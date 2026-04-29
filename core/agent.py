from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from agent.memory.compactor import _count_tokens_approx
from agent.tools import get_schemas

from .prompts import _build_system_prompt
from .turn import _post_turn_capture_and_summarize
from agent.ipc.controller import run_turn_ipc

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, config: "Config", store=None, embedder=None, asm_store=None, data_provider=None) -> None:
        from openai import AsyncOpenAI
        from agent.tools import load_all_tools

        # Ensure DataProvider exists; create from raw objects when not provided.
        if data_provider is None:
            from agent.data_provider import LocalDataProvider
            data_provider = LocalDataProvider(store=store, embedder=embedder, asm_store=asm_store, config=config)
        else:
            store = data_provider.get_store()
            embedder = data_provider.get_embedder()
            asm_store = data_provider.get_asm_store()

        self.config = config
        self._llm_defaults: dict = {
            "max_output_tokens": config.llm.max_output_tokens,
            "ctx_window": config.llm.ctx_window,
            "temperature": config.llm.temperature,
            "think_level": config.llm.think_level,
        }
        self.data_provider = data_provider
        self.store = store
        self.embedder = embedder
        self.asm_store = asm_store
        self.messages: list[dict] = []
        self._client = AsyncOpenAI(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
        )
        self._qa_logger = None
        self._facts_store = None
        self._side_log = None
        self._turn_id: int = 0
        self._notes_sys_idx: int | None = None  # index of current notes system message
        self._session_id: str | None = None
        self._project_memory_store = None  # project-level MemoryStore for session indexing
        self._last_turn_time: float = 0.0
        self._idle_compact_task: asyncio.Task | None = None
        self.stats: dict = {
            "input_tokens": 0,
            "output_tokens": 0,
            "content_tokens": 0,
            "reasoning_tokens": 0,
            "tool_tokens": 0,
            "calls": 0,
            "in_tps": 0.0,
            "out_tps": 0.0,
            "last_gen_seconds": 0.0,
            "last_output_tokens": 0,
            "last_content_tokens": 0,
            "last_reasoning_tokens": 0,
            "last_tool_tokens": 0,
        }
        self._pending_bg_tasks: set[asyncio.Task] = set()
        self.round_peak_tokens: int = 0
        self.last_round_peak_tokens: int = 0
        self._inject_queue: asyncio.Queue = asyncio.Queue()

        load_all_tools(config=config, data_provider=data_provider)

        indexed_count = store.stats()["chunks"] if store else 0
        system_content = _build_system_prompt(config, indexed_count=indexed_count)

        from agent.context import ensure_context_files, load_always_context, load_project_doc
        ensure_context_files(config, system_content)
        user_context = load_always_context(config)
        project_doc, project_doc_warning = load_project_doc(config)
        if project_doc_warning:
            logger.warning(project_doc_warning)
            import sys
            print(f"warning: {project_doc_warning}", file=sys.stderr)

        self.messages = [{"role": "system", "content": system_content}]
        if project_doc:
            self.messages.append({"role": "system", "content": project_doc})
        if user_context:
            self.messages.append({"role": "system", "content": user_context})

        # Notes are injected per-turn based on query relevance (_refresh_notes_context).
        # A full static load at startup is used only when no embedder is available.
        from agent.tools.notes import load_notes_context
        if not embedder:
            notes_ctx = load_notes_context(config)
            if notes_ctx:
                self._notes_sys_idx = len(self.messages)
                self.messages.append({"role": "system", "content": notes_ctx})

        # Project-level MemoryStore for cross-session session-summary indexing.
        try:
            from pathlib import Path
            from agent.memory.store import MemoryStore
            _agent_dir = Path(config.tools.working_dir) / config.tools.agent_dir
            self._project_memory_store = MemoryStore(_agent_dir / "memory.db")
        except Exception:
            self._project_memory_store = None

    def set_session_id(self, session_id: str) -> None:
        from agent.memory.qa_log import QALogger
        from agent.memory.facts_store import FactsStore
        from agent.memory.session import get_session_full_dir
        from agent.memory.side_log import SideLogWriter
        from agent.tools import recall as recall_tool
        from agent import failure_report as _fr
        self._session_id = session_id
        _fr.set_session(session_id)
        _fr.set_config(self.config)
        self._qa_logger = QALogger(session_id)
        self._facts_store = FactsStore(session_id, embedder=self.embedder)
        try:
            self._side_log = SideLogWriter(get_session_full_dir(session_id))
        except Exception as e:
            logger.warning("SideLogWriter init failed: %s", e)
            self._side_log = None
        recall_tool.setup(self._facts_store)

    def _refresh_notes_context(self, query: str, top_k: int = 6) -> None:
        """Inject (or replace) a notes system message relevant to *query*.

        Embeds the query, retrieves top-N notes from the project MemoryStore,
        and upserts a system message into self.messages just before the most
        recent user message. Tracks its index in _notes_sys_idx so the next
        call can remove the stale one.
        """
        if self.embedder is None:
            return
        from agent.tools.notes.notes import _get_store as _notes_store
        store = _notes_store()
        if store is None:
            return
        try:
            embedding = self.embedder.embed_one(query[:2000])
        except Exception:
            return
        hits = store.hybrid_search(query, embedding=embedding, scope="note", top_k=top_k)
        if not hits:
            # If no hits, clear stale notes message.
            if self._notes_sys_idx is not None:
                try:
                    del self.messages[self._notes_sys_idx]
                except IndexError:
                    pass
                self._notes_sys_idx = None
            return

        import json as _json
        lines = ["# Relevant saved notes\n"]
        for h in hits:
            title = h.get("title") or "(untitled)"
            body = h.get("body") or ""
            tags = h.get("tags")
            if isinstance(tags, str):
                try:
                    tags = _json.loads(tags)
                except Exception:
                    tags = []
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(f"## {title}{tag_str}\n{body}\n")
        notes_content = "\n".join(lines).strip()

        # Remove previous notes message.
        if self._notes_sys_idx is not None:
            try:
                del self.messages[self._notes_sys_idx]
            except IndexError:
                pass
            self._notes_sys_idx = None

        # Insert before the last user message (which was just appended).
        insert_at = len(self.messages) - 1
        if insert_at < 0:
            insert_at = 0
        self.messages.insert(insert_at, {"role": "system", "content": notes_content})
        self._notes_sys_idx = insert_at

    def pending_background_count(self) -> int:
        return sum(1 for t in self._pending_bg_tasks if not t.done())

    async def wait_background(self, timeout: float | None = None) -> int:
        tasks = [t for t in list(self._pending_bg_tasks) if not t.done()]
        if not tasks:
            return 0
        try:
            await asyncio.wait_for(
                asyncio.shield(asyncio.gather(*tasks, return_exceptions=True)),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            pass
        return sum(1 for t in tasks if not t.done())

    def inject(self, text: str) -> None:
        """Queue a user message to be injected on the next run_turn iteration."""
        self._inject_queue.put_nowait(text)

    def cancel_background(self) -> int:
        n = 0
        for t in list(self._pending_bg_tasks):
            if not t.done():
                t.cancel()
                n += 1
        return n

    def message_count(self) -> int:
        return len(self.messages)

    def get_messages(self) -> list[dict]:
        return list(self.messages)

    def set_messages(self, messages: list[dict]) -> None:
        self.messages = list(messages)

    def reset_messages(self) -> None:
        system = next((m for m in self.messages if m.get("role") == "system"), None)
        self.messages = [system] if system else []

    async def _idle_compact_loop(self, delay: float) -> None:
        """Wait `delay` seconds; if no new turn started, compact messages."""
        import time as _time
        stamp = self._last_turn_time
        await asyncio.sleep(delay)
        if self._last_turn_time != stamp:
            return  # new turn started — this fire is stale
        token_est = self.token_estimate()
        min_tokens = int(self.config.llm.ctx_window * 0.3)
        if token_est < min_tokens:
            return  # not enough content to bother compacting
        try:
            from agent.memory.compactor import compact
            self.messages = await compact(
                self.messages,
                self.config,
                self._client,
                facts_store=self._facts_store,
                project_memory_store=self._project_memory_store,
                session_id=self._session_id,
            )
            logger.debug("idle compaction: %d tokens after compact", self.token_estimate())
        except Exception:
            logger.debug("idle compaction failed", exc_info=True)

    async def compact_messages(self) -> None:
        from agent.memory.compactor import compact
        self.messages = await compact(self.messages, self.config, self._client)

    def token_estimate(self) -> int:
        return _count_tokens_approx(self.messages)

    def schema_tokens(self) -> int:
        try:
            from agent._tokens import count_tokens_approx
            return count_tokens_approx(json.dumps(get_schemas()))
        except Exception:
            return 0

    def context_breakdown(self) -> list[dict]:
        from agent._tokens import count_tokens_approx

        agent_prompt = 0
        user_context = 0
        user_input = 0
        assistant = 0
        tool_results = 0
        seen_system = False
        for m in self.messages:
            role = m.get("role")
            content = m.get("content") or ""
            if isinstance(content, list):
                text = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
            else:
                text = str(content)
            n = count_tokens_approx(text)
            if role == "system":
                if not seen_system:
                    agent_prompt += n
                    seen_system = True
                else:
                    user_context += n
            elif role == "user":
                user_input += n
            elif role == "assistant":
                assistant += n
                if m.get("tool_calls"):
                    assistant += count_tokens_approx(json.dumps(m["tool_calls"]))
            elif role == "tool":
                tool_results += n
        return [
            {"label": "agent_prompt", "tokens": agent_prompt},
            {"label": "user_context", "tokens": user_context},
            {"label": "tools_schema", "tokens": self.schema_tokens()},
            {"label": "skills",       "tokens": 0},
            {"label": "user_input",   "tokens": user_input},
            {"label": "assistant",    "tokens": assistant},
            {"label": "tool_results", "tokens": tool_results},
        ]

    def output_breakdown(self, scope: str = "session") -> list[dict]:
        s = self.stats
        if scope == "last":
            total = s.get("last_output_tokens", 0)
            reasoning = s.get("last_reasoning_tokens", 0)
            tool = s.get("last_tool_tokens", 0)
            content = s.get("last_content_tokens", 0)
        else:
            total = s.get("output_tokens", 0)
            reasoning = s.get("reasoning_tokens", 0)
            tool = s.get("tool_tokens", 0)
            content = s.get("content_tokens", 0)
        other = max(0, total - reasoning - tool - content)
        return [
            {"label": "reasoning", "tokens": reasoning},
            {"label": "tool",      "tokens": tool},
            {"label": "content",   "tokens": content},
            {"label": "other",     "tokens": other},
        ]

    def _record_usage(self, u: dict) -> None:
        s = self.stats
        s["input_tokens"] += u.get("input_tokens", 0)
        s["output_tokens"] += u.get("output_tokens", 0)
        s["content_tokens"] += u.get("content_tokens", 0)
        s["reasoning_tokens"] += u.get("reasoning_tokens", 0)
        s["tool_tokens"] += u.get("tool_tokens", 0)
        s["last_output_tokens"] = u.get("output_tokens", 0)
        s["last_content_tokens"] = u.get("content_tokens", 0)
        s["last_reasoning_tokens"] = u.get("reasoning_tokens", 0)
        s["last_tool_tokens"] = u.get("tool_tokens", 0)
        s["calls"] += 1
        gen = u.get("gen_seconds") or 0.0
        ttft = u.get("ttft")
        if ttft and ttft > 0 and u.get("input_tokens"):
            s["in_tps"] = u["input_tokens"] / ttft
        if gen > 0:
            s["out_tps"] = u.get("output_tokens", 0) / gen
            s["last_gen_seconds"] = gen

    async def chat(
        self,
        user_input: str,
        on_tool_call=None,
        on_tool_result=None,
        on_token=None,
        on_user_message=None,
        on_progress=None,
        on_loop_detected=None,
        on_phase=None,
        on_reasoning=None,
        on_context_size=None,
    ) -> str:
        self._turn_id += 1
        turn_id = self._turn_id

        # drain stale injections from a previous turn
        while not self._inject_queue.empty():
            try:
                self._inject_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self.last_round_peak_tokens = self.round_peak_tokens
        self.round_peak_tokens = self.token_estimate()

        def _track_ctx(n: int) -> None:
            if n > self.round_peak_tokens:
                self.round_peak_tokens = n
            if on_context_size is not None:
                try:
                    on_context_size(n)
                except Exception:
                    logger.exception("on_context_size callback failed")

        _turn_tool_calls: list[str] = []
        _turn_modified_files: list[str] = []
        original_on_tool_call = on_tool_call

        def _tracking_on_tool_call(name: str, args: str) -> None:
            _turn_tool_calls.append(name)
            if name in ("write_file", "patch_file", "edit_file"):
                try:
                    parsed = json.loads(args) if isinstance(args, str) else args
                    if name == "edit_file":
                        for ch in (parsed.get("chunks") or []):
                            p = ch.get("path", "") if isinstance(ch, dict) else ""
                            if p and p not in _turn_modified_files:
                                _turn_modified_files.append(p)
                    else:
                        path = parsed.get("path", "")
                        if path and path not in _turn_modified_files:
                            _turn_modified_files.append(path)
                except Exception:
                    pass
            if original_on_tool_call is not None:
                original_on_tool_call(name, args)

        pre_turn_len = len(self.messages)
        self.messages.append({"role": "user", "content": user_input})
        self._refresh_notes_context(user_input)
        if on_user_message is not None:
            on_user_message()
        try:
            response, self.messages = await run_turn_ipc(
                self.messages,
                self.config,
                self._client,
                on_token=on_token,
                on_tool_call=_tracking_on_tool_call,
                on_tool_result=on_tool_result,
                on_usage=self._record_usage,
                on_progress=on_progress,
                on_loop_detected=on_loop_detected,
                on_phase=on_phase,
                on_reasoning=on_reasoning,
                on_context_size=_track_ctx,
                facts_store=self._facts_store,
                turn_index=turn_id,
                side_log=self._side_log,
                inject_queue=self._inject_queue,
                project_memory_store=self._project_memory_store,
                session_id=self._session_id,
            )
        except Exception:
            # Roll back the user message so the next turn doesn't start with
            # consecutive user messages (which causes a 400 deadloop).
            self.messages = self.messages[:pre_turn_len]
            raise

        import time as _time
        self._last_turn_time = _time.monotonic()

        if self._qa_logger is not None:
            task = asyncio.create_task(
                _post_turn_capture_and_summarize(
                    self._qa_logger,
                    self.config,
                    turn_id,
                    user_input,
                    response,
                    list(_turn_tool_calls),
                    list(_turn_modified_files),
                )
            )
            self._pending_bg_tasks.add(task)
            task.add_done_callback(self._pending_bg_tasks.discard)

        idle_sec = getattr(self.config.token_limits, "idle_compaction_seconds", 0.0)
        if idle_sec > 0:
            if self._idle_compact_task is not None and not self._idle_compact_task.done():
                self._idle_compact_task.cancel()
            self._idle_compact_task = asyncio.create_task(
                self._idle_compact_loop(idle_sec)
            )
            self._pending_bg_tasks.add(self._idle_compact_task)
            self._idle_compact_task.add_done_callback(self._pending_bg_tasks.discard)

        return response

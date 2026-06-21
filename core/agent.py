from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

from agent.memory.compactor import _count_tokens_approx
from agent.tools import get_schemas

from .prompts import _build_system_prompt, load_base_rules, HARD_RULES_MARKER
from agent import prompt_compiler as _prompt_compiler
from .turn import _post_turn_capture_and_summarize, run_turn
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
            "autonomy": config.agent.autonomy,
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
        # Registry entry name for the active LLM endpoint — key under which the
        # turn loop persists throughput in model_stats.json.
        try:
            from agent.metrics.model_stats import resolve_entry_name
            self._model_entry_name = resolve_entry_name(config)
        except Exception:
            self._model_entry_name = config.llm.model or "default"
        self._facts_store = None
        self._side_log = None
        self._turn_id: int = 0
        self._notes_sys_idx: int | None = None  # kept for compat; notes removal now uses marker
        self._session_id: str | None = None
        self._similar_sessions_injected: bool = False
        self._project_memory_store = None  # project-level MemoryStore for session indexing
        self._last_turn_time: float = 0.0
        self._idle_compact_task: asyncio.Task | None = None
        self._active_step_skills: list[str] = []
        self._skill_loader = None
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
        self.on_turn_summarized = None
        self.round_peak_tokens: int = 0
        self.last_round_peak_tokens: int = 0
        self._inject_queue: asyncio.Queue = asyncio.Queue()

        from agent.core.output_store import init_store as _init_output_store
        _init_output_store(config.output_store)

        # Initialise GPU concurrency semaphore from config
        if config.concurrency.gpu_pool and config.concurrency.gpu_slots > 0:
            from agent.core.model_status import init_gpu_semaphore
            from agent.core.gpu_lock import resolve_lock_dir
            # Key the shared lock on the GPU endpoint so agents in different
            # working dirs hitting the same server coordinate.
            lock_dir = resolve_lock_dir(
                config.concurrency.gpu_lock_dir,
                config.llm.base_url,
            )
            init_gpu_semaphore(config.concurrency.gpu_slots, lock_dir)
            logger.info(
                "gpu_semaphore: %d slot(s) for pool %r; cross-process lock: %s",
                config.concurrency.gpu_slots, config.concurrency.gpu_pool,
                lock_dir or "disabled",
            )

        load_all_tools(config=config, data_provider=data_provider)

        indexed_stats = store.stats() if store else {"chunks": 0, "files": 0}
        indexed_count = indexed_stats["files"]
        total_files = indexed_count
        index_percent = 100
        if store and indexed_count > 0:
            try:
                from agent.rag.indexer import pending_files as _pending_files
                pf = _pending_files(config.tools.working_dir, store)
                total_files = pf["total"]
                if total_files > 0:
                    index_percent = round(100 * pf["indexed"] / total_files)
            except Exception:
                pass
        system_content = _build_system_prompt(
            config,
            indexed_count=indexed_count,
            total_files=total_files,
            index_percent=index_percent,
        )

        from agent.context import ensure_context_files, load_always_context, load_project_doc
        ensure_context_files(config, system_content)
        user_context = load_always_context(config)
        project_doc, project_doc_warning = load_project_doc(config)
        if project_doc_warning:
            logger.warning(project_doc_warning)
            import sys
            print(f"warning: {project_doc_warning}", file=sys.stderr)

        base_rules = load_base_rules()
        if base_rules:
            base_rules = _prompt_compiler.load("base_rules.txt", base_rules, config)

        # Project-level MemoryStore — init early so it can be shared with rules injection.
        try:
            from pathlib import Path
            from agent.memory.store import MemoryStore
            _agent_dir = Path(config.tools.working_dir) / config.tools.agent_dir
            self._project_memory_store = MemoryStore(_agent_dir / "memory.db")
        except Exception:
            self._project_memory_store = None

        self.messages = []
        if base_rules:
            self.messages.append({"role": "system", "content": base_rules, HARD_RULES_MARKER: True})

        # Inject learned behavioral rules from past sessions.
        try:
            from agent.memory.rules_loader import load_behavioral_rules
            hard_rules, soft_rules = load_behavioral_rules(config, store=self._project_memory_store)
            if hard_rules:
                self.messages.append({"role": "system", "content": hard_rules, HARD_RULES_MARKER: True})
            if soft_rules:
                self.messages.append({"role": "system", "content": soft_rules})
        except Exception:
            pass

        self.messages.append({"role": "system", "content": system_content})
        if project_doc:
            self.messages.append({"role": "system", "content": project_doc})
        if user_context:
            self.messages.append({"role": "system", "content": user_context})

        # Skills: build loader and inject index summary once at session start.
        from agent.skills import SkillLoader
        self._skill_loader = SkillLoader(config)
        self._active_step_skills = []
        skill_index = self._skill_loader.index_summary()
        if skill_index:
            self.messages.append({"role": "system", "content": skill_index})

        # Notes are injected per-turn based on query relevance (_refresh_notes_context).
        # A full static load at startup is used only when no embedder is available.
        from agent.tools.notes import load_notes_context
        if not embedder:
            notes_ctx = load_notes_context(config)
            if notes_ctx:
                self.messages.append({"role": "system", "content": notes_ctx, "_notes_marker": True})

    def set_session_id(self, session_id: str) -> None:
        from agent.memory.qa_log import QALogger
        from agent.memory.facts_store import FactsStore
        from agent.memory.session import get_session_full_dir
        from agent.memory.side_log import SideLogWriter
        from agent.tools import recall as recall_tool
        from agent.tools import rate_session as rate_session_tool
        from agent.tools import recall_history as recall_history_tool
        from agent import failure_report as _fr
        self._session_id = session_id
        self._similar_sessions_injected = False
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
        recall_history_tool.setup(self._qa_logger)
        rate_session_tool.set_session(session_id)
        try:
            from agent.tools.files.hint import reset_session_hints
            reset_session_hints()
        except Exception:
            pass

    def _inject_similar_sessions(self, query: str, top_k: int = 3, embedding=None) -> None:
        """On first turn, inject top-K similar rated sessions as context.

        Only fires once per session (guarded by _similar_sessions_injected).
        Requires project MemoryStore and at least one rated session in history.

        Args:
            embedding: Optional precomputed embedding for ``query[:2000]``.
                       When provided, skips the duplicate embedder call.
        """
        if self._similar_sessions_injected:
            return
        self._similar_sessions_injected = True

        store = self._project_memory_store
        if store is None:
            return

        if embedding is None and self.embedder is not None:
            try:
                embedding = self.embedder.embed_one(query[:2000])
            except Exception:
                pass

        # Try good outcomes first; fall back to ok if no good results
        hits = store.hybrid_search(
            query,
            embedding=embedding,
            scope="session_summary",
            top_k=top_k,
            tags_filter=["outcome:good"],
        )
        if not hits:
            hits = store.hybrid_search(
                query,
                embedding=embedding,
                scope="session_summary",
                top_k=top_k,
                tags_filter=["outcome:ok"],
            )
        if not hits:
            return

        lines = ["# Similar past sessions (rated successful)\n"]
        for h in hits:
            title = h.get("title") or "(untitled)"
            snippet = (h.get("body") or "")[:500]
            lines.append(f"## {title}\n{snippet}\n")
        content = "\n".join(lines).strip()

        insert_at = len(self.messages) - 1
        if insert_at < 0:
            insert_at = 0
        self.messages.insert(
            insert_at,
            {"role": "user", "content": content, "_similar_sessions_marker": True},
        )

    def _refresh_notes_context(self, query: str, top_k: int = 6, embedding=None) -> None:
        """Inject (or replace) a notes message relevant to *query*.

        Embeds the query, retrieves top-N notes from the project MemoryStore,
        and upserts a user message into self.messages just before the most
        recent user message. Uses a ``_notes_marker`` key to find and remove
        the previous injection regardless of how messages shifted since then.

        Args:
            embedding: Optional precomputed embedding for ``query[:2000]``.
                       When provided, skips the duplicate embedder call.
        """
        if self.embedder is None:
            return
        from agent.tools.notes.notes import _get_store as _notes_store
        store = _notes_store()
        if store is None:
            return
        if embedding is None:
            try:
                embedding = self.embedder.embed_one(query[:2000])
            except Exception:
                return
        hits = store.hybrid_search(query, embedding=embedding, scope="note", top_k=top_k)

        # Remove any previous notes injection (find by marker, not index).
        self.messages = [m for m in self.messages if not m.get("_notes_marker")]
        self._notes_sys_idx = None

        if not hits:
            return

        lines = ["# Relevant saved notes\n"]
        for h in hits:
            title = h.get("title") or "(untitled)"
            body = h.get("body") or ""
            tags = h.get("tags")
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except Exception:
                    tags = []
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(f"## {title}{tag_str}\n{body}\n")
        notes_content = "\n".join(lines).strip()

        # Insert before the last user message (which was just appended).
        insert_at = len(self.messages) - 1
        if insert_at < 0:
            insert_at = 0
        self.messages.insert(insert_at, {"role": "user", "content": notes_content, "_notes_marker": True})

    def _refresh_skills_context(self) -> None:
        """Swap skills system message when the active plan step changes."""
        if self._skill_loader is None:
            return
        step_skills: list[str] = []
        try:
            from agent.planning.plan import list_plans
            active = next((p for p in list_plans() if p.status == "active"), None)
            if active:
                in_progress = next((s for s in active.steps if s.status == "in_progress"), None)
                if in_progress:
                    step_skills = list(in_progress.skills)
        except Exception:
            pass

        if step_skills == self._active_step_skills:
            return
        self._active_step_skills = step_skills

        self.messages = [m for m in self.messages if not m.get("_skills_marker")]
        if not step_skills:
            return

        content = self._skill_loader.load(step_skills)
        if not content.strip():
            return

        # Inject after last static system message, before any user/assistant turns.
        insert_at = 0
        for i, m in enumerate(self.messages):
            if m.get("role") == "system":
                insert_at = i + 1
            else:
                break
        self.messages.insert(insert_at, {
            "role": "system",
            "content": f"# Active step skills\n\n{content}",
            "_skills_marker": True,
        })

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
        self.messages = [m for m in self.messages if m.get("role") == "system"]

    async def _idle_compact_loop(self, delay: float) -> None:
        """Wait `delay` seconds; if no new turn started, compact messages."""
        stamp = self._last_turn_time
        await asyncio.sleep(delay)
        if self._last_turn_time != stamp:
            return  # new turn started — this fire is stale
        # Deferred idle work (session naming/classification, backfill, …) runs
        # first, on the full transcript, before compaction may summarize it away.
        try:
            from agent.core.idle_tasks import run_pending
            await run_pending(self)
        except Exception:
            logger.debug("idle deferred tasks failed", exc_info=True)
        if self._last_turn_time != stamp:
            return  # user started typing while we worked
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
        # Pass the same stores as idle/auto compaction so a manual /compact also
        # persists the Tier-2 facts round (and carries original_request forward);
        # otherwise recall_facts loses everything elided by the manual compaction.
        self.messages = await compact(
            self.messages,
            self.config,
            self._client,
            facts_store=self._facts_store,
            project_memory_store=self._project_memory_store,
            session_id=self._session_id,
        )

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
        skills_tokens = sum(
            count_tokens_approx(m.get("content", "") if isinstance(m.get("content"), str) else "")
            for m in self.messages if m.get("_skills_marker")
        )
        return [
            {"label": "agent_prompt", "tokens": agent_prompt},
            {"label": "user_context", "tokens": user_context},
            {"label": "tools_schema", "tokens": self.schema_tokens()},
            {"label": "skills",       "tokens": skills_tokens},
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
            # Persist throughput so the daily-chat path feeds model_stats.json
            # (previously only the commit-message generator did). EWMA filter +
            # _MIN_TOKENS guard live inside update_stats.
            try:
                from agent.metrics.model_stats import update_stats
                update_stats(self._model_entry_name, u.get("output_tokens", 0), gen)
            except Exception:
                logger.debug("update_stats failed", exc_info=True)

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
        stop_event: asyncio.Event | None = None,
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
        _is_continue = user_input.strip().lower() in ("continue", "/continue", "/c")
        if not _is_continue:
            self.messages.append({"role": "user", "content": user_input})
            if self._facts_store is not None:
                self._facts_store.set_original_request(user_input)
            # Compute embedding once; reuse across both injection helpers.
            precomputed_embedding = None
            if self.embedder is not None:
                try:
                    precomputed_embedding = self.embedder.embed_one(user_input[:2000])
                except Exception:
                    pass
            self._inject_similar_sessions(user_input, embedding=precomputed_embedding)
            self._refresh_notes_context(user_input, embedding=precomputed_embedding)
        self._refresh_skills_context()
        if on_user_message is not None:
            on_user_message()
        _run_turn_fn = run_turn if not self.config.parallel.enabled else run_turn_ipc
        _excluded: set[str] = set()
        if self.config.web_search.require_worker:
            _excluded.update({"web_search", "web_fetch"})
        # Mode gate: in ultrasecure the privileged agent loses direct net access
        # and reaches the internet only through the quarantined ask_internet
        # broker; in fast mode ask_internet is hidden (broker is unconfigured).
        if getattr(self.config.agent, "mode", "fast") == "ultrasecure":
            _excluded.update({"web_search", "web_fetch"})
        else:
            _excluded.add("ask_internet")
        try:
            response, self.messages = await _run_turn_fn(
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
                stop_event=stop_event,
                excluded_tools=_excluded or None,
            )
        except BaseException:
            # Roll back the user message so the next turn doesn't start with
            # consecutive user messages (which causes a 400 deadloop).
            self.messages = self.messages[:pre_turn_len]
            raise

        self._last_turn_time = time.monotonic()

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
                    on_summarized=self.on_turn_summarized,
                )
            )
            self._pending_bg_tasks.add(task)
            task.add_done_callback(self._pending_bg_tasks.discard)

        idle_sec = self.config.token_limits.idle_compaction_seconds
        if idle_sec > 0:
            if self._idle_compact_task is not None and not self._idle_compact_task.done():
                self._idle_compact_task.cancel()
            self._idle_compact_task = asyncio.create_task(
                self._idle_compact_loop(idle_sec)
            )
            self._pending_bg_tasks.add(self._idle_compact_task)
            self._idle_compact_task.add_done_callback(self._pending_bg_tasks.discard)

        return response

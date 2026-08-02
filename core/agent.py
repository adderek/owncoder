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
from .changeset import open_window as _changeset_open_window, paths_from_tool_call
from agent.ipc.controller import run_turn_ipc
from agent.security.airgap import is_local_url

# Minimum embedding cosine similarity for a saved note to be injected as
# turn context when it has no meaningful token overlap with the query.
_NOTES_MIN_SIMILARITY = 0.5

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)


def _describe_turn_failure(exc: BaseException) -> str:
    """Why a round ended early, short enough to sit in the transcript."""
    if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
        return "stopped"
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {text}"[:200] if text else type(exc).__name__


class Agent:
    def __init__(self, config: "Config", store=None, embedder=None, asm_store=None, data_provider=None) -> None:
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
        from agent.core.llm_client import make_llm_client
        # Bound a single request so a wedged backend (e.g. a GPU/HSA stall on
        # the llama.cpp side) can't hang the agent forever. The per-chunk
        # stall watchdog in _stream_response catches mid-stream stalls sooner;
        # this is the outer hard ceiling. max_retries=0 — we own retry below.
        self._client = make_llm_client(config)
        self._qa_logger = None
        # Registry entry name for the active LLM endpoint — key under which the
        # turn loop persists throughput in model_stats.json.
        try:
            from agent.metrics.model_stats import resolve_entry_name
            self._model_entry_name = resolve_entry_name(config)
        except Exception:
            self._model_entry_name = config.llm.model or "default"
        # Cost tier of the main endpoint, classified once (config is immutable
        # for the session). Used to attribute every main-turn LLM call.
        try:
            from agent.config.registry import entry_tier
            _entry = (getattr(config, "model_entries", {}) or {}).get(self._model_entry_name)
            self._model_tier = entry_tier(_entry) if _entry is not None else "local"
        except Exception:
            self._model_tier = "local"
        self.last_round_model_calls: dict = {}
        self._facts_store = None
        self._side_log = None
        self._turn_id: int = 0
        self._notes_sys_idx: int | None = None  # kept for compat; notes removal now uses marker
        self._pending_note_grade: dict | None = None  # last injection awaiting usefulness grading
        self._session_id: str | None = None
        self._session_mode: str = "standard"  # "standard" | "incognito" | "private"
        self._similar_sessions_injected: bool = False
        self._project_memory_store = None  # project-level MemoryStore for session indexing
        self._last_turn_time: float = 0.0
        self.last_changeset = None  # core.changeset.Changeset for the last round
        self._turn_busy: bool = False  # True while a turn runs (scheduler defers)
        self._idle_compact_task: asyncio.Task | None = None
        self._active_step_skills: list[str] = []
        self._skill_loader = None
        self.stats: dict = {
            "input_tokens": 0,
            # Portion of input_tokens the endpoint served from its prompt cache.
            # Only some providers report it; 0 means "not reported", not "miss".
            "cached_input_tokens": 0,
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

        # Record how the tool surface changed since last session, with the
        # reason from git. See agent/core/tool_ledger.py.
        try:
            from agent.core import tool_ledger as _tool_ledger
            _tool_ledger.record_changes(config, get_schemas())
        except Exception:
            logger.debug("tool ledger not updated", exc_info=True)

        indexed_stats = store.stats() if store else {"chunks": 0, "files": 0}
        indexed_count = indexed_stats["files"]
        total_files = indexed_count
        index_percent = 100
        if store and indexed_count > 0:
            try:
                from agent.rag.indexer import pending_files as _pending_files
                pf = _pending_files(config.tools.working_dir, store, cfg=config.rag)
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

        # Core rules go first and are never compiled, compacted, or written by
        # the agent — the one part of the prompt with a human in the loop.
        # See agent/core/core_rules.py.
        try:
            from agent.core import core_rules as _core_rules
            core_text = _core_rules.load_core(config)
            if core_text:
                self.messages.append({"role": "system", "content": core_text,
                                      HARD_RULES_MARKER: True})
            _core_rules.record_state(config)
        except Exception:
            logger.warning("core rules not injected", exc_info=True)

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

        # Notes are injected per-turn based on query relevance
        # (_refresh_notes_context). With an embedder it uses hybrid search;
        # without one it falls back to FTS — no full static dump either way.

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
        from agent.tools.notes.notes import _get_store as _notes_store
        store = _notes_store()
        if store is None:
            return
        if self.embedder is None:
            # FTS-only fallback. Raw user text breaks fts5 MATCH syntax
            # (punctuation, quotes), so reduce it to an OR of word tokens.
            import re as _re
            # \w (unicode) — ASCII-only class split words like "prasówka"
            # into fragments that never match fts5 unicode61 tokens.
            tokens = _re.findall(r"\w{3,}", query)[:12]
            if not tokens:
                return
            hits = store.fts_search(
                " OR ".join(f'"{t}"' for t in tokens), scope="note", top_k=top_k
            )
        else:
            if embedding is None:
                try:
                    embedding = self.embedder.embed_one(query[:2000])
                except Exception:
                    return
            hits = store.hybrid_search(query, embedding=embedding, scope="note", top_k=top_k)

        # Remove any previous notes injection (find by marker, not index).
        self.messages = [m for m in self.messages if not m.get("_notes_marker")]
        self._notes_sys_idx = None

        # Relevance floor. hybrid_search's combined_score is normalized within
        # the result set (relative), so filter on absolute signals instead:
        # embedding similarity, or ≥2 query tokens appearing in the note.
        # Without this, a small store injects whatever it has (top_k fills up).
        import re as _re
        qtokens = {t.lower() for t in _re.findall(r"\w{3,}", query)}

        def _relevant(h: dict) -> bool:
            # Usefulness feedback: a note injected many times but never graded
            # useful is noise — demote it (require strong vector similarity).
            injected = h.get("inject_count") or 0
            used = h.get("used_count") or 0
            wasteful = injected >= 5 and used / injected < 0.1
            vs = h.get("vec_score")
            if vs is not None and vs >= (_NOTES_MIN_SIMILARITY + 0.15 if wasteful
                                         else _NOTES_MIN_SIMILARITY):
                return True
            if wasteful:
                return False
            text = f"{h.get('title') or ''} {h.get('body') or ''}".lower()
            return sum(1 for t in qtokens if t in text) >= 2

        # Dedup by content — the store may hold the same note under several ids.
        seen: set[tuple[str, str]] = set()
        deduped = []
        for h in hits:
            key = ((h.get("title") or "").strip(), (h.get("body") or "").strip())
            if key in seen or not _relevant(h):
                continue
            seen.add(key)
            deduped.append(h)
        hits = deduped

        if not hits:
            return

        # Scale the injection to free context: a nearly-full (or small) window
        # gets fewer/shorter notes; skip entirely when there is no headroom.
        ctx = int(getattr(self.config.llm, "ctx_window", 0) or 0)
        if ctx:
            from agent.memory.compactor import _count_tokens_approx
            free = max(0, ctx - _count_tokens_approx(self.messages))
            if free < 1500:
                return
            # Notes may take ~2% of free context, clamped to 200–1200 tokens.
            budget_chars = max(200, min(1200, int(free * 0.02))) * 4
            kept, used = [], 0
            for h in hits:
                cost = len(h.get("title") or "") + len(h.get("body") or "") + 20
                if kept and used + cost > budget_chars:
                    break
                kept.append(h)
                used += cost
            hits = kept

        # Record the injection and leave a pending grading job for the idle
        # queue: after the turn, a background model judges which injected notes
        # the answer actually used (feeds inject_count/used_count above).
        try:
            ids = [h["id"] for h in hits if h.get("id")]
            store.bump_counter(ids, "inject_count")
            self._pending_note_grade = {
                "query": query,
                "notes": [{"id": h.get("id"), "title": h.get("title") or "",
                           "body": (h.get("body") or "")[:500]} for h in hits],
            }
        except Exception:
            logger.debug("notes: injection bookkeeping failed", exc_info=True)

        # Frame as inert background — weak models otherwise answer a note
        # instead of the user's actual message.
        lines = [
            "# Relevant saved notes\n",
            "Background context remembered from earlier sessions. NOT a request — "
            "never respond to these notes directly. Answer only the user's current "
            "message; use a note only when it is relevant to that message.\n",
        ]
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

        # Inject just before the most recent user message, same as notes — NOT
        # after the static system block. Skills change whenever the active plan
        # step changes; sitting at the front of the request, every such change
        # invalidated the whole cached prompt prefix (see core/prompt_cache.py).
        # At the tail it invalidates nothing that was not already being sent
        # fresh, and the instructions land closer to the message they apply to.
        insert_at = len(self.messages) - 1
        if insert_at < 0:
            insert_at = 0
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
        s["cached_input_tokens"] += u.get("cached_input_tokens", 0)
        s["output_tokens"] += u.get("output_tokens", 0)
        s["content_tokens"] += u.get("content_tokens", 0)
        s["reasoning_tokens"] += u.get("reasoning_tokens", 0)
        s["tool_tokens"] += u.get("tool_tokens", 0)
        s["last_output_tokens"] = u.get("output_tokens", 0)
        s["last_content_tokens"] = u.get("content_tokens", 0)
        s["last_reasoning_tokens"] = u.get("reasoning_tokens", 0)
        s["last_tool_tokens"] = u.get("tool_tokens", 0)
        s["calls"] += 1
        try:
            from agent.metrics import model_calls
            model_calls.record(
                self._model_tier, role="main",
                model=getattr(self, "_model_entry_name", "")
                or getattr(self.config.llm, "model", ""),
                in_tokens=u.get("input_tokens", 0),
                out_tokens=u.get("output_tokens", 0),
            )
        except Exception:
            logger.debug("model_calls.record failed", exc_info=True)
        gen = u.get("gen_seconds") or 0.0
        ttft = u.get("ttft")
        # Prompt tokens the endpoint had to actually process — tokens it served
        # from its own cache cost no prefill time, so counting them would make
        # in-tok/s read absurdly high on a warm prompt.
        fresh_in = max(0, u.get("input_tokens", 0) - u.get("cached_input_tokens", 0))
        if ttft and ttft > 0 and fresh_in:
            s["in_tps"] = fresh_in / ttft
        if gen > 0:
            s["out_tps"] = u.get("output_tokens", 0) / gen
            s["last_gen_seconds"] = gen
        # Persist throughput so the daily-chat path feeds model_stats.json
        # (previously only the commit-message generator did). EWMA filter +
        # per-direction sample guards live inside update_stats.
        try:
            from agent.metrics.model_stats import update_stats
            update_stats(self._model_entry_name, u.get("output_tokens", 0), gen,
                         in_tokens=fresh_in, ttft=ttft or 0.0)
        except Exception:
            logger.debug("update_stats failed", exc_info=True)

    def _changeset_spill_dir(self, turn_id: int) -> "Path | None":
        """Where oversized diffs for this round go, or None when there is no
        session to hang them off (a one-shot run keeps them in memory)."""
        if not self._session_id:
            return None
        try:
            from pathlib import Path
            from agent.memory.session import get_session_full_dir
            return Path(get_session_full_dir(self._session_id)) / "changesets" / str(turn_id)
        except Exception:
            logger.debug("changeset: no spill dir for session %s", self._session_id,
                         exc_info=True)
            return None

    def _collect_changeset(self, turn_id: int, since_seq: int):
        """The round's changeset. Never raises — a summary is not worth a turn."""
        from agent.core import changeset as _cs
        if not getattr(self.config.ui.changeset, "enabled", True):
            return _cs.Changeset(turn_id=turn_id)
        try:
            return _cs.collect(
                since_seq,
                working_dir=self.config.tools.working_dir,
                limits=_cs.limits_from_config(self.config),
                turn_id=turn_id,
                spill_dir=self._changeset_spill_dir(turn_id),
            )
        except Exception:
            logger.exception("changeset: collection failed (round summary skipped)")
            return _cs.Changeset(turn_id=turn_id)

    def _changeset_prose_mode(self) -> str:
        """"off" | "background" | "always" from [ui.changeset], read defensively
        so a missing/odd config value degrades to "off" rather than raising."""
        raw = getattr(getattr(self.config.ui, "changeset", None), "prose_summary", "off")
        return str(raw or "off").lower()

    async def _apply_changeset_prose(self, cs) -> None:
        """"always" mode: compute the one-line intent summary and set it on
        *cs* before it's persisted. Never raises — a summary is not worth a
        round."""
        from agent.core.changeset_prose import summarize as _cs_summarize
        try:
            cs.prose = await _cs_summarize(self.config, cs)
        except Exception:
            logger.exception("changeset: prose summary failed (round unaffected)")

    def _checkpoint_note_for_failed_turn(self, exc: BaseException) -> str:
        """Handle file edits left behind by a turn that died. Returns a note or "".

        With ``[checkpoints] auto_rollback_on_error`` the newest auto checkpoint
        is restored, so the next turn starts from a clean tree. It is off by
        default because reverting files is destructive and half-finished edits
        are often still wanted — so the default is to *name* the rollback point
        and let the user decide.

        Cancellation is never a failure: Ctrl-C and a stop button mean "stop
        doing things", not "undo what you did".
        """
        import asyncio as _asyncio
        if isinstance(exc, (KeyboardInterrupt, _asyncio.CancelledError)):
            return ""
        try:
            from agent.core import checkpoint as _cp
            cp = _cp.latest_auto_checkpoint()
            if cp is None:
                return ""
            pending = _cp.edits_since(cp.id)
            if not pending:
                return ""
            if not getattr(self.config.checkpoints, "auto_rollback_on_error", False):
                return (f"[turn failed after {pending} file edit(s). They are still on disk; "
                        f"revert them with /checkpoint rollback {cp.id} ('{cp.label}').]")
            res = _cp.rollback_to(cp.id)
            if res.get("error"):
                return f"[turn failed; auto-rollback to {cp.id} failed: {res['error']}]"
            note = (f"[turn failed — auto-rolled back to checkpoint {cp.id}: "
                    f"{len(res['restored'])} file(s) restored, "
                    f"{len(res['deleted'])} removed. Retry from this clean state.]")
            logger.warning("checkpoints: auto-rollback to %s after turn error (%s)",
                           cp.id, type(exc).__name__)
            if res.get("errors"):
                note += f"  (partial: {res['errors']})"
            return note
        except Exception:
            logger.debug("checkpoint note for failed turn failed", exc_info=True)
            return ""

    async def chat(
        self,
        user_input: str,
        on_tool_call=None,
        on_tool_result=None,
        on_tool_record=None,
        on_token=None,
        on_user_message=None,
        on_progress=None,
        on_loop_detected=None,
        on_phase=None,
        on_injected_message=None,
        on_reasoning=None,
        on_context_size=None,
        on_changeset=None,
        stop_event: asyncio.Event | None = None,
        source: str = "terminal",
    ) -> str:
        self._turn_id += 1
        turn_id = self._turn_id

        # Start a fresh per-round model-call tally (local/free/bundled/paid).
        try:
            from agent.metrics import model_calls
            model_calls.reset_round()
        except Exception:
            pass

        # Refresh this agent's worktree presence beacon each turn so other agents
        # see a live heartbeat (TTL-based liveness in agent/coord/presence.py).
        try:
            from agent import coord as _coord
            _coord.heartbeat(
                self.config.tools.working_dir,
                agent="owncoder",
                tool="owncoder",
                note=self.config.llm.model,
            )
        except Exception:
            logger.debug("coord heartbeat failed", exc_info=True)

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
        # Journal position before the round: bounds the edits that belong to it.
        # Taken here rather than at the first edit so a round that starts with a
        # concurrent agent already writing still measures from its own start.
        _changeset_seq = _changeset_open_window()
        original_on_tool_call = on_tool_call

        def _tracking_on_tool_call(name: str, args: str) -> None:
            _turn_tool_calls.append(name)
            for p in paths_from_tool_call(name, args):
                if p not in _turn_modified_files:
                    _turn_modified_files.append(p)
            if original_on_tool_call is not None:
                original_on_tool_call(name, args)

        # Refuse a private-mode turn against a non-local endpoint before mutating
        # message state, so a raise here leaves nothing to roll back.
        self._validate_private_mode()

        pre_turn_len = len(self.messages)
        _is_continue = user_input.strip().lower() in ("continue", "/continue", "/c")
        if not _is_continue:
            # Deliver results of scheduled runs that finished since the last
            # turn — they land in detached sched-<name> sessions, so without
            # this the interactive conversation never learns their outcome.
            if source != "scheduler":
                try:
                    from agent.core import scheduler
                    _runs = scheduler.unseen_results(self.config)
                    if _runs:
                        _lines = []
                        for r in _runs:
                            line = f"- {r.get('name') or r.get('job')}: {r.get('status')}"
                            if r.get("result"):
                                line += f" — {r['result']}"
                            if r.get("session"):
                                line += f"  [session {r['session']}]"
                            _lines.append(line)
                        self.messages.append({
                            "role": "system",
                            "content": "[scheduled jobs finished since last turn]\n"
                                       + "\n".join(_lines),
                        })
                        if on_phase is not None:
                            try:
                                on_phase("sched_results",
                                         f"{len(_runs)} scheduled run(s) delivered")
                            except Exception:
                                pass
                except Exception:
                    logger.debug("sched result delivery failed", exc_info=True)

            # Deliver background shell jobs (run_argv_bg) that finished since the
            # last turn, so the model sees build/test results without polling.
            try:
                from agent.tools.shell.main import drain_bg_finished
                _bg = drain_bg_finished()
                if _bg:
                    _blines = []
                    for b in _bg:
                        cmd = " ".join(b.get("argv") or [])[:80]
                        res = b.get("result") or {}
                        rc = res.get("returncode")
                        tail = ""
                        if isinstance(res, dict):
                            _err = (res.get("stderr") or "").strip() or (res.get("stdout") or "").strip()
                            if res.get("error"):
                                _err = res["error"]
                            if _err:
                                tail = " — " + _err[-300:]
                        _blines.append(
                            f"- job {b['job_id']} ({b['status']}"
                            + (f", exit {rc}" if rc is not None else "")
                            + f"): {cmd}{tail}")
                    self.messages.append({
                        "role": "system",
                        "content": "[background shell jobs finished since last turn "
                                   "— full output via bg_output(job_id)]\n"
                                   + "\n".join(_blines),
                    })
                    if on_phase is not None:
                        try:
                            on_phase("bg_results", f"{len(_bg)} background job(s) delivered")
                        except Exception:
                            pass
            except Exception:
                logger.debug("bg shell delivery failed", exc_info=True)
            self.messages.append({"role": "user", "content": user_input})
            if self._facts_store is not None:
                self._facts_store.set_original_request(user_input)
            # Compute embedding once; reuse across both injection helpers.
            # Off the event loop: the embed call is sync network I/O and a
            # slow endpoint would otherwise freeze the whole UI (SSE, stop,
            # queued prompts) until it returns.
            precomputed_embedding = None
            if self.embedder is not None:
                try:
                    precomputed_embedding = await asyncio.to_thread(
                        self.embedder.embed_one, user_input[:2000])
                except Exception:
                    pass
            self._inject_similar_sessions(user_input, embedding=precomputed_embedding)
            self._refresh_notes_context(user_input, embedding=precomputed_embedding)
        self._refresh_skills_context()
        if on_user_message is not None:
            on_user_message()
        # Per-turn model tiering: fast model by default, strong for complex turns
        # (no-op unless config.auto_tier.enabled). Re-decided every turn so a
        # strong turn reverts to fast on the next one.
        try:
            from agent.core.model_tier import select_for_turn, apply_entry
            _tier = select_for_turn(self.config, user_input, source)
            if _tier and apply_entry(self, self.config, _tier):
                logger.info("auto-tier: turn on '%s' (model=%s url=%s)",
                            _tier, self.config.llm.model, self.config.llm.base_url)
        except Exception:
            logger.exception("auto-tier selection failed (ignored)")
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
        self._turn_busy = True
        try:
            response, self.messages = await _run_turn_fn(
                self.messages,
                self.config,
                self._client,
                on_token=on_token,
                on_tool_call=_tracking_on_tool_call,
                on_tool_result=on_tool_result,
                on_tool_record=on_tool_record,
                on_usage=self._record_usage,
                on_progress=on_progress,
                on_loop_detected=on_loop_detected,
                on_phase=on_phase,
                on_injected_message=on_injected_message,
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
        except BaseException as _turn_exc:
            # Roll back the turn's own additions so the next turn doesn't start
            # with consecutive user messages (which causes a 400 deadloop) —
            # but keep the question itself, followed by a note saying it never
            # got answered. Dropping the question outright made a stopped or
            # crashed round vanish from the resumed session even though the
            # user had watched it happen.
            self.messages = self.messages[:pre_turn_len]
            if user_input:
                self.messages.append({"role": "user", "content": user_input})
                self.messages.append({
                    "role": "assistant",
                    "content": f"[turn did not finish: {_describe_turn_failure(_turn_exc)}]",
                })
            _note = self._checkpoint_note_for_failed_turn(_turn_exc)
            if _note:
                self.messages.append({"role": "system", "content": _note})
            raise
        finally:
            self._turn_busy = False
            self._last_turn_time = time.monotonic()

        # Snapshot the round's model-call breakdown before background tasks
        # (qa-summary, idle compaction, naming) schedule their own LLM calls.
        try:
            from agent.metrics import model_calls
            self.last_round_model_calls = model_calls.round_counts()
            _round_detail = model_calls.round_detail()
            _round_duration = model_calls.round_duration()
        except Exception:
            self.last_round_model_calls = {}
            _round_detail = []
            _round_duration = 0.0

        # What the round changed, collected before anything else can move the
        # working tree. Cheap (a blob read and a file read per changed path) and
        # necessarily eager: the pre-images are pinned to the journal window,
        # so deferring this would measure the wrong thing.
        _changeset = self._collect_changeset(turn_id, _changeset_seq)
        self.last_changeset = _changeset

        # Optional one-line "intent" prose on top of the diffstat — a model
        # call, so it is opt-in (config.ui.changeset.prose_summary) and never
        # blocks the round except in "always" mode, which awaits it here so
        # it lands in the same A record written just below.
        _prose_mode = self._changeset_prose_mode()
        if _prose_mode == "always" and _changeset:
            await self._apply_changeset_prose(_changeset)

        if on_changeset is not None and _changeset:
            try:
                on_changeset(_changeset)
            except Exception:
                logger.exception("on_changeset callback failed")

        if self._qa_logger is not None:
            from agent.core.changeset import to_json as _changeset_to_json
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
                    model_calls=_round_detail,
                    duration=_round_duration,
                    changeset=_changeset_to_json(_changeset) if _changeset else None,
                )
            )
            self._pending_bg_tasks.add(task)
            task.add_done_callback(self._pending_bg_tasks.discard)
            from agent.core import background
            background.register_task(task, f"qa-summary turn {turn_id}", "post-turn")

            if _prose_mode == "background" and _changeset:
                from agent.core.changeset_prose import summarize_and_persist as _cs_summarize_and_persist

                async def _prose_bg(_cs=_changeset, _tid=turn_id, _logger=self._qa_logger):
                    try:
                        await _cs_summarize_and_persist(self.config, _cs, _logger, _tid)
                    except Exception:
                        logger.exception("changeset: background prose summary failed")

                prose_task = asyncio.create_task(_prose_bg())
                self._pending_bg_tasks.add(prose_task)
                prose_task.add_done_callback(self._pending_bg_tasks.discard)
                background.register_task(prose_task, f"changeset-prose turn {turn_id}", "post-turn")

        idle_sec = self.config.token_limits.idle_compaction_seconds
        if idle_sec > 0:
            if self._idle_compact_task is not None and not self._idle_compact_task.done():
                self._idle_compact_task.cancel()
            self._idle_compact_task = asyncio.create_task(
                self._idle_compact_loop(idle_sec)
            )
            self._pending_bg_tasks.add(self._idle_compact_task)
            self._idle_compact_task.add_done_callback(self._pending_bg_tasks.discard)
            from agent.core import background
            background.register_task(
                self._idle_compact_task, f"idle-compact in {int(idle_sec)}s", "idle")

        return response

    def set_session_mode(self, mode: str) -> None:
        """Record the session privacy mode and propagate it to persistence sinks.

        "standard" — normal. "incognito" — nothing persists (sessions, notes).
        "private" — incognito plus a hard requirement that every configured LLM
        endpoint is local, enforced per-turn by ``_validate_private_mode``.
        """
        self._session_mode = mode or "standard"
        # Pin mid-turn routing (auto-tier escalation) to local endpoints while
        # private mode is active, mirroring the per-turn _validate_private_mode
        # guard. Non-persisted runtime flag on config, read by escalate_mid_turn.
        try:
            self.config.runtime_local_only = (self._session_mode == "private")
        except Exception:
            logger.debug("set runtime_local_only failed (ignored)", exc_info=True)
        try:
            from agent.tools.notes import notes as _notes
            _notes.set_session_mode(self._session_mode)
        except Exception:
            logger.debug("notes.set_session_mode failed (ignored)", exc_info=True)

    def _validate_private_mode(self) -> None:
        """In private mode, refuse to run if any LLM endpoint is non-local.

        Checks the active endpoint plus every declared model entry, so an
        auto-tier swap can't silently route a private turn through the cloud.

        Raises:
            ValueError: if private mode is on and a non-local endpoint is found.
        """
        if getattr(self, "_session_mode", "standard") != "private":
            return

        candidates: list[tuple[str, str]] = [
            ("active LLM endpoint", getattr(self.config.llm, "base_url", "") or ""),
        ]
        for name, entry in getattr(self.config, "model_entries", {}).items():
            candidates.append((f"model '{name}'", getattr(entry, "base_url", "") or ""))

        for label, url in candidates:
            if url and not is_local_url(url):
                raise ValueError(
                    f"Private mode: {label} uses non-local endpoint {url!r}. "
                    f"Switch to a local endpoint or leave private mode."
                )



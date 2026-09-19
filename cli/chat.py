from __future__ import annotations

import json
import logging
import subprocess
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


# Index coverage moved to agent/core/index_coverage.py so every entrypoint —
# not just this one — injects the block base_rules.txt tells the model to read.
# Re-exported here under the old names for existing callers and tests.
from agent.core.index_coverage import (  # noqa: E402
    COVERAGE_MAX_DIRS as _COVERAGE_MAX_DIRS,
    coverage_map as _get_index_coverage,
    format_coverage as _format_index_coverage,
)


def _bg_update_index(store, embedder, config, result: dict) -> None:
    """Re-index changed files and prune deleted ones. Runs in a daemon thread."""
    try:
        from agent.rag.indexer import index_directory, prune_index
        from agent.rag.archive import ArchiveStore
        from agent.tools.rules.core import load_rules

        working_dir = config.tools.working_dir
        load_rules(working_dir)

        stats = index_directory(
            root=working_dir,
            store=store,
            embedder=embedder,
            cfg=config.rag,
            force=False,
        )
        result["indexed"] = stats["indexed"]
        result["chunks"] = stats["chunks"]

        archive = ArchiveStore(config.rag.archive_db_path)
        pruned = prune_index(working_dir, store, archive)
        archive.purge_expired(config.rag.archive_ttl_days)
        archive.close()
        result["pruned_files"] = len(pruned["paths"])
    except Exception as exc:
        logger.debug("bg index update failed: %s", exc)
        result["error"] = str(exc)


_UI_MODES = {
    "1": ("textual", "Textual — full TUI, scrollable panes, token bar"),
    "2": ("simple",  "Simple  — flowing terminal, Rich markdown, /commands"),
    "3": ("http",    "HTTP    — local web server, chat in your browser"),
}


def _pick_ui_mode(current: str) -> str:
    from agent.ui.prefs import load_prefs, save_prefs
    prefs = load_prefs()
    saved = prefs.get("ui_mode")
    if saved in {m for m, _ in _UI_MODES.values()}:
        return saved

    from rich.console import Console
    console = Console()
    console.print("\n[bold]Choose UI mode[/bold]")
    for key, (mode, desc) in _UI_MODES.items():
        marker = " [cyan]←[/cyan]" if mode == current else ""
        console.print(f"  [cyan]{key}[/cyan]  {desc}{marker}")
    console.print(f"\n  [dim]Enter to keep current ({current}), or set in agent.toml to skip this prompt[/dim]")
    try:
        choice = input("  Mode [1/2/3]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return current
    chosen = _UI_MODES.get(choice, (current,))[0]
    try:
        prefs["ui_mode"] = chosen
        save_prefs(prefs)
    except Exception:
        pass
    return chosen


def _is_first_run() -> bool:
    from agent.config.loader import CONFIG_FILENAMES
    paths = [
        *(Path.home() / ".config" / "agent" / name for name in CONFIG_FILENAMES),
        *(Path(name) for name in CONFIG_FILENAMES),
    ]
    return not any(p.exists() for p in paths)


def _extract_written_files(messages: list[dict]) -> list[str]:
    """Return file paths written/edited during a session from its message history."""
    files: list[str] = []
    write_tools = {"edit_file", "write_file", "patch_file", "replace_text"}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            if name not in write_tools:
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                continue
            if name == "edit_file":
                for chunk in args.get("chunks") or []:
                    p = chunk.get("path", "") if isinstance(chunk, dict) else ""
                    if p and p not in files:
                        files.append(p)
            else:
                p = args.get("path", "")
                if p and p not in files:
                    files.append(p)
    return files


def _audit_crash(console, sentinel: Path, messages: list[dict]) -> None:
    """If sentinel exists (prior crash), warn and show git-dirty written files."""
    if not sentinel.exists():
        return
    console.print("[yellow]Warning: previous run of this session may have crashed.[/yellow]")
    # These land before any UI exists; record them so an HTTP session replays
    # them into the browser instead of leaving them on a terminal nobody reads.
    from agent import ui_notice
    ui_notice.record("Warning: previous run of this session may have crashed.")
    written = _extract_written_files(messages)
    if not written:
        return
    dirty: list[str] = []
    for f in written:
        try:
            r = subprocess.run(
                ["git", "diff", "--name-only", "HEAD", "--", f],
                capture_output=True, text=True, timeout=5,
            )
            if r.stdout.strip():
                dirty.append(f)
        except Exception:
            pass
    if dirty:
        console.print("[yellow]Files modified last session differ from git HEAD:[/yellow]")
        for f in dirty:
            console.print(f"  {f}")
        console.print("  Restore: [bold]git checkout HEAD -- <file>[/bold]")
        ui_notice.record(
            "Files modified last session differ from git HEAD:\n"
            + "\n".join(f"  {f}" for f in dirty)
            + "\n  Restore: git checkout HEAD -- <file>")


def _run_teardown_steps(console, steps) -> None:
    """Run session-end steps with a live line each: what runs, how long, what it
    produced. Ctrl+C skips the current and remaining steps; errors are logged
    and the next step still runs."""
    import time
    console.print("[dim]Session end — learning from this session (Ctrl+C again to skip)…[/dim]")
    for label, unit, fn in steps:
        t0 = time.monotonic()
        try:
            with console.status(f"[dim]  {label}…[/dim]"):
                out = fn()
        except KeyboardInterrupt:
            console.print(f"[yellow]  skipped: {label} and the rest[/yellow]")
            return
        except Exception:
            logger.debug("teardown step %r failed", label, exc_info=True)
            console.print(f"[dim]  {label}: failed ({time.monotonic() - t0:.1f}s, see agent.log)[/dim]")
            continue
        n = len(out) if isinstance(out, (list, tuple)) else out if isinstance(out, int) else 0
        result = f"{n} {unit}" if n else "nothing new"
        console.print(f"[dim]  {label}: {result} ({time.monotonic() - t0:.1f}s)[/dim]")


def _warn_loop_guard_resume(console, messages: list[dict]) -> None:
    """Warn user when resuming a session that was stopped by the loop guard."""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            content = m.get("content") or ""
            from agent.core import markers
            if markers.strip(content).strip().startswith("[loop guard:"):
                note = (
                    "Note: last session ended with a loop-guard stop:\n"
                    f"  {content.strip()[:200]}\n"
                    "The agent will see this in history. Type a message to "
                    "redirect it (e.g. 're-read the file and retry')."
                )
                console.print(f"[yellow]{note.splitlines()[0]}[/yellow]")
                console.print(f"  {content.strip()[:200]}")
                console.print(
                    "[yellow]The agent will see this in history. "
                    "Type a message to redirect it (e.g. 're-read the file and retry').[/yellow]"
                )
                from agent import ui_notice
                ui_notice.record(note)
            break


def cmd_chat(args, config):
    # The background warmup (endpoint probes, heavy imports) was started before
    # the startup prompts. Everything below needs it done.
    from agent.cli import warmup
    warmup.join()
    from agent.rag.store import VectorStore
    from agent.rag.embedder import Embedder
    from agent.core.agent import Agent
    from agent.ui.terminal import run_ui
    from agent.memory.session import new_session, load_session, save_session
    from agent.data_provider import LocalDataProvider
    from rich.console import Console
    import os

    console = Console()

    if args.model:
        config.llm.model = args.model
    if args.ctx:
        config.llm.ctx_window = args.ctx

    if getattr(args, "ui", None):
        config.ui.mode = args.ui
    elif not os.environ.get("AGENT_UI_MODE"):
        config.ui.mode = _pick_ui_mode(config.ui.mode)
    if getattr(args, "http_sidecar", False):
        config.ui.http_sidecar = True

    # Extra Origin/Host allow-list hosts: config (agent.toml/agent.yaml) plus
    # any --allow-host flags. Published to the environment so validate_origin_host
    # (and router-spawned project processes) honour it without config access.
    for _h in (getattr(args, "allow_hosts", None) or []):
        _h = str(_h).strip()
        if _h and _h not in config.ui.allowed_hosts:
            config.ui.allowed_hosts.append(_h)
    _extra_hosts = [str(h).strip() for h in (config.ui.allowed_hosts or []) if str(h).strip()]
    if _extra_hosts:
        os.environ["AGENT_ALLOWED_HOSTS"] = ",".join(_extra_hosts)

    if _is_first_run():
        console.print(
            "[yellow]No agent.toml found.[/yellow] Using defaults "
            f"(model=[bold]{config.llm.model}[/bold]  endpoint=[bold]{config.llm.base_url}[/bold]).\n"
            "  Create [bold]agent.toml[/bold] in this directory to customise settings.\n"
        )

    import sys as _sys
    _agent_dir = Path(config.tools.working_dir) / config.tools.agent_dir
    _configured = _agent_dir / ".configured"
    _initialized = _agent_dir / ".initialized"

    if not _agent_dir.exists() or (not _configured.exists() and not _initialized.exists()):
        console.print("[red]Not initialized.[/red] Run [bold]agent init[/bold] first.")
        _sys.exit(1)

    _reuse_store = None
    _db_path_check = Path(config.rag.db_path)
    _is_indexed = False

    if _db_path_check.exists():
        try:
            _guard = VectorStore(config.rag)
            if _guard.stats()["files"] > 0:
                _is_indexed = True
                _reuse_store = _guard
                if not _initialized.exists():
                    _initialized.touch()
            else:
                _guard.close()
        except Exception:
            pass

    if not _is_indexed:
        console.print("\n[yellow]Project not indexed.[/yellow] Semantic search unavailable.")
        console.print("  [cyan]w[/cyan]  Index whole project now")
        console.print("  [cyan]p[/cyan]  Index a specific path (subtree)")
        console.print("  [cyan]s[/cyan]  Skip — use grep/read tools (faster start)")
        try:
            _idx_choice = input("  Index? [w/p/s]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            _idx_choice = "s"

        if _idx_choice in ("w", "whole"):
            from agent.cli.index import _run_indexing
            _run_indexing(config, console)
            if _db_path_check.exists():
                try:
                    _reuse_store = VectorStore(config.rag)
                    if _reuse_store.stats()["files"] > 0:
                        _is_indexed = True
                    else:
                        _reuse_store.close()
                        _reuse_store = None
                except Exception:
                    pass
        elif _idx_choice in ("p", "path"):
            try:
                _idx_path = input("  Path to index (relative to project root): ").strip()
            except (EOFError, KeyboardInterrupt):
                _idx_path = ""
            if _idx_path:
                from agent.cli.index import _run_indexing
                _idx_root = str(Path(config.tools.working_dir) / _idx_path)
                _run_indexing(config, console, root=_idx_root)
                if _db_path_check.exists():
                    try:
                        _reuse_store = VectorStore(config.rag)
                        if _reuse_store.stats()["files"] > 0:
                            _is_indexed = True
                        else:
                            _reuse_store.close()
                            _reuse_store = None
                    except Exception:
                        pass
            else:
                console.print("  No path given. Skipping.")
        else:
            console.print("[dim]Skipping indexing. Using grep/read/shell tools for code search.[/dim]")

    store = None
    embedder = None
    asm_store = None
    _bg_thread: threading.Thread | None = None
    _bg_result: dict = {}
    _start_maintainer = False
    db_path = Path(config.rag.db_path)
    if _is_indexed and db_path.exists():
        try:
            store = _reuse_store or VectorStore(config.rag)
            embedder = Embedder(config.embeddings)
            # An index whose vectors were produced by a different embedder must
            # not be silently updated: the background thread would write the new
            # model's vectors into the old table and blend two vector spaces.
            from agent.rag.mismatch import handle_embedding_mismatch
            store, _emb_action = handle_embedding_mismatch(
                store, config, console, interactive=_sys.stdin.isatty()
            )
            if _emb_action == "aborted":
                _sys.exit(1)
            if config.asm.enabled:
                from agent.rag.asm_store import AsmStore
                asm_store = AsmStore(config.rag)
            if _emb_action != "frozen" and config.rag.auto_index:
                # Started below, once the agent exists: it gates on turn state.
                _start_maintainer = True
            elif _emb_action != "frozen":
                _bg_thread = threading.Thread(
                    target=_bg_update_index,
                    args=(store, embedder, config, _bg_result),
                    daemon=True,
                    name="bg-index-update",
                )
                _bg_thread.start()
        except Exception as e:
            console.print(f"[yellow]Warning: could not load index: {e}[/yellow]")
            from agent import ui_notice
            ui_notice.record(f"Warning: could not load index: {e}", error=True)

    data_provider = LocalDataProvider(store=store, embedder=embedder, asm_store=asm_store, config=config)
    agent = Agent(config, data_provider=data_provider)

    maintainer = None
    if _start_maintainer:
        try:
            from agent.rag.maintainer import IndexMaintainer
            maintainer = IndexMaintainer(
                config,
                is_busy=lambda: agent._turn_busy,
                last_activity=lambda: agent._last_turn_time,
            )
            maintainer.start()
        except Exception:
            logger.warning("background index maintenance did not start", exc_info=True)

    # Index coverage is injected by Agent.__init__ for every entrypoint.

    _mode = "private" if getattr(args, "private", False) else (
        "incognito" if getattr(args, "incognito", False) else (
            "vault" if getattr(args, "vault", False) else "standard"
        )
    )
    if _mode == "vault":
        # Unlock before anything opens a store: sqlite paths and file targets
        # are chosen at open time from the mode, so a late unlock would leave
        # this session writing in the clear.
        from agent.security import vault as _vault
        _vault.set_mode("vault")
        try:
            _vault.prompt_and_unlock(
                Path(config.tools.working_dir) / config.tools.agent_dir)
        except _vault.VaultError as exc:
            # Refuse to start rather than fall back to standard mode: a session
            # that quietly persists in the clear is the one failure this feature
            # cannot have.
            console.print(f"[red]Vault: {exc}[/red]")
            raise SystemExit(1)
    if args.session:
        session, messages = load_session(args.session)
        if session is None:
            if _mode != "vault":
                # A sealed session is unreadable without the key, so "not found"
                # would be a misleading answer where a vault exists.
                from agent.security import vault as _vault
                if _vault.header_path(
                        Path(config.tools.working_dir) / config.tools.agent_dir).exists():
                    console.print("[yellow]This project has a vault. If that session "
                                  "was sealed, resume it with --vault.[/yellow]")
            session = new_session(short_name=args.session, mode=_mode)
            messages = []
        elif _mode != "standard":
            session.mode = _mode
        if messages:
            # Same rule as the UI server: only the transient note marker is
            # dropped. Sweeping every "_" key here cost the resumed session its
            # side-log links and stored reasoning (see
            # ui_server.local.LocalServer.load_session).
            messages = [{k: v for k, v in m.items() if k != "_notes_marker"}
                        for m in messages]
            agent.messages = messages
            console.print(f"Loaded session: {session.id} ({len(messages)} messages)")
            _warn_loop_guard_resume(console, messages)
    else:
        session = new_session(mode=_mode)
        console.print(f"New session: {session.id}")
    if session.mode != "standard":
        console.print(f"[yellow]Session mode: {session.mode}[/yellow]")

    from agent.memory.session import get_session_full_dir
    _sentinel = get_session_full_dir(session.id) / "running"
    if args.session:
        _audit_crash(console, _sentinel, agent.messages)
    try:
        _sentinel.parent.mkdir(parents=True, exist_ok=True)
        _sentinel.write_text("")
    except Exception:
        pass

    # Expose session on agent so planning helpers can tag plans with session_id.
    agent.session = session
    agent.set_session_mode(session.mode)

    # Scheduled jobs: idle-kind jobs hook into the idle sweep; time-based jobs
    # fire from a ticker thread that defers to the interactive agent.
    _sched_stop = None
    if getattr(config, "scheduler", None) and config.scheduler.enabled:
        try:
            from agent.core.scheduler import start_ticker, register_idle_hook
            register_idle_hook()
            _sched_stop = start_ticker(config, agent)
        except Exception:
            logger.debug("scheduler start failed", exc_info=True)

    # Unapproved project hooks: a cloned repo's agent.toml can ship shell that
    # would run un-sandboxed on the first tool call. They are inert until
    # approved; say so once, at the point the user can act on it.
    try:
        from agent.security.hook_trust import session_warning
        _hw = session_warning(agent.config)
        if _hw:
            console.print(f"[red]{_hw}[/red]")
    except Exception:
        logger.debug("hook trust warning failed", exc_info=True)

    # Optional action classifier: say once when it is not there; the user
    # accepts running without it via /classify accept.
    try:
        from agent.classify import startup_warning as _cls_warning
        _cw = _cls_warning(agent.config)
        if _cw:
            console.print(_cw, style="yellow", markup=False, highlight=False)
            from agent import ui_notice
            ui_notice.record(_cw)
    except Exception:
        logger.debug("classifier startup check failed", exc_info=True)

    # Tamper check: warn if sealed skills/config drifted, or pinned weights moved.
    try:
        from agent.security.integrity import warn_if_tampered
        from agent.security.weightvault import warn_if_drift
        for _w in (warn_if_tampered(agent.config), warn_if_drift(agent.config)):
            if _w:
                console.print(f"[red]{_w}[/red]")
                from agent import ui_notice
                ui_notice.record(_w, error=True)
    except Exception:
        pass

    # Multi-agent presence: announce this agent on the worktree and warn loudly
    # if another agent (owncoder/Claude/Gemini/Hermes) is already editing here.
    try:
        from agent import coord as _coord
        _wd = config.tools.working_dir
        _coord.prune(_wd)
        _coord.heartbeat(_wd, agent="owncoder", tool="owncoder", note=config.llm.model)
        _others = _coord.list_active(_wd)
        if _others:
            console.print(f"[yellow]⚠ {_coord.summary(_wd)}[/yellow]")
            console.print(
                "[yellow]  Shared worktree — coordinate before editing/building "
                "(see AGENTS.md → Multi-agent coordination).[/yellow]"
            )
            from agent import ui_notice
            ui_notice.record(
                f"⚠ {_coord.summary(_wd)}\n"
                "  Shared worktree — coordinate before editing/building "
                "(see AGENTS.md → Multi-agent coordination).")
    except Exception:
        logger.debug("coord presence announce failed", exc_info=True)

    try:
        active_session = run_ui(agent, session=session)
        if active_session is not None:
            session = active_session
    except BaseException as exc:
        try:
            if config.recovery.enabled and not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                from agent.planning import recovery as _rec
                last_user = ""
                for m in reversed(agent.messages or []):
                    if m.get("role") == "user":
                        last_user = str(m.get("content", ""))[:1000]
                        break
                active_plan_id = None
                try:
                    from agent.planning import list_plans
                    for p in list_plans():
                        if p.status == "active" and (not session or p.session_id == session.id):
                            active_plan_id = p.id
                            break
                except Exception:
                    pass
                _rec.record_crash(
                    session_id=session.id,
                    exc=exc,
                    plan_id=active_plan_id,
                    last_user_message=last_user,
                )
        except Exception:
            pass
        raise
    finally:
        if _sched_stop is not None:
            _sched_stop.set()
        try:
            _sentinel.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            from agent import coord as _coord
            _coord.clear(config.tools.working_dir, agent="owncoder")
        except Exception:
            pass
        save_session(session, agent.messages)
        # Session-end learning: up to three LLM calls (tens of seconds on a LAN
        # model). Shown step by step; a second Ctrl+C skips the rest instead of
        # aborting the whole teardown (stores and MCP would stay open).
        _facts = getattr(agent, "_facts_store", None)

        def _promote():
            from agent.memory.promoter import promote_session_to_notes
            return promote_session_to_notes(session_id=session.id, config=config, facts_store=_facts,
                                            embedder=embedder, session_mode=session.mode)

        def _reflect():
            from agent.memory.reflector import reflect_session
            return reflect_session(session_id=session.id, config=config, facts_store=_facts,
                                   embedder=embedder,
                                   store=getattr(agent, "_project_memory_store", None))

        def _distill():
            from agent.memory.skill_distiller import distill_session_skills
            return distill_session_skills(session_id=session.id, config=config, facts_store=_facts)

        def _evaluate_prompts():
            # A/B verdict on compiled prompts: recompile/pin variants that
            # measurably regress vs their original-text control arm.
            from agent import prompt_compiler
            return prompt_compiler.evaluate(config)

        _run_teardown_steps(console, [
            ("saving notes", "note(s)", _promote),
            ("reflecting on the session", "rule(s)", _reflect),
            ("distilling skills", "skill(s)", _distill),
            ("checking compiled prompts", "verdict(s)", _evaluate_prompts),
        ])
        try:
            from agent.mcp import shutdown_mcp
            shutdown_mcp()
        except Exception:
            pass
        if maintainer is not None:
            maintainer.stop()
        if _bg_thread and _bg_thread.is_alive():
            _bg_thread.join(timeout=5)
        if _bg_result.get("error"):
            console.print(f"[yellow]Background index update failed: {_bg_result['error']}[/yellow]")
        elif _bg_result.get("indexed", 0) > 0:
            console.print(
                f"[dim]Index updated: {_bg_result['indexed']} file(s) re-indexed "
                f"({_bg_result.get('chunks', 0)} chunks)"
                + (f", {_bg_result['pruned_files']} file(s) pruned" if _bg_result.get("pruned_files") else "")
                + "[/dim]"
            )
        if store:
            store.close()
        if asm_store:
            asm_store.close()

from __future__ import annotations

import json
import logging
import subprocess
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


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
        choice = input("  Mode [1/2]: ").strip()
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
    paths = [
        Path.home() / ".config" / "agent" / "agent.toml",
        Path("agent.toml"),
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


def cmd_chat(args, config):
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

    if _is_first_run():
        console.print(
            "[yellow]No agent.toml found.[/yellow] Using defaults "
            f"(model=[bold]{config.llm.model}[/bold]  endpoint=[bold]{config.llm.base_url}[/bold]).\n"
            "  Create [bold]agent.toml[/bold] in this directory to customise settings.\n"
        )

    import sys as _sys
    _marker = Path(config.tools.working_dir) / config.tools.agent_dir / ".initialized"
    _db_path_check = Path(config.rag.db_path)
    _reuse_store = None
    if not _marker.exists():
        _ERR_MSG = (
            "[red]Index not initialized.[/red] Run [bold]agent init[/bold] first.\n"
            "  The agent will not work correctly without a completed index."
        )
        if _db_path_check.exists():
            try:
                _guard = VectorStore(config.rag)
                if _guard.stats()["files"] > 0:
                    _marker.touch()
                    _reuse_store = _guard
                else:
                    _guard.close()
                    console.print(_ERR_MSG)
                    _sys.exit(1)
            except Exception:
                console.print(_ERR_MSG)
                _sys.exit(1)
        else:
            console.print(_ERR_MSG)
            _sys.exit(1)

    store = None
    embedder = None
    asm_store = None
    _bg_thread: threading.Thread | None = None
    _bg_result: dict = {}
    db_path = Path(config.rag.db_path)
    if db_path.exists():
        try:
            store = _reuse_store or VectorStore(config.rag)
            embedder = Embedder(config.embeddings)
            if config.asm.enabled:
                from agent.rag.asm_store import AsmStore
                asm_store = AsmStore(config.rag)
            _bg_thread = threading.Thread(
                target=_bg_update_index,
                args=(store, embedder, config, _bg_result),
                daemon=True,
                name="bg-index-update",
            )
            _bg_thread.start()
        except Exception as e:
            console.print(f"[yellow]Warning: could not load index: {e}[/yellow]")
    else:
        console.print(
            "[yellow]No index found[/yellow] — code search disabled. "
            "Run [bold]agent init[/bold] to build one."
        )

    data_provider = LocalDataProvider(store=store, embedder=embedder, asm_store=asm_store, config=config)
    agent = Agent(config, data_provider=data_provider)

    if args.session:
        session, messages = load_session(args.session)
        if session is None:
            session = new_session(short_name=args.session)
            messages = []
        if messages:
            messages = [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]
            agent.messages = messages
            console.print(f"Loaded session: {session.id} ({len(messages)} messages)")
    else:
        session = new_session()
        console.print(f"New session: {session.id}")

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
        try:
            _sentinel.unlink(missing_ok=True)
        except Exception:
            pass
        save_session(session, agent.messages)
        try:
            from agent.memory.promoter import promote_session_to_notes
            promote_session_to_notes(
                session_id=session.id,
                config=config,
                facts_store=getattr(agent, "_facts_store", None),
                embedder=embedder,
            )
        except Exception:
            pass
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

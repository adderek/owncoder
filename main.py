#!/usr/bin/env python3
"""local-code-agent — entry point."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def cmd_init(args, config):
    from agent.rag.indexer import index_directory, LANGUAGE_MAP
    from agent.rag.store import VectorStore
    from agent.rag.embedder import Embedder
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()
    store = VectorStore(config.rag)
    embedder = Embedder(config.embeddings)

    languages = args.languages.split(",") if args.languages else None
    exclude = args.exclude.split(",") if args.exclude else []
    working_dir = config.tools.working_dir

    console.print(f"[bold]Indexing[/bold] {Path(working_dir).resolve()}")
    if languages:
        console.print(f"Languages: {', '.join(languages)}")

    def progress_cb(path: str, chunk_count: int) -> None:
        console.print(f"  [dim]{path}[/dim] → {chunk_count} chunks")

    stats = index_directory(
        root=working_dir,
        store=store,
        embedder=embedder,
        cfg=config.rag,
        languages=languages,
        exclude=exclude,
        force=getattr(args, "force", False),
        progress_cb=progress_cb,
    )

    store.close()
    console.print(f"\n[green]Done.[/green] Indexed {stats['indexed']} files, "
                  f"skipped {stats['skipped']}, "
                  f"created {stats['chunks']} chunks.")


def cmd_index_update(args, config):
    from agent.rag.indexer import index_directory
    from agent.rag.store import VectorStore
    from agent.rag.embedder import Embedder
    from rich.console import Console
    import subprocess

    console = Console()

    # Get list of changed files from git
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=config.tools.working_dir,
            capture_output=True, text=True,
        )
        changed = [f.strip() for f in result.stdout.splitlines() if f.strip()]
    except Exception:
        changed = None

    store = VectorStore(config.rag)
    embedder = Embedder(config.embeddings)

    if changed is not None and not changed:
        console.print("No changed files detected. Index is up to date.")
        store.close()
        return

    stats = index_directory(
        root=config.tools.working_dir,
        store=store,
        embedder=embedder,
        cfg=config.rag,
    )
    store.close()
    console.print(f"Updated: {stats['indexed']} files re-indexed, {stats['skipped']} unchanged.")


def cmd_index_stats(args, config):
    from agent.rag.store import VectorStore
    from rich.console import Console

    store = VectorStore(config.rag)
    stats = store.stats()
    store.close()

    console = Console()
    console.print(f"[bold]Index stats:[/bold]")
    console.print(f"  Files:  {stats['files']}")
    console.print(f"  Chunks: {stats['chunks']}")
    console.print(f"  DB:     {config.rag.db_path}")


_UI_MODES = {
    "1": ("textual", "Textual — full TUI, scrollable panes, token bar"),
    "2": ("simple",  "Simple  — flowing terminal, Rich markdown, /commands"),
}


def _pick_ui_mode(current: str) -> str:
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
    return _UI_MODES.get(choice, (current,))[0]


def cmd_chat(args, config):
    from agent.rag.store import VectorStore
    from agent.rag.embedder import Embedder
    from agent.agent import Agent
    from agent.ui.terminal import run_ui
    from agent.memory.session import load_session, save_session
    from rich.console import Console
    import os

    console = Console()

    # Apply CLI overrides
    if args.model:
        config.llm.model = args.model
    if args.ctx:
        config.llm.ctx_window = args.ctx

    # UI mode: CLI flag > env var > prompt (skip prompt if config was explicit)
    if getattr(args, "ui", None):
        config.ui.mode = args.ui
    elif not os.environ.get("AGENT_UI_MODE"):
        config.ui.mode = _pick_ui_mode(config.ui.mode)

    store = None
    embedder = None
    db_path = Path(config.rag.db_path)
    if db_path.exists():
        try:
            store = VectorStore(config.rag)
            embedder = Embedder(config.embeddings)
        except Exception as e:
            console.print(f"[yellow]Warning: could not load index: {e}[/yellow]")
    else:
        console.print("[yellow]No index found. Run 'agent init' to build one.[/yellow]")

    agent = Agent(config, store=store, embedder=embedder)

    session_name = args.session or "default"
    messages, meta = load_session(session_name)
    if messages:
        agent.messages = messages
        console.print(f"Loaded session: {session_name} ({len(messages)} messages)")

    try:
        run_ui(agent, session_name=session_name)
    finally:
        save_session(session_name, agent.messages)
        if store:
            store.close()


def cmd_run(args, config):
    from agent.rag.store import VectorStore
    from agent.rag.embedder import Embedder
    from agent.agent import Agent
    from rich.console import Console

    console = Console()

    store = None
    embedder = None
    db_path = Path(config.rag.db_path)
    if db_path.exists():
        try:
            store = VectorStore(config.rag)
            embedder = Embedder(config.embeddings)
        except Exception:
            pass

    agent = Agent(config, store=store, embedder=embedder)

    def on_tool(name: str, args_str: str) -> None:
        console.print(f"  [dim]→ {name}[/dim]")

    async def _run():
        response = await agent.chat(args.prompt, on_tool_call=on_tool)
        console.print(response)
        if config.ui.show_token_count:
            console.print(f"[dim][tokens: {agent.token_estimate()}/{config.llm.ctx_window}][/dim]")

    asyncio.run(_run())

    if store:
        store.close()


def cmd_sessions(args, config):
    from agent.memory.session import list_sessions, load_session
    from rich.console import Console
    from rich.table import Table
    import datetime

    console = Console()

    if args.load:
        messages, meta = load_session(args.load)
        if not messages:
            console.print(f"Session '{args.load}' not found.")
            return
        console.print(f"Session '{args.load}': {len(messages)} messages")
        return

    sessions = list_sessions()
    if not sessions:
        console.print("No sessions found.")
        return

    table = Table(title="Sessions")
    table.add_column("Name")
    table.add_column("Messages")
    table.add_column("Saved At")

    for s in sessions:
        saved = datetime.datetime.fromtimestamp(s["saved_at"]).strftime("%Y-%m-%d %H:%M") if s["saved_at"] else "?"
        table.add_row(s["name"], str(s["message_count"]), saved)

    console.print(table)


def cmd_debug_context(args, config):
    from agent.memory.session import load_session
    from rich.console import Console
    import json

    console = Console()
    session_name = args.session or "default"
    messages, _ = load_session(session_name)
    if not messages:
        console.print("No session loaded.")
        return

    for i, m in enumerate(messages):
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            content = json.dumps(content)
        console.print(f"[bold]{i}. {role}:[/bold] {str(content)[:200]}")


def main():
    parser = argparse.ArgumentParser(prog="agent", description="Local code agent")
    parser.add_argument("--config", type=str, help="Path to agent.toml")

    sub = parser.add_subparsers(dest="command")

    # init
    init_p = sub.add_parser("init", help="Initialize index for current directory")
    init_p.add_argument("--languages", type=str, help="Comma-separated languages: py,js,kt,cpp")
    init_p.add_argument("--exclude", type=str, help="Comma-separated paths to exclude")
    init_p.add_argument("--force", action="store_true", help="Force re-index all files")

    # index
    idx_p = sub.add_parser("index", help="Manage index")
    idx_p.add_argument("--update", action="store_true", help="Re-index changed files")
    idx_p.add_argument("--stats", action="store_true", help="Show index statistics")

    # chat
    chat_p = sub.add_parser("chat", help="Start interactive session")
    chat_p.add_argument("--model", type=str, help="Override model name")
    chat_p.add_argument("--ctx", type=int, help="Override context window size")
    chat_p.add_argument("--session", type=str, help="Session name to load/save")
    chat_p.add_argument("--ui", type=str, choices=["textual", "simple"],
                        help="UI mode (skips the prompt)")

    # run
    run_p = sub.add_parser("run", help="Run a single prompt non-interactively")
    run_p.add_argument("prompt", type=str, help="Prompt to run")

    # sessions
    sess_p = sub.add_parser("sessions", help="Manage sessions")
    sess_p.add_argument("--list", action="store_true", help="List sessions")
    sess_p.add_argument("--load", type=str, help="Show session details")

    # debug
    dbg_p = sub.add_parser("debug", help="Debug utilities")
    dbg_p.add_argument("--context", action="store_true", help="Show full context of current session")
    dbg_p.add_argument("--session", type=str, help="Session name")

    args = parser.parse_args()

    from agent.config import load_config, check_reachability
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    if args.command == "init":
        cmd_init(args, config)
    elif args.command == "index":
        if args.update:
            cmd_index_update(args, config)
        elif args.stats:
            cmd_index_stats(args, config)
        else:
            parser.parse_args(["index", "--help"])
    elif args.command == "chat":
        check_reachability(config)
        cmd_chat(args, config)
    elif args.command == "run":
        check_reachability(config)
        cmd_run(args, config)
    elif args.command == "sessions":
        cmd_sessions(args, config)
    elif args.command == "debug":
        cmd_debug_context(args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

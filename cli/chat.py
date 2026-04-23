from __future__ import annotations

from pathlib import Path


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


def _is_first_run() -> bool:
    paths = [
        Path.home() / ".config" / "agent" / "agent.toml",
        Path("agent.toml"),
    ]
    return not any(p.exists() for p in paths)


def _find_project_root(start_dir: Path, search_parents: bool) -> Path | None:
    curr = start_dir.resolve()
    while True:
        if (curr / ".agent").is_dir():
            return curr
        if not search_parents or curr == curr.parent:
            break
        curr = curr.parent
    return None


def cmd_chat(args, config):
    from agent.rag.store import VectorStore
    from agent.rag.embedder import Embedder
    from agent.agent import Agent
    from agent.ui.terminal import run_ui
    from agent.memory.session import new_session, load_session, save_session
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

    store = None
    embedder = None
    asm_store = None
    db_path = Path(config.rag.db_path)
    if db_path.exists():
        try:
            store = VectorStore(config.rag)
            embedder = Embedder(config.embeddings)
            if config.asm.enabled:
                from agent.rag.asm_store import AsmStore
                asm_store = AsmStore(config.rag)
        except Exception as e:
            console.print(f"[yellow]Warning: could not load index: {e}[/yellow]")
    else:
        console.print(
            "[yellow]No index found[/yellow] — code search disabled. "
            "Run [bold]agent init[/bold] to build one."
        )

    agent = Agent(config, store=store, embedder=embedder, asm_store=asm_store)

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

    try:
        active_session = run_ui(agent, session=session)
        if active_session is not None:
            session = active_session
    finally:
        save_session(session, agent.messages)
        if store:
            store.close()
        if asm_store:
            asm_store.close()

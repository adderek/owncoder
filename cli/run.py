from __future__ import annotations

import asyncio
from pathlib import Path


def cmd_run(args, config):
    import sys
    from agent.rag.store import VectorStore
    from agent.rag.embedder import Embedder
    from agent.core.agent import Agent
    from agent.data_provider import LocalDataProvider
    from rich.console import Console

    console = Console()

    if args.prompt:
        args.prompt_text = args.prompt
    elif not sys.stdin.isatty():
        args.prompt_text = sys.stdin.read().strip()
        if not args.prompt_text:
            console.print("[red]No prompt provided on stdin.[/red]")
            return
    else:
        console.print("[red]Provide a prompt argument or pipe one via stdin.[/red]")
        return

    store = None
    embedder = None
    db_path = Path(config.rag.db_path)
    if db_path.exists():
        try:
            store = VectorStore(config.rag)
            embedder = Embedder(config.embeddings)
        except Exception:
            pass

    data_provider = LocalDataProvider(store=store, embedder=embedder, config=config)
    agent = Agent(config, data_provider=data_provider)

    def on_tool(name: str, args_str: str) -> None:
        console.print(f"  [dim]→ {name}[/dim]")

    def on_tool_result(name: str, ok: bool) -> None:
        if not ok:
            console.print(f"  [dim]✗ {name}[/dim]")

    async def _run():
        response = await agent.chat(args.prompt_text, on_tool_call=on_tool, on_tool_result=on_tool_result)
        console.print(response)
        if config.ui.show_token_count:
            console.print(f"[dim][tokens: {agent.token_estimate()}/{config.llm.ctx_window}][/dim]")

    asyncio.run(_run())

    if store:
        store.close()

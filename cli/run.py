from __future__ import annotations

import asyncio
import json
from pathlib import Path

# Response-text markers emitted by core/turn.py and the ask_user/mark_done
# signal tools (tools/turn_signals.py) — used to classify how a headless run
# ended without touching turn.py itself.
_CAP_MARKERS = ("[goal ceiling", "[iteration limit")
_GOAL_ACHIEVED_MARKER = "[goal achieved"


def _classify_exit(response: str) -> tuple[int, str]:
    """(exit_code, exit_reason) from the trailing marker text in *response*.

    0/"done" is the default for a normal completion (including >>>DONE and
    >>>ASK signals — those are legitimate stopping points for a single
    non-interactive run, not errors). 2/"iteration_cap" or "goal_cap" only
    when run_turn's own ceiling markers are present.
    """
    if _GOAL_ACHIEVED_MARKER in response:
        return 0, "goal_achieved"
    for marker in _CAP_MARKERS:
        if marker in response:
            return 2, "iteration_cap" if "iteration limit" in marker else "goal_cap"
    return 0, "done"


def _warn_unanswerable_asks(config, as_json: bool) -> list[str]:
    """Say up front that `ask` rules will deny here, and return what they are.

    `agent run` is non-interactive, so it registers no permission asker and an
    `ask` verdict fails closed. That is right — a headless run has nobody to
    approve — but it used to surface only as a tool error partway through a run,
    which reads like the agent malfunctioning rather than like policy.

    Deliberately a warning, not a flag that grants: nothing here should offer a
    way to turn `ask` into `allow` without a human answering.
    """
    import sys

    from agent.security import permissions as _permissions

    try:
        sources = _permissions.unanswerable_asks(config)
    except Exception:
        return []
    if not sources:
        return []
    shown = ", ".join(sources[:4]) + (f" (+{len(sources) - 4} more)" if len(sources) > 4 else "")
    # stderr, so --json stdout stays a single parseable object.
    print(f"warning: no interactive UI, so 'ask' permission verdicts will deny: {shown}\n"
          f"         review with 'agent permissions', or use 'agent chat' to answer them.",
          file=sys.stderr)
    return sources


def cmd_run(args, config):
    import sys
    from agent.cli import warmup
    warmup.join()
    from agent.rag.store import VectorStore
    from agent.rag.embedder import Embedder
    from agent.core.agent import Agent
    from agent.data_provider import LocalDataProvider
    from rich.console import Console

    as_json = bool(getattr(args, "json", False))
    console = Console(quiet=as_json)

    if args.prompt:
        args.prompt_text = args.prompt
    elif not sys.stdin.isatty():
        args.prompt_text = sys.stdin.read().strip()
        if not args.prompt_text:
            _fail(as_json, "No prompt provided on stdin.", console)
            sys.exit(1)
    else:
        _fail(as_json, "Provide a prompt argument or pipe one via stdin.", console)
        sys.exit(1)

    _warn_unanswerable_asks(config, as_json)

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

    tool_calls: list[str] = []

    def on_tool(name: str, args_str: str) -> None:
        tool_calls.append(name)
        console.print(f"  [dim]→ {name}[/dim]")

    def on_tool_result(name: str, ok: bool) -> None:
        if not ok:
            console.print(f"  [dim]✗ {name}[/dim]")

    result: dict = {}

    async def _run():
        try:
            response = await agent.chat(args.prompt_text, on_tool_call=on_tool, on_tool_result=on_tool_result)
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            return
        result["response"] = response

    asyncio.run(_run())

    if store:
        store.close()

    if "error" in result:
        _fail(as_json, result["error"], console)
        sys.exit(1)

    response = result["response"]
    exit_code, exit_reason = _classify_exit(response)

    if as_json:
        print(json.dumps({
            "result": response,
            "exit_reason": exit_reason,
            "tool_calls": tool_calls,
            "tokens": agent.token_estimate(),
        }, ensure_ascii=False))
    else:
        console.print(response)
        if config.ui.show_token_count:
            console.print(f"[dim][tokens: {agent.token_estimate()}/{config.llm.ctx_window}][/dim]")

    if exit_code:
        sys.exit(exit_code)


def _fail(as_json: bool, message: str, console) -> None:
    if as_json:
        print(json.dumps({"error": message}, ensure_ascii=False))
    else:
        console.print(f"[red]{message}[/red]")

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path


def cmd_commit(args, config):
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt

    console = Console()

    path = Path(args.path)
    if not path.is_absolute():
        path = Path(config.tools.working_dir) / path
    path = path.resolve()

    if not path.is_dir():
        console.print(f"[red]Not a directory: {path}[/red]")
        return

    result = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=str(path),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]Not a git repository: {path}[/red]")
        return

    def _git(*git_args: str) -> str:
        r = subprocess.run(["git"] + list(git_args), cwd=str(path), capture_output=True, text=True)
        return r.stdout.strip()

    staged_diff = _git("diff", "--cached")
    if not staged_diff:
        console.print("[yellow]No staged changes found. Stage files first with git add.[/yellow]")
        return

    status = _git("status", "--short")
    recent_log = _git("log", "--oneline", "-10")

    DIFF_CAP = 12000
    diff_chars = len(staged_diff)
    diff_text = staged_diff
    truncated = False
    if diff_chars > DIFF_CAP:
        diff_text = staged_diff[:DIFF_CAP] + "\n[... diff truncated ...]"
        truncated = True

    system_prompt = (
        "You write git commit messages. Output ONLY the commit message: "
        "no preamble, no explanation, no markdown fences, no quotes. "
        "First line: imperative-mood summary, <=72 chars. "
        "Optional body after a blank line, wrapped at 72 chars."
    )
    user_prompt = (
        f"Recent commits (style reference):\n{recent_log or '(none)'}\n\n"
        f"Git status:\n{status}\n\n"
        f"Staged diff:\n{diff_text}\n\n"
        "Write the commit message."
    )

    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text
    from rich.markup import escape as _markup_escape
    import time as _time
    from openai import AsyncOpenAI

    console.print(
        f"[dim]Generating commit message for {path} "
        f"(staged diff: {diff_chars:,} chars"
        f"{', truncated to ' + str(DIFF_CAP) if truncated else ''})…[/dim]"
    )

    state = {"tokens": 0, "buf": "", "start": _time.monotonic()}

    async def _generate() -> str:
        client = AsyncOpenAI(base_url=config.llm.base_url, api_key=config.llm.api_key)
        stream = await client.chat.completions.create(
            model=config.llm.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=config.token_limits.commit_message,
            stream=True,
            extra_body={"reasoning_effort": "low"},
        )
        parts: list[str] = []
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            reasoning = getattr(delta, "reasoning_content", None)
            if content:
                state["tokens"] += 1
                state["buf"] += content
                parts.append(content)
            elif reasoning:
                state["tokens"] += 1
                state["buf"] = ("…thinking: " + reasoning)[-120:]
        return "".join(parts)

    async def _run_with_status() -> str:
        task = asyncio.create_task(_generate())
        spinner = Spinner("dots", text="starting…")
        with Live(spinner, console=console, refresh_per_second=8, transient=True):
            while not task.done():
                elapsed = _time.monotonic() - state["start"]
                preview = _markup_escape(state["buf"].replace("\n", " ")[-60:])
                spinner.update(text=Text.from_markup(
                    f"[cyan]generating[/cyan] · {elapsed:5.1f}s · "
                    f"{state['tokens']} tok · [dim]{preview}[/dim]"
                ))
                await asyncio.sleep(0.15)
        return await task

    message = asyncio.run(_run_with_status()).strip()
    elapsed = _time.monotonic() - state["start"]
    console.print(f"[dim]done in {elapsed:.1f}s · {state['tokens']} tokens[/dim]")

    if message.startswith("```"):
        lines = message.splitlines()
        lines = [l for l in lines if not l.startswith("```")]
        message = "\n".join(lines).strip()

    if not message:
        console.print(
            "[red]Model returned an empty commit message "
            "(no content in stream — likely all output went to reasoning_content "
            "or max_tokens was exhausted during thinking). Aborting.[/red]"
        )
        return

    console.print(Panel(message, title="Proposed commit message", border_style="cyan"))
    choice = Prompt.ask("Commit with this message?", choices=["y", "n", "e"], default="y")

    if choice == "n":
        console.print("[dim]Aborted.[/dim]")
        return

    if choice == "e":
        import tempfile
        import os
        editor = os.environ.get("EDITOR", "vi")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(message)
            tmp = f.name
        subprocess.run([editor, tmp])
        message = Path(tmp).read_text().strip()
        Path(tmp).unlink(missing_ok=True)
        if not message:
            console.print("[red]Empty commit message. Aborting.[/red]")
            return
        console.print(Panel(message, title="Edited commit message", border_style="cyan"))
        confirm = Prompt.ask("Commit?", choices=["y", "n"], default="y")
        if confirm == "n":
            console.print("[dim]Aborted.[/dim]")
            return

    result = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(path),
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        console.print(f"[green]Committed.[/green]\n{result.stdout.strip()}")
    else:
        console.print(f"[red]Commit failed:[/red]\n{result.stderr.strip()}")

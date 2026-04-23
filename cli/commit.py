from __future__ import annotations

import asyncio
import re
import subprocess
from pathlib import Path


# Control tokens the model may emit during chunked summarization.
# The model is told about these in the per-step prompt; if it emits one on the
# first line of its response, we retry the step with extra context.
_REQUEST_PREV_RAW = "NEED_PREVIOUS_RAW"
_REQUEST_PREV_SUMMARY = "NEED_PREVIOUS_SUMMARY"


def _split_diff(diff: str, chunk_chars: int) -> list[str]:
    """Split a unified diff into chunks <= chunk_chars.

    Preference order for split boundaries: per-file (`diff --git`), then per-hunk
    (`@@ ...`), then hard char split. Never splits inside a line.
    """
    if len(diff) <= chunk_chars:
        return [diff]

    # Split into per-file blocks.
    parts = re.split(r"(?m)^(?=diff --git )", diff)
    parts = [p for p in parts if p]

    # Any single file block larger than chunk_chars is further split at hunk
    # boundaries, keeping the file header on each sub-chunk so the model has
    # context about which file it is looking at.
    expanded: list[str] = []
    for p in parts:
        if len(p) <= chunk_chars:
            expanded.append(p)
            continue
        m = re.search(r"(?m)^@@", p)
        header = p[: m.start()] if m else ""
        body = p[m.start():] if m else p
        hunks = re.split(r"(?m)^(?=@@ )", body)
        hunks = [h for h in hunks if h]
        for h in hunks:
            piece = header + h
            if len(piece) <= chunk_chars:
                expanded.append(piece)
            else:
                # Hard split on newline boundaries as last resort.
                lines = piece.splitlines(keepends=True)
                buf, size = [], 0
                for ln in lines:
                    if size + len(ln) > chunk_chars and buf:
                        expanded.append("".join(buf))
                        buf, size = [], 0
                    buf.append(ln)
                    size += len(ln)
                if buf:
                    expanded.append("".join(buf))

    # Coalesce small adjacent pieces up to chunk_chars to minimize round-trips.
    merged: list[str] = []
    for piece in expanded:
        if merged and len(merged[-1]) + len(piece) <= chunk_chars:
            merged[-1] += piece
        else:
            merged.append(piece)
    return merged


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

    chunk_chars = config.token_limits.commit_chunk_chars
    summary_tokens = config.token_limits.commit_summary_tokens
    diff_chars = len(staged_diff)
    chunks = _split_diff(staged_diff, chunk_chars) if diff_chars > chunk_chars else [staged_diff]
    chunked = len(chunks) > 1

    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text
    from rich.markup import escape as _markup_escape
    import time as _time
    from openai import AsyncOpenAI

    if chunked:
        console.print(
            f"[dim]Generating commit message for {path} "
            f"(staged diff: {diff_chars:,} chars → {len(chunks)} chunks of ≤{chunk_chars:,})…[/dim]"
        )
    else:
        console.print(
            f"[dim]Generating commit message for {path} "
            f"(staged diff: {diff_chars:,} chars)…[/dim]"
        )

    client = AsyncOpenAI(base_url=config.llm.base_url, api_key=config.llm.api_key)

    state = {"tokens": 0, "buf": "", "start": _time.monotonic(), "phase": "starting"}

    async def _stream(messages: list[dict], *, max_tokens: int) -> str:
        stream = await client.chat.completions.create(
            model=config.llm.model,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
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

    summary_system = (
        "You summarize a large git diff one chunk at a time. Goal: build a running "
        "summary concise enough that a later step can write a good commit message "
        "from it alone. Output ONLY the updated running summary — no preamble, no "
        "markdown fences. Group related changes; note file paths, added/removed "
        "functions, behavioral changes, and intent. Drop line-level noise.\n\n"
        "If the new chunk is ambiguous without more context you may instead reply "
        f"with EXACTLY `{_REQUEST_PREV_RAW}` on the first line (to get the previous "
        f"chunk in full) or `{_REQUEST_PREV_SUMMARY}` (to get the prior running "
        "summary re-included verbatim). Use these sparingly — at most once per "
        "chunk. Otherwise just output the updated summary."
    )

    async def _summarize_chunk(
        idx: int,
        total: int,
        chunk: str,
        running_summary: str,
        prev_raw: str,
    ) -> str:
        state["phase"] = f"summarize {idx + 1}/{total}"

        def _build_user(include_prev_raw: bool, include_prev_summary: bool) -> str:
            parts: list[str] = []
            parts.append(f"Chunk {idx + 1} of {total}.")
            if running_summary:
                parts.append(f"Running summary so far:\n{running_summary}")
            else:
                parts.append("Running summary so far: (none — this is chunk 1).")
            if include_prev_raw and prev_raw:
                parts.append(f"Previous chunk (raw, as requested):\n{prev_raw}")
            if include_prev_summary and running_summary:
                parts.append(
                    f"(Prior running summary re-included verbatim as requested:\n{running_summary})"
                )
            parts.append(f"New chunk:\n{chunk}")
            parts.append("Output the updated running summary.")
            return "\n\n".join(parts)

        messages = [
            {"role": "system", "content": summary_system},
            {"role": "user", "content": _build_user(False, False)},
        ]
        out = (await _stream(messages, max_tokens=summary_tokens)).strip()

        first_line = out.splitlines()[0].strip() if out else ""
        if first_line == _REQUEST_PREV_RAW and prev_raw:
            messages[-1]["content"] = _build_user(True, False)
            out = (await _stream(messages, max_tokens=summary_tokens)).strip()
        elif first_line == _REQUEST_PREV_SUMMARY and running_summary:
            messages[-1]["content"] = _build_user(False, True)
            out = (await _stream(messages, max_tokens=summary_tokens)).strip()
        return out

    final_system = (
        "You write git commit messages. Output ONLY the commit message: "
        "no preamble, no explanation, no markdown fences, no quotes. "
        "First line: imperative-mood summary, <=72 chars. "
        "Optional body after a blank line, wrapped at 72 chars."
    )

    async def _final_message(diff_or_summary: str, *, from_summary: bool) -> str:
        state["phase"] = "commit message"
        label = "Summary of staged diff" if from_summary else "Staged diff"
        user_prompt = (
            f"Recent commits (style reference):\n{recent_log or '(none)'}\n\n"
            f"Git status:\n{status}\n\n"
            f"{label}:\n{diff_or_summary}\n\n"
            "Write the commit message."
        )
        return (await _stream(
            [
                {"role": "system", "content": final_system},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=config.token_limits.commit_message,
        )).strip()

    async def _run() -> str:
        if not chunked:
            return await _final_message(chunks[0], from_summary=False)
        running = ""
        prev_raw = ""
        for i, chunk in enumerate(chunks):
            running = await _summarize_chunk(i, len(chunks), chunk, running, prev_raw)
            prev_raw = chunk
        return await _final_message(running, from_summary=True)

    async def _run_with_status() -> str:
        task = asyncio.create_task(_run())
        spinner = Spinner("dots", text="starting…")
        with Live(spinner, console=console, refresh_per_second=8, transient=True):
            while not task.done():
                elapsed = _time.monotonic() - state["start"]
                preview = _markup_escape(state["buf"].replace("\n", " ")[-60:])
                spinner.update(text=Text.from_markup(
                    f"[cyan]{state['phase']}[/cyan] · {elapsed:5.1f}s · "
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

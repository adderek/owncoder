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


def _save_problem_report(
    state: dict,
    final_message: str,
    chunked: bool,
    num_chunks: int,
    diff_chars: int,
    config,
    primary_model: str,
    summ_model: str,
    elapsed: float,
    repo_path: Path,
    user_description: str = "",
) -> Path | None:
    """Save raw model outputs + diagnostics for problem report.

    Creates {agent_dir}/problem/commit/{timestamp}/ for future automated analysis.
    Returns report path on success, None on failure.
    """
    import json
    import os
    import platform
    import sys
    from datetime import datetime, timezone

    raw_outputs = state.get("raw_outputs", [])

    # Git state at report time
    git_info = {}
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_path), capture_output=True, text=True,
        ).stdout.strip()
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_path), capture_output=True, text=True,
        ).stdout.strip()
        has_changes = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=str(repo_path),
        ).returncode != 0
        git_info = {
            "branch": branch,
            "sha": sha,
            "has_uncommitted_changes": has_changes,
        }
    except Exception:
        pass

    report = {
        "type": "problem-report",
        "subtype": "commit",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_description": user_description,
        "model": primary_model,
        "summarizer_model": summ_model,
        "chunked": chunked,
        "num_chunks": num_chunks,
        "tokens_used": state.get("tokens", 0),
        "elapsed_seconds": round(elapsed, 2),
        "diff_size_chars": diff_chars,
        "final_cleaned": final_message,
        "raw_outputs": raw_outputs,
        "config": {
            "reasoning_effort": "low",
            "temperature": 0.2,
            "ctx_window": config.llm.ctx_window,
            "commit_message_max_tokens": config.token_limits.commit_message_max_tokens,
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.system(),
            "release": platform.release(),
        },
        "git": git_info,
    }

    agent_dir = Path(config.tools.agent_dir)
    if not agent_dir.is_absolute():
        agent_dir = repo_path / agent_dir
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    report_dir = agent_dir / "problem" / "commit" / ts
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "meta.json"
    try:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        # Also save raw outputs as separate files for easy inspection
        for i, ro in enumerate(raw_outputs):
            raw_file = report_dir / f"raw_output_{i}.txt"
            raw_file.write_text(ro.get("content", ""), encoding="utf-8")
            if ro.get("reasoning"):
                reason_file = report_dir / f"raw_reasoning_{i}.txt"
                reason_file.write_text(ro["reasoning"], encoding="utf-8")
        return report_dir
    except OSError:
        return None


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

    # Resolve chunk size
    chunk_size_arg = getattr(args, "chunk_size", None)
    if chunk_size_arg:
        if chunk_size_arg.endswith("%"):
            try:
                percentage = float(chunk_size_arg[:-1]) / 100.0
                chunk_chars = int(config.llm.ctx_window * percentage)
            except (ValueError, TypeError):
                console.print(f"[red]Invalid chunk size percentage: {chunk_size_arg}[/red]")
                return
        else:
            try:
                chunk_chars = int(chunk_size_arg)
            except ValueError:
                console.print(f"[red]Invalid chunk size: {chunk_size_arg}. Must be integer or percentage (e.g. '50%').[/red]")
                return
    else:
        chunk_chars = config.token_limits.commit_chunk_chars
        if chunk_chars <= 0:
            # Auto-derive: leave room for running summary (output of prev step),
            # output budget for this step, and system/prompt overhead (~500 tok).
            # chars_per_token ≈ 4 for code/diffs.
            summary_tok = config.token_limits.commit_summary_tokens
            overhead_tok = summary_tok + summary_tok + 500
            chunk_chars = max(4000, (config.llm.ctx_window - overhead_tok) * 4)

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
    from agent.config import make_registry

    state = {"tokens": 0, "buf": "", "start": _time.monotonic(), "phase": "starting", "raw_outputs": []}

    # Resolve summarizer model entry (for chunked diff summarization only).
    registry = make_registry(config)
    summ_entry = None
    summ_override = getattr(args, "summarizer_model", None)
    if summ_override:
        summ_entry = registry.get(summ_override)
        if summ_entry is None:
            console.print(f"[red]Unknown model entry '{summ_override}'. "
                          f"Available: {registry.names()}[/red]")
            return
    elif config.model_roles.get("summarizer"):
        summ_entry = registry.summarizer

    primary_client = AsyncOpenAI(base_url=config.llm.base_url, api_key=config.llm.api_key)
    primary_model = config.llm.model

    if chunked:
        summ_label = f" · summarizer: {summ_entry.model}" if summ_entry else f" · summarizer: {primary_model}"
        console.print(
            f"[dim]Generating commit message for {path} "
            f"(staged diff: {diff_chars:,} chars → {len(chunks)} chunks of ≤{chunk_chars:,}{summ_label})…[/dim]"
        )
    else:
        _display_model = summ_entry.model if summ_override and summ_entry else primary_model
        console.print(
            f"[dim]Generating commit message for {path} "
            f"(staged diff: {diff_chars:,} chars · model: {_display_model})…[/dim]"
        )

    if summ_entry:
        summ_client = AsyncOpenAI(base_url=summ_entry.base_url, api_key=summ_entry.api_key)
        summ_model = summ_entry.model
    else:
        summ_client = primary_client
        summ_model = primary_model

    async def _stream(messages: list[dict], *, max_tokens: int, client=None, model: str = "") -> str:
        from agent.core.streaming import _clean_output
        _client = client or primary_client
        _model = model or primary_model
        stream = await _client.chat.completions.create(
            model=_model,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
            stream=True,
            extra_body={"reasoning_effort": "low"},
        )
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
                state["tokens"] += 1
                state["buf"] += delta.content
            if getattr(delta, "reasoning_content", None):
                reasoning_parts.append(delta.reasoning_content)
        raw_content = "".join(content_parts)
        raw_reasoning = "".join(reasoning_parts)
        state["raw_outputs"].append({"content": raw_content, "reasoning": raw_reasoning})
        full = _clean_output(raw_content)
        if not full:
            full = _clean_output(raw_reasoning)
        return full

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
        out = (await _stream(messages, max_tokens=summary_tokens,
                             client=summ_client, model=summ_model)).strip()

        first_line = out.splitlines()[0].strip() if out else ""
        if first_line == _REQUEST_PREV_RAW and prev_raw:
            messages[-1]["content"] = _build_user(True, False)
            out = (await _stream(messages, max_tokens=summary_tokens,
                                 client=summ_client, model=summ_model)).strip()
        elif first_line == _REQUEST_PREV_SUMMARY and running_summary:
            messages[-1]["content"] = _build_user(False, True)
            out = (await _stream(messages, max_tokens=summary_tokens,
                                 client=summ_client, model=summ_model)).strip()
        return out

    final_system = (
        "You write git commit messages. Output ONLY the commit message: "
        "no preamble, no explanation, no markdown fences, no quotes. "
        "First line: imperative-mood summary, <=72 chars. "
        "Optional body after a blank line, wrapped at 72 chars."
        f"\n\nIMPORTANT: You have a total budget of {config.token_limits.commit_message_max_tokens} tokens "
        f"(target output: ~{config.token_limits.commit_message} tokens). "
        f"Reserve at least {config.token_limits.commit_message_reserved} tokens for the final commit message content."
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
            max_tokens=config.token_limits.commit_message_max_tokens,
            client=summ_client if summ_override else None,
            model=summ_model if summ_override else "",
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
    choice = Prompt.ask("Commit with this message?", choices=["y", "n", "e", "rpt"], default="y")

    if choice == "rpt":
        desc = Prompt.ask("[yellow]Describe the issue[/yellow]",
                          default="leaked thinking/comments in output")
        report_dir = _save_problem_report(state, message, chunked, len(chunks),
                                          diff_chars, config, primary_model,
                                          summ_model, elapsed, path, desc)
        if report_dir:
            console.print(f"[dim]Problem report saved: {report_dir}[/dim]")
        else:
            console.print("[red]Failed to save problem report.[/red]")
        console.print("[dim]Aborted.[/dim]")
        return

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

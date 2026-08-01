from __future__ import annotations

import asyncio
import os
import re
import subprocess
from pathlib import Path


def _git_env() -> dict:
    """Environment that prevents git from blocking on an interactive prompt."""
    return {**os.environ, "GIT_TERMINAL_PROMPT": "0"}


def _fmt_tps(v: float) -> str:
    if v >= 10:
        return f"{v:.0f}"
    s = f"{v:.1f}"
    return s.lstrip("0") or "0"


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
            timeout=10, env=_git_env(),
        ).stdout.strip()
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_path), capture_output=True, text=True,
            timeout=10, env=_git_env(),
        ).stdout.strip()
        has_changes = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=str(repo_path),
            timeout=10, env=_git_env(),
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
        "model_failures": state.get("model_failures", []),
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


class CommitModelError(RuntimeError):
    """Every candidate model failed; ``failures`` holds per-model diagnostics."""

    def __init__(self, failures: list[dict]):
        super().__init__("no usable model for commit-message generation")
        self.failures = failures


def _ask(console, question: str, *, choices: list[str], default: str,
         on_eof: str | None = None) -> str | None:
    """Prompt.ask that survives having no terminal to ask on.

    Every other interactive prompt in cli/ guards EOFError; these did not, so
    running `agent commit` from a script or with piped output crashed with a
    traceback and wrote a crash dump instead of saying what was wrong.

    Returns *on_eof* when stdin is exhausted (None means "the caller decides"),
    and treats Ctrl-C as a decline rather than letting it unwind.
    """
    from rich.prompt import Prompt      # imported lazily, as elsewhere in this module
    try:
        return Prompt.ask(question, choices=choices, default=default)
    except (EOFError, KeyboardInterrupt):
        return on_eof


def _resolve_confirmation(args, console) -> str:
    """What to do with the proposed message: "y", "n", "e", "rpt", or "print".

    Extracted from cmd_commit so the non-interactive contract is testable — it
    is the branch where the command used to crash with an EOFError traceback.
    Exits 1 when there is nobody to ask: a script that expected a commit has to
    be able to tell, and committing a message no one approved is the one
    outcome worse than not committing.
    """
    if getattr(args, "print_only", False):
        return "print"
    if getattr(args, "yes", False):
        return "y"
    # on_eof stays None so the no-terminal case stays distinguishable from a
    # user typing "n" — they want different output and different exit codes.
    choice = _ask(console, "Commit with this message?",
                  choices=["y", "n", "e", "rpt"], default="y")
    if choice is None:
        console.print(
            "[yellow]Not committed: no terminal to confirm on.[/yellow]\n"
            "[dim]Re-run with '-y' to commit without asking, or '--print' to "
            "just emit the message.[/dim]"
        )
        raise SystemExit(1)
    return choice


def _err_brief(exc: Exception) -> str:
    """One-line ``Type: first line of message`` summary, capped at 200 chars."""
    s = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {s[:200]}" if s else type(exc).__name__


def _pick_fast_entry(registry, gpu_pool: list[str]):
    """Return (entry_name, entry) for fastest GPU model by measured tps, else first in pool."""
    from agent.metrics.model_stats import get_tps
    best_name, best_entry, best_tps = None, None, -1.0
    for name in gpu_pool:
        entry = registry.get(name)
        if entry is None:
            continue
        measured = get_tps(name)
        declared = entry.tokens_per_sec if entry.tokens_per_sec > 0 else 0.0
        tps = measured if measured > 0 else declared
        if best_name is None or tps > best_tps:
            best_name, best_entry, best_tps = name, entry, tps
    return best_name, best_entry


def _probe_endpoints(registry, timeout: int = 2) -> dict[str, "set[str] | None"]:
    """GET /models once per unique base_url, in parallel.

    Returns base_url → set of live model ids, or None when unreachable. Pure
    metadata: no completion is requested, so probing costs no tokens.
    """
    from concurrent.futures import ThreadPoolExecutor
    from agent.config.model_probe import list_endpoint_models

    targets: dict[str, str] = {}  # base_url → api_key
    for name in registry.names():
        entry = registry.get(name)
        if entry is not None and entry.base_url:
            targets.setdefault(entry.base_url, getattr(entry, "api_key", "") or "")
    if not targets:
        return {}
    with ThreadPoolExecutor(max_workers=min(16, len(targets))) as pool:
        results = pool.map(
            lambda kv: (kv[0], list_endpoint_models(kv[0], kv[1], timeout=timeout)),
            list(targets.items()),
        )
        return dict(results)


def _print_model_list(console, registry, gpu_pool: list[str], probe: bool = True) -> None:
    from rich.table import Table
    from agent.metrics.model_stats import load_stats
    stats = load_stats()
    live = _probe_endpoints(registry) if probe else {}
    table = Table(title="Available model entries", show_lines=False, box=None,
                  pad_edge=False, collapse_padding=True)
    table.add_column("name", style="cyan", no_wrap=True)
    table.add_column("model", no_wrap=True)
    if probe:
        table.add_column("live", justify="center", no_wrap=True)
        table.add_column("endpoint serves", style="dim")
    table.add_column("tags", style="dim")
    table.add_column("tps", justify="right", no_wrap=True)
    table.add_column("cfg", justify="right", no_wrap=True)
    table.add_column("gpu", justify="center", no_wrap=True)
    for name in registry.names():
        entry = registry.get(name)
        rec = stats.get(name, {})
        measured = f"{_fmt_tps(rec['tps_ewma'])} ({rec['samples']}s)" if rec else "-"
        declared = f"{_fmt_tps(entry.tokens_per_sec)}" if entry.tokens_per_sec > 0 else "-"
        gpu_mark = "✓" if name in gpu_pool else ""
        tags = ", ".join(entry.tags) if entry.tags else ""
        cells = [name, entry.model or name]
        if probe:
            cells.extend(_live_cells(entry, live))
        cells.extend([tags, measured, declared, gpu_mark])
        table.add_row(*cells)
    console.print(table)
    if probe:
        console.print(
            "[dim]live: ✓ served now · ~ endpoint up but serving something else "
            "('router:' can autoload, 'loaded:' is a one-model server needing a "
            "manual swap) · ✗ endpoint unreachable. "
            "Probe is a /models GET — no tokens spent.[/dim]"
        )


def _live_cells(entry, live: dict) -> tuple[str, str]:
    """Return (live marker, short description of what the endpoint serves)."""
    if not entry.base_url:
        return "", ""
    ids = live.get(entry.base_url)
    if ids is None:
        return "[red]✗[/red]", "[red]unreachable[/red]"
    from agent.config.model_probe import model_in_server
    if model_in_server(entry.model or "", ids):
        return "[green]✓[/green]", ""
    # Endpoint answers but does not advertise this entry's model. A one-model
    # llama-server lists only the loaded gguf (so this entry needs a manual
    # swap); a router lists every preset it can autoload.
    shown = ", ".join(sorted(_short_id(i) for i in ids)[:3]) or "(none)"
    if len(ids) > 3:
        shown += f", +{len(ids) - 3}"
    kind = "loaded" if len(ids) == 1 else "router"
    return "[yellow]~[/yellow]", f"{kind}: {shown}"


def _short_id(model_id: str) -> str:
    """Basename of a gguf-path model id; ids that are plain names pass through."""
    return model_id.rsplit("/", 1)[-1]


def cmd_commit(args, config):
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from agent.config import make_registry

    console = Console()
    registry = make_registry(config)
    gpu_pool: list[str] = config.concurrency.gpu_pool

    # -m or -s with no value → list models and exit
    summ_override = getattr(args, "summarizer_model", None)
    if summ_override == "__list__" or getattr(args, "model", None) == "__list__":
        _print_model_list(console, registry, gpu_pool,
                          probe=getattr(args, "probe", True))
        return

    path = Path(args.path)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    if not path.is_dir():
        console.print(f"[red]Not a directory: {path}[/red]")
        return

    result = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=str(path),
        capture_output=True, text=True,
        timeout=10, env=_git_env(),
    )
    if result.returncode != 0:
        console.print(f"[red]Not a git repository: {path}[/red]")
        return

    def _git(*git_args: str) -> str:
        try:
            r = subprocess.run(
                ["git"] + list(git_args), cwd=str(path), capture_output=True,
                text=True, timeout=30, env=_git_env(),
            )
        except subprocess.TimeoutExpired:
            return ""
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
                chunk_chars = int(config.llm.ctx_window * percentage) if config.llm.ctx_window > 0 else 0
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
        chunk_chars = 0

    if chunk_chars <= 0:
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

    state = {"tokens": 0, "buf": "", "start": _time.monotonic(), "phase": "starting",
             "raw_outputs": [], "fallback": False, "cand_idx": 0, "model_failures": []}

    # Resolve summarizer entry:
    # 1. explicit -m NAME flag
    # 2. [models] summarizer role in config
    # 3. auto-pick fastest GPU model from gpu_pool
    # 4. fall back to primary model
    summ_entry = None
    summ_entry_name: str = ""
    if summ_override:
        summ_entry = registry.get(summ_override)
        if summ_entry is None:
            console.print(f"[red]Unknown model entry '{summ_override}'. "
                          f"Run 'agent commit -m' to list available entries.[/red]")
            return
        summ_entry_name = summ_override
    elif config.model_roles.get("summarizer"):
        summ_entry = registry.summarizer
        summ_entry_name = config.model_roles["summarizer"]
    elif gpu_pool:
        name, entry = _pick_fast_entry(registry, gpu_pool)
        if entry is not None:
            summ_entry = entry
            summ_entry_name = name

    # Primary (final commit-message) model: an explicit `commit` role pin wins,
    # else the active default endpoint (config.llm).
    commit_entry = registry.for_role("commit")
    if commit_entry is not None and commit_entry.base_url:
        primary_base_url, primary_api_key = commit_entry.base_url, commit_entry.api_key
        primary_model = commit_entry.model or config.llm.model
    else:
        primary_base_url, primary_api_key = config.llm.base_url, config.llm.api_key
        primary_model = config.llm.model
    from agent.core.llm_client import make_llm_client
    primary_client = make_llm_client(config, base_url=primary_base_url, api_key=primary_api_key)

    if chunked:
        summ_label = f" · summarizer: {summ_entry.model}" if summ_entry else f" · summarizer: {primary_model}"
        console.print(
            f"[dim]Generating commit message for {path} "
            f"(staged diff: {diff_chars:,} chars → {len(chunks)} chunks of ≤{chunk_chars:,}{summ_label})…[/dim]"
        )
    else:
        _display_model = summ_entry.model if summ_entry else primary_model
        console.print(
            f"[dim]Generating commit message for {path} "
            f"(staged diff: {diff_chars:,} chars · model: {_display_model})…[/dim]"
        )

    if summ_entry:
        summ_client = make_llm_client(config, base_url=summ_entry.base_url, api_key=summ_entry.api_key)
        summ_model = summ_entry.model
    else:
        summ_client = primary_client
        summ_model = primary_model

    # Fallback chain: preferred summarizer first, then the primary endpoint,
    # then remaining GPU-pool entries, then any other local-tier entry. Tried
    # in order whenever a model errors out (connection refused, 5xx "cannot
    # load model", 404 unknown model, timeout); a failed candidate is skipped
    # for the rest of the run.
    candidates: list[dict] = []
    _seen_cand: set = set()

    def _add_candidate(name: str, base_url: str, api_key: str, model: str) -> None:
        key = (base_url, model)
        if not model or key in _seen_cand:
            return
        _seen_cand.add(key)
        candidates.append({"name": name or model, "base_url": base_url,
                           "api_key": api_key, "model": model, "client": None})

    if summ_entry:
        _add_candidate(summ_entry_name, summ_entry.base_url, summ_entry.api_key, summ_model)
    _add_candidate("primary", primary_base_url, primary_api_key, primary_model)
    for _name in gpu_pool:
        _e = registry.get(_name)
        if _e is not None:
            _add_candidate(_name, _e.base_url, _e.api_key, _e.model)
    from agent.config import entry_tier as _entry_tier
    for _name in registry.names():
        _e = registry.get(_name)
        if _e is not None and _entry_tier(_e) == "local":
            _add_candidate(_name, _e.base_url, _e.api_key, _e.model)

    for _c in candidates:
        if _c["base_url"] == primary_base_url and _c["model"] == primary_model:
            _c["client"] = primary_client
        elif summ_entry and _c["base_url"] == summ_entry.base_url and _c["model"] == summ_model:
            _c["client"] = summ_client

    def _cand_client(c: dict):
        if c["client"] is None:
            c["client"] = make_llm_client(config, base_url=c["base_url"], api_key=c["api_key"])
        return c["client"]

    async def _do_stream(client, model: str, messages: list[dict], max_tokens: int, entry_name: str) -> tuple[str, str, int, float]:
        """Low-level stream call. Returns (raw_content, raw_reasoning, completion_tokens, elapsed)."""
        from agent.metrics.model_stats import update_stats
        try:
            from agent.metrics import model_calls
            model_calls.record_entry_name(config, entry_name, role="commit")
        except Exception:
            pass
        t0 = _time.monotonic()
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"reasoning_effort": "low"},
        )
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage_completion_tokens: int = 0
        async for chunk in stream:
            if not chunk.choices:
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_completion_tokens = chunk.usage.completion_tokens or 0
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
                state["tokens"] += 1
                state["buf"] += delta.content
            if getattr(delta, "reasoning_content", None):
                reasoning_parts.append(delta.reasoning_content)
        elapsed = _time.monotonic() - t0
        tok_count = usage_completion_tokens if usage_completion_tokens > 0 else len(content_parts)
        if entry_name:
            try:
                update_stats(entry_name, tok_count, elapsed)
            except Exception:
                pass
        return "".join(content_parts), "".join(reasoning_parts), usage_completion_tokens, elapsed

    async def _stream(messages: list[dict], *, max_tokens: int) -> str:
        """Stream one completion, walking the fallback chain on any API error.

        ``state["cand_idx"]`` persists across calls so a candidate that failed
        once (endpoint down, model cannot be loaded, …) is not retried on later
        chunks. Raises CommitModelError when every candidate has failed.
        """
        from agent.core.streaming import _clean_output
        import openai as _openai
        last_exc: Exception | None = None
        while state["cand_idx"] < len(candidates):
            cand = candidates[state["cand_idx"]]
            try:
                raw_content, raw_reasoning, _, _ = await _do_stream(
                    _cand_client(cand), cand["model"], messages, max_tokens, cand["name"])
            except (_openai.OpenAIError, OSError) as exc:
                brief = _err_brief(exc)
                state["model_failures"].append(
                    {"entry": cand["name"], "model": cand["model"],
                     "base_url": cand["base_url"], "error": brief})
                state["fallback"] = True
                state["cand_idx"] += 1
                nxt = (candidates[state["cand_idx"]]
                       if state["cand_idx"] < len(candidates) else None)
                if nxt is not None:
                    console.print(
                        f"[yellow]Model '{cand['name']}' unavailable at {cand['base_url']} "
                        f"({_markup_escape(brief)}) — falling back to "
                        f"'{nxt['name']}' ({nxt['model']}).[/yellow]"
                    )
                last_exc = exc
                continue
            state["raw_outputs"].append({"content": raw_content, "reasoning": raw_reasoning})
            full = _clean_output(raw_content)
            if not full:
                full = _clean_output(raw_reasoning)
            return full
        raise CommitModelError(list(state["model_failures"])) from last_exc

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

    try:
        message = asyncio.run(_run_with_status()).strip()
    except CommitModelError as exc:
        elapsed = _time.monotonic() - state["start"]
        console.print("[red]Commit-message generation failed — no usable model.[/red]")
        for f in exc.failures:
            console.print(
                f"  [red]✗[/red] {f['entry']} ({f['model']}) @ {f['base_url']} — "
                f"{_markup_escape(f['error'])}"
            )
        console.print(
            "[dim]Check the endpoints above, or pick a working entry with "
            "'agent commit -m NAME' ('-m' alone lists entries).[/dim]"
        )
        report_dir = _save_problem_report(
            state, "", chunked, len(chunks), diff_chars, config, primary_model,
            summ_model, elapsed, path, "auto: all candidate models failed",
        )
        if report_dir:
            console.print(f"[dim]Error dump: {report_dir}[/dim]")
        return
    elapsed = _time.monotonic() - state["start"]
    _fb = ""
    if state["fallback"] and state["cand_idx"] < len(candidates):
        _fb = (f" · fell back to '{candidates[state['cand_idx']]['name']}' "
               f"({candidates[state['cand_idx']]['model']})")
    console.print(f"[dim]done in {elapsed:.1f}s · {state['tokens']} tokens{_fb}[/dim]")

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

    choice = _resolve_confirmation(args, console)
    if choice == "print":
        return

    if choice == "rpt":
        _default_desc = "leaked thinking/comments in output"
        try:
            desc = Prompt.ask("[yellow]Describe the issue[/yellow]", default=_default_desc)
        except (EOFError, KeyboardInterrupt):
            desc = _default_desc
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
        # EOF here means the editor closed stdin; decline rather than commit
        # an edited message that was never confirmed.
        confirm = _ask(console, "Commit?", choices=["y", "n"], default="y", on_eof="n")
        if confirm == "n":
            console.print("[dim]Aborted.[/dim]")
            return

    result = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(path),
        capture_output=True, text=True,
        timeout=30, env=_git_env(),
    )
    if result.returncode == 0:
        console.print(f"[green]Committed.[/green]\n{result.stdout.strip()}")
    else:
        console.print(f"[red]Commit failed:[/red]\n{result.stderr.strip()}")

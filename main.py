#!/usr/bin/env python3
"""local-code-agent — entry point."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path


def _write_exception_dump(
    exc: BaseException,
    argv: list[str] | None = None,
    config=None,
    log_path: Path | None = None,
) -> Path | None:
    """Write a human-readable exception dump to .agent/exception-<timestamp>.dump."""
    import platform

    try:
        # Determine dump directory
        if config is not None:
            dump_dir = Path(config.tools.working_dir) / config.tools.agent_dir
        else:
            dump_dir = Path(".agent")
        dump_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        dump_path = dump_dir / f"exception-{ts}.dump"

        lines: list[str] = []
        lines.append("=== Exception Dump ===")
        lines.append(f"Timestamp : {datetime.now().isoformat(timespec='seconds')}")
        lines.append(f"Python    : {sys.version}")
        lines.append(f"Platform  : {platform.platform()}")
        lines.append(f"Command   : {' '.join(argv or sys.argv)}")
        lines.append("")

        lines.append("=== Traceback ===")
        lines.append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip())
        lines.append("")

        if config is not None:
            lines.append("=== Config ===")
            try:
                lines.append(f"model      : {config.llm.model}")
                lines.append(f"base_url   : {config.llm.base_url}")
                lines.append(f"working_dir: {config.tools.working_dir}")
                lines.append(f"agent_dir  : {config.tools.agent_dir}")
                lines.append(f"ctx_window : {config.llm.ctx_window}")
            except Exception as ce:
                lines.append(f"(error reading config: {ce})")
            lines.append("")

        if log_path is not None and log_path.exists():
            lines.append("=== Recent Log (last 60 lines) ===")
            try:
                log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                lines.extend(log_lines[-60:])
            except Exception as le:
                lines.append(f"(error reading log: {le})")
            lines.append("")

        dump_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return dump_path
    except Exception:
        return None


def _setup_logging(agent_dir: str | None = None, logs_cfg=None) -> None:
    """Configure logging per [logs] config: rotating file + stderr + per-source levels."""
    from logging.handlers import RotatingFileHandler

    log_dir = Path(agent_dir) if agent_dir else Path(".agent")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "agent.log"

    level_name = (getattr(logs_cfg, "level", None) or "DEBUG").upper()
    stderr_level = (getattr(logs_cfg, "stderr_level", None) or "WARNING").upper()
    max_bytes = getattr(logs_cfg, "max_bytes", 20 * 1024 * 1024)
    backup_count = getattr(logs_cfg, "backup_count", 5)
    sources = getattr(logs_cfg, "sources", {}) or {}

    root = logging.getLogger()
    root.setLevel(getattr(logging, level_name, logging.DEBUG))

    fh = RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    fh.setLevel(getattr(logging, level_name, logging.DEBUG))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(getattr(logging, stderr_level, logging.WARNING))
    sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(sh)

    for source_name, source_level in sources.items():
        lvl = getattr(logging, str(source_level).upper(), None)
        if lvl is not None:
            logging.getLogger(source_name).setLevel(lvl)


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

    if getattr(args, "watch", False):
        _watch_and_reindex(config, console, languages=languages, exclude=exclude)


def _watch_and_reindex(config, console, languages=None, exclude=None):
    """Watch working_dir for changes and re-index modified files using watchdog."""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        console.print("[red]watchdog not installed. Run: pip install watchdog[/red]")
        return

    from agent.rag.indexer import index_directory
    from agent.rag.store import VectorStore
    from agent.rag.embedder import Embedder
    import threading
    import time

    working_dir = config.tools.working_dir
    debounce: dict[str, float] = {}
    lock = threading.Lock()

    class _Handler(FileSystemEventHandler):
        def on_modified(self, event):
            if event.is_directory:
                return
            with lock:
                debounce[event.src_path] = time.monotonic()

        on_created = on_modified

    observer = Observer()
    observer.schedule(_Handler(), working_dir, recursive=True)
    observer.start()
    console.print(f"[dim]Watching {working_dir} for changes. Ctrl+C to stop.[/dim]")

    try:
        while True:
            time.sleep(1)
            now = time.monotonic()
            with lock:
                ready = [p for p, t in list(debounce.items()) if now - t > 1.5]
                for p in ready:
                    del debounce[p]
            if ready:
                console.print(f"[dim]Re-indexing {len(ready)} changed file(s)…[/dim]")
                store = VectorStore(config.rag)
                embedder = Embedder(config.embeddings)
                index_directory(
                    root=working_dir,
                    store=store,
                    embedder=embedder,
                    cfg=config.rag,
                    languages=languages,
                    exclude=exclude or [],
                )
                store.close()
                console.print(f"[green]Re-indexed.[/green]")
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def _open_archive(config):
    from agent.rag.archive import ArchiveStore
    return ArchiveStore(config.rag.archive_db_path)


def cmd_index_update(args, config):
    from agent.rag.indexer import index_directory, prune_index
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
    archive = _open_archive(config)

    if changed is not None and not changed:
        console.print("No changed files detected. Index is up to date.")
    else:
        stats = index_directory(
            root=config.tools.working_dir,
            store=store,
            embedder=embedder,
            cfg=config.rag,
        )
        console.print(f"Updated: {stats['indexed']} files re-indexed, {stats['skipped']} unchanged.")

    pruned = prune_index(config.tools.working_dir, store, archive, reason="stale")
    if pruned["archived"]:
        console.print(f"Archived {pruned['archived']} chunks from {len(pruned['paths'])} file(s):")
        for p in pruned["paths"]:
            console.print(f"  [dim]- {p}[/dim]")

    purged = archive.purge_expired(config.rag.archive_ttl_days)
    if purged:
        console.print(f"Purged {purged} expired archived chunks (ttl={config.rag.archive_ttl_days}d).")

    store.close()
    archive.close()


def cmd_index_prune(args, config):
    from agent.rag.indexer import prune_index
    from agent.rag.store import VectorStore
    from rich.console import Console

    console = Console()
    store = VectorStore(config.rag)
    archive = _open_archive(config)

    pruned = prune_index(config.tools.working_dir, store, archive, reason="prune")
    if pruned["archived"]:
        console.print(f"Archived {pruned['archived']} chunks from {len(pruned['paths'])} file(s):")
        for p in pruned["paths"]:
            console.print(f"  [dim]- {p}[/dim]")
    else:
        console.print("Index is clean — nothing to archive.")

    ttl = getattr(args, "archive_ttl", None)
    if ttl is None:
        ttl = config.rag.archive_ttl_days
    purged = archive.purge_expired(ttl)
    if purged:
        console.print(f"Purged {purged} expired archived chunks (ttl={ttl}d).")

    store.close()
    archive.close()


def cmd_index_restore(args, config):
    from agent.rag.indexer import restore_paths
    from agent.rag.store import VectorStore
    from rich.console import Console

    console = Console()
    store = VectorStore(config.rag)
    archive = _open_archive(config)
    res = restore_paths(store, archive, [args.restore])
    if res["restored"]:
        console.print(f"Restored {res['restored']} chunks for:")
        for p in res["paths"]:
            console.print(f"  [green]+ {p}[/green]")
    else:
        console.print(f"No archived chunks found for path: {args.restore}")
    store.close()
    archive.close()


def cmd_index_purge_archive(args, config):
    from rich.console import Console
    console = Console()
    archive = _open_archive(config)
    ttl = getattr(args, "archive_ttl", None)
    if ttl is None:
        ttl = config.rag.archive_ttl_days
    purged = archive.purge_expired(ttl)
    console.print(f"Purged {purged} expired archived chunks (ttl={ttl}d).")
    archive.close()


def cmd_index_stats(args, config):
    from agent.rag.store import VectorStore
    from rich.console import Console
    import datetime as _dt

    store = VectorStore(config.rag)
    stats = store.stats()
    store.close()

    archive = _open_archive(config)
    astats = archive.stats()
    archive.close()

    console = Console()
    console.print(f"[bold]Index stats:[/bold]")
    console.print(f"  Files:  {stats['files']}")
    console.print(f"  Chunks: {stats['chunks']}")
    console.print(f"  DB:     {config.rag.db_path}")
    console.print(f"[bold]Archive stats:[/bold]")
    console.print(f"  Files:  {astats['files']}")
    console.print(f"  Chunks: {astats['chunks']}")
    console.print(f"  DB:     {config.rag.archive_db_path}")
    console.print(f"  TTL:    {config.rag.archive_ttl_days} days" + (" (disabled)" if config.rag.archive_ttl_days <= 0 else ""))
    oldest = astats.get("oldest_archived_at")
    if oldest:
        ts = _dt.datetime.fromtimestamp(oldest).isoformat(timespec="seconds")
        console.print(f"  Oldest: {ts}")


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
    """Return True if no agent.toml is found in standard locations."""
    paths = [
        Path.home() / ".config" / "agent" / "agent.toml",
        Path("agent.toml"),
    ]
    return not any(p.exists() for p in paths)


def _find_project_root(start_dir: Path, search_parents: bool) -> Path | None:
    """Search for a directory containing a .agent subdirectory."""
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

    # Explain first-run / uninitialized directory
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
            # Treat the arg as a short_name for a new session.
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


def cmd_run(args, config):
    import sys
    from agent.rag.store import VectorStore
    from agent.rag.embedder import Embedder
    from agent.agent import Agent
    from rich.console import Console

    console = Console()

    # Support piped input: `echo "fix this" | agent run`
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

    agent = Agent(config, store=store, embedder=embedder)

    def on_tool(name: str, args_str: str) -> None:
        console.print(f"  [dim]→ {name}[/dim]")

    async def _run():
        response = await agent.chat(args.prompt_text, on_tool_call=on_tool)
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
        session, messages = load_session(args.load)
        if session is None:
            console.print(f"Session '{args.load}' not found.")
            return
        label = session.name or session.short_name or session.id
        console.print(f"Session '{label}' ({session.id}): {len(messages)} messages")
        return

    sessions = list_sessions()
    if not sessions:
        console.print("No sessions found.")
        return

    table = Table(title="Sessions")
    table.add_column("ID")
    table.add_column("Short name")
    table.add_column("Name")
    table.add_column("Tags")
    table.add_column("Msgs", justify="right")
    table.add_column("Updated")

    for s in sessions:
        ts = s.get("updated_at") or s.get("created_at")
        updated = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "?"
        table.add_row(
            s["id"],
            s.get("short_name", ""),
            s.get("name", ""),
            ", ".join(s.get("tags", [])),
            str(s["message_count"]),
            updated,
        )

    console.print(table)


def cmd_commit(args, config):
    import subprocess
    import asyncio
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt

    console = Console()

    # Resolve path
    path = Path(args.path)
    if not path.is_absolute():
        path = Path(config.tools.working_dir) / path
    path = path.resolve()

    if not path.is_dir():
        console.print(f"[red]Not a directory: {path}[/red]")
        return

    # Verify it's a git repo
    result = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=str(path),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]Not a git repository: {path}[/red]")
        return

    def _git(*git_args: str) -> str:
        r = subprocess.run(
            ["git"] + list(git_args),
            cwd=str(path),
            capture_output=True, text=True,
        )
        return r.stdout.strip()

    staged_diff = _git("diff", "--cached")
    if not staged_diff:
        console.print("[yellow]No staged changes found. Stage files first with git add.[/yellow]")
        return

    status = _git("status", "--short")
    recent_log = _git("log", "--oneline", "-10")

    # Cap diff size — models hallucinate badly on very large diffs
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
    import time as _time
    from openai import AsyncOpenAI

    console.print(
        f"[dim]Generating commit message for {path} "
        f"(staged diff: {diff_chars:,} chars"
        f"{', truncated to ' + str(DIFF_CAP) if truncated else ''})…[/dim]"
    )

    state = {"tokens": 0, "buf": "", "start": _time.monotonic()}

    async def _generate() -> str:
        client = AsyncOpenAI(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
        )
        # Reasoning-capable models (e.g. Gemma-4) emit a thinking trace before
        # the final answer. Budget enough tokens for thinking + answer and
        # hint that minimal reasoning is fine. Capture `reasoning_content`
        # only for progress display, never as the commit message.
        stream = await client.chat.completions.create(
            model=config.llm.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=4096,
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
                # Show thinking in the spinner preview so the user sees life
                state["tokens"] += 1
                state["buf"] = ("…thinking: " + reasoning)[-120:]
        return "".join(parts)

    async def _run_with_status() -> str:
        task = asyncio.create_task(_generate())
        spinner = Spinner("dots", text="starting…")
        with Live(spinner, console=console, refresh_per_second=8, transient=True):
            while not task.done():
                elapsed = _time.monotonic() - state["start"]
                preview = state["buf"].replace("\n", " ")[-60:]
                spinner.update(text=Text.from_markup(
                    f"[cyan]generating[/cyan] · {elapsed:5.1f}s · "
                    f"{state['tokens']} tok · [dim]{preview}[/dim]"
                ))
                await asyncio.sleep(0.15)
        return await task

    message = asyncio.run(_run_with_status()).strip()
    elapsed = _time.monotonic() - state["start"]
    console.print(
        f"[dim]done in {elapsed:.1f}s · {state['tokens']} tokens[/dim]"
    )

    # Strip any accidental markdown fences
    if message.startswith("```"):
        lines = message.splitlines()
        lines = [l for l in lines if not l.startswith("```")]
        message = "\n".join(lines).strip()

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


def cmd_debug_context(args, config):
    from agent.memory.session import load_session
    from agent.memory.compactor import _count_tokens_approx
    from rich.console import Console
    from rich.table import Table
    import json

    console = Console()
    session_name = args.session or "default"
    session, messages = load_session(session_name)
    if not messages:
        console.print("No session loaded.")
        return

    label = (session.name or session.short_name or session.id) if session else session_name
    table = Table(title=f"Context: session '{label}'", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Role", width=10)
    table.add_column("Tokens", justify="right", width=7)
    table.add_column("Flags", width=14)
    table.add_column("Preview", ratio=1)

    total = 0
    for i, m in enumerate(messages):
        role = m.get("role", "?")

        content = m.get("content") or ""
        if isinstance(content, list):
            content_str = json.dumps(content)
        else:
            content_str = str(content)

        tool_calls = m.get("tool_calls", [])
        tool_names = [tc["function"]["name"] for tc in tool_calls if isinstance(tc, dict)]

        flags = []
        if m.get("_nudged"):
            flags.append("nudged")
        if "[SESSION SUMMARY]" in content_str:
            flags.append("compacted")
        if tool_names:
            flags.append("tools:" + ",".join(tool_names[:2]))

        toks = _count_tokens_approx([m])
        total += toks

        preview = content_str[:120].replace("\n", "↵")
        if len(content_str) > 120:
            preview += "…"

        table.add_row(str(i), role, str(toks), " ".join(flags), preview)

    console.print(table)
    console.print(f"[bold]Total:[/bold] {total} tokens across {len(messages)} messages")


def main():
    parser = argparse.ArgumentParser(prog="agent", description="Local code agent")
    parser.add_argument("--config", type=str, help="Path to agent.toml")

    sub = parser.add_subparsers(dest="command")

    # init
    init_p = sub.add_parser("init", help="Initialize index for current directory")
    init_p.add_argument("--languages", type=str, help="Comma-separated languages: py,js,kt,cpp")
    init_p.add_argument("--exclude", type=str, help="Comma-separated paths to exclude")
    init_p.add_argument("--force", action="store_true", help="Force re-index all files")
    init_p.add_argument("--watch", action="store_true", help="Watch for file changes and re-index automatically")

    # index
    idx_p = sub.add_parser("index", help="Manage index")
    idx_p.add_argument("--update", action="store_true", help="Re-index changed files (also prunes stale & purges expired archive)")
    idx_p.add_argument("--stats", action="store_true", help="Show index statistics")
    idx_p.add_argument("--prune", action="store_true", help="Archive chunks for files that are missing or now match .agent.ignore")
    idx_p.add_argument("--restore", type=str, metavar="PATH", help="Restore a previously archived path back into the live index")
    idx_p.add_argument("--purge-archive", action="store_true", help="Permanently delete archive rows older than archive_ttl_days")
    idx_p.add_argument("--archive-ttl", type=int, metavar="DAYS", help="Override archive_ttl_days for this run (0 = disable expiration)")

    # chat
    chat_p = sub.add_parser("chat", help="Start interactive session")
    chat_p.add_argument("--model", type=str, help="Override model name")
    chat_p.add_argument("--ctx", type=int, help="Override context window size")
    chat_p.add_argument("--session", type=str, help="Session name to load/save")
    chat_p.add_argument("--ui", type=str, choices=["textual", "simple"],
                        help="UI mode (skips the prompt)")

    # run
    run_p = sub.add_parser("run", help="Run a single prompt non-interactively")
    run_p.add_argument("prompt", type=str, nargs="?", default=None,
                       help="Prompt to run (reads stdin if omitted)")

    # sessions
    sess_p = sub.add_parser("sessions", help="Manage sessions")
    sess_p.add_argument("--list", action="store_true", help="List sessions")
    sess_p.add_argument("--load", type=str, help="Show session details")

    # commit
    commit_p = sub.add_parser("commit", help="Generate and apply a commit message for a subrepo")
    commit_p.add_argument("path", type=str, help="Path to subrepo (absolute or relative to working dir)")
    commit_p.add_argument("--model", type=str, help="Override model name")

    # exec
    exec_p = sub.add_parser("exec", help="Execute a system command in the project directory")
    exec_p.add_argument("prompt", type=str, help="Command to execute")

    # debug
    dbg_p = sub.add_parser("debug", help="Debug utilities")
    dbg_p.add_argument("--context", action="store_true", help="Show full context of current session")
    dbg_p.add_argument("--session", type=str, help="Session name")

    args = parser.parse_args()

    from agent.config import load_config, check_reachability, Config, ToolsConfig
    from agent.memory.session import configure as configure_sessions

    # 1. Find project root (only if not running 'init')
    project_root = None
    if args.command != "init":
        temp_tools = ToolsConfig()
        project_root = _find_project_root(Path.cwd(), temp_tools.search_parents)
        if project_root is None:
            print("Error: Current directory (and parents) is not a valid agent project.")
            print("Please run 'agent init' in the desired project directory.")
            sys.exit(1)

    # 2. Load config
    if args.config:
        config = load_config(Path(args.config))
    elif project_root and (project_root / "agent.toml").exists():
        config = load_config(project_root / "agent.toml")
    else:
        config = load_config(None)

    # 3. If we found a project root, ensure working_dir is set to it
    if project_root:
        config.tools.working_dir = str(project_root)

    # 4. Continue with rest of setup
    configure_sessions(config.tools.working_dir, config.tools.agent_dir)
    log_dir = Path(config.tools.working_dir) / config.tools.agent_dir
    _setup_logging(str(log_dir), config.logs)
    log_path = log_dir / "agent.log"

    try:
        if args.command == "init":
            cmd_init(args, config)
        elif args.command == "index":
            if args.update:
                cmd_index_update(args, config)
            elif args.stats:
                cmd_index_stats(args, config)
            elif args.prune:
                cmd_index_prune(args, config)
            elif args.restore:
                cmd_index_restore(args, config)
            elif getattr(args, "purge_archive", False):
                cmd_index_purge_archive(args, config)
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
        elif args.command == "commit":
            if getattr(args, "model", None):
                config.llm.model = args.model
            check_reachability(config)
            cmd_commit(args, config)
        elif args.command == "debug":
            cmd_debug_context(args, config)
        # exec command handler
        elif args.command == "exec":
            from agent.tools.exec_command import handle_exec_command
            handle_exec_command(args, config)
        else:
            parser.print_help()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        dump_path = _write_exception_dump(exc, argv=sys.argv, config=config, log_path=log_path)
        msg = f"\nUnhandled exception: {type(exc).__name__}: {exc}"
        if dump_path:
            msg += f"\nDump written to: {dump_path}"
        print(msg, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()




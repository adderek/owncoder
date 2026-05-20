from __future__ import annotations

from pathlib import Path


def _open_archive(config):
    from agent.rag.archive import ArchiveStore
    return ArchiveStore(config.rag.archive_db_path)


def _make_bg_worker(config, code_store, embedder):
    """Build a BgWorker from config. Returns None if summarization disabled."""
    if not config.summarization.enabled:
        return None
    from openai import OpenAI
    from agent.rag.describer import Describer
    from agent.rag.judge import Judge
    from agent.rag.bg_worker import BgWorker

    scfg = config.summarization
    model = scfg.describer_model or config.llm.model
    client = OpenAI(base_url=config.llm.base_url, api_key=config.llm.api_key)
    describer = Describer(client, model=model, ctx_tokens=scfg.ctx_tokens, max_output_tokens=scfg.max_output_tokens)
    judge = Judge(client, model=model, store=code_store)
    return BgWorker(store=code_store, describer=describer, judge=judge, embedder=embedder)


def cmd_init(args, config):
    from agent.rag.indexer import index_directory, pending_files, LANGUAGE_MAP
    from agent.rag.store import VectorStore
    from agent.rag.embedder import Embedder
    from agent.rag.code_store import CodeStore
    from agent.tools.rules import load_rules
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn

    console = Console()

    languages = args.languages.split(",") if args.languages else None
    exclude = args.exclude.split(",") if args.exclude else []
    working_dir = config.tools.working_dir
    _agent_dir = Path(working_dir) / config.tools.agent_dir
    _agent_dir.mkdir(parents=True, exist_ok=True)
    _in_progress = _agent_dir / ".init_in_progress"
    _initialized = _agent_dir / ".initialized"

    if _in_progress.exists() and not _initialized.exists():
        console.print("[yellow]Warning: previous 'agent init' was interrupted. Re-running.[/yellow]")
    _in_progress.touch()

    store = VectorStore(config.rag)
    embedder = Embedder(config.embeddings)
    code_store = CodeStore(config.summarization.db_path)

    load_rules(working_dir)

    console.print(f"[bold]Indexing[/bold] {Path(working_dir).resolve()}")
    if languages:
        console.print(f"Languages: {', '.join(languages)}")
    workers = config.embeddings.embed_workers
    if workers > 1:
        console.print(f"[dim]Embedding workers: {workers}[/dim]")

    # Pre-scan disk to get an accurate total for the progress bar.
    pre = pending_files(root=working_dir, store=store, languages=languages, exclude=exclude)
    total_to_index = pre["pending"] if not getattr(args, "force", False) else pre["total"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Indexing…", total=max(total_to_index, 1))

        def progress_cb(path: str, chunk_count: int) -> None:
            progress.advance(task)
            progress.update(task, description=f"[dim]{Path(path).name}[/dim]")

        stats = index_directory(
            root=working_dir,
            store=store,
            embedder=embedder,
            cfg=config.rag,
            languages=languages,
            exclude=exclude,
            force=getattr(args, "force", False),
            progress_cb=progress_cb,
            code_store=code_store,
        )

    store.close()
    console.print(
        f"[green]Done.[/green] Indexed {stats['indexed']} files, "
        f"skipped {stats['skipped']}, "
        f"created {stats['chunks']} chunks."
    )
    try:
        _in_progress.unlink(missing_ok=True)
        _initialized.touch()
    except Exception:
        pass

    if config.summarization.enabled and stats["indexed"] > 0:
        pending = code_store.stats().get("by_status", {}).get("pending", 0)
        if pending:
            depth = code_store.max_level()
            rounds_hint = f", ~{depth + 1} rollup round(s)" if depth > 0 else ""
            console.print(f"  {pending} chunks queued for summarization{rounds_hint} (running in background…)")
            worker = _make_bg_worker(config, code_store, embedder)
            if worker:
                worker.start()
                import signal, time as _time
                console.print("  [dim]Press Ctrl+C to stop early (queue persists).[/dim]")
                try:
                    while worker.is_alive():
                        remaining = code_store.stats().get("by_status", {})
                        r_pending = remaining.get("pending", 0)
                        r_stale = remaining.get("stale", 0)
                        if r_pending == 0 and r_stale == 0:
                            break
                        console.print(f"  [dim]remaining={r_pending + r_stale}[/dim]", end="\r")
                        _time.sleep(2)
                except KeyboardInterrupt:
                    pass
                worker.stop()
                final = code_store.stats().get("by_status", {})
                console.print(f"\n  Summarization: {final}")

    if getattr(args, "watch", False):
        _watch_and_reindex(config, console, languages=languages, exclude=exclude)


def _watch_and_reindex(config, console, languages=None, exclude=None):
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
                console.print("[green]Re-indexed.[/green]")
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def cmd_index_update(args, config):
    from agent.rag.indexer import index_directory, prune_index
    from agent.rag.store import VectorStore
    from agent.rag.embedder import Embedder
    from rich.console import Console
    import subprocess

    console = Console()

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
        stats = index_directory(root=config.tools.working_dir, store=store, embedder=embedder, cfg=config.rag)
        console.print(f"Updated: {stats['indexed']} files re-indexed, {stats['skipped']} unchanged.")

    # Resume any pending summarization left over from an interrupted run.
    if config.summarization.enabled:
        from agent.rag.code_store import CodeStore
        code_store = CodeStore(config.summarization.db_path)
        by_status = code_store.stats().get("by_status", {})
        pending = by_status.get("pending", 0) + by_status.get("stale", 0)
        if pending:
            import time as _time
            depth = code_store.max_level()
            rounds_hint = f", ~{depth + 1} rollup round(s)" if depth > 0 else ""
            console.print(f"  {pending} chunks pending summarization{rounds_hint} (resuming…)")
            worker = _make_bg_worker(config, code_store, embedder)
            if worker:
                worker.start()
                console.print("  [dim]Press Ctrl+C to stop early (queue persists).[/dim]")
                try:
                    while worker.is_alive():
                        remaining = code_store.stats().get("by_status", {})
                        r_pending = remaining.get("pending", 0)
                        r_stale = remaining.get("stale", 0)
                        if r_pending == 0 and r_stale == 0:
                            break
                        console.print(f"  [dim]remaining={r_pending + r_stale}[/dim]", end="\r")
                        _time.sleep(2)
                except KeyboardInterrupt:
                    pass
                worker.stop()
                final = code_store.stats().get("by_status", {})
                console.print(f"\n  Summarization: {final}")

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


def _daemon_pid_file(config) -> Path:
    return Path(config.tools.working_dir) / config.tools.agent_dir / "index-daemon.pid"


def _daemon_log_file(config) -> Path:
    return Path(config.tools.working_dir) / config.tools.agent_dir / "index-daemon.log"


def _read_daemon_pid(config) -> int | None:
    pid_file = _daemon_pid_file(config)
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except Exception:
        return None


def _daemon_running(pid: int) -> bool:
    import os
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — still running.
        return True


def cmd_index_daemon_start(args, config):
    import sys
    import subprocess
    from rich.console import Console

    console = Console()
    pid_file = _daemon_pid_file(config)
    log_file = _daemon_log_file(config)

    existing_pid = _read_daemon_pid(config)
    if existing_pid is not None:
        if _daemon_running(existing_pid):
            console.print(f"[yellow]Daemon already running (pid {existing_pid}).[/yellow]")
            console.print(f"  Log: {log_file}")
            return
        # Stale PID — clean it up.
        pid_file.unlink(missing_ok=True)

    cmd = [sys.executable, "-m", "agent.main", "index", "--watch"]
    if getattr(args, "languages", None):
        cmd += ["--languages", args.languages]

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as log_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=config.tools.working_dir,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
        )
    pid_file.write_text(str(proc.pid))
    console.print(f"[green]Index daemon started[/green] (pid {proc.pid})")
    console.print(f"  Log:  {log_file}")
    console.print(f"  Stop: agent index --stop")
    console.print(f"  [dim]Note: only one writer should index at a time.[/dim]")


def cmd_index_daemon_stop(args, config):
    import os
    from rich.console import Console

    console = Console()
    pid_file = _daemon_pid_file(config)
    pid = _read_daemon_pid(config)

    if pid is None:
        console.print("No daemon PID file found — daemon not running.")
        return

    if not _daemon_running(pid):
        console.print(f"[yellow]Daemon (pid {pid}) not running. Removing stale PID file.[/yellow]")
        pid_file.unlink(missing_ok=True)
        return

    try:
        os.kill(pid, 15)  # SIGTERM
    except ProcessLookupError:
        console.print(f"[yellow]Daemon (pid {pid}) already gone.[/yellow]")
    except Exception as e:
        console.print(f"[red]Failed to stop daemon: {e}[/red]")
        return

    # Wait up to 5 seconds for it to exit.
    import time
    for _ in range(10):
        time.sleep(0.5)
        if not _daemon_running(pid):
            break

    pid_file.unlink(missing_ok=True)
    console.print(f"[green]Index daemon stopped[/green] (pid {pid})")


def _daemon_watch_entry(config, languages=None, exclude=None):
    """Entry point for the detached daemon process.

    Installs a SIGTERM handler so the watchdog loop shuts down cleanly,
    writes the PID file, then delegates to _watch_and_reindex().
    """
    import os
    import signal
    from rich.console import Console

    pid_file = _daemon_pid_file(config)
    pid_file.write_text(str(os.getpid()))

    # Convert SIGTERM to KeyboardInterrupt so the watchdog loop catches it.
    def _sigterm(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _sigterm)

    console = Console(stderr=True)
    try:
        _watch_and_reindex(config, console, languages=languages, exclude=exclude)
    finally:
        pid_file.unlink(missing_ok=True)


def cmd_index_stats(args, config):
    from agent.rag.store import VectorStore
    from agent.rag.indexer import pending_files
    from agent.tools.rules import load_rules
    from rich.console import Console
    import datetime as _dt

    store = VectorStore(config.rag)
    stats = store.stats()

    load_rules(config.tools.working_dir)
    pending = pending_files(root=config.tools.working_dir, store=store)
    store.close()

    archive = _open_archive(config)
    astats = archive.stats()
    archive.close()

    console = Console()
    console.print("[bold]Index stats:[/bold]")
    console.print(f"  Files indexed: {stats['files']} / {pending['total']} on disk")
    if pending['pending']:
        console.print(f"  [yellow]Pending:  {pending['pending']} file(s) not yet indexed[/yellow]")
        if getattr(args, "list_pending", False):
            for p in pending["paths"]:
                console.print(f"    [dim]- {p}[/dim]")
    else:
        console.print("  [green]Index is up to date.[/green]")
    console.print(f"  Chunks: {stats['chunks']}")
    console.print(f"  DB:     {config.rag.db_path}")
    console.print("[bold]Archive stats:[/bold]")
    console.print(f"  Files:  {astats['files']}")
    console.print(f"  Chunks: {astats['chunks']}")
    console.print(f"  DB:     {config.rag.archive_db_path}")
    console.print(f"  TTL:    {config.rag.archive_ttl_days} days" + (" (disabled)" if config.rag.archive_ttl_days <= 0 else ""))
    oldest = astats.get("oldest_archived_at")
    if oldest:
        ts = _dt.datetime.fromtimestamp(oldest).isoformat(timespec="seconds")
        console.print(f"  Oldest: {ts}")

    daemon_pid = _read_daemon_pid(config)
    if daemon_pid is not None and _daemon_running(daemon_pid):
        console.print(f"[bold]Daemon:[/bold] [green]running[/green] (pid {daemon_pid})")
        console.print(f"  Log: {_daemon_log_file(config)}")
    else:
        if daemon_pid is not None:
            _daemon_pid_file(config).unlink(missing_ok=True)
        console.print("[bold]Daemon:[/bold] not running  (start with [bold]agent index --daemon[/bold])")

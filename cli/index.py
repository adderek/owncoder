from __future__ import annotations

from pathlib import Path


def _open_archive(config):
    from agent.rag.archive import ArchiveStore
    return ArchiveStore(config.rag.archive_db_path)


def cmd_init(args, config):
    from agent.rag.indexer import index_directory, LANGUAGE_MAP
    from agent.rag.store import VectorStore
    from agent.rag.embedder import Embedder
    from agent.tools.rules import load_rules
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()
    store = VectorStore(config.rag)
    embedder = Embedder(config.embeddings)

    languages = args.languages.split(",") if args.languages else None
    exclude = args.exclude.split(",") if args.exclude else []
    working_dir = config.tools.working_dir

    # Load .agent.ignore et al so index_directory respects them
    load_rules(working_dir)

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
    console.print(
        f"\n[green]Done.[/green] Indexed {stats['indexed']} files, "
        f"skipped {stats['skipped']}, "
        f"created {stats['chunks']} chunks."
    )

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
    console.print("[bold]Index stats:[/bold]")
    console.print(f"  Files:  {stats['files']}")
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

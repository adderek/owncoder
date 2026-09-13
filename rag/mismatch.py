"""Startup guard for an index whose vectors no longer match the configured embedder.

The vector half of the index is stamped with the model that produced it
(``_meta.embedding_model``) and the width of its vectors.  When either no longer
matches ``config.embeddings``, the index is a liability rather than an asset:

* a width mismatch (``float[N]`` vs ``float[M]``) makes cosine distance
  undefined, so a KNN query can return nothing but noise;
* a same-width model swap (bge-m3 vs qwen3, q4_k_m vs q8_0) still computes, but
  ranks by the weaker side of the pair;
* and the background index-update thread would happily write the *new* model's
  vectors into the *old* table, permanently blending two vector spaces in one
  index — the failure this module exists to prevent.

Because the safe action depends on what the operator actually wants (the index
may be expensive to rebuild, or the config change may be the mistake), this
module detects the state and asks, instead of silently picking a destructive
default.  Non-interactive callers get the conservative choice: run frozen.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.config import Config
    from agent.rag.store import VectorStore

log = logging.getLogger(__name__)

# Actions returned by handle_embedding_mismatch().
OK = ""
FROZEN = "frozen"          # run, leave the index untouched, keyword search only
ABORTED = "aborted"        # caller should stop; the store is already closed
REEMBEDDED = "reembedded"  # vectors were rebuilt to match the config


def mismatch_code(store: "VectorStore | None", config: "Config | None") -> str:
    """`""` | `"dims"` | `"model"` — see ``VectorStore.embedding_mismatch``."""
    if store is None or config is None:
        return ""
    try:
        return store.embedding_mismatch(
            config.embeddings.model, config.embeddings.dimensions
        )
    except Exception:
        log.debug("embedding mismatch check failed", exc_info=True)
        return ""


def index_summary(store: "VectorStore") -> str:
    """Human-readable 'model (N-dim)' description of what the index holds."""
    model = ""
    try:
        model = store.get_meta("embedding_model") or "unknown model"
    except Exception:
        model = "unknown model"
    dims = getattr(store, "_vec_dims", None)
    return f"{model} ({dims}-dim)" if dims else model


def backup_index(db_path: str) -> Path | None:
    """Copy the index DB (and its WAL/SHM sidecars) next to the original.

    Must be called with the store closed: copying a live WAL risks a snapshot
    that is missing committed frames.  Returns the backup path, or None when
    there is nothing to copy.
    """
    src = Path(db_path)
    if not src.exists():
        return None
    dst = src.with_name(f"{src.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(src, dst)
    for suffix in ("-wal", "-shm"):
        side = Path(str(src) + suffix)
        if side.exists():
            shutil.copy2(side, Path(str(dst) + suffix))
    return dst


def _reembed(store: "VectorStore", config: "Config", console) -> "tuple[VectorStore, bool]":
    """Back up the index, then replace every vector with the configured model's.

    Returns ``(store, ok)``; ``ok`` is False when the embedder never answered
    and the original vectors were left in place (caller then runs frozen).
    """
    from agent.rag.store import VectorStore
    from agent.rag.embedder import Embedder
    from agent.rag.indexer import reembed_all

    # Close before copying so the backup captures a quiescent WAL.
    store.close()
    backup = backup_index(config.rag.db_path)
    if backup is not None:
        console.print(f"  [dim]backup: {backup}[/dim]")

    store = VectorStore(config.rag)
    embedder = Embedder(config.embeddings)
    result = reembed_all(store, embedder, config.rag)
    if result["aborted"]:
        console.print(
            "[red]Re-embed aborted — index left untouched[/red] "
            "(backup kept). Running frozen instead."
        )
        return store, False
    console.print(
        f"  Re-embedded {result['embedded']}/{result['chunks']} chunks."
    )
    return store, True


def handle_embedding_mismatch(
    store: "VectorStore",
    config: "Config",
    console: Any,
    *,
    interactive: bool,
) -> tuple["VectorStore", str]:
    """Resolve an index/config embedding mismatch at startup.

    Returns ``(store, action)``.  ``store`` is the VectorStore to use for the
    rest of the session — a fresh handle after a re-embed, the original
    otherwise.  ``action`` is one of OK / FROZEN / ABORTED / REEMBEDDED; on
    ABORTED the store has been closed and the caller must stop.
    """
    code = mismatch_code(store, config)
    if not code:
        return store, OK

    emb = config.embeddings
    console.print(
        f"\n[yellow]Embedding mismatch:[/yellow] the index holds "
        f"[bold]{index_summary(store)}[/bold] vectors, but config now uses "
        f"[bold]{emb.model}[/bold] ([dim]{emb.dimensions} dims[/dim])."
    )
    if code == "dims":
        console.print(
            "  [dim]Widths differ — vector search cannot compare the two, so it "
            "would return noise. Keyword search still works.[/dim]"
        )
    else:
        console.print(
            "  [dim]Same width, different model — vector search still runs, but "
            "ranking is degraded.[/dim]"
        )

    if not interactive:
        console.print(
            "  [dim]Non-interactive: running frozen (index untouched).[/dim]"
        )
        return store, FROZEN

    console.print("  [cyan]1[/cyan]  Stop and exit")
    console.print("  [cyan]2[/cyan]  Run frozen — no indexing, keyword search only, vectors preserved")
    console.print("  [cyan]3[/cyan]  Re-embed the index now (backup kept)")
    try:
        choice = input("  Choose [1/2/3] (default 2): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = ""

    if choice in ("1", "stop", "exit", "quit"):
        console.print("[yellow]Stopping.[/yellow]")
        store.close()
        return store, ABORTED
    if choice in ("3", "reembed", "re-embed"):
        store, ok = _reembed(store, config, console)
        return store, REEMBEDDED if ok else FROZEN
    console.print("[dim]Running frozen — index left untouched.[/dim]")
    return store, FROZEN

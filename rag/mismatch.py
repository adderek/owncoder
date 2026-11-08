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


def stale_stamp(store: "VectorStore", config: "Config", n: int = 8) -> bool:
    """True when the stamp names another model but the stored vectors are, in
    fact, the configured model's — checked by re-embedding a random sample.

    The stamp is only metadata; the vectors are the truth. An index-update pass
    cut short used to write Qwen3 vectors and exit before restamping, so the
    stamp kept saying 'bge-m3' and every startup offered a full re-embed of an
    index that needed none (2026-11-08: 857/857 sampled vectors cos 1.000).
    Read-only; any failure means "can't tell" → False.
    """
    from agent.rag.embedder import Embedder
    from agent.rag.indexer import SAME_MODEL_COS, same_space

    try:
        sample = store.sample_vectors(n)
        if not sample:
            return False
        fresh = Embedder(config.embeddings).embed([text for _, text, _ in sample])
        return same_space([vec for _, _, vec in sample], fresh) >= SAME_MODEL_COS
    except Exception:
        log.debug("stale-stamp check failed", exc_info=True)
        return False


def run_reembed(store: "VectorStore", config: "Config", console,
                helpers: "list[str] | None" = None) -> dict:
    """Re-embed with progress, a served-model check and optional helpers.

    *helpers* are launcher specs ('gpu0', 'gpu1', 'cpu') started only for this
    run on free ports and stopped afterwards; they share batches with the
    configured endpoint. Returns reembed_all's result dict (``aborted`` set on a
    model mismatch without touching the index).
    """
    import dataclasses

    from agent.rag import embed_server
    from agent.rag.embedder import Embedder
    from agent.rag.indexer import progress_text, reembed_all

    embedder = Embedder(config.embeddings)
    problem = embedder.model_problem() if hasattr(embedder, "model_problem") else ""
    if problem:
        console.print(f"[red]Wrong embedding model:[/red] {problem} — index left untouched.")
        return {"chunks": 0, "embedded": 0, "failed": 0, "aborted": True,
                "workers": [], "dropped": [problem], "elapsed": 0.0}

    started = []
    if helpers:
        console.print(f"  [dim]starting helpers: {', '.join(helpers)}…[/dim]")
        started = embed_server.start_helpers(
            config, helpers, say=lambda m: console.print(f"  [dim]{m}[/dim]"))
    extras = [Embedder(dataclasses.replace(config.embeddings, base_url=h.base_url))
              for h in started]

    t0 = time.monotonic()
    last = [0.0]

    def _progress(done: int, total: int) -> None:
        now = time.monotonic()
        if now - last[0] >= 10 or done >= total:
            last[0] = now
            console.print(f"  [dim]{progress_text(done, total, now - t0)}[/dim]")

    try:
        result = reembed_all(store, embedder, config.rag, progress_cb=_progress,
                             extra_embedders=extras)
    finally:
        embed_server.stop_helpers(started)
    if result.get("workers"):
        console.print(f"  [dim]workers: {', '.join(result['workers'])}[/dim]")
    for d in result.get("dropped", []):
        console.print(f"  [yellow]helper dropped:[/yellow] {d}")
    return result


def ask_helpers(config: "Config", console) -> "list[str]":
    """Interactive: which temporary servers should join this re-embed."""
    from agent.rag import embed_server

    default = " ".join(config.rag.reembed_helpers or [])
    if not (config.rag.embed_server_command or "").strip():
        return []
    console.print(
        "  Speed up with temporary embed servers (stopped afterwards), e.g. "
        "[cyan]gpu0 gpu1[/cyan] — the configured endpoint keeps working too."
    )
    while True:
        try:
            raw = input(f"  Helpers [{default or 'none'}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = ""
        if raw.lower() in ("none", "no", "-"):
            return []
        try:
            return embed_server.parse_helpers(raw or default)
        except ValueError as e:
            console.print(f"  [red]{e}[/red]")


def _reembed(store: "VectorStore", config: "Config", console) -> "tuple[VectorStore, bool]":
    """Back up the index, then replace every vector with the configured model's.

    Returns ``(store, ok)``; ``ok`` is False when the embedder never answered
    and the original vectors were left in place (caller then runs frozen).
    """
    from agent.rag.store import VectorStore

    helpers = ask_helpers(config, console)
    # Close before copying so the backup captures a quiescent WAL.
    store.close()
    backup = backup_index(config.rag.db_path)
    if backup is not None:
        console.print(f"  [dim]backup: {backup}[/dim]")

    store = VectorStore(config.rag)
    result = run_reembed(store, config, console, helpers)
    if result["aborted"]:
        console.print(
            "[red]Re-embed aborted — index left untouched[/red] "
            "(backup kept). Running frozen instead."
        )
        return store, False
    console.print(
        f"  Re-embedded {result['embedded']}/{result['chunks']} chunks"
        f" in {result.get('elapsed', 0) / 60:.1f} min."
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
    if code == "model" and stale_stamp(store, config):
        prev = store.get_meta("embedding_model")
        store.record_embedding_model(emb.model, emb.base_url)
        console.print(
            f"[dim]Index stamp said {prev}, but sampled vectors match "
            f"{emb.model} — stamp corrected, no re-embed needed.[/dim]"
        )
        log.warning("embedding stamp %r was stale; vectors verified as %r", prev, emb.model)
        return store, OK
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

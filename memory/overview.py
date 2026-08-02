"""One high-level view of everything the agent remembers.

The tiers are real and separate — notes and summaries in memory.db, facts
rounds on disk per session, skills as markdown, code chunks in the RAG
index, the KB corpus — and each had its own tool or CLI flag. Nothing
answered the plain question: *what is stored, and is the index current?*

`overview()` is the single source the HTML panel, `/memory` and any future
status view all read. Counts and paths only: the content stays behind the
recall tools, so this stays cheap and safe to show anywhere.

The index-freshness scan walks the project tree, so it is opt-in (`deep`).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config.models import Config

logger = logging.getLogger(__name__)

# memory.db scopes, in the order a human would ask about them.
_SCOPE_LABELS = {
    "note": "notes",
    "session_summary": "session summaries",
    "context_chunk": "context chunks",
    "facts_round": "facts rounds",
    "behavioral_rule": "behavioral rules",
}


def _size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def rag_db_path(config: "Config") -> Path:
    """Absolute path of the code index.

    `rag.db_path` is written relative to the project, but VectorStore opens
    it as given — resolving it here keeps a caller in another cwd from
    creating a second, empty index.
    """
    db = Path(config.rag.db_path)
    return db if db.is_absolute() else Path(config.tools.working_dir) / db


def overview(config: "Config", session_id: str = "", mode: str = "",
             deep: bool = False, notes_limit: int = 10) -> dict:
    """Counts, paths and a few recent titles for every memory tier.

    Every section is independent and fail-soft: a fresh project, an
    off-the-record session or `kb.enabled = false` drops one row rather
    than the whole view.
    """
    out: dict = {"tiers": [], "notes": [], "skills": [], "warnings": [],
                 "mode": mode or "standard"}
    work = Path(config.tools.working_dir)
    agent_dir = work / config.tools.agent_dir
    out["agent_dir"] = str(agent_dir)

    def _tier(name: str, count, detail: str = "", path: Path | None = None) -> None:
        row = {"name": name, "count": count, "detail": detail}
        if path is not None:
            row["path"] = str(path)
            row["size"] = _size(path)
        out["tiers"].append(row)

    # Cross-session text memory: notes, session summaries, context chunks.
    try:
        from agent.memory.store import MemoryStore
        db = agent_dir / "memory.db"
        store = MemoryStore(db)
        for scope, count in sorted(store.counts_by_scope().items(),
                                   key=lambda kv: -kv[1]):
            _tier(_SCOPE_LABELS.get(scope, scope), count,
                  "memory.db  scope=" + scope, db)
        out["notes"] = [
            {"id": e.get("id"),
             "title": e.get("title") or "(untitled)",
             "tags": _tags(e.get("tags")),
             "updated_at": e.get("updated_at") or e.get("created_at")}
            for e in store.list_entries(scope="note", limit=notes_limit,
                                        order_by="updated_at DESC")
        ]
    except Exception:
        logger.debug("memory overview: store failed", exc_info=True)

    # Sessions on disk.
    try:
        from agent.memory.session import list_sessions
        sess = list_sessions()
        msgs = sum(int(s.get("message_count") or 0) for s in sess)
        _tier("sessions", len(sess), f"{msgs:,} messages on disk")
    except Exception:
        logger.debug("memory overview: sessions failed", exc_info=True)

    # Facts rounds of the *current* session (two-stage compaction tier 2).
    try:
        if session_id:
            from agent.memory.facts_store import FactsStore
            ids = FactsStore(session_id).list_round_ids()
            _tier("facts rounds (this session)", len(ids),
                  f"latest round {ids[-1]}" if ids else "no compaction yet")
    except Exception:
        logger.debug("memory overview: facts rounds failed", exc_info=True)

    # Procedural memory.
    try:
        from agent.skills import SkillLoader
        skills = SkillLoader(config).available()
        _tier("skills", len(skills), str(agent_dir / "skills"))
        out["skills"] = [{"name": n, "description": d} for n, d in skills[:20]]
    except Exception:
        logger.debug("memory overview: skills failed", exc_info=True)

    # Code index (RAG): chunks + embeddings, and optionally how stale it is.
    try:
        db = rag_db_path(config)
        if db.exists():
            import dataclasses
            from agent.rag.store import VectorStore
            store = VectorStore(dataclasses.replace(config.rag, db_path=str(db)))
            st = store.stats()
            model = st.get("embedding_model") or "no embeddings"
            detail = f"{st.get('files', 0)} files · {model}"
            if deep:
                try:
                    from agent.rag.indexer import pending_files
                    from agent.tools.rules import load_rules
                    load_rules(str(work))
                    p = pending_files(root=str(work), store=store, cfg=config.rag)
                    out["index"] = {"pending": p["pending"], "total": p["total"],
                                    "paths": p.get("paths", [])[:20]}
                    detail += (f" · {p['pending']} of {p['total']} files pending"
                               if p["pending"] else " · up to date")
                except Exception:
                    logger.debug("memory overview: pending scan failed", exc_info=True)
            _tier("code index chunks", st.get("chunks", 0), detail, db)
            store.close()
    except Exception:
        logger.debug("memory overview: code index failed", exc_info=True)

    # Knowledge base — a corpus shared across projects, not session state.
    try:
        kb_path = getattr(config.kb, "corpus_path", "")
        if getattr(config.kb, "enabled", False) and kb_path:
            from kb.api import Corpus
            with Corpus.open(kb_path) as corpus:
                nodes = corpus.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            _tier("kb nodes", nodes, kb_path)
    except Exception:
        logger.debug("memory overview: kb failed", exc_info=True)

    if out["mode"] not in ("standard", ""):
        out["warnings"].append(
            f"session mode '{out['mode']}' — writes to these stores are "
            "suppressed or sealed for this session")
    return out


def _tags(raw) -> list[str]:
    try:
        return json.loads(raw or "[]")
    except Exception:
        return []


def _fmt_size(n: int) -> str:
    if not n:
        return ""
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f}MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.0f}kB"
    return f"{n}B"


def format_overview(data: dict) -> str:
    """Plain text for the terminal UIs. No rich markup: `[...]` is eaten by
    the rich console the callers print through."""
    tiers = data.get("tiers") or []
    if not tiers:
        return "Nothing stored yet for this project."
    width = max(len(t["name"]) for t in tiers)
    lines = ["Memory and indexes:"]
    for t in tiers:
        extra = "  ".join(x for x in (t.get("detail", ""),
                                      _fmt_size(t.get("size", 0))) if x)
        lines.append(f"  {t['name']:<{width}}  {t['count']:>7,}"
                     + (f"   {extra}" if extra else ""))
    notes = data.get("notes") or []
    if notes:
        lines.append("")
        lines.append("Recent notes:")
        for n in notes:
            tags = "  " + " ".join(n["tags"]) if n.get("tags") else ""
            lines.append(f"  - {n['title']}{tags}")
    skills = data.get("skills") or []
    if skills:
        lines.append("")
        lines.append("Skills: " + ", ".join(s["name"] for s in skills))
    idx = data.get("index")
    if idx and idx.get("pending"):
        lines.append("")
        lines.append(f"Index: {idx['pending']} of {idx['total']} files not indexed"
                     " — run 'agent index --update'")
    if data.get("agent_dir"):
        lines.append("")
        lines.append("Agent dir: " + data["agent_dir"])
    for w in data.get("warnings") or []:
        lines.append("! " + w)
    return "\n".join(lines)


def run_memory_command(config: "Config", arg: str = "", session_id: str = "",
                       mode: str = "") -> str:
    """`/memory [index|notes [n]]` — the shared text view for every UI."""
    arg = (arg or "").strip()
    parts = arg.split()
    sub = parts[0].lower() if parts else ""

    if sub in ("", "status", "show"):
        return format_overview(overview(config, session_id=session_id, mode=mode))
    if sub == "index":
        data = overview(config, session_id=session_id, mode=mode, deep=True)
        idx = data.get("index")
        tier = next((t for t in data["tiers"] if t["name"] == "code index chunks"), None)
        if tier is None:
            return "No code index yet — run 'agent init' to build one."
        out = [f"Code index: {tier['count']:,} chunks   {tier['detail']}",
               f"  DB: {tier.get('path', '')}  {_fmt_size(tier.get('size', 0))}"]
        if idx and idx.get("paths"):
            out.append("  Not yet indexed:")
            out += [f"    - {p}" for p in idx["paths"]]
        return "\n".join(out)
    if sub == "notes":
        limit = 20
        if len(parts) > 1 and parts[1].isdigit():
            limit = max(1, min(200, int(parts[1])))
        data = overview(config, session_id=session_id, mode=mode, notes_limit=limit)
        notes = data.get("notes") or []
        if not notes:
            return "No notes saved for this project yet."
        lines = [f"Notes (newest {len(notes)}):"]
        for n in notes:
            tags = "  " + " ".join(n["tags"]) if n.get("tags") else ""
            lines.append(f"  - {n['title']}{tags}")
        return "\n".join(lines)
    return ("Usage: /memory  |  /memory index  |  /memory notes [n]")

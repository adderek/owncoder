"""Cache path helpers and index load/save/prune."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import agent.prompt_compiler._state as _s
from ._state import _Entry

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_KNOWN_PROMPT_FILES = ("system.txt", "analyze.txt", "synthesize.txt", "base_rules.txt")


def _cache_key(api_base: str, model: str, original: str) -> str:
    h = hashlib.sha256()
    h.update(api_base.encode("utf-8", errors="replace"))
    h.update(b"\x00")
    h.update(model.encode("utf-8", errors="replace"))
    h.update(b"\x00")
    h.update(original.encode("utf-8", errors="replace"))
    return h.hexdigest()[:24]


def _cache_dir(config: "Config") -> Path:
    base = Path(config.tools.working_dir) / config.compile_prompts.cache_dir
    base.mkdir(parents=True, exist_ok=True)
    return base


def _compiled_path(config: "Config", key: str) -> Path:
    return _cache_dir(config) / f"{key}.txt"


def _index_file(config: "Config") -> Path:
    return _cache_dir(config) / "index.json"


def _load_stripped(path: Path) -> str:
    """Read a prompt file, stripping comment lines (lines starting with '#')."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(l for l in lines if not l.startswith("#")).strip()


def _known_targets() -> list[tuple[str, str]]:
    """Enumerate (logical_name, original_text) for every shippable prompt."""
    out: list[tuple[str, str]] = []
    for fname in _KNOWN_PROMPT_FILES:
        p = _PROMPTS_DIR / fname
        if not p.is_file():
            continue
        text = _load_stripped(p) if fname == "base_rules.txt" else p.read_text(encoding="utf-8")
        if text:
            out.append((fname, text))
    for subdir in ("guidelines", "inline"):
        d = _PROMPTS_DIR / subdir
        if d.is_dir():
            for p in sorted(d.glob("*.txt")):
                out.append((f"{subdir}/{p.name}", p.read_text(encoding="utf-8")))
    return out


def _ensure_loaded(config: "Config") -> None:
    path = _index_file(config)
    if _s._index is not None and _s._index_path == path:
        return
    _s._index_path = path
    if not path.exists():
        _s._index = {}
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        _s._index = {k: _Entry(**v) for k, v in raw.items()}
    except Exception as e:
        logger.warning("compile_prompts: failed to read index %s: %s", path, e)
        _s._index = {}
    _prune_orphans(config)


def _prune_orphans(config: "Config") -> None:
    """Drop rows whose (name, model, api_base) tuple no longer matches live prompt content.
    Caller must hold _lock. Safe with missing prompt files (skipped, not pruned).
    """
    if _s._index is None or not _s._index:
        return
    try:
        live_keys: dict[tuple[str, str, str], str] = {}
        api_base = config.llm.base_url
        model = config.llm.model
        for pname, original in _known_targets():
            live_keys[(pname, model, api_base)] = _cache_key(api_base, model, original)
    except Exception as e:
        logger.warning("compile_prompts: prune skipped, _known_targets failed: %s", e)
        return
    removed = 0
    for key in list(_s._index.keys()):
        entry = _s._index[key]
        live = live_keys.get((entry.name, entry.model, entry.api_base))
        if live is None:
            continue
        if live != key:
            try:
                _compiled_path(config, key).unlink(missing_ok=True)
            except Exception:
                pass
            del _s._index[key]
            removed += 1
    if removed:
        _save_index()


def _save_index() -> None:
    """Write the in-memory index to disk. Caller must hold _lock."""
    if _s._index is None or _s._index_path is None:
        return
    tmp = _s._index_path.with_suffix(".json.tmp")
    payload = {k: asdict(v) for k, v in _s._index.items()}
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(_s._index_path)

"""Integrity / tamper detection for the agent's own trusted files (Tier-3 #14).

The agent's behavior is steered by files it treats as trusted: self-evolved skills
(`.agent/skills/*.md`, procedural memory) and config (`agent.toml` / `agent.yaml`).
A poisoned skill or a flipped config flag (e.g. disabling the sandbox) is a persistent
compromise that survives restarts and is invisible in normal use. This module lets the
operator SEAL those files (record an HMAC of each) and later CHECK for drift —
modified, added, or deleted — so tampering surfaces.

HMAC-SHA256 with a local random key kept at `<agent_dir>/integrity.key` (0600). stdlib
only, no external dependency, fully offline. The key never leaves the machine; an
attacker who cannot read the key cannot forge a matching manifest entry.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config


def _root(config) -> Path:
    return Path(getattr(getattr(config, "tools", None), "working_dir", ".") or ".").resolve()


def _agent_dir(config) -> Path:
    ad = Path(getattr(getattr(config, "tools", None), "agent_dir", ".agent") or ".agent")
    return ad if ad.is_absolute() else _root(config) / ad


def _key_path(config) -> Path:
    return _agent_dir(config) / "integrity.key"


def _manifest_path(config) -> Path:
    return _agent_dir(config) / "integrity.json"


def _load_or_create_key(config) -> bytes:
    p = _key_path(config)
    if p.exists():
        return p.read_bytes()
    p.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    # Write 0600 so the key is not world-readable.
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)
    return key


def _hmac_file(key: bytes, path: Path) -> str:
    h = hmac.new(key, digestmod=hashlib.sha256)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_files(config) -> list[Path]:
    """Trusted files whose tampering matters: skills + config. Existing only."""
    out: list[Path] = []
    skills_dir = _agent_dir(config) / "skills"
    if skills_dir.is_dir():
        out += sorted(skills_dir.glob("*.md"))  # non-recursive: skips .history/
    root = _root(config)
    for name in ("agent.toml", "agent.yaml", "agent.yml"):
        f = root / name
        if f.is_file():
            out.append(f)
    return out


def _rel(config, p: Path) -> str:
    try:
        return str(p.relative_to(_root(config)))
    except ValueError:
        return str(p)


def seal(config) -> int:
    """Record an HMAC for every protected file. Returns count sealed."""
    key = _load_or_create_key(config)
    entries = {_rel(config, p): _hmac_file(key, p) for p in protected_files(config)}
    mp = _manifest_path(config)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps({
        "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "algo": "hmac-sha256",
        "entries": entries,
    }, indent=2))
    return len(entries)


def load_manifest(config) -> dict:
    p = _manifest_path(config)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text()).get("entries", {})
    except (json.JSONDecodeError, OSError, AttributeError):
        return {}


def check(config) -> dict:
    """Compare current protected files against the sealed manifest.

    Returns {sealed, modified[], added[], deleted[], ok}. ``added`` = protected
    file present but not in manifest; ``deleted`` = in manifest but now missing.
    """
    manifest = load_manifest(config)
    if not manifest:
        return {"sealed": False, "modified": [], "added": [], "deleted": [], "ok": True}

    key = _load_or_create_key(config)
    current = {_rel(config, p): p for p in protected_files(config)}

    modified, added = [], []
    for rel, p in current.items():
        expected = manifest.get(rel)
        if expected is None:
            added.append(rel)
        elif not hmac.compare_digest(expected, _hmac_file(key, p)):
            modified.append(rel)
    deleted = [rel for rel in manifest if rel not in current]

    ok = not (modified or added or deleted)
    return {"sealed": True, "modified": sorted(modified), "added": sorted(added),
            "deleted": sorted(deleted), "ok": ok}


def warn_if_tampered(config) -> str | None:
    """One-line warning if drift is detected, else None. Safe to call on startup."""
    try:
        res = check(config)
    except Exception:  # noqa: BLE001
        return None
    if not res["sealed"] or res["ok"]:
        return None
    bits = []
    for k in ("modified", "added", "deleted"):
        if res[k]:
            bits.append(f"{len(res[k])} {k}")
    return f"INTEGRITY WARNING: trusted files changed since seal ({', '.join(bits)}). Run /security integrity check."


def run_integrity_command(config, arg: str) -> str:
    """Text handler for `/security integrity [seal|check|status]` (both UIs)."""
    v = arg.strip().lower()
    if v == "seal":
        n = seal(config)
        return f"Sealed {n} trusted file(s). Manifest: {_manifest_path(config)}"
    if v in ("", "check", "status"):
        res = check(config)
        if not res["sealed"]:
            return "No integrity manifest. Run /security integrity seal to establish a baseline."
        if res["ok"]:
            return f"Integrity OK — all {len(load_manifest(config))} sealed file(s) unchanged."
        lines = ["INTEGRITY DRIFT DETECTED:"]
        for k in ("modified", "added", "deleted"):
            for rel in res[k]:
                lines.append(f"  {k:8s} {rel}")
        lines.append("Review changes, then /security integrity seal to re-baseline if intended.")
        return "\n".join(lines)
    return "Usage: /security integrity [seal | check | status]"

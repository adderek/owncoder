"""Weight vault — pin and verify local model files (Tier-1/supply-chain #15).

owncoder's one true dependency in an air-gapped world is the model weights on disk.
If access to hosted models is pulled (the Fable/Mythos scenario), the local `.gguf`/
`.safetensors` files are all that's left — so their integrity matters: silent
corruption, an accidental swap to a weaker/biased model, or a tampered download should
not go unnoticed.

This records a sha256 + size + provenance for each pinned weight file and verifies on
demand. Weight files are large (GB), so two checks are offered: a cheap quickcheck
(size + mtime) for routine startup, and a full sha256 verify when integrity must be
proven. stdlib only, offline.

Note: the manifest itself is plain JSON; seal it with `/security integrity` (it lives
under .agent/) if you need tamper-evidence on the manifest too.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


def _agent_dir(config) -> Path:
    root = Path(getattr(getattr(config, "tools", None), "working_dir", ".") or ".").resolve()
    ad = Path(getattr(getattr(config, "tools", None), "agent_dir", ".agent") or ".agent")
    return ad if ad.is_absolute() else root / ad


def _manifest_path(config) -> Path:
    return _agent_dir(config) / "weights.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(config) -> dict:
    p = _manifest_path(config)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text()).get("weights", {})
    except (json.JSONDecodeError, OSError, AttributeError):
        return {}


def _save(config, weights: dict) -> None:
    p = _manifest_path(config)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "weights": weights,
    }, indent=2))


def pin(config, path: str, source: str = "") -> dict:
    """Record sha256 + size + mtime + provenance for a weight file."""
    p = Path(os.path.expanduser(path)).resolve()
    if not p.is_file():
        return {"error": f"not a file: {path}"}
    st = p.stat()
    entry = {
        "sha256": _sha256(p),
        "size": st.st_size,
        "mtime": int(st.st_mtime),
        "source": source.strip(),
        "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    weights = _load(config)
    weights[str(p)] = entry
    _save(config, weights)
    return {"ok": True, "path": str(p), "sha256": entry["sha256"], "size": entry["size"]}


def quickcheck(config) -> dict:
    """Fast verify: size + mtime only (no hashing). Returns drift summary."""
    weights = _load(config)
    if not weights:
        return {"pinned": False, "ok": True, "missing": [], "changed": []}
    missing, changed = [], []
    for path, e in weights.items():
        p = Path(path)
        if not p.is_file():
            missing.append(path)
            continue
        st = p.stat()
        if st.st_size != e["size"] or int(st.st_mtime) != e.get("mtime"):
            changed.append(path)
    return {"pinned": True, "ok": not (missing or changed),
            "missing": sorted(missing), "changed": sorted(changed)}


def verify(config, path: str | None = None) -> dict:
    """Full sha256 verify of one pinned file, or all if path is None."""
    weights = _load(config)
    if not weights:
        return {"pinned": False, "ok": True, "missing": [], "mismatched": []}
    targets = weights
    if path:
        rp = str(Path(os.path.expanduser(path)).resolve())
        if rp not in weights:
            return {"error": f"not pinned: {path}"}
        targets = {rp: weights[rp]}
    missing, mismatched, verified = [], [], []
    for p, e in targets.items():
        fp = Path(p)
        if not fp.is_file():
            missing.append(p)
            continue
        if _sha256(fp) != e["sha256"]:
            mismatched.append(p)
        else:
            verified.append(p)
    return {"pinned": True, "ok": not (missing or mismatched),
            "verified": sorted(verified), "missing": sorted(missing),
            "mismatched": sorted(mismatched)}


def warn_if_drift(config) -> str | None:
    """One-line warning on quickcheck drift, else None. Cheap; safe at startup."""
    try:
        res = quickcheck(config)
    except Exception:  # noqa: BLE001
        return None
    if not res["pinned"] or res["ok"]:
        return None
    bits = []
    if res["missing"]:
        bits.append(f"{len(res['missing'])} missing")
    if res["changed"]:
        bits.append(f"{len(res['changed'])} changed")
    return f"WEIGHT VAULT WARNING: pinned model file(s) {', '.join(bits)}. Run /security weights verify."


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def run_weights_command(config, arg: str) -> str:
    """Text handler for `/security weights [pin <path> [source] | verify [path] | quickcheck | list]`."""
    parts = arg.strip().split()
    sub = parts[0].lower() if parts else "list"

    if sub == "pin":
        if len(parts) < 2:
            return "Usage: /security weights pin <path> [source]"
        path = parts[1]
        source = " ".join(parts[2:])
        res = pin(config, path, source)
        if res.get("error"):
            return f"Error: {res['error']}"
        return f"Pinned {res['path']}\n  sha256={res['sha256']}\n  size={_human(res['size'])}"

    if sub in ("verify", "check"):
        path = parts[1] if len(parts) > 1 else None
        res = verify(config, path)
        if res.get("error"):
            return f"Error: {res['error']}"
        if not res["pinned"]:
            return "No weights pinned. Use /security weights pin <path>."
        if res["ok"]:
            return f"Weights OK — {len(res['verified'])} file(s) match pinned sha256."
        lines = ["WEIGHT VERIFY FAILED:"]
        for p in res["mismatched"]:
            lines.append(f"  MISMATCH  {p}")
        for p in res["missing"]:
            lines.append(f"  MISSING   {p}")
        return "\n".join(lines)

    if sub == "quickcheck":
        res = quickcheck(config)
        if not res["pinned"]:
            return "No weights pinned."
        if res["ok"]:
            return "Quickcheck OK (size+mtime unchanged). Use 'verify' for full sha256."
        lines = ["QUICKCHECK DRIFT:"]
        for p in res["changed"]:
            lines.append(f"  CHANGED  {p}")
        for p in res["missing"]:
            lines.append(f"  MISSING  {p}")
        return "\n".join(lines)

    if sub in ("list", "ls", ""):
        weights = _load(config)
        if not weights:
            return "No weights pinned. Use /security weights pin <path> [source]."
        lines = [f"Pinned weights ({len(weights)}):"]
        for p, e in sorted(weights.items()):
            src = f"  src={e['source']}" if e.get("source") else ""
            lines.append(f"  {p}  [{_human(e['size'])}]  {e['sha256'][:12]}…{src}")
        return "\n".join(lines)

    return "Usage: /security weights [pin <path> [source] | verify [path] | quickcheck | list]"

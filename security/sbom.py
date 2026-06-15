"""SBOM + dependency audit (Tier-1 #2).

Enumerates a project's declared dependencies (Python, Node, Rust, Go) into a Software
Bill of Materials, then — if a LOCAL offline vulnerability database is present — flags
components with known CVEs. No network: the vuln DB is a mirror the operator drops in,
matching the air-gapped premise of the suite. Without a DB, the SBOM itself is still
useful: it lists every dependency + version and flags floating (unpinned) ranges, which
are the supply-chain blast radius.

Offline vuln DB format (`<agent_dir>/vulndb/<ecosystem>.json`):

    { "<package>": [ {"id": "CVE-…", "severity": "high",
                      "introduced": "1.0.0", "fixed": "1.2.3",
                      "summary": "…"} ] }

A component matches an advisory when introduced <= version < fixed (fixed omitted =
all later versions affected). Version compare uses packaging.version when available,
else a numeric-tuple fallback. stdlib only otherwise.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Component:
    ecosystem: str
    name: str
    version: str        # "" if unknown/floating
    pinned: bool
    source: str         # file it came from (relative)


# ── version compare ──────────────────────────────────────────────────────────

def _vkey(v: str):
    try:
        from packaging.version import Version
        return ("pkg", Version(v))
    except Exception:  # noqa: BLE001 - packaging missing or unparseable
        parts = re.findall(r"\d+", v)
        return ("tup", tuple(int(x) for x in parts) or (0,))


def _vlt(a: str, b: str) -> bool:
    ka, kb = _vkey(a), _vkey(b)
    if ka[0] == kb[0]:
        try:
            return ka[1] < kb[1]
        except TypeError:
            return str(ka[1]) < str(kb[1])
    return str(a) < str(b)


def _affected(version: str, adv: dict) -> bool:
    if not version:
        return False
    introduced = adv.get("introduced")
    fixed = adv.get("fixed")
    if introduced and _vlt(version, introduced):
        return False
    if fixed and not _vlt(version, fixed):
        return False
    return True


# ── ecosystem parsers ──────────────────────────────────────────────────────

_REQ_RE = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*(==|>=|~=|<=|>|<|!=)?\s*([A-Za-z0-9._*+-]+)?")


def _parse_requirements(text: str, src: str) -> list[Component]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        line = line.split("#", 1)[0].split(";", 1)[0].strip()
        m = _REQ_RE.match(line)
        if not m or not m.group(1):
            continue
        name, op, ver = m.group(1), m.group(2), m.group(3) or ""
        out.append(Component("pypi", name.lower(), ver if op == "==" else "",
                             pinned=(op == "=="), source=src))
    return out


def _parse_pyproject(text: str, src: str) -> list[Component]:
    try:
        import tomllib
        data = tomllib.loads(text)
    except Exception:  # noqa: BLE001
        return []
    out = []
    deps = (data.get("project", {}) or {}).get("dependencies", []) or []
    for d in deps:
        m = _REQ_RE.match(d)
        if m and m.group(1):
            out.append(Component("pypi", m.group(1).lower(),
                                 m.group(3) or "" if m.group(2) == "==" else "",
                                 pinned=(m.group(2) == "=="), source=src))
    poetry = (data.get("tool", {}) or {}).get("poetry", {}) or {}
    for name, spec in (poetry.get("dependencies", {}) or {}).items():
        if name.lower() == "python":
            continue
        ver = spec if isinstance(spec, str) else (spec.get("version", "") if isinstance(spec, dict) else "")
        ver = ver.lstrip("^~=<> ")
        out.append(Component("pypi", name.lower(), ver, pinned=False, source=src))
    return out


def _parse_package_lock(text: str, src: str) -> list[Component]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    out = []
    pkgs = data.get("packages")
    if isinstance(pkgs, dict):  # lockfile v2/v3
        for path, meta in pkgs.items():
            if not path.startswith("node_modules/"):
                continue
            name = path.split("node_modules/")[-1]
            ver = (meta or {}).get("version", "")
            if name and ver:
                out.append(Component("npm", name, ver, pinned=True, source=src))
        if out:
            return out
    deps = data.get("dependencies")  # lockfile v1
    if isinstance(deps, dict):
        for name, meta in deps.items():
            ver = (meta or {}).get("version", "")
            out.append(Component("npm", name, ver, pinned=bool(ver), source=src))
    return out


def _parse_package_json(text: str, src: str) -> list[Component]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    out = []
    for key in ("dependencies", "devDependencies"):
        for name, ver in (data.get(key) or {}).items():
            clean = str(ver).lstrip("^~>=< ")
            out.append(Component("npm", name, clean, pinned=str(ver).lstrip()[:1].isdigit(), source=src))
    return out


def _parse_cargo_lock(text: str, src: str) -> list[Component]:
    try:
        import tomllib
        data = tomllib.loads(text)
    except Exception:  # noqa: BLE001
        return []
    return [Component("crates", p.get("name", ""), p.get("version", ""), True, src)
            for p in data.get("package", []) if p.get("name")]


def _parse_go_mod(text: str, src: str) -> list[Component]:
    out, in_block = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("require ("):
            in_block = True
            continue
        if in_block and s == ")":
            in_block = False
            continue
        if s.startswith("require "):
            s = s[len("require "):].strip()
        elif not in_block:
            continue
        parts = s.split()
        if len(parts) >= 2 and parts[0] != "//":
            out.append(Component("go", parts[0], parts[1].lstrip("v"), True, src))
    return out


# filename → parser
_PARSERS = [
    ("package-lock.json", _parse_package_lock),
    ("package.json", _parse_package_json),
    ("pyproject.toml", _parse_pyproject),
    ("Cargo.lock", _parse_cargo_lock),
    ("go.mod", _parse_go_mod),
]


# ── orchestration ────────────────────────────────────────────────────────────

def _agent_dir(config) -> Path:
    root = Path(getattr(getattr(config, "tools", None), "working_dir", ".") or ".").resolve()
    ad = Path(getattr(getattr(config, "tools", None), "agent_dir", ".agent") or ".agent")
    return ad if ad.is_absolute() else root / ad


def build_sbom(target: str) -> list[Component]:
    """Enumerate declared dependencies under *target* (top level + one nesting)."""
    root = Path(target).resolve()
    comps: list[Component] = []
    seen_files = set()

    # Top-level manifests + requirements*.txt anywhere shallow.
    candidates = list(root.glob("*")) + list(root.glob("*/*"))
    for f in candidates:
        if not f.is_file() or f in seen_files:
            continue
        rel = str(f.relative_to(root))
        if any(part in (".venv", "node_modules", ".git", "vendor") for part in f.parts):
            continue
        if f.name.startswith("requirements") and f.suffix == ".txt":
            comps += _parse_requirements(f.read_text(errors="ignore"), rel)
            seen_files.add(f)
            continue
        for fname, parser in _PARSERS:
            if f.name == fname:
                comps += parser(f.read_text(errors="ignore"), rel)
                seen_files.add(f)
                break

    # Dedupe by (ecosystem, name, version, source).
    uniq, keys = [], set()
    for c in comps:
        k = (c.ecosystem, c.name, c.version, c.source)
        if k not in keys:
            keys.add(k)
            uniq.append(c)
    return uniq


def load_vulndb(config) -> dict:
    """Load all <agent_dir>/vulndb/<ecosystem>.json into {ecosystem: {pkg: [adv]}}."""
    d = _agent_dir(config) / "vulndb"
    if not d.is_dir():
        return {}
    db = {}
    for f in d.glob("*.json"):
        try:
            db[f.stem] = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
    return db


def match_vulns(comps: list[Component], db: dict) -> list[dict]:
    """Return advisories that match SBOM components."""
    hits = []
    for c in comps:
        eco = db.get(c.ecosystem)
        if not eco:
            continue
        for adv in eco.get(c.name, []):
            if _affected(c.version, adv):
                hits.append({
                    "ecosystem": c.ecosystem, "name": c.name, "version": c.version,
                    "id": adv.get("id", "?"), "severity": adv.get("severity", "unknown"),
                    "fixed": adv.get("fixed", ""), "summary": adv.get("summary", ""),
                    "source": c.source,
                })
    return hits


def run_sbom_command(config, arg: str) -> str:
    """Text handler for `/security sbom [path]`."""
    parts = arg.strip().split()
    target = parts[0] if parts else (
        getattr(getattr(config, "tools", None), "working_dir", ".") or ".")
    if not Path(target).is_dir():
        return f"Error: not a directory: {target}"

    comps = build_sbom(target)
    if not comps:
        return f"No dependency manifests found under {target} (pyproject/requirements/package*.json/Cargo.lock/go.mod)."

    by_eco: dict[str, int] = {}
    floating = 0
    for c in comps:
        by_eco[c.ecosystem] = by_eco.get(c.ecosystem, 0) + 1
        if not c.pinned:
            floating += 1

    eco_line = "  ".join(f"{k}={v}" for k, v in sorted(by_eco.items()))
    lines = [
        f"# SBOM — {Path(target).resolve()}",
        f"- components: {len(comps)}  ({eco_line})",
        f"- unpinned (floating ranges): {floating}",
    ]

    db = load_vulndb(config)
    if not db:
        lines.append("- vuln DB: none installed (SBOM only — drop OSV mirror in .agent/vulndb/)")
    else:
        hits = match_vulns(comps, db)
        lines.append(f"- vuln DB: {', '.join(sorted(db))} → {len(hits)} known vulnerable component(s)")
        if hits:
            lines += ["", "| sev | id | package | version | fixed | summary |",
                      "|---|---|---|---|---|---|"]
            for h in hits:
                lines.append(f"| {h['severity']} | {h['id']} | {h['name']} | "
                             f"{h['version']} | {h['fixed'] or '—'} | "
                             f"{h['summary'].replace('|', '/')[:80]} |")
    return "\n".join(lines)


def to_json(comps: list[Component]) -> str:
    return json.dumps([asdict(c) for c in comps], indent=2)

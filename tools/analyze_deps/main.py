"""`analyze_dependencies` tool — dependency hygiene report for a project.

Builds on the existing SBOM layer (``agent.security.sbom`` parses
requirements*.txt / pyproject / package-lock / Cargo.lock / go.mod and matches
the offline vuln DB) and adds the analyses the SBOM doesn't do:

- unused        — declared Python deps never imported by project code.
                  Dist→module mapping comes from the project venv's
                  ``*.dist-info`` metadata when available (accurate), else a
                  normalization + alias heuristic (flagged as such).
- conflicts     — ``pip check`` in the project venv (authoritative resolver
                  view: broken requires, version clashes).
- outdated      — ``pip list --outdated`` in the project venv. Network; skipped
                  under airgap or with check_outdated=false.
- unpinned      — floating versions from the SBOM (supply-chain drift risk).
- vulnerabilities — offline vuln DB matches from the SBOM layer.

Unused/conflicts/outdated are Python-first (the manifest formats named in the
request); other ecosystems still appear in the component counts + vuln match.
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import re
import subprocess
from typing import TYPE_CHECKING, Any

from agent.tools import register
from agent.tools._common import working_dir

if TYPE_CHECKING:
    from agent.config import Config

_config: "Config | None" = None

_PIP_TIMEOUT_S = 60
_OUTDATED_TIMEOUT_S = 120
_MAX_PY_FILES = 5000

# Dist name → import name(s) for common mismatches (heuristic fallback when no
# venv metadata is available).
_IMPORT_ALIASES: dict[str, tuple[str, ...]] = {
    "pyyaml": ("yaml",),
    "pillow": ("PIL",),
    "beautifulsoup4": ("bs4",),
    "scikit-learn": ("sklearn",),
    "opencv-python": ("cv2",),
    "python-dateutil": ("dateutil",),
    "python-dotenv": ("dotenv",),
    "msgpack-python": ("msgpack",),
    "protobuf": ("google",),
    "pycryptodome": ("Crypto",),
    "typing-extensions": ("typing_extensions",),
    "importlib-metadata": ("importlib_metadata",),
    "setuptools": ("setuptools", "pkg_resources"),
}

# Dists that are used without ever being imported by project code.
_INDIRECT_PREFIXES = ("pytest-", "flake8-", "mypy-", "types-", "ruff-")
_INDIRECT_NAMES = frozenset({
    "pip", "wheel", "build", "tox", "pre-commit", "coverage", "uvloop",
    "gunicorn", "uvicorn", "supervisor",
})


def setup(config: "Config") -> None:
    global _config
    _config = config


def _find_venv(root: str) -> str | None:
    for rel in (".venv", "venv"):
        py = os.path.join(root, rel, "bin", "python")
        if os.access(py, os.X_OK):
            return os.path.join(root, rel)
    return None


def _venv_dist_modules(venv: str) -> dict[str, set[str]]:
    """Map normalized dist name → top-level import names, from dist-info."""
    out: dict[str, set[str]] = {}
    for info in glob.glob(os.path.join(venv, "lib", "python*", "site-packages",
                                       "*.dist-info")):
        base = os.path.basename(info).removesuffix(".dist-info")
        name = (base.rsplit("-", 1)[0] if "-" in base else base).lower().replace("_", "-")
        modules: set[str] = set()
        tl = os.path.join(info, "top_level.txt")
        try:
            if os.path.exists(tl):
                modules = {ln.strip() for ln in open(tl, encoding="utf-8") if ln.strip()}
            else:
                # Fall back to RECORD: top-level dirs/modules the dist installs.
                rec = os.path.join(info, "RECORD")
                if os.path.exists(rec):
                    for ln in open(rec, encoding="utf-8", errors="replace"):
                        top = ln.split(",", 1)[0].split("/", 1)[0]
                        if top and not top.endswith((".dist-info", ".pth")) \
                                and not top.startswith("_") and "." not in top:
                            modules.add(top)
        except OSError:
            continue
        if modules:
            out[name] = modules
    return out


def _collect_imports(root: str) -> set[str]:
    """Top-level module names imported anywhere in the project's .py files."""
    skip = {".git", ".venv", "venv", "node_modules", "__pycache__", ".tox",
            "site-packages", "dist", "build"}
    imports: set[str] = set()
    seen = 0
    pat = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w]*)|import\s+([A-Za-z_][\w]*))",
                     re.MULTILINE)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for f in filenames:
            if not f.endswith(".py"):
                continue
            seen += 1
            if seen > _MAX_PY_FILES:
                return imports
            try:
                text = open(os.path.join(dirpath, f), encoding="utf-8",
                            errors="replace").read()
            except OSError:
                continue
            for m in pat.finditer(text):
                imports.add(m.group(1) or m.group(2))
    return imports


def _norm(name: str) -> str:
    return name.lower().replace("_", "-")


def _find_unused(declared: list, imports: set[str],
                 dist_modules: dict[str, set[str]]) -> list[dict]:
    unused = []
    for comp in declared:
        name = _norm(comp.name)
        if name in _INDIRECT_NAMES or name.startswith(_INDIRECT_PREFIXES):
            continue
        modules = dist_modules.get(name)
        if modules is not None:
            if modules & imports:
                continue
            confidence = "venv-verified"
        else:
            candidates = set(_IMPORT_ALIASES.get(name, ()))
            candidates.add(name.replace("-", "_"))
            candidates.add(name)
            if candidates & imports:
                continue
            confidence = "heuristic"
        unused.append({"name": comp.name, "source": comp.source,
                       "confidence": confidence})
    return unused


def _run_pip(venv: str, args: list[str], timeout_s: int) -> tuple[int, str]:
    py = os.path.join(venv, "bin", "python")
    try:
        proc = subprocess.run([py, "-m", "pip", *args], capture_output=True,
                              text=True, timeout=timeout_s)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"pip {' '.join(args)} timed out after {timeout_s}s"
    except OSError as e:
        return 127, str(e)


def _pip_conflicts(venv: str) -> tuple[list[str], str | None]:
    rc, out = _run_pip(venv, ["check"], _PIP_TIMEOUT_S)
    if rc == 0:
        return [], None
    if rc in (124, 127):
        return [], out
    lines = [ln.strip() for ln in out.splitlines()
             if ln.strip() and "pip is available" not in ln]
    return lines[:50], None


def _pip_outdated(venv: str) -> tuple[list[dict], str | None]:
    rc, out = _run_pip(venv, ["list", "--outdated", "--format=json",
                              "--disable-pip-version-check"], _OUTDATED_TIMEOUT_S)
    if rc != 0:
        return [], out.strip()[:500] or f"pip list --outdated failed (rc {rc})"
    try:
        rows = json.loads(out[out.index("["):out.rindex("]") + 1])
    except Exception:
        return [], "could not parse pip list --outdated output"
    return [{"name": r.get("name", ""), "installed": r.get("version", ""),
             "latest": r.get("latest_version", "")} for r in rows][:100], None


@register(
    "analyze_dependencies",
    {
        "description": (
            "Dependency hygiene report: unused declared dependencies (declared in "
            "requirements/pyproject but never imported), version conflicts (pip check "
            "in the project venv), outdated packages (pip list --outdated; skipped "
            "when air-gapped), unpinned/floating versions, and offline vuln-DB "
            "matches. Python-first; other ecosystems (npm/cargo/go) are included in "
            "component counts and vuln matching via the SBOM layer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Project directory. Default: working dir.",
                },
                "check_outdated": {
                    "type": "boolean",
                    "description": "Query the package index for newer versions (network). Default true.",
                },
            },
            "required": [],
        },
    },
)
async def analyze_dependencies(path: str = "", check_outdated: bool = True) -> dict[str, Any]:
    from agent.security import sbom

    root = os.path.abspath(os.path.expanduser(path.strip() or working_dir(_config)))
    if not os.path.isdir(root):
        return {"error": f"not a directory: {root}"}

    comps = await asyncio.to_thread(sbom.build_sbom, root)
    if not comps:
        return {"error": "no dependency manifests found "
                         "(requirements*.txt, pyproject.toml, package-lock.json, "
                         "Cargo.lock, go.mod)"}

    notes: list[str] = []
    pypi = [c for c in comps if c.ecosystem == "pypi"]
    by_eco: dict[str, int] = {}
    for c in comps:
        by_eco[c.ecosystem] = by_eco.get(c.ecosystem, 0) + 1

    # Unused (python): declared vs imported.
    unused: list[dict] = []
    if pypi:
        venv = _find_venv(root)
        dist_modules = await asyncio.to_thread(_venv_dist_modules, venv) if venv else {}
        if not dist_modules:
            notes.append("no project venv metadata; unused-detection is heuristic "
                         "(name/alias matching) and may have false positives")
        imports = await asyncio.to_thread(_collect_imports, root)
        unused = _find_unused(pypi, imports, dist_modules)
        if unused:
            notes.append("unused-detection sees only static imports; packages loaded "
                         "via importlib/plugin entry points (e.g. tree-sitter language "
                         "packs) can be false positives — verify before removing")

    # Conflicts + outdated need an installed environment.
    conflicts: list[str] = []
    outdated: list[dict] = []
    venv = _find_venv(root)
    if venv:
        conflicts, err = await asyncio.to_thread(_pip_conflicts, venv)
        if err:
            notes.append(f"pip check unavailable: {err}")
        if check_outdated:
            from agent.security.airgap import is_enabled as _airgapped
            if _config is not None and _airgapped(_config):
                notes.append("air-gapped: outdated-package check skipped")
            else:
                outdated, err = await asyncio.to_thread(_pip_outdated, venv)
                if err:
                    notes.append(f"outdated check failed: {err}")
    else:
        notes.append("no project venv found; conflict and outdated checks skipped")

    unpinned = [{"name": c.name, "ecosystem": c.ecosystem, "source": c.source}
                for c in comps if not c.pinned][:100]

    vulns: list[dict] = []
    try:
        db = sbom.load_vulndb(_config) if _config is not None else {}
        if db:
            vulns = sbom.match_vulns(comps, db)
        else:
            notes.append("no offline vuln DB configured; vulnerability match skipped")
    except Exception as e:
        notes.append(f"vuln match failed: {e}")

    return {
        "components": by_eco,
        "unused": unused,
        "conflicts": conflicts,
        "outdated": outdated,
        "unpinned": unpinned,
        "vulnerabilities": vulns,
        "notes": notes,
        "ok": not (unused or conflicts or vulns),
    }

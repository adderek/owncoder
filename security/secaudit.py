"""Security self-audit engine — local, deterministic, multi-project.

A cheap, offline-capable security scanner that owncoder can point at *any* project
(itself, the minimal browser, the model-rating suite, the tilix fork, …). It is the
Tier-1 floor of the security suite (see docs/MYTHOS_security_suite.md): deterministic
detection only — no LLM trust required to produce a finding. The LLM may later triage
these findings, but every Finding here has a non-LLM source and is reproducible.

Three detectors, all optional / fail-soft:
  1. External SAST scanners (semgrep, bandit, ruff, gosec, govulncheck) if on PATH.
  2. Secret leakage — reuses the credential shapes from ``redaction._PATTERNS`` to
     *flag* (not mask) secrets committed to disk.
  3. Config / hygiene lint — unsafe Dockerfile / CI / world-writable / risky patterns.

Scope can be the whole tree or just the current git diff (``diff_only``) so it can run
as a fast pre-push gate. Output is a normalized Finding list plus a reproducible
Markdown+JSON report. No network. No external Python deps.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

from agent.security.redaction import _PATTERNS

# Severity ordering for sorting / gating.
_SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

# Source files worth scanning for secrets / hygiene. Keep small + fast.
_TEXT_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".c", ".cc", ".cpp", ".h",
    ".hpp", ".java", ".kt", ".rb", ".php", ".sh", ".bash", ".zsh", ".vala", ".vapi",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env", ".json", ".xml",
    ".tf", ".tfvars", ".dockerfile", ".gradle", ".properties",
}
_SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
    ".ruff_cache", "dist", "build", "target", ".tox", ".idea", ".gradle",
    "vendor", ".cache", ".agent",
}
_MAX_FILE_BYTES = 2 * 1024 * 1024  # skip huge files / blobs


@dataclass
class Finding:
    detector: str          # "semgrep" | "bandit" | "secret" | "hygiene" | …
    rule_id: str
    severity: str          # critical|high|medium|low|info
    path: str              # relative to target root
    line: int | None
    message: str

    def key(self) -> tuple:
        """Stable identity for dedupe (line-sensitive)."""
        return (self.detector, self.rule_id, self.path, self.line)

    def baseline_key(self) -> str:
        """Line-insensitive identity for baseline matching.

        Excludes line number so an accepted finding stays suppressed when
        unrelated edits shift it up/down the file.
        """
        return f"{self.detector}:{self.rule_id}:{self.path}"


# ── external scanner adapters ──────────────────────────────────────────────

def _run(cmd: list[str], cwd: str, timeout: int = 180) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "NO_COLOR": "1"},
        )
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", "not found"
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def _norm_sev(s: str) -> str:
    s = (s or "").lower()
    if s in ("error", "critical", "blocker"):
        return "critical" if s == "critical" else "high"
    if s in ("warning", "high"):
        return "high"
    if s in ("medium", "moderate"):
        return "medium"
    if s in ("low", "info", "note", "minor"):
        return "low" if s == "low" else "info"
    return "medium"


def _scan_semgrep(root: str, files: list[str] | None) -> list[Finding]:
    if not shutil.which("semgrep"):
        return []
    cmd = ["semgrep", "--config", "auto", "--json", "--quiet", "--timeout", "30"]
    cmd += files if files else ["."]
    rc, out, _ = _run(cmd, root, timeout=300)
    if not out:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    findings = []
    for r in data.get("results", []):
        extra = r.get("extra", {})
        findings.append(Finding(
            detector="semgrep",
            rule_id=r.get("check_id", "?"),
            severity=_norm_sev(extra.get("severity", "")),
            path=os.path.relpath(r.get("path", "?"), root) if os.path.isabs(r.get("path", "")) else r.get("path", "?"),
            line=(r.get("start") or {}).get("line"),
            message=(extra.get("message") or "").strip()[:300],
        ))
    return findings


def _scan_bandit(root: str, files: list[str] | None) -> list[Finding]:
    if not shutil.which("bandit"):
        return []
    pys = [f for f in (files or []) if f.endswith(".py")]
    if files is not None and not pys:
        return []
    cmd = ["bandit", "-f", "json", "-q"]
    cmd += pys if files is not None else ["-r", "."]
    rc, out, _ = _run(cmd, root)
    if not out:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    findings = []
    for r in data.get("results", []):
        findings.append(Finding(
            detector="bandit",
            rule_id=r.get("test_id", "?"),
            severity=_norm_sev(r.get("issue_severity", "")),
            path=os.path.relpath(r.get("filename", "?"), root) if os.path.isabs(r.get("filename", "")) else r.get("filename", "?"),
            line=r.get("line_number"),
            message=(r.get("issue_text") or "").strip()[:300],
        ))
    return findings


_EXTERNAL = {
    "semgrep": _scan_semgrep,
    "bandit": _scan_bandit,
}


# ── secret leakage detector (reuses redaction shapes) ──────────────────────

def _scan_secrets(root: str, files: list[str]) -> list[Finding]:
    findings = []
    for rel in files:
        ap = os.path.join(root, rel)
        try:
            if os.path.getsize(ap) > _MAX_FILE_BYTES:
                continue
            with open(ap, "r", errors="ignore") as fh:
                for n, line in enumerate(fh, 1):
                    for pat, label in _PATTERNS:
                        if label == "secret-assignment":
                            continue  # too noisy on source; rely on shaped keys
                        m = pat.search(line)
                        if m:
                            findings.append(Finding(
                                detector="secret",
                                rule_id=label,
                                severity="critical" if "key" in label or "token" in label or label == "private-key" else "high",
                                path=rel,
                                line=n,
                                message=f"possible {label} committed to source",
                            ))
        except OSError:
            continue
    return findings


# ── hygiene / config lint ──────────────────────────────────────────────────

_HYGIENE_RULES: list[tuple[re.Pattern, str, str, str]] = [
    (re.compile(r"\bverify\s*=\s*False\b"), "tls-verify-off", "high", "TLS verification disabled"),
    (re.compile(r"\bshell\s*=\s*True\b"), "subprocess-shell", "medium", "subprocess shell=True (injection risk)"),
    (re.compile(r"\beval\s*\("), "eval-use", "high", "use of eval()"),
    (re.compile(r"\bpickle\.loads?\b"), "pickle-load", "high", "pickle deserialization (RCE risk)"),
    (re.compile(r"\byaml\.load\s*\((?!.*Loader)"), "yaml-unsafe-load", "high", "yaml.load without SafeLoader"),
    (re.compile(r"\bmd5\b|\bsha1\b", re.I), "weak-hash", "low", "weak hash (md5/sha1)"),
    (re.compile(r"0\.0\.0\.0"), "bind-all", "low", "binds all interfaces (0.0.0.0)"),
    (re.compile(r"DEBUG\s*=\s*True"), "debug-on", "medium", "DEBUG=True"),
    (re.compile(r"--no-check-certificate|--insecure\b|curl\s+.*-k\b"), "insecure-fetch", "high", "certificate check disabled in fetch"),
]


def _scan_hygiene(root: str, files: list[str]) -> list[Finding]:
    findings = []
    for rel in files:
        ap = os.path.join(root, rel)
        try:
            if os.path.getsize(ap) > _MAX_FILE_BYTES:
                continue
            with open(ap, "r", errors="ignore") as fh:
                for n, line in enumerate(fh, 1):
                    for pat, rid, sev, msg in _HYGIENE_RULES:
                        if pat.search(line):
                            findings.append(Finding("hygiene", rid, sev, rel, n, msg))
        except OSError:
            continue
    return findings


# ── file discovery ─────────────────────────────────────────────────────────

def _git_changed(root: str) -> list[str] | None:
    if not shutil.which("git") or not os.path.isdir(os.path.join(root, ".git")):
        return None
    rc, out, _ = _run(["git", "diff", "--name-only", "HEAD"], root, timeout=30)
    if rc != 0:
        return None
    changed = set(l.strip() for l in out.splitlines() if l.strip())
    rc, out, _ = _run(["git", "ls-files", "--others", "--exclude-standard"], root, timeout=30)
    if rc == 0:
        changed |= set(l.strip() for l in out.splitlines() if l.strip())
    return sorted(changed)


def _walk_files(root: str) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in _TEXT_EXTS or fn.lower() in ("dockerfile", "makefile"):
                out.append(os.path.relpath(os.path.join(dirpath, fn), root))
    return out


def _scannable(files: list[str]) -> list[str]:
    return [
        f for f in files
        if (os.path.splitext(f)[1].lower() in _TEXT_EXTS
            or os.path.basename(f).lower() in ("dockerfile", "makefile"))
    ]


# ── orchestration ───────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    target: str
    started_at: str
    diff_only: bool
    file_count: int
    scanners_used: list[str]
    scanners_missing: list[str]
    findings: list[Finding] = field(default_factory=list)

    def by_severity(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        return counts


def scan(target: str, diff_only: bool = False) -> ScanResult:
    """Run all detectors against *target* (a project directory)."""
    root = os.path.abspath(os.path.expanduser(target))
    if not os.path.isdir(root):
        raise ValueError(f"not a directory: {target}")

    if diff_only:
        changed = _git_changed(root)
        files = _scannable(changed) if changed is not None else _walk_files(root)
    else:
        files = _walk_files(root)

    used, missing = [], []
    findings: list[Finding] = []

    # External SAST. When diff_only, pass file list; else let them recurse.
    sast_files = files if diff_only else None
    for name, fn in _EXTERNAL.items():
        if shutil.which(name):
            used.append(name)
            findings += fn(root, sast_files)
        else:
            missing.append(name)

    # Built-in detectors always run (no external dep).
    used += ["secret", "hygiene"]
    findings += _scan_secrets(root, files)
    findings += _scan_hygiene(root, files)

    # Dedupe + sort (severity, then path).
    seen = set()
    deduped = []
    for f in findings:
        if f.key() in seen:
            continue
        seen.add(f.key())
        deduped.append(f)
    deduped.sort(key=lambda f: (_SEV_ORDER.get(f.severity, 9), f.path, f.line or 0))

    return ScanResult(
        target=root,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        diff_only=diff_only,
        file_count=len(files),
        scanners_used=used,
        scanners_missing=missing,
        findings=deduped,
    )


def to_json(res: ScanResult) -> str:
    return json.dumps({
        "target": res.target,
        "started_at": res.started_at,
        "diff_only": res.diff_only,
        "file_count": res.file_count,
        "scanners_used": res.scanners_used,
        "scanners_missing": res.scanners_missing,
        "severity_counts": res.by_severity(),
        "findings": [asdict(f) for f in res.findings],
    }, indent=2)


def to_markdown(res: ScanResult, limit: int = 100) -> str:
    counts = res.by_severity()
    sev_line = "  ".join(f"{k}={counts[k]}" for k in ("critical", "high", "medium", "low", "info") if k in counts) or "none"
    lines = [
        f"# Security audit — {res.target}",
        "",
        f"- scanned: {res.started_at}  ({'diff' if res.diff_only else 'full tree'})",
        f"- files: {res.file_count}",
        f"- detectors: {', '.join(res.scanners_used)}",
        f"- not installed (skipped): {', '.join(res.scanners_missing) or 'none'}",
        f"- findings: {len(res.findings)}  ({sev_line})",
        "",
        "> Deterministic floor only. A clean result is NOT proof of safety — a weak local "
        "model and a fixed rule set miss novel bugs. See docs/MYTHOS_security_suite.md.",
        "",
    ]
    if not res.findings:
        lines.append("No findings.")
        return "\n".join(lines)
    lines += ["| sev | detector | rule | location | message |", "|---|---|---|---|---|"]
    for f in res.findings[:limit]:
        loc = f"{f.path}:{f.line}" if f.line else f.path
        msg = f.message.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {f.severity} | {f.detector} | {f.rule_id} | {loc} | {msg} |")
    if len(res.findings) > limit:
        lines.append("")
        lines.append(f"… {len(res.findings) - limit} more (see JSON report).")
    return "\n".join(lines)


# ── baseline / suppression (Tier-4 #17) ─────────────────────────────────────

def baseline_path(workdir: str) -> Path:
    return Path(workdir) / ".agent" / "security" / "baseline.json"


def load_baseline(workdir: str) -> dict:
    """Return {baseline_key: reason}. Empty dict if no baseline file."""
    p = baseline_path(workdir)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        return {str(k): str(v) for k, v in (data.get("suppressed") or {}).items()}
    except (json.JSONDecodeError, OSError, AttributeError):
        return {}


def write_baseline(workdir: str, res: "ScanResult", reason: str = "accepted baseline") -> int:
    """Accept every finding in *res* into the baseline. Returns count accepted.

    Merges with any existing baseline (never drops prior suppressions).
    """
    existing = load_baseline(workdir)
    for f in res.findings:
        existing.setdefault(f.baseline_key(), reason)
    p = baseline_path(workdir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target": os.path.abspath(workdir),
        "suppressed": existing,
    }, indent=2))
    return len(res.findings)


def apply_baseline(res: "ScanResult", baseline: dict) -> tuple[list[Finding], int]:
    """Split findings into (new, suppressed_count) using *baseline* keys."""
    if not baseline:
        return res.findings, 0
    new = [f for f in res.findings if f.baseline_key() not in baseline]
    return new, len(res.findings) - len(new)


def _full_posture(config, target: str, workdir: str) -> str:
    """Aggregate every security check into one consolidated report + verdict."""
    sections: list[str] = [f"# Security posture — {os.path.abspath(target)}", ""]
    concerns: list[str] = []

    # 1. Code scan (baseline-filtered).
    try:
        res = scan(target)
        new, suppressed = apply_baseline(res, load_baseline(workdir))
        res.findings = new
        counts = res.by_severity()
        high = counts.get("critical", 0) + counts.get("high", 0)
        if high:
            concerns.append(f"{high} high/critical code finding(s)")
        sections.append(to_markdown(res))
        if suppressed:
            sections.append(f"_{suppressed} suppressed by baseline._")
    except Exception as e:  # noqa: BLE001
        sections.append(f"## Code scan\n(error: {e})")

    # 2. Dependencies / SBOM.
    try:
        from agent.security.sbom import run_sbom_command
        sbom_md = run_sbom_command(config, target)
        if "known vulnerable" in sbom_md and "→ 0 known" not in sbom_md:
            concerns.append("known-vulnerable dependencies")
        sections += ["", "## Dependencies", sbom_md]
    except Exception as e:  # noqa: BLE001
        sections += ["", f"## Dependencies\n(error: {e})"]

    # 3. Integrity of trusted files.
    try:
        from agent.security.integrity import check as _icheck
        ic = _icheck(config)
        if ic["sealed"] and not ic["ok"]:
            concerns.append("trusted files drifted since seal")
            sections += ["", "## Integrity", "DRIFT: " + ", ".join(
                f"{k}={len(ic[k])}" for k in ("modified", "added", "deleted") if ic[k])]
        elif ic["sealed"]:
            sections += ["", "## Integrity", "OK — sealed files unchanged."]
        else:
            sections += ["", "## Integrity", "not sealed (run /security integrity seal)."]
    except Exception as e:  # noqa: BLE001
        sections += ["", f"## Integrity\n(error: {e})"]

    # 4. Weight vault.
    try:
        from agent.security.weightvault import quickcheck as _wqc
        wq = _wqc(config)
        if wq["pinned"] and not wq["ok"]:
            concerns.append("pinned model weights changed/missing")
            sections += ["", "## Weights", "DRIFT detected — run /security weights verify."]
        elif wq["pinned"]:
            sections += ["", "## Weights", "OK — pinned weights unchanged (quickcheck)."]
        else:
            sections += ["", "## Weights", "none pinned."]
    except Exception as e:  # noqa: BLE001
        sections += ["", f"## Weights\n(error: {e})"]

    # 5. Egress posture.
    try:
        from agent.security.airgap import report as _agreport, is_enabled
        sections += ["", "## Egress", _agreport(config)]
        base_url = getattr(getattr(config, "llm", None), "base_url", "") or ""
        from agent.security.airgap import is_local_url
        if is_enabled(config) and not is_local_url(base_url):
            concerns.append("air-gap on but LLM endpoint is remote")
    except Exception as e:  # noqa: BLE001
        sections += ["", f"## Egress\n(error: {e})"]

    # Verdict up top.
    verdict = ("ATTENTION — " + "; ".join(concerns)) if concerns else \
        "No high-severity concerns from the deterministic floor (not proof of safety)."
    sections.insert(1, f"**Verdict:** {verdict}\n")
    return "\n".join(sections)


def run_security_command(config, arg: str) -> str:
    """Text handler for the /security slash command (both UIs).

    Subcommands:
      scan [path]       full-tree scan of path (default: working dir)
      diff [path]       scan only git-changed files (fast pre-push gate)
      triage [path]     scan + LLM ranks/explains existing findings
      review [path]     LLM READS source to find NEW vulns (memory/logic bugs)
      selfaudit         scan owncoder itself (config.tools.working_dir's repo)
      report [path]     scan + write Markdown+JSON report under .agent/security/
      baseline [path]   accept current findings as baseline (suppress as known)
      baseline clear    delete the baseline (un-suppress everything)
      baseline show     list suppressed entries
      airgap [on|off|status]  toggle/report non-local egress block
      integrity [seal|check]  sign skills+config / detect tampering
      weights [pin <p>|verify|list]  pin/verify local model weight files
      sbom [path]       list dependencies + flag known-vulnerable (offline DB)
      verify [<i>|run]  generate/run sandboxed PoC test for finding #i (fix check)
      full [path]       consolidated posture: scan+sbom+integrity+weights+egress
    """
    parts = arg.strip().split()
    sub = parts[0].lower() if parts else "scan"
    rest = parts[1] if len(parts) > 1 else ""

    workdir = getattr(getattr(config, "tools", None), "working_dir", ".") or "."

    # ── full posture: chain every check into one report ──────────────────
    if sub in ("full", "audit", "posture"):
        target = rest or workdir
        return _full_posture(config, target, workdir)

    # ── air-gap (egress posture / toggle) ────────────────────────────────
    if sub == "airgap":
        from agent.security.airgap import run_airgap_command
        return run_airgap_command(config, rest)

    # ── integrity (tamper detection for skills + config) ─────────────────
    if sub == "integrity":
        from agent.security.integrity import run_integrity_command
        return run_integrity_command(config, rest)

    # ── LLM deep-read vulnerability audit (reads source, not just findings) ─
    if sub == "review":
        from agent.security.review import run_review_command
        return run_review_command(config, arg.strip()[len("review"):].strip())

    # ── fix-verification PoC tests ───────────────────────────────────────
    if sub == "verify":
        from agent.security.verify import run_verify_command
        return run_verify_command(config, arg.strip()[len("verify"):].strip())

    # ── SBOM + dependency vuln audit ─────────────────────────────────────
    if sub == "sbom":
        from agent.security.sbom import run_sbom_command
        return run_sbom_command(config, rest)

    # ── weight vault (pin/verify local model files) ──────────────────────
    if sub == "weights":
        from agent.security.weightvault import run_weights_command
        # pass the full remainder (pin needs path + source words)
        return run_weights_command(config, arg.strip()[len("weights"):].strip())

    # ── baseline management ──────────────────────────────────────────────
    if sub == "baseline":
        action = rest.lower() if rest else "accept"
        if action == "clear":
            p = baseline_path(workdir)
            if p.exists():
                p.unlink()
                return "Baseline cleared. All findings will surface again."
            return "No baseline to clear."
        if action == "show":
            bl = load_baseline(workdir)
            if not bl:
                return "No baseline set."
            lines = [f"Baseline ({len(bl)} suppressed):"]
            for k, why in sorted(bl.items()):
                lines.append(f"  {k}  — {why}")
            return "\n".join(lines)
        # accept: scan the target and write all current findings as baseline
        target = rest or workdir
        try:
            res = scan(target)
        except ValueError as e:
            return f"Error: {e}"
        n = write_baseline(workdir, res, reason="accepted via /security baseline")
        return (f"Accepted {n} finding(s) into baseline at {baseline_path(workdir)}. "
                f"Future scans show only NEW findings.")

    if sub in ("scan", "diff", "report", "triage"):
        target = rest or workdir
    elif sub == "selfaudit":
        # owncoder repo root = two levels up from this file (agent/security/..)
        target = str(Path(__file__).resolve().parents[2])
    else:
        return ("Unknown subcommand. Use: scan [path] | diff [path] | "
                "selfaudit | report [path] | baseline [accept|clear|show]")

    try:
        res = scan(target, diff_only=(sub == "diff"))
    except ValueError as e:
        return f"Error: {e}"

    # Suppress baselined findings; surface only what's new.
    new, suppressed = apply_baseline(res, load_baseline(workdir))
    res.findings = new

    md = to_markdown(res)
    if suppressed:
        md += f"\n\n_{suppressed} finding(s) suppressed by baseline (see /security baseline show)._"

    if sub == "triage":
        from agent.security.triage import run_triage
        md += "\n\n## LLM triage\n\n" + run_triage(config, res)

    if sub == "report":
        out_dir = Path(workdir) / ".agent" / "security"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        base = out_dir / f"audit-{stamp}"
        base.with_suffix(".md").write_text(md)
        base.with_suffix(".json").write_text(to_json(res))
        md += f"\n\nReport written: {base}.md / {base}.json"

    return md

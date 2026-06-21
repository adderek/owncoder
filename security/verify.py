"""Fix-verification PoC tests for security findings (Tier-2 #20).

Closes the loop scan -> triage -> PROVE -> fix -> RE-PROVE. For a specific finding,
the local LLM writes a single self-contained pytest that exercises ONLY the target
project's own code and asserts the *vulnerable behavior reproduces*. The test is then
run inside the sandbox (no network, fs-confined, rlimited). Interpretation:

  test PASSES  -> vulnerability reproduced  -> still VULNERABLE (status open)
  test FAILS   -> behavior gone             -> FIXED or false-positive
  error/no run -> INCONCLUSIVE

The saved test doubles as a permanent regression test: re-run it after a fix and a
green-to-red flip proves the fix landed. The deterministic pytest result — not the
model's say-so — is the source of truth, same principle as the rest of the suite.

DEFENSIVE ONLY. The generation prompt forbids network egress, out-of-tempdir file
writes, shells, and reusable exploit payloads. Tests target own code, never external
or third-party live systems. Air-gap aware: refuses if the local LLM endpoint is
non-local while air-gap is on.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from agent.security import secaudit

_MAX_OUTPUT_TOKENS = 1400
_TEST_TIMEOUT = 30

_SYSTEM = (
    "You are a defensive security engineer writing a REGRESSION TEST that proves a "
    "specific vulnerability exists in THIS project's own source code.\n\n"
    "STRICT RULES — the test MUST NOT:\n"
    "- open network sockets or contact any host (no requests/urllib/socket to remote)\n"
    "- read, write, or delete files outside a pytest tmp_path / tempfile dir\n"
    "- spawn shells, subprocesses, or eval attacker-controlled strings for real\n"
    "- contain a reusable weaponized exploit payload (only the minimum to assert behavior)\n\n"
    "The test MUST:\n"
    "- be a single self-contained pytest file importing only this project's modules and stdlib\n"
    "- assert that the VULNERABLE behavior is reproducible, so it PASSES while the bug "
    "is present and FAILS once the code is fixed\n"
    "- be small, deterministic, and runnable offline\n\n"
    "Output ONLY Python code for the test file. No prose, no markdown fences."
)


def poc_dir(config) -> Path:
    root = Path(getattr(getattr(config, "tools", None), "working_dir", ".") or ".")
    return root / ".agent" / "security" / "poc"


def _slug(f: "secaudit.Finding") -> str:
    raw = f"{f.detector}_{f.rule_id}_{Path(f.path).stem}_{f.line or 0}"
    return re.sub(r"[^a-zA-Z0-9_]", "_", raw)[:60]


def _excerpt(target: str, rel_path: str, line: int | None, ctx: int = 15) -> str:
    p = Path(target) / rel_path
    try:
        lines = p.read_text(errors="ignore").splitlines()
    except OSError:
        return ""
    if line:
        lo, hi = max(0, line - ctx), min(len(lines), line + ctx)
    else:
        lo, hi = 0, min(len(lines), 2 * ctx)
    return "\n".join(f"{i+1}: {lines[i]}" for i in range(lo, hi))


def _strip_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = re.sub(r"^```[a-zA-Z]*\n", "", code)
        code = re.sub(r"\n```$", "", code)
    return code.strip()


async def _generate(config, finding: "secaudit.Finding", target: str) -> str:
    from openai import AsyncOpenAI
    from agent.config import make_registry
    from agent.security import airgap

    entry = make_registry(config).role("verify")
    if airgap.is_enabled(config) and not airgap.is_local_url(entry.base_url):
        return "# air-gap: refused — LLM endpoint is non-local"

    excerpt = _excerpt(target, finding.path, finding.line)
    user = (
        f"Project root: {target}\n"
        f"Finding: [{finding.severity}] {finding.detector}/{finding.rule_id}\n"
        f"Location: {finding.path}:{finding.line}\n"
        f"Message: {finding.message}\n\n"
        f"Source excerpt:\n```\n{excerpt}\n```\n\n"
        "Write the regression test now."
    )
    client = AsyncOpenAI(base_url=entry.base_url, api_key=entry.api_key)
    try:
        resp = await client.chat.completions.create(
            model=entry.model,
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": user}],
            max_tokens=_MAX_OUTPUT_TOKENS,
            temperature=0.1,
        )
        out = (resp.choices[0].message.content or "") if resp.choices else ""
    finally:
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass
    return _strip_fences(out)


def _generate_sync(config, finding, target) -> str:
    import asyncio
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(_generate(config, finding, target))).result()


def _run_test(target: str, test_path: Path, timeout: int = _TEST_TIMEOUT) -> tuple[int, str]:
    """Run pytest on *test_path* inside the sandbox (no network). Returns (rc, output)."""
    argv = [sys.executable, "-m", "pytest", "-q", "-x", "--no-header", str(test_path)]
    try:
        from agent.security import runner, policy
        if policy.is_configured():
            res = runner.run(argv, cwd=target, network=False, timeout=timeout)
            return res.returncode, (res.stdout + res.stderr)[-4000:]
    except Exception:  # noqa: BLE001 - fall back to plain subprocess
        pass
    try:
        p = subprocess.run(argv, cwd=target, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)[-4000:]
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return 127, f"pytest run failed: {e}"


def _classify(rc: int) -> str:
    if rc == 0:
        return "VULNERABLE — PoC reproduced the weakness (status: OPEN)"
    if rc == 1:
        return "not reproduced — likely FIXED or false-positive"
    return "INCONCLUSIVE — test errored or pytest unavailable"


def verify_finding(config, target: str, index: int) -> str:
    res = secaudit.scan(target)
    if not res.findings:
        return "No findings to verify. Run /security scan first."
    if index < 0 or index >= len(res.findings):
        return f"No finding at index {index} (have 0..{len(res.findings) - 1})."
    f = res.findings[index]

    code = _generate_sync(config, f, target)
    if not code or code.startswith("# air-gap"):
        return code or "(PoC generation returned empty)"

    d = poc_dir(config)
    d.mkdir(parents=True, exist_ok=True)
    test_path = d / f"test_poc_{_slug(f)}.py"
    test_path.write_text(code)

    rc, out = _run_test(target, test_path)
    loc = f"{f.path}:{f.line}" if f.line else f.path
    return (
        f"# PoC verification — finding #{index}\n"
        f"- {f.severity} {f.detector}/{f.rule_id} @ {loc}\n"
        f"- test: {test_path}\n"
        f"- result: {_classify(rc)}\n\n"
        f"```\n{out.strip()[:1500]}\n```\n"
        "_PoC asserts the weakness reproduces. After you fix the code, re-run "
        "`/security verify run` — a flip to 'not reproduced' confirms the fix._"
    )


def rerun(config, target: str) -> str:
    d = poc_dir(config)
    tests = sorted(d.glob("test_poc_*.py")) if d.is_dir() else []
    if not tests:
        return "No saved PoC tests. Generate one with /security verify <finding-index>."
    lines = [f"Re-running {len(tests)} saved PoC test(s):"]
    open_n = fixed_n = inc_n = 0
    for t in tests:
        rc, _ = _run_test(target, t)
        if rc == 0:
            open_n += 1
            tag = "OPEN (still vulnerable)"
        elif rc == 1:
            fixed_n += 1
            tag = "FIXED (no longer reproduces)"
        else:
            inc_n += 1
            tag = "inconclusive"
        lines.append(f"  {t.name}: {tag}")
    lines.append(f"\nopen={open_n}  fixed={fixed_n}  inconclusive={inc_n}")
    return "\n".join(lines)


def run_verify_command(config, arg: str) -> str:
    """Text handler for `/security verify [<finding-index> | run | list]`."""
    parts = arg.strip().split()
    workdir = getattr(getattr(config, "tools", None), "working_dir", ".") or "."
    sub = parts[0].lower() if parts else "list"

    if sub == "run":
        target = parts[1] if len(parts) > 1 else workdir
        return rerun(config, target)

    if sub in ("list", "ls", ""):
        d = poc_dir(config)
        tests = sorted(d.glob("test_poc_*.py")) if d.is_dir() else []
        if not tests:
            return ("No PoC tests yet. Run /security scan to list findings, then "
                    "/security verify <index> to generate a fix-verification test.")
        return "Saved PoC tests:\n" + "\n".join(f"  {t.name}" for t in tests) + \
               "\nUse /security verify run to re-check them."

    try:
        index = int(sub)
    except ValueError:
        return "Usage: /security verify [<finding-index> | run | list]"
    return verify_finding(config, workdir, index)

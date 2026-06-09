# Security Improvements Plan

Findings from security sweep of `agent/` (2026-06-09). Sandbox core is solid
(bwrap/firejail + seccomp BPF + rlimits + env scrub + fs gate). Gaps below are
tools/paths that route **around** the harness. Ordered by priority.

Each item: where, what, why, fix, how to verify. Independent — agents can pick
one without touching others.

---

## P0 — SSRF via HTTP redirect (no re-validation)

**Files:** `agent/tools/web_search/http_executor.py` (`_FETCHER_SCRIPT`),
`agent/security/query_gate.py` (`gate_fetch`, `_validate_url`).

**Problem:**
- `gate_fetch()` validates only the *initial* URL (blocks private/link-local/
  loopback/IPv4-mapped IPs, DNS-rebind check).
- The in-sandbox fetcher uses `opener.open()`, which **auto-follows HTTP
  redirects with no IP re-validation**. The manual redirect logic
  (`redirect_count`, `current_url`, `max_redirects`) is **dead code** — never
  used; urllib follows redirects internally and unbounded.
- web_fetch runs `network=True` → bwrap launched **without** `--unshare-net` →
  child shares host network.
- Attack: fetch `http://attacker/r` → 302 → `http://169.254.169.254/...`
  (cloud metadata) or `http://10.x/internal`. Initial gate passes (attacker
  host public); redirect target never checked.

**Fix:**
- In `_FETCHER_SCRIPT`, install a custom `urllib.request.HTTPRedirectHandler`
  subclass whose `redirect_request()` re-validates each hop's host/IP against
  the same blocklist (`_BLOCKED_NETWORKS` logic) before allowing the redirect,
  and enforces `max_redirects`. Or disable auto-redirect and handle the loop
  manually (re-resolve + re-check each `Location`).
- Blocklist must live in the fetcher (runs in sandbox, no import of query_gate).
  Inline the IP-range check, or pass the blocklist in via the stdin JSON request.

**Verify:**
- Unit test: mock server returning 302 → `http://127.0.0.1/` ; assert fetch
  rejected (not followed).
- Test redirect to `http://169.254.169.254/` rejected.
- Test legitimate redirect (https→https, public host) still works.
- Confirm `max_redirects` actually caps hop count.

---

## P0 — DNS rebind TOCTOU between gate and fetch

**Files:** `agent/security/query_gate.py` (`_validate_url`),
`agent/tools/web_search/http_executor.py`.

**Problem:** Gate resolves DNS in the parent process and validates IPs; the
sandboxed fetcher then re-resolves the hostname **independently**. Attacker DNS
returns a public IP on the first lookup, a private IP on the second → gate
bypassed. Same root cause as redirect SSRF: the IP that gets connected to is
never the one that was validated.

**Fix:** Resolve once in the gate, pin the validated IP, and have the fetcher
connect to *that IP* (Host header preserved for vhost/SNI). Cleanest combined
fix with P0-redirect: do all validation + connection inside one place that
controls the actual socket target. Consider a custom `HTTPConnection` that
overrides `connect()` to use the pre-validated IP.

**Verify:** Test with a resolver/hosts stub that returns different IPs on
successive lookups; assert the fetch connects to the validated IP only.

---

## P1 — `grep_code` escapes project root, bypasses sandbox

**File:** `agent/tools/search/grep.py` (`grep_code`).

**Problem:**
```python
search_root = Path(path) if path else Path(working_dir)
if not search_root.is_absolute():
    search_root = Path(working_dir) / search_root
search_root = search_root.resolve()      # NO confinement check
...
proc = subprocess.run(cmd, ...)          # host grep, NOT _runner.run
```
LLM-controlled `path` resolved with no `_within(root)` check; grep runs
**directly on host**, not via sandbox. `grep_code(pattern=..., path="/etc")` or
`path="~/.ssh"` reads arbitrary files outside the project — filenames + matching
line content leak into context. Contrast `read_file`, which correctly routes
through `paths._resolve()` / `security.fs.safe_resolve()`.

**Fix:** Gate `search_root` through `security.fs.safe_resolve()` (or reuse
`tools/files/paths._resolve`) before running. Reject paths outside working_dir
with a clear error. Optionally route the grep through `_runner.run` so it's
sandboxed like everything else.

**Verify:**
- Test `grep_code(path="/etc")` returns escape error.
- Test `grep_code(path="../../outside")` rejected.
- Test in-root search still works.

---

## P2 — `run_argv` skips destructive-command tripwire

**File:** `agent/tools/shell/main.py` (`run_argv` vs `run_command`).

**Problem:** `_check_dangerous` (confirm gate for rm/dd/mkfs/fdisk/shutdown/
sudo/doas) runs only in `run_command`, not `run_argv`. `run_argv(["rm","-rf",
"x"])` skips confirmation. Sandbox bind-mounts confine writes to project root,
so impact is bounded **when a sandbox backend exists** — but the confirm UX is
inconsistent, and on host-exec fallback (see P2-posture) there's no confinement.

**Fix:** Call `_check_dangerous(shlex.join(argv))` (or check `argv[0]` basename
against `_CONFIRM_COMMANDS`) at the top of `run_argv`, returning the same
`requires_confirm` dict.

**Verify:** Test `run_argv(["rm","-rf","foo"])` returns `requires_confirm=True`.

---

## P2 — Default posture is silent host-exec

**Files:** `agent/config/models.py` (`SecurityConfig`),
`agent/security/runner.py` (`select_backend`).

**Problem:** Defaults `require_sandbox=False`, `sandbox_backend="auto"`. If
neither bwrap nor firejail is installed/functional, `select_backend()` logs
`"running commands on host!"` and proceeds — backend `"none"` runs argv with
**no FS confinement** (only env scrub + rlimits). Combined with P1/P0, host-exec
means full FS read. Degradation is only a log line; users may not see it.

**Fix (pick per risk appetite):**
- Default `require_sandbox=True` (fail-closed; needs bwrap/firejail installed), OR
- Surface the degradation loudly: print a visible warning to the UI/console on
  first host-exec, not just `logger.warning`. Make it a one-time prominent
  banner.
- Document the tradeoff in CLAUDE.md / config docs.

**Verify:** With no sandbox backend + `require_sandbox=True`, assert shell tools
return a clear "sandbox required" error instead of running.

---

## P3 — seccomp filter only applied on bwrap path

**Files:** `agent/security/runner.py` (`_firejail_argv`, `_bwrap_argv`),
`agent/security/seccomp_filter.py`.

**Problem:** Curated `_BLOCKED_SYSCALLS` (unshare/setns/mount/ptrace/bpf/keyctl/
userfaultfd/...) applied only via `--add-seccomp-fd` on bwrap. firejail relies
on its generic `--seccomp` default, which doesn't match the curated list.
Asymmetric hardening.

**Fix:** Either generate a firejail-compatible seccomp filter
(`--seccomp.block-secondary` / custom filter file) mirroring the list, or
document the asymmetry and prefer bwrap. At minimum add a code comment so the
gap is intentional/visible.

**Verify:** Under firejail backend, attempt a blocked syscall (e.g. `unshare`)
from inside the sandbox; document actual behavior.

---

## Verified clean (no action needed — context for reviewers)

- fs gate (`security/fs.py`): O_NOFOLLOW + literal-component symlink walk +
  pinned (dev,ino) root identity. Solid vs symlink/TOCTOU root swaps. 0o600 writes.
- env scrub (`security/policy.py`): deny `*_TOKEN/_KEY/_SECRET`, cloud prefixes,
  DB URLs; HOME repointed to project root.
- audit log (`security/audit.py`): hashes stdout/stderr, never stores raw —
  no secret leak to disk.
- git tool (`tools/git/main.py`): paths after `--`, no flag injection;
  read-only subcommands only.
- injection_shield: structural `<web_result>` wrapping + Human:/Assistant:
  escaping + zero-width strip.
- query_gate: blocks `file://`/`gopher://`/`data:` schemes, IPv4-mapped-IPv6
  (`::ffff:`), secret patterns in queries. (Gaps are redirect/rebind above, not
  the initial-URL checks.)

---

## Suggested order

1. P0 redirect SSRF + P0 DNS rebind together (same code area, same fix surface).
2. P1 grep_code confinement (small, isolated).
3. P2 run_argv tripwire (small) + P2 posture decision (config/UX).
4. P3 firejail seccomp parity (or document).

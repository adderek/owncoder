# Security Hardening Plan — Round 2 (evil-AI threat model)

Follow-up to `security-improvements-plan.md` (P0–P3, now landed). Sandbox core
is solid: bwrap/firejail + seccomp BPF + rlimits + env allowlist + fs gate +
SSRF/redirect/rebind guards. This round assumes the **AI running inside the
harness is hostile**: it will try to escape the sandbox, persist across
sessions, weaken its own config, and exfiltrate anything it can read.

Each item: **where, what, why, fix, verify.** Items are independent — pick one
without touching the others. Ordered by blast radius (escape/persistence first,
then DoS, then leak).

Conventions used elsewhere in this repo:
- All LLM-initiated command execution must route through
  `agent/security/runner.py:run()`. Never add a raw `subprocess.*` in a tool.
- All LLM-initiated file I/O must route through
  `agent/security/fs.py` (`safe_resolve`/`safe_open`) or
  `agent/tools/files/paths.py` (`_resolve`).
- Defaults live in `agent/config/models.py` (`SecurityConfig`).
- Tests go under `agent/tests/unit/`, run with `agent/.venv/bin/pytest`.

---

## H1 — Project root is bind-mounted read-write → host escape + self-config rewrite

**Severity: critical (sandbox escape + persistence).**

**Files:**
- `agent/security/runner.py` (`_bwrap_argv`, `_firejail_argv`) — root mounted rw.
- `agent/security/fs.py` (`safe_open`) — no write-denylist.
- `agent/tools/rules/` (`check_write`) — existing write-rule hook to extend.

**Problem:**
`_bwrap_argv` does `--bind str(root) str(root)` — the whole project root is
writable inside the sandbox. A hostile agent can write files that get executed
**on the host, outside the sandbox**, or that weaken the next session:

1. **Git hook escape.** Write `.git/hooks/pre-commit` (or `post-checkout`,
   `post-merge`). Next time the user runs `git commit` on the host (normal dev
   action, no sandbox), the hook runs with the user's full privileges. Full
   escape.
2. **Self-config rewrite.** Edit `agent.toml`, `.agent/`, `CLAUDE.md`/`AGENT.md`,
   `.claude/settings*.json`, `.agent.*` override files. Next session loads
   `require_sandbox=false`, `network='on'`, emptied `env_deny_patterns`, or a
   poisoned system prompt. Persistence + privilege gain across runs.

The fs gate confines *location* (can't escape root) but does **not** restrict
*which files inside root* may be written. And the gate is bypassable anyway via
a raw shell argv (`run_argv(['sh','-c','printf ... > .git/hooks/pre-commit'])`),
so a gate-only fix is insufficient.

**Fix (defense in depth — do both layers):**

1. **Mount sensitive paths read-only in the sandbox** (the real control, since
   it survives shell bypass). In `_bwrap_argv`, after the `--bind root root`,
   append `--ro-bind` (or `--ro-bind-try`) overlays for each protected path that
   exists:
   - `<root>/.git`
   - `<root>/agent.toml`, `<root>/.agent.toml`
   - `<root>/.agent` (state dir — but the agent legitimately writes sessions/
     index here; see note below)
   - `<root>/CLAUDE.md`, `<root>/AGENT.md`
   - `<root>/.claude`
   - `<root>/.agent.ignore`, `<root>/.agent.priorities.toml`, other `.agent.*`
   - Mirror the same `--read-only`/`--blacklist` for `_firejail_argv`.

   **Note on `.agent/`:** the agent writes sessions, logs, audit, plans there
   during normal operation. Do **not** blanket-ro it. Protect only config-like
   children (e.g. any `*.toml`, plan-policy files), or move agent state outside
   the bind-root entirely (preferred long-term: state dir lives outside project
   root, only sources are bound). For this task: ro-bind `.git`, top-level
   config files, and `.claude`; leave `.agent/` writable but add it to the gate
   denylist for config files (layer 2).

2. **Write-denylist in the fs gate** (catches the non-shell tool path early
   with a clear error). In `safe_open`, when mode is a write mode, reject paths
   whose root-relative form matches a protected glob. Reuse/extend
   `agent/tools/rules` `check_write` so the list is configurable. Default
   denied-write globs:
   ```
   .git/**, agent.toml, .agent.toml, .agent.*, CLAUDE.md, AGENT.md,
   .claude/**, .agent/**/*.toml
   ```
   Return a `PathEscape`-style error: "write to protected path denied: <p>".

**Verify:**
- Test: `safe_open(".git/hooks/pre-commit", "w")` raises / returns denied.
- Test: `safe_open("agent.toml", "w")` denied; `safe_open("CLAUDE.md", "w")` denied.
- Test: writing a normal source file (`src/foo.py`) still works.
- Integration (bwrap available): inside the sandbox,
  `run_argv(['sh','-c','echo x > .git/hooks/pre-commit'])` fails with EROFS /
  permission denied (proves ro-bind, not just gate).
- Integration: `run_argv(['sh','-c','echo x > agent.toml'])` fails.

---

## H2 — `patch_file` bypasses the sandbox entirely

**Severity: medium (unsandboxed host exec + DoS).**

**File:** `agent/tools/files/patch.py` (`_apply_unified_diff`).

**Problem:** Lines 22 and 26 call host `patch` via raw `subprocess.run` — not
routed through `security.runner.run`. So it runs:
- outside bwrap/firejail (no FS confinement, no seccomp),
- with **no timeout** → a crafted patch / pathological input can hang the agent
  process indefinitely (DoS),
- with no rlimits.

Inputs are temp files the tool controls, so it's not a direct arbitrary-write,
but it is an unsandboxed host binary invocation that violates the
"all execution goes through runner" invariant, and the missing timeout is a real
hang vector.

**Fix (pick one, prefer A):**
- **A.** Add `timeout=` to both `subprocess.run` calls (e.g. 15s) and wrap in
  `try/except subprocess.TimeoutExpired` returning a clean error. Smallest fix,
  closes the DoS.
- **B.** Stop shelling out: apply the unified diff in pure Python (the codebase
  already prefers `edit_file`; consider a Python diff-apply so no `patch` binary
  is needed at all). Removes the host-exec entirely.

**Verify:**
- Test: a patch that would hang returns a timeout error within the limit, not a
  hang.
- Test: a normal valid patch still applies.
- Grep check: `grep -rn 'subprocess' agent/tools/files/patch.py` shows either no
  raw subprocess (option B) or every call carries a `timeout=` (option A).

---

## H3 — No egress confinement when network is enabled

**Severity: high (exfil + internal-service access).**

**Files:** `agent/security/runner.py` (`_bwrap_argv` drops `--unshare-net` when
`network=True`), `agent/tools/shell/main.py` (`run_argv(network=...)`),
`agent/tools/web_search/`.

**Problem:** When network is allowed, bwrap omits `--unshare-net` → the child
gets the **host's full network namespace**. It can then reach:
- the local LLM endpoint (`http://localhost:8080`),
- any LAN / RFC1918 host,
- cloud metadata (`169.254.169.254`),
- the loopback services of the dev machine.

The SSRF/redirect/rebind guards (`security/query_gate.py`) only protect the
**HTTP fetch tool path**. A raw `run_argv(['curl', ...], network=True)` (when
`security.network='on'`) or any socket-capable binary bypasses all of that.
Network-on currently means "trusted with the entire host network."

**Fix:** Never hand over the raw host network. Options, strongest first:
- **A.** Keep `--unshare-net` always; provide outbound via slirp4netns or a
  small filtering proxy bound into the namespace, with an **egress allowlist**
  (only the configured LLM host + an optional user allowlist). Block loopback,
  RFC1918, link-local, metadata IPs at the proxy.
- **B.** If A is too heavy: keep a network namespace but install firewall rules
  (nftables in the netns) that drop loopback + RFC1918 + 169.254.0.0/16 +
  ::1/fc00::/fe80:: before exec.
- **C.** Minimum stopgap: document that `security.network='on'` exposes the full
  host network and gate it behind an explicit, loud opt-in (it already requires
  config, but the docstring should state the exposure).

**Verify:**
- Test (network on): from inside the sandbox, a connect to `127.0.0.1:8080` and
  to an RFC1918 address is refused; a connect to an allowlisted public host
  works.
- Test: `169.254.169.254` unreachable from inside the sandbox.

---

## H4 — Fork bomb inside bwrap (no pid cap)

**Severity: medium (local DoS).**

**File:** `agent/security/runner.py` (`_rlimit_preexec`, `_bwrap_argv`).

**Problem:** `RLIMIT_NPROC` is deliberately **skipped on the bwrap path**
(`runner.py:150-157`) because it would break `unshare(CLONE_NEWPID)` when the
user already has many processes. `--unshare-pid` gives a fresh pid view but no
upper bound on process count → a `:(){ :|:& };:` fork bomb inside the sandbox can
exhaust the host's pid/task space.

**Fix:** Cap tasks via cgroup v2 instead of rlimit:
- Launch the sandbox under `systemd-run --user --scope -p TasksMax=<nproc>` (when
  available), or
- create a transient cgroup, write `pids.max`, and put the bwrap child in it
  (`--unshare-cgroup` already present), or
- as a fallback, apply `RLIMIT_NPROC` only when the host process count is well
  under the limit (best-effort).
Make it best-effort + logged, like the other rlimits (never hard-fail the run).

**Verify:**
- Test (bwrap): a fork-bomb command inside the sandbox is capped (hits TasksMax /
  pids.max) and the parent reaps it within `wall_seconds`, host stays responsive.

---

## H5 — Project secrets are readable inside root → exfil pairing

**Severity: medium (leak, amplifies H3).**

**Files:** `agent/security/fs.py` (read path), `agent/tools/files/`,
`agent/tools/search/grep.py`.

**Problem:** Env vars are scrubbed (`policy.env_for_child`), but the project
tree itself often holds secrets: `.env`, `*.pem`, `id_rsa`/`id_ed25519`,
`*.key`, `.netrc`, `.aws/credentials`, committed tokens. The fs gate happily
serves these to the read tool and `grep_code`. Pair with **any** egress
(see H3) and the agent exfiltrates them. Confinement-to-root does not protect
secrets that live in root.

**Fix:**
- Add a **read-deny / redact list** in the fs gate (and grep output filter):
  default globs `.env*`, `*.pem`, `*.key`, `id_rsa*`, `id_ed25519*`, `*.p12`,
  `.netrc`, `**/.aws/credentials`, `**/.ssh/*`. Reading these returns a redacted
  placeholder or a clear "secret file read blocked" error.
- Add a lightweight **secret-pattern scrub** on tool output (high-entropy /
  known token prefixes like `AKIA`, `ghp_`, `sk-`, `-----BEGIN ... KEY-----`)
  before it enters context. Conservative — only redact high-confidence hits.
- Make both lists config-driven so legit projects can opt out per-path.

**Verify:**
- Test: `read_file(".env")` returns blocked/redacted, not contents.
- Test: `grep_code('SECRET', path='.')` does not echo `.env` line contents.
- Test: a normal source file with no secrets reads through unchanged.

---

## H6 — Default posture is fail-open (`require_sandbox=False`)

**Severity: medium (silent loss of all FS confinement).**

**Files:** `agent/config/models.py` (`SecurityConfig.require_sandbox`),
`agent/security/runner.py` (`select_backend`).

**Problem:** Default `require_sandbox=False`, `sandbox_backend='auto'`. If no
backend is installed/functional, `select_backend()` logs a warning and runs with
backend `'none'` — argv on the host with **no FS confinement** (only env scrub +
rlimits). For a hostile-AI model this is fail-open: combined with H1/H5 it means
full host read/write. The degradation is only a log line + one-time banner.

**Fix:** Flip the default to **fail-closed** for the evil-AI threat model:
- `require_sandbox=True` by default. With no working backend, shell/file-exec
  tools return a clear "sandbox required — install bubblewrap or firejail, or set
  security.require_sandbox=false to allow unsandboxed host exec" error instead of
  running.
- Keep the loud one-time banner only for users who explicitly opt out
  (`require_sandbox=false`).
- Document the tradeoff in `CLAUDE.md` (Context reliability / security section)
  and the config docs.

**Verify:**
- Test: `require_sandbox=True` + no backend → `runner.run([...])` raises
  `SandboxUnavailable` / shell tool returns the sandbox-required error, does NOT
  execute.
- Test: `require_sandbox=False` + no backend → runs on host with the banner
  (existing behavior preserved for opt-out).

---

## H7 — seccomp asymmetry: firejail lacks the curated blocklist (P3 carryover)

**Severity: low (documented gap, weaker firejail path).**

**Files:** `agent/security/runner.py` (`_firejail_argv`),
`agent/security/seccomp_filter.py`.

**Problem:** Curated `_BLOCKED_SYSCALLS` (unshare/setns/mount/ptrace/bpf/keyctl/
userfaultfd/…) is applied **only on the bwrap path** via `--add-seccomp-fd`.
firejail relies on its generic `--seccomp` default, which is broader but
different and not controlled by our list. Asymmetric, already documented in the
seccomp_filter.py docstring.

**Fix (pick one):**
- Generate a firejail-compatible filter (`--seccomp=<comma-list>` or a custom
  filter file) mirroring `_BLOCKED_SYSCALLS`, OR
- Demote firejail to a last-resort backend (prefer bwrap; warn when firejail is
  selected that seccomp coverage differs), OR
- At minimum keep the existing docstring note and add a startup log line when the
  firejail backend is chosen.

**Verify:**
- Under firejail backend, attempt a blocked syscall (e.g. `unshare -U`) from
  inside the sandbox; document the actual behavior in a test/skip-marker.

---

## Suggested order

1. **H1** (root ro-bind + write-denylist) — biggest blast radius: turns "escape"
   into "confined." Do first.
2. **H6** (fail-closed default) — small config change, removes the silent
   bypass that amplifies everything else.
3. **H3** (egress confinement) — high value, more work; can land after H1/H6.
4. **H5** (secret read guard) — pairs with H3; do together if possible.
5. **H2** (patch timeout) and **H4** (pid cap) — small, isolated DoS fixes.
6. **H7** (firejail seccomp parity) — lowest priority; document if not fixed.

## Ground rules for the implementing agent

- One item per change/commit. Keep diffs surgical.
- Never weaken an existing control to make a test pass.
- Add a unit test for every item (see each Verify block).
- Run `agent/.venv/bin/pytest` before and after; don't break existing
  `tests/unit/test_security.py`, `test_http_executor_ssrf.py`.
- Update `FEATURES_LIST.md` only if a config key or user-visible behavior
  changes (H6 changes the default; document it).
- If a fix needs a new `SecurityConfig` field, give it a safe default and
  mention it in `agent.toml.example`.

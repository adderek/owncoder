# Hook trust-boundary spec (S4)

Status: **decided** — implementation steps at the end are PLAN_NORMAL-class.

Scope: `core/hooks.py` (pre_tool / post_tool shell hooks) and the trust level
of the config layers that can define them. Written against the current code,
which — unlike the assumption in PLAN_STRONG — already ships *active* hooks:
a `block=true` pre_tool hook denies the call on non-zero exit, and post_tool
output surfaces as a transient note.

## Finding 0 (live vulnerability, fix first)

`cli/main.py` loads `<project_root>/agent.toml` as a config layer, and the
`hooks` section is merged from every layer. Therefore **cloning a hostile
repo that contains an `agent.toml` with `[[hooks.entries]]` and running the
agent inside it executes attacker-authored shell on the first tool call** —
no sandbox (hooks run un-sandboxed in the project dir, with the user's full
environment), no prompt, no injection needed.

The same merge path lets a project config weaken `[security]`
(`write_deny_globs = []`, `require_sandbox = false`, `redact_tool_output =
false`, …). That is broader than this spec (tracked as its own follow-up),
but the hook fix below establishes the mechanism the broader fix will reuse.

## Decisions

### D1 — May pre_tool block a call? **Yes** (status quo, kept)

Non-zero exit from a `block=true` pre_tool hook denies the call and returns
the hook's output to the model as the error. This mirrors Claude Code and is
already shipped; nothing changes.

**Constraint**: hooks must never gate security-suite internal operations
(secaudit, verify, evolve, injection_scan). Today those runners invoke
subprocess/fs directly rather than going through `execute_tool()`, so the
property holds structurally. Lock it in with a test: a `tools = ["*"]`,
always-failing, `block=true` hook must not prevent `secaudit` from
completing its scan. If a future refactor routes suite operations through
the tool layer, they must set the same internal-bypass flag the permission
model uses (`_internal_security=True`, see docs/permissions-design.md) and
hooks must honor it.

### D2 — May hook stdout inject into the conversation? **Yes, marked, never as a role**

- pre_tool block output: returned as the *tool error* for the denied call —
  already clearly attributed ("blocked by hook X"), model-visible by design
  so it can re-plan. Keep, but always prefix with the hook's identity:
  `[hook pre_tool:<first 40 chars of command>] <output>`.
- post_tool output (non-zero exit): surfaces as a transient note — the
  existing system-reminder-style channel. Keep. It must never be injected as
  a `user` or `assistant` message, and the note must carry the same
  `[hook post_tool:…]` attribution prefix so the model can weigh it as
  environment feedback, not instruction.
- Hook output is *data crossing into the context*, so it goes through the
  same redaction pass as tool output (`redact_tool_output`) before
  injection. A hook that cats a secret file must not paste the secret into
  the conversation.

### D3 — Config trust levels: **project-layer hooks require one-time approval by content hash**

- Hooks from the user layers (`~/.config/agent/agent*.toml`) are trusted:
  the user wrote them.
- Hooks from the **project layer** (`<project_root>/agent.toml` or any path
  passed as `extra_path`) are untrusted input — attacker-controlled in any
  cloned repo.
- Rule: at config load, each project-layer hook entry is fingerprinted:
  `sha256(event + "\0" + command + "\0" + ",".join(tools) + "\0" + str(block))`.
  A project hook only becomes active if its fingerprint is present in the
  user-level approval store `~/.config/agent/approved_hooks.json`
  (**user-level on purpose** — an approval store inside the repo would be
  attacker-writable, defeating the point).
- Unapproved project hooks: **not executed**, surfaced once per session as a
  warning listing event/tools/command, with the exact `/hooks approve`
  command to run. Approval is interactive and shows the full command text.
  Any edit to the hook changes its hash and re-requires approval.
- No auto-approve config knob. A knob would immediately become the thing
  hostile READMEs tell users to set.

### D4 — Ultrasecure mode: **quarantined side fires no hooks**

Hooks run on the privileged side only. The quarantined `ask_internet`
subagent's tool calls never consult the hooks config: hooks are user
automation with full-environment shell access, and the quarantined side
exists precisely because its inputs are hostile. Letting quarantined events
trigger privileged-side shell would tunnel straight through the boundary.
(Symmetric with the permission-model decision that quarantined-side calls
skip permission rules — the broker's fixed tool surface is the whole
policy there.)

## Implementation hand-off (mechanical)

1. `security/` or `core/hooks.py`: fingerprint function + origin tagging.
   Loader tags each `HookConfig` with `origin: "user" | "project"` (loader
   knows which file each layer came from; thread it through `_coerce_hooks`).
2. Approval store `~/.config/agent/approved_hooks.json` (create 0600),
   `/hooks` command: list (with origin + approval state), `approve <n>`,
   `revoke <n>`. Follow `path_grants.py` persistence idioms.
3. `_matching()` in `core/hooks.py`: skip unapproved project-origin entries;
   collect them for the once-per-session warning.
4. Attribution prefixes (D2) + redaction pass on hook output before it
   reaches the model/notes.
5. Tests: hostile-repo fixture (project agent.toml with a `block=true`
   `tools=["*"]` hook writing a marker file) — assert hook does not run and
   warning fires; approval flow round-trip; hash invalidation on edit;
   secaudit-unblocked test (D1); quarantined-side no-hooks test (D4).
6. Follow-up issue (separate from hooks): project-layer `[security]`
   overrides — decide clamp-or-approve with the same fingerprint machinery.

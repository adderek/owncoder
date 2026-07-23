# Permission model design (S1)

Status: **design approved for implementation** — implementation is a
PLAN_NORMAL-class task; every decision that requires judgment is made here.

## Problem

The security suite is strong at scan/audit (secaudit, review, verify, evolve)
and at coarse confinement (sandbox, write/read-deny globs, path grants,
air-gap, ultrasecure). What's missing is *interactive permissioning* between
those extremes: per-tool allow/ask/deny rules with argument-level matching,
like Claude Code's permission rules or Codex's approval modes. Today a tool
is either callable or it isn't.

## Non-goals

- Not a replacement for the sandbox, fs gate, or deny globs. Those are
  enforcement layers below the LLM; permission rules are a *policy* layer
  that decides whether a tool call is attempted at all.
- No UI redesign: the ask flow reuses the existing `ask_user` signal path
  (Textual, HTML, and remote-relay Q/A plumbing already exist).
- No per-model or per-session-type rules in v1. One rule set, three verdicts.

## Rule grammar

Config lives in `[permissions]` (SecurityConfig sibling, own section — it is
policy, not sandbox mechanics):

```toml
[permissions]
default = "allow"          # verdict when no rule matches: allow | ask | deny
                           # ("allow" preserves current behavior; setting
                           #  "ask" turns the agent into approve-every-tool)

[[permissions.rules]]
tool = "run_command"        # tool-name glob (fnmatch): "run_command", "web_*", "*"
match = "git push*"         # optional arg matcher (see below)
verdict = "ask"             # allow | ask | deny
reason = "pushes publish"   # optional; shown in the ask prompt / deny error
```

### Matching semantics

- **Evaluation order**: rules are evaluated in config order, **first match
  wins**. No implicit specificity ranking — explicit order is auditable and
  trivially testable. `default` applies when nothing matches.
- **`tool`**: fnmatch glob against the tool name. Required.
- **`match`**: optional, matched against the tool's *primary argument*:
  - `run_command` → the command string, matched as a **glob** by default;
    prefix with `re:` for a regex (`match = "re:^git\\s+push"`). Glob
    default because regex-by-default invites catastrophic mistakes
    (`.` matching everything).
  - file tools (`edit_file`, `write_file`, `read_file`, …) → the `path`
    argument, root-relative, glob only (`match = "src/**"`). Regex on paths
    is refused at config-load time — paths are what globs are for.
  - tools with no natural primary argument: `match` is a config error,
    caught at load (fail loud, not silently-never-matching).
- The primary-argument table is a static dict in code
  (`PERMISSION_MATCH_ARG: dict[str, str]`), not config — which argument is
  "primary" is a property of the tool, not user policy.

### Verdicts

- `allow` — proceed.
- `deny` — the tool call fails with a structured error naming the rule and
  its `reason`; the model sees it and can re-plan. Not silent.
- `ask` — fire the ask flow (below). Timeout/no-answer resolves to **deny**
  (fail closed).

## Composition with existing modes — the narrowing invariant

**Invariant: permission rules can only narrow, never widen.** Enforced
structurally, not by config validation:

```
tool call → mode gates (ultrasecure / air-gap / sandbox / deny-globs)
          → permission rules (allow | ask | deny)
          → execution
```

Permission evaluation runs *after* the existing gates, and its `allow`
verdict merely means "no additional restriction". Concretely:

- **Air-gap**: a rule `tool = "web_fetch", verdict = "allow"` changes
  nothing — airgap already refused the call before rules were consulted.
  There is no code path where a permission verdict re-enables an egress
  the air-gap layer blocked.
- **Ultrasecure**: same ordering on the privileged side. On the quarantined
  (broker) side, permission rules are **not consulted at all** — the
  quarantined agent's tool surface is fixed by the broker, and letting
  repo-level config alter it would hand page content an indirect knob.
  (Mirrors the S4 decision that quarantined-side events fire no hooks.)
- **Write/read-deny globs, path grants**: unaffected; they gate the fs layer
  below rules. A rule can `deny` a path the fs gate would have allowed —
  never the reverse.

The invariant gets a dedicated test: for every mode-blocked call, assert the
permission engine was never even invoked (call-order test, not just outcome).

### Security-suite internal calls

Tool calls issued by the security suite itself (secaudit, verify, evolve
runners) bypass `ask`/`deny` rules — same reasoning as S4's "hooks must not
block audit internals": a repo-shipped config must not be able to blind the
audit that would catch it. Bypass is keyed on an internal call-site flag
(`_internal_security=True` threaded from the suite's executors), never on
anything the model or config controls.

## Ask flow

Reuses the existing `ask_user` signal path end-to-end:

- Question: tool name, the matched rule's `reason`, and a compact rendering
  of the arguments (command string / path). Argument rendering passes through
  the existing redaction layer first — an ask prompt must not leak a secret
  the tool output would have had masked.
- Choices: **Allow once** / **Allow for session** / **Deny** /
  **Deny for session**. "For session" answers append an in-memory rule at the
  *front* of the rule list (first-match-wins makes the grant/refusal sticky
  for exactly matching calls).
- Remote relay: works unchanged (ask_user signals already support remote
  answers); the timeout that the signal path already applies resolves to
  deny, fail closed.
- No "always allow, persist" choice in the ask prompt itself. Durable rules
  are written by the human editing config or via an explicit
  `/permissions` command — a one-keystroke path from "the agent wants X" to
  "X is allowed forever" is how users train themselves to grant everything.

## Persistence

- **Session grants**: in-memory only, front of the rule list, die with the
  process. Never written to disk.
- **Durable rules**: two sources, merged in this order (later = higher
  precedence at the front of the list):
  1. `[permissions]` in the normal config layers (user/device/local/project
     TOML) — ordinary config, subject to the existing layer merge.
  2. `.agent/permissions.json` — rules added via `/permissions` at runtime.
- `.agent/permissions.json` **must be in the built-in `write_deny_globs`**
  (same treatment as `path_grants.json`, which already established the
  pattern and the reason: the agent's own file tools must not be able to
  edit the policy that constrains them — otherwise a prompt-injected agent
  rewrites its own permissions and every rule is theater).
- Project-layer TOML `[permissions]` rules from a *cloned repo* are
  untrusted input. v1 keeps it simple and safe: project-layer rules may only
  contain `deny`/`ask` verdicts; an `allow` rule in the project layer is
  ignored with a warning (a hostile repo may restrict the agent, never
  loosen it). Revisit alongside S4's content-hash approval mechanism if
  project-level allows turn out to be wanted.

## Failure modes

- Malformed rule (bad glob, `re:` that doesn't compile, `match` on a tool
  with no primary arg): config-load error, session refuses to start. Policy
  must not degrade silently.
- Permission engine internal error at call time: resolve to `ask` if
  interactive, `deny` otherwise. Never fail open.
- `.agent/permissions.json` unreadable/corrupt: warn + ignore the file
  (config-layer rules still apply); do not fail open by skipping the whole
  engine.

## Implementation sketch (PLAN_NORMAL hand-off)

1. `config/models.py`: `PermissionRule` + `PermissionsConfig` dataclasses;
   loader wiring + load-time validation (glob/regex compile, primary-arg
   check, project-layer allow-stripping). ~Tests: load/validate matrix.
2. `security/permissions.py`: `evaluate(tool_name, args) -> Verdict` with
   the rule walk, session-grant list, `PERMISSION_MATCH_ARG` table, and the
   `_internal_security` bypass. Pure function + tiny state object; the bulk
   of the test surface lives here, offline.
3. `core/tool_calls.py`: invoke `evaluate()` in `execute_tool()` *after* the
   existing mode gates, before hooks/pre_tool. Deny → structured error;
   ask → existing ask_user signal, map answer to verdict + session grant.
4. `.agent/permissions.json` read/write + `write_deny_globs` addition +
   `/permissions` command (list rules, add durable rule, drop session
   grants). Follow `path_grants.py` persistence idioms.
5. Call-order tests for the narrowing invariant (ultrasecure, air-gap,
   deny-globs) and the ask-timeout→deny path.

Each step is mechanically verifiable; none requires judgment beyond this
document.

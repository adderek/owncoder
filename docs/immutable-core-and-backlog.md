# The immutable core, the tool ledger, and the backlog

2026-07-25. Answers a question worth stating plainly: *what part of this agent's
instructions can the agent itself change?*

Before this: all of them. `reflector` writes behavioral rules, `skill_distiller`
writes procedures, `promoter` writes facts, and `prompt_compiler` hands the
prompt text itself — including `base_rules.txt` — to a local model to rewrite,
with a token-savings check as the only gate. Nothing in that loop involves a
human, and a compressor that drops a safety line to save tokens passes that gate.

## 1. The core (`prompts/core.txt`, `agent/core/core_rules.py`)

Defined by what it forbids:

| Property | Mechanism |
|---|---|
| Human input only | no write path exists in any tool; `propose_core_change` files a backlog item instead |
| Never compiled | `prompt_compiler.NEVER_COMPILE`, checked **before** the enabled flag, so no config can opt in |
| Never compacted | injected first, with `HARD_RULES_MARKER` |
| Not writable by the agent | fs-gate write-deny globs, imported from `core_rules.WRITE_DENY_GLOBS` so list and rationale cannot drift |
| Accountable | `.agent/core_history.jsonl`: every observed change, per-file digests, and the reason from the git commit behind it |

Two sources, concatenated: shipped `prompts/core.txt`, then optional
`.agent/core.md` per project. A project may **add** rules, never remove shipped
ones — narrowing only, the same direction the permission layer allows.

History is triggered by **observation at session start**, not by the act of
editing. A human edit, a branch switch and a bad merge all change the core, and
only one of them goes through anything the agent runs.

Where the reason comes from: `git log -1` on the file. An untracked file records
`"unrecorded — file is not tracked by git"` rather than an invented reason.
That is also the argument for keeping the core in git: the commit message is the
only account of intent that survives.

### Proposing a change

`propose_core_change(title, rationale, proposed_text, replaces)` → backlog item
of type `core_change`, status `raw`, `source=agent`. The tool result says the
core is unchanged, so a filed proposal cannot be reported as an applied rule.
The module has no `write_text` and no `open(` — enforced by a test, because a
guarded write is still a write.

Applying one means a human editing the file, in a commit whose message says why.

## 2. The tool ledger (`agent/core/tool_ledger.py`)

Tool schemas are the agent's API to the world and change behaviour silently: a
renamed parameter, a tightened `required`, a description rewritten to steer the
model differently. Nothing recorded any of it, so "it used to do this correctly"
had nowhere to be checked.

`.agent/tool_history.jsonl` records `added` / `removed` / `changed` per tool,
with a schema digest, which sections moved, and the reason from the commit that
last touched the implementing file. The file is resolved through
`inspect.getsourcefile` on the live registry — exact, not grepped.

`required` is called out separately from `parameters`: tightening it breaks
calls that used to work.

State is replayed from the entries, so a truncated ledger loses history and
re-baselines rather than reporting a change that never happened.

Query it with `tool_change_history(tool=…)`.

## 3. The backlog (`agent/ideas/`, `agent todo`)

Already existed as a SQLite store with statuses `raw → evaluated → planned →
implementing → verifying → done | rejected`, reachable only via `/idea` in a
chat session. Now also `agent todo`: `list`, `add`, `show`, `set`, `done`,
`reject`, `export`, `import`.

This is a holding place until there is a real tracker, and it is built to be
left: `export` emits versioned JSON of the whole backlog, `import` restores it
with **ids preserved**, so `plan_ref`, `session_ref` and any id written into a
commit message keep pointing at the same item. Migration to Jira or anything
else should be a script over an export.

Deliberately absent: assignees, sprints, workflow rules, notifications. Adding
them makes this a tracker to maintain instead of a place to put work items.

## What this does not do

- **Does not verify that a compiled prompt kept its meaning.** The core is
  exempt from compilation entirely, which sidesteps the problem rather than
  solving it. `base_rules.txt` and `system.txt` are still compiled with only a
  token-savings check.
- **Does not seal the core with `security/integrity.py`.** The fs gate stops the
  agent's own tools; it does not detect an edit made by something else. Sealing
  `prompts/` would close that, and is worth doing.
- **Does not route learning-loop output into `core_change` proposals.**
  `reflector` still writes behavioral rules directly. A rule that contradicts a
  core rule should surface as a proposal; today nothing checks for that.

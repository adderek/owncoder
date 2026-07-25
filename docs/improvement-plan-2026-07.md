# Improvement plan — 2026-07 (competitive gap closure)

Goal: close the feedback-loop and productization gaps found comparing `agent/`
against Claude Code, Codex, Cursor, Windsurf and Hermes-style agents. Feature
breadth is already ahead in security, RAG/ASM, graph and self-improvement; what
is missing is *fast correctness feedback*, *interactive policy*, and *cost
control*.

## Context

- Key files: `agent/core/turn.py` (turn loop, 1300+ lines), `agent/core/hooks.py`,
  `agent/security/policy.py`, `agent/config/models.py` (all config dataclasses),
  `agent/core/checkpoint.py`, `agent/prompt_compiler/`, `agent/evals/run.py`,
  `agent/ui_server/`.
- Constraints: no breaking config changes (every new knob defaults to today's
  behavior); local-first must keep working with zero cloud deps; each step lands
  as its own commit in the `agent` submodule plus a parent `bump agent:` commit.
- Verification per step: `.venv/bin/python -m pytest agent/tests/unit -q`.

## Steps

### S1 — Post-edit diagnostics (fast, file-scoped)

**Why:** `VerifyConfig` already runs a *project-wide* command at end of turn.
The Cursor/Windsurf edge is different: a *per-edit*, *file-scoped*, sub-second
checker whose output lands in the tool result the model is already reading. Turns
a 3-turn "edit → run tests → read failure → fix" cycle into a 1-turn fix.

**Do:** new `agent/core/diagnostics.py` — language→checker table (ruff/pyright for
`.py`, tsc/eslint for `.ts`, etc.), config `[diagnostics]` (disabled by default,
`auto` detection of installed checkers), invoked from `turn.py` after a
successful mutating tool call, result appended as `_diagnostics` on the tool
result JSON. Budgeted: hard timeout, only the edited file, findings capped.

**Accept:** disabled → byte-identical tool results; enabled → syntax error in an
edited file surfaces in the same tool result; timeout never blocks the turn.

### S2 — Interactive permission layer

**Why:** `docs/permissions-design.md` is approved-but-unbuilt. Today a tool is
callable or not — no ask/deny between "full sandbox" and "nothing". This is the
central CC/Codex UX and it unblocks the diff-review UI (S6).

**Do:** implement the approved design — `[permissions]` config, rule matching
(tool glob + arg matcher + verdict), enforcement at tool-dispatch in `turn.py`,
`ask` routed through the existing `ask_user` signal path.

**Accept:** default config = today's behavior; `deny` returns a policy error to
the model without executing; `ask` blocks on the existing UI Q/A path.

### S3 — Prompt caching + cache-stable prefix

**Why:** no `cache_control` anywhere. Cloud tiers re-pay full input every step.

**Do:** guarantee stable prefix ordering in the compiled prompt, emit provider
cache breakpoints when the endpoint advertises support, expose hit/miss in the
existing usage accounting.

**Accept:** local/OpenAI-compatible endpoints unaffected; cached runs show
reduced billed input tokens in `/context`.

### S4 — Eval gate + failure-corpus miner

**Why:** `evals/` exists (12 fixtures) but is manual, so prompt/harness changes
are still ungated. And `failure_report.py`/`crash_report.py` collect failures
that nothing aggregates.

**Do:** CI-callable eval entry point with a score threshold; a miner that turns
recurring real-session failures into new eval fixtures.

**Accept:** one command returns non-zero on regression; miner produces a fixture
from a recorded failure.

### S5 — Persist checkpoints

**Why:** `core/checkpoint.py` is explicitly in-memory/session-scoped. CC and
Cursor persist rewind across restarts; losing rollback on crash is exactly when
it is needed.

**Do:** journal to `.agent/checkpoints/`, restore on startup, prune by age/size.

**Accept:** checkpoint → restart → rollback still works.

### S6 — VS Code client over `ui_server`

**Why:** `UIServerProtocol` + HTTP sidecar already abstract the backend. No IDE
client exists, and that is the distribution channel for every competitor.

**Do:** thin extension: webview chat against the sidecar, diff review using S2's
ask verdict.

**Accept:** extension connects, chats, renders a diff and accepts/rejects it.

### S7 — Split `core/turn.py` + repo hygiene

**Why:** 1300-line turn loop with 20+ nested closures is the riskiest file in the
codebase; `docs/` carries dead `LEGACY_HELPER_v0..v5`-class files that pollute
retrieval; the repo root carries stray logs/backups.

**Do:** extract model-call, tool-dispatch, loop-guard and compaction into modules
behind the current `run_turn` signature; archive dead docs; clean root junk.

**Accept:** no behavior change, unit + integration tests green.

**Done** (agent `5c054bd`, parent `chore: archive superseded docs…`). Deviations
from the plan as written:

- Extracted four modules — `turn_setup` (tool selection + API-message fixups),
  `turn_batch` (dedup/execute/time/compact one batch), `turn_guards` (result
  rewriting guards), `turn_errors` (endpoint failure policy) — 1351 → 933 lines.
  *Compaction* was **not** extracted: its two call sites are three lines each
  and closing over the turn's `messages`/`budget`/`_phase`, so a module boundary
  there would have cost more indirection than it removed. Verify stayed too:
  tests patch `_run_verify_command` on `core.turn`.
- Root junk needed no commit — `crash-*.txt`, `even-terminal-*.log`, `*.bak` and
  `/_` are already gitignored and untracked, i.e. local-only files that are the
  user's to delete. One *tracked* stray was removed: root `streaming.py`, a
  comment fragment ending in `pass` that shadowed `agent/core/streaming.py`.
- Dead docs were **archived, not deleted**: `docs/archive/` + `.agent.ignore`,
  which is the mechanism the indexer already has for this (`search_archive`
  still reaches hidden entries). Only provably superseded files moved (17 of
  58); "looks old" was not treated as evidence.
- Found and fixed an unrelated real defect on the way: `apply_concurrency_pragmas`
  ran `PRAGMA journal_mode=WAL` before setting `busy_timeout`, and that pragma
  does not wait on the busy handler — a concurrent writer made a lazily-opened
  connection raise "database is locked", which the best-effort stores swallow as
  a dropped write. Was a ~40% flake in `test_model_reliability.py`.

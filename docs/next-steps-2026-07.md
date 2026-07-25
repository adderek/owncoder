# What is left after the 2026-07 improvement plan

Written 2026-07-25, at the end of the session that executed
[improvement-plan-2026-07.md](improvement-plan-2026-07.md) (S1–S7, all landed)
plus two follow-ups from that plan's own "what is left" list: miner staleness
detection and sidecar permission prompts.

State at the time of writing: unit suite **1954 passed, 1 skipped**; nothing
pushed (see N1 — this is the only item whose risk grows while it waits).

Ordering below is by cost of *not* doing it, not by effort.

## N1 — Push, or decide not to

Everything from this plan exists on one machine only.

- `agent` submodule: 17 commits ahead of `origin/master`
- parent repo: `review-fixes` has no upstream set; 44 ahead of
  `fractal/review-fixes`, ~101 ahead of `master`

`.githooks/pre-push` runs the unit suite, and `core.hooksPath` is set, so a push
self-checks. The `review-fixes` → `master` gap is an older decision than this
session and should be settled separately from pushing the branch.

Not automated deliberately: pushing is outward-facing and the branch topology is
the user's call.

## N2 — Load the VS Code extension in a real editor once

`clients/vscode/` has never been run in a VS Code window. What *is* covered:
the table parser and client logic (`clients/vscode/test/smoke.js`, 15 checks).
What is not covered by anything: activation, webview creation, the CSP nonce
path, and the CSS.

Check: `F5` in `clients/vscode/`, then send one prompt whose answer contains a
pipe table. Failure modes to expect are the untested ones above, not the parser.

Until this is done, treat the extension as unverified regardless of green tests
— a passing parser suite in node says nothing about whether the panel opens.

## N3 — ~~Run the integration suite~~ (done 2026-07-25 — and it was not what it looked like)

Recorded here because the diagnosis in the first draft of this document was
wrong, and the wrong version is the kind that gets believed.

`tests/integration` was 1 skipped, and this document assumed it needed a live
model. It does not: it is the kb fixture corpus, no LLM and no network, and it
runs in 0.15s. It skipped because pytest's `pythonpath` was `[".."]` while `kb`
is src-layout, so `kb.model` never imported no matter what was checked out — a
skip is a pass, so nothing ever complained. Fixed by adding `../kb/src`; all 9
tests pass, and they are now in the pre-push gate.

The real gap the wrong version was pointing at still stands and is **not**
covered by any test: no test in this repo exercises a live tool-calling round
trip. The `e2e` marker is registered and excluded by default `addopts` — and no
test in the repo uses it, so that suite is empty rather than merely unrun. That
is where a live round trip belongs, against the LAN workhorse
(192.168.31.42:8093) rather than a paid endpoint.

## N4 — Reconcile the local-only working tree

Four pre-existing uncommitted things in the parent repo, untouched all session
and left for their owner to decide:

| Path | State | Note |
|------|-------|------|
| `agent.toml.new` | untracked, **not** gitignored | will keep showing up in `git status` until resolved or ignored |
| `docs/FEATURES_LIST.md` | modified | |
| `.claude/settings.local.json` | modified | |
| `kb` | submodule pointer moved | needs its own commit inside `kb` first |

Root junk (`crash-*.txt`, `even-terminal-*.log` ~718KB, `agent.toml.bak`, `_`) is
untracked *and already gitignored* — local files, safe to delete, not mine to
delete.

## N5 — Staleness beyond the traceback

`evals/mine.py` now judges a mode `live`/`stale`/`?` by comparing the files in
its traceback against git. Two known limits, both acceptable today:

- Records with no traceback (invalid tool calls — a large share of them) are
  always `?`. Attributing those would mean mapping a tool name to its
  implementing module, which is guessable but not reliable.
- "The file changed" is not "the bug is fixed". Stale modes are therefore still
  listed and still scaffoldable, just demoted.

A better signal exists and is more work than it is worth right now: check
whether the failing call still fails, by replaying the recorded arguments
against the current tool in a sandbox. That is a real eval, which is what the
scaffold asks a human to write.

## N6 — Markdown in the VS Code webview

The webview renders tables only. The rest of the agent's markdown shows as
literal text there, while both browser UIs render it via
`agent/ui/static/md.js`.

Deliberate: full rendering means assigning `innerHTML` in a webview that shares
the editor's process. If this is picked up, the way to do it safely is to reuse
`md.js` and keep cell/text insertion on `textContent`, not to widen
`tables.js`'s remit.

## N7 — Headless `agent run` still denies every `ask`

The sidecar closed this for `--ui simple`. Headless `agent run` has no surface
at all, so an `ask` verdict is a denial there.

This is correct fail-closed behavior, not a bug, and the fix is a *policy*
decision rather than a UI one: either document that `run` needs rules with no
`ask` verdicts, or add a `--permissions-deny-ask`/`--yes` style flag that makes
the refusal explicit at invocation time. Do not add a prompt to a
non-interactive command.

## Not on this list

Things considered and rejected, so they do not get re-proposed:

- Extracting compaction or verify out of `run_turn` (S7 deviation) — two call
  sites of three lines, closing over turn state; a module boundary costs more
  than it removes.
- Deleting the 41 older docs left un-archived in S7 — "not touched since May" is
  not evidence a doc is wrong. Only provably superseded docs were archived
  (17 of 58), via `.agent.ignore` rather than deletion.

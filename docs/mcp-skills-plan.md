# MCP + skills rollout plan

Created 2026-11-11. Executed by separate low-context agent sessions, one task per session.

## Protocol for executing agent

1. Pick first task with `[ ]` whose deps are `[x]`. Read only this file + files the task names.
2. `scripts/coord beat` + `claim` per AGENTS.md.
3. Do task. Run its acceptance. Tests: `agent/.venv/bin/pytest <relevant paths> -x`.
4. Mark `[x]`, append `done YYYY-MM-DD, commit <sha>` + one-line note (deviations, numbers measured).
5. Commit in `agent/` (plain message), then root bump per AGENTS.md. Don't start next task in same session.

## Context (why)

- owncoder = local-first. Used where subsidized agents (Claude Code, Gemini, Codex) can't be: offline work, topics a model's vendor censors (e.g. Chinese models + Tiananmen/Uyghurs), security work Claude refuses, private code. Design for **small local models, small context, offline by default**.
- Skills are cheap: prompt carries one index line each (`agent/skills.py`, `skills_index_max`), body loaded on demand.
- MCP is expensive: `agent/mcp/manager.py:load_mcp_tools()` registers every allowed tool schema at startup → billed every turn. REA allowlist alone ~16k tok on fractal (per `~/.config/agent/agent.fractal.yaml` comment, not measured).
- Any network-dependent addition must be opt-in and degrade gracefully offline.

## Skill format

Bundled: `agent/prompts/skills/<name>.md`, read-only. Project-local: `.agent/skills/<name>.md`.
```
# Title
---
description: one line, used in index — make it trigger-worthy
---
- terse imperative bullets
```
Keep bodies ≤ ~40 lines (token-efficiency first; local models). Procedures > philosophy. No meta-instructions about tone.

## Tasks

### T1 [ ] Lazy MCP loading
Deps: none. Highest value: unlocks T7/T8 without per-turn cost.
- Add per-server `lazy: bool` to `MCPServerConfig` (`agent/config/models.py:1199`), default `false` (no behavior change).
- Lazy server: at startup register nothing but its name + one-line `description` (new field) in a compact list; expose one tool, e.g. `mcp_load(server)`, that starts the server and registers its allowed tools for the rest of the session. Optional `mcp_unload`.
- Tool registry must handle tools appearing mid-session (check how `tools/__init__.py:159` registers; next request must include new schemas).
- `/mcp` slash (`run_mcp_command`) shows lazy/loaded state.
- Respect existing `tools_allow`/`tools_deny`, sandbox, project-config clamp (`loader._clamp_project_mcp`).
- Docs: `agent/docs/rea.md` mention `lazy: true` for REA; `FEATURES_LIST.md` entry (config field + tool).
- Acceptance: unit tests — lazy server not spawned at startup; `mcp_load` spawns + registers; schema count in request before/after. Measure prompt tokens with REA lazy vs eager, record number here.

### T2 [ ] Remove dead MCP examples
Deps: none.
- `agent.yaml` (repo root) lists `fs` and `git` MCP servers (disabled) — duplicate built-in tools. Replace with one commented example of a lazy server (after T1) or a plain stdio example.
- Acceptance: config loads (`agent/.venv/bin/pytest agent/tests -k config -x`).

### T3 [ ] Skill: spec-and-plan
Deps: none.
- `agent/prompts/skills/spec-and-plan.md`: before code — restate goal, inputs/outputs, constraints, acceptance check; split into small atomic steps each independently verifiable; when to write a `/plan` (link `planning` skill, don't duplicate it).
- Merges TODO.md items "spec before code" + "plan small atomic tasks"; tick them in `TODO.md:195` list.
- Acceptance: `agent/.venv/bin/pytest agent/tests/unit/test_skills.py -x`; skill appears in `/skills`.

### T4 [ ] Skill: self-review
Deps: none.
- `agent/prompts/skills/self-review.md`: before declaring done — read own diff (`git diff`), check: matches request, no unrelated edits, no debug leftovers, error paths, tests run + result stated, simplification pass (dead code, duplication). Mention `review_changes` tool if it fits (read `agent/tools/review_changes/` first).
- Covers TODO "review" + "code-simplify".
- Acceptance: as T3.

### T5 [ ] Skill: debug
Deps: none.
- `agent/prompts/skills/debug.md`: reproduce first (exact command), read full error, one hypothesis at a time, minimal probe, bisect (`git bisect` / halving input), fix root cause not symptom, add regression test (`tests/regression/`).
- Acceptance: as T3.

### T6 [ ] Skill: github (optional, via `gh`)
Deps: none.
- `agent/prompts/skills/github.md`: PR/issue/CI workflow through `gh` CLI over shell — not an MCP (zero schema cost, works with any model). Cover: `gh pr create/view/checks/diff`, `gh issue list/view`, `gh run view --log-failed`, `gh api`. Rules: never push/comment without user confirmation; respect AGENTS.md immutable commits; check `gh auth status` first, if unavailable say so and stop (offline is a normal state).
- Description must make clear it's only for repos with a GitHub remote.
- Acceptance: as T3.

### T7 [ ] Optional MCP: browser (Playwright)
Deps: T1.
- Document (not enable) in `agent/docs/mcp-recipes.md`: pinned install, `lazy: true`, `sandbox: bwrap`, `tools_allow` ≤ ~6 tools (navigate, snapshot, click, type, screenshot, console). Use case: verify HTTP UI, read JS-rendered pages.
- Measure schema size of the allowlist (tokens) and record it.
- Acceptance: manually load once via `mcp_load`, navigate to local HTTP UI, record result here. If not possible in session, say so.

### T8 [ ] Optional MCP: library docs lookup
Deps: T1.
- Evaluate a docs MCP (e.g. Context7) vs offline alternative (local docs index via existing RAG `agent index` on vendored docs). Network service → leaks queries; must be opt-in, flagged unsuitable for private mode (see memory/notes on private vs cloud modes).
- Output: recipe in `agent/docs/mcp-recipes.md` or a "rejected, use X" note here. No code unless trivially small.

### T9 [ ] Language skills (only if used)
Deps: none. Low priority.
- `bash.md` (set -euo pipefail, quoting, shellcheck) and `c-cpp.md` (UB, sanitizers, `-Wall -Wextra`, embedded/UART notes). Only write if user confirms frequent C/shell work; otherwise mark `[-] skipped`.

## Not doing (decided)

- GitHub MCP — `gh` skill (T6) does same at zero per-turn cost.
- fs/git MCP — duplicate built-ins.
- "ship/deploy" skill — wait until stable-as-container-image exists (AGENTS.md "Repos & branches").
- Tone/meta skills (karpathy/caveman style) for owncoder — local models follow concrete procedures better than meta-guidance.

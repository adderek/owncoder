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
- MCP: `agent/mcp/manager.py:load_mcp_tools()` spawns every enabled server at startup and registers all allowed schemas. With `[tool_discovery] enabled = true` (set in user config) only core schemas + a catalog go per turn; MCP schemas load via `find_tools`. Without discovery, full cost (~16k tok for REA, `docs/rea.md`). Remaining MCP costs: catalog lines (fixed by T1), startup spawn time/RAM.
- Any network-dependent addition must be opt-in and degrade gracefully offline.

## Skill format

Bundled: `agent/prompts/skills/<name>.md`, read-only. Project-local: `.agent/skills/<name>.md`.
```
---
description: one line, used in index — make it trigger-worthy
---

# Title
- terse imperative bullets
```
Frontmatter MUST be line 1 (`parse_skill()`); test enforces it.
Keep bodies ≤ ~40 lines (token-efficiency first; local models). Procedures > philosophy. No meta-instructions about tone.

## Tasks

### T1 [x] MCP tools in discovery catalog: one line per server
done 2026-11-11. Original plan (lazy loading) was based on wrong premise: progressive tool disclosure (`core/tool_discovery.py`, `find_tools`) already keeps MCP schemas out of the per-turn request. Residual cost was one catalog line per MCP tool (REA 42 tools ≈ 4 KB ≈ 1k tok/turn, estimate from synthetic test, not measured on real REA). Done: `render_catalog()` collapses `mcp__<srv>__*` to one line; new `MCPServerConfig.description` field feeds it (fallback: sample tool names). `find_tools` unchanged. Tests in `tests/unit/test_tool_discovery.py`.
Follow-up for user: add `description:` to REA in both `agent.fractal.yaml` + `agent.sharkoon.yaml`.

### T1b [ ] Lazy MCP spawn (optional)
Deps: T1. Only if startup time/RAM of servers hurts (REA spawns JVM/Ghidra lazily? check first). Add `lazy: bool` to `MCPServerConfig`; lazy server spawns on first `find_tools` match against its name or first call. Measure startup time before/after; skip with `[-]` if gain < 1 s.

### T2 [x] Remove dead MCP examples
done 2026-11-11 (root repo commit). fs/git/http examples replaced by one commented stdio example; `load_config()` verified.
Deps: none.
- `agent.yaml` (repo root) lists `fs` and `git` MCP servers (disabled) — duplicate built-in tools. Replace with one commented stdio example incl. `description`.
- Acceptance: config loads (`agent/.venv/bin/pytest agent/tests -k config -x`).

### T3 [x] Skill: spec-and-plan
done 2026-11-11. Also fixed: all bundled skills had `# Title` before frontmatter → `parse_skill()` ignored `description:` and leaked `---` into body. Frontmatter now first; regression test `test_bundled_skills_have_frontmatter_description`.
Deps: none.
- `agent/prompts/skills/spec-and-plan.md`: before code — restate goal, inputs/outputs, constraints, acceptance check; split into small atomic steps each independently verifiable; when to write a `/plan` (link `planning` skill, don't duplicate it).
- Merges TODO.md items "spec before code" + "plan small atomic tasks"; tick them in `TODO.md:195` list.
- Acceptance: `agent/.venv/bin/pytest agent/tests/unit/test_skills.py -x`; skill appears in `/skills`.

### T4 [x] Skill: self-review
done 2026-11-11.
Deps: none.
- `agent/prompts/skills/self-review.md`: before declaring done — read own diff (`git diff`), check: matches request, no unrelated edits, no debug leftovers, error paths, tests run + result stated, simplification pass (dead code, duplication). Mention `review_changes` tool if it fits (read `agent/tools/review_changes/` first).
- Covers TODO "review" + "code-simplify".
- Acceptance: as T3.

### T5 [x] Skill: debug
done 2026-11-11.
Deps: none.
- `agent/prompts/skills/debug.md`: reproduce first (exact command), read full error, one hypothesis at a time, minimal probe, bisect (`git bisect` / halving input), fix root cause not symptom, add regression test (`tests/regression/`).
- Acceptance: as T3.

### T6 [x] Skill: github (optional, via `gh`)
done 2026-11-11.
Deps: none.
- `agent/prompts/skills/github.md`: PR/issue/CI workflow through `gh` CLI over shell — not an MCP (zero schema cost, works with any model). Cover: `gh pr create/view/checks/diff`, `gh issue list/view`, `gh run view --log-failed`, `gh api`. Rules: never push/comment without user confirmation; respect AGENTS.md immutable commits; check `gh auth status` first, if unavailable say so and stop (offline is a normal state).
- Description must make clear it's only for repos with a GitHub remote.
- Acceptance: as T3.

### T7 [ ] Optional MCP: browser (Playwright)
Deps: none (T1 done).
- Document (not enable) in `agent/docs/mcp-recipes.md`: pinned install, `description`, `sandbox: bwrap`, `tools_allow` ≤ ~6 tools (navigate, snapshot, click, type, screenshot, console). Use case: verify HTTP UI, read JS-rendered pages.
- Measure schema size of the allowlist (tokens) and record it.
- Acceptance: manually load once via `mcp_load`, navigate to local HTTP UI, record result here. If not possible in session, say so.

### T8 [ ] Optional MCP: library docs lookup
Deps: none.
- Evaluate a docs MCP (e.g. Context7) vs offline alternative (local docs index via existing RAG `agent index` on vendored docs). Network service → leaks queries; must be opt-in, flagged unsuitable for private mode (see memory/notes on private vs cloud modes).
- Output: recipe in `agent/docs/mcp-recipes.md` or a "rejected, use X" note here. No code unless trivially small.

### T9 [x] Language skills
done 2026-11-12. User confirmed frequent shell + C (llama.cpp). `bash.md`, `c-cpp.md` (llama.cpp CMake options verified against local checkout: `LLAMA_/GGML_SANITIZE_{ADDRESS,UNDEFINED,THREAD}`, `GGML_{HIP,CUDA,VULKAN}`, C11/C++17).

## Not doing (decided)

- GitHub MCP — `gh` skill (T6) does same at zero per-turn cost.
- fs/git MCP — duplicate built-ins.
- "ship/deploy" skill — wait until stable-as-container-image exists (AGENTS.md "Repos & branches").
- Tone/meta skills (karpathy/caveman style) for owncoder — local models follow concrete procedures better than meta-guidance.

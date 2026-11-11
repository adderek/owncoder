---
description: Before writing code for a non-trivial request — pin goal, constraints, acceptance check, then split into small verifiable steps
---

# Spec and plan before code

Use when the request touches more than one file, is ambiguous, or you can't name how to verify it.

1. Restate the goal in one sentence. Name what "done" looks like from the user's side.
2. Inputs/outputs: what goes in, what comes out, formats, edge cases (empty, huge, invalid, offline).
3. Constraints: files you must not touch, APIs to keep stable, priorities from CLAUDE.md / `.agent.priorities.toml`.
4. Acceptance check: the exact command or observation that proves it works (test name, CLI run, output line). If none exists, the first step is to create one.
5. Read the code you'll change before deciding how — `find_symbol`, `explore`, `read_file`. Don't plan from memory.
6. Split into steps that each:
   - change one thing,
   - leave the code working (tests pass),
   - have their own check.
   Order: test or reproduction first, then smallest change, then cleanup.
7. Ambiguity: pick the most likely reading, say "Assuming X", continue. Ask only if a wrong guess is destructive or wastes the whole task.
8. More than ~4 steps or work spanning sessions → make it a plan (load skill `planning`).

Don't: write a spec longer than the change; plan steps you can't verify; refactor along the way unless asked.

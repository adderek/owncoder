---
description: Before declaring a change done — review own diff for scope, leftovers, error paths, tests, simplification
---

# Self-review before done

1. Read the diff: `git_diff` (or `git diff --stat` then per file). Review what's actually there, not what you meant to write.
2. Scope: every hunk serves the request. Revert unrelated edits, reformatting, drive-by renames.
3. Leftovers: debug prints, commented-out code, TODOs you added, temp files, hardcoded paths/secrets.
4. Correctness: error paths and edge cases (empty input, missing file, None, offline, timeout). Callers of changed functions still match (`find_symbol`).
5. Simplify: dead code, duplication, needless abstraction, a helper used once. Smallest diff that solves it.
6. Tests: run the relevant tests (`run_tests`). Report command + result. New behavior or fixed bug → test exists for it.
7. Docs: user-facing change (CLI, config, slash cmd) → docs updated per project rules.
8. Optional second opinion: `review_changes` (diff + security scan + review).
9. Report: what changed, how verified, what was not verified. Failures and skipped steps stated plainly.

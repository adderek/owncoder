---
description: Only for repos with a GitHub remote — PRs, issues, CI logs through the `gh` CLI (no MCP)
---

# GitHub via gh CLI

Check first: `git remote -v` shows github.com, and `gh auth status` succeeds. Otherwise say so and stop — offline and non-GitHub repos are normal.

Read (safe):
- `gh pr list` / `gh pr view <n> [--comments]` / `gh pr diff <n>` / `gh pr checks <n>`
- `gh issue list [--label X]` / `gh issue view <n> --comments`
- `gh run list --branch <b>` / `gh run view <id> --log-failed` (CI failures: read only failed logs)
- `gh api repos/{owner}/{repo}/...` for anything else; add `--jq` to keep output small.

Write (ask user first, every time): `gh pr create`, `gh pr comment`, `gh pr merge`, `gh issue create/comment/close`, `git push`.
- PR body: what + why + how verified. No @mentions of people unless user names them.
- Never force-push, amend or rebase pushed commits.

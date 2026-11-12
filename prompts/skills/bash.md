---
description: Writing or fixing shell scripts — strict mode, quoting, arrays, portable checks, shellcheck
---

# Bash scripts
- Shebang `#!/usr/bin/env bash`, then `set -euo pipefail`. Know the traps: `set -e` is ignored inside `if`/`&&`/`||` and in `$(...)` used as a function arg; `cmd | head` can fail on SIGPIPE under pipefail.
- Quote every expansion: `"$var"`, `"$@"`, `"${arr[@]}"`. Unquoted only when word splitting is intended — comment it.
- Lists of args → arrays: `args=(--flag "$x"); cmd "${args[@]}"`. Never build commands in strings + `eval`.
- `[[ ... ]]` for tests, `(( ... ))` for arithmetic. `local` in functions. `readonly` for constants.
- Defaults/required: `${VAR:-default}`, `${VAR:?VAR must be set}`.
- Temp files: `tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT`.
- Paths relative to script: `cd "$(dirname "${BASH_SOURCE[0]}")"` or `here=$(dirname "$(realpath "${BASH_SOURCE[0]}")")`.
- Loops over files: `find ... -print0 | while IFS= read -r -d '' f; do ...; done` or globs with `shopt -s nullglob`. Never `for f in $(ls)`.
- Read lines: `while IFS= read -r line; do ...; done < file`.
- Check tools exist: `command -v jq >/dev/null || { echo "need jq" >&2; exit 1; }`.
- Errors/prose to stderr (`>&2`), data to stdout, meaningful exit codes. Scripts for automation: no prompts.
- Dates: `date +%F`, `date '+%F %T'` (ISO-8601). Don't parse `ls` or locale output.
- Interactive entry point `_`: bare `_` prints help + offers choice; subcommands for actions. Automation calls a long-named script, not `_`.
- Verify: `bash -n script` (syntax), `shellcheck script` if installed, then run it once on a harmless input.

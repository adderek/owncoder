# Eval harness

A small, offline-testable harness for measuring the agent's success rate on
repeatable coding tasks. Use it to gate harness/prompt/model changes on a
score instead of vibes.

## Layout

```
evals/
  run.py             # the runner
  tasks/<id>.yaml    # task definitions
  fixtures/<id>/     # starting workspace for each task (a few files)
```

## Running

Run everything against the real agent (default invocation is
`<repo>/.venv/bin/agent run {prompt}`):

```
.venv/bin/python evals/run.py
```

List the available tasks without running anything:

```
.venv/bin/python evals/run.py --list
```

Run a subset of tasks:

```
.venv/bin/python evals/run.py --tasks fix-failing-test,rename-symbol
```

Keep the temp workspaces around for inspection (paths are printed at the end):

```
.venv/bin/python evals/run.py --keep
```

Write machine-readable results (per-task status, per-check pass/fail, timing)
to a file, e.g. for CI gating:

```
.venv/bin/python evals/run.py --json /tmp/eval-results.json
```

Override how the agent is invoked. `{prompt}` is substituted with the
shell-quoted task prompt; if the placeholder is absent the prompt is appended
as the final argument. This is also how the runner is tested offline, and how
you can point it at a different model/harness/prompt variant:

```
.venv/bin/python evals/run.py --agent-cmd 'python /path/to/fake_agent.py {prompt}'
```

Example fully-offline smoke test (no LLM involved) for the `answer-question`
task, using a "fake agent" that just writes the expected answer directly:

```
.venv/bin/python evals/run.py --tasks answer-question \
  --agent-cmd 'python -c "open(\"ANSWER.txt\", \"w\").write(\"load_config\")"'
```

The runner exits `0` if every requested task passed, `1` otherwise, so it can
be used as a CI gate.

## Judged scoring (LLM-as-judge)

Mechanical checks decide pass/fail. `--judge` adds a secondary 0-10 quality
score per task, judging dimensions checks can't see: minimal diff, style
match, no drive-by changes.

```
.venv/bin/python evals/run.py --judge --json /tmp/results.json
```

How it works (see `evals/judge.py`):

- After the checks run, the workspace is diffed against its pristine fixture
  (`diff -ruN`, `.agent`/`__pycache__` excluded).
- The judge model sees ONLY the task prompt and that diff — never the
  candidate agent's own output — so a candidate can't talk its way to a
  higher score.
- The call goes through `call_role_with_failover(config, "judge", …)`: it is
  failover-safe, honors air-gap mode, and is pinnable to your strongest
  model via `[model_roles] judge = "<entry>"` (falls back verify → default).
- Mechanical guards run before/after the LLM and can't be argued with:
  - empty diff → score 0, no LLM call;
  - `judge.forbid_paths` glob hit → score 0, no LLM call;
  - diff longer than `judge.max_diff_lines` (default 400) → score clamped
    to 4 no matter what the judge said.
- A judge failure (endpoint down, unparseable reply) is reported per-task as
  `judge.error` and never affects mechanical pass/fail or the exit code.

Per-task judge config in the task YAML (all optional):

```yaml
judge:
  type: fix            # rubric: fix | edit | locate (default fix)
  max_diff_lines: 400   # mechanical length-penalty threshold
  forbid_paths:         # globs the diff must not touch (checked mechanically)
    - "test_*.py"
```

## Absolute floor

`--baseline` catches *movement*. A suite that is already failing half its tasks
compares clean against a matching baseline and reports success. `--fail-under`
adds an absolute floor on the pass rate, and can be combined with `--baseline`:

```
.venv/bin/python evals/run.py --fail-under 0.8
```

## Mining eval tasks from real failures (`evals/mine.py`)

`failure_report.py` records every invalid tool call, tool exception and runtime
exception under `.agent/failures/`. `evals/mine.py` clusters those records into
recurring failure *modes* — normalising away paths, numbers and timestamps so
forty one-off records collapse into one ranked entry — and scaffolds an eval
task from a mode you pick.

```
python evals/mine.py                        # rank failure modes in this project
python evals/mine.py --project ~/src/other  # …repeatable, merges projects
python evals/mine.py --json /tmp/modes.json
python evals/mine.py --scaffold 1           # write a task from mode #1
python evals/mine.py --live-only            # hide modes that look already fixed
```

### The STATE column

A failure record is history, not a bug report — the code it came from may have
been rewritten since. Each mode is therefore judged against git: the files named
in its traceback are compared with when the failure was last seen.

| STATE | Meaning |
|-------|---------|
| `live` | at least one implicated file has not been committed to since the last sighting — the bug can still be there |
| `stale` | every implicated file has changed since — probably already fixed, so it sinks to the bottom of the ranking and its scaffold carries a warning |
| `?` | nothing to judge on: no traceback, no file found for the tool, or no git history for it |

Records with a traceback are attributed to the project files in it. Records
without one — invalid tool calls, the largest class — fall back to the module
that registers the named tool, found by grep rather than by importing another
checkout's registry. That is a weaker attribution (the fault may be in the
schema or the prompt, not the implementation), so a traceback always wins when
there is one.

`?` ranks with `live`: a mode that cannot be judged must not be demoted on no
evidence. `--no-staleness` skips the git lookups entirely. This is a heuristic —
"the file changed" is not "the bug is fixed", which is why stale modes are still
listed and still scaffoldable.

A scaffold writes `tasks/regress-<tool>.yaml` plus
`fixtures/regress-<tool>/FAILURE.json` (the evidence: real arguments, real
error, how often, over how many sessions). The task is marked `draft: true`, so
the runner skips it — a scaffold has placeholder prompts and checks and must not
fail the suite before a human finishes it. Run drafts explicitly with `--drafts`.

This is deliberately not automatic: a failure record proves something went
wrong, it does not specify what right looks like.

## Regression gating against a baseline

```
.venv/bin/python evals/run.py --judge --json /tmp/new.json --baseline /tmp/old.json
```

With `--baseline`, the exit code reflects regressions instead of raw
pass/fail: a task that flipped mechanical pass→fail, or whose judged score
dropped by more than 2 points, fails the run (`1`). New tasks, removed
tasks, and improvements are not regressions.

## How a task works

For each task the runner:

1. Copies the task's fixture directory (`evals/fixtures/<fixture>/`) into a
   fresh temp directory (the "workspace").
2. Runs the agent command with `cwd` set to the workspace and the task
   `prompt` filled in, enforcing `timeout_s`.
3. Runs each of the task's `checks` with `cwd` set to the workspace.
4. Records pass/fail per check, wall time, and the agent's exit code.

A task **passes** only if the agent didn't time out and every check passed.

A missing fixture directory or a malformed task file is reported as an
`error` for that one task — it does not abort the rest of the run.

## Task definition schema

```yaml
id: fix-failing-test          # required, must match the filename stem
prompt: "..."                  # required, the instruction given to the agent
fixture: fix-failing-test      # required, a directory under evals/fixtures/
timeout_s: 300                 # optional, defaults to 300
checks:                        # list of checks, ALL must pass
  - type: command               # run a shell command, cwd = workspace, exit 0 = pass
    run: "python -m pytest test_calc.py -q"
  - type: file_contains          # path must exist and contain text
    path: calc.py
    text: "return a + b"
  - type: file_not_contains      # path must exist and NOT contain text
    path: module_a.py
    text: "procces"
  - type: file_exists            # path must exist
    path: calc.py
```

Task files are YAML by default (pyyaml is already a project dependency). If
pyyaml were ever unavailable the runner falls back to loading `*.json` task
files with the same schema instead.

## Adding a task

1. Create `evals/fixtures/<id>/` with a small, self-contained starting
   workspace (a few files). Any pytest files in a fixture must be runnable
   with plain `python -m pytest` and must not depend on the `agent` project's
   own packages — they test the *fixture*, not the agent codebase.
2. Create `evals/tasks/<id>.yaml` describing the prompt, fixture, and checks
   (see schema above). `id` should match the filename stem.
3. Sanity-check it end-to-end without an LLM by writing a one-off
   `--agent-cmd` that does what a correct agent would do, and confirm the
   task passes; then confirm it *fails* if you skip the fix, to make sure the
   checks are actually discriminating.
4. Run `.venv/bin/python evals/run.py --tasks <id>` for real to confirm it
   also works against the live agent.

## Seed tasks

| id | what it tests |
| --- | --- |
| `fix-failing-test` | find and fix a one-line bug so a failing test passes |
| `add-function` | implement a function from a spec, verified by a test |
| `rename-symbol` | rename a symbol consistently across two files |
| `add-cli-flag` | add an argparse flag with the expected side effect |
| `answer-question` | read code and answer a question about it in a file |
| `multi-file-edit` | change a constant consistently across three files |
| `fix-off-by-one` | fix a loop bound bug that drops the last element |
| `fix-mutable-default` | fix a shared-mutable-default-argument bug |
| `fix-exception-type` | catch the correct exception type instead of a wrong one |
| `add-input-validation` | add raise-on-invalid-input logic verified by tests |
| `extract-duplicate-logic` | extract a shared helper out of two near-duplicate functions |
| `add-quiet-flag` | add an argparse flag that suppresses specific output |
| `fix-recursion-base-case` | fix a recursive function that never recurses |
| `fix-dict-key-typo` | fix a typo'd dict key lookup |
| `add-context-manager` | add `__enter__`/`__exit__` to an existing resource class |
| `fix-string-format-bug` | fix a `%`-formatting call missing an argument |

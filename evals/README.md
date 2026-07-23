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

---
description: Fix a bug or failing test — reproduce, isolate with one hypothesis at a time, fix root cause, add regression test
---

# Debugging procedure

1. Reproduce first. Get the exact command that fails and run it. No repro → no fix; collect more info (logs, inputs, versions) instead of guessing.
2. Read the whole error: last frame of a traceback is the symptom, follow it to your code. Read the code at each frame.
3. One hypothesis at a time. State it, pick the cheapest probe that would disprove it (print, assert, small script, `git_log`/`git_blame` on the lines), run it.
4. Narrow: halve the input, the config, or the commit range (`git bisect run <cmd>` when it used to work).
5. Fix the root cause, not the symptom. No blanket try/except, no retry loops hiding errors, no skipping the test.
6. Regression test: reproduces the bug, fails before the fix, passes after (`tests/regression/` if the project has it).
7. Re-run the original repro and the surrounding test suite.
8. Stuck after 3 disproven hypotheses: stop, summarize what was ruled out and what evidence exists, ask the user.

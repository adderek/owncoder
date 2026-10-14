# Binary analysis — suggested improvements

Status: proposed, not started. Context: `analyze_binary` (`rag/binary_pipeline.py`,
`rag/asm_describer.py`), setup and limits in `rea.md`, workflow in skill `legacy-re`.

## Two-tier function descriptions (short for all, deep on demand)

**Problem.** Every function gets the full description today, `behaviour`
included: rules, constants, inputs → outputs. That is the expensive part of the
reply: the WndProc of moveme.exe took 735 output tokens (9B model, 1135 prompt
tokens). On local hardware run time grows roughly with output tokens; in the
cloud output tokens are the priciest. Most functions in a legacy binary are
plumbing (UI, runtime, CRT, library code) whose rules nobody will port, and
`legacy-re` only drills into a few modules anyway.

**Proposal.**
- Tier 1, in `analyze_binary`, for all functions: `purpose`, `inferred_name`,
  `apis`, `confidence`, `side_effects`. Enough for naming, the call-graph
  hierarchy, search and `asm_tree`. Own budget, e.g. `token_limits.asm_function_short`
  (~256).
- Tier 2, new tool `describe_deep(unit_id)` (or `asm_tree(..., deep=true)`):
  fetches the pseudocode again through REA and asks for `behaviour` plus a
  spec-shaped answer (inputs, outputs, rules with constants and addresses,
  errors, state/globals). Stored on the unit (`key_patterns` or a new
  `spec` column) and marked `deep`, so it runs once per function. Budget
  `token_limits.asm_function` (1024).
- Optional subtree mode: `describe_deep(unit_id, recursive=true)` for every
  function under a group, i.e. one module to be rewritten.
- Skill `legacy-re` step 4/5: call `describe_deep` on the functions chosen for
  a spec instead of reading every pseudocode by hand.

**Expected gain (estimate, not measured).** Tier-1 replies around 150–250
tokens vs 400–800 today → 3–4× less output across the whole binary; deep cost
paid only for the modules being rewritten.

**Before building: measure.** Index one large binary (thousands of functions)
with today's pipeline and record the per-function prompt/completion tokens
(`usage`), wall time, and how often a reply hits `finish=length`. Use the
numbers to set the tier-1 budget and to confirm the gain. Without that, the
3–4× is a guess.

**Risks.**
- Tier-1 group summaries lose detail that `behaviour` fed into them. Check the
  root summary on the same binary in both modes.
- Re-fetching pseudocode for tier 2 needs the binary to be openable again (same
  path, REA running). Store `sha256` with the unit and refuse on mismatch.

# Weak-model system-prompt rework (S5)

Status: **mechanism shipped, default OFF; content is eval-gated.** The overlay
framework and a first weak-tier overlay exist in code; enabling them for any
model is a measured decision, not a merge. This doc is the standing protocol,
not a one-time task — grep discipline is a prompt+affordance property, not a
model property, and the only way to know an overlay helps a given model is to
run it through the S2 judge.

## What shipped

- `prompts/overlays/tier_local.txt` — the weak-model overlay: short,
  imperative, example-driven; enforces grep-before-search, read-before-edit,
  small diffs, verify-after-edit, stop-when-done. Comment lines (`#`) are
  dev notes stripped before the model sees it.
- `core/prompts._tier_overlay()` / `_resolve_default_tier()` — append the
  overlay for the running model's cost tier, gated by
  `agent.tier_prompt_overlay`.
- Config `agent.tier_prompt_overlay: "off" | "auto" | "local"`, **default
  `off`** — a fresh config changes nothing.
  - `auto` — append the overlay matching the default entry's tier
    (`entry_tier` → `local`/`free`/`paid`/`bundled`); only `local` has an
    overlay today.
  - `local` — force the weak overlay regardless of tier (for A/B testing on a
    strong model, or when the operator knows their endpoint is weak but
    mis-tiered).

## Why an overlay, not a rewrite of `system.txt`

The base prompt is tuned for capable models and is the cache-stable prefix
(S3). Rewriting it risks regressing strong-model behavior to help weak ones,
and burns the shared prompt cache. An *appended* overlay:
- keeps the base prompt byte-stable (strong models unaffected when overlay
  off);
- rides the existing `prompt_compiler` A/B pipeline as its own entry
  (`overlays/tier_local.txt`), so the compiled/compressed variant is measured
  separately;
- lands at the end of the stable prefix — per S3 it must stay
  per-session-stable (don't toggle mid-session) or prompt caching is lost.

## The overlay content, and why each rule

Weak local models exhibit four dominant failure modes (observed, to be
re-confirmed per model via evals):
1. **Over-use semantic search** for things grep answers exactly (symbols,
   error strings) → slow, wrong file. Rule 1 forces grep/graph first.
2. **Edit without reading** → hallucinated context, broken edits. Rule 2.
3. **Oversized edits** — reformat/rename/"clean up" beyond the task → exactly
   what the S2 judge penalizes as drive-by churn. Rule 3.
4. **Skip verification** → declare done on an unrun change. Rule 4 (pairs
   with W1's edit post-check feedback: the model is told to react to the
   post-check error the tool already surfaces).

The two worked examples (good vs bad) are deliberate: weak models follow
concrete tool-call traces far better than abstract instructions.

## Measurement protocol (required before enabling)

This is the gate. An overlay ships enabled for a model only after it wins on
that model's evals.

1. Baseline the target weak model with the overlay **off**:
   `evals/run.py --judge --json baseline_<model>.json` (16 tasks; the S2
   judge should be pinned to a *strong* model via `[model_roles] judge`, so
   the weak model under test never grades itself).
2. Turn the overlay on (`agent.tier_prompt_overlay = "local"`) and re-run
   with `--baseline baseline_<model>.json`.
3. **Keep only wins.** Enable for that model/tier iff: no mechanical
   pass→fail flips AND mean judged score does not drop (ideally rises,
   especially on `extract-duplicate-logic`, `add-quiet-flag`, and the
   fix-* tasks where drive-by churn and skipped verification show up).
4. Iterate on the overlay *text* against the same harness. Each edit is a new
   `prompt_compiler` entry; spot-check that the compiled variant keeps the
   imperative structure (S2 judge on a couple of tasks — compression can
   flatten "do X, then Y" into prose the weak model ignores).
5. Add the S3 long-context tasks when they exist — the "stop when done" and
   "small diff" rules matter most under context pressure.

## Per-tier expansion (future)

`_TIER_OVERLAY_FILE` maps tier→file; only `local` is populated. If evals show
a distinct `free`-tier failure profile (fast cloud models that are capable
but terse), add `overlays/tier_free.txt` the same way — mechanism is already
there. Do **not** add a `paid` overlay speculatively; the base prompt targets
that tier already.

## Handoff

The framework is mechanical and done. The remaining work is the eval loop
above, per model, run on live weak endpoints (not doable offline). It is
explicitly ongoing: re-validate when the base prompt, the overlay text, or
the target model changes.

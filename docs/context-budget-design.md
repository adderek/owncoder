# Context budget architecture (S3)

Status: **design decided**; implementation is PLAN_NORMAL-class and MUST be
validated with the measurement loop (`evals/run.py --judge --baseline`)
before and after each step — a wrong eviction priority degrades every turn
with no test failure, so eval evidence gates every merge here.

## What exists today (reuse, don't rebuild)

- `core/context_budget.py` — usage tiers (NORMAL→CRITICAL) with transition
  callbacks. Good trigger machinery; knows nothing about *what* to shrink.
- `memory/compactor.py` — whole-history compaction, `keep_last=4` tail,
  goal-drift check on the compacted summary.
- `agent/tool_compactor.py` + `core/output_store.py` — per-tool-result
  compaction and store-with-retrieval truncation.
- `core/cache_tracker.py` — per-(base_url, model) prompt-cache-TTL tracking.
- System-message stack built once in `core/agent.py.__init__`: hard rules →
  behavioral rules (hard, soft) → system prompt → project doc → user context
  → skill index.

The gap: these act independently. RAG excerpts, memory notes, transient
notes, similar-session injections, and tool results all compete for the same
window blind, and nothing orders messages for cache stability on purpose.

## Budget classes

Every message gets a class at append time (tag in message metadata dict, not
content):

| class        | contents                                                    | volatility |
|--------------|-------------------------------------------------------------|------------|
| `pinned`     | hard rules, system prompt, project doc, user context, skills | never changes in-session |
| `memory`     | behavioral soft rules, similar-session inject, memory notes | changes rarely |
| `rag`        | semantic-search excerpts injected as context                | per-turn |
| `tool`       | tool results in history                                     | per-call |
| `tail`       | recent conversation turns (user/assistant)                  | always kept |
| `note`       | transient notes, loop-guard notes, hook post_tool notes     | short-lived |

## Per-class budgets, per model tier

Fractions of the model's context window, resolved from the entry's tier
(`MODE_TIERS` / `entry_tier` already classify local | free | paid):

| class    | local | free | paid |
|----------|-------|------|------|
| `pinned` | ≤15%  | ≤10% | ≤8%  |
| `memory` | ≤5%   | ≤5%  | ≤5%  |
| `rag`    | ≤8%   | ≤12% | ≤15% |
| `tool`   | ≤25%  | ≤35% | ≤40% |
| `note`   | ≤3%   | ≤3%  | ≤3%  |
| `tail`   | remainder (floor 30%)                |

Rationale for the tier skew: weak local models degrade sharply with long
irrelevant context (the S5 observation), so they get *tighter* `rag` and
`tool` budgets and proportionally more `tail`; strong paid models extract
value from bigger excerpt/tool budgets. These starting numbers are guesses
**by design** — the implementation must land them as config
(`[context_budget.classes]`) and the tuning happens against W4/S2 judged
scores, not by editing this table in place.

Budget enforcement is *at append time*: a RAG injection that would exceed
the `rag` budget is trimmed (drop lowest-relevance excerpts first — the
relevance floor machinery from the notes-dedup work already ranks them). A
tool result over the `tool` class headroom goes through the existing
output_store truncation with a tighter cap rather than being appended whole.

## Ordering: cache-stable prefix vs volatile tail

Rule: **message order is `pinned` → `memory` → (conversation: everything
else in chronological order)**. Never insert anything above the
chronological section mid-session:

- Transient notes, similar-session injections, RAG excerpts, loop-guard
  notes: always *append* (or attach to the next user turn), never splice
  before existing messages. A single splice above N cached tokens
  invalidates the provider prefix cache for every following request —
  worse than the note is worth, especially on 5-min-TTL cloud caches
  (`cache_tracker` already knows the TTL; the assembler should log a
  cache-burn warning if any code path ever violates ordering).
- prompt_compiler A/B arms: the compiled system-prompt variant *replaces*
  the system prompt inside `pinned` at session start and then must be
  byte-stable for the whole session. Switching arms mid-session is
  forbidden (it would look like a "better prompt" in isolation and lose
  more to cache misses than it gains — measure arms across sessions, not
  within one).
- Compaction rewrites history wholesale — that's an accepted full cache
  burn, already paid today. The design changes nothing there except that
  the rebuilt history preserves the same class ordering.

## Eviction: which class shrinks first

Wired into the existing `ContextBudget` tier callbacks, replacing blind
"compact when full":

1. **WARNING (80%)**: stop injecting new `rag` and `note` content (skip,
   log); existing messages untouched. Cheapest lever, zero cache cost for
   the stable prefix.
2. **COMPACT (85%)**: evict `note` class entirely (transient by contract);
   re-truncate `tool`-class messages older than the last `keep_last` turns
   down to their output_store stubs (full text stays retrievable via
   `retrieve_output` — nothing is lost, only demoted). This edits history →
   cache burn, but it happens exactly when the alternative is compaction,
   which burns cache anyway.
3. **DANGER (90%)**: run the full compactor (as today), which now operates
   on a history whose fat was already trimmed — summaries stay focused on
   conversation, not on re-summarizing tool dumps. Goal-drift check runs as
   today on the result.
4. **CRITICAL (95%)**: emergency truncation (as today), unchanged.

Why this order: `note` is short-lived by definition; `rag` is re-derivable
on demand (the model can re-search); `tool` full text is retrievable from
the store so demotion is lossless; `tail` and `pinned` are evicted last
because they're the request identity and the behavior contract. `memory`
sits with `pinned` — it's small and cheap by budget, not worth a rung.

Interaction with goal-drift: eviction stages 1-2 never touch user/assistant
turns, so the drift check's inputs (original request vs q_view) are
unaffected; only stage 3 involves it, exactly as today.

## Measurement protocol (blocking gate for implementation)

1. Before any change: `evals/run.py --judge --json baseline.json` on the
   16-task set, once per model tier that matters (at minimum: one local
   weak model, one strong model).
2. Each implementation step (tagging, budgets, ordering, eviction rungs)
   lands separately and re-runs with `--baseline baseline.json`. Mechanical
   flip or judged drop >2 blocks the merge.
3. Add 2-3 long-context eval tasks (fixture with a large distractor file +
   a task whose answer needs the tail) so eviction actually fires during
   evals — current tasks are too small to exercise the pressure path.
   (Task-count growth here is fine now that S2's judge exists.)

## Implementation hand-off (mechanical, in order)

1. Class tagging at append sites (`core/agent.py`, `core/turn.py`,
   RAG/note/memory inject paths) + `[context_budget.classes]` config with
   the table above as defaults. No behavior change yet — tags only.
2. Append-time budget enforcement (rag trim, tool tighter cap) behind a
   config flag, default off; eval before/after, flip default on evidence.
3. Ordering audit: find every mid-history splice (grep for insert(…) on
   self.messages), convert to append-or-attach; add the cache-burn warning
   log to the assembler.
4. Eviction rungs 1-2 wired into ContextBudget callbacks; rung 3/4 stay as
   today. Eval gate again.
5. Long-context eval tasks (step 3 of the protocol).

# Token watch — scenarios + counter-actions on logprobs

`core/token_watch.py`. Reads [token_stats](token_stats.md) rows of every model call, names situations, acts. Needs token_stats capture (local/LAN endpoint, server with logprobs + tools + stream).

## Config

```toml
[token_watch]
enabled = true
derail = "retry"      # off|mark|retry
collapse = "retry"    # off|mark|retry
tool_doubt = "note"   # off|mark|note
claim = "mark"        # off|mark|note   (note = one extra round per answer)
tail = "mark"
drift = "mark"
no_probs = "mark"
max_retries = 1       # per turn
max_notes = 1         # per turn, per scenario
```
Thresholds: see `TokenWatchConfig` (`config/models.py`).

## Calibration — layered, online

`core/token_watch_calib.py`. Thresholds (derail_entropy, derail_p) + drift baseline per **configuration fingerprint** = model + endpoint + top_logprobs + think_level + prompts/*.txt hash + agent commit.

| layer | data | used for |
|---|---|---|
| 0 prior | `data/token_watch_priors.json` (fnmatch on model), else global `[token_watch]`; weight `prior_weight`=3 sessions | starting thresholds, baseline ppl |
| 1 slow | OTHER finished sessions; per-session decay `session_decay`=0.95, one session ≤ weight 1 | thresholds = weighted blend prior+slow; baseline ppl |
| 2 fast | current session, last `fast_window`=30 calls, in memory | session_drift vs slow; may LOOSEN (≤ `fast_loosen_cap`) only while slow weight < `thin_slow_weight` |

- `per_model` overrides beat all layers.
- Current session never sets its own thresholds (step-by-step degradation can't self-normalise).
- Session calls → `pending[session]`; merged into slow when quiet `pending_stale_minutes`=30 (by any process — crashed sessions too). Skipped: < `session_min_calls`, alarmed calls (never pended), sessions flagged `drifted` (unless `/tokwatch accept`).
- New fingerprint inherits parent slow × `inherit_weight`=0.3 → `regime_change` event (info) listing changed components.
- `session_drift`: fast window mean ppl > slow baseline × `session_drift_ratio`=1.3 (per-call `drift` stays 1.8×). Reported once, again only if 20% worse. Pearson r(ctx tokens, ppl) ≥ 0.6 → hint "compaction may help".
- Store `~/.config/agent/token_watch/calibration.json`, exclusive flock, read-merge-write (multi-process safe, tested with 2 processes). v1 `token_watch_calibration.json` migrated as `legacy-*` fingerprint.
- Resolution limit: top_logprobs=5 → flat top-5 nH≈0.90; hesitant models need larger k or llama.cpp full entropy (ask 1).
- Cost: ~7 ms/call (300 tok), ~95 ms (4000 tok), run in a worker thread after the stream.

```toml
[token_watch]
prior_weight = 3.0
session_decay = 0.95
pending_stale_minutes = 30
session_drift_ratio = 1.3
[token_watch.per_model."ornith15-9B-mtp"]
derail_entropy = 0.7
```

`/tokwatch` — fingerprints, layers, effective values · `/tokwatch accept` — this session is normal (merge now, even if drifted) · `/tokwatch reset [<model>]` · `/tokwatch diag [days]`.

## Diagnostics — for later review / refactor

`core/token_watch_diag.py` → `~/.config/agent/token_watch/diag/YYYY-MM.jsonl` (0600, `diag_max_mb`=50/month, `diag=false` off). Schema `v=1`:

- `call`: session, turn, model, fingerprint, ctx_tokens, metrics {n, windows, ppl, h_max, p_min, mean_H, min_p, low_p, tool_ppl}, thresholds {values, source prior|slow|fast-loosened|override|default, prior name, slow_weight/sessions, fast_calls, loosened, baseline_ppl}, events {kind, action, value, threshold, cut, text}, tripped, learned / skip reason.
- `outcome`: after retry/note — kind, action, cleared (scenario absent on next call), next_events, next_tool_calls, temperature.
- `merge` (incl. `skipped: session drifted`), `regime_change` (changed components).

Review checklist (`/tokwatch diag 90` first, raw JSONL for replay):
1. Event rate per model — claim/tool_doubt above ~5% of calls = thresholds too loose for that model.
2. `outcome derail→retry` cleared rate — low = retry useless → switch derail to mark or change temperature.
3. `outcome claim→note` / `tool_doubt→note` — cleared + next_tool_calls>0 = model verified.
4. Threshold trajectory per fingerprint — oscillation = decay too fast; flat after regime_change = inherit too strong.
5. session_drift with ctx_corr — if mostly context-driven, hook compaction instead of mark.
6. Replay: feed `call` metrics through new thresholds offline before changing defaults.

## Scenarios

| kind | signal | meaning | default action |
|---|---|---|---|
| derail | 48-tok window: mean H/ln(k+1) ≥ 0.6 AND mean p ≤ 0.5 | lost thread, gibberish, KV/context breakdown | retry at T=0.2; cut mid-stream |
| collapse | 64-tok window: mean p ≥ 0.97, distinct tokens ≤ 15% | degenerate loop, earlier than text repetition guard | retry at T=0.9; cut mid-stream |
| tool_doubt | ≥3 tokens p<0.15 (or outside top-k) within 8, kind `t` | guessed tool name / param / path | note after tool results |
| claim | same cluster, kind `c` | invented specifics (address, URL, number) | mark; `note` asks model to verify or hedge |
| tail | ≥5% tokens sampled outside top-k | sampler too hot | mark |
| drift | call ppl > 1.8 × baseline (slow layer / prior) | one call far off | mark |
| session_drift | last 8 calls mean ppl > 1.3 × baseline | persistent shift this session (context, server) | mark |
| regime_change | configuration fingerprint changed | calibration restarts from parent | mark (info) |
| no_probs | ≥20% new-format rows without alternatives | server fake p=1 (old speculative path) | mark |

Calibration 2026-10-03, 82 recorded calls (ornith10-35B, -iq4nl): healthy 48-tok windows normalised H ≤ 0.43, mean p ≥ 0.69. Replay: 2 events total, both `claim`, both real fabrications (`Shell, ul. Kijowska 17, Kraków`; `motoforum.pl, forum.oil-…`). Single low-p tokens in prose = word choice, not flagged. Reasoning (`r`) not cluster-checked.

Normalised entropy: H is lower bound over top-k + tail bucket, ceiling ln(k+1).

## Actions

- **mark** — `token_watch` phase row (orange), ⚠ on ◔ fold summary, ⚠ kinds on the turn's work fold (red border for alert), event list + outlined token spans inside the fold. Log warning.
- **note** — harness `_injected("token watch", …)` message. tool_doubt: after the round's tool results (keeps tool_calls paired). claim: answer kept in history, note asks to verify/hedge, next answer replaces it.
- **retry** — call discarded before history, re-run with `temperature_override(allow_lower=True)`. Streaming `LiveWatch` cuts the stream when the window trips (only while retries remain → `watch_cut_ok`).

## Data

- Event: `{kind, severity: info|warn|alert, action, start, end, value, threshold, text, detail, [temperature, spans, texts, cut]}`; start/end index kept `tokens`.
- Live: `record["watch"]` in the `tokstats` SSE event. Persisted: same key inside `tokstats.jsonl` record + `tokwatch.jsonl` (`{turn, tokstats_ref, model, events}`); transcript adds `tokwatch: {ref: [...]}` so closed replayed folds show ⚠.
- Drift baseline: slow-layer ppl of other sessions (prior `ppl` until then).

## Limits

- Text-form tool calls (`call:fn{...}` in content) score as `c` → reach `claim`, not `tool_doubt`.
- Confident-wrong = invisible (p high). Needs external check (tool result vs claim).
- Thresholds per model + KV quant; recalibrate by replaying a session's `tokstats.jsonl` records through `token_watch.evaluate`.
- Retry discards streamed text already shown in UI (stays in work fold as intermediate).

## Asks for llama.cpp fork (would sharpen detectors)

1. **Full-vocab entropy per token** — `entropy` field on each `logprobs.content` entry, computed on device from pre-sampling softmax (already materialised for `n_probs`). Replaces lower-bound H → derail threshold model-independent of k. Scalar only.
2. **Draft flag per token** — `draft: true|false` on rows from accepted speculative/MTP tokens. Lets owncoder separate verify-path numerics from normal decode, and correlate acceptance with confidence.
3. **EOS probability per token** (`p_eos`) — model wanting to stop mid tool-call/JSON = template/format fight; also early signal for runaway generation.
4. **Prompt logprobs (P3, docs/token_stats.md)** — surprise of tool *output* tokens: injected instructions / garbled file reads show as high-surprise spans in input. Detector `input_surprise` waits on this.
5. **Per-request KV stats in final chunk** — `n_ctx_used`, `cache_n` (reused), KV type. Drift detector can then split "long context" vs "quantised KV" causes.
6. **Sampler state echo** — effective temperature/top-p/min-p actually used per request (after server defaults + overrides) in final chunk; verifies a retry's temperature took effect.

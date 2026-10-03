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

## Per-model calibration

`core/token_watch_calib.py`. Each model has own normal (size, quant, KV type) → thresholds per model.

- Profile per model name (`config.llm.model`) in `~/.config/agent/token_watch_calibration.json` (user-global: model property, not project). Learned only from calls with no derail/collapse/no_probs/drift event.
- Stored: histograms (50 bins) of 48-tok window mean normalised entropy + mean p (every 16 rows), EWMA call ppl (drift baseline, persisted).
- Precedence: `per_model` override > learned (after `calib_min_calls`=20) > global.
- Learned: `derail_entropy = clamp(q99.5(H) + calib_margin, 0.75×global, 0.95)`; `derail_p = clamp(q0.5(p) − bin − margin, 0.2, 1.25×global)`.
- Resolution limit: with `top_logprobs=5` a flat top-5 is nH≈0.90 — a very hesitant model's learned ceiling can reach it. Raise `top_logprobs` (or llama.cpp full entropy, ask 1) for hesitant models.
- `/tokwatch` shows profiles + effective thresholds; `/tokwatch reset [<model>]` drops them. Events carry `calib: default|learned|override`.

```toml
[token_watch]
calibrate = true
calib_min_calls = 20
calib_margin = 0.1
[token_watch.per_model."ornith15-9B-mtp"]
derail_entropy = 0.7
cluster_p = 0.1
```
Overridable: derail_entropy, derail_p, collapse_p, collapse_distinct, cluster_p, cluster_min, cluster_span, tail_share, drift_ratio, derail_temperature, collapse_temperature.

## Scenarios

| kind | signal | meaning | default action |
|---|---|---|---|
| derail | 48-tok window: mean H/ln(k+1) ≥ 0.6 AND mean p ≤ 0.5 | lost thread, gibberish, KV/context breakdown | retry at T=0.2; cut mid-stream |
| collapse | 64-tok window: mean p ≥ 0.97, distinct tokens ≤ 15% | degenerate loop, earlier than text repetition guard | retry at T=0.9; cut mid-stream |
| tool_doubt | ≥3 tokens p<0.15 (or outside top-k) within 8, kind `t` | guessed tool name / param / path | note after tool results |
| claim | same cluster, kind `c` | invented specifics (address, URL, number) | mark; `note` asks model to verify or hedge |
| tail | ≥5% tokens sampled outside top-k | sampler too hot | mark |
| drift | call ppl > 1.8 × model EWMA baseline (≥3 samples) | context degradation → compaction / KV quant | mark |
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
- Drift baseline: per-model profile `ppl` (persisted); alarmed calls not folded in.

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

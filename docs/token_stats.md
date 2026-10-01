# Token confidence overlay

Per-token logprob capture for model output, shown in the HTTP UI. Off by default.

## Use

```toml
[token_stats]
enabled = true
top_logprobs = 5          # 1..20; entropy/margin need >= 2
local_only = true         # only loopback/private endpoints get asked
speculative_type = "none" # per-request llama.cpp override while capturing; "" = leave server setting
max_tokens = 4000         # rows kept per model call (newest)
```

HTTP UI: each model call gets a `◔ token confidence` fold in the work journal.
Summary: token count, perplexity, min p, count below p=0.5, mean entropy, tool-arg ppl.
Open → raw tokens shaded by metric (improbability / entropy / thin margin);
wavy underline = sampled outside top-k; hover = exact numbers. Replayed sessions
fetch rows lazily (`GET /api/tokstats?id=<sid>&seq=<ref>`).

## Data

- Row: `[text, logprob, entropy, margin, rank, kind]`, kind `c|r|t` (content, reasoning, native tool args).
- Entropy = top-k + one tail bucket → lower bound.
- Pre-sampling distribution (llama.cpp default): model belief, not what temperature did.
- Raw server tokens, before `_clean_output`; no alignment with rendered markdown.
- Persisted: session side-log `tokstats.jsonl`, message key `_tokstats_ref` (stripped before API).
- Live: `core/token_stats.sink` ContextVar → SSE event `tokstats`. Only the primary HTTP UI sets it (not the sidecar, not IPC-worker mode).
- Streaming path only: `agent run` (non-streaming, no `on_token`) captures nothing.

## Interpretation limits

- Low p / high entropy = model hesitation. Not truth: confident-wrong facts score high.
- Quantized KV (TurboQuant) shifts absolute logprobs → thresholds per model+KV type.
- Confident + invalid tool call = suspect (hallucination / template issue). Detector not built yet; needs this data first.

## Server gaps (llama.cpp, checked upstream a8681a0 + domvox-turboquant 89231e3b4)

1. `tools/server/server-common.cpp` (`Handle "logprobs" field`): throws
   `logprobs is not supported with tools + stream`. owncoder always streams with
   tools → first request 400s, owncoder disables capture for that endpoint until
   restart (warning logged). **Blocker for real use.**
2. `tools/server/server-context.cpp` speculative accept loop: accepted draft
   tokens get `prob = 1.0f` and `// TODO: set result.probs` → no rows / fake
   confidence. Mitigated by `speculative_type = "none"` (fork commit 901f0234f
   makes `speculative.type` per-request).
3. Prompt tokens: logits only for last prompt token (`batch.set_output(size-1)`),
   `echo` refused → no input-side overlay possible.

### Measured 2026-10-01 (fork 89231e3b4, CPU build, Qwen2.5-0.5B)

- stream, no tools: OK; rows = completion_tokens − 1 (stop token has no row).
- stream + tools: `400 logprobs is not supported with tools + stream` — owncoder falls back, warns once, turn OK.
- non-stream + tools: OK, rows == completion_tokens, raw `<tool_call>` tokens included.
- `speculative.type` "none" and "bogus" both 200 on a server started without speculative → field not validated / ignored there.
- owncoder vs mock emulating P1: rows for reasoning + tool args captured, side-log written, UI fold renders + lazy replay works.

### Patch spec for the fork

P1 — logprobs with tools + stream
- Remove the guard. Verify per-chunk `choices[0].logprobs.content` is emitted for
  chunks whose delta is `tool_calls` / `reasoning_content`, and that tokens held back
  by the tool-call / reasoning parser are flushed with their probs (no drops, no dupes:
  total logprob rows == completion_tokens).
- Test: streamed chat with tools + `logprobs:true, top_logprobs:5`, model emits a tool call;
  assert row count == usage.completion_tokens.

P2 — probs for accepted speculative tokens
- In the accept loop, fill `result.probs` / `result.prob` from the target model's
  verification logits for each accepted position (already computed for acceptance; no
  extra decode). Respect `post_sampling_probs`.
- Test: same prompt, spec on vs off, temperature 0 → identical tokens and logprobs within
  KV-quant tolerance.

P3 — prompt logprobs (optional, later)
- Request field `prompt_logprobs: k` (0 = off). When set, mark every prompt token in the
  batch as output; after decode compute, per position i, logprob of token i+1, top-k,
  entropy. Do log-softmax + gather on device; return scalars only (full logits are
  n_vocab×4 B per token).
- Only tokens actually decoded are scored (prompt-cache hit → only the new suffix). Return
  absolute positions so the client can map rows.
- Response: final chunk / non-stream body `prompt_logprobs: [{pos, token, logprob, top_logprobs, entropy}]`.
- Cost to report: prefill slowdown %, memory with `prompt_logprobs` on vs off.

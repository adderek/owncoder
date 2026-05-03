# Output cleaning & reproducibility — step-by-step plan

Goal: fix `<|imend>` / reasoning-only / pipe-leak regressions with testable checkpoints.

## Steps

### Step 1: Add `_CHATML_TOKEN_RE` to `_clean_output`

Strips `<|imend>`, `<|imendend>`, `<|im_end|>`, etc. Produced by Gemma 4, Qwen 3.5.

**Files**: `core/streaming.py`, `tests/unit/test_streaming.py`

**Change**:
- Add `_CHATML_TOKEN_RE = re.compile(r"<\|[^>]*>")` constant
- Add `text = _CHATML_TOKEN_RE.sub("", text)` in `_clean_output` after `_LEAK_RE` line

**Test**: `_clean_output("Done.<|imend>") == "Done."`, `_clean_output("Done.<|imendend>") == "Done."`, `_clean_output("Done.<|im_end|>") == "Done."`

**Verify**: `pytest tests/unit/test_streaming.py -k "test_chatml_tokens_stripped or test_channel_tokens_stripped or test_think_blocks_stripped" -x`

---

### Step 2: Add reasoning-content fallback

When model produces only `reasoning_content` (no `delta.content`), use reasoning as fallback content instead of returning "" (which shows "(done)" in UI).

**Files**: `core/turn.py`, `tests/unit/test_streaming.py`

**Change**:
- In `core/turn.py`, after empty-content check (line ~474), add:
  ```python
  if turn_reasoning.strip():
      content = turn_reasoning
  ```
- No retry logic — retry caused `<|imend>` regression

**Test**: existing `test_channel_tokens_stripped` in `test_streaming.py` ensures stream cleaning still works. Need to verify entire turn logic handles empty content + reasoning gracefully.

**Verify**: `pytest -x`

---

### Step 3: Add token counter to phase detail

Show token count during "generating" phase so user sees progress even during tool call construction (no visible text tokens).

**Files**: `core/streaming.py`, `core/turn.py`

**Change**:
- `_stream_response`: add `on_stream_progress` param, `_gen_tokens`/`_reasoning_tokens` counters, increment on content/reasoning/tool-arg deltas, periodic 0.2s throttle callback
- `turn.py`: define `_on_progress` callback that calls `_phase("generating", f"iter X · {total} tok")`, pass it to `_stream_response`

**Verify**: `pytest -x`. Manual: observe "iter 1/50 · 42 tok" in TUI phase detail during generation.

---

### Step 4: Add seed to config for deterministic reproduction

Allows setting `seed = 42` + `temperature = 0.0` for reproducible model output.

**Files**: `config/models.py`, `config/loader.py`, `core/prompts.py`

**Change**:
- Add `seed: int | None = None` to `LLmConfig` and `ModelEntry`
- Map `AGENT_LLM_SEED` env var
- Propagate from `ModelEntry` → `LLmConfig` in loader
- Pass `seed` kwarg in `_build_call_kwargs` when not None

**Verify**: `pytest -x`. Set `AGENT_LLM_SEED=42` env var, run two identical prompts, check model output matches.

---

### Step 5 (if needed): Add `re.IGNORECASE` fix for role-word cleaning

When `re.IGNORECASE` is global, `(?=[A-Z])` lookahead matches lowercase too, stripping legitimate text.

**Files**: `core/streaming.py`, `tests/unit/test_streaming.py`

**Change**:
- Replace `flags=re.IGNORECASE` with `(?i:...)` on role words only
- `(?=[A-Z])` stays strict uppercase

**Test**: `_clean_output("thoughtLet me fix this") == "Let me fix this"`

**Verify**: `pytest tests/unit/test_streaming.py::TestCleanOutput::test_role_words_mid_text_handled -x`

---

### Step 6 (if needed): Add leading pipe strip

Gemma 4 chat template leaks `| ` before response.

**Files**: `core/streaming.py`, `tests/unit/test_streaming.py`

**Change**:
- Add `text = re.sub(r"^\|\s*", "", text)` in `_clean_output`

**Test**: `_clean_output("| Add login button.") == "Add login button."`

**Verify**: `pytest tests/unit/test_streaming.py -k pipe -x`

---

## Test procedure per step

1. Apply step changes
2. `cd agent && .venv/bin/pytest --no-header -q`
3. If pass, commit. If fail, fix before proceeding.
4. Push to refactor/split-modules branch after each commit.

## Rollback

`git checkout -- <file>` for any step that breaks tests. Don't accumulate uncommitted changes across multiple steps.

## Key insight (from previous attempts)

Empty response retry CAUSES `<|imend>` regression. Never add retry with "Please respond." — the model naturally stops when done. If it produces reasoning but no content, use reasoning fallback. If it produces nothing, return empty — UI shows "(done)" which is honest.

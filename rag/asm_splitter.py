from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import AsmAnalysisConfig, LLMConfig

logger = logging.getLogger(__name__)

_SPLIT_PROMPT = """\
You are analyzing disassembled binary code. Identify where new logical units begin.

A logical unit is typically a function, interrupt handler, jump trampoline, or data table.
Indicators of a new unit starting:
- Stack frame setup (push rbp/mov rbp,rsp or equivalent)
- First instruction after an unconditional jmp/ret/retq that ends the previous unit
- Alignment padding (series of nop or int3/cc bytes) followed by code
- Explicit label ending in a colon (rare in stripped binaries but possible)
- Change from code-like to data-like content or vice versa

Lines {start_line}–{end_line} of {path}:
```
{window_content}
```

Reply with a JSON array of line numbers where a new logical unit STARTS (use the original \
line numbers, not window-relative numbers).
Only include lines that are confident boundaries. Example: [42, 107, 203]
If the entire window is one unit, reply: []
"""

_RETRY_PROMPT = """\
Your previous response could not be parsed as a JSON array of integers.
Please reply ONLY with a JSON array of line numbers, for example: [42, 107]
If the entire window is one unit, reply: []
"""


def _approx_tokens(text: str) -> int:
    return len(text) // 4


def _parse_boundary_response(text: str) -> list[int] | None:
    text = text.strip()
    # Try to find a JSON array in the response
    match = re.search(r"\[[\d,\s]*\]", text)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list) and all(isinstance(x, int) for x in result):
                return result
        except (json.JSONDecodeError, TypeError):
            pass
    # Direct parse attempt
    try:
        result = json.loads(text)
        if isinstance(result, list) and all(isinstance(x, (int, float)) for x in result):
            return [int(x) for x in result]
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _merge_proposals(proposals: list[int], num_lines: int) -> list[int]:
    """Cluster proposals within ±3 lines and take median; always include line 1."""
    if not proposals:
        return [1]
    proposals = sorted(set(proposals))
    clusters: list[list[int]] = []
    for p in proposals:
        if clusters and abs(p - clusters[-1][-1]) <= 3:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    boundaries = []
    for cluster in clusters:
        median = sorted(cluster)[len(cluster) // 2]
        boundaries.append(median)
    # Ensure we start at line 1
    if not boundaries or boundaries[0] != 1:
        boundaries = [1] + [b for b in boundaries if b != 1]
    return boundaries


class AsmLogicalSplitter:
    def __init__(self, llm_client, cfg: "AsmAnalysisConfig", llm_cfg: "LLMConfig") -> None:
        self._client = llm_client
        self._cfg = cfg
        self._llm_cfg = llm_cfg
        self._model = cfg.describer_model or llm_cfg.model

    def split(self, path: str, lines: list[str]) -> list[tuple[int, int]]:
        """Return list of (start_line, end_line) 1-indexed, non-overlapping, covering all lines."""
        num_lines = len(lines)
        if num_lines == 0:
            return []

        window_size_chars = self._cfg.splitter_ctx_tokens * 4
        overlap = self._cfg.splitter_overlap_lines

        all_proposals: list[int] = []
        window_start = 0  # 0-indexed

        while window_start < num_lines:
            # Build window
            window_lines = []
            char_count = 0
            for i in range(window_start, num_lines):
                line = lines[i]
                if char_count + len(line) > window_size_chars and window_lines:
                    break
                window_lines.append(line)
                char_count += len(line)

            window_end = window_start + len(window_lines) - 1  # 0-indexed inclusive
            start_1 = window_start + 1  # 1-indexed
            end_1 = window_end + 1

            window_content = "".join(
                f"{start_1 + i:6d}  {line}" for i, line in enumerate(window_lines)
            )

            prompt = _SPLIT_PROMPT.format(
                start_line=start_1,
                end_line=end_1,
                path=path,
                window_content=window_content,
            )

            boundaries = self._call_llm_for_boundaries(prompt, path, start_1, end_1)
            all_proposals.extend(boundaries)

            # Advance by window size minus overlap
            advance = len(window_lines) - overlap
            if advance <= 0:
                advance = len(window_lines)
            window_start += advance

        # Merge proposals
        merged = _merge_proposals(all_proposals, num_lines)

        # Validate and produce intervals
        intervals = self._proposals_to_intervals(merged, num_lines)

        # Validate coverage
        if not self._validate_intervals(intervals, num_lines):
            logger.warning("AsmLogicalSplitter: validation failed for %s, using fallback", path)
            return self._label_fallback(lines, path)

        return intervals

    def _call_llm_for_boundaries(
        self, prompt: str, path: str, start_1: int, end_1: int
    ) -> list[int]:
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(2):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=256,
                    temperature=0,
                )
                text = response.choices[0].message.content or ""
                parsed = _parse_boundary_response(text)
                if parsed is not None:
                    # Filter to valid line range
                    return [b for b in parsed if start_1 <= b <= end_1]
                # Retry
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": _RETRY_PROMPT})
            except Exception as e:
                logger.warning("AsmLogicalSplitter LLM call failed: %s", e)
                break
        # Treat window as single unit
        return [start_1]

    def _proposals_to_intervals(
        self, boundaries: list[int], num_lines: int
    ) -> list[tuple[int, int]]:
        intervals = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] - 1 if i + 1 < len(boundaries) else num_lines
            intervals.append((start, end))
        return intervals

    def _validate_intervals(self, intervals: list[tuple[int, int]], num_lines: int) -> bool:
        if not intervals:
            return False
        if intervals[0][0] != 1:
            return False
        if intervals[-1][1] != num_lines:
            return False
        for i in range(len(intervals) - 1):
            if intervals[i][1] + 1 != intervals[i + 1][0]:
                return False
        return True

    def _label_fallback(self, lines: list[str], path: str) -> list[tuple[int, int]]:
        """Fall back to label-based splitting."""
        import re as _re
        label_re = _re.compile(r"^\s*\w+:")
        boundaries = [1]
        for i, line in enumerate(lines, 1):
            if label_re.match(line) and i > 1:
                boundaries.append(i)
        return self._proposals_to_intervals(boundaries, len(lines))

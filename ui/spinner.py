"""Readline-mode spinner and status field helpers."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def _spinner_status_fields(server, status: str, elapsed: float) -> list[str]:
    """Return status fields in priority order (most meaningful first).

    Priority order (can be reordered by user preference in future):
      1. ctx%    — context fill % — most actionable; warns when near limit
      2. tokens  — used/total tokens — detail behind ctx%
      3. msgs    — conversation depth (message count)
      4. status  — current operation text (thinking / tool name)
      5. files   — number of indexed files (if RAG store present)
      6. chunks  — number of indexed chunks (if RAG store present)
      7. model   — model name (useful when switching models)
      8. time    — elapsed seconds for current operation
    """
    fields: list[str] = []

    if server is not None:
        info = server.get_llm_info()
        ctx = info["ctx_window"]
        used = server.token_estimate()

        if ctx:
            pct = int(used / ctx * 100)
            fields.append(f"ctx {pct}%")
            k_used = f"{used / 1000:.1f}k" if used >= 1000 else str(used)
            k_ctx = f"{ctx // 1000}k" if ctx >= 1000 else str(ctx)
            fields.append(f"{k_used}/{k_ctx}")

        msg_count = max(0, server.message_count() - 1)  # exclude system prompt
        fields.append(f"{msg_count} msg")

        s = server.stats()
        if s and s.get("calls", 0) > 0:

            def _k(n: int) -> str:
                return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

            fields.append(f"↑{_k(s['input_tokens'])}")
            fields.append(f"↓{_k(s['output_tokens'])}")
            if s.get("in_tps"):
                fields.append(f"{s['in_tps']:.0f}↑t/s")
            if s.get("out_tps"):
                fields.append(f"{s['out_tps']:.1f}↓t/s")
            if s.get("reasoning_tokens"):
                fields.append(f"think {_k(s['reasoning_tokens'])}")
            if s.get("tool_tokens"):
                fields.append(f"tool {_k(s['tool_tokens'])}")

    fields.append(status)

    if server is not None:
        store_s = server.get_store_stats()
        if store_s is not None:
            fields.append(f"{store_s['files']} files")
            fields.append(f"{store_s['chunks']} chunks")
        model = info["model"]
        if model:
            fields.append(model)

    fields.append(f"{elapsed:.1f}s")

    return fields


async def _run_spinner(status_ref: list[str], stop: asyncio.Event, server=None) -> None:
    import sys
    import shutil
    import time as _time

    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    ANIM_WIDTH = 2  # frame char + space
    SEP = "  "  # separator between fields
    i = 0
    t0 = _time.monotonic()
    while not stop.is_set():
        frame = frames[i % len(frames)]
        elapsed = _time.monotonic() - t0
        term_width = shutil.get_terminal_size((80, 24)).columns
        available = term_width - ANIM_WIDTH

        fields = _spinner_status_fields(server, status_ref[0], elapsed)

        parts: list[str] = []
        remaining = available
        for field in fields:
            needed = len(field) + (len(SEP) if parts else 0)
            if needed <= remaining:
                parts.append(field)
                remaining -= needed
            elif not parts:
                parts.append(field[:available])
                break

        info = SEP.join(parts)
        sys.stdout.write(f"\r\033[2m{frame} {info}\033[0m")
        sys.stdout.flush()
        i += 1
        await asyncio.sleep(0.08)
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

from agent.tools import register

if TYPE_CHECKING:
    from agent.config import Config
    from agent.data_provider import DataProviderProtocol

_config = None
_data_provider: "DataProviderProtocol | None" = None

# Shared interrupt flag so the UI can cancel a running analysis
_interrupt_flag: threading.Event = threading.Event()

# Optional UI callback set by the Textual layer. Called with (formatted_msg: str)
# from the executor thread — the UI layer must use call_from_thread internally.
_ui_progress_cb: "callable | None" = None


def set_ui_progress_cb(cb: "callable | None") -> None:
    """Register a callable(msg: str) that the UI calls to display progress in-app."""
    global _ui_progress_cb
    _ui_progress_cb = cb


def setup(config, data_provider) -> None:
    global _config, _data_provider
    _config = config
    _data_provider = data_provider


def get_interrupt_flag() -> threading.Event:
    return _interrupt_flag


@register(
    "analyze_asm",
    {
        "description": (
            "Analyze an assembly file with LLM-driven logical splitting and hierarchical "
            "summarization. Stores descriptions at multiple abstraction levels for semantic search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the assembly file to analyze",
                },
                "resume": {
                    "type": "boolean",
                    "description": "Resume interrupted analysis (skip already-described units)",
                },
                "force": {
                    "type": "boolean",
                    "description": "Re-analyze everything regardless of checksums",
                },
                "max_levels": {
                    "type": "integer",
                    "description": "Override max hierarchy levels for this run",
                },
            },
            "required": ["path"],
        },
    },
)
def analyze_asm(
    path: str,
    resume: bool = False,
    force: bool = False,
    max_levels: int | None = None,
) -> dict:
    asm_store = _data_provider.get_asm_store() if _data_provider else None
    embedder = _data_provider.get_embedder() if _data_provider else None
    if _config is None or asm_store is None:
        return {"error": "analyze_asm not initialized. Run 'agent init' first."}

    if not _config.asm.enabled:
        return {
            "error": (
                "Assembly analysis is disabled. "
                "Set asm_analysis.enabled = true in agent.toml or AGENT_ASM_ENABLED=1."
            )
        }

    p = Path(path)
    if not p.exists():
        return {"error": f"File not found: {path}"}

    from openai import OpenAI
    from agent.rag.asm_splitter import AsmLogicalSplitter
    from agent.rag.asm_describer import AsmDescriber
    from agent.rag.asm_pipeline import AsmAnalysisPipeline

    cfg = _config.asm
    if max_levels is not None:
        # Temporarily override max_levels without mutating config
        from dataclasses import replace

        cfg = replace(cfg, max_levels=max_levels)

    llm_client = OpenAI(
        base_url=_config.llm.base_url,
        api_key=_config.llm.api_key,
    )

    _start_time = time.time()
    _last_line_len = [0]

    def _progress_cb(event: str, data: dict) -> None:
        elapsed = time.time() - _start_time
        ts = f"[{elapsed:6.0f}s]"
        if event == "splitting_window":
            w, tw = data["window"], data["total_windows"]
            pct = 100 * w // tw
            cached_marker = " [cached]" if data.get("cached") else ""
            msg = f"{ts} Phase 1/4 Splitting:  window {w:>5}/{tw} ({pct:3d}%)  lines {data['start_line']:>8}-{data['end_line']:<8} of {data['total_lines']}{cached_marker}"
        elif event == "split_complete":
            suffix = " (cached)" if data.get("cached") else ""
            msg = f"{ts} Phase 1/4 complete:   {data['chunks']} chunks in {data['total_lines']} lines{suffix}"
        elif event == "embedding":
            i, tot = data["index"], data["total"]
            pct = 100 * i // tot
            msg = f"{ts} Phase 2/4 Embedding:  {i:>5}/{tot} ({pct:3d}%)"
        elif event == "described":
            i, tot = data.get("_index", "?"), data.get("_total", "?")
            pct = (100 * i // tot) if isinstance(i, int) and isinstance(tot, int) else 0
            name = data.get("inferred_name") or "?"
            msg = f"{ts} Phase 3/4 Describing: {i:>5}/{tot} ({pct:3d}%)  {name}"
        elif event == "grouped":
            gi, gt = data.get("_group_index", "?"), data.get("_group_total", "?")
            lvl = data.get("level", "?")
            pct = (100 * gi // gt) if isinstance(gi, int) and isinstance(gt, int) else 0
            msg = f"{ts} Phase 4/4 Hierarchy:  level {lvl}  group {gi:>4}/{gt} ({pct:3d}%)"
        else:
            return
        # Pad to overwrite previous line on stderr
        padded = msg.ljust(_last_line_len[0])
        _last_line_len[0] = len(msg)
        print(f"\r{padded}", end="", flush=True, file=sys.stderr)
        if event in ("split_complete",):
            print(file=sys.stderr)  # newline after phase-complete markers
        # Log key milestones so they appear in agent.log
        if event == "split_complete":
            logger.info(msg)
        elif event in ("described", "grouped") and (
            isinstance(data.get("_index"), int) and data["_index"] % 50 == 0
        ):
            logger.info(msg)
        # Push to Textual UI if registered
        if _ui_progress_cb is not None:
            try:
                _ui_progress_cb(msg)
            except Exception:
                pass

    splitter = AsmLogicalSplitter(
        llm_client, cfg, _config.llm, progress_cb=_progress_cb,
        token_limits=_config.token_limits,
    )
    describer = AsmDescriber(
        llm_client, cfg, _config.llm, token_limits=_config.token_limits,
    )

    _interrupt_flag.clear()

    pipeline = AsmAnalysisPipeline(
        asm_store=asm_store,
        embedder=embedder,
        splitter=splitter,
        describer=describer,
        cfg=cfg,
        interrupt_flag=_interrupt_flag,
        progress_cb=_progress_cb,
    )

    result = pipeline.analyze_file(str(p), force=force)
    print(file=sys.stderr)  # final newline after last \r progress line

    if result.get("cached"):
        result["message"] = (
            f"Already fully analyzed ({result.get('chunks', 0)} chunks, file unchanged)."
        )
    elif result.get("interrupted"):
        result["message"] = (
            f"Interrupted after {result.get('described', 0)} chunks described. "
            "Resume with analyze_asm(path=..., resume=True)."
        )
    else:
        result["message"] = (
            f"Analysis complete: {result.get('chunks', 0)} chunks, "
            f"{result.get('described', 0)} described, "
            f"{result.get('levels_built', 0)} hierarchy levels."
        )

    return result

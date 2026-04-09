from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from agent.tools import register

if TYPE_CHECKING:
    from agent.config import Config
    from agent.rag.asm_store import AsmStore
    from agent.rag.embedder import Embedder

_config = None
_asm_store = None
_embedder = None

# Shared interrupt flag so the UI can cancel a running analysis
_interrupt_flag: threading.Event = threading.Event()


def setup(config, asm_store, embedder) -> None:
    global _config, _asm_store, _embedder
    _config = config
    _asm_store = asm_store
    _embedder = embedder


def get_interrupt_flag() -> threading.Event:
    return _interrupt_flag


@register("analyze_asm", {
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
})
def analyze_asm(
    path: str,
    resume: bool = False,
    force: bool = False,
    max_levels: int | None = None,
) -> dict:
    if _config is None or _asm_store is None:
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

    splitter = AsmLogicalSplitter(llm_client, cfg, _config.llm)
    describer = AsmDescriber(llm_client, cfg, _config.llm)

    _interrupt_flag.clear()

    pipeline = AsmAnalysisPipeline(
        asm_store=_asm_store,
        embedder=_embedder,
        splitter=splitter,
        describer=describer,
        cfg=cfg,
        interrupt_flag=_interrupt_flag,
    )

    result = pipeline.analyze_file(str(p), force=force)

    if result.get("interrupted"):
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

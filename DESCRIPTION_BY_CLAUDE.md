# Project description (by Claude)

> Auto-generated overview of the **owncoder agent**. Companion to `README.md`.

## Summary

owncoder is a **local-first coding agent** with a strong security posture. It
runs against your own model server (llama.cpp / vLLM / ollama) or any
OpenAI-compatible API (deepseek, openai). It is built for heavy analysis of
low-structure languages — assembler in particular — that need indexing and
understanding before an agent can work on them.

## Why use it

- Keep code on your own machine; nothing leaves the box unless you allow it.
- Sandboxed, audited tool execution instead of unrestricted shell access.
- Deep code understanding via several indexing layers, not just one RAG store.
- Purpose-built for assembler and other languages tree-sitter alone can't model.

## Architecture

- **`core/`** — agent loop, turn handling, streaming, checkpoints, history ops.
- **`rag/`** — indexing & retrieval: tree-sitter chunker, embedder, vector store
  (sqlite-vec), background indexer, archive, assembler pipeline, summarizer.
- **`security/`** — seccomp sandbox, path grants, airgap, prompt-injection scan,
  redaction, taint tracking, audit log, SBOM, integrity/verify.
- **`memory/`** — facts store, Q&A log, session summaries, compaction, recall.
- **`tools/`** — file edit, git, shell, search, web search, graph, checkpoints,
  skills, security audit, assembler analysis, KB.
- **`ui/` + `ui_server/`** — Textual TUI plus simple/readline text-only modes.
- **`mcp/`** — Model Context Protocol support for external tools.
- **`config/`** — typed config (TOML + env overrides) for llm, embeddings, rag,
  asm analysis, summarization, tools, ui, kb.

## Code indexing / retrieval layers

Each layer is optional and used as needed. Note how each is built:

1. **RAG** — tree-sitter splits code into chunks; the embeddings model vectorizes
   them into sqlite-vec; hybrid (vector + keyword) search at query time
   (`.agent/index.db`).
2. **Archive** — pruned chunks kept with a TTL so old/deleted code stays
   searchable (`.agent/index-archive.db`).
3. **Summarization** — the LLM writes a terse description per chunk, then rolls
   them up into a multi-level summary pyramid (`.agent/summaries.db`).
4. **Assembler analysis** — same LLM describe-and-rollup pyramid (up to 6 levels),
   tuned for low-structure code tree-sitter can't model.
5. **Graph** — static dependency/call graph export (graphify); no model needed.
6. **KB** — optional external knowledge-base corpus.
7. **Memory / recall** — facts, Q&A log, and session history, distilled and
   compacted by the LLM.

## Prompt / skill / tool compilation

Static prompts and skills/tools are **compiled per model**: the prompt-compiler
(`prompt_compiler/`) compresses static prompt files for the active (model, api)
pair and caches the result, tracking per-variant success/error counts. Tool
results are compacted by the LLM (`tool_compactor.py`) before re-entering
context. Net effect: smaller context, same meaning.

## Getting started

See `README.md`. In short:

```
pip install -e .
agent init   # index files (only needed for low-structure code)
agent chat   # run the agent
```

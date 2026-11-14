# Code understanding

owncoder builds several layers of knowledge about your project. Each adds something different and costs something different, so you can run with only the layers you need.

| Layer | What it gives you | Cost |
|---|---|---|
| Keyword search over chunks | Exact-term lookup across the project | Nothing extra |
| Vector search (embeddings) | Find code by meaning, merged with keyword results | An embeddings model (runs fine on CPU) |
| Archive | Pruned chunks kept for a while and restorable | Disk only |
| LLM summaries (pyramid) | Plain-language description per file, module and project | LLM calls |
| Assembler / binary analysis | The same pyramid for assembly and compiled binaries | LLM calls (binaries also need REA) |
| Call and dependency graph, KB | Who calls what, symbol lookup | Nothing extra to query; built by an external graph tool |
| Memory and recall | Facts, Q&A log, session history | Nothing extra |
| Background maintainer | Keeps the above fresh while idle | CPU/LLM time while idle |

## Index and search

Source files are split into chunks along syntax boundaries with tree-sitter and stored in a local sqlite database with sqlite-vec for embeddings. With `hybrid = true` a query is answered from vector and keyword matches together.

Default grammars: Python, C, C++, bash. JavaScript, TypeScript, Rust, Go, Java and Kotlin come with the optional extra:

```sh
pip install -e .[lang]
```

A file whose grammar is missing falls back to line-based chunking, with a one-time warning.

```toml
[rag]
db_path          = ".agent/index.db"
chunk_max_tokens = 400
top_k            = 8
hybrid           = true
index_text_files = true   # also .md, .toml, Makefile, other text files
```

Commands:

```sh
agent index --update        # re-index changed files
agent index --stats         # coverage and sizes
agent index --check [--fix] # consistency report, repair what needs no embedder
agent index --reembed       # re-apply embeddings after switching model
agent index --watch         # foreground watcher
agent index --daemon        # background watcher (--stop to end it)
```

Embeddings come from the entry named by `[models] embeddings`. If no embeddings endpoint is reachable, indexing falls back to keyword-only and vectors are filled in later.

## Archive

Chunks for files that disappear or become ignored are moved to a separate archive database instead of being dropped. They expire after `archive_ttl_days`.

```sh
agent index --prune
agent index --restore path/to/file.py
agent index --purge-archive
```

## Summaries

The describer asks an LLM to summarise chunks, then summarises groups of summaries, building a multi-level pyramid from function up to project. The summarizer is the model set by `[models] summarizer`. Summaries let the agent orient in a large codebase without reading files.

## Assembler and binaries

Low-structure code (assembly) has no reliable syntax to chunk on. With `[asm_analysis] enabled = true` (off by default, toggle with `/analyze-asm`) a splitter finds boundaries, an LLM describes units, and groups are rolled up into levels (`group_size`, `max_levels`).

For compiled binaries, the `analyze_binary` and `asm_tree` tools use facts from a REA (reverse-engineering MCP) server: function boundaries, imports, strings and callees. The LLM then only names and explains functions, callees before callers, and functions that call each other most are grouped. Runs are resumable, and a changed binary is re-analysed. See the `docs/rea.md` file in the repository for REA setup.

## Call graph and knowledge base

The agent can query a per-project knowledge base built from a static call and dependency graph: `kb_search`, `kb_get`, `kb_callers`, plus `find_symbol` for definitions and references. `rag.auto_kb` refreshes it after index passes.

## Memory and recall

Facts the agent should keep, the Q&A log and session history survive across turns and compaction. `recall_facts` lets the agent re-check key constraints after its context was compacted.

## Background maintainer

While the agent is idle and the machine is not loaded, a maintainer keeps things current:

```toml
[rag]
auto_index = true      # incremental index passes after file changes
auto_kb    = true      # refresh the graph and KB
auto_describe = true   # drain the summary queue a little per idle pass
```

Background description uses only summarizer endpoints on localhost or a private network, so code is not sent to a cloud model from the background. A localhost embeddings server is not used by the maintainer unless `auto_index_allow_local_embed = true`.

## Do I need to index?

- Small project, a few files: no. Reading files directly is enough.
- Medium or large project: yes. Run `agent index --update` once, then let the maintainer keep it fresh.
- No embeddings model: keyword search still works. Add embeddings later and run `agent index --reembed`.
- Want orientation across a large codebase: also enable summaries.
- Assembly or binaries: enable `[asm_analysis]` or use `analyze_binary`; the normal index is not built for them.

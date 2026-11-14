# Configuration

owncoder reads TOML or YAML. The annotated reference with every section is
[agent.toml.example](https://github.com/adderek/owncoder/blob/master/agent.toml.example)
(also installed inside the package). `owncoder setup` writes a working starting point.

## Where config lives

Files are loaded in this order; later layers override earlier ones.

| # | File | Purpose |
|---|---|---|
| 1 | `~/.config/agent/agent.{toml,yaml,yml}` | Your user layer: models, keys, preferences. |
| 2 | `~/.config/agent/agent.<hostname>.{toml,yaml,yml}` | Per-machine layer (endpoints, paths). Picked by hostname, so one synced directory serves several machines. |
| 3 | `~/.config/agent/agent.local.{toml,yaml,yml}` | Unsynced local overrides. |
| 4 | `<project>/agent.{toml,yaml,yml}` | Project layer. Ships with the repo. |
| 5 | Environment variables `AGENT_*` | Override file values. |

Sections deep-merge key by key, and model entries merge per name. Lists are replaced as a whole,
so keep a given list in one layer. Within one directory, YAML overrides TOML.
`agent --config <file>` loads that file as the project layer instead.

!!! note "Project config is untrusted"
    A config inside a cloned repo can only tighten `[security]` and permissions, never loosen them,
    and cannot set `[agent] mode = "fast"`. Put anything that widens access in your user layer.

Unknown sections or keys are reported and ignored. Set `AGENT_CONFIG_STRICT=1` to abort instead.
`[llm]` and `[embeddings]` are not config sections any more: endpoints go in `[models.<entry>]`,
behaviour knobs in `[agent]`.

## Minimal config

```toml
[models]
default = "local"

[models.local]
base_url = "http://localhost:8080/v1"   # must end in /v1
api_key  = "local"                        # any value for a local server
model    = "qwen3-coder-30b"              # id the server serves
```

## Several models

Define one `[models.<name>]` table per endpoint and assign roles in `[models]`.

```toml
[models]
default    = "local"        # chat, run, commit
summarizer = "hosted"       # chunk and Q/A summaries
embeddings = "embed"        # RAG index

[models.local]
base_url   = "http://localhost:8080/v1"
api_key    = "local"
model      = "qwen3-coder-30b"
ctx_window = 32768          # 0 or unset = ask the server

[models.hosted]
base_url = "https://api.example.com/v1"
api_key  = "env:MY_API_KEY"
model    = "your-model-id"
tier     = "paid"           # "" | local | free | paid | bundled

[models.embed]
base_url   = "http://localhost:8080/v1"
api_key    = "local"
model      = "nomic-embed-text"
dimensions = 768
```

Useful entry keys: `base_url`, `api_key`, `model`, `ctx_window`, `max_output_tokens`,
`temperature`, `tags`, `tier`, `local`, `dimensions` (embeddings only).
Choose a model for one session with `agent chat --model <name-or-id>`, or in chat with
`/model`. `[models]` also accepts further roles such as `background`; see the example file.

## Embeddings

Indexing and semantic search need an embeddings model, assigned with `embeddings = "<entry>"` in
`[models]`. `dimensions` must match the model. A small CPU-served model is enough. After you change
the embeddings model, re-embed the existing index with `agent index --reembed`.

## Useful sections

| Section | What it controls | Key settings |
|---|---|---|
| `[agent]` | Runtime behaviour | `mode` (`ultrasecure` default, or `fast`), `model_mode` (`any`, `local-only`, `lan-only`, `free-cloud`, `free-hybrid`, `paid-cloud`), `max_iterations`, `compaction_threshold`, `think_level` |
| `[rag]` | Index and retrieval | `db_path`, `top_k`, `hybrid`, `chunk_max_tokens`, `index_text_files` |
| `[tools]` | Tool limits | `allow_shell`, `shell_timeout`, `agent_dir` (default `.agent`), `search_parents` |
| `[ui]` | Interface | `mode` (`textual`, `simple`, `http`), `syntax_highlight`, `show_token_count`, `allowed_hosts` |
| `[loop_guard]` | Stops repeated identical tool calls | `enabled`, `window`, `repeat_threshold` |
| `[web_search]` | Web search and fetch, off by default | `enabled`, `backend`, `searxng_url`, `execution_mode` (`sandboxed` or `direct`) |
| `[planning]` | Multi-step plans | `enabled`, `auto_commit_on_step_complete`, `max_steps` |
| `[parallel]` | Parallel worker subagents, off by default | `enabled`, `workers`, `global_max_concurrent`, `worker_tools` |
| `[security]` | Sandbox and limits | `sandbox_backend` (`auto`, `bwrap`, `firejail`, `none`), `require_sandbox`, `network`, `cpu_seconds`, `wall_seconds`, `rss_mb` |
| `[permissions]` | Allow, ask or deny rules per tool call | `default`, `builtin_rules`, `[[permissions.rules]]` with `tool`, `match`, `verdict`, `reason` |
| `[notify]` | Push signals to external channels, off by default | `enabled`, `events`, `on_timeout`, `answer_timeout_s`, `remote_answers` |
| `[speech]` | Speech-to-text input, off by default (needs the `speech` extra) | `enabled`, `backend`, `model`, `language` |
| `[housekeeping]` | Cleanup of `.agent/` | `enabled`, `interval_hours`, `keep_index_backups` |

`[security] require_sandbox` defaults to `true`: with no `bubblewrap` or `firejail`, shell and web
tools are refused. Setting it to `false` allows unsandboxed execution on the host; a project
config cannot do that. `[agent] mode = "ultrasecure"` removes direct web access from the main
agent and routes it through a quarantined subagent.

## Environment variables

Environment values beat every file. Booleans accept `1`, `true`, `yes`.

| Variable | Sets |
|---|---|
| `AGENT_LLM_BASE_URL`, `AGENT_LLM_API_KEY`, `AGENT_LLM_MODEL` | Endpoint, key and model id of the default model |
| `AGENT_LLM_MAX_OUTPUT_TOKENS`, `AGENT_LLM_TEMPERATURE`, `AGENT_LLM_REQUEST_TIMEOUT` | Default-model sampling and timeout |
| `AGENT_EMBEDDINGS_BASE_URL`, `AGENT_EMBEDDINGS_MODEL`, `AGENT_EMBEDDINGS_DIMENSIONS` | Embeddings endpoint |
| `AGENT_MODEL_ROLE_DEFAULT`, `AGENT_MODEL_ROLE_SUMMARIZER`, `AGENT_MODEL_ROLE_EMBEDDINGS`, `AGENT_MODEL_ROLE_BACKGROUND` | Entry name for a role |
| `AGENT_MODEL_MODE` | `[agent] model_mode` |
| `AGENT_UI_MODE` | `[ui] mode` (`textual`, `simple`, `http`) |
| `AGENT_RAG_DB_PATH`, `AGENT_RAG_TOP_K` | `[rag]` |
| `AGENT_TOOLS_ALLOW_SHELL`, `AGENT_TOOLS_SHELL_TIMEOUT`, `AGENT_TOOLS_WORKING_DIR` | `[tools]` |
| `AGENT_SECURITY_BACKEND`, `AGENT_SECURITY_REQUIRE_SANDBOX`, `AGENT_SECURITY_NETWORK` | `[security]` |
| `AGENT_WEB_SEARCH_ENABLED`, `AGENT_WEB_SEARCH_BACKEND` | `[web_search]` |
| `AGENT_LOOP_GUARD_ENABLED` | `[loop_guard] enabled` |
| `AGENT_NOTIFY_ENABLED` | `[notify] enabled` |
| `AGENT_CONFIG_STRICT` | Abort on config errors instead of ignoring them |

Setting `AGENT_LLM_MODEL`, `AGENT_LLM_BASE_URL` or `AGENT_MODEL_ROLE_DEFAULT` pins the default role:
a `/model` pick in a session cannot override it. Variables exist for most other keys too;
they are named `AGENT_<SECTION>_<KEY>`, and the full list is in `config/loader.py`.

Example without any file:

```sh
export AGENT_LLM_BASE_URL="https://api.example.com/v1"
export AGENT_LLM_API_KEY="sk-…"
export AGENT_LLM_MODEL="your-model-id"
```

## Keeping keys out of files

- Use `api_key = "env:MY_API_KEY"` in a model entry. The variable is read at start. If it is
  unset, the request fails with the provider's 401; the literal string is never sent as a key.
  `owncoder setup` writes this form when the variable is already exported.
- Set the variable in your shell profile or a secrets manager, not in the repo.
- A config that holds a literal key should be mode 0600 (`chmod 600 ~/.config/agent/agent.toml`).
  `owncoder setup` creates it that way.
- Never commit a config containing a key. Prefer keeping keys in the user layer, and keep the
  project `agent.toml` free of credentials so it can be shared.

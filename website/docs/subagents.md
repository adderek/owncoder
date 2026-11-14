# Subagents & models

owncoder can fan work out to several worker agents, each on its own model entry, and can assign different models to background roles.

## When fan-out helps

- Independent subtasks: research three modules, review five files, try one question on several models.
- Mixing models: a strong hosted model for the hard part, a small local one for bulk or cheap work.
- Keeping the main context clean: workers return conclusions, not file bodies.

It does not help for steps that depend on each other. It also costs extra model calls.

## Enabling it

The `spawn_agents` tool is off by default:

```toml
[parallel]
enabled = true
```

## Example configuration

```toml
[models.local-fast]
base_url   = "http://localhost:8080/v1"
api_key    = "local"
model      = "small-local-model"
ctx_window = 16384
local      = true

[models.hosted-strong]
base_url   = "https://api.example.com/v1"
api_key    = "env:EXAMPLE_API_KEY"
model      = "large-hosted-model"
ctx_window = 128000
thinking   = true
tier       = "paid"

[parallel]
enabled                = true
worker_tools           = "readonly"
worker_timeout_seconds = 120
global_max_concurrent  = 4

[parallel.groups.local]
models         = ["local-fast"]
max_concurrent = 1

[parallel.groups.cloud]
models         = ["hosted-strong"]
max_concurrent = 5

[parallel.decision]
enabled         = true
prefer_local    = 2.0
cost_weight     = 1.0
strength_weight = 0.5
```

Each group has its own concurrency limit, so a single local GPU is not flooded while the hosted API takes several calls at once. A model in no group falls under `global_max_concurrent`.

## How a task picks a model

A call looks like this:

```json
{"tasks": [
  {"task": "Summarise how config loading works", "model": "local-fast"},
  {"task": "Find races in the scheduler", "hint": {"min_strength": 30, "needs_thinking": true}}
]}
```

Each task takes:

| field | meaning |
|---|---|
| `task` | the instruction |
| `model` | name of a `[models.<name>]` entry; wins over everything else |
| `hint` | `est_in_tokens`, `est_out_tokens`, `min_strength` (billions of parameters), `needs_thinking` |
| `context` | pre-fetched text injected before the task, so weak workers need no search |
| `output_schema` | JSON Schema for the answer (below) |

Without `model`:

- With `[parallel.decision] enabled = true`, the decision-maker filters out models whose context window is too small, that lack thinking when needed, or that are weaker than `min_strength`. It scores the rest by cost, a `prefer_local` bonus and a strength penalty. It respects the active model mode, so a local-only mode never picks a cloud entry. If the best group is full or unreachable, the task spills to the next live group.
- Otherwise the flat `workers` list is used round-robin.

## Worker tools

`worker_tools` controls what workers may call:

- `readonly` (default): file reading, listing, search, grep, `recall`, and read-only git.
- `all`: the full tool set minus `spawn_agents`.
- `internet`: `web_search` and `web_fetch` only.

Workers cannot spawn workers. Workers share the index read-only and keep their own message history.

## Structured output

Add `output_schema` to get checked data instead of prose. The supported subset is `type`, `properties`, `required`, `additionalProperties: false`, `items`, `enum` and `minItems`. If the answer does not validate, the worker gets one repair round without tools. You then receive either the data or a `schema_error` saying what was wrong.

## Limits

- A worker that hits `worker_timeout_seconds` returns its partial text and the tools it already ran instead of nothing.
- `[concurrency] gpu_pool` and `gpu_slots` cap simultaneous requests to GPU-backed entries, with a lock that holds across several agent processes sharing an endpoint.
- Groups and `global_max_concurrent` cap parallel calls per endpoint.

## Related tools

- `explore` (`[explore]`, on by default): one read-only worker with isolated history that answers a codebase question and returns only conclusions with file and line references. It works even when `[parallel]` is off. `model` picks the entry, with `max_iterations` and `timeout_seconds` as limits.
- `ask_internet`: a disposable subagent that alone holds web tools and returns a sanitised answer. See [Security](security.md).
- `delegate`: hands a request to another named agent connected to the same relay. It is fire-and-forget: the peer treats it as a new turn. See [Relay & Android](relay.md).

## Per-role models

Besides `[models] default`, you can pin background roles to named entries: `summarizer`, `background`, `namer`, `compaction`, `review`, `triage`, `verify`, `evolve`, `commit`.

```toml
[models]
default    = "hosted-strong"
summarizer = "local-fast"
namer      = "local-fast"
```

Switch live with `/model <role>=<entry>`; `/model` with no argument shows the matrix. Unset roles fall back to `default`. `/use` offers a simpler selector: where (`auto`, `local`, `cloud`), model (`auto`, `fast`, `balanced`, `strong`) and effort.

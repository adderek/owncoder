# Scheduled jobs (cron-like + delayed execution)

Jobs are prompts the agent runs unattended, each in a **fresh agent with its own
session** (`sched-<name>`), so scheduled work never touches the user's
conversation. Module: `agent/core/scheduler.py`. Store:
`<agent_dir>/schedule/jobs.json` (flock-guarded), run log `schedule/runs.jsonl`.

## Schedule specs

| Spec | Meaning |
|------|---------|
| `in 20m` / `in 2h` / `in 1d` | one-shot delay |
| `at 07:00` / `at 2026-07-03T07:00` | one-shot at local time (time-only rolls to tomorrow if past) |
| `every 30m` / `every 6h` | recurring interval |
| `0 7 * * *` | 5-field cron, local time (`m h dom mon dow`; `*/n`, lists, ranges) |
| `@hourly` `@daily` `@midnight` `@weekly` | cron shorthands (`@daily` = 07:00) |
| `idle` | one-shot at the next idle sweep (no user↔agent interaction ongoing) |

## Three trigger paths (one shared store)

1. **In-process ticker** — a long-running `agent chat` starts a daemon thread
   (`[scheduler] tick_seconds`, default 60 s). It defers to the user: skips
   while a turn runs and until `min_quiet_seconds` (30 s) after the last one.
2. **External scheduler** — `agent cron run` claims and runs due jobs, then
   exits. Point crontab or a systemd timer at the project:

       */5 * * * *  cd /home/adderek/src/assistant && agent cron run

   Claiming happens under the store flock, so a crontab entry and a running
   chat ticker never double-run the same job.
3. **Idle sweep** — jobs with spec `idle` fire from the idle-task queue
   (`core/idle_tasks.py`), i.e. only when the agent is otherwise doing nothing.

## Interfaces

- Slash: `/schedule` (`/sched`) — `list | add <spec> :: <prompt> [:: <name>] |
  rm <id|name> | on/off <id|name> | runs | run`
- CLI: `agent cron run [--job <id|name>] | add | rm | enable | disable | runs | list`
- Agent tools: `schedule_task(spec, prompt, name)` — the agent defers its own
  work ("remind me", "check this again in 20m"); `list_scheduled()`;
  `cancel_scheduled(id_or_name)`.

- Agent tools: `sleep_until(note, after, until)` / `cancel_wake(id|all)` —
  pause the **current** conversation instead (`core/wake.py`). Wakes on the
  first of: `after` (`10m`, `in 2h`, `at 07:00`), `until` event (`job:<id>` of
  `run_argv_bg`, `pid:<n>`, `file:<path>`, `url:<url>`, `cmd:<shell>` — same
  semantics as `/watch`), or the user's next message (supersedes all pending
  waits; the model gets a note listing them). Delivered as a `[wake <id>] ...`
  turn through `LocalUIServer.submit_external_prompt`, only while no turn runs.
  In-process: lost on exit. Readline UI has no prompt handler → wake waits for
  next input. `cmd:` is ask-gated (builtin permission rule) and needs
  `tools.allow_shell`.

The scheduled prompt must be **self-contained** — the future run is a fresh
agent with no memory of the conversation that created the job.

## Config

```toml
[scheduler]
enabled = true          # ticker + idle hook in `agent chat`
tick_seconds = 60
min_quiet_seconds = 30  # defer jobs this long after the last user turn
```

## Typical daily-assistant jobs

```
agent cron add "@daily"    "Gather a press summary of today's key news and save it with save_note." --name press
agent cron add "0 8 * * *" "Check the key indicators I track and summarize changes."               --name indicators
agent cron add "idle"      "Introspect recent sessions for recurring issues; save findings."       --name introspect
```

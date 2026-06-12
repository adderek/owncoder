# Notification Channel

Push agent progress/questions to external endpoints; optionally receive answers.
Goal: follow and steer the agent without the full terminal UI. Long-term: this
wire protocol becomes a full UI transport (graphical UI, remote control).

## Architecture

```
turn signals (>>>ASK/BLOCKED/DONE/...)        answers (phase 2)
        │                                          ▲
        ▼                                          │
LocalUIServer.chat() ── NotifyBroker ──► Channel 1 (command, display)
                            │       └──► Channel 2 (relay, chat)
                            └─ pending questions {id → Future}
```

- `agent/notify/messages.py` — `Notice`, `Question`, `Answer` + JSON wire envelope.
- `agent/notify/channels.py` — channel impls + `build_channel()` (bad config → skip + warn, never blocks startup).
- `agent/notify/broker.py` — fan-out, event filter, pending-question registry.
- Wired in `ui_server/local.py`: broker built in `__init__`, `handle_signal()` after each parsed turn signal. Non-blocking (fire-and-forget tasks).

## Capability tiers

| tier | receives | can send back |
|------|----------|---------------|
| `display` | notices + questions rendered as text | nothing |
| `choices` | notices + questions | one of the offered options |
| `chat`    | everything | options + free text |

Multiple channels run in parallel; first valid answer wins, the rest get an
"answered by ..." notice.

## Wire envelope (JSON, one object per message)

```json
{"type":"notice",  "id":"n-...","kind":"done","text":"...","session":"..."}
{"type":"question","id":"q-...","kind":"ask_user","text":"...",
 "options":["accept","refuse"],"free_text":true,"session":"...","expires_at":0.0}
{"type":"answer",  "id":"q-...","choice":"accept","text":"","from":"user"}
```

`from` may be `user` or `agent:<id>` — other agents can be parties on the
channel; per-question policy decides whether agent answers are accepted.

## Security invariants

- An answer is data resolving exactly one pending question. It must match a
  pending question id; `choice` must be one of the offered options; free text
  only when the question allowed it. Ids are single-use; late/duplicate/expired
  answers are dropped and logged. The channel must never become an instruction
  injection path into the agent.
- Command channels: message goes to the command's stdin, never interpolated
  into the command line.
- Relay (phase 2): agent connects outbound only (no listening port), token
  auth, TLS required.

## Config

`[notify]` in `agent.toml` or `agent.yaml`. Off by default.

```yaml
notify:
  enabled: true
  events: [ask_user, blocked, done]   # signal kinds that push
  answer_timeout_s: 600
  on_timeout: continue                # continue (use question default) | wait
  channels:
    - type: command                   # pipe to stdin: ntfy, signal-cli, ...
      cmd: ntfy publish mytopic
      capability: display
      format: text                    # text | json
```

Env: `AGENT_NOTIFY_ENABLED`, `AGENT_NOTIFY_ANSWER_TIMEOUT`, `AGENT_NOTIFY_ON_TIMEOUT`.
Slash: `/notify [on | off | status]`.

## Relay (phase 2)

Server (runs on your own host; TLS via reverse proxy when exposed):

```
pip install 'local-code-agent[notify]'   # websockets
python -m agent.notify.relay_server --port 8970 --token-file ~/.config/agent/relay.token
```

- In-band auth: first websocket message must be
  `{"type":"hello","role":"agent"|"client","token":"...","name":"..."}`;
  bad/missing hello → close 4401 (constant-time token compare).
- Routing: agent → all clients (+ replay buffer, last `--replay` messages
  resent to newly connected clients); client → all agents.
- Server forwards opaque JSON only; all answer validation stays in the
  agent-side broker.

Agent channel:

```yaml
notify:
  enabled: true
  remote_answers: true        # ask_user/blocked wait for remote answer
  channels:
    - type: relay
      url: wss://myhost:8970
      token_file: ~/.config/agent/relay.token
      capability: chat
```

`RelayChannel`: outbound-only persistent connection, local queue (drops oldest
at 100), reconnect with exponential backoff (1s→60s), incoming `answer`
messages → `NotifyBroker.submit_answer()`.

**remote_answers**: when true and a choices/chat channel exists, `ask_user` /
`blocked` signals are pushed as a `Question` and the meta-loop waits up to
`answer_timeout_s` for a remote answer; the answer is fed back into the turn as
user input (auto-step counter resets). On timeout the turn returns to the
terminal UI as usual. Trade-off: terminal input is blocked while waiting —
that's why it's off by default.

Test a client by hand:

```
wscat -c ws://localhost:8970
> {"type":"hello","role":"client","token":"..."}
< {"type":"question","id":"q-...","text":"deploy?","options":["yes","no"],...}
> {"type":"answer","id":"q-...","choice":"yes","from":"user"}
```

## Phases

1. **(done)** `command` channel, display-only. Self-hosted ntfy gives Android
   push today; ntfy action buttons (max 3, HTTP callback) can cover the
   `choices` tier with zero app code.
2. **(done)** `relay` channel + `relay_server.py` + `remote_answers` loop —
   answer from any websocket client steers the agent mid-run.
3. Dedicated Android app speaking the relay protocol (full `chat` tier:
   option lists, free text, session picker, push via FCM or persistent
   connection). Only when ntfy/wscat-grade clients are outgrown.

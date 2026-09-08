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
- Relay: agent connects outbound only (no listening port), token auth. Do not
  expose the relay on the WAN (token is in-band) — run it behind WireGuard
  (bind to wg0); WG encrypts the wire and admits only authenticated peers.
- End-to-end encryption (`e2e_key_file`): second secret known only to agent +
  client — never to the relay. All payloads AES-256-GCM
  (key = HKDF-SHA256(secret, salt="owncoder-notify", info="e2e-v1"), random
  12-byte nonce, AAD "owncoder-notify-v1"); relay forwards opaque
  `{"type":"enc","v":1,"n":...,"c":...}` envelopes and can neither read nor
  forge messages. Fail closed both ways: key unreadable / cryptography missing
  → channel disabled (no plaintext fallback); incoming plaintext dropped when
  e2e on (no downgrade). Cross-implementation vector pinned in
  `tests/unit/test_notify_e2e.py` and the Android client's `CryptoTest.kt`.
  Generate: `openssl rand -base64 32 > ~/.config/agent/notify-e2e.key`.

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

Server (runs on your own host; bind to the wg0 address and reach it over
WireGuard when exposed — no TLS terminator, the WG tunnel encrypts the wire):

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
      url: ws://10.0.0.1:8970                          # relay wg0 address; ws:// — WireGuard encrypts the wire
      token_file: ~/.config/agent/relay.token
      e2e_key_file: ~/.config/agent/notify-e2e.key   # E2E; relay sees ciphertext only
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
3. **(done)** Android client `clients/android/` (repo root): foreground
   service holds the relay connection; questions arrive as notifications with
   option action buttons; free-text answers in-app; E2E encryption with 🔒
   indicator. See `clients/android/README.md`.

## Remote UI: rendering profiles

`ui_server/projection.py` turns the `ViewModel` into a per-client payload, so a
client renders instead of re-implementing the fold and the truncation rules
(`static/app.js` and the Android client each do that today, and they drift).

| profile | client | streaming | reasoning | markdown | answer cap |
|---|---|---|---|---|---|
| `full` | desktop TUI / browser | yes | yes | yes | — |
| `compact` | phone | yes | no | yes | 400 tokens |
| `glance` | smart glasses | no | no | no | 100 tokens, last turn only, ≤3 question lines |

Unicode is preserved in all profiles — monochrome is a colour-depth limit, not a
charset limit. Unknown profile names fall back to `full`.

**Profiles are a rendering contract, not a security boundary.** The name is
self-asserted by the client and may only narrow what is displayed. Authority is
enforced separately, below.

## Remote UI: what a remote client may do

A relay client token authenticates the *channel*, not the *device*. `[ui_server]
remote_actions` therefore lists the control actions a remote client may perform;
the default excludes `set`:

```yaml
ui_server:
  remote: true
  relay_url: ws://wg0:8970
  relay_token_file: ~/.config/agent/relay.token
  # set is omitted on purpose: a lost or unlocked phone must not be able to
  # rewrite the model, autonomy or plan. Add it to opt in.
  remote_actions: [chat, answer, stop, inject, changeset_diff]
```

A frame whose action is not listed is logged and dropped, never raised: the relay
is a shared channel, so killing the agent's link over someone else's frame would
be worse than ignoring it. The local TUI is unaffected — it does not use the
relay. WireGuard protects the transport; e2e protects the payload from the relay
host; this list protects the *agent's configuration* from a client.

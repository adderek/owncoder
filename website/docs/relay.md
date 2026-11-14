# Relay & Android

The relay lets you reach a running agent from a phone or another machine without opening any port on the agent host. It is a small websocket server that forwards messages between agents and clients.

!!! note "About the screenshots"
    Screenshots in this documentation come from a demo sandbox with a scripted fake model.

```
  agent host                relay                    phone
 +-----------+   ws    +-------------+    ws    +-----------+
 |  agent    | ------> |   relay     | <------- |  client   |
 | (outbound |         |  (forwards) |          |  (app)    |
 |   only)   | <------ |             | -------> |           |
 +-----------+         +-------------+          +-----------+
```

The agent connects outbound. Clients connect to the relay. The relay checks a shared token and forwards JSON without interpreting it.

## Running the relay

On the agent host, install with the `notify` extra:

```
uv tool install "owncoder[notify] @ git+https://github.com/adderek/owncoder"
```

On the machine that runs the relay, use a source checkout (see [Install](install.md)) so the module is importable:

```
uv pip install -e ".[notify]"
.venv/bin/python -m agent.notify.relay_server --port 8970 --token-file ~/.config/agent/relay.token
```

Options include `--host` (default `0.0.0.0`), `--agent-token-file` / `--client-token-file` for per-role tokens, `--replay` (messages replayed to newly connected clients, default 100), `--max-agents` and `--max-clients`. The first websocket message must be a `hello` carrying the token; otherwise the relay closes the connection.

!!! warning
    The token travels in-band. Do not expose the relay to the open internet as is; put it behind a VPN such as WireGuard, or terminate TLS in front of it. The placeholder `wss://relay.example.org` below stands for such a setup.

## Two links

| | `[notify]` channel | `[ui_server]` remote |
|---|---|---|
| Carries | Notices (`ask_user`, `blocked`, `done`) and questions | Full UI event stream |
| Client can | Answer a question: pick an option, or free text | Send control frames: `chat`, `answer`, `stop`, `inject`, `changeset_diff` |
| Off by default | yes | yes |

The control frame `set` (changing model, autonomy, plan or notify settings) is excluded by default. A lost or unlocked phone should not be able to rewrite agent configuration. You can opt in by adding `"set"` to `remote_actions`. Frames not on the list are logged and dropped.

### Notifications and answers

```toml
[notify]
enabled = true
remote_answers = true      # ask_user/blocked wait for a remote answer
answer_timeout_s = 600
on_timeout = "continue"    # or "wait"

[[notify.channels]]
type = "relay"
url = "wss://relay.example.org"
token_file = "~/.config/agent/relay.token"
e2e_key_file = "~/.config/agent/notify-e2e.key"
capability = "chat"        # display | choices | chat
```

With `remote_answers = true`, terminal input is blocked while the agent waits for an answer. After `answer_timeout_s` the turn continues (default option) or keeps waiting, per `on_timeout`.

### Full remote UI

```toml
[ui_server]
remote = true
relay_url = "wss://relay.example.org"
relay_token_file = "~/.config/agent/relay.token"
e2e_key_file = "~/.config/agent/notify-e2e.key"
# "set" is left out on purpose
remote_actions = ["chat", "answer", "stop", "inject", "changeset_diff"]
```

The local UI keeps working in parallel. The remote client renders a profile suited to its screen (for example a compact one for phones).

## End-to-end encryption

Set `e2e_key_file` on both links. The key is a secret shared only by the agent and the client; the relay never sees it. Payloads are encrypted with AES-256-GCM and the relay forwards opaque ciphertext envelopes, so it can neither read nor forge messages. Failure is closed: if the key cannot be read, the channel is disabled instead of falling back to plaintext, and plaintext messages are dropped once e2e is on.

Generate a key:

```
openssl rand -base64 32 > ~/.config/agent/notify-e2e.key
```

Copy the same key to the client by a path other than the relay.

## Voice

A client can record audio and stream it to the agent host as chunked `voice` frames. The host transcribes it with faster-whisper and either answers the pending question or starts a new turn. Enable with:

```
uv tool install "owncoder[speech] @ git+https://github.com/adderek/owncoder"
```

and the `[speech]` section of the config. It is off by default.

## Android client

The Android client talks to the agent through the relay. It can:

- show notifications for agent notices and questions, with answers from the lock screen
- chat with the agent
- capture voice for transcription on the host
- use the end-to-end key described above

Its source is not part of this repository. For the wire protocol and the full config reference, see `docs/notify.md` in the owncoder source tree.

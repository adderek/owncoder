# Remote SSH hosts (`[remote]`)

Lets the agent run commands on lab machines you name: a router/AP that can
capture packets on an isolated interface, the device under test behind it,
traffic generators. Off by default.

**What you are granting:** `remote_exec` runs any command the model writes on
that machine, with the configured account's rights, outside the local
sandbox. List only devices you accept the agent fully controlling (lab gear,
not your production router or a box with credentials to other systems).

## Intended use

Prototyping and on-device work (bring-up, firmware/config iteration, packet
capture on the device or on the AP in front of it), security research, and
vulnerability/pen testing of equipment you own or are authorised to test.
Targets are disposable lab devices that can be reflashed or reset.

## Accepted risks

Known and accepted for that use; the controls above are guard rails against
mistakes, not a containment boundary against the agent:

- Once on a host the agent is root (or whatever the account is): it can add
  its own key to `authorized_keys`, change passwords, install services, and
  keep access after you revoke the approval or remove the host from config.
- It can pivot from a host to anything that host reaches (`dbclient`/`ssh`,
  `nc`, keys and credentials stored on the device).
- With `security.network = "on"` the local sandbox gets the host network, so
  a `run_argv(..., network=True)` script can open raw TCP to LAN devices
  (password/no-auth SSH, telnet, HTTP admin UIs) without the `remote_*`
  confirmation, host list or audit events — only the plain `run_argv` audit.
  Keep `network = "off"` if that matters.

After a session, treat touched devices as agent-controlled: reflash/reset them
or at least review `authorized_keys`, users and running services before
reusing them outside the lab.

## Setup (user does every step)

1. Generate a dedicated key per lab, no passphrase:

   ```sh
   ssh-keygen -t ed25519 -N '' -C agent-lab -f ~/.ssh/agent_lab
   ```

   Refused: default names (`id_ed25519`, `id_rsa`, ...), keys with a
   passphrase, keys readable by group/others, keys not owned by you.

2. Authorise it on the device only (OpenWrt: `/etc/dropbear/authorized_keys`;
   OpenSSH: `~/.ssh/authorized_keys` of the target account). Consider a
   `from="<your-ip>"` prefix and a non-root account where root is not needed.

3. Declare hosts in your **user** config (`~/.config/agent/agent.yaml` or
   `agent.<hostname>.yaml`). A project/repo config cannot add hosts — the
   loader drops `[remote]` from it (it may only set `enabled: false`).

   ```yaml
   remote:
     enabled: true
     hosts:
       - name: router
         host: 192.168.50.1
         user: root
         identity_file: ~/.ssh/agent_lab
         description: OpenWrt AP, no internet; DUT on br-lan (eth1); tcpdump installed
         confirm: session          # session | always | never
       - name: dut
         host: 192.168.50.20       # reachable only from the router
         user: root
         identity_file: ~/.ssh/agent_lab
         jump: router              # ProxyCommand through 'router'
         description: device under test
         confirm: always
         # ssh_options: ["HostKeyAlgorithms=+ssh-rsa"]   # old dropbear
   ```

4. Pin host keys (jump host first), comparing the fingerprint with the device:

   ```sh
   agent remote trust router
   agent remote trust dut
   agent remote check            # validates key + host key, test-connects
   ```

   Keys go to `remote.known_hosts` (default
   `~/.config/agent/remote_known_hosts`), separate from `~/.ssh/known_hosts`.

## Behaviour

- Tools: `remote_exec(host, cmd, timeout)`, `remote_copy(host, direction,
  remote_path, local_path)`. The model only picks a host name (schema enum
  with your `description`s); it never supplies address, user, key or options.
- `confirm`: `session` warns and asks once per host per agent process;
  `always` asks for every call; `never` skips the prompt (needed for
  headless `agent run`, where nobody can answer).
- ssh is run with `-F /dev/null` (your `~/.ssh/config` is ignored),
  `IdentitiesOnly`, `IdentityAgent=none` (ssh-agent never used),
  `BatchMode`, `StrictHostKeyChecking=yes`, no agent/X11/port forwarding, no
  connection sharing, minimal env. `ssh_options` can add options but cannot
  override these (ssh keeps the first value).
- `remote_copy` streams via `cat` (works on busybox/dropbear). The local path
  goes through the same fs gate and `.agent.*` rules as file tools; size cap
  `max_copy_bytes`.
- Blocked when `security.airgap` is on. Output is treated as untrusted
  (injection guard). Every call is logged to `.agent/audit.jsonl`.
- Long captures: bound them on the device, e.g.
  `timeout 20 tcpdump -i br-lan -nn -c 500 -w /tmp/c.pcap`, then
  `remote_copy(get)` and analyse locally.

## In the UI (TUI, readline, HTTP)

`/remote` (= `status`) lists hosts, readiness (key, pinned host key) and
whether each is approved for this session. `/remote revoke [name]` drops the
session approval, so the next command asks again — in the HTTP UI it runs
even while a turn is in progress. `/remote check [name]` test-connects.
`/remote approve <name>` answers the session prompt ahead of time;
`/remote off|on <name>` switches a host off/on for this agent process (off
also drops its approval and blocks every host that jumps through it; the
model's host list shrinks accordingly).

HTTP UI: left drawer → **remote — SSH hosts** (next to *access — allowed
paths*). Per host: state (`will ask` / `approved` / `asks each time` /
`no prompt` / `off` / `not ready` + the fix), buttons ✓ approve (with a
browser confirm), ✗ revoke, ⇄ test-connect, ⏸ off / ▶ on. Changes are
audited (`remote_ui` events).
Pinning host keys and editing hosts are deliberately not available from the
UI (the HTTP UI may be reached remotely): use `agent remote trust` and your
user config.

## Config reference

| key | default | |
|---|---|---|
| `remote.enabled` | `false` | |
| `remote.known_hosts` | `~/.config/agent/remote_known_hosts` | pinned host keys |
| `remote.connect_timeout_s` | `10` | |
| `remote.command_timeout_s` | `120` | per call unless `timeout` given |
| `remote.max_copy_bytes` | `268435456` | `remote_copy` cap |
| `hosts[].name/host/user/port` | `—/—/root/22` | |
| `hosts[].identity_file` | required | dedicated key |
| `hosts[].jump` | `""` | another host's `name` |
| `hosts[].description` | `""` | shown to the model |
| `hosts[].confirm` | `session` | `session`/`always`/`never` |
| `hosts[].ssh_options` | `[]` | extra `Key=Value` |
| `hosts[].enabled` | `true` | |

# Extra isolation (optional)

owncoder is **security-first and local-first**. It already ships an internal
sandbox: every command the agent runs is wrapped in **bubblewrap/firejail**
(network namespace, seccomp, dropped capabilities, filesystem confined to the
project root). That guards the boundary *command → agent*.

An **extra** layer of security guards the boundary *agent → host*: put the
whole agent process inside one more sandbox box. The internal sandbox stays;
this wraps everything around it. Pick the box that fits your threat model.

---

## Option 1 — Podman (recommended)

Rootless, daemonless, one line. Best simplicity/security ratio. If the agent
process is ever compromised, it cannot touch the host.

```sh
podman run --rm -it \
  --userns=keep-id \
  -v "$PWD:/work:Z" -w /work \
  -v owncoder-agent:/work/.agent \
  --network=host \
  owncoder
```

- Rootless by default — no root daemon to escape through.
- `--network=host` only if the local LLM endpoint lives on the host; for
  tighter egress control drop it and expose just the LLM/relay ports.
- Build the image once: `podman build -t owncoder .` (see `Containerfile`).

## Option 2 — Firecracker (most secure)

microVM: separate real kernel behind a hardware boundary, minimal device
emulation (Rust, ~5 devices). Strongest escape resistance — use when running
untrusted code (foreign PoCs, harvested material, multi-tenant). Heavier to set
up (kernel image, rootfs, jailer); no one-liner.

## Option 3 — none

Run the agent directly on the host. Valid when the host is trusted and the
internal bubblewrap sandbox is enough. This is the default posture.

---

## Why this is not part of owncoder

These options **wrap** owncoder to raise security; they do not belong inside it.

- A security boundary must sit *above* the thing it confines — an agent that
  built its own outer sandbox could also fail to build it. The wrapper stays
  outside the agent's control.
- Container vs VM vs host is a **deployment decision** (network, storage,
  kernel, density), not agent logic. The same owncoder image runs unchanged on
  the host, in Podman, or in a microVM.
- The user picks the box that matches their threat model. owncoder ships the
  internal sandbox and this guide; the operator supplies the outer layer.

> Note: `Containerfile` uses standard OCI syntax — Podman prefers that name,
> Docker also reads it. One file, both engines.

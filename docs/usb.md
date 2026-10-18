# USB devices (`[usb]`)

Hands chosen USB devices to the agent's sandboxed commands (`run_argv`):
an ESP32 HID/CDC gadget, a capture dongle, a programmer. Off by default.
For a UART console prefer `[serial]` (`docs/serial.md`) — it keeps the port
open between turns and has expect/arm.

**What you are granting:** sandboxed commands may open every `/dev` node the
device exposes (hidraw, tty, input, video, `/dev/bus/usb/BBB/DDD` = full
libusb control incl. reset). Sharing a keyboard shares its keystrokes. Block
nodes (USB storage) are never bound — raw disks stay denied.

## How it works

- A device is declared by `id` (`vvvv:pppp`, as `lsusb` prints) and optional
  `serial`. Bus/device numbers change on replug, so nothing keys on them.
- Before each sandboxed command the matching devices are looked up in sysfs
  and their current nodes `--dev-bind`-ed into bwrap. Replug needs nothing.
- The system prompt names each device and its current nodes. No `/sys` inside
  the sandbox: open the `/dev` path directly (libusb/hidapi enumeration may
  not find the device).
- Unix permissions still apply — the agent runs as you. Default udev modes
  (hidraw `0600 root`) need a rule.
- User config layer only; a project layer may only set `enabled: false`.
- Firejail backend: not bound (bwrap only).

## Setup — HTTP UI

Left drawer → **usb — devices** → `share…` on the device. The panel shows:

1. the YAML to paste into `~/.config/agent/agent.<host>.yaml` (devices are
   per machine),
2. a udev rule (`/etc/udev/rules.d/70-agent-<name>.rules`, `MODE 0660`, your
   group, `uaccess`) + the `udevadm` reload command,
3. **reload [usb] from config** — re-reads `[usb]` only, no restart.

Per-node ✓/✗ shows whether you can open it now.

## Config

```yaml
usb:
  enabled: true
  devices:
    - name: knob                 # what the model calls it
      id: "303a:4001"
      serial: "44095c21641c"     # optional: this unit only
      description: "ESP32-S3 radial knob, HID input reports on hidraw"
      # enabled: false           # keep the entry, stop sharing
```

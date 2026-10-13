# Serial consoles (`[serial]`)

Lets the agent use the UART console of a lab device through a USB-UART
adapter: bring-up, U-Boot, recovery when SSH is dead, watching boot logs.
Off by default. For bulk work once the device is on the network, prefer
`remote_exec` (`docs/remote.md`).

**What you are granting:** a UART console is usually an unauthenticated root
shell, and the bootloader can reflash or erase the device. Same accepted risks
as `[remote]` (`docs/remote.md` → *Accepted risks*); list only lab devices.

## Hardware

- **3.3 V** adapter (CP2102, FT232, CH340; FT4232H = 4 ports on one cable).
  Most router UARTs are 3.3 V; 5 V can kill the SoC.
- Wire TX→RX, RX→TX, GND→GND. **Leave VCC unconnected.**
- 115200 8N1 on most routers; set `baud` otherwise.

## Setup (user does every step)

1. Access to the device node. Arch: group `uucp`; Debian/Ubuntu: `dialout`.

   ```sh
   sudo usermod -aG uucp "$USER"     # then log in again
   ```

2. A stable name, so ports don't swap on replug. Either use
   `/dev/serial/by-id/usb-...` directly, or a udev rule by adapter serial
   (`udevadm info -a /dev/ttyUSB0 | grep '{serial}'`):

   ```
   # /etc/udev/rules.d/70-lab-serial.rules
   SUBSYSTEM=="tty", ATTRS{serial}=="A50285BI", SYMLINK+="lab/router-uart0", GROUP="uucp", MODE="0660"
   ```

   `sudo udevadm control --reload && sudo udevadm trigger`

3. `pip install -e '.[serial]'` (pyserial).

4. Declare ports in your **user** config (`~/.config/agent/agent.yaml` or
   `agent.<hostname>.yaml`). A project config cannot add ports (loader keeps
   only `enabled: false`).

   ```yaml
   serial:
     enabled: true
     # log: true                    # .agent/serial/<name>/{console.log,actions.jsonl}
     ports:
       - name: router-uart0
         url: /dev/lab/router-uart0  # or socket://host:port / rfc2217://host:port (ser2net)
         baud: 115200                # bytesize 8, parity N, stopbits 1 by default
         description: OpenWrt console, root no password; U-Boot "Hit any key to stop autoboot"
         confirm: session            # session | always | never
   ```

## Behaviour

- A port opens on first use and stays open, **exclusively** (flock), for the
  agent process. A background reader keeps everything it prints (ring of
  `buffer_bytes`), so output between model turns is not lost. picocom on the
  same port fails while the agent holds it; watch the log instead.
- Tools (port = schema enum with your `description`s):
  - `serial_send(port, data, newline, key, expect, timeout)` — types `data`
    (+ `line_ending`, default CR), or a `key` (`enter ctrl_c ctrl_d ctrl_z esc
    break`). Returns output from the moment of sending (echo included): until
    `expect` regex matches, else until the line is quiet `idle_ms`.
  - `serial_read(port, wait_for, timeout)` — output since the last read;
    lists armed responses.
  - `serial_arm(port, pattern, response, newline, once, ttl_s | clear)` — the
    reader types `response` the instant `pattern` appears. For prompts with a
    1–3 s window (U-Boot autoboot, OpenWrt failsafe) a model round trip is too
    slow; arm, then reboot.
- Output: ANSI stripped, CR removed, head+tail truncated to
  `max_output_chars`; `lost_bytes` when the ring overflowed. Untrusted
  (injection guard). Adapter unplugged → `error`, next call reopens.
- `confirm`: as `[remote]` (`never` needed for headless `agent run`).
  Network URLs blocked in air-gap. Calls audited in `.agent/audit.jsonl`.
- Transcript (`log: true`, default): `.agent/serial/<name>/console.log` raw
  console bytes (`tail -f` it to watch), `actions.jsonl` every open/send/arm/
  trigger/error with timestamp.

## Config reference

| key | default | |
|---|---|---|
| `enabled` | false | |
| `log` | true | transcript files |
| `buffer_bytes` | 1 MiB | per-port ring |
| `idle_ms` | 500 | quiet time ending a read without `expect` |
| `command_timeout_s` | 30 | default `timeout` |
| `max_output_chars` | 16000 | per tool result |
| `ports[].url` | | device path or pyserial URL |
| `ports[].baud / bytesize / parity / stopbits / rtscts / xonxoff` | 115200 8 N 1 off off | |
| `ports[].line_ending` | `"\r"` | |
| `ports[].confirm` | session | |

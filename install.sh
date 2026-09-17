#!/bin/sh
# owncoder installer — Linux and macOS.
#
#   curl -LsSf https://raw.githubusercontent.com/adderek/owncoder/master/install.sh | sh
#
# Installs uv if absent, installs owncoder as an isolated uv tool (its own
# Python and its own dependency set — nothing is added to any system or
# project environment), then runs the first-start wizard.
#
# Windows is not supported natively: the sandbox is bubblewrap/seccomp and the
# locking uses fcntl. Use WSL2 and run this script inside it (install.ps1 sets
# that up for you).
set -eu

REPO="https://github.com/adderek/owncoder"
# Override to install a branch or a local checkout:
#   OWNCODER_SOURCE=. sh install.sh
#   OWNCODER_SOURCE="git+$REPO@some-branch" sh install.sh
SOURCE="${OWNCODER_SOURCE:-git+$REPO}"

say()  { printf '%s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# ── platform ────────────────────────────────────────────────────────────────
case "$(uname -s)" in
    Linux)  PLATFORM=linux ;;
    Darwin) PLATFORM=macos ;;
    *)
        die "unsupported platform '$(uname -s)'.
Windows users: install WSL2 (wsl --install), open the Linux shell, and run
this script there."
        ;;
esac

# ── prerequisites the agent shells out to ───────────────────────────────────
missing=""
have git || missing="$missing git"
[ -n "$missing" ] && die "missing required command(s):$missing
Install them with your package manager and re-run this script."

have rg || warn "ripgrep (rg) not found — search falls back to a slower path.
  Debian/Ubuntu: sudo apt install ripgrep | macOS: brew install ripgrep"

if [ "$PLATFORM" = linux ] && ! have bwrap && ! have firejail; then
    warn "neither bubblewrap nor firejail found.
  Shell commands and web fetches run sandboxed through one of them; without
  either, those tools refuse to run.
  Debian/Ubuntu: sudo apt install bubblewrap | Fedora: sudo dnf install bubblewrap"
fi

# ── uv ──────────────────────────────────────────────────────────────────────
if ! have uv; then
    say "uv not found — installing it from https://astral.sh/uv ..."
    if have curl; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif have wget; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        die "need curl or wget to fetch uv. Install one, or install uv yourself:
  https://docs.astral.sh/uv/getting-started/installation/"
    fi
    # The uv installer drops the binary here and only edits shell rc files,
    # which this non-interactive shell has not sourced.
    for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
        [ -x "$d/uv" ] && PATH="$d:$PATH"
    done
    export PATH
    have uv || die "uv installed but not on PATH. Open a new shell and re-run."
fi

# ── install ─────────────────────────────────────────────────────────────────
say "Installing owncoder from $SOURCE ..."
# --force so re-running the script upgrades an existing install in place.
uv tool install --force "$SOURCE"

# `uv tool dir --bin` is authoritative (it honours UV_TOOL_BIN_DIR and the
# platform's conventions); older uv versions lack the flag.
BIN_DIR="$(uv tool dir --bin 2>/dev/null || true)"
[ -n "$BIN_DIR" ] || BIN_DIR="$HOME/.local/bin"

if ! have owncoder; then
    say ""
    say "Installed to $BIN_DIR, which is not on your PATH. Add it:"
    say "  echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.profile"
    say "  export PATH=\"$BIN_DIR:\$PATH\""
    PATH="$BIN_DIR:$PATH"
    export PATH
fi

say ""
say "Installed: $(owncoder --version 2>/dev/null || echo owncoder)"
say "Commands:  owncoder   (alias: agent)"
say "Upgrade:   uv tool upgrade owncoder"
say "Remove:    uv tool uninstall owncoder"

# ── first start ─────────────────────────────────────────────────────────────
if [ -f "$HOME/.config/agent/agent.toml" ]; then
    say ""
    say "Existing config found at ~/.config/agent/agent.toml — leaving it alone."
    say "Next: cd into a project, then 'agent init' and 'agent chat'."
    exit 0
fi

# Piped into sh, stdin is the script itself, so the wizard cannot read answers
# from it. Re-attach the terminal when there is one; otherwise just say what to
# run next.
if { : < /dev/tty; } 2>/dev/null; then
    say ""
    owncoder setup < /dev/tty || say "Setup skipped. Run 'owncoder setup' when ready."
else
    say ""
    say "Next: run 'owncoder setup' to choose a model provider."
fi

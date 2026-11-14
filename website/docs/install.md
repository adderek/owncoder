# Install

owncoder runs on Linux and macOS. On Windows it runs inside WSL2. It is not on PyPI yet, so all
installs below pull from the GitHub repository.

## Requirements

| Requirement | Needed | Notes |
|---|---|---|
| Python 3.11 or newer | yes | `uv` provides its own interpreter, so you do not need to install one. |
| `git` | yes | The installer stops if it is missing. |
| `uv` | for the one-liner and `uv tool` | The one-liner installs it if absent. |
| `ripgrep` (`rg`) | no | Faster search. Without it a slower path is used. |
| `bubblewrap` or `firejail` | Linux | The sandbox for shell and web tools. Without one, those tools refuse to run. |
| An LLM endpoint | yes | Any OpenAI-compatible server: llama.cpp, vLLM, Ollama, LM Studio, or a hosted API. |

!!! warning "No sandbox, no shell tools"
    On Linux, `[security] require_sandbox` defaults to `true`. If neither `bubblewrap` nor `firejail`
    is installed, shell and web tools are refused. Install one, for example
    `sudo apt install bubblewrap` or `sudo dnf install bubblewrap`.

## Install

=== "Linux / macOS"

    ```sh
    curl -LsSf https://raw.githubusercontent.com/adderek/owncoder/master/install.sh | sh
    ```

    The script installs `uv` if it is missing, installs owncoder as an isolated `uv` tool, and then
    starts `owncoder setup` if you have no user config yet. Re-running it upgrades the install.
    If `~/.local/bin` is not on your `PATH`, the script prints the line to add.

    Read the script first if you prefer: [install.sh](https://github.com/adderek/owncoder/blob/master/install.sh).

=== "Windows (WSL2)"

    owncoder does not run natively on Windows: the sandbox is bubblewrap/seccomp and file locking
    uses `fcntl`. In PowerShell:

    ```powershell
    irm https://raw.githubusercontent.com/adderek/owncoder/master/install.ps1 | iex
    ```

    The script checks for WSL and a Linux distribution, installs `curl`, `git`, `ripgrep` and
    `bubblewrap` inside it if `curl` is missing, and runs the Linux installer there.
    If WSL is missing, run `wsl --install` in an Administrator shell, or pass `-InstallWsl`
    to the script from an Administrator shell. `-Distro <name>` picks a distribution.

    Keep projects on the Linux filesystem (`~/code/...`), not under `/mnt/c/`. Cross-filesystem
    access is slow enough to dominate indexing time.

=== "uv"

    ```sh
    uv tool install git+https://github.com/adderek/owncoder
    ```

    Installs into an isolated environment and puts `owncoder` and `agent` on your `PATH`
    (`uv tool dir --bin` shows where).

=== "From source"

    ```sh
    git clone https://github.com/adderek/owncoder.git agent
    cd agent
    uv venv .venv
    uv pip install -e ".[dev]"
    .venv/bin/agent --version
    ```

    Name the clone directory `agent`: the test suite resolves the package from the parent directory.
    Normal use does not depend on the name.

`owncoder` and `agent` are the same program; the docs use both.

## Optional extras

Extras are declared in `pyproject.toml`. With `uv tool`, add one like this:

```sh
uv tool install --force "owncoder[lang] @ git+https://github.com/adderek/owncoder"
```

From a source checkout use `uv pip install -e ".[lang]"`.

| Extra | Adds |
|---|---|
| `lang` | tree-sitter grammars for JavaScript, TypeScript, Rust, Go, Java, Kotlin. Without them those files are chunked by lines. |
| `graph` | static dependency/call graph export (`graphifyy`). |
| `notify` | `websockets` and `cryptography` for push notifications and the relay. |
| `vault` | `cryptography` for encrypted-at-rest sessions (`agent chat --vault`). |
| `serial` | `pyserial` for UART console tools. |
| `speech` | `faster-whisper` for speech-to-text input. |
| `dev` | test dependencies (`pytest`, `httpx`, ...). |

Python, C, C++ and bash grammars ship by default.

## Verify

```sh
owncoder --version
```

Prints `owncoder 0.1.0` or the current version. Next: [Quickstart](quickstart.md).

## Upgrade and uninstall

```sh
uv tool upgrade owncoder        # or re-run the install one-liner
uv tool uninstall owncoder
```

Uninstalling does not remove `~/.config/agent/` or the per-project `.agent/` directories.

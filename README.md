# What is it?

Local-first coding agent with solid security. Runs against your own llama.cpp/vLLM/ollama, or any OpenAI-compatible API (deepseek, openai). Use it when you want a coding agent that keeps code on your machine and is built for heavy analysis of low-structure languages like assembler.

Similar to
* claude code (best for complex tasks)
* gemini cli (good even in free version)
* openai chatgpt codex (currently best offer)
* hermes (good and works locally - **use it instead of my owncoder** )
* cursor (IDE that works)
* windsurf (IDE that's on develop, but cheaper than cursor and sometimes has better features)


Normally you don't index so the agent works right away, but languages like assembler lack structure and need initial code analysis.


# Install

**Linux and macOS** — one line, no Python setup of your own:

```sh
curl -LsSf https://raw.githubusercontent.com/adderek/owncoder/master/install.sh | sh
```

It installs [uv](https://docs.astral.sh/uv/) if you do not have it, installs
owncoder into its own isolated environment (its own Python, its own
dependencies — nothing is added to your system or project environments), and
then runs the first-start wizard.

**Windows** — owncoder runs under WSL2, not natively: its shell and web tools
execute inside a bubblewrap/seccomp sandbox and its locking uses `fcntl`, none
of which exist on Windows. In PowerShell:

```powershell
irm https://raw.githubusercontent.com/adderek/owncoder/master/install.ps1 | iex
```

That sets up WSL2 if needed and runs the Linux installer inside it. Keep your
projects on the Linux filesystem (`~/code/...`), not `/mnt/c/` — cross-filesystem
access is slow enough to dominate indexing time.

Already have uv? Then just:

```sh
uv tool install owncoder        # or: uv tool install git+https://github.com/adderek/owncoder
```

Upgrade with `uv tool upgrade owncoder`, remove with `uv tool uninstall owncoder`.
Both `owncoder` and `agent` are installed as commands; they are the same program.

Prerequisites the agent shells out to: `git` (required), `ripgrep` (faster
search), and `bubblewrap` or `firejail` on Linux (the sandbox that shell and web
tools run inside — without one, those tools refuse to run).

## First start

```sh
owncoder setup    # pick provider + model, verify it, write ~/.config/agent/agent.toml
```

The wizard asks where the models are hosted — a local server
(llama.cpp / Ollama / LM Studio), OpenRouter, DeepSeek, or any other
OpenAI-compatible URL — lists the models that endpoint actually serves, sends
one test completion, and only then writes the config (mode 0600). If the API key
is already exported (`OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`,
`AGENT_LLM_API_KEY`), the config stores a reference to the variable
(`api_key = "env:OPENROUTER_API_KEY"`) rather than the key itself, so the file
stays safe to sync or paste into a bug report.

Then, in a project:

```sh
agent init    # optional: index the code
agent chat
```

Running any of `init`, `chat` or `run` with no config at all offers to run the
wizard for you.

## From source

```sh
git clone https://github.com/adderek/owncoder.git agent
cd agent
uv venv .venv
uv pip install -e ".[dev]"
```

The directory must be named `agent`: the tests resolve the package from the
parent directory (`pythonpath = [".."]`). Packaging does not care — `setup.py`
maps the repo root onto the `agent` package, which is why
`uv tool install git+...` works from any checkout name.

# Usage

```
agent commit .
```

checks the current directory (assuming it is a git repo) and creates a commit
message for it, chunking the diff the standard way.

# About

Local-first heavy coding agent meant for assembler language.

![Self description](https://adderek.github.io/owncoder/img/001-UI_and_features.png)

Key features:
* **Local-first** — point it at llama.cpp / vLLM / ollama, or any OpenAI-compatible endpoint; nothing leaves your box unless you say so
* **Security suite** — seccomp sandbox, path grants, airgap mode, prompt-injection scanning, output redaction, audit log, SBOM
* **Code understanding** — tree-sitter parsing + sqlite-vec semantic search/embeddings (CPU is enough)
* **Rich toolset** — file edit, git, shell, code/web search, checkpoints, skills
* **Interfaces** — Textual TUI, plus simple/readline text-only modes
* **Extensible** — MCP support

Layers of code indexing / retrieval (each optional, used as needed):
* **RAG** — tree-sitter splits code into chunks, the embeddings model vectorizes them into sqlite-vec; hybrid (vector + keyword) search at query time
* **Archive** — pruned chunks kept with a TTL, so old/deleted code stays searchable
* **Summarization** — the LLM writes terse descriptions per chunk, then rolls them up into a multi-level summary pyramid
* **Assembler analysis** — same LLM describe-and-rollup pyramid (up to 6 levels), tuned for low-structure code tree-sitter can't model
* **Graph** — static dependency/call graph export (graphify), no model needed
* **KB** — per-project code knowledge base (`.agent/kb`): the graph's symbols and edges, summaries that still match the code, and your saved notes attached to the code they mention. `kb_get` / `kb_callers` / `find_symbol` take plain names
* **Memory / recall** — facts, Q&A log, and session history, distilled and compacted by the LLM

While a chat is idle the agent keeps these current by itself (`agent/rag/maintainer.py`): changed files are re-indexed within seconds, the graph and KB are refreshed, and a few pending summaries are written per pass, most-used files first. It waits while a turn runs or the machine is loaded, never embeds through a localhost server (CPU embeddings can overload a desktop) and never sends code to a cloud model; `rag.auto_index`, `rag.auto_kb` and `rag.auto_describe` turn the parts off. `agent index --watch` runs the same loop without a chat.

Prompts and skills/tools are **compiled per model**: the prompt-compiler compresses static prompt files for the active (model, api) pair and caches them, and tool results are compacted by the LLM before re-entering context — smaller context, same meaning.

```
agent init  # index files
agent chat  # run agent
AGENT_UI_MODE=simple  agent chat  # if you wish text-only mode with no text-panels
agent run "do something"
agent index --stats
```

example:
`python -m agent.main chat`


## What you need more

Embeddings model:
* to index files for the agent
* it works on CPU good enough

# Configuration

`owncoder setup` writes `~/.config/agent/agent.toml` for you. To do it by hand,
copy `agent.toml.example` to the project root or to `~/.config/agent/agent.toml`

```
[llm]
base_url = "http://localhost:8080/v1"   # your llama-server or OpenAI
api_key = "local"                        # or your real API key
model = "qwen3-coder-30b"

[embeddings]
base_url = "http://localhost:8080/v1"   # embedding endpoint
model = "nomic-embed-text"
```

Or env var overrides (no file needed)

```
export AGENT_LLM_BASE_URL="https://api.openai.com/v1"
export AGENT_LLM_API_KEY="sk-..."
export AGENT_LLM_MODEL="gpt-4o"
export AGENT_EMBEDDINGS_BASE_URL="https://api.openai.com/v1"
export AGENT_EMBEDDINGS_MODEL="text-embedding-3-small"
```

## User interface

Readline mode (no Textual UI)
```
[ui]
mode = "readline"
```

or `AGENT_UI_MODE=readline agent chat`


# Screenshots

![Ask it to compare](https://adderek.github.io/owncoder/img/002-TrueComparisonQuestion.png)
![Discussion summary](https://adderek.github.io/owncoder/img/003-TrueComparisonSummary.png)
![Questions summary](https://adderek.github.io/owncoder/img/004-QuestionSummary.png)
![UI](https://adderek.github.io/owncoder/img/005-UI.png)
![Details of tool call](https://adderek.github.io/owncoder/img/006-ToolCallDetails.png)
![More details of tool call](https://adderek.github.io/owncoder/img/007-ToolCallDetails.png)
![Strenghts](https://adderek.github.io/owncoder/img/100-Strengths.png)
![Strengths - more](https://adderek.github.io/owncoder/img/101-Strengths.png)
![Weaknesses](https://adderek.github.io/owncoder/img/110-Weaknesses.png)
![Honest verdict](https://adderek.github.io/owncoder/img/200-HonestVerdict.png)


# Fancy commands if you wish to play with it more

## ollama, llama, vLLM, etc.

* ollama is simple and good, but only to run "stable" things
* llama.cpp is fast and has decent features
* vLLM is compatible and offers some unique features (and recently is a bit promoted yb AMD)

## Code structure

* owncoder/agent/ = inner agent repo, master branch (dev) - you are looking at it now
* actual toolset (tests, documentation, etc.) - not published :)

## My local setup

I have a worktree with stable version, although I would probably prefer claude/hermes/gemini cli if I broke master branch

```
git -C ~/src/owncoder-stable/agent fetch ~/src/owncoder/agent master
git -C ~/src/owncoder-stable/agent merge FETCH_HEAD
```

## To test terminal caps

`python3 scripts/terminal_probe.py --out agent.toml.probe`

My owncoder agent is meant for special modification of:
* tilix terminal
* VTE (used by tilix)
* llama.cpp


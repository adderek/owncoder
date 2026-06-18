# What is it?

Local-first coding agent (but works with openai like deepseek) with solid security

Meant for heavy analysis of languages that need it (like assembler) (xxx)

Similar to
* claude code (best for complex tasks)
* gemini cli (good even in free version)
* openai chatgpt codex (currently best offer)
* hermes (good and works locally - **use it instead of my owncoder** )
* cursor (IDE that works)
* windsurf (IDE that's on develop, but cheaper than cursor and sometimes has better features)


(xxx) Normally you don't index so agent can work rightaway, but languages like assembler lack structure and require initial code analysis


# Usage

```
git clone https://github.com/adderek/owncoder.git
cd owncoder/
pip install -e .
./.venv/bin/agent
```

or
```
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

# About

Local-first heavy coding agent meant for assembler language

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

copy agent.toml to project root or ~/.config/agent/agent.toml

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

My owmcoder agent is meant for special modification of:
* tilix terminal
* VTE (used by tilix)
* llama.cpp


# Usage

cd agent/
pip install -e .

or
```
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

agent init
agent chat
AGENT_UI_MODE=simple agent chat
agent run "do something"
agent index --stats

example:
python -m agent.main chat

# configuration

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

Readline mode (no Textual UI)
```
[ui]
mode = "readline"
```
or `AGENT_UI_MODE=readline agent chat`

# ollama

```
cd /home/adderek/src/ollama-turboquant
./go.sh
```

owncoder/agent/ = inner agent repo, master branch (dev)
owncoder-stable/agent/ = worktree of same repo but branch "stable"

When I am sure that master is stable:
```
git -C /home/adderek/src/owncoder/agent merge master
git -C /home/adderek/src/owncoder-stable/agent pull
```

or

```
git -C /home/adderek/src/owncoder/agent checkout stable
git -C /home/adderek/src/owncoder/agent merge master
git -C /home/adderek/src/owncoder/agent checkout master
```

or

```
git -C /home/adderek/src/owncoder-stable/agent fetch /home/adderek/src/owncoder/agent master
git -C /home/adderek/src/owncoder-stable/agent merge FETCH_HEAD
```

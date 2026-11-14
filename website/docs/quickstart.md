# Quickstart

Five minutes from install to a first answer about your code. You need an OpenAI-compatible
endpoint: a local llama.cpp, Ollama or LM Studio server, or a hosted API.

## 1. Install

```sh
curl -LsSf https://raw.githubusercontent.com/adderek/owncoder/master/install.sh | sh
```

Other platforms and options: [Install](install.md).

## 2. Point it at a model

```sh
owncoder setup
```

The wizard asks for a provider (local server, OpenRouter, DeepSeek, or any other OpenAI-compatible
URL), the base URL and a key, lists the models the endpoint serves, sends one test completion,
and only then writes `~/.config/agent/agent.toml` with mode 0600. If the key variable is already
exported (for example `OPENROUTER_API_KEY`), the file stores `env:OPENROUTER_API_KEY` instead of
the key. `owncoder setup --force` overwrites an existing config.

Or write the config yourself. Pick one:

=== "Local server"

    ```toml
    [models]
    default = "local"

    [models.local]
    base_url = "http://localhost:8080/v1"
    api_key  = "local"
    model    = "qwen3-coder-30b"   # an id your server serves
    local    = true
    ```

=== "Hosted API"

    ```toml
    [models]
    default = "hosted"

    [models.hosted]
    base_url = "https://api.example.com/v1"
    api_key  = "env:MY_API_KEY"     # read from the environment at start
    model    = "your-model-id"
    ```

    ```sh
    export MY_API_KEY="sk-…"
    ```

## 3. Initialize a project

```sh
cd path/to/your/project
agent init
```

`agent init` is required once per project: it creates `.agent/`, and `agent chat` refuses to start
without it. It then offers to index the code, which is what lets the agent search by meaning.
Indexing needs an embeddings model (see [Configuration](configuration.md)). To skip it for now:
`agent init --skip-index`; index later with `agent index --update`.

## 4. Chat

```sh
agent chat                  # terminal UI
agent chat --ui simple      # plain text
agent chat --ui http        # browser UI
```

See [Interfaces](interfaces.md) for the differences.

## Prompts that suit it

owncoder is built for reading large, unfamiliar codebases. Start with questions:

- `Explain how request routing works in this project and point me to the functions involved.`
- `Summarise what changed between v1.4.0 and HEAD, grouped by module.`
- `Find where configuration errors are handled and list each place with file and line.`

Replace the names with your own. The agent searches the index and reads the files it needs, then
answers with locations you can open.

## Other commands

```sh
agent run "<prompt>"        # one-shot, non-interactive
agent commit .              # generate and apply a commit message for the repo
agent index --stats         # index statistics
agent sessions --list       # saved sessions
agent --help                # everything else
```

Next: [Configuration](configuration.md).

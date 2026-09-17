"""`agent setup` — first-start wizard that writes the user config layer.

Runs before any project exists and before any config is loaded: a fresh
install has no ~/.config/agent/agent.toml at all, and every other command
needs one to know where the models live. The wizard asks three things —
provider, credential, model — verifies the answers against the endpoint, and
writes ~/.config/agent/agent.toml (0600).

Kept deliberately dependency-free (urllib + input()) so it still works when
the install is half-broken, which is exactly when a user runs it.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


CONFIG_DIR = Path.home() / ".config" / "agent"
CONFIG_PATH = CONFIG_DIR / "agent.toml"


@dataclass(frozen=True)
class Provider:
    key: str          # entry name written into [models.<key>]
    label: str
    base_url: str
    env_var: str      # "" = no credential needed
    hint: str


PROVIDERS: tuple[Provider, ...] = (
    Provider("local", "Local server (llama.cpp / Ollama / LM Studio)",
             "http://localhost:8080/v1", "",
             "no key, no network; the server must already be running"),
    Provider("openrouter", "OpenRouter", "https://openrouter.ai/api/v1",
             "OPENROUTER_API_KEY", "one key, many models, pay per token"),
    Provider("deepseek", "DeepSeek", "https://api.deepseek.com/v1",
             "DEEPSEEK_API_KEY", "cheap, strong at code"),
    Provider("custom", "Other OpenAI-compatible endpoint", "", "AGENT_LLM_API_KEY",
             "anything that serves /v1/models and /v1/chat/completions"),
)


# ── plumbing ─────────────────────────────────────────────────────────────────

def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(130)
    return answer or default


def _request(url: str, api_key: str, payload: dict | None = None,
             timeout: int = 20) -> tuple[int, dict | None, str]:
    """One JSON call. Returns (status, parsed_body, error_message)."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 method="POST" if data else "GET")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            try:
                return resp.status, json.loads(body), ""
            except ValueError:
                return resp.status, None, "response was not JSON"
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        return exc.code, None, f"HTTP {exc.code}: {detail or exc.reason}"
    except (urllib.error.URLError, OSError) as exc:
        return 0, None, f"cannot reach {url}: {exc}"


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


# ── steps ────────────────────────────────────────────────────────────────────

def _pick_provider() -> Provider:
    print("\nWhere are the models hosted?\n")
    for i, p in enumerate(PROVIDERS, 1):
        print(f"  {i}) {p.label}")
        print(f"     {p.hint}")
    while True:
        choice = _ask("\nChoice", "1")
        if choice.isdigit() and 1 <= int(choice) <= len(PROVIDERS):
            return PROVIDERS[int(choice) - 1]
        print(f"  Enter a number between 1 and {len(PROVIDERS)}.")


def _pick_base_url(provider: Provider) -> str:
    default = provider.base_url
    while True:
        url = _ask("Base URL (must end in /v1)", default)
        if url.startswith(("http://", "https://")):
            return url.rstrip("/")
        print("  Needs to start with http:// or https://")


def _pick_credential(provider: Provider, base_url: str) -> tuple[str, str]:
    """Return (value_written_to_config, key_used_for_probing).

    The two differ when the key lives in an environment variable: the config
    stores the reference "env:VAR" while probing needs the real value.
    """
    if not provider.env_var:
        # The "local server" choice means a llama.cpp/Ollama/LM Studio box —
        # loopback or LAN, either way it takes any bearer token. Keyed off the
        # provider rather than the URL so a LAN address (192.168.x, a hostname)
        # is not mistaken for a cloud endpoint and made to ask for a key.
        return "local", "local"

    env_var = provider.env_var or "AGENT_LLM_API_KEY"
    from_env = os.environ.get(env_var, "")
    if from_env:
        print(f"\nUsing ${env_var} from the environment "
              f"(...{from_env[-4:]}); the config will reference the variable, "
              f"not the key.")
        return f"env:{env_var}", from_env

    print(f"\nAPI key. Leave empty if the endpoint needs none.")
    print(f"  Export it as ${env_var} to keep it out of the config file, "
          f"then re-run `agent setup`.")
    key = _ask("API key")
    return (key or "local"), (key or "local")


def _pick_model(base_url: str, api_key: str) -> str:
    status, body, err = _request(f"{base_url}/models", api_key)
    models: list[str] = []
    if body and isinstance(body.get("data"), list):
        models = sorted(str(m.get("id", "")) for m in body["data"] if m.get("id"))

    if err:
        print(f"\n  Could not list models — {err}")
    if not models:
        print("  Type the model id by hand (e.g. deepseek-chat, "
              "qwen/qwen3-coder).")
        while True:
            name = _ask("Model id")
            if name:
                return name

    if len(models) > 30:
        print(f"\n{len(models)} models available.")
        needle = _ask("Filter by substring (empty = show first 30)").lower()
        if needle:
            models = [m for m in models if needle in m.lower()] or models

    shown = models[:30]
    print(f"\nAvailable models{' (first 30)' if len(models) > 30 else ''}:\n")
    for i, m in enumerate(shown, 1):
        print(f"  {i:>2}) {m}")
    while True:
        choice = _ask("\nPick a number, or type a model id", "1")
        if choice.isdigit() and 1 <= int(choice) <= len(shown):
            return shown[int(choice) - 1]
        if choice and not choice.isdigit():
            return choice
        print("  Not in range.")


def _smoke_test(base_url: str, api_key: str, model: str) -> bool:
    print(f"\nTesting {model} at {base_url} ...")
    status, body, err = _request(
        f"{base_url}/chat/completions", api_key,
        payload={
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the word: ok"}],
            "max_tokens": 8,
            "temperature": 0,
        },
    )
    if err:
        print(f"  FAILED — {err}")
        return False
    try:
        reply = body["choices"][0]["message"]["content"]
    except (TypeError, KeyError, IndexError):
        print(f"  FAILED — unexpected response shape: {str(body)[:200]}")
        return False
    print(f"  OK — model replied: {reply.strip()[:60]!r}")
    return True


def _render_config(provider: Provider, base_url: str, api_key: str,
                   model: str) -> str:
    is_local = base_url.startswith(("http://localhost", "http://127.0.0.1"))
    return f"""# Written by `agent setup`. Hand-edit freely; it is never rewritten.
#
# This is the user layer. Per-project overrides go in <project>/agent.toml,
# per-machine ones in ~/.config/agent/agent.<hostname>.toml.
# Full reference: agent.toml.example in the installed package.

[models]
default = "{provider.key}"

[models.{provider.key}]
base_url          = "{_toml_escape(base_url)}"
api_key           = "{_toml_escape(api_key)}"
model             = "{_toml_escape(model)}"
max_output_tokens = 8192
temperature       = 0.7
local             = {str(is_local).lower()}
"""


def _write_config(text: str, force: bool) -> bool:
    if CONFIG_PATH.exists() and not force:
        print(f"\n{CONFIG_PATH} already exists. Re-run with --force to replace it.")
        print("Proposed content:\n")
        print(text)
        return False
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # 0600 before the write, not after: the key must never exist on disk
    # world-readable, not even for the instant between create and chmod.
    fd = os.open(CONFIG_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.chmod(CONFIG_PATH, 0o600)
    print(f"\nWrote {CONFIG_PATH} (mode 0600)")
    return True


# ── entry point ──────────────────────────────────────────────────────────────

def cmd_setup(args) -> int:
    if not sys.stdin.isatty():
        print("agent setup needs an interactive terminal.", file=sys.stderr)
        return 1

    print("agent setup — one-time model configuration")

    provider = _pick_provider()
    base_url = _pick_base_url(provider)
    stored_key, probe_key = _pick_credential(provider, base_url)
    model = _pick_model(base_url, probe_key)

    if not _smoke_test(base_url, probe_key, model) and not getattr(args, "force", False):
        print("\nNot writing a config that does not work.")
        print("Fix the endpoint (is the server up? is the key right?) and "
              "re-run `agent setup`.")
        return 1

    if not _write_config(_render_config(provider, base_url, stored_key, model),
                         force=getattr(args, "force", False)):
        return 1

    print("\nNext: cd into a project and run `agent init`, then `agent chat`.")
    return 0


def user_config_exists() -> bool:
    """True when any user-layer config file is present.

    Mirrors the user-layer half of agent.config.loader.load_config's search
    order; it cannot call into it because the point is to decide whether
    loading is worth attempting at all.
    """
    for name in ("agent.toml", "agent.yaml", "agent.yml"):
        if (CONFIG_DIR / name).exists():
            return True
    return bool(list(CONFIG_DIR.glob("agent.*.toml"))) if CONFIG_DIR.is_dir() else False

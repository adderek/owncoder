from __future__ import annotations

from typing import Any, Callable

_registry: dict[str, Callable] = {}
_schemas: list[dict] = []
_tools_loaded: bool = False


def register(name: str, schema: dict):
    def decorator(fn: Callable) -> Callable:
        _registry[name] = fn
        # Avoid duplicates if the decorator fires more than once (e.g. during testing).
        if not any(s["function"]["name"] == name for s in _schemas):
            _schemas.append({"type": "function", "function": {**schema, "name": name}})
        return fn
    return decorator


def get_tool(name: str) -> Callable | None:
    return _registry.get(name)


def get_schemas() -> list[dict]:
    return list(_schemas)


def load_all_tools(config=None, store=None, embedder=None) -> None:
    global _tools_loaded
    from agent.tools import files, shell, git, search
    # Always update config/store/embedder so a second call refreshes dependencies.
    files.setup(config)
    shell.setup(config)
    git.setup(config)
    search.setup(config, store, embedder)
    _tools_loaded = True

from __future__ import annotations

from typing import Any, Callable

_registry: dict[str, Callable] = {}
_schemas: list[dict] = []


def register(name: str, schema: dict):
    def decorator(fn: Callable) -> Callable:
        _registry[name] = fn
        _schemas.append({"type": "function", "function": {**schema, "name": name}})
        return fn
    return decorator


def get_tool(name: str) -> Callable | None:
    return _registry.get(name)


def get_schemas() -> list[dict]:
    return list(_schemas)


def load_all_tools(config=None, store=None, embedder=None) -> None:
    from agent.tools import files, shell, git, search
    files.setup(config)
    shell.setup(config)
    git.setup(config)
    search.setup(config, store, embedder)

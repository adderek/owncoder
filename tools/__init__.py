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


def load_all_tools(config=None, store=None, embedder=None, asm_store=None, data_provider=None) -> None:
    global _tools_loaded
    from agent.tools import files, shell, git, search, analyze_asm, edit_file, recall  # noqa: F401
    from agent.tools.rules import load_rules

    # Wrap raw objects in DataProvider when caller hasn't provided one.
    if data_provider is None:
        from agent.data_provider import LocalDataProvider
        data_provider = LocalDataProvider(store=store, embedder=embedder, asm_store=asm_store, config=config)

    # Load rule files (.agent.ignore, .agent.ro, .agent.config, etc.)
    working_dir = config.tools.working_dir if config else "."
    load_rules(working_dir)

    # Initialize security harness before any tool runs.
    if config is not None:
        from agent.security import policy as _sec_policy, fs as _sec_fs
        _sec_policy.setup(config)
        try:
            _sec_fs.init_root_pin()
        except Exception:
            # Root pin failure is fatal in theory, but we log and continue
            # so existing test fixtures that use tmp dirs keep working.
            import logging
            logging.getLogger(__name__).warning(
                "security.fs.init_root_pin failed", exc_info=True
            )

    # edit_file schema depends on [edit] config → register after load_rules.
    edit_file._register_edit_file()

    # Always update config/store/embedder so a second call refreshes dependencies.
    files.setup(config)
    shell.setup(config)
    git.setup(config)
    search.setup(config, data_provider)
    analyze_asm.setup(config, data_provider)

    if config is not None and getattr(config.planning, "increments_enabled", False):
        from agent.tools import increment_tools  # noqa: F401
        increment_tools.setup(config)

    if config is not None and getattr(getattr(config, "parallel", None), "enabled", False):
        from agent.tools import parallel  # noqa: F401
        parallel.setup(config, data_provider)

    _tools_loaded = True

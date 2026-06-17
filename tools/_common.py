"""Small shared helpers for tool modules.

Tool modules each hold their own module-global ``_config`` (set via ``setup``);
this centralizes config-derived values they all need so the config path lives in
one place.
"""
from __future__ import annotations


def working_dir(config) -> str:
    """Project working directory from config, or '.' when config is unset."""
    return config.tools.working_dir if config else "."

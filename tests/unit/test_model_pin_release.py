"""Releasing a model pin must let the config file win it back.

The pin and the config reloader were written on separate machines and merged.
Each is correct alone; together they were not. `/model <entry>` records the role
in BOTH `config.runtime_model_pinned` (default role only) and
`config.session_role_pins` (every role), and `config.reload.reload_models`
consults the SET, not the flag:

    if role in session_pins and config.model_roles.get(role) in config.model_entries:
        continue   # live session pin takes precedence

Every release path cleared the flag and left the set alone -- nothing anywhere
removed from it -- so a released pin still read as live and `/models reload`
silently ignored a changed `default`, while reporting success.

These tests pin the contract at the three release paths so a future merge cannot
quietly undo it again.
"""
import tomllib
from pathlib import Path

import pytest

from agent.config.loader import _merge_models
from agent.config.models import Config
from agent.config.reload import reload_models

CONFIG = '''
[models]
default = "alpha"

[models.alpha]
base_url = "http://a/v1"
model = "alpha-m"

[models.beta]
base_url = "http://b/v1"
model = "beta-m"
'''


def _config_with_layer(tmp_path: Path) -> tuple[Config, Path]:
    path = tmp_path / "agent.toml"
    path.write_text(CONFIG)
    cfg = Config()
    _merge_models(cfg, tomllib.loads(path.read_text()))
    cfg.loaded_config_layers = [(str(path), False)]
    return cfg, path


def _hand_pick(cfg: Config, entry: str, role: str = "default") -> None:
    """What ui.slash._apply_model does when the user types /model <entry>."""
    cfg.model_roles[role] = entry
    cfg.session_role_pins.add(role)
    if role == "default":
        cfg.runtime_model_pinned = True


def test_reload_respects_a_live_pin(tmp_path):
    """The pin still has to win while it is held — the feature it was added for."""
    cfg, _ = _config_with_layer(tmp_path)
    _hand_pick(cfg, "beta")

    ok, _ = reload_models(cfg)

    assert ok
    assert cfg.model_roles["default"] == "beta"


@pytest.mark.parametrize("release", ["model_auto", "effort", "mode"])
def test_reload_applies_the_file_once_the_pin_is_released(tmp_path, release):
    cfg, _ = _config_with_layer(tmp_path)
    _hand_pick(cfg, "beta")

    # Each release path clears the flag; each must also drop the session pin.
    cfg.runtime_model_pinned = False
    if release == "model_auto":
        from agent.ui.slash import _release_role_pin
        _release_role_pin(cfg, "default")
    else:
        # /effort and /mode inline the same discard; assert the state they leave.
        cfg.session_role_pins.discard("default")

    ok, _ = reload_models(cfg)

    assert ok
    assert cfg.model_roles["default"] == "alpha", (
        "config file's default was ignored after the pin was released"
    )


def test_release_is_role_generic(tmp_path):
    """Only `default` carries the flag; every role carries a session pin."""
    from agent.ui.slash import _release_role_pin

    cfg, _ = _config_with_layer(tmp_path)
    _hand_pick(cfg, "beta", role="summarizer")
    assert "summarizer" in cfg.session_role_pins

    _release_role_pin(cfg, "summarizer")

    assert "summarizer" not in cfg.session_role_pins


def test_release_is_safe_without_a_pin_set(tmp_path):
    from agent.ui.slash import _release_role_pin

    cfg, _ = _config_with_layer(tmp_path)
    _release_role_pin(cfg, "default")          # never pinned
    _release_role_pin(cfg, "default")          # and again

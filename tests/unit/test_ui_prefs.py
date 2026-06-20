import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from agent.ui.prefs import load_prefs, save_prefs, get_prefs_path


@pytest.fixture
def tmp_prefs_dir(tmp_path):
    """Fixture to provide a temporary directory for prefs."""
    prefs_dir = tmp_path / ".agent"
    prefs_dir.mkdir()
    return prefs_dir


def test_get_prefs_path():
    assert get_prefs_path() == Path(".agent/ui_prefs.json")


def test_save_and_load_prefs(tmp_path, monkeypatch):
    # Mock get_prefs_path to use a temporary file
    prefs_file = tmp_path / "ui_prefs.json"
    monkeypatch.setattr("agent.ui.prefs.get_prefs_path", lambda: prefs_file)

    test_data = {"chat_wrap": "wrap", "other": 123}

    # Test saving
    save_prefs(test_data)
    assert prefs_file.exists()
    assert json.loads(prefs_file.read_text()) == test_data

    # Test loading
    loaded_data = load_prefs()
    assert loaded_data == test_data


def test_load_prefs_empty(monkeypatch):
    # Mock get_prefs_path to a non-existent file
    monkeypatch.setattr(
        "agent.ui.prefs.get_prefs_path", lambda: Path("non_existent.json")
    )
    assert load_prefs() == {}


def test_load_prefs_corrupt(tmp_path, monkeypatch):
    # Mock get_prefs_path to a corrupt file
    prefs_file = tmp_path / "corrupt.json"
    prefs_file.write_text("not valid json")
    monkeypatch.setattr("agent.ui.prefs.get_prefs_path", lambda: prefs_file)

    assert load_prefs() == {}


@pytest.mark.parametrize("payload", ["null", "[]", '"a string"', "42", "true"])
def test_load_prefs_valid_but_non_object_returns_empty_dict(payload, tmp_path, monkeypatch):
    # Valid JSON that isn't an object must not leak through: callers do
    # prefs.get(...) (e.g. chat startup) and would crash on a list/None/str.
    prefs_file = tmp_path / "nonobj.json"
    prefs_file.write_text(payload)
    monkeypatch.setattr("agent.ui.prefs.get_prefs_path", lambda: prefs_file)

    result = load_prefs()
    assert result == {}
    # Contract: always a dict, so .get is always safe.
    assert result.get("anything") is None

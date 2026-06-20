import json
from pathlib import Path
from typing import Any, Dict


def get_prefs_path() -> Path:
    return Path(".agent/ui_prefs.json")


def load_prefs() -> Dict[str, Any]:
    path = get_prefs_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Must be a JSON object: a valid but non-object prefs file (null,
            # [], "str", a number) would otherwise be returned as-is, and
            # callers doing prefs.get(...) — e.g. the chat startup path — would
            # raise AttributeError. Honor the Dict return contract.
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_prefs(prefs: Dict[str, Any]) -> None:
    path = get_prefs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prefs, indent=2) + "\n", encoding="utf-8")

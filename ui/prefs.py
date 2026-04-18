import json
from pathlib import Path
from typing import Any, Dict


def get_prefs_path() -> Path:
    return Path(".agent/ui_prefs.json")


def load_prefs() -> Dict[str, Any]:
    path = get_prefs_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_prefs(prefs: Dict[str, Any]) -> None:
    path = get_prefs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prefs, indent=2) + "\n", encoding="utf-8")

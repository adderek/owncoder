"""Tilix OSC-777 fold escape sequences (custom patched tilix/VTE).

Wraps terminal scrollback regions in foldable sections. Requires the patched
tilix from ~/src/tilix (see its doc/FOLD.md for the protocol); other terminals
ignore unknown OSC 777 subcommands, but emission is still opt-in via
``ui.tilix_folds`` because the fold viewport changes terminal behaviour.

Protocol:
    ESC ] 777 ; tilix-fold-start ; id=X ; title=T ; group=G  BEL
    ESC ] 777 ; tilix-fold-end   ; id=X ; summary=S ; status=success|warning|error  BEL

Only meaningful for the plain readline UI (linear scrollback). The Textual UI
runs on the alternate screen and manages folding itself.
"""
from __future__ import annotations

import re
import sys

# Values ride inside an OSC string with ';'-separated key=value params:
# strip control chars (would terminate/corrupt the OSC) and ';' (param split).
_UNSAFE = re.compile(r"[\x00-\x1f\x7f;]")

_VALID_STATUS = ("success", "warning", "error")


def _sanitize(value: str, limit: int = 120) -> str:
    v = _UNSAFE.sub(" ", str(value or "")).strip()
    return v[:limit]


def _emit(subcommand: str, params: list[str]) -> None:
    try:
        sys.stdout.write(f"\033]777;{subcommand};" + ";".join(params) + "\007")
        sys.stdout.flush()
    except Exception:
        pass


def fold_start(fold_id: str, title: str = "", group: str = "") -> None:
    """Open a foldable section. *group* auto-collapses the previous fold in it."""
    params = [f"id={_sanitize(fold_id, 40)}"]
    if title:
        params.append(f"title={_sanitize(title)}")
    if group:
        params.append(f"group={_sanitize(group, 40)}")
    _emit("tilix-fold-start", params)


def fold_end(fold_id: str, summary: str = "", status: str = "") -> None:
    """Close a foldable section. *summary* shows inline when collapsed."""
    params = [f"id={_sanitize(fold_id, 40)}"]
    if summary:
        params.append(f"summary={_sanitize(summary)}")
    if status in _VALID_STATUS:
        params.append(f"status={status}")
    _emit("tilix-fold-end", params)

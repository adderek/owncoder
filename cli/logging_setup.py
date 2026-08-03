from __future__ import annotations

import logging
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path


def _write_exception_dump(
    exc: BaseException,
    argv: list[str] | None = None,
    config=None,
    log_path: Path | None = None,
) -> Path | None:
    import platform

    try:
        from agent.security import vault
        if not vault.persist_allowed():
            # A crash dump carries the command line, config and the tail of the
            # log — an off-the-record session leaves none of it behind.
            return None
        if config is not None:
            dump_dir = Path(config.tools.working_dir) / config.tools.agent_dir
        else:
            dump_dir = Path(".agent")
        dump_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        dump_path = dump_dir / f"exception-{ts}.dump"

        lines: list[str] = []
        lines.append("=== Exception Dump ===")
        lines.append(f"Timestamp : {datetime.now().isoformat(timespec='seconds')}")
        lines.append(f"Python    : {sys.version}")
        lines.append(f"Platform  : {platform.platform()}")
        lines.append(f"Command   : {' '.join(argv or sys.argv)}")
        lines.append("")
        lines.append("=== Traceback ===")
        lines.append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip())
        lines.append("")

        if config is not None:
            lines.append("=== Config ===")
            try:
                lines.append(f"model      : {config.llm.model}")
                lines.append(f"base_url   : {config.llm.base_url}")
                lines.append(f"working_dir: {config.tools.working_dir}")
                lines.append(f"agent_dir  : {config.tools.agent_dir}")
                lines.append(f"ctx_window : {config.llm.ctx_window}")
            except Exception as ce:
                lines.append(f"(error reading config: {ce})")
            lines.append("")

        if log_path is not None and log_path.exists() and not vault.encrypting():
            lines.append("=== Recent Log (last 60 lines) ===")
            try:
                log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                lines.extend(log_lines[-60:])
            except Exception as le:
                lines.append(f"(error reading log: {le})")
            lines.append("")

        vault.write_text(dump_path, "\n".join(lines) + "\n")
        return dump_path
    except Exception:
        return None


class _VaultFileHandler(logging.Handler):
    """File handler that seals each record, for vault-mode sessions.

    Debug logs quote prompts, tool arguments and file contents, so they are as
    sensitive as the transcript. Records go through vault.append_jsonl as sealed
    frames rather than through a rotating text file; there is no rotation, which
    is a deliberate simplification — a vault session is a working session, not a
    long-lived daemon.

    Read it back with ``agent vault log`` (agent/cli/vault_cli.py).
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self._emitting = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        from agent.security import vault
        if not vault.encrypting():
            return
        # vault logs its own failures; without this guard a failing seal would
        # emit a record that tries to seal, and so on.
        if getattr(self._emitting, "active", False):
            return
        self._emitting.active = True
        try:
            vault.append_jsonl(self.path, {
                "ts": record.created,
                "level": record.levelname,
                "name": record.name,
                "msg": self.format(record),
            })
        except Exception:      # logging must never take the process down
            self.handleError(record)
        finally:
            self._emitting.active = False


def _default_stderr_level(logs_cfg, ui_mode: str | None) -> str:
    """Resolve the stderr threshold, quieter when the UI is not the terminal.

    In HTTP mode the terminal is a supervision console: the user reads the
    browser, so routine warnings there are noise nobody acts on (they are still
    in the file log, and operator-facing ones go through agent/ui_notice.py).
    An explicit ``logs.stderr_level`` in agent.toml always wins; "explicit"
    means "differs from the dataclass default", which is the best signal the
    loader leaves behind.
    """
    from agent.config.models import LogsConfig

    configured = getattr(logs_cfg, "stderr_level", None)
    if configured and configured.upper() != LogsConfig.stderr_level.upper():
        return configured.upper()
    if ui_mode == "http":
        return "ERROR"
    return (configured or LogsConfig.stderr_level).upper()


def _setup_logging(agent_dir: str | None = None, logs_cfg=None,
                   ui_mode: str | None = None) -> None:
    """Attach handlers for the current privacy mode.

    Called again by ``Agent.set_session_mode`` when the mode changes mid-session,
    so a ``/incognito`` typed at turn 5 detaches the file handler there and then
    rather than at the next start.
    """
    from logging.handlers import RotatingFileHandler
    from agent.security import vault

    log_dir = Path(agent_dir) if agent_dir else Path(".agent")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "agent.log"

    level_name = (getattr(logs_cfg, "level", None) or "DEBUG").upper()
    stderr_level = _default_stderr_level(logs_cfg, ui_mode)
    max_bytes = getattr(logs_cfg, "max_bytes", 20 * 1024 * 1024)
    backup_count = getattr(logs_cfg, "backup_count", 5)
    sources = getattr(logs_cfg, "sources", {}) or {}

    root = logging.getLogger()
    root.setLevel(getattr(logging, level_name, logging.DEBUG))

    # Re-entrant: drop the handlers this function owns before adding new ones,
    # so a mode switch replaces them instead of logging to both. Handlers set up
    # by anything else (tests, embedders) are left alone.
    for handler in list(root.handlers):
        if getattr(handler, "_owncoder", False):
            root.removeHandler(handler)
            handler.close()

    fh: logging.Handler | None = None
    if vault.encrypting():
        fh = _VaultFileHandler(log_dir / "agent.log.jsonl")
    elif vault.persist_allowed():
        fh = RotatingFileHandler(log_path, maxBytes=max_bytes,
                                 backupCount=backup_count, encoding="utf-8")
    # else: incognito / private — no file handler at all. stderr still works.
    if fh is not None:
        fh.setLevel(getattr(logging, level_name, logging.DEBUG))
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
        fh._owncoder = True
        root.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(getattr(logging, stderr_level, logging.WARNING))
    sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    sh._owncoder = True
    root.addHandler(sh)

    for source_name, source_level in sources.items():
        lvl = getattr(logging, str(source_level).upper(), None)
        if lvl is not None:
            logging.getLogger(source_name).setLevel(lvl)

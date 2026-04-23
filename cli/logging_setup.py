from __future__ import annotations

import logging
import sys
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

        if log_path is not None and log_path.exists():
            lines.append("=== Recent Log (last 60 lines) ===")
            try:
                log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                lines.extend(log_lines[-60:])
            except Exception as le:
                lines.append(f"(error reading log: {le})")
            lines.append("")

        dump_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return dump_path
    except Exception:
        return None


def _setup_logging(agent_dir: str | None = None, logs_cfg=None) -> None:
    from logging.handlers import RotatingFileHandler

    log_dir = Path(agent_dir) if agent_dir else Path(".agent")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "agent.log"

    level_name = (getattr(logs_cfg, "level", None) or "DEBUG").upper()
    stderr_level = (getattr(logs_cfg, "stderr_level", None) or "WARNING").upper()
    max_bytes = getattr(logs_cfg, "max_bytes", 20 * 1024 * 1024)
    backup_count = getattr(logs_cfg, "backup_count", 5)
    sources = getattr(logs_cfg, "sources", {}) or {}

    root = logging.getLogger()
    root.setLevel(getattr(logging, level_name, logging.DEBUG))

    fh = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    fh.setLevel(getattr(logging, level_name, logging.DEBUG))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(getattr(logging, stderr_level, logging.WARNING))
    sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(sh)

    for source_name, source_level in sources.items():
        lvl = getattr(logging, str(source_level).upper(), None)
        if lvl is not None:
            logging.getLogger(source_name).setLevel(lvl)

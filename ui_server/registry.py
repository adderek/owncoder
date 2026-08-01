"""Project registry for multi-project UI access (MULTI_PROJECT_PLAN s3).

Holds the set of known projects and answers "what projects exist" and "which
process serves project X". It is a *registry + proxy map*, NOT a manager: it
never spawns or stops project processes (D11, discover-only).

Project identity:
  - `project_id` is stable across restarts and unique between hosts:
        sha256(canonical_workdir + "\\x00" + host_id)[:16]
    `host_id` is a random, persisted per-host value (avoids leaking the
    filesystem path and survives directory moves only via canonical path).
  - Local projects are registered at startup / explicit add, canonicalized,
    and gated by an explicit directory whitelist (paths are never set over the
    network — §4.2, §7.5).
  - Remote projects are ingested from relay presence/roster frames (§4.4):
    {name, host, project_id, label, workdir_hash} — no raw path on the wire.

Lifecycle (D11): a live local project is one whose pidfile points at a live
PID (`_pid_alive`). Stale pidfiles are reaped so /api/projects never fills with
ghosts. Remote projects track presence with a timeout: no keep-alive within
`PRESENCE_TIMEOUT_S` ⇒ treated as left (D9).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

# Host identity: random, persisted. Shared by all projects on this machine so
# project_id is unique even across hosts that happen to share a workdir path.
_HOST_ID_FILE = "owncoder-host-id"


def load_host_id(base_dir: str | os.PathLike | None = None) -> str:
    """Return this host's stable id, creating+persisting one if absent."""
    base = Path(base_dir or os.environ.get("XDG_CONFIG_HOME") or (
        Path.home() / ".config"))
    path = base / _HOST_ID_FILE
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except OSError:
        pass
    new = os.urandom(16).hex()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new, encoding="utf-8")
    except OSError:
        pass  # read-only config dir → ephemeral host id for this process
    return new


def project_id(workdir: str, host_id: str | None = None) -> str:
    """Stable, host-unique id for a canonical workdir (16 hex chars = 64 bits)."""
    h = host_id or load_host_id()
    canonical = os.path.realpath(workdir)
    return hashlib.sha256(f"{canonical}\x00{h}".encode("utf-8")).hexdigest()[:16]


def _pid_alive(pid: int) -> bool:
    import errno

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


# Presence timeout for remote projects: how long a remote peer may be silent
# before it is treated as left. Relay has no explicit keep-alive today, so this
# is the upper bound on disconnect detection (D9).
PRESENCE_TIMEOUT_S = 300.0


@dataclass
class ProjectRecord:
    project_id: str
    label: str                 # local: canonical path; remote: user-provided
    host: str = "local"        # "local" or a remote host name
    workdir: str = ""          # local only; never sent to remote peers
    workdir_hash: str = ""     # remote only
    status: str = "unknown"    # local | remote | disconnected
    pid: int = 0               # local: serving process pid
    port: int = 0              # local: serving port
    last_seen: float = field(default_factory=time.time)

    def to_dict(self, *, for_wire: bool = False) -> dict:
        """Serialize. `for_wire` drops local-only fields (path, pid, port)."""
        d = {
            "project_id": self.project_id,
            "label": self.label,
            "host": self.host,
            "status": self.status,
        }
        if for_wire:
            d["workdir_hash"] = self.workdir_hash
        else:
            d["workdir"] = self.workdir
            d["pid"] = self.pid
            d["port"] = self.port
            d["last_seen"] = self.last_seen
        return d


class ProjectRegistry:
    """Known projects + proxy map. Not responsible for process lifecycle."""

    def __init__(
        self,
        *,
        whitelist: "list[str] | None" = None,
        host_id: str | None = None,
        presence_timeout: float = PRESENCE_TIMEOUT_S,
    ) -> None:
        # Canonical whitelist of directories that may host a project. Empty
        # list = no projects allowed at all; None = unset (caller decides).
        self._whitelist: "list[str] | None" = None
        if whitelist is not None:
            self._whitelist = [os.path.realpath(w) for w in whitelist]
        self._host_id = host_id or load_host_id()
        self._presence_timeout = presence_timeout
        self._local: "dict[str, ProjectRecord]" = {}     # project_id -> record
        self._remote: "dict[str, ProjectRecord]" = {}    # project_id -> record

    # ── local registration ────────────────────────────────────────────────

    def allowed(self, workdir: str) -> bool:
        """Whether a canonicalized workdir is permitted by the whitelist."""
        if self._whitelist is None:
            return True
        canonical = os.path.realpath(workdir)
        return any(
            canonical == w or canonical.startswith(w + os.sep)
            for w in self._whitelist
        )

    def register_local(
        self, workdir: str, *, pid: int = 0, port: int = 0
    ) -> "ProjectRecord | None":
        """Register a local project. Returns None if not whitelisted."""
        if not self.allowed(workdir):
            return None
        canonical = os.path.realpath(workdir)
        pid_ = project_id(canonical, self._host_id)
        rec = self._local.get(pid_)
        if rec is None:
            rec = ProjectRecord(
                project_id=pid_,
                label=canonical,
                host="local",
                workdir=canonical,
                status="local",
            )
            self._local[pid_] = rec
        rec.pid = pid or rec.pid
        rec.port = port or rec.port
        rec.last_seen = time.time()
        rec.status = "local"
        return rec

    def unregister_local(self, workdir: str) -> None:
        self._local.pop(project_id(workdir, self._host_id), None)

    # ── remote ingestion from presence ────────────────────────────────────

    def apply_presence(self, frame: dict) -> "list[ProjectRecord]":
        """Ingest a relay presence/roster frame; return current remote list.

        Peers without a project_id are not projects (they may be plain agents
        or clients) and are ignored. Disappeared peers are marked left.
        """
        peers = frame.get("peers") if isinstance(frame, dict) else None
        if not isinstance(peers, dict):
            return list(self._remote.values())
        now = time.time()
        seen: "set[str]" = set()
        for name, meta in peers.items():
            if not isinstance(meta, dict):
                continue
            pid_ = meta.get("project_id")
            if not isinstance(pid_, str) or not pid_:
                continue  # not a project peer (plain agent / client)
            seen.add(pid_)
            rec = self._remote.get(pid_)
            if rec is None:
                rec = ProjectRecord(
                    project_id=pid_,
                    label=meta.get("label") or name,
                    host=meta.get("host") or name,
                    workdir_hash=meta.get("workdir_hash", ""),
                    status="remote",
                )
                self._remote[pid_] = rec
            rec.last_seen = now
            rec.status = "remote"
            if meta.get("label"):
                rec.label = meta["label"]
            if meta.get("host"):
                rec.host = meta["host"]
        # Mark peers that dropped off the roster as disconnected.
        for pid_ in list(self._remote):
            if pid_ not in seen and now - self._remote[pid_].last_seen > self._presence_timeout:
                self._remote[pid_].status = "disconnected"
        return self.remote_projects()

    # ── queries ───────────────────────────────────────────────────────────

    def local_projects(self) -> "list[ProjectRecord]":
        return list(self._local.values())

    def remote_projects(self) -> "list[ProjectRecord]":
        return list(self._remote.values())

    def projects(self) -> "list[ProjectRecord]":
        return self.local_projects() + self.remote_projects()

    def get(self, project_id_: str) -> "ProjectRecord | None":
        return self._local.get(project_id_) or self._remote.get(project_id_)

    def is_remote(self, project_id_: str) -> bool:
        return project_id_ in self._remote

    def reap_stale_local(self) -> int:
        """Drop local records whose pid is dead. Returns number removed."""
        dead = [
            rec.project_id for rec in self._local.values()
            if rec.pid and not _pid_alive(rec.pid)
        ]
        for pid_ in dead:
            self._local.pop(pid_, None)
        return len(dead)

    def to_dict(self, *, for_wire: bool = False) -> dict:
        return {
            "projects": [
                rec.to_dict(for_wire=for_wire) for rec in self.projects()
            ],
            "host_id": self._host_id,
        }

    def save(self, path: str | os.PathLike) -> None:
        """Persist the registry (local records) to a JSON file."""
        data = {
            "host_id": self._host_id,
            "local": [rec.to_dict() for rec in self._local.values()],
        }
        p = Path(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(p)
        except OSError:
            pass

    def load(self, path: str | os.PathLike) -> None:
        """Restore local records from a persisted registry file."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data.get("host_id"), str):
            self._host_id = data["host_id"]
        for item in data.get("local", []):
            if not isinstance(item, dict) or not item.get("project_id"):
                continue
            self._local[item["project_id"]] = ProjectRecord(
                project_id=item["project_id"],
                label=item.get("label", item.get("workdir", "")),
                host="local",
                workdir=item.get("workdir", ""),
                status=item.get("status", "local"),
                pid=item.get("pid", 0),
                port=item.get("port", 0),
                last_seen=item.get("last_seen", time.time()),
            )

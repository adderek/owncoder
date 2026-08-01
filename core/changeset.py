"""What a round changed, as a foldable summary rather than a wall of journal.

A round's action journal is the right thing for a technical reader watching the
agent work, and the wrong thing for someone who only wants to know what came out
of it. This builds the other half: a structured changeset per round —
``3 files changed, +41 -17`` — that unfolds to a file list and then to a diff.

Source of truth is core/checkpoint.py's edit journal, **not** ``git diff``.
The journal already records the exact pre-image of every successful agent edit,
so bounding it by the journal position at round start gives exact per-round
attribution. Deriving the same thing from git would instead report every
uncommitted change in those files (including earlier rounds and the user's own
editing), report nothing for newly created untracked files, and produce nothing
at all outside a git repo.

Three tiers of initial disclosure, chosen from the size of the change:

    inline   small enough to just show the diff
    list     one row per file, each unfolding to its diff
    count    one line; unfolds to the file list, then to a diff

Rendering is always lazy — nothing here draws anything. Collection is eager,
and has to be: the pre-images live in the journal and the working tree keeps
moving, so "compute it later" silently produces the wrong answer.
"""
from __future__ import annotations

import difflib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

# Tools whose arguments name files they are about to change.
MUTATING_TOOLS = ("write_file", "patch_file", "edit_file")

# A file with a NUL byte in its head is not something to diff.
_BINARY_SNIFF_BYTES = 8192


@dataclass
class Limits:
    """Thresholds for tiering and for how much diff text to keep.

    Defaults live here rather than in the config dataclass so this module stays
    usable (and testable) without a Config; the ``[ui.changeset]`` section
    overrides them via :func:`limits_from_config`.
    """
    inline_max_files: int = 3
    inline_max_lines: int = 80
    list_max_files: int = 50
    max_diff_bytes: int = 262144
    max_total_bytes: int = 4194304
    context_lines: int = 3


def limits_from_config(config) -> Limits:
    """Read ``[ui.changeset]`` when present, else defaults. Never raises."""
    section = getattr(getattr(config, "ui", None), "changeset", None)
    if section is None:
        return Limits()
    values = {}
    for f in Limits.__dataclass_fields__:
        raw = getattr(section, f, None)
        if raw is None:
            continue
        try:
            values[f] = int(raw)
        except (TypeError, ValueError):
            logger.debug("changeset: ignoring non-numeric limit %s=%r", f, raw)
    return Limits(**values)


@dataclass
class FileChange:
    path: str
    added: int = 0
    removed: int = 0
    status: str = "modified"          # "added" | "modified" | "deleted"
    diff: str | None = None           # unified diff; None when spilled or skipped
    diff_ref: str | None = None       # file holding the diff, relative to spill_dir
    binary: bool = False
    truncated: bool = False           # diff too large to keep inline
    foreign_edit: bool = False        # somebody else wrote this file too
    foreign_actors: list[str] = field(default_factory=list)  # empty → unattributed

    @property
    def churn(self) -> int:
        return self.added + self.removed

    def note(self) -> str:
        """Human-readable warning for this file, or "" when nothing is wrong."""
        if not self.foreign_edit:
            return ""
        if self.foreign_actors:
            return "also edited by " + ", ".join(self.foreign_actors)
        return ("changed outside any agent (user or external tool); "
                "diff shown is this agent's edit only")


@dataclass
class Changeset:
    turn_id: int = 0
    files: list[FileChange] = field(default_factory=list)
    tier: str = "count"               # "inline" | "list" | "count"
    truncated: bool = False           # diff capture stopped at max_total_bytes
    prose: str = ""                   # one-line model-generated intent summary, opt-in

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_added(self) -> int:
        return sum(f.added for f in self.files)

    @property
    def total_removed(self) -> int:
        return sum(f.removed for f in self.files)

    @property
    def foreign(self) -> list[FileChange]:
        return [f for f in self.files if f.foreign_edit]

    def __bool__(self) -> bool:
        return bool(self.files)

    def headline(self) -> str:
        n = self.file_count
        head = f"{n:,} file{'' if n == 1 else 's'} changed"
        if self.total_added or self.total_removed:
            head += f", +{self.total_added:,} -{self.total_removed:,}"
        return head


def paths_from_tool_call(name: str, args) -> list[str]:
    """Files a mutating tool call is about to touch, in call order, deduped.

    The single definition of this parse. It was copied into core/agent.py, the
    Textual event handler and app.js, with a test pinning the copies together;
    callers import this instead.
    """
    if name not in MUTATING_TOOLS or not args:
        return []
    try:
        parsed = json.loads(args) if isinstance(args, str) else args
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, dict):
        return []
    if name == "edit_file":
        raw = [c.get("path") for c in (parsed.get("chunks") or []) if isinstance(c, dict)]
        # edit_file also accepts a flat path+anchor+replacement form.
        raw.append(parsed.get("path"))
    else:
        raw = [parsed.get("path")]
    out: list[str] = []
    for p in raw:
        if isinstance(p, str) and p and p not in out:
            out.append(p)
    return out


def open_window() -> int:
    """Journal position now. Pass the result to :func:`collect` at round end."""
    from agent.core.checkpoint import current_seq
    return current_seq()


def _read_current(root: Path, path: str) -> tuple[str | None, bool]:
    """(text, is_binary) of *path* on disk now. text None → the file is gone."""
    try:
        fpath = Path(path) if Path(path).is_absolute() else root / path
        if not fpath.is_file():
            return None, False
        raw = fpath.read_bytes()
    except OSError:
        return None, False
    if b"\x00" in raw[:_BINARY_SNIFF_BYTES]:
        return None, True
    return raw.decode("utf-8", errors="replace"), False


def _unified(path: str, before: str, after: str, context: int) -> tuple[str, int, int]:
    """(diff text, added, removed). Counts come from the diff, so a line that
    moved unchanged is not counted twice."""
    lines = list(difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}", n=context,
    ))
    added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
    return "".join(lines), added, removed


def pick_tier(file_count: int, churn: int, limits: Limits) -> str:
    """Which disclosure level to open at.

    ``inline`` needs *both* few files and few lines: two files carrying five
    thousand lines between them is not something to unfold in someone's face.
    """
    if file_count <= limits.inline_max_files and churn <= limits.inline_max_lines:
        return "inline"
    if file_count <= limits.list_max_files:
        return "list"
    return "count"


def collect(
    since_seq: int,
    *,
    working_dir: "str | Path | None" = None,
    actor: "str | None" = None,
    limits: Limits | None = None,
    turn_id: int = 0,
    spill_dir: "str | Path | None" = None,
) -> Changeset:
    """Build the changeset for the edits after *since_seq*.

    *actor* restricts the changeset to one agent's edits (default: this
    process). Edits to the same files by *other* actors inside the window are
    not included, but are reported as ``foreign_edit`` on the affected files —
    concurrent agents in one tree already interleave into this journal, so
    "whose edit was this" cannot be assumed.
    """
    from agent.core import checkpoint

    limits = limits or Limits()
    root = Path(working_dir).resolve() if working_dir else Path.cwd()
    me = actor if actor is not None else checkpoint.actor()

    mine: dict[str, list[dict]] = {}
    others: dict[str, list[str]] = {}
    for entry in checkpoint.journal_entries(since_seq):
        who = entry.get("actor")
        path = entry["path"]
        # An entry with no actor predates attribution. Claim it only when we
        # have no identity of our own to compare against, so a genuinely
        # concurrent agent is never silently absorbed into our changeset.
        if who == me or (who is None and me is None):
            mine.setdefault(path, []).append(entry)
        elif who is not None and who not in others.setdefault(path, []):
            others[path].append(who)

    cs = Changeset(turn_id=turn_id)
    spill = Path(spill_dir) if spill_dir else None
    budget = limits.max_total_bytes

    for path, entries in mine.items():
        entries.sort(key=lambda e: e["seq"])
        before = entries[0].get("before")           # earliest pre-image in the window
        last_sha = entries[-1].get("after_sha")
        after, binary = _read_current(root, path)

        fc = FileChange(path=path, binary=binary)
        fc.foreign_actors = list(others.get(path, ()))

        # Did the file move after our last write? A known-good hash that no
        # longer matches means somebody else wrote it; an unknown hash means we
        # cannot tell, which must not be reported as "unchanged".
        if after is not None and last_sha:
            fc.foreign_edit = checkpoint.sha_text(after) != last_sha
        elif after is None and not binary:
            fc.foreign_edit = True                  # gone from under us
        if fc.foreign_actors:
            fc.foreign_edit = True

        if after is None:
            fc.status = "deleted" if not binary else "modified"
            fc.removed = len((before or "").splitlines())
            cs.files.append(fc)
            continue
        if before is None:
            fc.status = "added"
        if binary:
            cs.files.append(fc)
            continue

        diff, fc.added, fc.removed = _unified(path, before or "", after, limits.context_lines)
        size = len(diff.encode("utf-8", "surrogatepass"))
        if size > limits.max_diff_bytes or size > budget:
            fc.truncated = True
            cs.truncated = True
            if spill is not None:
                fc.diff_ref = _spill(spill, path, diff)
        else:
            fc.diff = diff
            budget -= size
        cs.files.append(fc)

    # Biggest change first: with many files, the top of the list is all anyone
    # reads, so it should carry the most information.
    cs.files.sort(key=lambda f: (-f.churn, f.path))
    cs.tier = pick_tier(cs.file_count, cs.total_added + cs.total_removed, limits)
    return cs


def _spill(spill_dir: Path, path: str, diff: str) -> str | None:
    """Write an oversized diff beside the session and return its relative name."""
    from agent.core.checkpoint import sha_text
    name = f"{sha_text(path)[:16]}.diff"
    try:
        spill_dir.mkdir(parents=True, exist_ok=True)
        (spill_dir / name).write_text(diff, encoding="utf-8", errors="surrogatepass")
        return name
    except OSError:
        logger.warning("changeset: could not spill diff for %s", path)
        return None


def load_diff(cs_file: FileChange, spill_dir: "str | Path | None") -> str | None:
    """The diff for one file, reading it back from disk when it was spilled."""
    if cs_file.diff is not None:
        return cs_file.diff
    if not (cs_file.diff_ref and spill_dir):
        return None
    try:
        return (Path(spill_dir) / cs_file.diff_ref).read_text(
            encoding="utf-8", errors="surrogatepass")
    except OSError:
        return None


def stored_diff(session_id: str, turn_id: int, path: str) -> dict:
    """The diff one round captured for one file, read back from the session.

    Answers a different question from ``git diff``: this is what turn 3 wrote,
    which must not change because turn 7 edited the same file again. Backs both
    the browser's ``/api/changeset`` and the relay's diff-on-demand reply, so
    the two cannot drift. Returns ``{"error": …}`` rather than raising.
    """
    path = (path or "").strip()
    if not path:
        return {"error": "invalid path"}
    if not session_id:
        return {"error": "no active session"}
    try:
        turn_id = int(turn_id)
    except (TypeError, ValueError):
        return {"error": "invalid turn"}
    try:
        from agent.memory.qa_log import read_history_sync
        a_data = next((a for tid, _q, a in read_history_sync(session_id) if tid == turn_id), None)
    except Exception:
        logger.debug("changeset: stored_diff could not read qa log", exc_info=True)
        return {"error": "session log unreadable"}
    if a_data is None:
        return {"error": f"turn {turn_id} not found"}
    fc = next((f for f in from_a_data(a_data).files if f.path == path), None)
    if fc is None:
        return {"error": f"'{path}' was not changed in turn {turn_id}"}
    try:
        from agent.memory.session import get_session_full_dir
        spill_dir = Path(get_session_full_dir(session_id)) / "changesets" / str(turn_id)
    except Exception:
        spill_dir = None
    return {"path": fc.path, "diff": load_diff(fc, spill_dir) or "",
            "status": fc.status, "binary": fc.binary, "truncated": fc.truncated}


def render_text(cs: Changeset, *, tier: str | None = None, indent: str = "  ") -> list[str]:
    """Plain lines for the terminal UIs. No markup — callers add their own.

    *tier* overrides the collected tier, which is how "unfold" is implemented:
    the same changeset re-rendered one level down.
    """
    if not cs:
        return []
    tier = tier or cs.tier
    lines = [cs.headline() + (" (diffs truncated)" if cs.truncated else "")]
    if tier == "count":
        return lines

    for f in cs.files:
        mark = {"added": "+", "deleted": "-"}.get(f.status, "~")
        stat = "binary" if f.binary else f"+{f.added:,} -{f.removed:,}"
        row = f"{indent}{mark} {f.path}  {stat}"
        if f.truncated:
            row += "  (diff too large)"
        lines.append(row)
        if f.note():
            lines.append(f"{indent}  ! {f.note()}")
        if tier == "inline" and f.diff:
            lines.extend(f"{indent}  {l}" for l in f.diff.rstrip("\n").splitlines())
    return lines


def to_json(cs: Changeset) -> dict:
    return {
        "turn_id": cs.turn_id, "tier": cs.tier, "truncated": cs.truncated,
        "prose": cs.prose,
        "files": [asdict(f) for f in cs.files],
    }


def from_json(data: dict) -> Changeset:
    """Rebuild a stored changeset. Unknown keys are dropped so a session written
    by a newer agent still loads."""
    known = set(FileChange.__dataclass_fields__)
    files = [
        FileChange(**{k: v for k, v in (f or {}).items() if k in known})
        for f in (data.get("files") or []) if isinstance(f, dict)
    ]
    return Changeset(
        turn_id=int(data.get("turn_id") or 0),
        files=files,
        tier=str(data.get("tier") or "count"),
        truncated=bool(data.get("truncated")),
        prose=str(data.get("prose") or ""),
    )


def merge_changesets(changesets: "list[Changeset]") -> Changeset:
    """Aggregate a session's worth of round changesets into one, by path.

    A path touched in more than one round is counted once, with its added/
    removed churn summed across rounds. Diff text is not merged (the rollup is
    a file list, not a stack of diffs — drill into a single round's changeset
    for that). Foreign-edit notes and actors accumulate across rounds too.

    Lives here rather than in a UI module because all three UIs render the
    rollup and must agree on what it says.
    """
    merged: dict[str, FileChange] = {}
    first_status: dict[str, str] = {}
    order: list[str] = []
    for cs in changesets:
        if not cs:
            continue
        for f in cs.files:
            if f.path not in merged:
                merged[f.path] = FileChange(path=f.path, status=f.status, binary=f.binary)
                first_status[f.path] = f.status
                order.append(f.path)
            m = merged[f.path]
            m.added += f.added
            m.removed += f.removed
            # Status is measured from the session start, not from the last
            # round: a file created in round 1 and edited in round 5 is still
            # "added" as far as the session is concerned.
            if f.status == "deleted":
                m.status = "deleted"
            elif first_status[f.path] == "added":
                m.status = "added"
            else:
                m.status = f.status
            m.binary = f.binary
            if f.foreign_edit:
                m.foreign_edit = True
                for actor in f.foreign_actors:
                    if actor not in m.foreign_actors:
                        m.foreign_actors.append(actor)

    files = [merged[p] for p in order]
    files.sort(key=lambda f: (-f.churn, f.path))
    tier = pick_tier(len(files), sum(f.churn for f in files), Limits())
    return Changeset(files=files, tier=tier)


def session_changesets(session_id: str) -> list[Changeset]:
    """Every round's changeset for *session_id*, rebuilt from the QA log.

    Reading the log rather than an in-memory list is what makes a *resumed*
    session roll up the whole session instead of only the rounds since the UI
    started. Returns [] (never raises) when the log is unreadable — a missing
    rollup is a cosmetic loss, not a reason to break a round.
    """
    try:
        from agent.memory.qa_log import read_history_sync
        records = list(read_history_sync(session_id))
    except Exception:
        logger.debug("changeset: could not read qa log for %r", session_id, exc_info=True)
        return []
    out: list[Changeset] = []
    for _tid, _q, a in records:
        cs = from_a_data(a if isinstance(a, dict) else {})
        if cs:
            out.append(cs)
    return out


def session_rollup(session_id: str, extra: "list[Changeset] | None" = None) -> Changeset:
    """The whole session's changeset: the QA log plus rounds not yet in it.

    *extra* carries live rounds the caller already holds (the round that just
    finished is written to the log at turn end, so it may or may not be there
    yet). A round present in both is taken once — matched on ``turn_id``, with
    an untagged (turn_id 0) extra always kept, since it cannot be matched.
    """
    rounds = session_changesets(session_id) if session_id else []
    seen = {cs.turn_id for cs in rounds if cs.turn_id}
    for cs in (extra or []):
        if not cs:
            continue
        if cs.turn_id and cs.turn_id in seen:
            continue
        if cs.turn_id:
            seen.add(cs.turn_id)
        rounds.append(cs)
    return merge_changesets(rounds)


class SessionRollup:
    """A session's rounds, accumulated once and merged on demand.

    Every UI needs the same two things: the rounds already on disk (so a
    resumed session rolls up the whole session) and the rounds it has watched
    since (which reach it before the log does). This holds both, reads the log
    exactly once — a round end must not re-read every A-record in a long
    session — and drops a round it has already seen, matched on ``turn_id``.
    """

    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id
        self._rounds: list[Changeset] = []
        self._seen: set[int] = set()
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True                     # set first: a failed read is final
        if self.session_id:
            for cs in session_changesets(self.session_id):
                self._take(cs)

    def _take(self, cs: "Changeset | None") -> None:
        if not cs:
            return
        # A UI may offer the same round twice (two views of one turn). turn_id
        # catches that for a tagged round; identity catches an untagged one.
        if any(r is cs for r in self._rounds):
            return
        if cs.turn_id:
            if cs.turn_id in self._seen:
                return
            self._seen.add(cs.turn_id)
        self._rounds.append(cs)

    def add(self, cs: "Changeset | None") -> None:
        """Record a round the UI just watched finish."""
        self._load()                            # log rounds first, so they sort earlier
        self._take(cs)

    def changeset(self) -> Changeset:
        self._load()
        return merge_changesets(self._rounds)

    def line(self) -> str:
        return rollup_line(self.changeset())

    def __bool__(self) -> bool:
        return bool(self.changeset())


def rollup_line(cs: "Changeset | None") -> str:
    """One-line session total, worded identically in every UI. "" when empty."""
    if not cs:
        return ""
    return "session: " + cs.headline()


def rollup_json(cs: "Changeset | None") -> dict | None:
    """Wire form of a rollup for the browser UI: the line plus its numbers."""
    if not cs:
        return None
    return {
        "line": rollup_line(cs),
        "files": cs.file_count,
        "added": cs.total_added,
        "removed": cs.total_removed,
    }


# ── config toggles ───────────────────────────────────────────────────────────
#
# Read defensively: a bare/stub config (tests, older config files) must produce
# the default rather than an AttributeError.

def _changeset_config(config):
    """The ``[ui.changeset]`` section, or None if config has no ``ui``."""
    return getattr(getattr(config, "ui", None), "changeset", None)


def changeset_enabled(config) -> bool:
    cfg = _changeset_config(config)
    return bool(getattr(cfg, "enabled", True)) if cfg is not None else True


def session_rollup_enabled(config) -> bool:
    cfg = _changeset_config(config)
    return bool(getattr(cfg, "session_rollup", True)) if cfg is not None else True


def from_a_data(a_data: dict) -> Changeset:
    """Rebuild a Changeset from a stored A-record, old or new.

    Sessions written before this feature have no ``changeset`` key, only
    ``modified_files`` — a list of either path strings or ``{path, added,
    removed}`` dicts (both shapes occur in the wild). Every UI that reads
    history needs the same fallback, so it lives here once instead of being
    reimplemented in each one.
    """
    try:
        turn_id = int((a_data.get("turn_id") if isinstance(a_data, dict) else 0) or 0)
    except (TypeError, ValueError):
        turn_id = 0
    try:
        stored = a_data.get("changeset") if isinstance(a_data, dict) else None
        if isinstance(stored, dict) and stored:
            cs = from_json(stored)
            # The A-record is authoritative for which turn this was; a stored
            # changeset predating turn tagging carries 0.
            return cs if cs.turn_id else Changeset(turn_id, cs.files, cs.tier, cs.truncated, cs.prose)

        modified = a_data.get("modified_files") if isinstance(a_data, dict) else None
        if not modified:
            return Changeset(turn_id=turn_id)

        seen: set[str] = set()
        files: list[FileChange] = []
        for entry in modified:
            if isinstance(entry, str):
                path = entry
                added = removed = 0
            elif isinstance(entry, dict):
                path = entry.get("path")
                added = entry.get("added") or 0
                removed = entry.get("removed") or 0
            else:
                continue
            if not isinstance(path, str) or not path or path in seen:
                continue
            seen.add(path)
            try:
                files.append(FileChange(path=path, added=int(added), removed=int(removed)))
            except (TypeError, ValueError):
                files.append(FileChange(path=path))

        if not files:
            return Changeset(turn_id=turn_id)

        limits = Limits()
        tier = pick_tier(len(files), sum(f.churn for f in files), limits)
        return Changeset(turn_id=turn_id, files=files, tier=tier)
    except Exception:
        logger.debug("changeset: from_a_data could not parse A-record", exc_info=True)
        return Changeset()

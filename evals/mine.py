"""Turn recorded failures into eval tasks.

`failure_report.py` has been writing every invalid tool call, tool exception and
runtime exception to `.agent/failures/` for a long time, and `agent diag`
summarises them per project. Nothing closed the loop: the failures that recur
most often were never the ones the eval suite covered, so harness and prompt
changes were gated on tasks chosen by hand rather than on what actually breaks.

This reads those journals, clusters them into recurring failure *modes*, ranks
the modes, and scaffolds an eval task from one so a human only has to write the
fixture and the check.

    python evals/mine.py                       # rank failure modes here
    python evals/mine.py --project ~/src/x     # …across other projects too
    python evals/mine.py --scaffold 1          # write a task from mode #1

Deliberately not automatic: a mined failure is evidence that *something* went
wrong, not a specification of correct behavior. The scaffold carries the
evidence and marks the parts a human must fill in.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
TASKS_DIR = EVALS_DIR / "tasks"
FIXTURES_DIR = EVALS_DIR / "fixtures"

# Volatile substrings that make otherwise-identical errors look distinct.
# Normalising them is what turns 40 one-off records into one ranked mode.
_NOISE = [
    (re.compile(r"0x[0-9a-fA-F]+"), "0xADDR"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\S*"), "TIMESTAMP"),
    (re.compile(r"(/[^\s'\"]+)+"), "PATH"),
    (re.compile(r"\bline \d+"), "line N"),
    (re.compile(r"\b\d+\b"), "N"),
]


def normalize(text: str, limit: int = 160) -> str:
    """Collapse a raw error string to its shape, so equal shapes group."""
    out = (text or "").strip()
    for pattern, replacement in _NOISE:
        out = pattern.sub(replacement, out)
    out = re.sub(r"\s+", " ", out)
    return out[:limit]


_TB_FILE = re.compile(r'^\s*File "([^"]+)", line \d+', re.MULTILINE)


def implicated_files(record: dict, project: Path) -> list[str]:
    """Project-relative source files named in *record*'s traceback, deepest last.

    Only files under *project* count: a traceback is mostly stdlib and
    site-packages frames, and "json/encoder.py changed" says nothing about
    whether this agent still has the bug.
    """
    out: list[str] = []
    for raw in _TB_FILE.findall(str(record.get("traceback") or "")):
        try:
            relative = Path(raw).resolve().relative_to(project.resolve())
        except (ValueError, OSError):
            continue
        name = relative.as_posix()
        if name not in out:
            out.append(name)
    return out


def tool_source_files(tool: str, project: Path) -> list[str]:
    """Files declaring `@register("<tool>")`, the module that implements it.

    Most failure records are invalid tool calls, which carry no traceback at
    all — so without this the majority of modes could never be judged. Found by
    grepping rather than importing: mine.py runs against other checkouts, and
    importing another project's tool registry to ask where a tool lives is both
    slow and a code-execution decision this has no business making.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", tool or ""):
        return []
    try:
        proc = subprocess.run(
            ["git", "-C", str(project), "grep", "-l", "-F", f'"{tool}"', "--", "*.py"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    # git grep is line-based and the decorator routinely wraps:
    #     @register(
    #         "read_file",
    # so candidates are narrowed by the literal name first, then confirmed
    # against the whole file text.
    pattern = re.compile(rf"""@?register\(\s*["']{re.escape(tool)}["']""")
    out = []
    for name in proc.stdout.splitlines():
        name = name.strip()
        if not name:
            continue
        try:
            text = (project / name).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if pattern.search(text):
            out.append(name)
    return out


def last_changed(path: str, project: Path) -> str:
    """ISO timestamp of the last commit touching *path*, or "" if unknown.

    git, not the filesystem mtime: a checkout or a `touch` would otherwise read
    as "someone fixed this", which is exactly the false negative that makes a
    staleness signal worse than none.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(project), "log", "-1", "--format=%cI", "--", path],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


@dataclass
class Mode:
    """One recurring failure shape."""
    kind: str
    tool: str
    reason: str
    shape: str
    count: int = 0
    first_seen: str = ""
    last_seen: str = ""
    sessions: set = field(default_factory=set)
    samples: list = field(default_factory=list)   # full records, newest first
    files: list = field(default_factory=list)     # implicated project files
    dated_files: list = field(default_factory=list)     # those with git history
    changed_since: list = field(default_factory=list)   # (file, commit ts) pairs

    @property
    def key(self) -> tuple:
        return (self.kind, self.tool, self.reason, self.shape)

    @property
    def stale(self) -> bool | None:
        """True if every implicated file changed after the last sighting.

        None means "no idea": no traceback, no project files in it, or no git
        history for them. Ranking must not treat that as fresh *or* stale — it
        is the common case for invalid-tool-call records, which carry no
        traceback at all.
        """
        if not self.files or not self.dated_files:
            return None
        if len(self.changed_since) < len(self.files):
            return False        # something implicated has not been touched since
        return True

    def as_dict(self) -> dict:
        return {
            "kind": self.kind, "tool": self.tool, "reason": self.reason,
            "shape": self.shape, "count": self.count, "sessions": len(self.sessions),
            "first_seen": self.first_seen, "last_seen": self.last_seen,
            "files": list(self.files), "stale": self.stale,
            "changed_since": [{"file": f, "committed": ts} for f, ts in self.changed_since],
        }


def failure_dirs(projects: list[Path]) -> list[Path]:
    out = []
    for project in projects:
        candidate = project / ".agent" / "failures"
        if candidate.is_dir():
            out.append(candidate)
    return out


def load_records(directory: Path) -> list[dict]:
    """Records from one failures/ dir: index.jsonl, enriched from detail files.

    The index carries a truncated error and the detail file name; the detail
    file has the arguments that actually reproduce the failure. Reading both
    means the scaffold can quote the real call.
    """
    index = directory / "index.jsonl"
    if not index.is_file():
        return []
    records: list[dict] = []
    for line in index.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        detail_name = record.get("file")
        if detail_name:
            detail_path = directory / str(detail_name)
            if detail_path.is_file():
                try:
                    detail = json.loads(detail_path.read_text(encoding="utf-8", errors="replace"))
                    if isinstance(detail, dict):
                        merged = dict(detail)
                        merged.update({k: v for k, v in record.items() if v})
                        record = merged
                except ValueError:
                    pass
        record["_dir"] = str(directory)
        records.append(record)
    return records


def cluster(records: list[dict]) -> list[Mode]:
    modes: dict[tuple, Mode] = {}
    for record in records:
        shape = normalize(str(record.get("error", "")) or str(record.get("reason", "")))
        mode = Mode(
            kind=str(record.get("kind", "?")),
            tool=str(record.get("tool", "") or ""),
            reason=str(record.get("reason", "") or ""),
            shape=shape,
        )
        mode = modes.setdefault(mode.key, mode)
        mode.count += 1
        session = record.get("session_id")
        if session:
            mode.sessions.add(session)
        timestamp = str(record.get("ts") or record.get("timestamp") or "")
        if timestamp:
            mode.first_seen = min(mode.first_seen or timestamp, timestamp)
            mode.last_seen = max(mode.last_seen, timestamp)
        mode.samples.append(record)
    for mode in modes.values():
        mode.samples.sort(key=lambda r: str(r.get("ts") or r.get("timestamp") or ""), reverse=True)
    return sorted(modes.values(), key=lambda m: (-m.count, -len(m.sessions), m.tool))


def _as_datetime(text: str):
    try:
        parsed = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def annotate_staleness(modes: list[Mode], last_changed=None) -> None:
    """Fill in `files` and `changed_since` on each mode, in place.

    Answers the one question the report could not: has the code this failure
    came from been touched since the failure last happened? A mode whose every
    implicated file has been rewritten since is probably already fixed, and
    ranking it first sends a human to write an eval for a bug that no longer
    exists — which is what happened the first two times this miner was used.
    """
    lookup = last_changed or globals()["last_changed"]
    sources = globals()["tool_source_files"]
    cache: dict[tuple[str, str], str] = {}
    for mode in modes:
        sample = next((s for s in mode.samples if s.get("traceback")), None) \
            or (mode.samples[0] if mode.samples else None)
        if sample is None:
            continue
        directory = sample.get("_dir")
        if not directory:
            continue
        project = Path(directory).parent.parent
        mode.files = implicated_files(sample, project)
        if not mode.files and mode.tool:
            # No traceback — the invalid-tool-call case, and the majority of
            # records. The tool's own module is a weaker attribution than a
            # traceback (the bug may be in the prompt or the schema, not the
            # implementation), but it is the only file the record points at.
            mode.files = sources(mode.tool, project)
        seen_at = _as_datetime(mode.last_seen)
        if not seen_at:
            continue
        for name in mode.files:
            key = (str(project), name)
            if key not in cache:
                cache[key] = lookup(name, project)
            committed = _as_datetime(cache[key])
            if not committed:
                continue
            mode.dated_files.append(name)
            if committed > seen_at:
                mode.changed_since.append((name, cache[key]))


def rank(modes: list[Mode]) -> list[Mode]:
    """Frequency order, but modes that look already-fixed sink to the bottom.

    Unknown staleness ranks with the live ones: an unjudgeable mode must not be
    demoted on no evidence.
    """
    return sorted(modes, key=lambda m: (bool(m.stale), -m.count, -len(m.sessions), m.tool))


def _stale_flag(mode: Mode) -> str:
    return {True: "stale", False: "live", None: "?"}[mode.stale]


def print_report(modes: list[Mode], limit: int) -> None:
    if not modes:
        print("No failure records found. Nothing to mine.")
        return
    total = sum(m.count for m in modes)
    stale = sum(1 for m in modes if m.stale)
    print(f"{len(modes)} failure mode(s) across {total} record(s)"
          + (f", {stale} likely already fixed" if stale else "") + "\n")
    header = (f"{'#':>3}  {'N':>5} {'SESS':>5} {'STATE':<6} {'KIND':<20} "
              f"{'TOOL':<18} SHAPE")
    print(header)
    print("-" * min(len(header) + 30, 110))
    for i, mode in enumerate(modes[:limit], 1):
        print(f"{i:>3}  {mode.count:>5} {len(mode.sessions):>5} {_stale_flag(mode):<6} "
              f"{mode.kind[:20]:<20} {mode.tool[:18]:<18} {mode.shape[:50]}")
    print("\nSTATE: live = implicated code unchanged since the failure last "
          "happened;\n       stale = every implicated file has been changed since, "
          "so it may\n       already be fixed; ? = no traceback to attribute it to a file.")
    print("\nScaffold an eval task from a mode with: "
          "python evals/mine.py --scaffold <#>")


def _slug(text: str, limit: int = 30) -> str:
    out = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (out or "failure")[:limit].strip("-")


def scaffold(mode: Mode, tasks_dir: Path = TASKS_DIR,
             fixtures_dir: Path = FIXTURES_DIR) -> tuple[Path, Path]:
    """Write a task YAML + fixture dir for *mode*. Returns (task, fixture)."""
    base = f"regress-{_slug(mode.tool or mode.kind)}"
    task_id = base
    n = 2
    while (tasks_dir / f"{task_id}.yaml").exists():
        task_id = f"{base}-{n}"
        n += 1

    fixture_dir = fixtures_dir / task_id
    fixture_dir.mkdir(parents=True, exist_ok=True)

    sample = mode.samples[0] if mode.samples else {}
    evidence = {
        "kind": mode.kind,
        "tool": mode.tool,
        "reason": mode.reason,
        "occurrences": mode.count,
        "sessions": len(mode.sessions),
        "first_seen": mode.first_seen,
        "last_seen": mode.last_seen,
        "example_arguments": sample.get("arguments") or sample.get("raw_arguments"),
        "example_error": str(sample.get("error", ""))[:500],
        "implicated_files": list(mode.files),
        "stale": mode.stale,
        "changed_since_last_seen": [{"file": f, "committed": ts}
                                    for f, ts in mode.changed_since],
    }
    (fixture_dir / "FAILURE.json").write_text(
        json.dumps(evidence, indent=2, default=str) + "\n", encoding="utf-8")

    task_path = tasks_dir / f"{task_id}.yaml"
    task_path.write_text(_task_yaml(task_id, mode, evidence), encoding="utf-8")
    return task_path, fixture_dir


def _yaml_quote(text: str) -> str:
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _stale_warning(mode: Mode) -> str:
    if not mode.stale:
        return ""
    files = ", ".join(f"{f} ({ts[:10]})" for f, ts in mode.changed_since)
    return (f"""#
# WARNING: this mode may already be fixed. Every file its traceback implicates
# has been committed to since the failure was last seen ({mode.last_seen or "?"}):
#   {files}
# Confirm the bug still reproduces before writing an eval for it.
""")


def _task_yaml(task_id: str, mode: Mode, evidence: dict) -> str:
    mined_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"""# Mined from recorded failures on {mined_at} by evals/mine.py.
{_stale_warning(mode)}\

#
# This is a SCAFFOLD, not a finished task. A failure record proves something
# went wrong; it does not say what right looks like. Before enabling it:
#   1. Put a minimal repro in fixtures/{task_id}/ (FAILURE.json holds the
#      evidence — the real arguments and error — and is not itself a fixture).
#   2. Write a prompt that provokes the same situation.
#   3. Replace the placeholder check with one that fails today and passes once
#      the underlying problem is fixed.
#   4. Delete the `draft: true` line.
#
# Evidence: {mode.count} occurrence(s) across {len(mode.sessions)} session(s),
# last seen {mode.last_seen or "unknown"}.
# Failure shape: {mode.shape}

id: {task_id}
draft: true
fixture: {task_id}
prompt: {_yaml_quote("TODO: prompt that reproduces: " + (mode.reason or mode.kind))}

checks:
  - type: file_contains
    path: TODO.txt
    text: {_yaml_quote("TODO: an assertion that fails before the fix")}
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster recorded agent failures and scaffold eval tasks from them.")
    parser.add_argument("--project", action="append", default=None,
                        help="project root to mine (repeatable; default: cwd)")
    parser.add_argument("--limit", type=int, default=20,
                        help="how many modes to show (default 20)")
    parser.add_argument("--min-count", type=int, default=1,
                        help="ignore modes seen fewer than this many times")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="write the ranked modes to a JSON file")
    parser.add_argument("--scaffold", type=int, default=None, metavar="N",
                        help="write an eval task scaffold from mode #N")
    parser.add_argument("--live-only", action="store_true",
                        help="drop modes whose implicated files all changed since "
                             "the failure last happened (probably already fixed)")
    parser.add_argument("--no-staleness", action="store_true",
                        help="skip the git lookups that judge staleness")
    parser.add_argument("--tasks-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fixtures-dir", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    projects = [Path(p).expanduser() for p in (args.project or ["."])]

    records: list[dict] = []
    for directory in failure_dirs(projects):
        records.extend(load_records(directory))

    modes = [m for m in cluster(records) if m.count >= args.min_count]

    if not args.no_staleness:
        annotate_staleness(modes)
        if args.live_only:
            modes = [m for m in modes if not m.stale]
        modes = rank(modes)

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps([m.as_dict() for m in modes], indent=2) + "\n", encoding="utf-8")

    if args.scaffold is not None:
        if not 1 <= args.scaffold <= len(modes):
            print(f"--scaffold: no mode #{args.scaffold} (have {len(modes)})", file=sys.stderr)
            return 1
        tasks_dir = Path(args.tasks_dir) if args.tasks_dir else TASKS_DIR
        fixtures_dir = Path(args.fixtures_dir) if args.fixtures_dir else FIXTURES_DIR
        task_path, fixture_dir = scaffold(modes[args.scaffold - 1], tasks_dir, fixtures_dir)
        print(f"wrote {task_path}")
        print(f"wrote {fixture_dir}/FAILURE.json")
        print("\nThe task is marked `draft: true` and will not run until you "
              "finish the fixture, prompt and check, then remove that line.")
        return 0

    print_report(modes, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
import sys
from collections import Counter
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

    @property
    def key(self) -> tuple:
        return (self.kind, self.tool, self.reason, self.shape)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind, "tool": self.tool, "reason": self.reason,
            "shape": self.shape, "count": self.count, "sessions": len(self.sessions),
            "first_seen": self.first_seen, "last_seen": self.last_seen,
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


def print_report(modes: list[Mode], limit: int) -> None:
    if not modes:
        print("No failure records found. Nothing to mine.")
        return
    total = sum(m.count for m in modes)
    print(f"{len(modes)} failure mode(s) across {total} record(s)\n")
    header = f"{'#':>3}  {'N':>5} {'SESS':>5}  {'KIND':<20} {'TOOL':<18} SHAPE"
    print(header)
    print("-" * min(len(header) + 30, 110))
    for i, mode in enumerate(modes[:limit], 1):
        print(f"{i:>3}  {mode.count:>5} {len(mode.sessions):>5}  "
              f"{mode.kind[:20]:<20} {mode.tool[:18]:<18} {mode.shape[:50]}")
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
    }
    (fixture_dir / "FAILURE.json").write_text(
        json.dumps(evidence, indent=2, default=str) + "\n", encoding="utf-8")

    task_path = tasks_dir / f"{task_id}.yaml"
    task_path.write_text(_task_yaml(task_id, mode, evidence), encoding="utf-8")
    return task_path, fixture_dir


def _yaml_quote(text: str) -> str:
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _task_yaml(task_id: str, mode: Mode, evidence: dict) -> str:
    mined_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"""# Mined from recorded failures on {mined_at} by evals/mine.py.
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

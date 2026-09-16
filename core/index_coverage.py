"""Index coverage block — which parts of the tree semantic search can answer for.

`base_rules.txt` tells the model its coverage is stated under "# Index coverage".
That block used to be injected by `cli/chat.py` alone, so every other entrypoint
(`cli/run.py`, the scheduler, sub-agents) carried a rule pointing at text that
was never there. It lives here now and is injected once, in `Agent.__init__`.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Directories listed individually before the tail is summarised as a count.
COVERAGE_MAX_DIRS = 25


def coverage_map(store, working_dir: str) -> dict[str, int]:
    """Return {top-level-dir: indexed_file_count}; root-level files count under "."."""
    if store is None:
        return {}
    try:
        paths = store.list_paths()
        coverage: dict[str, int] = {}
        wd = Path(working_dir)
        for p in paths:
            try:
                rel = Path(p).relative_to(wd)
                # A file at the root has one part and is NOT a directory —
                # counting it as its own "directory" turned the coverage list
                # into one bogus "collect.py/: 1 file(s)" line per file.
                top = rel.parts[0] if len(rel.parts) > 1 else "."
            except ValueError:
                top = "."
            coverage[top] = coverage.get(top, 0) + 1
        return coverage
    except Exception:
        return {}


def format_coverage(coverage: dict[str, int]) -> str:
    """Render coverage as a map the model can route on: biggest areas first."""
    total = sum(coverage.values())
    root_n = coverage.get(".", 0)
    dirs = sorted(((d, n) for d, n in coverage.items() if d != "."),
                  key=lambda t: (-t[1], t[0]))
    lines = [f"# Index coverage\nSemantic search available over {total} indexed file(s)."]
    shown = dirs[:COVERAGE_MAX_DIRS]
    if shown:
        lines.append("Top-level directories:")
        lines += [f"  {d}/: {n} file(s)" for d, n in shown]
    if len(dirs) > len(shown):
        rest = sum(n for _, n in dirs[len(shown):])
        lines.append(f"  ... {len(dirs) - len(shown)} more directories ({rest} file(s))")
    if root_n:
        lines.append(f"  (repository root): {root_n} file(s)")
    return "\n".join(lines)


#: Shown when nothing is indexed. Names real tools only: it used to recommend
#: `shell_exec`, which does not exist — the shell tool is `run_argv`.
NO_INDEX_MESSAGE = (
    "# Index coverage\n"
    "Not indexed. Semantic search unavailable.\n"
    "Use grep_code for exact text, find_symbol for a known name, read_file and "
    "list_files to navigate; run_argv for anything those do not cover.\n"
    "Recommend indexing to user when semantic search would materially help."
)


def coverage_message(store, working_dir: str) -> str:
    """The "# Index coverage" system block for this session."""
    coverage = coverage_map(store, working_dir)
    return format_coverage(coverage) if coverage else NO_INDEX_MESSAGE

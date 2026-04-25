from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


def _find_project_root(start_dir: Path, search_parents: bool) -> Path | None:
    curr = start_dir.resolve()
    while True:
        if (curr / ".agent").is_dir():
            return curr
        if not search_parents or curr == curr.parent:
            break
        curr = curr.parent
    return None


def main() -> None:
    sys.setrecursionlimit(5000)
    parser = argparse.ArgumentParser(prog="agent", description="Local code agent")
    parser.add_argument("--config", type=str, help="Path to agent.toml")

    sub = parser.add_subparsers(dest="command")

    # init
    init_p = sub.add_parser("init", help="Initialize index for current directory")
    init_p.add_argument("--languages", type=str, help="Comma-separated languages: py,js,kt,cpp")
    init_p.add_argument("--exclude", type=str, help="Comma-separated paths to exclude")
    init_p.add_argument("--force", action="store_true", help="Force re-index all files")
    init_p.add_argument("--watch", action="store_true", help="Watch for file changes and re-index automatically")

    # index
    idx_p = sub.add_parser("index", help="Manage index")
    idx_p.add_argument("--update", action="store_true", help="Re-index changed files (also prunes stale & purges expired archive)")
    idx_p.add_argument("--stats", action="store_true", help="Show index statistics")
    idx_p.add_argument("--prune", action="store_true", help="Archive chunks for files that are missing or now match .agent.ignore")
    idx_p.add_argument("--restore", type=str, metavar="PATH", help="Restore a previously archived path back into the live index")
    idx_p.add_argument("--purge-archive", action="store_true", help="Permanently delete archive rows older than archive_ttl_days")
    idx_p.add_argument("--archive-ttl", type=int, metavar="DAYS", help="Override archive_ttl_days for this run (0 = disable expiration)")

    # chat
    chat_p = sub.add_parser("chat", help="Start interactive session")
    chat_p.add_argument("--model", type=str, help="Override model name")
    chat_p.add_argument("--ctx", type=int, help="Override context window size")
    chat_p.add_argument("--session", type=str, help="Session name to load/save")
    chat_p.add_argument("--ui", type=str, choices=["textual", "simple"],
                        help="UI mode (skips the prompt)")

    # run
    run_p = sub.add_parser("run", help="Run a single prompt non-interactively")
    run_p.add_argument("prompt", type=str, nargs="?", default=None,
                       help="Prompt to run (reads stdin if omitted)")

    # sessions
    sess_p = sub.add_parser("sessions", help="Manage sessions")
    sess_p.add_argument("--list", action="store_true", help="List sessions")
    sess_p.add_argument("--load", type=str, help="Show session details")
    sess_p.add_argument(
        "--split", type=str, metavar="ID",
        help="Retro-extract verbose tool-call/reasoning blobs into sibling "
             "tool_calls.jsonl / reasoning.jsonl. Leaves a backup as "
             "session.json.bak. Pass session id (or 'all').",
    )
    sess_p.add_argument(
        "--dry-run", action="store_true",
        help="Show what --split would do without modifying files.",
    )

    # commit
    commit_p = sub.add_parser("commit", help="Generate and apply a commit message for a subrepo")
    commit_p.add_argument("path", type=str, help="Path to subrepo (absolute or relative to working dir)")
    commit_p.add_argument("--model", type=str, help="Override model name (primary + summarization)")
    commit_p.add_argument("--summarizer-model", type=str, dest="summarizer_model",
                          help="Named model entry to use for chunked diff summarization"
                               " (overrides [models] summarizer role)")

    # exec
    exec_p = sub.add_parser("exec", help="Execute a system command in the project directory")
    exec_p.add_argument("prompt", type=str, help="Command to execute")

    # prompts (compiled-prompt cache)
    pr_p = sub.add_parser("prompts", help="Manage compiled-prompt cache")
    pr_sub = pr_p.add_subparsers(dest="prompts_action")
    pr_sub.add_parser("status", help="Show cache entries with stats")
    pr_rec = pr_sub.add_parser("recompile", help="Mark entries pending so the next run recompiles them")
    pr_rec.add_argument("name", nargs="?", help="Prompt name (e.g. system.txt). Omit for all.")
    pr_clr = pr_sub.add_parser("clear", help="Delete cached compiled variants")
    pr_clr.add_argument("name", nargs="?", help="Prompt name. Omit for all.")

    # debug
    dbg_p = sub.add_parser("debug", help="Debug utilities")
    dbg_p.add_argument("--context", action="store_true", help="Show full context of current session")
    dbg_p.add_argument("--session", type=str, help="Session name")

    args = parser.parse_args()

    from agent.config import load_config, check_reachability, Config, ToolsConfig
    from agent.memory.session import configure as configure_sessions
    from agent.cli.logging_setup import _write_exception_dump, _setup_logging

    project_root = None
    if args.command != "init":
        temp_tools = ToolsConfig()
        project_root = _find_project_root(Path.cwd(), temp_tools.search_parents)
        if project_root is None:
            print("Error: Current directory (and parents) is not a valid agent project.")
            print("Please run 'agent init' in the desired project directory.")
            sys.exit(1)

    if args.config:
        config = load_config(Path(args.config))
    elif project_root and (project_root / "agent.toml").exists():
        config = load_config(project_root / "agent.toml")
    else:
        config = load_config(None)

    if project_root:
        config.tools.working_dir = str(project_root)

    configure_sessions(config.tools.working_dir, config.tools.agent_dir)
    from agent.planning import configure_plans
    from agent.planning.recovery import configure as configure_recovery
    configure_plans(config.tools.working_dir, config.tools.agent_dir)
    configure_recovery(config.tools.working_dir, config.tools.agent_dir)
    log_dir = Path(config.tools.working_dir) / config.tools.agent_dir
    _setup_logging(str(log_dir), config.logs)
    log_path = log_dir / "agent.log"

    try:
        if args.command == "init":
            from agent.cli.index import cmd_init
            cmd_init(args, config)
        elif args.command == "index":
            from agent.cli.index import (
                cmd_index_update, cmd_index_stats, cmd_index_prune,
                cmd_index_restore, cmd_index_purge_archive,
            )
            if args.update:
                cmd_index_update(args, config)
            elif args.stats:
                cmd_index_stats(args, config)
            elif args.prune:
                cmd_index_prune(args, config)
            elif args.restore:
                cmd_index_restore(args, config)
            elif getattr(args, "purge_archive", False):
                cmd_index_purge_archive(args, config)
            else:
                parser.parse_args(["index", "--help"])
        elif args.command == "chat":
            from agent.cli.chat import cmd_chat
            check_reachability(config)
            if config.recovery.enabled:
                from agent.planning import recovery as _rec
                try:
                    _rec.handle_pending_at_startup(config.recovery.prompt_mode)
                except Exception:
                    pass
            cmd_chat(args, config)
        elif args.command == "run":
            from agent.cli.run import cmd_run
            check_reachability(config)
            cmd_run(args, config)
        elif args.command == "sessions":
            from agent.cli.sessions import cmd_sessions
            cmd_sessions(args, config)
        elif args.command == "commit":
            from agent.cli.commit import cmd_commit
            if getattr(args, "model", None):
                config.llm.model = args.model
            check_reachability(config)
            cmd_commit(args, config)
        elif args.command == "prompts":
            from agent.cli.debug import cmd_prompts
            cmd_prompts(args, config)
        elif args.command == "debug":
            from agent.cli.debug import cmd_debug_context
            cmd_debug_context(args, config)
        elif args.command == "exec":
            from agent.tools.exec_command import handle_exec_command
            handle_exec_command(args, config)
        else:
            parser.print_help()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        dump_path = _write_exception_dump(exc, argv=sys.argv, config=config, log_path=log_path)
        try:
            from agent import failure_report as _fr
            _fr.report_exception(
                exc,
                kind="unhandled_exception",
                context={"argv": sys.argv, "command": getattr(args, "command", None)},
                config=config,
            )
        except Exception:
            pass
        msg = f"\nUnhandled exception: {type(exc).__name__}: {exc}"
        if dump_path:
            msg += f"\nDump written to: {dump_path}"
        print(msg, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

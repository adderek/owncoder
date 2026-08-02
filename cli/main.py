from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config


def _find_project_root(start_dir: Path, search_parents: bool) -> Path | None:
    curr = start_dir.resolve()
    while True:
        if (curr / ".agent").is_dir():
            return curr
        if not search_parents or curr == curr.parent:
            break
        curr = curr.parent
    return None


def _resolve_project(args) -> tuple[Path | None, "Config"]:
    """Find the project root and load that project's config.

    Without ``--working-dir`` the search starts at the cwd (unchanged
    behaviour). With it, the search starts at the canonicalized flag path,
    so the flag also decides which ``agent.toml`` is loaded — different
    model, different grants, different ``.agent/``.
    """
    from agent.config import load_config, ToolsConfig
    from agent.config.loader import CONFIG_FILENAMES

    working_dir = getattr(args, "working_dir", None)
    start = Path.cwd()
    if working_dir:
        start = Path(os.path.realpath(os.path.expanduser(working_dir)))
        if not start.is_dir():
            print(f"Error: --working-dir is not a directory: {working_dir}")
            sys.exit(1)

    project_root = None
    if args.command != "init":
        temp_tools = ToolsConfig()
        project_root = _find_project_root(start, temp_tools.search_parents)
        if project_root is None:
            where = str(start) if working_dir else "Current directory"
            print(f"Error: {where} (and parents) is not a valid agent project.")
            print("Please run 'agent init' in the desired project directory.")
            sys.exit(1)

    if args.config:
        config = load_config(Path(args.config))
    else:
        project_cfgs = [
            p for name in CONFIG_FILENAMES
            if project_root and (p := project_root / name).exists()
        ]
        config = load_config(project_cfgs or None)

    if project_root:
        config.tools.working_dir = str(project_root)
    elif working_dir:
        # `init`: no root to find yet, but the flag still says where to work.
        config.tools.working_dir = str(start)

    return project_root, config


def _friendly_error(exc: Exception) -> str:
    """Return a human-readable error message for known exception types."""
    name = type(exc).__name__
    msg = str(exc)

    # OpenAI / httpx connection errors
    try:
        from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError
        if isinstance(exc, APIConnectionError):
            return (
                f"\nError: cannot reach LLM endpoint.\n"
                f"  Check that your model server is running and the endpoint URL is correct.\n"
                f"  (Configure via agent.toml or AGENT_LLM_BASE_URL)"
            )
        if isinstance(exc, APITimeoutError):
            return "\nError: LLM request timed out. Server may be overloaded."
        if isinstance(exc, AuthenticationError):
            return "\nError: LLM authentication failed. Check your API key."
        if isinstance(exc, RateLimitError):
            return "\nError: LLM rate limit hit. Try again later."
    except ImportError:
        pass

    # Generic fallback — show type + message but no traceback
    return f"\nError ({name}): {msg}"


def build_parser() -> argparse.ArgumentParser:
    """The full `agent` CLI surface.

    Split out of main() so the flags can be tested without running a
    command — previously nothing could assert that an option existed,
    parsed to the name the handler reads, or kept its default.
    """
    parser = argparse.ArgumentParser(prog="agent", description="Local code agent")
    parser.add_argument("--config", type=str, help="Path to agent.toml")
    parser.add_argument("--working-dir", type=str, metavar="PATH",
                        help="Work in this project directory instead of the "
                             "current one. Also decides which agent.toml is "
                             "loaded (model, grants, .agent/). No runtime "
                             "switching: for several projects, run one UI per "
                             "directory.")
    parser.add_argument("--ultrasecure", action="store_true",
                        help="Run in ultrasecure mode: the agent reaches the "
                             "internet only via a quarantined subagent broker "
                             "(ask_internet); web_search/web_fetch are stripped "
                             "from the main agent. Overrides agent.mode.")

    sub = parser.add_subparsers(dest="command")

    # init
    init_p = sub.add_parser("init", help="Initialize project config; optionally index")
    init_p.add_argument("--languages", type=str, help="Comma-separated languages: py,js,kt,cpp")
    init_p.add_argument("--exclude", type=str, help="Comma-separated paths to exclude")
    init_p.add_argument("--force", action="store_true", help="Force re-index all files")
    init_p.add_argument("--watch", action="store_true", help="Watch for file changes and re-index automatically")
    init_p.add_argument("--path", type=str, metavar="PATH", help="Index only this subtree (relative to project root)")
    init_p.add_argument("--skip-index", action="store_true", help="Configure only; skip indexing prompt")

    # index
    idx_p = sub.add_parser("index", help="Manage index")
    idx_p.add_argument("--update", action="store_true", help="Re-index changed files (also prunes stale & purges expired archive)")
    idx_p.add_argument("--stats", action="store_true", help="Show index statistics")
    idx_p.add_argument("--list-pending", action="store_true", help="With --stats, list files not yet indexed")
    idx_p.add_argument("--prune", action="store_true", help="Archive chunks for files that are missing or now match .agent.ignore")
    idx_p.add_argument("--restore", type=str, metavar="PATH", help="Restore a previously archived path back into the live index")
    idx_p.add_argument("--purge-archive", action="store_true", help="Permanently delete archive rows older than archive_ttl_days")
    idx_p.add_argument("--archive-ttl", type=int, metavar="DAYS", help="Override archive_ttl_days for this run (0 = disable expiration)")
    idx_p.add_argument("--daemon", action="store_true", help="Start background index watcher daemon")
    idx_p.add_argument("--stop", action="store_true", help="Stop background index watcher daemon")
    idx_p.add_argument("--watch", action="store_true", help="Watch for file changes and re-index (foreground; used internally by daemon)")

    # chat
    emb_p = sub.add_parser("embed", help="Manage the local embeddings server (external launcher script)")
    emb_p.add_argument("--start", action="store_true", help="Start the server (rag.embed_server_command)")
    emb_p.add_argument("--stop", action="store_true", help="Stop the server")
    emb_p.add_argument("--status", action="store_true", help="Show server status (default action)")
    emb_dev = emb_p.add_mutually_exclusive_group()
    emb_dev.add_argument("--gpu", action="store_true", help="Run the model on GPU")
    emb_dev.add_argument("--cpu", action="store_true", help="Run the model on CPU")

    chat_p = sub.add_parser("chat", help="Start interactive session")
    chat_p.add_argument("--model", type=str, help="Override model name")
    chat_p.add_argument("--ctx", type=int, help="Override context window size")
    chat_p.add_argument("--session", type=str, help="Session name to load/save")
    chat_p.add_argument("--ui", type=str, choices=["textual", "simple", "http"],
                        help="UI mode (skips the prompt)")
    chat_p.add_argument("--http-sidecar", action="store_true",
                        help="With --ui textual/simple: also run a companion "
                             "browser view (read-mostly mirror + chat input) "
                             "alongside it — see agent/ui/http_sidecar.py")
    chat_p.add_argument("--incognito", action="store_true",
                        help="Don't persist this session or any notes it produces")
    chat_p.add_argument("--private", action="store_true",
                        help="Incognito + refuse to run against non-local LLM endpoints")
    chat_p.add_argument("--vault", action="store_true",
                        help="Persist everything encrypted at rest (prompts for a "
                             "passphrase; lose it and the session is unrecoverable)")

    # vault
    vault_p = sub.add_parser("vault", help="Read back files sealed by vault mode")
    vault_sub = vault_p.add_subparsers(dest="vault_action")
    vault_sub.add_parser("status", help="Is there a vault here, and how much is sealed")
    vault_show = vault_sub.add_parser("show", help="Decrypt one sealed file to stdout")
    vault_show.add_argument("path", type=str, help="Logical path (with or without .enc)")
    vault_log = vault_sub.add_parser("log", help="Decrypt the sealed agent log")
    vault_log.add_argument("--tail", type=int, default=0, help="Last N lines only")

    # run
    run_p = sub.add_parser("run", help="Run a single prompt non-interactively")
    run_p.add_argument("prompt", type=str, nargs="?", default=None,
                       help="Prompt to run (reads stdin if omitted)")
    run_p.add_argument("--json", action="store_true",
                       help="Print a JSON envelope instead of plain text; "
                            "sets exit code 0=done, 1=error, 2=iteration/goal cap")

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
    sess_p.add_argument(
        "--prune-empty", action="store_true",
        help="Delete saved sessions that hold no conversation (left behind by "
             "older versions). Lists them; add -y to actually delete.",
    )
    sess_p.add_argument(
        "--include-empty", action="store_true",
        help="Include content-free sessions in the listing.",
    )
    sess_p.add_argument(
        "-y", "--yes", action="store_true",
        help="Answer yes to the confirmation (for scripts).",
    )

    # commit
    commit_p = sub.add_parser("commit", help="Generate and apply a commit message for a subrepo")
    commit_p.add_argument("path", nargs="?", default=".",
                          help="Path to git repo (default: current directory)")
    commit_p.add_argument("-m", "--model", type=str, nargs="?", const="__list__", default=None,
                          help="-m alone: list available models (with live availability); "
                               "-m NAME: override model name (primary + summarization)")
    commit_p.add_argument("-s", "-ms", "--summarizer-model", dest="summarizer_model",
                          nargs="?", const="__list__", default=None,
                          help="-s alone: list available models; -s NAME: use that model for summarization")
    commit_p.add_argument("--no-probe", dest="probe", action="store_false", default=True,
                          help="When listing models, skip the /models availability probe")
    commit_p.add_argument("-c", "--chunk-size", type=str, default="50%",
                          help="Chunk size (integer chars or percentage, e.g. '12000' or '50%%'); default: 50%% of context window")
    commit_p.add_argument("-y", "--yes", action="store_true",
                          help="Commit without the confirmation prompt (for scripts and CI)")
    commit_p.add_argument("--print", dest="print_only", action="store_true",
                          help="Print the generated message and exit without committing")

    # exec
    exec_p = sub.add_parser("exec", help="Execute a system command in the project directory")
    exec_p.add_argument("prompt", type=str, help="Command to execute")

    # prompts (compiled-prompt cache)
    pr_p = sub.add_parser("prompts", help="Manage compiled-prompt cache")
    pr_sub = pr_p.add_subparsers(dest="prompts_action")
    pr_sub.add_parser("status", help="Show cache entries with stats")
    pr_sub.add_parser("evaluate", help="Run the A/B verdict pass (recompile/pin regressed variants)")
    pr_rec = pr_sub.add_parser("recompile", help="Mark entries pending so the next run recompiles them")
    pr_rec.add_argument("name", nargs="?", help="Prompt name (e.g. system.txt). Omit for all.")
    pr_clr = pr_sub.add_parser("clear", help="Delete cached compiled variants")
    pr_clr.add_argument("name", nargs="?", help="Prompt name. Omit for all.")

    # cron (scheduled jobs)
    cron_p = sub.add_parser("cron", help="Manage/run scheduled jobs (see /schedule in chat)")
    cron_sub = cron_p.add_subparsers(dest="cron_action")
    cron_run = cron_sub.add_parser("run", help="Run all due jobs and exit (crontab/systemd-timer entry point)")
    cron_run.add_argument("--job", type=str, help="Force this job (id or name) to run now")
    cron_add = cron_sub.add_parser("add", help="Add a job")
    cron_add.add_argument("spec", type=str, help="Schedule: 'in 20m' | 'at 07:00' | 'every 6h' | '0 7 * * *' | '@daily' | 'idle'")
    cron_add.add_argument("prompt", type=str, help="Prompt the agent runs")
    cron_add.add_argument("--name", type=str, help="Job name (addressable, unique)")
    cron_rm = cron_sub.add_parser("rm", help="Remove a job")
    cron_rm.add_argument("job", type=str, help="Job id or name")
    cron_en = cron_sub.add_parser("enable", help="Enable a job")
    cron_en.add_argument("job", type=str, help="Job id or name")
    cron_dis = cron_sub.add_parser("disable", help="Disable a job")
    cron_dis.add_argument("job", type=str, help="Job id or name")
    cron_sub.add_parser("runs", help="Show recent scheduled runs")
    cron_sub.add_parser("list", help="List jobs (default)")

    # diag
    # todo — the backlog (.agent/ideas.db), also reachable as /idea in chat
    todo_p = sub.add_parser("todo", help="Backlog: list, add, update, export")
    todo_sub = todo_p.add_subparsers(dest="todo_action")
    todo_list = todo_sub.add_parser("list", help="List backlog items (default)")
    todo_list.add_argument("--status", type=str, help="raw|evaluated|planned|implementing|verifying|done|rejected")
    todo_list.add_argument("--type", type=str, help="feature|bug|optimization|integration|module|idea|core_change")
    todo_list.add_argument("--limit", type=int, default=50)
    todo_list.add_argument("--json", action="store_true", help="Raw JSON for scripting")
    todo_add = todo_sub.add_parser("add", help="Add an item")
    todo_add.add_argument("title", nargs="+")
    todo_add.add_argument("--body", type=str, default="")
    todo_add.add_argument("--type", type=str, default="idea")
    todo_add.add_argument("--tags", type=str, default="", help="comma-separated")
    todo_add.add_argument("--priority", type=int, default=3, help="1 (low) – 5 (critical)")
    todo_show = todo_sub.add_parser("show", help="Show one item in full")
    todo_show.add_argument("id")
    todo_set = todo_sub.add_parser("set", help="Update fields: status=… priority=… tags=a,b")
    todo_set.add_argument("id")
    todo_set.add_argument("fields", nargs="+")
    todo_done = todo_sub.add_parser("done", help="Mark done")
    todo_done.add_argument("id")
    todo_reject = todo_sub.add_parser("reject", help="Mark rejected")
    todo_reject.add_argument("id")
    todo_export = todo_sub.add_parser(
        "export", help="Export the whole backlog as JSON (migration to a real tracker)")
    todo_export.add_argument("--out", type=str, help="File to write (default: stdout)")
    todo_import = todo_sub.add_parser("import", help="Import a previously exported backlog")
    todo_import.add_argument("file")

    diag_p = sub.add_parser("diag", help="Tool health report from audit.jsonl")
    diag_p.add_argument("--json", action="store_true", help="Output raw JSON (for scripting)")

    # debug
    dbg_p = sub.add_parser("debug", help="Debug utilities")
    dbg_p.add_argument("--context", action="store_true", help="Show full context of current session")
    dbg_p.add_argument("--session", type=str, help="Session name")

    # serve
    serve_p = sub.add_parser("serve", help="Start chunk browser web UI")
    serve_p.add_argument("--port", type=int, default=8765, help="Port (default 8765; auto-increments if busy)")

    return parser


def main() -> None:
    sys.setrecursionlimit(5000)
    parser = build_parser()
    args = parser.parse_args()

    from agent.config import check_reachability
    from agent.memory.session import configure as configure_sessions
    from agent.cli.logging_setup import _write_exception_dump, _setup_logging

    project_root, config = _resolve_project(args)

    if getattr(args, "ultrasecure", False):
        config.agent.mode = "ultrasecure"

    configure_sessions(config.tools.working_dir, config.tools.agent_dir)
    from agent.planning import configure_plans
    from agent.planning.recovery import configure as configure_recovery
    from agent.ideas import configure as configure_ideas
    configure_plans(config.tools.working_dir, config.tools.agent_dir)
    configure_recovery(config.tools.working_dir, config.tools.agent_dir)
    configure_ideas(config.tools.working_dir, config.tools.agent_dir)
    log_dir = Path(config.tools.working_dir) / config.tools.agent_dir
    _setup_logging(str(log_dir), config.logs)
    log_path = log_dir / "agent.log"

    from agent.core.model_state_store import load_disabled
    from agent.core.model_control import disabled_set
    disabled_set(config).update(load_disabled(str(log_dir)))

    try:
        if args.command == "init":
            from agent.cli.index import cmd_init
            cmd_init(args, config)
        elif args.command == "index":
            from agent.cli.index import (
                cmd_index_update, cmd_index_stats, cmd_index_prune,
                cmd_index_restore, cmd_index_purge_archive,
                cmd_index_daemon_start, cmd_index_daemon_stop,
                _daemon_watch_entry,
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
            elif getattr(args, "daemon", False):
                cmd_index_daemon_start(args, config)
            elif getattr(args, "stop", False):
                cmd_index_daemon_stop(args, config)
            elif getattr(args, "watch", False):
                # Foreground watcher — also the path taken by the daemon subprocess.
                _daemon_watch_entry(
                    config,
                    languages=getattr(args, "languages", None),
                    exclude=getattr(args, "exclude", None),
                )
            else:
                parser.parse_args(["index", "--help"])
        elif args.command == "embed":
            from agent.rag import embed_server
            if getattr(args, "stop", False):
                print(embed_server.stop(config))
            elif getattr(args, "start", False):
                device = "gpu" if args.gpu else ("cpu" if args.cpu else None)
                print(embed_server.start(config, device))
            else:
                print(embed_server.status(config))
        elif args.command == "chat":
            from agent.cli.chat import cmd_chat
            from agent.cli import warmup
            from agent.config.profile_detect import run_startup_profile_check
            # Probe every endpoint and import the agent while the profile and
            # relay questions are on screen: none of that depends on the
            # answers, and it used to run afterwards, one endpoint at a time.
            warmup.start(config)
            run_startup_profile_check(config, interactive=sys.stdin.isatty())
            check_reachability(config)
            if config.recovery.enabled:
                from agent.planning import recovery as _rec
                try:
                    _rec.handle_pending_at_startup(config.recovery.prompt_mode)
                except Exception:
                    pass
            cmd_chat(args, config)
        elif args.command == "vault":
            from agent.cli.vault_cli import cmd_vault
            sys.exit(cmd_vault(args, config))
        elif args.command == "run":
            from agent.cli.run import cmd_run
            from agent.cli import warmup
            from agent.config.profile_detect import run_startup_profile_check
            warmup.start(config)
            run_startup_profile_check(config, interactive=False)
            check_reachability(config)
            cmd_run(args, config)
        elif args.command == "sessions":
            from agent.cli.sessions import cmd_sessions
            cmd_sessions(args, config)
        elif args.command == "commit":
            from agent.cli.commit import cmd_commit
            _model = getattr(args, "model", None)
            if _model == "__list__":
                # Listing only — no LLM call, so skip the reachability probe.
                cmd_commit(args, config)
                return
            if _model:
                config.llm.model = _model
            check_reachability(config)
            cmd_commit(args, config)
        elif args.command == "prompts":
            from agent.cli.debug import cmd_prompts
            cmd_prompts(args, config)
        elif args.command == "debug":
            from agent.cli.debug import cmd_debug_context
            cmd_debug_context(args, config)
        elif args.command == "cron":
            from agent.cli.cron import cmd_cron
            if getattr(args, "cron_action", None) == "run":
                check_reachability(config)
            cmd_cron(args, config)
        elif args.command == "todo":
            from agent.cli.todo import cmd_todo
            raise SystemExit(cmd_todo(args, config))
        elif args.command == "diag":
            from agent.cli.diag import cmd_diag
            cmd_diag(args, config)
        elif args.command == "serve":
            from agent.cli.serve import cmd_serve
            cmd_serve(args, config)
        elif args.command == "exec":
            from agent.tools.exec_command import handle_exec_command
            handle_exec_command(args, config)
        else:
            parser.print_help()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        # A plain Ctrl+C during teardown otherwise surfaces as a stack of
        # CancelledError tracebacks — exit quietly with the conventional code.
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        import logging as _logging
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

        # Log full traceback to file; show friendly message on screen.
        _logging.getLogger(__name__).error(
            "Unhandled exception in command %r", getattr(args, "command", None),
            exc_info=True,
        )

        _user_msg = _friendly_error(exc)
        if dump_path:
            _user_msg += f"\nDetails logged to: {dump_path}"
        print(_user_msg, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

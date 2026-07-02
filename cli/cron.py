from __future__ import annotations

"""`agent cron` — run/manage scheduled jobs from an external scheduler.

Point a crontab or systemd-timer at the project and let it fire due jobs:

    */5 * * * *  cd /home/user/src/assistant && agent cron run

`run` claims due jobs under the store flock, so it coexists safely with a
long-running `agent chat` ticker on the same project.
"""


def cmd_cron(args, config):
    from rich.console import Console
    from agent.core import scheduler

    console = Console()
    action = getattr(args, "cron_action", None) or "list"

    if action == "run":
        if getattr(args, "job", None):
            # Force one job now, regardless of its next_run.
            jobs = scheduler.list_jobs(config)
            job = next((j for j in jobs if j.id == args.job or j.name == args.job), None)
            if job is None:
                console.print(f"[red]No job '{args.job}'.[/red]")
                return
            import asyncio
            console.print(f"Running job '{job.name or job.id}' now…")
            reply = asyncio.run(scheduler.execute_job(config, job))
            console.print(reply or "[dim](no reply)[/dim]")
            return
        n = scheduler.run_due_jobs(
            config,
            kinds=("cron", "every", "at", "idle"),
            on_progress=lambda msg: console.print(f"[dim]{msg}[/dim]"),
        )
        console.print(f"Ran {n} due job(s)." if n else "No jobs due.")
        return

    if action == "add":
        console.print(scheduler.run_schedule_command(
            config,
            "add " + args.spec + " :: " + args.prompt
            + (f" :: {args.name}" if getattr(args, "name", None) else ""),
        ))
        return

    if action == "rm":
        console.print(scheduler.run_schedule_command(config, f"rm {args.job}"))
        return

    if action in ("enable", "disable"):
        console.print(scheduler.run_schedule_command(
            config, f"{'on' if action == 'enable' else 'off'} {args.job}"
        ))
        return

    if action == "runs":
        console.print(scheduler.run_schedule_command(config, "runs"))
        return

    console.print(scheduler.run_schedule_command(config, "list"))

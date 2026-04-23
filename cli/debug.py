from __future__ import annotations


def cmd_debug_context(args, config):
    from agent.memory.session import load_session
    from agent.memory.compactor import _count_tokens_approx
    from rich.console import Console
    from rich.table import Table
    import json

    console = Console()
    session_name = args.session or "default"
    session, messages = load_session(session_name)
    if not messages:
        console.print("No session loaded.")
        return

    label = (session.name or session.short_name or session.id) if session else session_name
    table = Table(title=f"Context: session '{label}'", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Role", width=10)
    table.add_column("Tokens", justify="right", width=7)
    table.add_column("Flags", width=14)
    table.add_column("Preview", ratio=1)

    total = 0
    for i, m in enumerate(messages):
        role = m.get("role", "?")
        content = m.get("content") or ""
        if isinstance(content, list):
            content_str = json.dumps(content)
        else:
            content_str = str(content)

        tool_calls = m.get("tool_calls", [])
        tool_names = [tc["function"]["name"] for tc in tool_calls if isinstance(tc, dict)]

        flags = []
        if m.get("_nudged"):
            flags.append("nudged")
        if "[SESSION SUMMARY]" in content_str:
            flags.append("compacted")
        if tool_names:
            flags.append("tools:" + ",".join(tool_names[:2]))

        toks = _count_tokens_approx([m])
        total += toks

        preview = content_str[:120].replace("\n", "↵")
        if len(content_str) > 120:
            preview += "…"

        table.add_row(str(i), role, str(toks), " ".join(flags), preview)

    console.print(table)
    console.print(f"[bold]Total:[/bold] {total} tokens across {len(messages)} messages")


def cmd_prompts(args, config):
    """`agent prompts {status|recompile|clear}` — manage compiled-prompt cache."""
    from agent import prompt_compiler
    from pathlib import Path

    action = getattr(args, "prompts_action", None) or "status"
    if action == "status":
        rows = prompt_compiler.status(config)
        if not rows:
            print("No compiled prompts cached yet.")
            print(f"  cache dir: {Path(config.tools.working_dir) / config.compile_prompts.cache_dir}")
            print(f"  enabled:   {prompt_compiler.is_enabled(config)}")
            return
        print(
            f"{'name':30}{'model':18}{'status':10}{'why':16}"
            f"{'calls':>6}{'err%':>6}{'tok in':>8}{'tok out':>9}{'save%':>7}{'saved Σ':>10}"
        )
        total_saved = 0
        for r in rows:
            save_pct = r["savings_ratio"] * 100 if r["savings_ratio"] else 0
            total_saved += r["tokens_saved_total"]
            print(
                f"{r['name'][:29]:30}"
                f"{r['model'][:17]:18}"
                f"{r['status']:10}"
                f"{(r['disabled_reason'] or '-')[:15]:16}"
                f"{r['calls']:>6}"
                f"{int(r['error_rate']*100):>5}%"
                f"{r['original_tokens']:>8}"
                f"{r['compiled_tokens']:>9}"
                f"{save_pct:>6.0f}%"
                f"{r['tokens_saved_total']:>10}"
            )
        print(f"\nLifetime tokens saved across all variants: {total_saved}")
        return
    if action == "recompile":
        target = getattr(args, "name", None)
        prompt_compiler.recompile(config, name=target)
        print("Compiling prompts synchronously (model must be idle)...")
        results = prompt_compiler.compile_all(config, name=target)
        if not results:
            print("No prompts matched.")
            return
        for pname, status, msg in results:
            print(f"  {status:10} {pname:30} {msg}")
        return
    if action == "clear":
        n = prompt_compiler.clear(config, name=getattr(args, "name", None))
        print(f"Removed {n} cached entries.")
        return
    print(f"Unknown prompts action: {action}")

from __future__ import annotations


def cmd_sessions(args, config):
    from agent.memory.session import list_sessions, load_session
    from rich.console import Console
    from rich.table import Table
    import datetime

    console = Console()

    if getattr(args, "split", None):
        _split_sessions(args.split, console, dry_run=bool(getattr(args, "dry_run", False)))
        return

    if args.load:
        session, messages = load_session(args.load)
        if session is None:
            console.print(f"Session '{args.load}' not found.")
            return
        label = session.name or session.short_name or session.id
        console.print(f"Session '{label}' ({session.id}): {len(messages)} messages")
        return

    sessions = list_sessions()
    if not sessions:
        console.print("No sessions found.")
        return

    table = Table(title="Sessions")
    table.add_column("ID")
    table.add_column("Short name")
    table.add_column("Name")
    table.add_column("Tags")
    table.add_column("Msgs", justify="right")
    table.add_column("Updated")

    for s in sessions:
        ts = s.get("updated_at") or s.get("created_at")
        updated = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "?"
        table.add_row(
            s["id"],
            s.get("short_name", ""),
            s.get("name", ""),
            ", ".join(s.get("tags", [])),
            str(s["message_count"]),
            updated,
        )

    console.print(table)


def _split_sessions(target: str, console, dry_run: bool = False) -> None:
    """Retro-extract verbose tool-call blobs from existing session.json files."""
    import json
    from agent.memory.session import list_sessions, _get_session_dir, get_session_full_dir
    from agent.memory.side_log import SideLogWriter

    if target == "all":
        ids = [s["id"] for s in list_sessions()]
    else:
        ids = [target]

    for sid in ids:
        try:
            sdir = get_session_full_dir(sid)
        except Exception as e:
            console.print(f"[{sid}] cannot resolve dir: {e}")
            continue

        session_json = sdir / "session.json"
        if not session_json.exists():
            flat = _get_session_dir() / f"{sid}.json"
            if flat.exists():
                session_json = flat
                sdir = flat.parent
            else:
                console.print(f"[{sid}] session.json not found at {session_json}")
                continue

        try:
            data = json.loads(session_json.read_text(encoding="utf-8"))
        except Exception as e:
            console.print(f"[{sid}] parse error: {e}")
            continue

        messages = data.get("messages", [])
        if not isinstance(messages, list):
            console.print(f"[{sid}] unexpected messages type")
            continue

        writer = SideLogWriter(sdir)
        new_messages: list[dict] = []
        reasoning_refs_added = 0
        tool_rows_added = 0
        i = 0
        while i < len(messages):
            m = messages[i]
            if (
                isinstance(m.get("content"), str)
                and "_tool_refs" not in m
                and (
                    (m.get("role") == "system" and m["content"].startswith("[tools:"))
                    or ("<agent_exec " in m.get("content", ""))
                )
            ):
                if not dry_run:
                    seq = writer.append("tool_calls.jsonl", {
                        "turn": None,
                        "tool_call_id": None,
                        "tool": "unknown (retro-split)",
                        "arguments": {},
                        "result": m["content"],
                        "source": "retro_split_from_summary",
                    })
                    new_messages.append({**m, "_tool_refs": [seq]})
                else:
                    new_messages.append(m)
                tool_rows_added += 1
                i += 1
                continue

            if m.get("role") == "assistant" and m.get("tool_calls"):
                j = i + 1
                results: list[dict] = []
                while j < len(messages) and messages[j].get("role") == "tool":
                    results.append(messages[j])
                    j += 1
                refs: list[int] = []
                parts: list[str] = []
                for tc in m["tool_calls"]:
                    if not isinstance(tc, dict):
                        continue
                    name = tc.get("function", {}).get("name", "?")
                    args_raw = tc.get("function", {}).get("arguments", "{}")
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except Exception:
                        args = args_raw
                    raw_result = ""
                    for r in results:
                        if r.get("tool_call_id") == tc.get("id"):
                            raw_result = r.get("content", "")
                            break
                    if not dry_run:
                        seq = writer.append("tool_calls.jsonl", {
                            "turn": None,
                            "tool_call_id": tc.get("id"),
                            "tool": name,
                            "arguments": args,
                            "result": raw_result,
                            "source": "retro_split",
                        })
                        refs.append(seq)
                    tool_rows_added += 1
                    try:
                        arg_str = ", ".join(f"{k}={repr(v)[:40]}" for k, v in list(args.items())[:2]) if isinstance(args, dict) else str(args)[:60]
                    except Exception:
                        arg_str = ""
                    parts.append(f"{name}({arg_str}) → (split)")

                if m.get("content") and str(m["content"]).strip():
                    new_messages.append({"role": "assistant", "content": m["content"]})
                # Build <agent_exec> tags for each tool call in the summary
                exec_tags = "\n".join(
                    f'<agent_exec tool="(retro)" args="{p}">(old-session)</agent_exec>'
                    for p in parts
                ) if parts else "(no tools)"
                summary_msg: dict = {"role": "assistant", "content": exec_tags}
                if refs:
                    summary_msg["_tool_refs"] = refs
                new_messages.append(summary_msg)
                i = j
                continue

            new_messages.append(m)
            i += 1

        if dry_run:
            console.print(
                f"[{sid}] dry-run: would write {tool_rows_added} tool_calls.jsonl rows, "
                f"{reasoning_refs_added} reasoning rows"
            )
            continue

        backup = session_json.with_suffix(".json.bak")
        if not backup.exists():
            backup.write_text(session_json.read_text(encoding="utf-8"), encoding="utf-8")
        data["messages"] = new_messages
        session_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(
            f"[{sid}] split {tool_rows_added} tool rows → {sdir / 'tool_calls.jsonl'} "
            f"(backup at {backup.name})"
        )

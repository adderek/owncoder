"""Slash command handlers for /idea and /ideas."""
from __future__ import annotations

from datetime import datetime


def _fmt_idea_row(idea: dict, short: bool = True) -> str:
    tags = idea.get("tags") or []
    tag_str = f" [{','.join(tags)}]" if tags else ""
    pri = idea.get("priority", 3)
    src = idea.get("source", "?")[0].upper()
    ts = datetime.fromtimestamp(idea["created_at"]).strftime("%Y-%m-%d")
    id_short = idea["id"][-9:]
    status = idea.get("status", "raw")
    title = idea.get("title", "")[:70]
    if short:
        return f"  [{id_short}] [{status:<12}] P{pri} {src}  {ts}  {title}{tag_str}"
    body = idea.get("body", "")
    lines = [
        f"id:      {idea['id']}",
        f"title:   {idea.get('title', '')}",
        f"type:    {idea.get('type', '?')}",
        f"status:  {status}",
        f"priority:{pri}",
        f"source:  {idea.get('source', '?')}",
        f"created: {ts}",
        f"tags:    {', '.join(tags) if tags else '—'}",
    ]
    if idea.get("effort_score") is not None:
        lines.append(f"effort:  {idea['effort_score']}")
    if idea.get("value_score") is not None:
        lines.append(f"value:   {idea['value_score']}")
    if idea.get("plan_ref"):
        lines.append(f"plan:    {idea['plan_ref']}")
    if idea.get("requirements_ref"):
        lines.append(f"reqs:    {idea['requirements_ref']}")
    if body:
        lines += ["", body]
    return "\n".join(lines)


def _apply_idea(agent, arg: str) -> tuple[bool, str]:
    """Handle /idea [subcommand] …  Returns (ok, message)."""
    from agent import ideas as _ideas

    store = _ideas.get_store()
    if store is None:
        return False, "Ideas store not configured."

    parts = arg.strip().split(maxsplit=1)

    # No subcommand — treat entire arg as quick-add title
    if not parts:
        return True, (
            "Usage:\n"
            "  /idea <title>                quick add\n"
            "  /idea add [--type T] [--tags t1,t2] [--priority N] <title> [| <body>]\n"
            "  /idea show <id>\n"
            "  /idea update <id> status=<s> [priority=N] [tags=t1,t2]\n"
            "  /idea done <id>\n"
            "  /idea reject <id>\n"
            "  /ideas                       list open ideas"
        )

    first = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    if first not in ("add", "show", "update", "done", "reject", "list"):
        # Treat as title for quick-add (most common human use)
        title = arg.strip()
        idea_id = store.add(title=title, source="human")
        return True, f"Idea saved [{idea_id[-9:]}]: {title}"

    if first == "add":
        return _cmd_add(store, rest, source="human")

    if first == "show":
        if not rest.strip():
            return False, "Usage: /idea show <id>"
        return _cmd_show(store, rest.strip())

    if first == "update":
        return _cmd_update(store, rest.strip())

    if first in ("done", "reject"):
        if not rest.strip():
            return False, f"Usage: /idea {first} <id>"
        status = "done" if first == "done" else "rejected"
        target = _resolve_id(store, rest.strip())
        if target is None:
            return False, f"Idea not found: {rest.strip()}"
        store.update(target, status=status)
        return True, f"Idea {target[-9:]} → {status}"

    if first == "list":
        return _cmd_list(store, rest.strip())

    return False, f"Unknown subcommand '{first}'."


def _cmd_add(store, arg: str, source: str = "human") -> tuple[bool, str]:
    from agent.ideas.store import IDEA_TYPES

    type_ = "idea"
    tags: list[str] = []
    priority = 3

    # Parse flags: --type T --tags t1,t2 --priority N
    tokens = arg.split()
    i = 0
    title_parts: list[str] = []
    body = ""
    while i < len(tokens):
        t = tokens[i]
        if t == "--type" and i + 1 < len(tokens):
            type_ = tokens[i + 1] if tokens[i + 1] in IDEA_TYPES else "idea"
            i += 2
        elif t == "--tags" and i + 1 < len(tokens):
            tags = [x.strip() for x in tokens[i + 1].split(",") if x.strip()]
            i += 2
        elif t == "--priority" and i + 1 < len(tokens):
            try:
                priority = max(1, min(5, int(tokens[i + 1])))
            except ValueError:
                pass
            i += 2
        else:
            title_parts.append(t)
            i += 1

    # Support "title | body" split
    joined = " ".join(title_parts)
    if "|" in joined:
        title, _, body = joined.partition("|")
        title = title.strip()
        body = body.strip()
    else:
        title = joined.strip()

    if not title:
        return False, "Usage: /idea add <title>"

    idea_id = store.add(
        title=title, body=body, type=type_, tags=tags,
        source=source, priority=priority,
    )
    tag_str = f" [{','.join(tags)}]" if tags else ""
    return True, f"Idea saved [{idea_id[-9:]}] ({type_}){tag_str}: {title}"


def _cmd_show(store, id_frag: str) -> tuple[bool, str]:
    target = _resolve_id(store, id_frag)
    if target is None:
        return False, f"Idea not found: {id_frag}"
    idea = store.get(target)
    if idea is None:
        return False, f"Idea not found: {id_frag}"
    return True, _fmt_idea_row(idea, short=False)


def _cmd_update(store, arg: str) -> tuple[bool, str]:
    tokens = arg.split()
    if not tokens:
        return False, "Usage: /idea update <id> field=value …"
    id_frag = tokens[0]
    target = _resolve_id(store, id_frag)
    if target is None:
        return False, f"Idea not found: {id_frag}"
    fields: dict = {}
    for tok in tokens[1:]:
        if "=" in tok:
            k, _, v = tok.partition("=")
            k = k.strip()
            v = v.strip()
            if k == "tags":
                fields["tags"] = [x.strip() for x in v.split(",") if x.strip()]
            elif k == "priority":
                try:
                    fields["priority"] = max(1, min(5, int(v)))
                except ValueError:
                    pass
            elif k in ("status", "title", "type", "body", "effort_score", "value_score",
                       "plan_ref", "requirements_ref"):
                fields[k] = v
    if not fields:
        return False, "No valid fields to update. Example: /idea update <id> status=evaluated priority=4"
    store.update(target, **fields)
    changed = " ".join(f"{k}={v}" for k, v in fields.items())
    return True, f"Updated [{target[-9:]}]: {changed}"


def _cmd_list(store, arg: str) -> tuple[bool, str]:
    status = arg.strip() or None
    ideas = store.list(status=status, limit=50)
    if not ideas:
        label = f"status={status}" if status else "any status"
        return True, f"No ideas ({label})."
    lines = [f"Ideas ({len(ideas)}):"]
    lines += [_fmt_idea_row(i) for i in ideas]
    return True, "\n".join(lines)


def _apply_ideas(agent, arg: str) -> tuple[bool, str]:
    """Handle /ideas [status_filter]."""
    from agent import ideas as _ideas

    store = _ideas.get_store()
    if store is None:
        return False, "Ideas store not configured."
    status = arg.strip() or None
    ideas = store.list(status=status, limit=50)
    if not ideas:
        label = f"with status={status}" if status else "(none yet)"
        return True, f"No ideas {label}."
    lines = [f"Ideas ({len(ideas)}):"]
    lines += [_fmt_idea_row(i) for i in ideas]
    return True, "\n".join(lines)


def _resolve_id(store, id_frag: str) -> str | None:
    """Match idea by full ID or trailing fragment."""
    ideas = store.list(limit=500)
    for idea in ideas:
        if idea["id"] == id_frag or idea["id"].endswith(id_frag):
            return idea["id"]
    return None

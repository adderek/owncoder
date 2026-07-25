"""`agent todo` — the backlog from a terminal, and a way out of it.

The idea store (`.agent/ideas.db`) has been reachable only from inside a chat
session, via `/idea`. That makes it invisible to everything else a person uses:
scripts, cron, a second terminal, a review before starting work. This exposes
the same store as a command.

Deliberately small, and deliberately exportable. This is a place to keep work
items until there is a real tracker, not an attempt to become one: no
assignees, no sprints, no workflow engine. `export`/`import` round-trip the
whole backlog as JSON with ids preserved, so moving to Jira or anything else is
a script over that file rather than a migration of this one.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

#: Export payload version. Bump when the record shape changes incompatibly, so
#: an importer can refuse a file it would misread rather than half-apply it.
EXPORT_VERSION = 1


def _store(config):
    from agent import ideas as _ideas

    _ideas.configure(config.tools.working_dir, config.tools.agent_dir)
    return _ideas.get_store()


def _row(idea: dict) -> str:
    tags = idea.get("tags") or []
    created = datetime.fromtimestamp(idea["created_at"]).strftime("%Y-%m-%d")
    return (f"{idea['id'][-9:]}  {idea.get('status', 'raw'):<12} "
            f"P{idea.get('priority', 3)} {idea.get('type', 'idea'):<13} {created}  "
            f"{(idea.get('title') or '')[:60]}"
            + (f"  [{','.join(tags)}]" if tags else ""))


def _detail(idea: dict) -> str:
    lines = [f"id:       {idea['id']}",
             f"title:    {idea.get('title', '')}",
             f"type:     {idea.get('type', 'idea')}",
             f"status:   {idea.get('status', 'raw')}",
             f"priority: {idea.get('priority', 3)}",
             f"source:   {idea.get('source', '?')}",
             f"created:  {datetime.fromtimestamp(idea['created_at']).isoformat(' ', 'seconds')}",
             f"updated:  {datetime.fromtimestamp(idea['updated_at']).isoformat(' ', 'seconds')}"]
    for key, label in (("tags", "tags"), ("plan_ref", "plan"),
                       ("requirements_ref", "reqs"), ("session_ref", "session")):
        value = idea.get(key)
        if value:
            lines.append(f"{label + ':':<10}{', '.join(value) if isinstance(value, list) else value}")
    if idea.get("body"):
        lines += ["", idea["body"]]
    return "\n".join(lines)


def _resolve(store, needle: str) -> dict | None:
    """Accept a full id or the short suffix the listing prints.

    Ambiguity is an error rather than a pick: acting on the wrong backlog item
    is silent, and the user has no way to notice it happened.
    """
    exact = store.get(needle)
    if exact is not None:
        return exact
    matches = [i for i in store.list(limit=10_000) if i["id"].endswith(needle)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(f"'{needle}' matches {len(matches)} items; use the full id")
    return None


def _parse_pairs(pairs: list[str]) -> dict:
    fields = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"expected key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        key = key.strip()
        value = value.strip()
        if key == "tags":
            fields[key] = [t.strip() for t in value.split(",") if t.strip()]
        elif key in ("priority",):
            fields[key] = int(value)
        elif key in ("effort_score", "value_score"):
            fields[key] = float(value)
        else:
            fields[key] = value
    return fields


def cmd_todo(args, config) -> int:
    from agent.ideas.store import IDEA_STATUSES, IDEA_TYPES

    store = _store(config)
    if store is None:
        print("backlog unavailable: could not open .agent/ideas.db", file=sys.stderr)
        return 1
    action = getattr(args, "todo_action", None) or "list"

    if action == "list":
        status = getattr(args, "status", None)
        if status and status not in IDEA_STATUSES:
            print(f"unknown status {status!r}; one of: {', '.join(IDEA_STATUSES)}",
                  file=sys.stderr)
            return 1
        items = store.list(status=status, limit=getattr(args, "limit", 50))
        wanted_type = getattr(args, "type", None)
        if wanted_type:
            items = [i for i in items if i.get("type") == wanted_type]
        if getattr(args, "json", False):
            print(json.dumps(items, ensure_ascii=False, indent=2))
            return 0
        if not items:
            print("backlog empty" if not status else f"nothing with status {status!r}")
            return 0
        for idea in items:
            print(_row(idea))
        print(f"\n{len(items)} item(s); {store.count()} total")
        return 0

    if action == "add":
        title = " ".join(args.title).strip()
        if not title:
            print("nothing to add: give a title", file=sys.stderr)
            return 1
        if args.type not in IDEA_TYPES:
            print(f"unknown type {args.type!r}; one of: {', '.join(IDEA_TYPES)}",
                  file=sys.stderr)
            return 1
        idea_id = store.add(
            title=title, body=args.body or "", type=args.type,
            tags=[t.strip() for t in (args.tags or "").split(",") if t.strip()],
            source="human", priority=max(1, min(5, args.priority)),
        )
        print(idea_id)
        return 0

    if action in ("show", "set", "done", "reject"):
        idea = _resolve(store, args.id)
        if idea is None:
            print(f"no backlog item matching {args.id!r}", file=sys.stderr)
            return 1
        if action == "show":
            print(_detail(idea))
            return 0
        if action == "done":
            fields = {"status": "done"}
        elif action == "reject":
            fields = {"status": "rejected"}
        else:
            fields = _parse_pairs(args.fields)
            status = fields.get("status")
            if status is not None and status not in IDEA_STATUSES:
                print(f"unknown status {status!r}; one of: {', '.join(IDEA_STATUSES)}",
                      file=sys.stderr)
                return 1
        if not store.update(idea["id"], **fields):
            print("nothing updated (no writable fields given)", file=sys.stderr)
            return 1
        print(f"{idea['id']}  {', '.join(f'{k}={v}' for k, v in fields.items())}")
        return 0

    if action == "export":
        payload = {
            "version": EXPORT_VERSION,
            "exported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "items": store.list(limit=100_000),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        out = getattr(args, "out", None)
        if out:
            Path(out).write_text(text + "\n", encoding="utf-8")
            print(f"{len(payload['items'])} item(s) → {out}")
        else:
            print(text)
        return 0

    if action == "import":
        try:
            payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"cannot read {args.file}: {e}", file=sys.stderr)
            return 1
        version = payload.get("version") if isinstance(payload, dict) else None
        if isinstance(payload, dict) and version != EXPORT_VERSION:
            print(f"export version {version!r} != {EXPORT_VERSION}; refusing to "
                  f"guess at the format", file=sys.stderr)
            return 1
        items = payload.get("items") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            print("no items in that file", file=sys.stderr)
            return 1
        written = sum(1 for item in items if isinstance(item, dict) and store.upsert(item))
        print(f"{written} item(s) imported")
        return 0

    print(f"unknown todo action {action!r}", file=sys.stderr)
    return 1

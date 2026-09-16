"""Single source of truth for *what the agent may do with a path*.

Before this module the answer was spread over four places that could not see
each other: root-relative deny globs in ``fs.py``, a read-only bind list in
``runner.py``, ``.agent.ignore`` patterns in ``tools/rules``, and the user's
``[security] grant_ceiling`` in ``path_grants.py``. Each judged a path against
a different base, so a grant outside the project root silently switched which
rules applied, and "too big to scan" and "too dangerous to touch" were written
in the same list.

The model here has three independent axes:

``Access``
    The *ceiling* on what may ever be done with a path: ``NONE`` (not even its
    existence is exposed), ``LIST`` (name visible, contents not), ``READ``,
    ``WRITE``. A grant can only ever lower this, never raise it.

``exact_only``
    The path is reachable only by a grant that names it exactly. A grant of a
    parent directory does not confer access. This is how ``/dev`` works: the
    user can hand over ``/dev/video0`` without handing over ``/dev/mem``.

``hidden``
    A *performance and noise* flag, not a security one. Hidden paths are
    skipped by the indexer, by search, and by the sandbox's per-command walk.
    A hidden path may still be read and written — this is the distinction the
    old ``.agent.ignore`` hack conflated.

Rule scope
----------
``Scope.ALWAYS`` rules hold everywhere, including inside the project root: the
agent's own control plane and private keys stay protected even when the user
starts the agent in the directory that holds them. ``Scope.OUTSIDE`` rules
(shell rc files, ``/etc``, system trees) are lifted inside the project root —
starting the agent in ``~`` is a deliberate act that says "this tree is what I
am working on".

Overrides
---------
A user-config ceiling entry that names a path *exactly* may raise a rule's
ceiling, up to that entry's own mode. A ceiling entry covering a *parent*
never does: pre-approving ``/home/me`` must not quietly hand over
``~/.ssh/id_ed25519``. Rules marked ``no_override`` cannot be raised at all —
those are the paths whose contents decide what the agent is allowed to do
(grants, permissions, the ceiling itself, the core prompt), where a raise
would be the agent approving its own access one file removed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from pathlib import Path


class Access(IntEnum):
    NONE = 0     # existence not exposed; every operation refused
    LIST = 1     # may appear in a directory listing; contents refused
    READ = 2
    WRITE = 3


_MODE_TO_ACCESS = {"none": Access.NONE, "list": Access.LIST,
                   "ro": Access.READ, "rw": Access.WRITE}
_ACCESS_TO_MODE = {Access.NONE: "none", Access.LIST: "list",
                   Access.READ: "ro", Access.WRITE: "rw"}


def mode_to_access(mode: str) -> Access:
    return _MODE_TO_ACCESS.get(mode, Access.NONE)


def access_to_mode(access: Access) -> str:
    return _ACCESS_TO_MODE[access]


class Scope(IntEnum):
    ALWAYS = 0      # holds inside the project root too
    OUTSIDE = 1     # lifted for paths under the project root


@dataclass(frozen=True)
class Rule:
    pattern: str            # absolute glob; `<agent_dir>` expanded at match time
    max: Access
    why: str
    scope: Scope = Scope.ALWAYS
    exact_only: bool = False
    hidden: bool = False
    no_override: bool = False


@dataclass(frozen=True)
class Decision:
    """What the built-in rules allow for one path, before any grant."""
    max: Access
    exact_only: bool
    hidden: bool
    why: str
    no_override: bool

    @property
    def mode(self) -> str:
        return access_to_mode(self.max)


# ── Rule table ─────────────────────────────────────────────────────────────
#
# `<agent_dir>` is substituted with the project's agent directory at match
# time, so a relocated `tools.agent_dir` is covered without restating the list.

#: The agent's own control plane. Readable — the agent should be able to see
#: the policy binding it — but never writable, and never raisable: every file
#: here is read back by the host to decide what the agent may do next, so a
#: write is a grant the user never approved.
_CONTROL_PLANE: tuple[Rule, ...] = (
    Rule("<agent_dir>/path_grants.json", Access.READ,
         "the grant list itself — writable means self-granting", no_override=True),
    Rule("<agent_dir>/permissions.json", Access.READ,
         "the permission rules binding the agent", no_override=True),
    Rule("<agent_dir>/**/session.json", Access.READ,
         "carries a path_grants snapshot restored on session switch",
         no_override=True),
    Rule("<agent_dir>/**/*.toml", Access.READ,
         "agent configuration", no_override=True),
    Rule("<agent_dir>/agent.preamble", Access.READ,
         "read verbatim into every system prompt", no_override=True),
    Rule("<agent_dir>/compiled_prompts/**", Access.READ,
         "served in place of the shipped system prompt", no_override=True),
    Rule("<agent_dir>/core.md", Access.READ,
         "immutable core of the system prompt", no_override=True),
    Rule("<agent_dir>/core_history.jsonl", Access.READ,
         "accountability record for core prompt changes", no_override=True),
    Rule("<agent_dir>/audit.jsonl", Access.READ,
         "record of what the agent ran", no_override=True),
    Rule("<agent_dir>/audit.*.jsonl", Access.READ,
         "rotated audit log", no_override=True),
    Rule("<agent_dir>/audit/**", Access.READ,
         "record of what the agent ran", no_override=True),
    Rule("<agent_dir>/checkpoints/**", Access.READ,
         "record of the agent's own edits", no_override=True),
    Rule("<agent_dir>/diagnostics/**", Access.READ,
         "crash and failure records: readable, not forgeable", no_override=True),
    Rule("<agent_dir>/web_search/**", Access.READ,
         "app-generated fetcher runs with network access", no_override=True),
    # Tool-mediated stores: reached through save_note / index_code / submit_idea,
    # which open sqlite in the host process below this gate. A file write here
    # is either corruption or the agent editing what it must only reach by tool.
    Rule("<agent_dir>/*.db", Access.READ, "tool-mediated store", no_override=True),
    Rule("<agent_dir>/*.db-wal", Access.READ, "tool-mediated store", no_override=True),
    Rule("<agent_dir>/*.db-shm", Access.READ, "tool-mediated store", no_override=True),
    Rule("<agent_dir>/*.db.enc", Access.READ, "sealed store image", no_override=True),
    Rule("<agent_dir>/**/memory.db*", Access.READ, "tool-mediated store",
         no_override=True),
    # Project-level policy files, by name anywhere in the tree.
    Rule("**/agent.toml", Access.READ, "agent configuration", no_override=True),
    Rule("**/.agent.toml", Access.READ, "agent configuration", no_override=True),
    Rule("**/.agent.ignore", Access.READ, "path rules", no_override=True),
    Rule("**/.agent.ro", Access.READ, "path rules", no_override=True),
    Rule("**/.agent.config", Access.READ, "path rules", no_override=True),
    Rule("**/.agent.sandbox", Access.READ, "sandbox allowlist", no_override=True),
    Rule("**/.agent.approve", Access.READ, "approval rules", no_override=True),
    Rule("**/.agent.log", Access.READ, "audit configuration", no_override=True),
    Rule("**/.agent.boundary", Access.READ, "boundary rules", no_override=True),
    Rule("**/.agent.priorities.toml", Access.READ, "agent configuration",
         no_override=True),
    Rule("**/prompts/core.txt", Access.READ, "immutable core of the system prompt",
         no_override=True),
    # Other agents' configuration: writing it hijacks that agent, not this one,
    # which is someone else's session to lose. Raisable by an exact entry.
    Rule("**/.claude/**", Access.READ, "another agent's configuration"),
    Rule("**/.claude/**", Access.READ,
         "another agent's configuration: its hooks run commands"),
    Rule("**/.config/agent/**", Access.READ,
         "the user config that defines the grant ceiling", no_override=True),
    Rule("**/.config/agent/*.token", Access.NONE, "credential", no_override=True),
    Rule("**/.config/agent/*.key", Access.NONE, "credential", no_override=True),
    # git: config decides what `git` executes (core.hooksPath, aliases), hooks
    # are executables git runs on ordinary commands.
    Rule("**/.git/**", Access.READ, "git internals", hidden=True),
    Rule("**/.git/hooks/**", Access.READ,
         "executed by ordinary git commands", no_override=True),
    Rule("**/.git/config", Access.READ,
         "core.hooksPath and aliases decide what git executes", no_override=True),
    Rule("**/.gitconfig", Access.READ,
         "core.hooksPath and aliases decide what git executes", no_override=True),
    Rule("**/.config/git/**", Access.READ,
         "core.hooksPath and aliases decide what git executes", no_override=True),
)

#: Secrets. NONE rather than READ: a listing that shows `id_ed25519` exists is
#: already half the attack. Raisable by an exact ceiling entry — a user who
#: pre-approves one specific key file by full path means it.
_SECRETS: tuple[Rule, ...] = (
    Rule("**/.ssh/**", Access.NONE, "private keys", hidden=True),
    Rule("**/.gnupg/**", Access.NONE, "private keys", hidden=True),
    Rule("**/.aws/credentials", Access.NONE, "cloud credentials"),
    Rule("**/.aws/config", Access.NONE, "cloud credentials"),
    Rule("**/.netrc", Access.NONE, "stored passwords"),
    Rule("**/.pgpass", Access.NONE, "stored passwords"),
    Rule("**/.env", Access.NONE, "environment secrets"),
    Rule("**/.env.*", Access.NONE, "environment secrets"),
    Rule("**/*.pem", Access.NONE, "private key material"),
    Rule("**/*.key", Access.NONE, "private key material"),
    Rule("**/*.p12", Access.NONE, "private key material"),
    Rule("**/id_rsa*", Access.NONE, "private key material"),
    Rule("**/id_ed25519*", Access.NONE, "private key material"),
    Rule("**/id_ecdsa*", Access.NONE, "private key material"),
    Rule("**/.kube/config", Access.NONE, "cluster credentials"),
    Rule("**/.docker/config.json", Access.NONE, "registry credentials"),
)

#: Paths whose contents decide what gets executed later, outside the project.
#: READ, so the agent can explain what is there; raising one to WRITE is the
#: user's call and needs an exact ceiling entry. Lifted inside the project
#: root: `agent chat` started in `~` means the user is working on these files.
_EXEC_SURFACES: tuple[Rule, ...] = (
    Rule("**/.bashrc", Access.READ, "sourced by every shell", Scope.OUTSIDE),
    Rule("**/.bash_profile", Access.READ, "sourced at login", Scope.OUTSIDE),
    Rule("**/.profile", Access.READ, "sourced at login", Scope.OUTSIDE),
    Rule("**/.zshrc", Access.READ, "sourced by every shell", Scope.OUTSIDE),
    Rule("**/.zshenv", Access.READ, "sourced by every shell", Scope.OUTSIDE),
    Rule("**/.zprofile", Access.READ, "sourced at login", Scope.OUTSIDE),
    Rule("**/.config/fish/**", Access.READ, "sourced by every shell", Scope.OUTSIDE),
    Rule("**/.config/environment.d/**", Access.READ,
         "environment for the user's session", Scope.OUTSIDE),
    Rule("**/.config/systemd/**", Access.READ, "user services", Scope.OUTSIDE),
    Rule("**/.local/share/systemd/**", Access.READ, "user services", Scope.OUTSIDE),
    Rule("**/.config/autostart/**", Access.READ, "runs at login", Scope.OUTSIDE),
    Rule("**/.local/bin/**", Access.READ, "on PATH", Scope.OUTSIDE),
    # Home-anchored, not `**/bin/**`: an ordinary repo's `bin/` is just
    # scripts the user may want edited, but `~/bin` is on their PATH.
    Rule("/home/*/bin/**", Access.READ, "on PATH", Scope.OUTSIDE),
    Rule("/root/bin/**", Access.READ, "on PATH", Scope.OUTSIDE),
    Rule("/var/spool/cron/**", Access.NONE, "scheduled command execution",
         Scope.OUTSIDE, no_override=True),
    Rule("/etc/cron*/**", Access.READ, "scheduled command execution", Scope.OUTSIDE,
         no_override=True),
)

#: System trees. Readable so the agent can inspect the machine it runs on;
#: never writable without an exact entry, and never by way of a broad one.
_SYSTEM: tuple[Rule, ...] = (
    Rule("/etc/**", Access.READ, "system configuration", Scope.OUTSIDE),
    Rule("/etc/shadow*", Access.NONE, "password hashes", Scope.OUTSIDE,
         no_override=True),
    Rule("/etc/gshadow*", Access.NONE, "password hashes", Scope.OUTSIDE,
         no_override=True),
    Rule("/etc/sudoers", Access.NONE, "privilege escalation", Scope.OUTSIDE,
         no_override=True),
    Rule("/etc/sudoers.d/**", Access.NONE, "privilege escalation", Scope.OUTSIDE,
         no_override=True),
    Rule("/etc/ssl/private/**", Access.NONE, "private key material", Scope.OUTSIDE,
         no_override=True),
    Rule("/usr/**", Access.READ, "system files", Scope.OUTSIDE),
    Rule("/bin/**", Access.READ, "system files", Scope.OUTSIDE),
    Rule("/sbin/**", Access.READ, "system files", Scope.OUTSIDE),
    Rule("/lib/**", Access.READ, "system files", Scope.OUTSIDE),
    Rule("/lib64/**", Access.READ, "system files", Scope.OUTSIDE),
    Rule("/opt/**", Access.READ, "system files", Scope.OUTSIDE),
    Rule("/boot/**", Access.NONE, "boot chain", Scope.OUTSIDE, no_override=True),
    Rule("/root/**", Access.NONE, "another user's home", Scope.OUTSIDE),
    # Kernel interfaces: individually useful, catastrophic wholesale. A grant
    # must name the exact file, so `/proc` or `/sys` as a subtree is never
    # handed over by one click.
    Rule("/proc/**", Access.READ, "kernel interface", Scope.OUTSIDE,
         exact_only=True),
    Rule("/proc/*/mem", Access.NONE, "another process's memory", Scope.OUTSIDE,
         no_override=True),
    Rule("/proc/sys/**", Access.READ, "kernel tunables", Scope.OUTSIDE,
         exact_only=True, no_override=True),
    Rule("/proc/*/environ", Access.NONE, "another process's secrets",
         Scope.OUTSIDE, no_override=True),
    Rule("/sys/**", Access.READ, "kernel interface", Scope.OUTSIDE,
         exact_only=True),
    Rule("/sys/firmware/efi/efivars/**", Access.NONE, "firmware variables",
         Scope.OUTSIDE, no_override=True),
    # Devices: grantable, but one at a time and by full path. `/dev/video0` for
    # a capture task is reasonable; `/dev` as a subtree never is.
    Rule("/dev/**", Access.WRITE, "device node", Scope.OUTSIDE, exact_only=True),
    Rule("/dev/mem", Access.NONE, "physical memory", Scope.OUTSIDE, no_override=True),
    Rule("/dev/kmem", Access.NONE, "kernel memory", Scope.OUTSIDE, no_override=True),
    Rule("/dev/port", Access.NONE, "I/O ports", Scope.OUTSIDE, no_override=True),
    Rule("/dev/kmsg", Access.NONE, "kernel log", Scope.OUTSIDE, no_override=True),
    Rule("/dev/sd*", Access.NONE, "raw disk", Scope.OUTSIDE, no_override=True),
    Rule("/dev/nvme*", Access.NONE, "raw disk", Scope.OUTSIDE, no_override=True),
    Rule("/dev/vd*", Access.NONE, "raw disk", Scope.OUTSIDE, no_override=True),
    Rule("/dev/hd*", Access.NONE, "raw disk", Scope.OUTSIDE, no_override=True),
    Rule("/dev/mapper/**", Access.NONE, "raw disk", Scope.OUTSIDE, no_override=True),
    Rule("/dev/dm-*", Access.NONE, "raw disk", Scope.OUTSIDE, no_override=True),
    Rule("/dev/loop*", Access.NONE, "raw disk", Scope.OUTSIDE, no_override=True),
    # Sockets that are an escape hatch to another privilege domain.
    Rule("/var/run/docker.sock", Access.NONE, "container escape", Scope.OUTSIDE,
         no_override=True),
    Rule("/run/docker.sock", Access.NONE, "container escape", Scope.OUTSIDE,
         no_override=True),
    Rule("/run/user/*/bus", Access.NONE, "session bus", Scope.OUTSIDE,
         no_override=True),
    Rule("/run/**", Access.READ, "runtime state", Scope.OUTSIDE, exact_only=True),
    Rule("/var/run/**", Access.READ, "runtime state", Scope.OUTSIDE,
         exact_only=True),
)

#: Hidden, not forbidden. Skipped by the indexer, by search and by the
#: sandbox's per-command walk because they are large and churn, *not* because
#: they are dangerous: access is whatever the surrounding rules say. Keeping
#: this separate is the point — the old `.agent.ignore` had to make a path
#: unreadable in order to make it uncrawled.
_HIDDEN: tuple[Rule, ...] = (
    Rule("<agent_dir>/**", Access.WRITE, "agent state: large and churns",
         hidden=True),
    # Any project's agent directory, not only this one's: a checkout the user
    # granted brings its own sessions and stores with it.
    Rule("**/.agent/**", Access.WRITE, "agent state: large and churns",
         hidden=True),
    Rule("**/.coord/**", Access.WRITE, "coordination state", hidden=True),
    Rule("**/node_modules/**", Access.WRITE, "dependency tree", hidden=True),
    Rule("**/.venv/**", Access.WRITE, "installed libraries", hidden=True),
    Rule("**/__pycache__/**", Access.WRITE, "build artifact", hidden=True),
    Rule("**/.mypy_cache/**", Access.WRITE, "tool cache", hidden=True),
    Rule("**/.pytest_cache/**", Access.WRITE, "tool cache", hidden=True),
    Rule("**/.ruff_cache/**", Access.WRITE, "tool cache", hidden=True),
    Rule("**/.gradle-cache/**", Access.WRITE, "tool cache", hidden=True),
    Rule("**/venv/**", Access.WRITE, "installed libraries", hidden=True),
    Rule("**/build/**", Access.WRITE, "build output", hidden=True),
    Rule("**/dist/**", Access.WRITE, "build output", hidden=True),
    Rule("**/graphify-out/**", Access.WRITE, "generated output", hidden=True),
    Rule("**/.git/objects/**", Access.READ, "object store: large", hidden=True),
)

RULES: tuple[Rule, ...] = (
    _HIDDEN + _SYSTEM + _EXEC_SURFACES + _CONTROL_PLANE + _SECRETS
)


# ── Matching ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=4096)
def _compile(pattern: str) -> re.Pattern:
    """Translate a path glob to a regex.

    `**` spans directories, `*` and `?` do not. A trailing `/**` also matches
    the directory itself, so a rule on `~/.ssh/**` covers `~/.ssh`.
    """
    out = ["^"]
    i = 0
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if pattern.startswith("/**/", i):
            out.append("/(?:.*/)?")
            i += 4
        elif pattern.startswith("/**", i) and i + 3 == n:
            out.append("(?:/.*)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    out.append("$")
    return re.compile("".join(out))


def _specificity(pattern: str) -> tuple[int, int, int]:
    """Rank patterns so the most specific rule wins a tie.

    Ordered by: number of literal path segments, then literal characters, then
    fewer wildcards. `**/.git/hooks/**` therefore beats `**/.git/**`, and
    `/dev/mem` beats `/dev/**`.
    """
    segments = [s for s in pattern.split("/") if s and "*" not in s and "?" not in s]
    literal = len(re.sub(r"[*?]", "", pattern))
    wildcards = pattern.count("*") + pattern.count("?")
    return (len(segments), literal, -wildcards)


def _expand(pattern: str, agent_dir: Path | None) -> str | None:
    if pattern.startswith("<agent_dir>"):
        if agent_dir is None:
            return None
        return str(agent_dir) + pattern[len("<agent_dir>"):]
    if pattern.startswith("~"):
        return str(Path(pattern).expanduser())
    return pattern


def _under(parent: Path, child: Path) -> bool:
    return child == parent or parent in child.parents


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(path: str | Path, *, root: Path | None = None,
             agent_dir: Path | None = None,
             rules: tuple[Rule, ...] = RULES) -> Decision:
    """The built-in ceiling for *path*, before grants and before the user's
    configured ceiling.

    *root* and *agent_dir* default to the live policy when one is set up, so
    callers inside the app can omit them; tests pass them explicitly.
    """
    if root is None or agent_dir is None:
        from . import policy as _policy
        if _policy.is_configured():
            pol = _policy.get()
            root = root if root is not None else pol.root
            agent_dir = agent_dir if agent_dir is not None else pol.agent_dir

    p = Path(path)
    s = str(p)
    in_project = root is not None and _under(root, p)

    best: Rule | None = None
    best_rank: tuple[int, int, int] = (-1, -1, -1)
    hidden = False
    for rule in rules:
        if rule.scope is Scope.OUTSIDE and in_project:
            continue
        pattern = _expand(rule.pattern, agent_dir)
        if pattern is None:
            continue
        if not _compile(pattern).match(s):
            continue
        if rule.hidden:
            hidden = True
        rank = _specificity(pattern)
        # A tie goes to the stricter rule: two equally specific patterns that
        # disagree should not depend on table order.
        if rank > best_rank or (rank == best_rank and best is not None
                                and rule.max < best.max):
            best, best_rank = rule, rank

    if best is None:
        return Decision(Access.WRITE, False, hidden, "", False)
    return Decision(best.max, best.exact_only, hidden or best.hidden, best.why,
                    best.no_override)


def hidden_for_scan(path: str | Path, *, root: Path | None = None,
                    agent_dir: Path | None = None) -> bool:
    """True when *path* should be skipped by indexing, search and the
    per-command sandbox walk. Says nothing about access."""
    return evaluate(path, root=root, agent_dir=agent_dir).hidden


def max_access(path: str | Path, *, root: Path | None = None,
               agent_dir: Path | None = None,
               exact_ceiling: Access | None = None) -> Decision:
    """Built-in ceiling for *path*, raised by an exact ceiling entry.

    *exact_ceiling* is the mode of a user-config ceiling entry naming this path
    exactly (not a parent). It can raise a raisable rule up to its own mode; a
    ``no_override`` rule is unaffected.
    """
    d = evaluate(path, root=root, agent_dir=agent_dir)
    if exact_ceiling is None or d.no_override or exact_ceiling <= d.max:
        return d
    return Decision(exact_ceiling, d.exact_only, d.hidden,
                    f"{d.why} (raised by an exact grant_ceiling entry)", False)


def describe(path: str | Path, **kw) -> str:
    """One line for the UI and for refusal messages."""
    d = max_access(path, **kw)
    if d.max is Access.WRITE and not d.exact_only:
        return f"{path}: no built-in restriction"
    bits = [f"max {d.mode}"]
    if d.exact_only:
        bits.append("exact grant only")
    if d.hidden:
        bits.append("hidden from indexing")
    if d.no_override:
        bits.append("not raisable")
    tail = f" — {d.why}" if d.why else ""
    return f"{path}: {', '.join(bits)}{tail}"


def hidden_dir_names() -> frozenset[str]:
    """Directory names that every tree walk should prune.

    The indexer, grep, the RAG maintainer and the sandbox's per-command scan
    each carried their own copy of this list, and they had drifted apart. They
    read it from here instead, so "don't descend into node_modules" is written
    once — and stays separate from "must not be read", which is `Access`.

    The project's own agent directory is included when a policy is set up: it
    is the biggest churn source of all (one subtree per session).
    """
    names = set()
    for rule in RULES:
        # Only prunable subtrees. A NONE rule is hidden too (`~/.ssh`), but the
        # sandbox's secret-mask walk has to *find* those files in order to mask
        # them — pruning the directory would quietly unmask everything in it.
        if not rule.hidden or rule.max < Access.READ:
            continue
        parts = rule.pattern.split("/")
        if (len(parts) == 3 and parts[0] == "**" and parts[2] == "**"
                and "*" not in parts[1]):
            names.add(parts[1])
    try:
        from . import policy as _policy
        if _policy.is_configured():
            names.add(_policy.get().agent_dir.name)
    except Exception:       # policy not importable in a bare unit test
        pass
    return frozenset(names)


def project_readonly_names() -> tuple[str, ...]:
    """Root-relative names the sandbox binds read-only inside the project.

    Derived from the rule table rather than restated, so the bwrap bind list
    and the fs gate cannot disagree about what a shell may overwrite. Only
    single-segment `**/name` and `**/name/**` rules qualify — anything deeper
    is found by the per-command walk instead.
    """
    out = []
    for rule in RULES:
        # READ-capped only. A NONE path (a key, `.env`) is masked with
        # /dev/null by the secret pass; binding it read-only as well would
        # leave a readable secret behind the first matching bind.
        if rule.scope is not Scope.ALWAYS or rule.max != Access.READ:
            continue
        parts = rule.pattern.split("/")
        if parts[0] != "**" or "*" in "".join(parts[1:2]):
            continue
        if len(parts) == 2 or (len(parts) == 3 and parts[2] == "**"):
            if parts[1] not in out:
                out.append(parts[1])
    return tuple(out)

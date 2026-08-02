"""Startup profile detection — probe endpoints, suggest a model-mode profile.

Entirely coded (no LLM): every distinct configured endpoint is probed in
parallel via GET /models, offline hosts are reported, and the best matching
model-mode profile is suggested from which cost tiers actually answered.
In interactive chat the user confirms/overrides the suggestion before the
session starts; non-interactive runs apply the suggestion automatically.

The chosen mode governs AUTOMATIC selection only (background/idle work,
spawn_agents decision-maker) — an explicitly pinned [models] default is
always honored, same contract as `/mode`.
"""
from __future__ import annotations

import concurrent.futures
import logging
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable
from urllib.parse import urlparse

from agent.config.registry import MODE_TIERS, entry_tier, is_lan_entry

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

# Per-endpoint probe timeout; probes run in parallel so worst case the whole
# check costs ~one timeout, not one per endpoint.
PROBE_TIMEOUT_S = 3

_MODE_ORDER = ["local-only", "lan-only", "free-cloud", "free-hybrid", "paid-cloud",
               "manual", "any"]


@dataclass
class HostStatus:
    """Aggregated probe result for one host (a host may serve several ports)."""
    host: str
    online: bool                       # any endpoint on this host answered
    tiers: set = field(default_factory=set)    # cost tiers of its entries
    entries: list = field(default_factory=list)  # entry names, sorted
    has_embeddings: bool = False


@dataclass
class ProfileReport:
    hosts: list                        # [HostStatus], offline first
    reachable_tiers: set               # cost tiers with >=1 live endpoint
    suggested: str                     # suggested model-mode
    reason: str                        # one-line why
    embeddings_ok: bool                # any embeddings-capable endpoint live
    embed_start_hint: bool = False     # rag.embed_server_command configured


def _host_label(base_url: str) -> str:
    try:
        p = urlparse(base_url)
        host = p.hostname or base_url
        return f"{host}:{p.port}" if p.port else host
    except Exception:
        return base_url


def _default_probe(base_url: str, api_key: str) -> bool:
    from agent.config.loader import _probe_models
    return _probe_models(base_url, api_key, timeout=PROBE_TIMEOUT_S) is not None


def _is_embeddings_entry(name: str, entry, config: "Config | None") -> bool:
    if "embeddings" in (getattr(entry, "tags", None) or []):
        return True
    if config is None:
        return False
    pool = config.model_pools.get("embeddings", [])
    return name == config.model_roles.get("embeddings") or name in pool


def suggest_mode(reachable_tiers: set, current: str) -> tuple[str, str]:
    """Map the set of live cost tiers to the best matching model-mode.

    Preference mirrors the failover philosophy: keep local hardware in the mix
    when it answers, spend free cloud before paid, fall to paid-cloud only when
    nothing free is alive.
    """
    r = reachable_tiers
    if "local" in r and "free" in r:
        return "free-hybrid", "local + free cloud reachable"
    if "local" in r:
        return "local-only", "only local endpoints reachable"
    # LAN box up but this desktop's router is not: keep the work on own
    # hardware instead of falling to the cloud tier LAN entries share.
    if "lan" in r:
        return "lan-only", "LAN server reachable; no local endpoints"
    if "free" in r:
        return "free-cloud", "no local/LAN endpoints; free cloud reachable"
    if "paid" in r or "bundled" in r:
        return "paid-cloud", "only paid cloud endpoints reachable"
    return current, "NO endpoint reachable — keeping current mode"


def prefetch_endpoints(config: "Config") -> None:
    """Warm the probe cache for every configured endpoint, in parallel.

    Called as early as startup can manage — the answers are the same whatever
    profile is chosen, so they can be gathered while the report is being read
    and the prompts answered, instead of one at a time afterwards.
    """
    try:
        from agent.config import probe_cache
        from agent.config.loader import _probe_models_uncached
        pairs = [((getattr(e, "base_url", "") or ""), (getattr(e, "api_key", "") or ""))
                 for e in config.model_entries.values()]
        probe_cache.prefetch(pairs, PROBE_TIMEOUT_S, _probe_models_uncached)
    except Exception:
        logger.debug("endpoint prefetch failed", exc_info=True)


def detect(config: "Config", probe: Callable[[str, str], bool] | None = None) -> ProfileReport:
    """Probe every distinct configured endpoint (parallel) and build a report."""
    probe = probe or _default_probe

    # Group entries by base_url (one probe per URL), then aggregate per host so
    # a dead machine shows as one line, not one line per port.
    by_url: dict[str, list[tuple[str, object]]] = {}
    for name, entry in config.model_entries.items():
        bu = (getattr(entry, "base_url", "") or "").rstrip("/")
        if bu:
            by_url.setdefault(bu, []).append((name, entry))

    results: dict[str, bool] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {
            ex.submit(probe, url, getattr(pairs[0][1], "api_key", "") or ""): url
            for url, pairs in by_url.items()
        }
        for fut in concurrent.futures.as_completed(futs):
            url = futs[fut]
            try:
                results[url] = bool(fut.result())
            except Exception:
                results[url] = False

    hosts: dict[str, HostStatus] = {}
    reachable_tiers: set = set()
    embeddings_ok = False
    for url, pairs in by_url.items():
        try:
            hkey = urlparse(url).hostname or url
        except Exception:
            hkey = url
        st = hosts.setdefault(hkey, HostStatus(host=hkey, online=False))
        online = results.get(url, False)
        st.online = st.online or online
        for name, entry in pairs:
            tier = entry_tier(entry)
            st.tiers.add(tier)
            # Location marker alongside the cost tier: LAN entries are cost
            # tier "free" (own electricity, no per-token price), so only this
            # extra marker lets suggest_mode tell them from cloud freebies.
            if is_lan_entry(entry):
                st.tiers.add("lan")
                if online:
                    reachable_tiers.add("lan")
            st.entries.append(name)
            is_emb = _is_embeddings_entry(name, entry, config)
            st.has_embeddings = st.has_embeddings or is_emb
            if online:
                reachable_tiers.add(tier)
                embeddings_ok = embeddings_ok or is_emb

    for st in hosts.values():
        st.entries = sorted(set(st.entries))

    suggested, reason = suggest_mode(reachable_tiers, config.agent.model_mode)
    ordered = sorted(hosts.values(), key=lambda s: (s.online, s.host))
    return ProfileReport(
        hosts=ordered,
        reachable_tiers=reachable_tiers,
        suggested=suggested,
        reason=reason,
        embeddings_ok=embeddings_ok,
        embed_start_hint=bool(getattr(config.rag, "embed_server_command", "")),
    )


def _format_report(report: ProfileReport) -> str:
    lines = ["profile check:"]
    for st in report.hosts:
        mark = "ok " if st.online else "OFFLINE"
        names = ", ".join(st.entries[:4])
        if len(st.entries) > 4:
            names += f" +{len(st.entries) - 4} more"
        tiers = "/".join(sorted(st.tiers))
        lines.append(f"  [{mark:^7s}] {st.host} [{tiers}] — {names}")
    if not report.embeddings_ok:
        hint = " (start: agent embed --start)" if report.embed_start_hint else ""
        lines.append(
            "  ! no embeddings endpoint reachable — RAG index/search will fail "
            f"until an embeddings server is started{hint}"
        )
    lines.append(f"suggested profile: {report.suggested} ({report.reason})")
    return "\n".join(lines)


def maybe_start_embed_server(config: "Config", interactive: bool) -> None:
    """Offer to start the local embeddings server when none answered.

    Driven by ``config.rag.embed_server_autostart``: "ask" (prompt on a tty),
    "cpu"/"gpu" (start unattended on that device), "off" (warn only). Starting
    needs ``rag.embed_server_command`` — without it this only prints how to set
    it, since the agent does not bundle an inference server.
    """
    setting = (getattr(config.rag, "embed_server_autostart", "ask") or "ask").lower()
    if setting == "off":
        return
    if not (getattr(config.rag, "embed_server_command", "") or "").strip():
        print(
            "  ! no embeddings launcher configured — set rag.embed_server_command "
            'to a script that starts an OpenAI-compatible embeddings server on '
            "the embeddings entry's base_url (called with one argument, "
            '"cpu" or "gpu"), then: agent embed --start',
            flush=True,
        )
        return

    if setting in ("cpu", "gpu"):
        device = setting
    elif interactive and sys.stdin.isatty():
        default = (getattr(config.rag, "embed_server_device", "cpu") or "cpu").lower()
        try:
            raw = input(
                f"start local embeddings server? [{default}] (cpu|gpu|no): "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if raw in ("n", "no", "off"):
            return
        device = raw or default
        if device not in ("cpu", "gpu"):
            print(f"unknown device {device!r} — not starting", flush=True)
            return
    else:
        return  # non-interactive "ask": the warning above is all we do

    print(f"  starting embeddings server [{device}] — model load can take ~30s …",
          flush=True)
    try:
        from agent.rag import embed_server
        print("  " + embed_server.start(config, device), flush=True)
    except Exception as e:  # launcher problems must not block the session
        logger.warning("embeddings server start failed", exc_info=True)
        print(f"  embeddings server start failed: {e}", flush=True)


def run_startup_profile_check(config: "Config", interactive: bool) -> None:
    """Detect endpoint availability, print the report, pick a profile.

    ``config.agent.startup_profile``: "ask" (prompt when tty), "auto"
    (apply suggestion silently), "off" (skip entirely — including the
    embeddings offer below).

    When no embeddings endpoint answered, ``maybe_start_embed_server`` offers
    to bring the local one up before the profile prompt.
    """
    setting = getattr(config.agent, "startup_profile", "ask")
    if setting == "off":
        return
    try:
        report = detect(config)
    except Exception:
        logger.warning("startup profile check failed", exc_info=True)
        return

    print(_format_report(report), flush=True)

    if not report.embeddings_ok:
        maybe_start_embed_server(config, interactive)

    chosen = report.suggested
    if interactive and setting == "ask" and sys.stdin.isatty():
        try:
            raw = input(
                f"profile [{chosen}] (enter=accept, or "
                f"{'|'.join(_MODE_ORDER)}): "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            raw = ""
        if raw:
            if raw in MODE_TIERS:
                chosen = raw
            else:
                print(f"unknown profile {raw!r} — using {chosen}", flush=True)

    if chosen != config.agent.model_mode:
        config.agent.model_mode = chosen
        msg = f"model-mode set to {chosen} (session only; config default unchanged)"
        print(msg, flush=True)
        logger.info(msg)

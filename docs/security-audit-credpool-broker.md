# Adversarial audit: credpool + ultrasecure broker (S6)

Scope: `security/credpool.py` (encrypted credential vault + domain-gated
cookie injection) and `tools/ask_internet/main.py` (dual-LLM quarantine
broker). One section each; findings feed the harvest→quarantine→distill KB
flow so fixes become regression rules.

## Component 1 — credpool

### Finding C1 (HIGH) — cross-domain cookie leak on redirect — **FIXED**

The cookie was gated to the fetch URL's domain by `headers_for()` **once,
before the request**. Redirects are then followed *inside* the sandboxed
fetcher subprocess (`web_search/http_executor.py`), which had no credpool
awareness and reused the request headers verbatim on every hop. Result: an
open-redirect on a bound domain — or any hostile page served under it —
could 3xx to an attacker host and the fetcher would send the bound session
`Cookie` (and any `Authorization`) straight to it. The module docstring's
claim that "a page that redirects the fetch to an unbound host gets NO
cookie" was true only for the *initial* gate, false for the redirect path
that actually matters.

Fix (commit `e605663`): `headers_for()` returns the bound domain;
`fetch()` threads it as `cred_domain`; the worker drops
`Cookie`/`Authorization` on any hop whose host is not the bound domain or a
dot-boundary subdomain of it (`_creds_for_hop` / `_domain_matches`, mirroring
`credpool._domain_matches`). Absence of `cred_domain` leaves headers
untouched (backward-compatible). Regression tests cover bound/subdomain/
off-domain, case-insensitive header names, and the no-injection path.

**KB rule to distill**: "credential headers attached below the LLM must be
re-gated at every redirect hop, not only at request construction." Any future
transport that follows redirects (MCP http, a new fetcher backend) must
re-apply the domain gate.

### C2 (LOW, accepted) — domain-gate matching

`_domain_matches` is exact-or-dot-boundary-suffix: `evilnews.ycombinator.com`
and `...ycombinator.com.attacker.com` correctly do **not** match
`news.ycombinator.com` (tested). It does not consult a public-suffix list, so
a hypothetical account bound to a registrable-suffix-adjacent string is not
specially handled — not exploitable in practice (accounts are bound to real
service domains) and out of proportion to add a PSL dependency. Accepted.

### C3 (LOW, accepted) — key at rest

AES-256-GCM, per-record in one vault blob; key at
`<agent_dir>/credpool/credpool.key`, mode 0600, created with
`O_CREAT|O_TRUNC` + explicit 0600. Ciphertext also 0600. The key sits beside
the ciphertext, so credpool protects against *casual* disclosure (backups,
world-readable dirs, accidental commits) and against the cookie reaching the
LLM — not against an attacker who already has arbitrary read on the agent
dir. That is the documented threat model (same as integrity.key / notify
e2e). No change; noted so the boundary is explicit.

### C4 (informational) — lock/failure modes

`_load` swallows all decrypt errors and returns an empty pool
(`{"accounts": []}`). Correct fail-closed posture: a corrupt/locked vault
yields *no* credentials rather than crashing the fetch, so the agent falls
back to anonymous access. `headers_for` returns `({}, None, None)` whenever
`credpool.enabled` is false or no account matches — no cookie, no bound
domain, no partial state. Verified.

### C5 (informational) — Set-Cookie capture is domain-gated

`capture_cookies` re-selects the bound account by the *response* URL's domain
before persisting `Set-Cookie`, so a cross-domain response cannot write
cookies into an unrelated account's jar. With C1 fixed, the response URL the
caller passes is the original bound URL (redirects that left the domain no
longer carry the cookie), keeping capture consistent with injection.

## Component 2 — ultrasecure broker (`ask_internet`)

Architecture confirmed sound: in `mode == "ultrasecure"` the main agent has
`web_search`/`web_fetch` stripped and can only reach the network via
`ask_internet`, which spawns a disposable subagent holding **only**
`_INTERNET_TOOLS = {web_search, web_fetch}` (allowlist by set-subtraction
from all registered tools — a newly registered tool is excluded by default,
which is the safe direction). Fresh isolated history; air-gap refuses the
call; per-turn web rate-limit counters are isolated via `make_worker_limiter`.

### Finding B1 (MEDIUM) — no cap on laundered `quotes`/`sources` size

The quarantine boundary re-writes findings through the subagent LLM, but
`quotes` is explicitly verbatim attacker-controlled text and `sources` is
unvalidated strings; both cross to the privileged agent. `injection_scan.
guard_tool_output` banners the serialized result, but:
- there is **no length or count cap** on `quotes`/`sources`. A hostile page
  can induce a large `quotes` array that (a) bloats privileged context and
  (b) dilutes/segments an injection payload across many entries, lowering the
  odds the pattern scan flags it.
- `_coerce`'s fallback puts the subagent's *entire* output into `answer` when
  JSON parsing fails — still LLM-generated, but unbounded.

**Recommended hardening** (PLAN_NORMAL-class): cap `quotes` to N entries ×
M chars and `sources` to N entries, truncating with an explicit marker,
*before* `guard_tool_output`. Validate `sources` entries are http(s) URLs and
drop the rest. Keeps the scanner operating on bounded input and prevents
context-bloat as a cheap DoS/obfuscation lever.

### B2 (LOW, accepted) — quotes are attacker text by design

Even bounded, `quotes` crosses verbatim; that is inherent to the feature
(the privileged agent sometimes needs an exact quote). Mitigation is the
untrusted-data banner + the model's own instruction to treat sources/quotes
as data. This is the residual risk the whole quarantine design accepts;
B1's caps shrink it but don't remove it. Documented so it isn't rediscovered
as "new."

### B3 (informational) — no hooks / no permission rules on the quarantined side

Confirmed consistent with the S1 permission-model and S4 hook-trust specs:
the quarantined subagent's tool surface is fixed by the broker and must not
be widened or gated by repo-level config. `run_turn` is invoked with
`excluded_tools=excluded`; no hook or permission path runs for it. This is
the intended boundary; the two design docs already encode "quarantined side
consults neither."

### B4 (informational) — model routing for the quarantined LLM

`_quarantine_config` can pin the subagent to a dedicated `internet` role
model. If unset it falls back to the default endpoint — meaning the
quarantined and privileged agents may share a model/endpoint. That is a
*capability* isolation (fresh history, tool allowlist), not a *model*
isolation; the security property does not depend on a separate model, but an
operator wanting stronger separation should map `model_roles.internet` to a
distinct endpoint. Noted for the docs, no code change.

## Summary

| id | component | severity | status |
|----|-----------|----------|--------|
| C1 | credpool  | HIGH     | fixed (e605663) |
| C2 | credpool  | LOW      | accepted |
| C3 | credpool  | LOW      | accepted (threat model) |
| C4 | credpool  | info     | verified fail-closed |
| C5 | credpool  | info     | verified |
| B1 | broker    | MEDIUM   | hardening handed to PLAN_NORMAL |
| B2 | broker    | LOW      | accepted (inherent residual) |
| B3 | broker    | info     | consistent with S1/S4 |
| B4 | broker    | info     | operator guidance |

Distill into KB as regression rules: C1 (re-gate creds per redirect hop) and
B1 (bound + validate laundered fields before they cross the quarantine
boundary).

# REA via MCP (reverse engineering legacy artifacts)

[REA](https://github.com/morluto/rea) (MIT, Node) is a stdio MCP server for
inspecting artifacts without source: native binaries (incl. 16-bit DOS MZ/COM),
.NET assemblies, PE resources, Go/Rust build metadata, SQLite, JS/Electron
bundles, firmware. Plugs into `[mcp]` as-is; no agent code needed.

Use it when a legacy project ships binaries whose source is lost or untrusted.
For old *source* code use the index (`agent index`), not REA.

## Install

Node ≥22.19. Pinned, per machine (REA moves fast; tool names change between
releases — `inspect_with_objdump` is on REA `main`, not in 6.4.0):

```sh
npm install --prefix ~/.local/share/agent/mcp/rea --ignore-scripts rea-agents@6.4.0
mkdir -p ~/rea-targets        # binaries to analyse go here (jail sees it read-only)
```

Native pseudocode/xrefs need a deep-analysis provider. Ghidra 12.1.x + JDK 21
(Arch: `sudo pacman -S ghidra jdk21-openjdk`, installs to `/opt/ghidra`).
Without one, only static tools (`inspect_artifact`, managed/.NET, PE
resources, JS bundles) work.

## Config

Device layer (`~/.config/agent/agent.<host>.yaml`): install is per machine.
Full catalog = 170 tools, ~270 KB of model-facing schema (~68k tokens every
turn). `REA_MCP_INPUT_SCHEMA_PROFILE=compact` barely helps us — the bulk it
trims is `outputSchema`, which we never send. **Always set `tools_allow`.**
Measured (default profile, as registered): static list 19 tools ≈ 38 KB;
static + native 42 tools ≈ 64 KB (~16k tokens). `inspect_analysis_view` alone
≈ 10 KB. `analyze_function` left out: one dossier ≈ 45 KB of result;
`procedure_pseudo_code` ≈ 4–7 KB gives the code.

```yaml
mcp:
  enabled: true
  servers:
    - name: rea
      # node + script, not the npm `.bin/rea` symlink: the shim resolves
      # outside the bwrap binds and the jailed server dies on start.
      command: node
      args: ["~/.local/share/agent/mcp/rea/node_modules/rea-agents/scripts/rea.mjs", "mcp"]
      env:
        GHIDRA_INSTALL_DIR: /opt/ghidra
      sandbox: bwrap
      sandbox_network: false
      # /opt/ghidra: jail binds /usr but not /opt. /etc/java21-openjdk: Arch JDK
      # conf/ symlinks there; without it JVM dies "Error loading java.security file".
      sandbox_ro: ["~/.local/share/agent/mcp/rea", "~/rea-targets", "{project}",
                   "/opt/ghidra", "/etc/java21-openjdk"]
      sandbox_rw: ["{project}/.rea-out"]   # export_evidence_bundle target
      init_timeout_s: 60
      call_timeout_s: 600                  # Ghidra auto-analysis on first open
      tools_allow:
        # static — no provider needed
        - analyze_javascript_application
        - analyze_web_bundle
        - recover_javascript_sources
        - trace_web_module_imports
        - compare_bundles
        - inspect_artifact
        - extract_artifact
        - inspect_binary_layout
        - inspect_macho
        - inspect_signature
        - inspect_plist
        - inspect_managed_artifact
        - inspect_managed_members
        - inspect_evm_interface
        - inspect_web_network_capture
        - demangle_swift
        - inspect_analysis_view
        - export_evidence_bundle
        - inspect_pe_resources
        # native — Ghidra
        - open_binary
        - close_binary
        - binary_session
        - binary_overview
        - inspect_native_load_image
        - list_procedures
        - list_strings
        - list_segments
        - list_names
        - search_procedures
        - search_strings
        - procedure_info
        - procedure_pseudo_code
        - procedure_assembly
        - read_function_instructions
        - procedure_callers
        - procedure_callees
        - xrefs
        - resolve_containing_procedure
        - get_call_graph
        - inspect_native_api
        - inspect_native_data_type
        - inspect_managed_native_boundaries
      tools_deny: ["capture_*", "observe_*"]   # these execute the target
```

Without Ghidra drop `env`, the last two `sandbox_ro` entries and the native list.

## Usage

Session-based: `open_binary {path, provider_id: ghidra}` first; native tools
then act on the open binary (`procedure: FUN_00401023`, `entry`, …). First
Ghidra call per server process ≈ 7–11 s (auto-analysis). Results carry
evidence ids and SHA-256 digests on every step — verbose; expect several KB per
call (`inspect_artifact` on a tiny ELF ≈ 6 KB).

DOS specifics (16-bit, Borland, seed files): REA `docs/ghidra-dos.md`,
`REA_GHIDRA_LANGUAGE_ID` / `REA_GHIDRA_COMPILER_SPEC_ID` env.

## Indexing a binary (`analyze_binary`)

The chat can drive REA tools directly, but on anything beyond a toy that burns
tokens per function. `analyze_binary(path)` does the bulk pass offline-style:
REA/Ghidra facts (procedures, callees, imports, strings, pseudocode) → one LLM
description per function, callees first → groups along the call graph → one
root summary. Stored in the asm index (`.agent/` db), searchable with `search`,
navigable with `asm_tree`. Needs `asm_analysis.enabled: true` and the REA server
running (`asm_analysis.rea_server`, default `rea`).

- Budget: `max_functions=N` per call; re-run continues. Same sha256 → cached.
- Output budget per function: `token_limits.asm_function` (default 1024; the
  WndProc of moveme.exe needed 735). JSON keys are ordered short-first, so a
  reply cut at the limit keeps purpose/name/apis/confidence and only the tail
  of `behaviour` is lost; cut earlier → one retry at double budget.
- `annotate=true` writes names + comments back into the open Ghidra program;
  add `annotate_native_function` to `tools_allow` (refused otherwise, reported
  as `annotate_error`).
- Callback functions referenced only as pointers (WndProc, thread procs) are
  not in Ghidra's callee lists; they stay loners until grouping packs them.
- Workflow for legacy rewrite: `load_skill legacy-re`.

## Verified (REA 6.4.0, Ghidra 12.1.2, JDK 21, 2026-10-13)

Through `load_config()` + `security.policy` + `agent.mcp.manager`, bwrap:

- 16-bit DOS MZ (REA fixture `tests/conformance/ghidra/dos-mz-fixture.mjs`):
  `dos-mz/x86`, Ghidra `MzLoader`, `x86:LE:16:Real Mode`, pseudocode with far
  calls and `swi(0x21)`.
- PE32 MASM32 tray app (github.com/adderek/moveme, 78 KB, mostly icons): 4
  internal procedures + 28 imports; `procedure_pseudo_code` of WinMain /
  WndProc / thread matches the `.asm` 1:1 (`Sleep(0x38a4)` = 14500 ms,
  `WM_USER+5/+6`, icon ids 500/501, `SendInput` ±1 px). `inspect_pe_resources`
  lists icon groups.
- static-only block (no Ghidra) on a second host: 19 tools register,
  `inspect_pe_resources` works.

Not yet: `agent run`/`chat` loop with a model driving these tools (schemas use
`$ref`/`$defs`), .NET target.

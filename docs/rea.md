# REA via MCP (reverse engineering legacy artifacts)

[REA](https://github.com/morluto/rea) (MIT, Node) is a stdio MCP server for
inspecting artifacts without source: native binaries (incl. 16-bit DOS MZ/COM),
.NET assemblies, PE resources, Go/Rust build metadata, SQLite, JS/Electron
bundles, firmware. Plugs into `[mcp]` as-is; no agent code needed.

Use it when a legacy project ships binaries whose source is lost or untrusted.
For old *source* code use the index (`agent index`), not REA.

## Install

Node ≥22.19. User-global, pinned (REA moves fast; tool names change between
releases — `inspect_with_objdump` is on REA `main`, not in 6.4.0):

```sh
npm install -g --ignore-scripts rea-agents@6.4.0
```

Native pseudocode/xrefs need a deep-analysis provider. Ghidra 12.1.x + JDK 21
(Arch: `sudo pacman -S ghidra jdk21-openjdk`, installs to `/opt/ghidra`).
Without one, only static tools (`open_binary`, `inspect_artifact`, managed/.NET,
PE resources) work.

## Config

Full catalog = 170 tools, ~270 KB of model-facing schema (~68k tokens every
turn). `REA_MCP_INPUT_SCHEMA_PROFILE=compact` barely helps us — the bulk it
trims is `outputSchema`, which we never send. **Always set `tools_allow`.**
The list below = 27 tools, ~30 KB (~7.5k tokens). `analyze_function` left out
on purpose: one dossier ≈ 45 KB; `procedure_pseudo_code` ≈ 4 KB gives the code.

```yaml
mcp:
  enabled: true
  servers:
    - name: rea
      # node + script, not the npm `bin/rea` symlink: the shim lives outside
      # the bwrap binds and the jailed server dies on start.
      command: /usr/bin/node
      args: [~/.local/share/npm/lib/node_modules/rea-agents/scripts/rea.mjs, mcp]
      env:
        GHIDRA_INSTALL_DIR: /opt/ghidra
      init_timeout_s: 60
      call_timeout_s: 600          # Ghidra auto-analysis is slow
      sandbox: bwrap
      sandbox_ro:
        - ~/.local/share/npm/lib/node_modules/rea-agents
        - /opt/ghidra              # jail binds /usr but not /opt
        - /etc/java21-openjdk      # Arch JDK conf/ symlinks here; else JVM dies: "Error loading java.security file"
        - /path/to/legacy/binaries # targets; open_binary wants absolute paths
      tools_allow: [open_binary, close_binary, binary_session, binary_overview,
        inspect_artifact, inspect_native_load_image, list_procedures, list_strings,
        list_segments, list_names, search_procedures, search_strings,
        procedure_info, procedure_pseudo_code, procedure_assembly,
        read_function_instructions, procedure_callers, procedure_callees, xrefs,
        resolve_containing_procedure, get_call_graph,
        inspect_native_api, inspect_native_data_type, inspect_managed_artifact,
        inspect_managed_members, inspect_managed_native_boundaries,
        inspect_pe_resources]
```

## Usage

Session-based: `open_binary {path}` first, then the rest act on the open
binary. `provider_id: "ghidra"` on `open_binary` forces Ghidra. Results carry
evidence ids and SHA-256 digests on every step — verbose; expect several KB per
call (`inspect_artifact` on a tiny ELF ≈ 6 KB).

DOS specifics (16-bit, Borland, seed files): REA `docs/ghidra-dos.md`,
`REA_GHIDRA_LANGUAGE_ID` / `REA_GHIDRA_COMPILER_SPEC_ID` env.

## Verified (REA 6.4.0, Ghidra 12.1.2, JDK 21, 2026-10-13)

Through `agent.mcp.manager`, unjailed and `bwrap`, with the config above:

- handshake, `tools/list`, `tools_allow` filtering
- ELF: `open_binary`, `inspect_artifact`
- 16-bit DOS MZ (REA fixture `tests/conformance/ghidra/dos-mz-fixture.mjs`):
  `open_binary` → `dos-mz/x86`, `list_procedures` (Ghidra `MzLoader`,
  `x86:LE:16:Real Mode`, ~7 s cold), `procedure_pseudo_code {procedure: entry}`
  → C with far calls and `swi(0x21)`

Not yet: chat loop with a local model (schemas use `$ref`/`$defs`), .NET target.

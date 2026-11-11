---
description: Analyze a legacy binary without source (REA + asm index), extract its logic as specs and characterization tests, rewrite module by module
---

# Legacy reverse engineering

Goal: legacy app nobody dares to touch → documented logic → tested rewrite on a
supported stack. Output is files in the repo, not chat: every finding written
down as you go (context gets compacted, sessions die).

## Rules
- Facts beat descriptions. Ghidra facts (calls, imports, signatures, strings) are exact; LLM descriptions in the index are hypotheses. Every rule you write cites its address (`0x401023`).
- `confidence: low` or `UNVERIFIED claims` in asm_tree → read the pseudocode yourself before relying on it.
- Never run the target on the dev host (`capture_*`/`observe_*` denied). Execution only on a lab host (`remote_exec`) or emulator.
- Token order: `asm_tree` → `search` → REA `procedure_pseudo_code` for a few functions. Don't dump `list_procedures`/`list_strings` of a big binary into chat.

## 1. Triage
- Copy binary into REA jail (`~/rea-targets` or project). Record sha256, version, where it runs, who depends on it → `docs/re/<app>/README.md`.
- `inspect_artifact`: format, compiler, packer, .NET? .NET/Java → managed tools, decompiled code is near-source; skip the native pass. Packed → unpack first (lab host).

## 2. Index
- `analyze_binary(path, max_functions=200)`; repeat until no `pending`. Report `unverified` count.
- Changed binary (new version) = new analysis; keep the old db for diffing behaviour.

## 3. Map
- `asm_tree(path)` → root purpose + subsystems. Write `docs/re/<app>/overview.md`: purpose, one line per subsystem with unit id, external interfaces.
- External surface from imports + strings: files/paths, registry, network, DB/SQL, IPC, printers, COM ports, data formats. This is the compatibility contract of the rewrite.

## 4. Drill (only where the goal needs it)
- `asm_tree(unit_id)` down to functions. Business rules live in the middle layers: validation, calculations, state machines, format parsers. Skip UI/runtime/library plumbing.
- For each chosen function: pseudocode → rename/comment in Ghidra (`annotate_native_function`) when sure, so callers' pseudocode improves.

## 5. Spec per module — `docs/re/<app>/<module>.md`
Purpose · Inputs (types, ranges, encodings) · Outputs · Rules (each with constants and `addr`) · Side effects · Errors/edge cases · State/globals · Open questions.
Note suspected bugs and vulnerabilities (overflows, hardcoded credentials, weak crypto) separately: for each, decide keep-for-compat or fix, and say which.

## 6. Characterization tests
- Golden I/O from the running legacy app (lab host / emulator): inputs, outputs, files written. Cover the rules from step 5, boundary values, error paths.
- These are the acceptance tests of the rewrite; store under `tests/regression/<app>/`.

## 7. Rewrite
- Strangler: one module at a time behind the same external interface; legacy keeps running for the rest.
- New code from the spec + golden tests; traceability table rule → `addr` → new function in the module doc.
- Done for a module = golden tests pass on the new code, open questions closed or explicitly accepted.

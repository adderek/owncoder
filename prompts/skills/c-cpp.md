---
description: C/C++ work (e.g. llama.cpp/ggml) — CMake builds, warnings, sanitizers, UB pitfalls, debugging crashes
---

# C / C++
- Read repo `AGENTS.md`/`CONTRIBUTING.md` first. A workspace may hold several source trees and many `build-*` dirs — confirm which tree + build dir the task targets before editing or building.
- Style (llama.cpp): C++17 in `src/`/`common/`, C11 in `ggml/`; 4-space indent. Match surrounding code, don't modernize it.
- Build out of tree: `cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j`. Reuse an existing `build*/` dir and its flags (`build/CMakeCache.txt`) instead of reconfiguring.
- GPU backends (`-DGGML_HIP=ON`, `-DGGML_CUDA=ON`, `-DGGML_VULKAN=ON`) build slowly — reuse the matching existing build dir; reproduce on CPU first when the bug isn't backend-specific.
- Narrow rebuilds: `cmake --build build --target <name> -j`. Tests: `ctest --test-dir build -R <regex> --output-on-failure`.
- Warnings on: `-Wall -Wextra` (project may set more). New warnings in your diff = fix them.
- Crashes / memory bugs: rebuild with `-DCMAKE_C_FLAGS=-fsanitize=address,undefined -DCMAKE_CXX_FLAGS=-fsanitize=address,undefined` (llama.cpp: `-DLLAMA_SANITIZE_ADDRESS=ON -DGGML_SANITIZE_ADDRESS=ON`; also `_UNDEFINED`, `_THREAD`), rerun repro. Then `gdb -batch -ex run -ex bt --args <cmd>`.
- UB checklist: signed overflow, shifts ≥ width, out-of-bounds, uninitialized reads, strict aliasing (use `memcpy` to type-pun), misaligned access, use-after-free, `size_t` vs `int` mixing in loops/comparisons.
- Sizes: `size_t`/`int64_t` for element counts and byte offsets (tensors exceed 2^31). Check multiplications for overflow before `malloc`.
- Ownership: who frees what — follow the existing `*_free` / RAII pattern; every early return releases resources.
- C API boundaries: no exceptions or STL types across `extern "C"`; keep ABI (struct layout, enum values) unless told to break it.
- Headers: include what you use; no `using namespace` in headers.
- Performance claims need a measurement (same build type, same input, several runs; e.g. `llama-bench`). Debug builds are not for timing.
- Verify: build clean (no new warnings), run relevant tests, run the actual binary once on a small model/input.

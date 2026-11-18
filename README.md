# Crucible

**An open-source DXIL → Metal Shading Language compiler.**

Crucible (`cruciblec`) smelts DirectX Intermediate Language (`.dxil` /
DXContainer) shader binaries down into clean, idiomatic **MSL source code**
plus JSON binding reflection, then compiles them with Apple's native toolchain
(`xcrun metal` / `metallib`) for execution on Apple Silicon GPUs.

## Why

Apple's proprietary `metal-shaderconverter` is closed-source, lags behind
modern Shader Model 6.8/6.9 intrinsics, and forces a heavyweight Tier-2
argument-buffer ABI that inflates register pressure. Crucible instead emits
zero-overhead direct `[[buffer(n)]]` bindings and human-readable MSL you can
diff, profile, and tune.

```
HLSL --dxc--> DXIL --cruciblec--> .metal + .json --xcrun metal--> .metallib
```

Example — this HLSL:

```hlsl
RWStructuredBuffer<float> Out : register(u0);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  Out[id.x] = Out[id.x] * 2.0f + 1.0f;
}
```

becomes exactly this MSL (no IR dump, no temp soup):

```cpp
#include <metal_stdlib>
using namespace metal;

kernel void main0(
    device float* uav0 [[buffer(0)]],
    uint3 dtid [[thread_position_in_grid]]) {
  uav0[dtid.x] = uav0[dtid.x] * 2.0f + 1.0f;
}
```

## Architecture

| Stage | Component | What it does |
|---|---|---|
| 1 | `lib/Frontend/ContainerLoader` | `llvm::object::DXContainer` part extraction (DXIL, PSV0, signatures); bare bitcode passthrough |
| 2 | `lib/Frontend/DXILUpgrader` | `llvm::parseBitcodeFile` auto-upgrades frozen LLVM 3.7 bitcode (incl. data-layout fix via `ParserCallbacks`); `DXILUpgradePass` + mem2reg/SROA/SimplifyCFG cleanup; `!dx.*` metadata reflection |
| 3 | `lib/Transforms/ResourceLowering` | Deterministic `(class, space, register) → [[buffer(n)]]` direct binding assignment |
| 4 | `lib/Transforms/CFGStructurizer` | Imports the module into MLIR and lifts the unstructured CFG to `scf.if`/`scf.while` regions via `LiftControlFlowToSCF` (goto-free by construction; value yields double as PHI demotion) |
| 5 | `lib/Backend/MSLEmitter` | Structured SCF regions → MSL printer: if/else, do-while/while loops, use-count-based expression inlining, signedness recovery |
| 6 | `lib/Backend/IntrinsicMap` | `@dx.op.*` → `metal_stdlib` mapping table |
| 7 | `tools/cruciblec` | CLI driver + `xcrun metal`/`metallib` toolchain wrapper (`--metallib`) |
| — | `tools/crucible-run` | Metal compute dispatch harness for GPU-verified lit tests |

Design guardrails:

1. **Never emit Apple `.air` bitcode directly** — always human-readable MSL
   compiled through the supported Apple toolchain.
2. **Reuse upstream LLVM** for container & bitcode ingestion — no custom
   binary parsers.
3. **Every codegen feature ships with a lit test** that both FileChecks the
   MSL text *and* compiles it with `xcrun metal -Werror`, plus a GPU
   execution test asserting exact buffer contents.

## Building

Requirements: CMake ≥ 3.24, Ninja, LLVM ≥ 16 (Homebrew `llvm` works), `dxc`,
Xcode command line tools, and `lit` (`pip install lit`).

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
ninja -C build check-crucible   # lit/FileCheck + GPU tests
```

Useful configure options:

- `-DLLVM_DIR=/path/to/lib/cmake/llvm` — explicit LLVM
- `-DDXC_EXECUTABLE=/path/to/dxc` — explicit DXC for tests
- `-DLLVM_EXTERNAL_LIT=/path/to/lit` — explicit lit (binary or `lit.py`)

## Usage

```sh
cruciblec input.dxil -o out.metal --reflect out.json --metallib out.metallib
crucible-run --lib out.metallib --entry main0 --threads 64,1,1 \
             --groups 1,1,1 --buffer 0:f32:64:iota
```

The reflection JSON carries the exact `(space, register, kind) → metalIndex`
mapping, thread-group size, and the (possibly sanitized) MSL kernel name.

## Testing & compliance

All testing is lit/FileCheck-driven (`test/FileCheck` for golden MSL +
`-Werror` Apple-compile checks, `test/E2E` for GPU-verified dispatches). CI
records every run's lit JSON with `scripts/record_offload_compliance.py` and
publishes the pass-rate dashboard in `pages/` to GitHub Pages — the same
pipeline that powers [Nacelle's compliance site](https://farzon.org/Nacelle/).
The recorder is format-compatible with
[llvm/offload-test-suite](https://github.com/llvm/offload-test-suite) lit
output, which becomes the compliance target in Milestone 4.

## Third-party foundation

Crucible deliberately builds on best-in-class open source instead of
reinventing it:

- **[LLVM + MLIR](https://llvm.org)** — DXContainer object parsing, bitcode
  auto-upgrade (LLVM 3.7 → modern opaque-pointer IR), pass infrastructure,
  `llvm::json`, `Error`/`Expected` diagnostics, CommandLine; MLIR's
  `LiftControlFlowToSCF` (the "perfect reconstructability" algorithm)
  structurizes arbitrary shader CFGs instead of a hand-rolled structurizer.
- **[DirectXShaderCompiler (dxc)](https://github.com/microsoft/DirectXShaderCompiler)**
  — reference HLSL → DXIL compiler used by the test suite.
- **[LLVM lit + FileCheck](https://llvm.org/docs/CommandGuide/lit.html)** —
  the industry-standard compiler testing harness.
- **[Nacelle](https://github.com/farzonl/Nacelle)** — DXIL loader patterns,
  lit conventions, benchmark harness, and the compliance dashboard were
  lifted from our DXIL software rasterizer.

## Roadmap

- [x] **Milestone 1** — DXContainer loader, compute kernels, structured
  buffers, direct bindings, reflection JSON, `xcrun` integration, GPU-verified
  lit suite.
- [x] **Milestone 2** — CFG structurization via MLIR `ControlFlowToSCF`
  (`if`/`else`, do-while/while loops, PHI demotion), cbuffers (incl. mixed
  float/uint rows), raw buffer ops, signedness recovery.
- [ ] **Milestone 3** — groupshared + barriers, atomics, wave ops
  (`simd_*`), SM 6.8 packed/vector dot products, benchmarks vs.
  `metal-shaderconverter`.
- [ ] **Milestone 4** — offload-test-suite Metal integration and public
  compliance dashboard; vertex/fragment stages.

## License

Apache-2.0 WITH LLVM-exception.

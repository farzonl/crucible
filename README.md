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
`-Werror` Apple-compile checks, `test/E2E` for GPU-verified dispatches).

### offload-test-suite integration

Crucible runs [llvm/offload-test-suite](https://github.com/llvm/offload-test-suite)
natively: `tools/crucible-offload` is a drop-in `offloader` replacement that
reuses the suite's YAML pipeline parser and ULP/epsilon verification rules,
but compiles the DXIL with crucible and dispatches on Metal with
reflection-driven direct bindings.

**Crucible advertises itself as a DirectX implementation** — the MoltenVK
model: identity is the API contract you implement (DXIL + D3D12 binding
semantics), not the backend you use (Metal). The lit feature set is therefore
`DirectX + Crucible` (driver tag) rather than `Metal`, so crucible is held to
the D3D12 contract instead of inheriting metal-shaderconverter's waivers.

```sh
cmake -S . -B build -G Ninja \
  -DCRUCIBLE_OFFLOAD_SUITE_DIR=/path/to/offload-test-suite   # auto-detected
ninja -C build check-crucible-offload
```

Pass `--validation-layer` to `crucible-offload` to enable Metal API Validation.
The offload lit harness uses `crucible-api-query` for host information when
the suite checkout includes its `HostInfo` support code.

Baseline at Milestone 2: **110 passed / 219 failed / 289 unsupported** of 619
discovered tests (graphics, RT, wave ops, 16/64-bit types, and textures are
still unimplemented). CI records every run with
`scripts/record_offload_compliance.py` and publishes the dashboard in
`pages/` to GitHub Pages — the same pipeline that powers
[Nacelle's compliance site](https://farzon.org/Nacelle/).

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

## Benchmarks

`scripts/bench.py` benchmarks crucible against Apple's metal-shaderconverter
on the same DXIL: both convert identical dxc output, dispatch through the same
minimal native runner (`crucible-run --bench-json`, GPU timestamps), and the
results publish to the `pages/bench.html` trend dashboard (ported from
Nacelle). MSC's argument-buffer ABI is honored via root-descriptor GPU
addresses at its fixed bind point — no IRConverter SDK required at runtime.

First results on Apple Silicon (median GPU ms, lower is better):

| Workload | crucible | metal-shaderconverter |
|---|---|---|
| Streaming FMA (memory-bound) | 1.59 | 1.59 |
| Mandelbrot loop (ALU-bound) | 1.54 | **1.16** |
| Groupshared reduction (barrier-bound) | **1.26** | 1.64 |
| Wave prefix ops (simd-bound) | 1.55 | 1.54 |

Direct bindings already win on binding-heavy workloads; loop codegen quality
is the current optimization target.

## Roadmap

- [x] **Milestone 1** — DXContainer loader, compute kernels, structured
  buffers, direct bindings, reflection JSON, `xcrun` integration, GPU-verified
  lit suite.
- [x] **Milestone 2** — CFG structurization via MLIR `ControlFlowToSCF`
  (`if`/`else`, do-while/while loops, PHI demotion), cbuffers (incl. mixed
  float/uint rows), raw buffer ops, signedness recovery.
- [x] **Milestone 3** — groupshared + barriers, atomics, wave ops
  (`simd_*`), SM 6.4/6.8 packed dot products, benchmarks vs.
  `metal-shaderconverter`.
- [x] **Milestone 4 (partial)** — offload-test-suite integration
  (`crucible-offload` + `check-crucible-offload`) and public compliance
  dashboard. Scalar `half`, `i16`, and `i64` are enabled in compliance tests.
  Typed buffer loads and stores preserve DXIL's packed 32-bit word layout
  for scalar and two-component 64-bit integers.
  Long-vector integer `abs` and integer/float `dot` use LLVM's Scalarizer
  rather than relying on Metal vector types wider than four lanes.
  Wave and quad operations use Metal SIMD builtins. The backend uses
  ballot and shuffle for 64-bit arithmetic reductions/scans and for
  16-bit integer sums that need to preserve the active-lane mask.
  32-bit resource atomics cover raw, structured, and typed UAV buffers
  and integer `RWTexture2D`. When binding a texture whose reflection has
  `requiresTextureAtomics` set, use `MTLTextureUsageShaderAtomic` and an
  `R32Uint` or `R32Sint` pixel format.
  Remaining: broader texture and sampler support, resource arrays, 64-bit
  atomics, and fixed wave-size selection.
- [x] **Graphics (initial)** — DXIL vertex/pixel signatures lower to Metal
  attributes, interpolated varyings (including array and matrix varyings),
  position, primitive ID, clip-distance outputs, and color and depth outputs.
  The offload runner executes single-draw, single-color-target raster
  pipelines with buffer bindings, fragment discard, and derivatives. It
  uses a transient depth attachment for shaders that write depth.
  Graphics textures and samplers, stencil, multisampling, cull distance,
  geometry, and tessellation remain unsupported.

## License

Apache-2.0 WITH LLVM-exception.

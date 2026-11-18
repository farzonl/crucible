#!/usr/bin/env python3
"""Crucible benchmark harness: crucible vs. metal-shaderconverter.

Compiles a set of HLSL compute workloads to DXIL with dxc, converts them to
Metal libraries with both crucible (DXIL -> MSL source -> metallib) and
Apple's metal-shaderconverter (DXIL -> AIR metallib), dispatches both through
the same minimal native runner (crucible-run), and emits Nacelle-compatible
benchmark JSON rows.

Only the GPU dispatch lives in native code; everything else is orchestrated
here.
"""

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Workload:
    id: str
    name: str
    source: str  # file in bench/shaders/
    threads: str
    groups: str
    # crucible-run --buffer specs. Buffer indices follow crucible's direct
    # binding order (SRVs, then UAVs, then CBVs).
    buffers: list = field(default_factory=list)
    # Root-parameter order for the metal-shaderconverter top-level argument
    # buffer, as crucible-run buffer indices (matches the [RootSignature]
    # embedded in the HLSL).
    ir_root_order: str = "0"


WORKLOADS = [
    Workload(
        id="bench_vecfma",
        name="Streaming FMA (memory-bound)",
        source="vecfma.hlsl",
        threads="256,1,1",
        groups="32768,1,1",  # 8M floats
        buffers=["0:f32:8388608:iota"],
        ir_root_order="0",
    ),
    Workload(
        id="bench_mandelbrot",
        name="Mandelbrot escape loop (ALU-bound)",
        source="mandelbrot.hlsl",
        threads="64,1,1",
        groups="65536,1,1",  # 2048x2048
        buffers=["0:f32:4194304:zero"],
        ir_root_order="0",
    ),
    Workload(
        id="bench_reduce",
        name="Groupshared tree reduction (barrier-bound)",
        source="reduce.hlsl",
        threads="256,1,1",
        groups="32768,1,1",  # 8M floats
        buffers=["1:f32:8388608:iota", "0:u32:256:zero"],
        ir_root_order="1,0",  # SRV t0 then UAV u0
    ),
    Workload(
        id="bench_waveprefix",
        name="Wave prefix/reduction ops (simd-bound)",
        source="waveprefix.hlsl",
        threads="256,1,1",
        groups="32768,1,1",  # 8M uints
        buffers=["0:u32:8388608:iota"],
        ir_root_order="0",
    ),
]


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def timed(cmd):
    t0 = time.perf_counter()
    run(cmd)
    return (time.perf_counter() - t0) * 1000.0


def bench_backend(runner, lib, entry, w, iters, warmup, ir_ab=None):
    cmd = [str(runner), "--lib", str(lib), "--entry", entry,
           "--threads", w.threads, "--groups", w.groups,
           "--iters", str(iters), "--warmup", str(warmup), "--bench-json"]
    for b in w.buffers:
        cmd += ["--buffer", b]
    if ir_ab:
        cmd += ["--ir-ab", ir_ab]
    out = run(cmd).stdout
    return json.loads(out)


def make_row(w, backend, shader_format, device):
    return {
        "benchId": w.id,
        "benchName": w.name,
        "category": "Compute",
        "backend": f"{backend}: {device}",
        "shaderFormat": shader_format,
        "simdWidth": 32,
        "ran": False,
        "compileMs": 0.0,
        "firstDispatchMs": 0.0,
        "coldMs": 0.0,
        "meanMs": 0.0,
        "medianMs": 0.0,
        "minMs": 0.0,
        "status": "SKIPPED",
    }


def finalize(row, compile_ms, timing):
    samples = timing["samplesMs"]
    row["ran"] = True
    row["status"] = "OK"
    row["compileMs"] = round(compile_ms, 3)
    row["firstDispatchMs"] = round(timing["firstDispatchMs"], 3)
    row["coldMs"] = round(compile_ms + timing["firstDispatchMs"], 3)
    row["meanMs"] = round(statistics.fmean(samples), 4)
    row["medianMs"] = round(statistics.median(samples), 4)
    row["minMs"] = round(min(samples), 4)


def main():
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-dir", type=Path, default=repo / "build")
    ap.add_argument("--shader-dir", type=Path, default=repo / "bench" / "shaders")
    ap.add_argument("--dxc", default=shutil.which("dxc"))
    ap.add_argument("--msc", default=shutil.which("metal-shaderconverter"))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("-o", "--out", type=Path, default=Path("bench-results.json"))
    args = ap.parse_args()

    cruciblec = args.build_dir / "tools" / "cruciblec" / "cruciblec"
    runner = args.build_dir / "tools" / "crucible-run" / "crucible-run"
    for tool, name in [(cruciblec, "cruciblec"), (runner, "crucible-run")]:
        if not tool.exists():
            sys.exit(f"bench: {name} not found at {tool}; build first")
    if not args.dxc:
        sys.exit("bench: dxc not found")
    if not args.msc:
        print("bench: metal-shaderconverter not found; recording crucible only",
              file=sys.stderr)

    device = run(["system_profiler", "SPDisplaysDataType", "-json"])
    try:
        device = json.loads(device.stdout)["SPDisplaysDataType"][0][
            "sppci_model"]
    except Exception:
        device = "Unknown GPU"

    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for w in WORKLOADS:
            src = args.shader_dir / w.source
            dxil = tmp / (w.id + ".dxil")
            run([args.dxc, "-T", "cs_6_0", "-E", "main", "-O3",
                 "-Fo", str(dxil), str(src)])

            # --- crucible: DXIL -> MSL -> metallib (xcrun) ---
            row = make_row(w, "Metal (crucible)", "DXIL->MSL", device)
            try:
                metal = tmp / (w.id + ".metal")
                lib = tmp / (w.id + ".crucible.metallib")
                compile_ms = timed([str(cruciblec), str(dxil), "-o", str(metal),
                                    "--metallib", str(lib)])
                timing = bench_backend(runner, lib, "main0", w,
                                       args.iters, args.warmup)
                finalize(row, compile_ms, timing)
            except subprocess.CalledProcessError as e:
                row["status"] = f"FAILED ({e.stderr.strip().splitlines()[-1] if e.stderr else e})"
            rows.append(row)
            print(f"{w.id:20s} crucible    {row['status']:8s} "
                  f"median={row['medianMs']}ms compile={row['compileMs']}ms")

            # --- metal-shaderconverter: DXIL -> AIR metallib ---
            row = make_row(w, "Metal (IRConverter)", "DXIL->AIR", device)
            if args.msc:
                try:
                    lib = tmp / (w.id + ".msc.metallib")
                    compile_ms = timed([args.msc, str(dxil), "-o", str(lib)])
                    timing = bench_backend(runner, lib, "main", w,
                                           args.iters, args.warmup,
                                           ir_ab=w.ir_root_order)
                    finalize(row, compile_ms, timing)
                except subprocess.CalledProcessError as e:
                    row["status"] = f"FAILED ({e.stderr.strip().splitlines()[-1] if e.stderr else e})"
            rows.append(row)
            print(f"{w.id:20s} IRConverter {row['status']:8s} "
                  f"median={row['medianMs']}ms compile={row['compileMs']}ms")

    args.out.write_text(json.dumps({"version": 1, "results": rows}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

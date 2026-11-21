#!/usr/bin/env python3
"""Crucible benchmark harness: Crucible vs. Apple's metal-shaderconverter.

Compiles the Nacelle benchmark shader suite (bench/nacelle/shaders/) to DXIL
with dxc (caching compiled .dxil files in build/bench-dxil/), then invokes
the in-process native runner (crucible-bench) to run both:
  - Metal (crucible):    DXIL -> MSL (libcrucible) -> MTLDevice newLibraryWithSource
  - Metal (IRConverter): DXIL -> AIR (libmetalirconverter) -> MTLDevice newLibraryWithData

Emits Nacelle-compatible benchmark JSON for pages/bench.html.
"""

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
from pathlib import Path

# (hlsl_file, profile, entry_or_None, output_dxil)
DXIL_TARGETS = [
    # Compute
    ("bench_raymarch.hlsl", "cs_6_0", "main", "bench_raymarch.dxil"),
    ("matmul.hlsl", "cs_6_0", "CSMain", "matmul.dxil"),
    ("tiled_matmul.hlsl", "cs_6_0", "CSMain", "tiled_matmul.dxil"),
    ("prefix_sum_groupshared.hlsl", "cs_6_0", "CSMain", "prefix_sum_groupshared.dxil"),
    ("divergent_cf_atomics.hlsl", "cs_6_0", "CSMain", "divergent_cf_atomics.dxil"),
    ("wave_intrinsics.hlsl", "cs_6_0", "CSMain", "wave_intrinsics.dxil"),
    ("GDeflate.hlsl", "cs_6_0", "CSMain", "gdeflate.dxil"),
    ("ETHashMine.hlsl", "cs_6_0", "main", "ethash_mine.dxil"),
    ("bench_coalesced_uniform.hlsl", "cs_6_0", "CSMain", "bench_coalesced_uniform.dxil"),
    # Vertex
    ("bench_vertex_stress.hlsl", "vs_6_0", "VSMain", "bench_vertex_stress_vs.dxil"),
    ("bench_vertex_stress.hlsl", "ps_6_0", "PSMain", "bench_vertex_stress_ps.dxil"),
    ("bench_small_tri_ps.hlsl", "vs_6_0", "VSMain", "bench_small_tri_ps_vs.dxil"),
    ("bench_small_tri_ps.hlsl", "ps_6_0", "PSMain", "bench_small_tri_ps_ps.dxil"),
    ("bench_vertex_mesh.hlsl", "vs_6_0", "VSMain", "bench_vertex_mesh_vs.dxil"),
    ("bench_vertex_mesh.hlsl", "ps_6_0", "PSMain", "bench_vertex_mesh_ps.dxil"),
    # Pixel
    ("fullscreen_quad.hlsl", "vs_6_0", "VSMain", "fullscreen_quad_vs.dxil"),
    ("light_cycle.hlsl", "ps_6_0", "PSMain", "light_cycle.dxil"),
    ("aurora_paint.hlsl", "ps_6_0", "PSMain", "aurora_paint.dxil"),
    ("sunflower_fields.hlsl", "ps_6_0", "PSMain", "sunflower_fields.dxil"),
    ("mt_fuji_watercolor.hlsl", "ps_6_0", "PSMain", "mt_fuji_watercolor.dxil"),
    ("disco_maze.hlsl", "ps_6_0", "PSMain", "disco_maze.dxil"),
    ("aetherial_cathedral.hlsl", "ps_6_0", "PSMain", "aetherial_cathedral.dxil"),
    ("bench_state_heavy_ps.hlsl", "vs_6_0", "VSMain", "bench_state_heavy_ps_vs.dxil"),
    ("bench_state_heavy_ps.hlsl", "ps_6_0", "PSMain", "bench_state_heavy_ps_ps.dxil"),
    ("textured_filter.hlsl", "ps_6_0", "PSMain", "textured_filter.dxil"),
    ("textured_pbr_quad.hlsl", "ps_6_0", "PSMain", "textured_pbr_quad.dxil"),
    # Geometry
    ("bench_geometry_stress.hlsl", "vs_6_0", "VSMain", "bench_geometry_stress_vs.dxil"),
    ("bench_geometry_stress.hlsl", "gs_6_0", "GSMain", "bench_geometry_stress_gs.dxil"),
    ("bench_geometry_stress.hlsl", "ps_6_0", "PSMain", "bench_geometry_stress_ps.dxil"),
    # Tessellation
    ("bench_tess_tri.hlsl", "vs_6_0", "VSMain", "bench_tess_tri_vs.dxil"),
    ("bench_tess_tri.hlsl", "hs_6_0", "HSMain", "bench_tess_tri_hs.dxil"),
    ("bench_tess_tri.hlsl", "ds_6_0", "DSMain", "bench_tess_tri_ds.dxil"),
    ("bench_tess_tri.hlsl", "ps_6_0", "PSMain", "bench_tess_tri_ps.dxil"),
    ("bench_tess_heavy.hlsl", "vs_6_0", "VSMain", "bench_tess_heavy_vs.dxil"),
    ("bench_tess_heavy.hlsl", "hs_6_0", "HSMain", "bench_tess_heavy_hs.dxil"),
    ("bench_tess_heavy.hlsl", "ds_6_0", "DSMain", "bench_tess_heavy_ds.dxil"),
    ("bench_tess_heavy.hlsl", "ps_6_0", "PSMain", "bench_tess_heavy_ps.dxil"),
    ("graphics_features.hlsl", "vs_6_4", "TessPointVS", "gfx_tess_point_vs.dxil"),
    ("graphics_features.hlsl", "hs_6_4", "TessPointHS", "gfx_tess_point_hs.dxil"),
    ("graphics_features.hlsl", "ds_6_4", "TessPointDS", "gfx_tess_point_ds.dxil"),
    ("graphics_features.hlsl", "gs_6_4", "TessPointGS", "gfx_tess_point_gs.dxil"),
    ("graphics_features.hlsl", "ps_6_4", "TessPointPS", "gfx_tess_point_ps.dxil"),
    # RayTracing
    ("raytracing_inline.hlsl", "cs_6_5", "RayQueryMain", "raytracing_inline_cs.dxil"),
    ("raytracing_pipeline.hlsl", "lib_6_5", None, "raytracing_pipeline.dxil"),
    ("bench_rt_inline_heavy.hlsl", "cs_6_5", "RayQueryMain", "bench_rt_inline_heavy_cs.dxil"),
    ("bench_rt_pipeline_heavy.hlsl", "lib_6_5", None, "bench_rt_pipeline_heavy.dxil"),
    # Mesh
    ("bench_mesh_stress.hlsl", "ms_6_5", "MSMain", "bench_mesh_stress_ms.dxil"),
    ("bench_mesh_stress.hlsl", "ps_6_5", "PSMain", "bench_mesh_stress_ps.dxil"),
    ("bench_mesh_cull.hlsl", "as_6_5", "ASMain", "bench_mesh_cull_as.dxil"),
    ("bench_mesh_cull.hlsl", "ms_6_5", "MSMain", "bench_mesh_cull_ms.dxil"),
    ("bench_mesh_cull.hlsl", "ps_6_5", "PSMain", "bench_mesh_cull_ps.dxil"),
]


def compile_one(dxc: str, shader_dir: Path, dxil_dir: Path, spec: tuple):
    hlsl, profile, entry, out_name = spec
    src = shader_dir / hlsl
    dst = dxil_dir / out_name
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return None
    cmd = [dxc, "-T", profile, "-O3", "-Fo", str(dst), str(src)]
    if entry:
        cmd += ["-E", entry]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return f"{out_name}: {r.stderr.strip()}"
    return None


def main():
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-dir", type=Path, default=repo / "build")
    ap.add_argument("--shader-dir", type=Path,
                    default=repo / "bench" / "nacelle" / "shaders")
    ap.add_argument("--dxil-dir", type=Path, default=None)
    ap.add_argument("--dxc", default=shutil.which("dxc"))
    ap.add_argument("--llvm-as", default=(
        shutil.which("llvm-as")
        or ("/opt/homebrew/opt/llvm/bin/llvm-as"
            if Path("/opt/homebrew/opt/llvm/bin/llvm-as").exists() else None)))
    ap.add_argument("--msc", default=shutil.which("metal-shaderconverter"),
                    help="Unused (kept for CLI compatibility); IRConverter is linked in-process")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--filter", default="")
    ap.add_argument("--backend", default="all",
                    choices=["all", "crucible", "irconverter"])
    ap.add_argument("-o", "--out", type=Path, default=Path("bench-results.json"))
    args = ap.parse_args()

    bench_bin = args.build_dir / "tools" / "crucible-bench" / "crucible-bench"
    if not bench_bin.exists():
        sys.exit(f"bench: crucible-bench not found at {bench_bin}; build first")
    if not args.dxc:
        sys.exit("bench: dxc not found")

    dxil_dir = args.dxil_dir or (args.build_dir / "bench-dxil")
    dxil_dir.mkdir(parents=True, exist_ok=True)

    # Compile all HLSL shaders to DXIL in parallel (cached by mtime).
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        futs = [ex.submit(compile_one, args.dxc, args.shader_dir, dxil_dir, t)
                for t in DXIL_TARGETS]
        for f in concurrent.futures.as_completed(futs):
            err = f.result()
            if err:
                errors.append(err)

    # Assemble linalg_matmul.ll if llvm-as is available.
    linalg_ll = args.shader_dir / "linalg_matmul.ll"
    linalg_dxil = dxil_dir / "linalg_matmul.dxil"
    if args.llvm_as and linalg_ll.exists():
        if not linalg_dxil.exists() or linalg_dxil.stat().st_mtime < linalg_ll.stat().st_mtime:
            subprocess.run([args.llvm_as, str(linalg_ll), "-o", str(linalg_dxil)],
                           check=False)

    if errors:
        print("bench: some DXIL compilations failed:", file=sys.stderr)
        for e in errors:
            print("  ", e, file=sys.stderr)

    cmd = [
        str(bench_bin),
        "--shader-dir", str(dxil_dir),
        "--iters", str(args.iters),
        "--warmup", str(args.warmup),
        "--width", str(args.width),
        "--height", str(args.height),
        "--steps", str(args.steps),
        "--backend", args.backend,
        "-o", str(args.out),
    ]
    if args.filter:
        cmd += ["--filter", args.filter]

    rc = subprocess.call(cmd)
    sys.exit(rc)


if __name__ == "__main__":
    main()

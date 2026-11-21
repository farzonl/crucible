//===- main.mm - crucible-bench CLI driver --------------------------------===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Runs the full Nacelle benchmark catalog head-to-head on Metal (crucible)
// and Metal (IRConverter) in-process, emitting Nacelle-compatible JSON.
//
//===----------------------------------------------------------------------===//

#include "bench_types.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace crucible_bench;

static std::string serializeResultJson(const MetalBenchResult &r) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(4) << "    {"
     << "\"benchId\": \"" << jsonEscape(r.benchId) << "\", "
     << "\"benchName\": \"" << jsonEscape(r.benchName) << "\", "
     << "\"category\": \"" << jsonEscape(r.category) << "\", "
     << "\"backend\": \"" << jsonEscape(r.backend) << "\", "
     << "\"shaderFormat\": \"" << jsonEscape(r.shaderFormat) << "\", "
     << "\"simdWidth\": " << r.simdWidth << ", "
     << "\"ran\": " << (r.ran ? "true" : "false") << ", "
     << "\"compileMs\": " << std::setprecision(3) << r.compileMs << ", "
     << "\"firstDispatchMs\": " << r.firstDispatchMs << ", "
     << "\"coldMs\": " << r.coldMs << ", "
     << "\"meanMs\": " << std::setprecision(4) << r.meanMs << ", "
     << "\"medianMs\": " << r.medianMs << ", "
     << "\"minMs\": " << r.minMs << ", "
     << "\"shortJobMs\": " << r.shortJobMs << ", "
     << "\"longJobMs\": " << r.longJobMs << ", "
     << "\"checksum\": " << std::setprecision(2) << r.checksum << ", "
     << "\"status\": \"" << jsonEscape(r.status) << "\""
     << "}";
  return os.str();
}

int main(int argc, char **argv) {
  std::string shaderDir;
  std::string outPath = "bench-results.json";
  std::string filter;
  std::string backend = "all"; // all, crucible, irconverter
  uint32_t width = 256;
  uint32_t height = 256;
  uint32_t maxSteps = 48;
  uint32_t iters = 20;
  uint32_t warmup = 3;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> const char * {
      if (++i >= argc) {
        fprintf(stderr, "crucible-bench: missing value for %s\n", arg.c_str());
        exit(1);
      }
      return argv[i];
    };
    if (arg == "--shader-dir")
      shaderDir = next();
    else if (arg == "-o" || arg == "--out")
      outPath = next();
    else if (arg == "--filter")
      filter = next();
    else if (arg == "--backend")
      backend = next();
    else if (arg == "--width")
      width = static_cast<uint32_t>(strtoul(next(), nullptr, 10));
    else if (arg == "--height")
      height = static_cast<uint32_t>(strtoul(next(), nullptr, 10));
    else if (arg == "--steps")
      maxSteps = static_cast<uint32_t>(strtoul(next(), nullptr, 10));
    else if (arg == "--iters")
      iters = static_cast<uint32_t>(strtoul(next(), nullptr, 10));
    else if (arg == "--warmup")
      warmup = static_cast<uint32_t>(strtoul(next(), nullptr, 10));
    else if (arg == "-h" || arg == "--help") {
      printf("usage: crucible-bench --shader-dir <dxil_dir> [-o out.json] "
             "[--iters N] [--warmup W] [--width W] [--height H] [--steps S] "
             "[--filter substr] [--backend all|crucible|irconverter]\n");
      return 0;
    } else {
      fprintf(stderr, "crucible-bench: unknown argument '%s'\n", arg.c_str());
      return 1;
    }
  }

  if (shaderDir.empty()) {
    fprintf(stderr, "crucible-bench: --shader-dir is required\n");
    return 1;
  }

  bool runCrucible = (backend == "all" || backend == "crucible");
  bool runIRConv = (backend == "all" || backend == "irconverter");

  auto catalog = buildBenchmarkCatalog();
  std::vector<MetalBenchResult> allResults;

  printf("%-24s %-12s | %-11s %-9s %-10s | %-11s %-9s %-10s | %s\n",
         "BENCHMARK", "CATEGORY", "CRUC_COMP", "CRUC_MED", "CRUC_ST",
         "MSC_COMP", "MSC_MED", "MSC_ST", "SPEEDUP");
  printf("%s\n", std::string(118, '-').c_str());

  int crucOK = 0, mscOK = 0, total = 0;

  for (const auto &spec : catalog) {
    if (!filter.empty() && spec.id.find(filter) == std::string::npos)
      continue;
    ++total;

    auto stamp = [&](MetalBenchResult &r) {
      r.benchId = spec.id;
      r.benchName = spec.name;
      r.category = categoryName(spec.category);
    };

    MetalBenchResult crucRes, mscRes;
    crucRes.backend = "Metal (crucible)";
    crucRes.shaderFormat = "DXIL->MSL";
    mscRes.backend = "Metal (IRConverter)";
    mscRes.shaderFormat = "DXIL->AIR";

    if (spec.kind == PipelineKind::Compute) {
      std::vector<uint8_t> dxil;
      if (!readBinary(shaderDir + "/" + spec.csDxil, dxil)) {
        crucRes.status = mscRes.status = "SKIPPED (missing " + spec.csDxil + ")";
      } else {
        if (runCrucible)
          crucRes = runCrucibleComputeBench(spec.id, dxil, width, height,
                                            maxSteps, iters, warmup);
        if (runIRConv)
          mscRes = runIRConverterComputeBench(spec.id, dxil, width, height,
                                              maxSteps, iters, warmup);
      }
    } else if (spec.kind == PipelineKind::GraphicsVSPS ||
               spec.kind == PipelineKind::GraphicsVSGSPS ||
               spec.kind == PipelineKind::GraphicsTessGS) {
      std::vector<uint8_t> vs, ps, hs, ds, gs;
      bool ok = readBinary(shaderDir + "/" + spec.vsDxil, vs) &&
                readBinary(shaderDir + "/" + spec.psDxil, ps);
      if (ok && !spec.hsDxil.empty())
        ok = readBinary(shaderDir + "/" + spec.hsDxil, hs);
      if (ok && !spec.dsDxil.empty())
        ok = readBinary(shaderDir + "/" + spec.dsDxil, ds);
      if (ok && !spec.gsDxil.empty())
        ok = readBinary(shaderDir + "/" + spec.gsDxil, gs);
      if (!ok) {
        crucRes.status = mscRes.status = "SKIPPED (missing stage dxil)";
      } else {
        if (runCrucible)
          crucRes = runCrucibleGraphicsBench(
              spec.id, categoryName(spec.category), vs, ps, width, height,
              iters, warmup, hs, ds, gs, spec.hasTexture);
        if (runIRConv)
          mscRes = runIRConverterGraphicsBench(
              spec.id, categoryName(spec.category), vs, ps, width, height,
              iters, warmup, hs, ds, gs, spec.hasTexture);
      }
    } else if (spec.kind == PipelineKind::GraphicsMSPS) {
      std::vector<uint8_t> as, ms, ps;
      bool ok = readBinary(shaderDir + "/" + spec.msDxil, ms) &&
                readBinary(shaderDir + "/" + spec.psDxil, ps);
      if (ok && !spec.asDxil.empty())
        ok = readBinary(shaderDir + "/" + spec.asDxil, as);
      if (!ok) {
        crucRes.status = mscRes.status = "SKIPPED (missing mesh dxil)";
      } else {
        if (runCrucible)
          crucRes = runCrucibleMeshBench(spec.id, as, ms, ps, iters, warmup);
        if (runIRConv)
          mscRes = runIRConverterMeshBench(spec.id, ms, ps, iters, warmup);
      }
    } else if (spec.kind == PipelineKind::RayTracingInline ||
               spec.kind == PipelineKind::RayTracingPipeline) {
      std::vector<uint8_t> rt;
      if (!readBinary(shaderDir + "/" + spec.rtDxil, rt)) {
        crucRes.status = mscRes.status = "SKIPPED (missing " + spec.rtDxil + ")";
      } else {
        bool isInline = (spec.kind == PipelineKind::RayTracingInline);
        if (runCrucible)
          crucRes = runCrucibleRaytracingBench(spec.id, isInline, rt, width,
                                               height, iters, warmup);
        if (runIRConv)
          mscRes = runIRConverterRaytracingBench(spec.id, isInline, rt, width,
                                                 height, iters, warmup);
      }
    }

    stamp(crucRes);
    stamp(mscRes);
    if (runCrucible)
      allResults.push_back(crucRes);
    if (runIRConv)
      allResults.push_back(mscRes);

    if (crucRes.ran)
      ++crucOK;
    if (mscRes.ran)
      ++mscOK;

    std::string speedup = "-";
    if (crucRes.ran && mscRes.ran && crucRes.medianMs > 0.0) {
      char buf[32];
      snprintf(buf, sizeof(buf), "%.2fx", mscRes.medianMs / crucRes.medianMs);
      speedup = buf;
    }

    auto shortStatus = [](const MetalBenchResult &r) -> std::string {
      if (r.ran)
        return "OK";
      if (r.status.rfind("SKIPPED", 0) == 0)
        return "SKIP";
      return "FAIL";
    };

    printf("%-24s %-12s | %8.2fms %7.3fms %-10s | %8.2fms %7.3fms %-10s | %s\n",
           spec.id.c_str(), categoryName(spec.category), crucRes.compileMs,
           crucRes.medianMs, shortStatus(crucRes).c_str(), mscRes.compileMs,
           mscRes.medianMs, shortStatus(mscRes).c_str(), speedup.c_str());
    if (!crucRes.ran && runCrucible && crucRes.status != "SKIPPED")
      printf("    [crucible] %s\n", crucRes.status.c_str());
    if (!mscRes.ran && runIRConv && mscRes.status != "SKIPPED")
      printf("    [IRConv  ] %s\n", mscRes.status.c_str());
    fflush(stdout);
  }

  printf("%s\n", std::string(118, '-').c_str());
  printf("Summary: Crucible OK %d/%d | IRConverter OK %d/%d\n", crucOK, total,
         mscOK, total);

  std::ofstream ofs(outPath);
  if (!ofs) {
    fprintf(stderr, "crucible-bench: cannot write %s\n", outPath.c_str());
    return 1;
  }
  ofs << "{\n  \"version\": 1,\n  \"results\": [\n";
  for (size_t i = 0; i < allResults.size(); ++i) {
    ofs << serializeResultJson(allResults[i])
        << (i + 1 < allResults.size() ? ",\n" : "\n");
  }
  ofs << "  ]\n}\n";
  printf("wrote %s\n", outPath.c_str());
  return 0;
}

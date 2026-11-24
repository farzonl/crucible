//===- main.cpp - crucible-run multi-backend compute runner ------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL/SPIR-V compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Minimal compute dispatch harness for lit/FileCheck E2E tests across GPU
// backends (Metal, Vulkan/MoltenVK/Lavapipe):
//
//   crucible-run [--backend metal] --lib k.metallib --entry main0 \
//                --threads 64,1,1 --groups 1,1,1 --buffer 0:f32:64:iota
//
//   crucible-run [--backend vulkan] --spv k.spv --entry main \
//                --groups 1,1,1 --buffer 0:f32:64:iota [--cbuffer 1:f32:4:...]
//
// Buffers are declared as <index>:<type>:<count>:<init> where <type> is one
// of f32/u32/i32 and <init> is "iota", "zero", or comma-separated values.
// After dispatch every storage buffer is printed as "buffer<index>: v0 v1 ..."
// so FileCheck can assert exact GPU results.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace crucible_run;

namespace {

[[noreturn]] void printUsageAndFail() {
  fail("usage: crucible-run [--backend metal|vulkan] "
       "[--lib <metallib> | --spv <file.spv>] --entry <kernel> "
       "[--threads x,y,z] [--groups x,y,z] --buffer i:type:count:init ... "
       "[--cbuffer i:type:count:init ...] "
       "[--specialization id:hexbytes|default ...] "
       "[--iters N --warmup W --bench-json] [--ir-ab i0,i1,...]");
}

std::string normalizeBackend(const std::string &B) {
  if (B == "vk")
    return "vulkan";
  return B;
}

} // namespace

int main(int argc, char **argv) {
  RunnerOptions Opts;

  for (int I = 1; I < argc; ++I) {
    std::string Arg = argv[I];
    auto Next = [&]() -> const char * {
      if (++I >= argc)
        fail("missing value for " + Arg);
      return argv[I];
    };
    if (Arg == "--version") {
      printf("crucible-run version %s\n", CRUCIBLE_VERSION);
      return 0;
    } else if (Arg == "--backend") {
      Opts.Backend = Next();
    } else if (Arg.rfind("--backend=", 0) == 0) {
      Opts.Backend = Arg.substr(sizeof("--backend=") - 1);
    } else if (Arg == "--lib") {
      Opts.LibPath = Next();
    } else if (Arg == "--spv") {
      Opts.SpvPath = Next();
    } else if (Arg == "--entry") {
      Opts.Entry = Next();
    } else if (Arg == "--threads") {
      if (!parseTriple(Next(), Opts.Threads))
        fail("bad --threads (want x,y,z)");
    } else if (Arg == "--groups") {
      if (!parseTriple(Next(), Opts.Groups))
        fail("bad --groups (want x,y,z)");
    } else if (Arg == "--buffer") {
      Opts.Buffers.push_back(parseBufferSpec(Next(), /*IsUniform=*/false));
    } else if (Arg == "--cbuffer") {
      Opts.Buffers.push_back(parseBufferSpec(Next(), /*IsUniform=*/true));
    } else if (Arg == "--specialization") {
      Opts.SpecializationArgs.push_back(Next());
    } else if (Arg == "--iters") {
      Opts.Iters = static_cast<uint32_t>(strtoul(Next(), nullptr, 10));
    } else if (Arg == "--warmup") {
      Opts.Warmup = static_cast<uint32_t>(strtoul(Next(), nullptr, 10));
    } else if (Arg == "--bench-json") {
      Opts.BenchJSON = true;
    } else if (Arg == "--ir-ab") {
      Opts.UseIRAB = true;
      for (const std::string &Tok : split(Next(), ','))
        Opts.IRRootOrder.push_back(
            static_cast<uint32_t>(strtoul(Tok.c_str(), nullptr, 10)));
    } else {
      fail("unknown argument '" + Arg + "'");
    }
  }

  std::string Backend = normalizeBackend(Opts.Backend);
  if (Backend.empty()) {
    if (!Opts.LibPath.empty() && !Opts.SpvPath.empty())
      fail("both --lib and --spv specified; pass --backend explicitly");
    if (!Opts.LibPath.empty())
      Backend = "metal";
    else if (!Opts.SpvPath.empty())
      Backend = "vulkan";
    else
      printUsageAndFail();
  }

  if (Backend == "metal") {
#ifdef CRUCIBLE_RUN_HAS_METAL
    return runMetal(Opts);
#else
    fail("backend 'metal' is not built on this platform");
#endif
  }

  if (Backend == "vulkan") {
#ifdef CRUCIBLE_RUN_HAS_VULKAN
    return runVulkan(Opts);
#else
    fail("backend 'vulkan' is not built (Vulkan/MoltenVK not found at configure "
         "time)");
#endif
  }

  fail("unknown backend '" + Opts.Backend + "' (expected 'metal' or 'vulkan')");
}

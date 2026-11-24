//===- Common.h - Shared types and helpers for crucible-run ------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL/SPIR-V compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_RUN_COMMON_H
#define CRUCIBLE_RUN_COMMON_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace crucible_run {

struct BufferSpec {
  uint32_t Index = 0; // Metal buffer index / Vulkan descriptor binding
  std::string Type;   // f32, u32, i32
  size_t Count = 0;
  std::string Init;
  bool IsUniform = false;
};

struct RunnerOptions {
  std::string Backend; // metal, vulkan, d3d12/dxil (empty = auto-detect from input)
  std::string LibPath;
  std::string SpvPath;
  std::string DxilPath;
  std::string Entry;
  uint32_t Threads[3] = {1, 1, 1};
  uint32_t Groups[3] = {1, 1, 1};
  std::vector<BufferSpec> Buffers;

  // Metal-specific options.
  uint32_t Iters = 0;
  uint32_t Warmup = 0;
  bool BenchJSON = false;
  std::vector<uint32_t> IRRootOrder;
  bool UseIRAB = false;

  // Vulkan-specific options.
  std::vector<std::string> SpecializationArgs;
};

[[noreturn]] void fail(const std::string &Msg);
bool parseTriple(const char *S, uint32_t Out[3]);
std::vector<std::string> split(const std::string &S, char Sep);
BufferSpec parseBufferSpec(const std::string &Arg, bool IsUniform);
void fillBuffer(const BufferSpec &Spec, void *Data);
void printBuffer(const BufferSpec &Spec, const void *Data);

#ifdef CRUCIBLE_RUN_HAS_METAL
int runMetal(const RunnerOptions &Opts);
#endif

#ifdef CRUCIBLE_RUN_HAS_VULKAN
int runVulkan(const RunnerOptions &Opts);
#endif

#ifdef CRUCIBLE_RUN_HAS_D3D12
int runD3D12(const RunnerOptions &Opts);
#endif

} // namespace crucible_run

#endif // CRUCIBLE_RUN_COMMON_H

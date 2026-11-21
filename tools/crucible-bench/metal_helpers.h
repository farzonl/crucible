//===- metal_helpers.h - Shared Metal benchmark helpers ---------*- ObjC++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#import <Metal/Metal.h>

#include <chrono>
#include <cstdint>

namespace crucible_bench {

inline double measureCmdMs(id<MTLCommandBuffer> cmdBuf,
                           std::chrono::steady_clock::time_point s0,
                           std::chrono::steady_clock::time_point s1) {
  double gpuMs = (cmdBuf.GPUEndTime - cmdBuf.GPUStartTime) * 1000.0;
  if (gpuMs > 0.0)
    return gpuMs;
  return std::chrono::duration<double, std::milli>(s1 - s0).count();
}

template <typename Fn>
inline void runWarmup(uint32_t minIters, Fn &&fn) {
  double elapsedMs = 0.0;
  for (uint32_t i = 0; i < minIters || (elapsedMs < 8.0 && i < 64); ++i)
    elapsedMs += fn();
}

inline bool deviceSupportsMeshShaders(id<MTLDevice> device) {
  if (![device respondsToSelector:@selector
               (newRenderPipelineStateWithMeshDescriptor:options:reflection:error:)])
    return false;
  if (@available(macOS 13.0, *)) {
    if (![device supportsFamily:MTLGPUFamilyApple7] &&
        ![device supportsFamily:MTLGPUFamilyMetal3])
      return false;
  }
  return true;
}

inline bool encoderSupportsMeshShaders(id<MTLRenderCommandEncoder> enc) {
  return [enc respondsToSelector:@selector(setObjectBuffer:offset:atIndex:)] &&
         [enc respondsToSelector:@selector(setMeshBuffer:offset:atIndex:)] &&
         [enc respondsToSelector:@selector
              (drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:)];
}

} // namespace crucible_bench

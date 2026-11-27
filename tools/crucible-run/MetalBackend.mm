//===- MetalBackend.mm - Metal compute backend for crucible-run -*- ObjC++ -*-===//
//
// Part of Crucible, an open-source DXIL/SPIR-V compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace crucible_run {

int runMetal(const RunnerOptions &Opts) {
  if (Opts.LibPath.empty() || Opts.Entry.empty())
    fail("usage: crucible-run [--backend metal] --lib <metallib> "
         "--entry <kernel> [--threads x,y,z] [--groups x,y,z] "
         "--buffer i:type:count:init ... "
         "[--iters N --warmup W --bench-json] [--ir-ab i0,i1,...]");

  @autoreleasepool {
    id<MTLDevice> Device = MTLCreateSystemDefaultDevice();
    if (!Device)
      fail("no Metal device available");

    NSError *Err = nil;
    NSURL *URL = [NSURL fileURLWithPath:@(Opts.LibPath.c_str())];
    id<MTLLibrary> Lib = [Device newLibraryWithURL:URL error:&Err];
    if (!Lib)
      fail(std::string("cannot load metallib: ") +
           Err.localizedDescription.UTF8String);

    id<MTLFunction> Fn = [Lib newFunctionWithName:@(Opts.Entry.c_str())];
    if (!Fn)
      fail("kernel '" + Opts.Entry + "' not found in library");

    auto PipelineStart = std::chrono::steady_clock::now();
    MTLComputePipelineDescriptor *CPDesc =
        [[MTLComputePipelineDescriptor alloc] init];
    CPDesc.computeFunction = Fn;
    uint32_t TGThreads = Opts.Threads[0] * Opts.Threads[1] * Opts.Threads[2];
    if (TGThreads > 0 && (TGThreads % 32u) == 0u)
      CPDesc.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;

    // Optional on-disk MTLBinaryArchive PSO cache via CRUCIBLE_METAL_CACHE.
    id<MTLBinaryArchive> Archive = nil;
    NSURL *ArchiveURL = nil;
    if (const char *CacheEnv = std::getenv("CRUCIBLE_METAL_CACHE");
        CacheEnv && CacheEnv[0] != '\0') {
      if (@available(macOS 11.0, *)) {
        ArchiveURL = [NSURL fileURLWithPath:@(CacheEnv)];
        MTLBinaryArchiveDescriptor *BADesc =
            [[MTLBinaryArchiveDescriptor alloc] init];
        if ([[NSFileManager defaultManager] fileExistsAtPath:@(CacheEnv)])
          BADesc.url = ArchiveURL;
        NSError *ArchErr = nil;
        Archive = [Device newBinaryArchiveWithDescriptor:BADesc error:&ArchErr];
        if (!Archive && BADesc.url) {
          BADesc.url = nil;
          Archive = [Device newBinaryArchiveWithDescriptor:BADesc error:&ArchErr];
        }
        if (Archive)
          CPDesc.binaryArchives = @[ Archive ];
      }
    }

    id<MTLComputePipelineState> PSO =
        [Device newComputePipelineStateWithDescriptor:CPDesc
                                              options:MTLPipelineOptionNone
                                           reflection:nil
                                                error:&Err];
    if (PSO && Archive && ArchiveURL) {
      if (@available(macOS 11.0, *)) {
        NSError *ArchErr = nil;
        if ([Archive addComputePipelineFunctionsWithDescriptor:CPDesc
                                                         error:&ArchErr]) {
          [Archive serializeToURL:ArchiveURL error:&ArchErr];
        }
      }
    }
    auto PipelineEnd = std::chrono::steady_clock::now();
    if (!PSO)
      fail(std::string("pipeline creation failed: ") +
           Err.localizedDescription.UTF8String);
    double PipelineMs =
        std::chrono::duration<double, std::milli>(PipelineEnd - PipelineStart)
            .count();

    std::vector<id<MTLBuffer>> MTLBuffers;
    for (const BufferSpec &Spec : Opts.Buffers) {
      id<MTLBuffer> B = [Device newBufferWithLength:Spec.Count * 4
                                            options:MTLResourceStorageModeShared];
      if (!B)
        fail("buffer allocation failed");
      fillBuffer(Spec, B.contents);
      MTLBuffers.push_back(B);
    }

    // Top-level argument buffer for the metal-shaderconverter ABI: root
    // descriptors are raw 64-bit GPU virtual addresses in root-parameter
    // order.
    id<MTLBuffer> ABBuf = nil;
    constexpr unsigned kIRArgumentBufferBindPoint = 2;
    if (Opts.UseIRAB) {
      ABBuf = [Device newBufferWithLength:Opts.IRRootOrder.size() * 8
                                  options:MTLResourceStorageModeShared];
      auto *Slots = static_cast<uint64_t *>(ABBuf.contents);
      for (size_t I = 0; I < Opts.IRRootOrder.size(); ++I) {
        uint32_t BufIdx = Opts.IRRootOrder[I];
        bool Found = false;
        for (size_t J = 0; J < Opts.Buffers.size(); ++J)
          if (Opts.Buffers[J].Index == BufIdx) {
            Slots[I] = MTLBuffers[J].gpuAddress;
            Found = true;
          }
        if (!Found)
          fail("--ir-ab references undeclared buffer index");
      }
    }

    id<MTLCommandQueue> Queue = [Device newCommandQueue];

    auto Dispatch = [&]() -> id<MTLCommandBuffer> {
      id<MTLCommandBuffer> Cmd = [Queue commandBuffer];
      id<MTLComputeCommandEncoder> Enc = [Cmd computeCommandEncoder];
      [Enc setComputePipelineState:PSO];
      if (Opts.UseIRAB) {
        [Enc setBuffer:ABBuf offset:0 atIndex:kIRArgumentBufferBindPoint];
        for (id<MTLBuffer> B : MTLBuffers)
          [Enc useResource:B
                     usage:MTLResourceUsageRead | MTLResourceUsageWrite];
      } else {
        for (size_t I = 0; I < Opts.Buffers.size(); ++I)
          [Enc setBuffer:MTLBuffers[I] offset:0 atIndex:Opts.Buffers[I].Index];
      }
      [Enc dispatchThreadgroups:MTLSizeMake(Opts.Groups[0], Opts.Groups[1],
                                            Opts.Groups[2])
          threadsPerThreadgroup:MTLSizeMake(Opts.Threads[0], Opts.Threads[1],
                                            Opts.Threads[2])];
      [Enc endEncoding];
      [Cmd commit];
      [Cmd waitUntilCompleted];
      if (Cmd.status == MTLCommandBufferStatusError)
        fail(std::string("GPU execution failed: ") +
             (Cmd.error ? Cmd.error.localizedDescription.UTF8String
                        : "unknown"));
      return Cmd;
    };

    auto FirstStart = std::chrono::steady_clock::now();
    Dispatch();
    auto FirstEnd = std::chrono::steady_clock::now();
    double FirstDispatchMs =
        std::chrono::duration<double, std::milli>(FirstEnd - FirstStart)
            .count();

    if (Opts.Iters > 0) {
      for (uint32_t I = 0; I < Opts.Warmup; ++I)
        Dispatch();
      std::vector<double> Samples;
      Samples.reserve(Opts.Iters);
      for (uint32_t I = 0; I < Opts.Iters; ++I) {
        id<MTLCommandBuffer> Cmd = Dispatch();
        Samples.push_back((Cmd.GPUEndTime - Cmd.GPUStartTime) * 1000.0);
      }
      if (Opts.BenchJSON) {
        printf("{\"pipelineMs\": %.6f, \"firstDispatchMs\": %.6f, "
               "\"samplesMs\": [",
               PipelineMs, FirstDispatchMs);
        for (size_t I = 0; I < Samples.size(); ++I)
          printf("%s%.6f", I ? ", " : "", Samples[I]);
        printf("]}\n");
      }
    }

    if (!Opts.BenchJSON)
      for (size_t I = 0; I < Opts.Buffers.size(); ++I)
        printBuffer(Opts.Buffers[I], MTLBuffers[I].contents);
  }
  return 0;
}

} // namespace crucible_run

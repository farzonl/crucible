//===- main.mm - crucible-run Metal compute runner ---------------*- ObjC++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Minimal Metal compute dispatch harness for lit/FileCheck tests:
//
//   crucible-run --lib k.metallib --entry main0 --threads 64,1,1 \
//                --groups 1,1,1 --buffer 0:f32:64:iota
//
// Buffers are declared as <index>:<type>:<count>:<init> where <type> is one
// of f32/u32/i32 and <init> is "iota", "zero", or comma-separated values.
// After dispatch every buffer is printed as "buffer<index>: v0 v1 ..." so
// FileCheck can assert exact GPU results.
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct BufferSpec {
  unsigned Index = 0;
  std::string Type; // f32, u32, i32
  size_t Count = 0;
  std::string Init;
};

[[noreturn]] void fail(const std::string &Msg) {
  fprintf(stderr, "crucible-run: error: %s\n", Msg.c_str());
  exit(1);
}

bool parseTriple(const char *S, unsigned Out[3]) {
  return sscanf(S, "%u,%u,%u", &Out[0], &Out[1], &Out[2]) == 3;
}

std::vector<std::string> split(const std::string &S, char Sep) {
  std::vector<std::string> Parts;
  size_t Start = 0;
  while (true) {
    size_t Pos = S.find(Sep, Start);
    Parts.push_back(S.substr(Start, Pos - Start));
    if (Pos == std::string::npos)
      break;
    Start = Pos + 1;
  }
  return Parts;
}

BufferSpec parseBufferSpec(const std::string &Arg) {
  auto Parts = split(Arg, ':');
  if (Parts.size() != 4)
    fail("bad --buffer spec '" + Arg + "' (want index:type:count:init)");
  BufferSpec Spec;
  Spec.Index = static_cast<unsigned>(strtoul(Parts[0].c_str(), nullptr, 10));
  Spec.Type = Parts[1];
  Spec.Count = strtoull(Parts[2].c_str(), nullptr, 10);
  Spec.Init = Parts[3];
  if (Spec.Type != "f32" && Spec.Type != "u32" && Spec.Type != "i32")
    fail("unsupported buffer type '" + Spec.Type + "'");
  if (Spec.Count == 0)
    fail("buffer count must be nonzero");
  return Spec;
}

void fillBuffer(const BufferSpec &Spec, void *Data) {
  std::vector<std::string> Values;
  bool Iota = Spec.Init == "iota";
  bool Zero = Spec.Init == "zero";
  if (!Iota && !Zero)
    Values = split(Spec.Init, ',');
  for (size_t I = 0; I < Spec.Count; ++I) {
    double V = 0;
    if (Iota)
      V = static_cast<double>(I);
    else if (!Zero)
      V = I < Values.size() ? strtod(Values[I].c_str(), nullptr) : 0;
    if (Spec.Type == "f32")
      static_cast<float *>(Data)[I] = static_cast<float>(V);
    else if (Spec.Type == "u32")
      static_cast<uint32_t *>(Data)[I] = static_cast<uint32_t>(V);
    else
      static_cast<int32_t *>(Data)[I] = static_cast<int32_t>(V);
  }
}

void printBuffer(const BufferSpec &Spec, const void *Data) {
  printf("buffer%u:", Spec.Index);
  for (size_t I = 0; I < Spec.Count; ++I) {
    if (Spec.Type == "f32")
      printf(" %g", static_cast<const float *>(Data)[I]);
    else if (Spec.Type == "u32")
      printf(" %u", static_cast<const uint32_t *>(Data)[I]);
    else
      printf(" %d", static_cast<const int32_t *>(Data)[I]);
  }
  printf("\n");
}

} // namespace

int main(int argc, char **argv) {
  const char *LibPath = nullptr;
  const char *Entry = nullptr;
  unsigned Threads[3] = {1, 1, 1};
  unsigned Groups[3] = {1, 1, 1};
  std::vector<BufferSpec> Buffers;

  for (int I = 1; I < argc; ++I) {
    std::string Arg = argv[I];
    auto Next = [&]() -> const char * {
      if (++I >= argc)
        fail("missing value for " + Arg);
      return argv[I];
    };
    if (Arg == "--lib")
      LibPath = Next();
    else if (Arg == "--entry")
      Entry = Next();
    else if (Arg == "--threads") {
      if (!parseTriple(Next(), Threads))
        fail("bad --threads (want x,y,z)");
    } else if (Arg == "--groups") {
      if (!parseTriple(Next(), Groups))
        fail("bad --groups (want x,y,z)");
    } else if (Arg == "--buffer")
      Buffers.push_back(parseBufferSpec(Next()));
    else
      fail("unknown argument '" + Arg + "'");
  }
  if (!LibPath || !Entry)
    fail("usage: crucible-run --lib <metallib> --entry <kernel> "
         "[--threads x,y,z] [--groups x,y,z] --buffer i:type:count:init ...");

  @autoreleasepool {
    id<MTLDevice> Device = MTLCreateSystemDefaultDevice();
    if (!Device)
      fail("no Metal device available");

    NSError *Err = nil;
    NSURL *URL = [NSURL fileURLWithPath:@(LibPath)];
    id<MTLLibrary> Lib = [Device newLibraryWithURL:URL error:&Err];
    if (!Lib)
      fail(std::string("cannot load metallib: ") +
           Err.localizedDescription.UTF8String);

    id<MTLFunction> Fn = [Lib newFunctionWithName:@(Entry)];
    if (!Fn)
      fail(std::string("kernel '") + Entry + "' not found in library");

    id<MTLComputePipelineState> PSO =
        [Device newComputePipelineStateWithFunction:Fn error:&Err];
    if (!PSO)
      fail(std::string("pipeline creation failed: ") +
           Err.localizedDescription.UTF8String);

    std::vector<id<MTLBuffer>> MTLBuffers;
    for (const BufferSpec &Spec : Buffers) {
      id<MTLBuffer> B = [Device newBufferWithLength:Spec.Count * 4
                                            options:MTLResourceStorageModeShared];
      if (!B)
        fail("buffer allocation failed");
      fillBuffer(Spec, B.contents);
      MTLBuffers.push_back(B);
    }

    id<MTLCommandQueue> Queue = [Device newCommandQueue];
    id<MTLCommandBuffer> Cmd = [Queue commandBuffer];
    id<MTLComputeCommandEncoder> Enc = [Cmd computeCommandEncoder];
    [Enc setComputePipelineState:PSO];
    for (size_t I = 0; I < Buffers.size(); ++I)
      [Enc setBuffer:MTLBuffers[I] offset:0 atIndex:Buffers[I].Index];
    [Enc dispatchThreadgroups:MTLSizeMake(Groups[0], Groups[1], Groups[2])
        threadsPerThreadgroup:MTLSizeMake(Threads[0], Threads[1], Threads[2])];
    [Enc endEncoding];
    [Cmd commit];
    [Cmd waitUntilCompleted];
    if (Cmd.status == MTLCommandBufferStatusError)
      fail(std::string("GPU execution failed: ") +
           (Cmd.error ? Cmd.error.localizedDescription.UTF8String : "unknown"));

    for (size_t I = 0; I < Buffers.size(); ++I)
      printBuffer(Buffers[I], MTLBuffers[I].contents);
  }
  return 0;
}

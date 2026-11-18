//===- main.cpp - crucible-offload: offload-test-suite runner ----*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Drop-in replacement for llvm/offload-test-suite's `offloader`:
//
//   crucible-offload pipeline.yaml shader.o [-quiet] [-o out]
//
// Crucible presents itself as a *DirectX* implementation whose backend is
// Metal -- the MoltenVK model. It reuses the suite's YAML pipeline parser and
// result-verification rules, but compiles the DXIL with crucible and executes
// it on Metal with reflection-driven direct [[buffer(n)]] bindings instead of
// metal-shaderconverter's argument-buffer ABI. Compute pipelines only.
//
//===----------------------------------------------------------------------===//

#include "Support/Check.h"
#include "Support/Pipeline.h"

#include "crucible/Compiler.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "SingleHeader/Metal.hpp"

#include <map>
#include <string>

using namespace llvm;
using namespace offloadtest;

static cl::opt<std::string>
    InputPipeline(cl::Positional, cl::Required,
                  cl::desc("<input pipeline description>"),
                  cl::value_desc("filename"));
static cl::list<std::string> InputShader(cl::Positional,
                                         cl::desc("<input compiled shader>"),
                                         cl::value_desc("filename"));
static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));
static cl::opt<bool>
    Quiet("quiet", cl::desc("Suppress printing the pipeline as output"));

// Crucible implements the D3D12 shader/binding contract (the MoltenVK model);
// Metal is the backend. Accept the offloader's -api flag for drop-in
// compatibility, but only the DirectX personality is valid.
static cl::opt<std::string>
    APIToUse("api", cl::desc("GPU API to use (only 'dx' is supported)"),
             cl::init("dx"));

namespace {

Error mtlError(const Twine &Msg, NS::Error *Err) {
  std::string Detail =
      Err ? std::string(Err->localizedDescription()->utf8String()) : "unknown";
  return createStringError(inconvertibleErrorCode(),
                           (Msg + ": " + Detail).str().c_str());
}

std::optional<crucible::ResourceClass> classify(ResourceKind Kind) {
  switch (Kind) {
  case ResourceKind::Buffer:
  case ResourceKind::StructuredBuffer:
  case ResourceKind::ByteAddressBuffer:
    return crucible::ResourceClass::SRV;
  case ResourceKind::RWBuffer:
  case ResourceKind::RWStructuredBuffer:
  case ResourceKind::RWByteAddressBuffer:
    return crucible::ResourceClass::UAV;
  case ResourceKind::ConstantBuffer:
    return crucible::ResourceClass::CBV;
  default:
    return std::nullopt; // textures/samplers/accel structures: unsupported
  }
}

Error executeCompute(Pipeline &P) {
  if (!P.isCompute())
    return createStringError(
        inconvertibleErrorCode(),
        "crucible-offload: only compute pipelines are supported");

  MTL::Device *Device = MTL::CreateSystemDefaultDevice();
  if (!Device)
    return createStringError(inconvertibleErrorCode(),
                             "no Metal device available");
  MTL::CommandQueue *Queue = Device->newCommandQueue();

  // One MTLBuffer per CPUBuffer, shared across shaders.
  std::map<CPUBuffer *, MTL::Buffer *> GPUBuffers;
  auto GetGPUBuffer = [&](CPUBuffer *CPU) -> MTL::Buffer * {
    auto It = GPUBuffers.find(CPU);
    if (It != GPUBuffers.end())
      return It->second;
    MTL::Buffer *B = Device->newBuffer(CPU->Data[0].get(),
                                       std::max<size_t>(CPU->Size, 16),
                                       MTL::ResourceStorageModeShared);
    GPUBuffers[CPU] = B;
    return B;
  };

  for (Shader &S : P.Shaders) {
    if (S.Stage != Stages::Compute)
      return createStringError(inconvertibleErrorCode(),
                               "non-compute shader stage in compute pipeline");

    crucible::Options Opts;
    Opts.EntryPoint = S.Entry;
    auto Compiled =
        crucible::compileDXILToMSL(S.Shader->getMemBufferRef(), Opts);
    if (!Compiled)
      return Compiled.takeError();
    const crucible::Reflection &R = Compiled->Reflect;

    NS::Error *NErr = nullptr;
    MTL::CompileOptions *CO = MTL::CompileOptions::alloc()->init();
    // Compliance runs match the D3D contract: NaN/INF must propagate
    // (sin(inf) == NaN), which requires safe math -- Metal's relaxed mode
    // still loses NaN through transcendental builtins. The remaining gap is
    // denormal flushing (D3D FTZ vs. Metal preserve in safe mode).
    // Benchmarks opt into fast math separately. (setFastMathEnabled rather
    // than setMathMode: the latter silently no-ops on current runtimes.)
    CO->setFastMathEnabled(false);
    MTL::Library *Lib = Device->newLibrary(
        NS::String::string(Compiled->MSL.c_str(), NS::UTF8StringEncoding), CO,
        &NErr);
    if (!Lib)
      return mtlError("MSL compilation failed", NErr);
    MTL::Function *Fn = Lib->newFunction(
        NS::String::string(R.MSLEntryName.c_str(), NS::UTF8StringEncoding));
    if (!Fn)
      return createStringError(inconvertibleErrorCode(),
                               "kernel '%s' not found after compilation",
                               R.MSLEntryName.c_str());
    MTL::ComputePipelineState *PSO =
        Device->newComputePipelineState(Fn, &NErr);
    if (!PSO)
      return mtlError("pipeline state creation failed", NErr);

    // Resolve every YAML descriptor to a crucible binding slot. YAML
    // resources the shader never declares are skipped (dxc strips unused
    // resources).
    SmallVector<std::pair<uint32_t, MTL::Buffer *>, 8> Binds;
    for (DescriptorSet &Set : P.Sets) {
      for (Resource &Res : Set.Resources) {
        auto Class = classify(Res.Kind);
        if (!Class)
          return createStringError(
              inconvertibleErrorCode(),
              "resource '%s': kind is not supported yet (textures/samplers)",
              Res.Name.c_str());
        if (!Res.DXBinding)
          return createStringError(inconvertibleErrorCode(),
                                   "resource '%s' has no DirectXBinding",
                                   Res.Name.c_str());
        if (Res.getArraySize() != 1)
          return createStringError(
              inconvertibleErrorCode(),
              "resource '%s': resource arrays are not supported yet",
              Res.Name.c_str());

        const crucible::ResourceBinding *Match = nullptr;
        for (const crucible::ResourceBinding &RB : R.Resources) {
          if (RB.Class == *Class && RB.Space == Res.DXBinding->Space &&
              Res.DXBinding->Register >= RB.LowerBound &&
              Res.DXBinding->Register < RB.LowerBound + RB.RangeSize) {
            Match = &RB;
            break;
          }
        }
        if (!Match)
          continue; // not referenced by this shader
        if (!Res.BufferPtr)
          return createStringError(inconvertibleErrorCode(),
                                   "resource '%s' has no backing buffer",
                                   Res.Name.c_str());
        Binds.push_back({Match->MetalIndex, GetGPUBuffer(Res.BufferPtr)});
      }
    }

    MTL::CommandBuffer *Cmd = Queue->commandBuffer();
    MTL::ComputeCommandEncoder *Enc = Cmd->computeCommandEncoder();
    Enc->setComputePipelineState(PSO);
    for (auto &[Index, Buf] : Binds)
      Enc->setBuffer(Buf, 0, Index);
    const auto &G = P.DispatchParameters.DispatchGroupCount;
    Enc->dispatchThreadgroups(
        MTL::Size::Make(G[0], G[1], G[2]),
        MTL::Size::Make(R.NumThreads[0], R.NumThreads[1], R.NumThreads[2]));
    Enc->endEncoding();
    Cmd->commit();
    Cmd->waitUntilCompleted();
    if (Cmd->status() == MTL::CommandBufferStatusError)
      return mtlError("GPU execution failed", Cmd->error());
  }

  // Read results back into the pipeline's CPU buffers for verification.
  for (auto &[CPU, GPU] : GPUBuffers)
    memcpy(CPU->Data[0].get(), GPU->contents(), CPU->Size);
  return Error::success();
}

} // namespace

int main(int ArgC, char **ArgV) {
  const InitLLVM X(ArgC, ArgV);
  cl::ParseCommandLineOptions(
      ArgC, ArgV, "crucible offload-test-suite execution tool\n");
  const ExitOnError ExitOnErr("crucible-offload: error: ");

  if (APIToUse != "dx")
    ExitOnErr(createStringError(
        std::errc::invalid_argument,
        "crucible implements the DirectX personality only (got -api=%s)",
        APIToUse.c_str()));

  auto PipelineBuf = MemoryBuffer::getFileOrSTDIN(InputPipeline);
  ExitOnErr(errorCodeToError(PipelineBuf.getError()));

  Pipeline PipelineDesc;
  yaml::Input YIn((*PipelineBuf)->getBuffer());
  YIn >> PipelineDesc;
  ExitOnErr(errorCodeToError(YIn.error()));

  if (InputShader.size() != PipelineDesc.Shaders.size())
    ExitOnErr(createStringError(
        std::errc::invalid_argument,
        "Pipeline description expects %zu shader(s), %zu provided",
        PipelineDesc.Shaders.size(), InputShader.size()));
  for (size_t I = 0; I < InputShader.size(); ++I) {
    auto Buf = MemoryBuffer::getFileOrSTDIN(InputShader[I]);
    ExitOnErr(errorCodeToError(Buf.getError()));
    PipelineDesc.Shaders[I].Shader = std::move(*Buf);
  }

  NS::AutoreleasePool *Pool = NS::AutoreleasePool::alloc()->init();
  Error ExecErr = executeCompute(PipelineDesc);
  Pool->release();
  ExitOnErr(std::move(ExecErr));

  Error ResultErr = Error::success();
  for (const auto &R : PipelineDesc.Results)
    ResultErr = joinErrors(std::move(ResultErr), verifyResult(R));
  if (ResultErr) {
    logAllUnhandledErrors(std::move(ResultErr), errs());
    return 1;
  }

  if (Quiet)
    return 0;

  std::error_code EC;
  auto Out = std::make_unique<ToolOutputFile>(OutputFilename, EC,
                                              sys::fs::OF_Text);
  ExitOnErr(errorCodeToError(EC));
  yaml::Output YOut(Out->os());
  YOut << PipelineDesc;
  Out->keep();
  return 0;
}

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
// metal-shaderconverter's argument-buffer ABI.
//
//===----------------------------------------------------------------------===//

#include "API/FormatConversion.h"
#include "Support/Check.h"
#include "Support/Pipeline.h"

#include "crucible/Compiler.h"

#include "llvm/Object/DXContainer.h"
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

#include <cerrno>
#include <cstdlib>
#include <map>
#include <tuple>
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
static cl::opt<bool>
    Validation("validation-layer", cl::desc("Enable Metal API Validation"));

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
  case ResourceKind::Texture1D:
  case ResourceKind::Texture1DArray:
  case ResourceKind::Texture2D:
  case ResourceKind::Texture2DArray:
  case ResourceKind::Texture3D:
  case ResourceKind::TextureCube:
  case ResourceKind::TextureCubeArray:
    return crucible::ResourceClass::SRV;
  case ResourceKind::RWBuffer:
  case ResourceKind::RWStructuredBuffer:
  case ResourceKind::RWByteAddressBuffer:
  case ResourceKind::RWTexture1D:
  case ResourceKind::RWTexture1DArray:
  case ResourceKind::RWTexture2D:
  case ResourceKind::RWTexture2DArray:
  case ResourceKind::RWTexture3D:
    return crucible::ResourceClass::UAV;
  case ResourceKind::ConstantBuffer:
    return crucible::ResourceClass::CBV;
  case ResourceKind::Sampler:
    return crucible::ResourceClass::Sampler;
  default:
    return std::nullopt; // accel structures: unsupported
  }
}

MTL::SamplerMinMagFilter metalFilter(FilterMode F) {
  return F == FilterMode::Nearest ? MTL::SamplerMinMagFilterNearest
                                  : MTL::SamplerMinMagFilterLinear;
}

MTL::SamplerAddressMode metalAddress(AddressMode A) {
  switch (A) {
  case AddressMode::Clamp:
    return MTL::SamplerAddressModeClampToEdge;
  case AddressMode::Repeat:
    return MTL::SamplerAddressModeRepeat;
  case AddressMode::Mirror:
    return MTL::SamplerAddressModeMirrorRepeat;
  case AddressMode::Border:
    return MTL::SamplerAddressModeClampToBorderColor;
  case AddressMode::MirrorOnce:
    return MTL::SamplerAddressModeMirrorClampToEdge;
  }
  return MTL::SamplerAddressModeClampToEdge;
}

MTL::CompareFunction metalCompare(CompareFunction C) {
  switch (C) {
  case CompareFunction::Never:
    return MTL::CompareFunctionNever;
  case CompareFunction::Less:
    return MTL::CompareFunctionLess;
  case CompareFunction::Equal:
    return MTL::CompareFunctionEqual;
  case CompareFunction::LessEqual:
    return MTL::CompareFunctionLessEqual;
  case CompareFunction::Greater:
    return MTL::CompareFunctionGreater;
  case CompareFunction::NotEqual:
    return MTL::CompareFunctionNotEqual;
  case CompareFunction::GreaterEqual:
    return MTL::CompareFunctionGreaterEqual;
  case CompareFunction::Always:
    return MTL::CompareFunctionAlways;
  }
  return MTL::CompareFunctionNever;
}

bool isTextureResource(ResourceKind Kind) {
  switch (Kind) {
  case ResourceKind::Texture1D:
  case ResourceKind::RWTexture1D:
  case ResourceKind::Texture1DArray:
  case ResourceKind::RWTexture1DArray:
  case ResourceKind::Texture2D:
  case ResourceKind::RWTexture2D:
  case ResourceKind::Texture2DArray:
  case ResourceKind::RWTexture2DArray:
  case ResourceKind::Texture3D:
  case ResourceKind::RWTexture3D:
  case ResourceKind::TextureCube:
  case ResourceKind::TextureCubeArray:
    return true;
  default:
    return false;
  }
}

bool isWritableTexture(ResourceKind Kind) {
  switch (Kind) {
  case ResourceKind::RWTexture1D:
  case ResourceKind::RWTexture1DArray:
  case ResourceKind::RWTexture2D:
  case ResourceKind::RWTexture2DArray:
  case ResourceKind::RWTexture3D:
    return true;
  default:
    return false;
  }
}

MTL::TextureType metalTextureType(ResourceKind Kind) {
  switch (Kind) {
  case ResourceKind::Texture1D:
  case ResourceKind::RWTexture1D:
  case ResourceKind::Texture2D:
  case ResourceKind::RWTexture2D:
    return MTL::TextureType2D;
  case ResourceKind::Texture1DArray:
  case ResourceKind::RWTexture1DArray:
  case ResourceKind::Texture2DArray:
  case ResourceKind::RWTexture2DArray:
    return MTL::TextureType2DArray;
  case ResourceKind::Texture3D:
  case ResourceKind::RWTexture3D:
    return MTL::TextureType3D;
  case ResourceKind::TextureCube:
    return MTL::TextureTypeCube;
  case ResourceKind::TextureCubeArray:
    return MTL::TextureTypeCubeArray;
  default:
    return MTL::TextureType2D;
  }
}

Expected<MTL::PixelFormat> pixelFormatFor(DataFormat Fmt, int Channels) {
  switch (Fmt) {
  case DataFormat::Float32:
    if (Channels == 1) return MTL::PixelFormatR32Float;
    if (Channels == 2) return MTL::PixelFormatRG32Float;
    if (Channels == 4) return MTL::PixelFormatRGBA32Float;
    break;
  case DataFormat::Float16:
    if (Channels == 1) return MTL::PixelFormatR16Float;
    if (Channels == 2) return MTL::PixelFormatRG16Float;
    if (Channels == 4) return MTL::PixelFormatRGBA16Float;
    break;
  case DataFormat::UInt32:
  case DataFormat::Hex32:
    if (Channels == 1) return MTL::PixelFormatR32Uint;
    if (Channels == 2) return MTL::PixelFormatRG32Uint;
    if (Channels == 4) return MTL::PixelFormatRGBA32Uint;
    break;
  case DataFormat::Int32:
    if (Channels == 1) return MTL::PixelFormatR32Sint;
    if (Channels == 2) return MTL::PixelFormatRG32Sint;
    if (Channels == 4) return MTL::PixelFormatRGBA32Sint;
    break;
  case DataFormat::UInt16:
  case DataFormat::Hex16:
    if (Channels == 1) return MTL::PixelFormatR16Uint;
    if (Channels == 2) return MTL::PixelFormatRG16Uint;
    if (Channels == 4) return MTL::PixelFormatRGBA16Uint;
    break;
  case DataFormat::Int16:
    if (Channels == 1) return MTL::PixelFormatR16Sint;
    if (Channels == 2) return MTL::PixelFormatRG16Sint;
    if (Channels == 4) return MTL::PixelFormatRGBA16Sint;
    break;
  case DataFormat::Depth32:
    if (Channels == 1) return MTL::PixelFormatDepth32Float;
    break;
  default:
    break;
  }
  return createStringError(inconvertibleErrorCode(),
                           "unsupported texture format/channel count (%d)",
                           Channels);
}

Error executeRaster(Pipeline &P) {
  auto Fail = [](const char *Message) {
    return createStringError(inconvertibleErrorCode(), "%s", Message);
  };
  if (!P.isTraditionalRaster() || P.Shaders.size() != 2)
    return Fail("raster execution requires one vertex and one pixel shader");
  auto &B = P.Bindings;
  if (B.SampleCount != 1 || P.Stencil || B.getViewportCount() != 1 ||
      !P.PushConstants.empty() ||
      P.ShadingRate != FragmentShadingRate::Rate1x1)
    return Fail("unsupported raster pipeline state");

  CPUBuffer *Output = B.RTargetBufferPtr;
  if (!Output || Output->OutputProps.Width < 1 ||
      Output->OutputProps.Height < 1 || Output->OutputProps.Depth > 1 ||
      Output->OutputProps.ArraySlices > 1 || Output->OutputProps.MipLevels > 1)
    return Fail("raster execution requires a single 2D color target");
  auto PF = pixelFormatFor(Output->Format, Output->Channels);
  if (!PF)
    return PF.takeError();
  if (*PF == MTL::PixelFormatDepth32Float)
    return Fail("depth render targets are not supported");
  size_t Width = Output->OutputProps.Width;
  size_t Height = Output->OutputProps.Height;
  size_t PixelSize = Output->getSingleElementSize() * Output->Channels;
  if (!PixelSize || Width > Output->Size / PixelSize / Height)
    return Fail("color target exceeds its backing storage");
  size_t Pitch = Width * PixelSize;

  NS::SharedPtr<MTL::Device> Device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!Device)
    return Fail("no Metal device available");
  auto Desc = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
  Desc->colorAttachments()->object(0)->setPixelFormat(*PF);
  auto VertexDesc = MTL::VertexDescriptor::vertexDescriptor();
  NS::Error *NErr = nullptr;
  auto CO = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
  CO->setFastMathEnabled(false);
  bool HasVertex = false, HasPixel = false, UsesVertexBuffer = false;
  bool WritesDepth = false;
  constexpr unsigned VertexDataSlot = 30;
  std::map<CPUBuffer *, NS::SharedPtr<MTL::Buffer>> RasterBuffers;
  SmallVector<std::tuple<bool, unsigned, MTL::Buffer *>, 8> BufferBindings;
  size_t Stride = B.VertexBufferPtr ? B.VertexBufferPtr->Stride : 0;
  if (!Stride)
    Stride = B.getVertexStride();
  for (const Shader &S : P.Shaders) {
    bool Vertex = S.Stage == Stages::Vertex;
    if ((!Vertex && S.Stage != Stages::Pixel) ||
        (Vertex ? HasVertex : HasPixel))
      return Fail("raster execution requires one vertex and one pixel shader");
    crucible::Options Opts;
    Opts.EntryPoint = S.Entry;
    auto Compiled = crucible::compileDXILToMSL(S.Shader->getMemBufferRef(), Opts);
    if (!Compiled)
      return Compiled.takeError();
    const auto &R = Compiled->Reflect;
    if (R.Stage != (Vertex ? crucible::ShaderStage::Vertex
                           : crucible::ShaderStage::Pixel))
      return Fail("DXIL shader stage does not match the raster pipeline");
    for (const auto &RB : R.Resources) {
      if (RB.RangeSize != 1 || RB.HasCounter ||
          (RB.Kind != crucible::ResourceKind::StructuredBuffer &&
           RB.Kind != crucible::ResourceKind::RawBuffer &&
           RB.Kind != crucible::ResourceKind::TypedBuffer &&
           RB.Kind != crucible::ResourceKind::CBuffer))
        return Fail("unsupported raster shader resource kind");
      if (RB.MetalIndex >= VertexDataSlot)
        return Fail("raster buffer binding exceeds the available slots");
      CPUBuffer *CPU = nullptr;
      for (auto &Set : P.Sets)
        for (auto &Res : Set.Resources)
          if (Res.DXBinding && classify(Res.Kind) == RB.Class &&
              !isTextureResource(Res.Kind) && !Res.HasCounter &&
              !Res.IsReserved &&
              Res.DXBinding->Space == RB.Space &&
              Res.DXBinding->Register == RB.LowerBound &&
              Res.getArraySize() == 1)
            CPU = Res.BufferPtr;
      if (!CPU || CPU == Output || !CPU->Size || CPU->Data.size() != 1 ||
          !CPU->Data[0])
        return Fail("missing backing buffer for raster shader resource");
      auto &GPU = RasterBuffers[CPU];
      if (!GPU)
        GPU = NS::TransferPtr(Device->newBuffer(CPU->Data[0].get(), CPU->Size,
                                               MTL::ResourceStorageModeShared));
      if (!GPU)
        return Fail("cannot allocate raster shader buffer");
      BufferBindings.emplace_back(Vertex, RB.MetalIndex, GPU.get());
    }
    if (!Vertex && llvm::any_of(R.Outputs, [](const auto &P) {
          return P.SemanticIndex != 0;
        }))
      return Fail("only color attachment zero is supported");
    auto Lib = NS::TransferPtr(Device->newLibrary(
        NS::String::string(Compiled->MSL.c_str(), NS::UTF8StringEncoding), CO.get(),
        &NErr));
    if (!Lib)
      return mtlError("MSL compilation failed", NErr);
    auto Fn = NS::TransferPtr(Lib->newFunction(
        NS::String::string(R.MSLEntryName.c_str(), NS::UTF8StringEncoding)));
    if (!Fn)
      return Fail("compiled raster entry point was not found");
    if (!Vertex) {
      WritesDepth = llvm::any_of(R.Outputs, [](const auto &P) {
        return P.Kind >= 17 && P.Kind <= 19;
      });
      Desc->setFragmentFunction(Fn.get());
      HasPixel = true;
      continue;
    }
    Desc->setVertexFunction(Fn.get());
    HasVertex = true;
    for (const auto &Input : R.Inputs) {
      if (Input.Kind != 0)
        continue;
      std::string Indexed = Input.Semantic + std::to_string(Input.SemanticIndex);
      auto Attr = llvm::find_if(B.VertexAttributes, [&](const VertexAttribute &A) {
        return StringRef(A.Name).equals_insensitive(Indexed) ||
               (Input.SemanticIndex == 0 &&
                StringRef(A.Name).equals_insensitive(Input.Semantic));
      });
      if (Attr == B.VertexAttributes.end() || Input.ID >= 31 ||
          Attr->Channels < 1 || Attr->Channels > 4 || Attr->Offset < 0 ||
          static_cast<size_t>(Attr->Offset) + Attr->size() > Stride)
        return Fail("missing or invalid vertex attribute layout");
      static constexpr MTL::VertexFormat FloatFormats[] = {
          MTL::VertexFormatFloat, MTL::VertexFormatFloat2,
          MTL::VertexFormatFloat3, MTL::VertexFormatFloat4};
      static constexpr MTL::VertexFormat IntFormats[] = {
          MTL::VertexFormatInt, MTL::VertexFormatInt2,
          MTL::VertexFormatInt3, MTL::VertexFormatInt4};
      static constexpr MTL::VertexFormat UIntFormats[] = {
          MTL::VertexFormatUInt, MTL::VertexFormatUInt2,
          MTL::VertexFormatUInt3, MTL::VertexFormatUInt4};
      const auto *Formats = Attr->Format == DataFormat::Float32 ? FloatFormats
                          : Attr->Format == DataFormat::Int32 ? IntFormats
                          : Attr->Format == DataFormat::UInt32 ? UIntFormats
                          : nullptr;
      if (!Formats)
        return Fail("unsupported Metal vertex attribute format");
      auto *A = VertexDesc->attributes()->object(Input.ID);
      A->setFormat(Formats[Attr->Channels - 1]);
      A->setOffset(Attr->Offset);
      A->setBufferIndex(VertexDataSlot);
      UsesVertexBuffer = true;
    }
  }
  if (UsesVertexBuffer) {
    VertexDesc->layouts()->object(VertexDataSlot)->setStride(Stride);
    Desc->setVertexDescriptor(VertexDesc);
  }
  size_t VertexCount = P.DispatchParameters.VertexCount.value_or(0);
  if (!P.DispatchParameters.VertexCount && B.VertexBufferPtr && Stride &&
      B.VertexBufferPtr->Size % Stride == 0)
    VertexCount = B.VertexBufferPtr->Size / Stride;
  if (!VertexCount || (UsesVertexBuffer &&
       (!B.VertexBufferPtr || VertexCount > B.VertexBufferPtr->Size / Stride)))
    return Fail("invalid raster vertex count or buffer");
  MTL::PrimitiveType Primitive;
  switch (B.Topology) {
  case PrimitiveTopology::TriangleList: Primitive = MTL::PrimitiveTypeTriangle; break;
  case PrimitiveTopology::LineList: Primitive = MTL::PrimitiveTypeLine; break;
  case PrimitiveTopology::PointList: Primitive = MTL::PrimitiveTypePoint; break;
  default: return Fail("unsupported raster primitive topology");
  }
  if (WritesDepth)
    Desc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);
  auto PSO = NS::TransferPtr(Device->newRenderPipelineState(Desc.get(), &NErr));
  if (!PSO)
    return mtlError("render pipeline state creation failed", NErr);
  auto *TexDesc = MTL::TextureDescriptor::texture2DDescriptor(*PF, Width, Height, false);
  TexDesc->setStorageMode(MTL::StorageModeShared);
  TexDesc->setUsage(MTL::TextureUsageRenderTarget);
  auto Target = NS::TransferPtr(Device->newTexture(TexDesc));
  auto Queue = NS::TransferPtr(Device->newCommandQueue());
  if (!Target || !Queue)
    return Fail("cannot allocate Metal render target or queue");
  NS::SharedPtr<MTL::Buffer> Vertices;
  if (UsesVertexBuffer) {
    Vertices = NS::TransferPtr(Device->newBuffer(B.VertexBufferPtr->Data[0].get(),
        B.VertexBufferPtr->Size, MTL::ResourceStorageModeShared));
    if (!Vertices)
      return Fail("cannot allocate Metal vertex buffer");
  }
  auto VP = B.getViewports().front();
  auto Sc = B.getScissors().front();
  if (Sc.X < 0 || Sc.Y < 0 || static_cast<size_t>(Sc.X) + Sc.Width > Width ||
      static_cast<size_t>(Sc.Y) + Sc.Height > Height)
    return Fail("scissor exceeds the render target");
  auto *Pass = MTL::RenderPassDescriptor::renderPassDescriptor();
  NS::SharedPtr<MTL::Texture> DepthTarget;
  NS::SharedPtr<MTL::DepthStencilState> DepthState;
  if (WritesDepth) {
    auto *DepthDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatDepth32Float, Width, Height, false);
    DepthDesc->setStorageMode(MTL::StorageModePrivate);
    DepthDesc->setUsage(MTL::TextureUsageRenderTarget);
    DepthTarget = NS::TransferPtr(Device->newTexture(DepthDesc));
    auto DS = NS::TransferPtr(MTL::DepthStencilDescriptor::alloc()->init());
    DS->setDepthCompareFunction(MTL::CompareFunctionLess);
    DS->setDepthWriteEnabled(true);
    DepthState = NS::TransferPtr(Device->newDepthStencilState(DS.get()));
    if (!DepthTarget || !DepthState)
      return Fail("cannot allocate Metal depth target or state");
    auto *Attachment = Pass->depthAttachment();
    Attachment->setTexture(DepthTarget.get());
    Attachment->setLoadAction(MTL::LoadActionClear);
    Attachment->setClearDepth(1.0);
    Attachment->setStoreAction(MTL::StoreActionDontCare);
  }
  auto *Color = Pass->colorAttachments()->object(0);
  Color->setTexture(Target.get());
  Color->setLoadAction(MTL::LoadActionClear);
  Color->setClearColor(MTL::ClearColor(0, 0, 0, 0));
  Color->setStoreAction(MTL::StoreActionStore);
  auto *Cmd = Queue->commandBuffer();
  if (!Cmd)
    return Fail("cannot allocate Metal render command buffer");
  auto *Enc = Cmd->renderCommandEncoder(Pass);
  if (!Enc)
    return Fail("cannot allocate Metal render command encoder");
  Enc->setRenderPipelineState(PSO.get());
  if (DepthState)
    Enc->setDepthStencilState(DepthState.get());
  Enc->setFrontFacingWinding(MTL::WindingCounterClockwise);
  Enc->setViewport({VP.X, VP.Y, VP.Width, VP.Height, VP.MinDepth, VP.MaxDepth});
  Enc->setScissorRect({static_cast<size_t>(Sc.X), static_cast<size_t>(Sc.Y),
                       Sc.Width, Sc.Height});
  if (Vertices)
    Enc->setVertexBuffer(Vertices.get(), 0, VertexDataSlot);
  for (auto [Vertex, Slot, GPU] : BufferBindings)
    if (Vertex)
      Enc->setVertexBuffer(GPU, 0, Slot);
    else
      Enc->setFragmentBuffer(GPU, 0, Slot);
  Enc->drawPrimitives(Primitive, NS::UInteger{0}, VertexCount);
  Enc->endEncoding();
  Cmd->commit();
  Cmd->waitUntilCompleted();
  if (Cmd->status() == MTL::CommandBufferStatusError)
    return mtlError("GPU rendering failed", Cmd->error());
  Target->getBytes(Output->Data[0].get(), Pitch,
                   MTL::Region::Make2D(0, 0, Width, Height), 0);
  for (auto &[CPU, GPU] : RasterBuffers)
    memcpy(CPU->Data[0].get(), GPU->contents(), CPU->Size);
  return Error::success();
}

Error executeCompute(Pipeline &P) {
  if (!P.isCompute())
    return createStringError(
        inconvertibleErrorCode(),
        "only compute and vertex/pixel raster pipelines are supported");

  MTL::Device *Device = MTL::CreateSystemDefaultDevice();
  if (!Device)
    return createStringError(inconvertibleErrorCode(),
                             "no Metal device available");
  MTL::CommandQueue *Queue = Device->newCommandQueue();

  // One MTLBuffer / MTLTexture per (CPUBuffer, ArraySlice), shared across shaders.
  using BufKey = std::pair<CPUBuffer *, uint32_t>;
  std::map<BufKey, MTL::Buffer *> GPUBuffers;
  std::map<BufKey, MTL::Buffer *> GPUCounters;
  std::map<BufKey, MTL::Texture *> GPUTextures;
  std::map<CPUBuffer *, MTL::TextureUsage> TextureUsages;
  std::map<CPUBuffer *, ResourceKind> TextureKinds;
  std::map<YAMLSampler *, MTL::SamplerState *> GPUSamplers;
  auto GetGPUSampler = [&](YAMLSampler *S) -> MTL::SamplerState * {
    auto It = GPUSamplers.find(S);
    if (It != GPUSamplers.end())
      return It->second;
    MTL::SamplerDescriptor *Desc = MTL::SamplerDescriptor::alloc()->init();
    Desc->setMinFilter(metalFilter(S->MinFilter));
    Desc->setMagFilter(metalFilter(S->MagFilter));
    Desc->setMipFilter(S->MinFilter == FilterMode::Nearest
                           ? MTL::SamplerMipFilterNearest
                           : MTL::SamplerMipFilterLinear);
    MTL::SamplerAddressMode Addr = metalAddress(S->Address);
    Desc->setSAddressMode(Addr);
    Desc->setTAddressMode(Addr);
    Desc->setRAddressMode(Addr);
    Desc->setLodMinClamp(S->MinLOD);
    Desc->setLodMaxClamp(S->MaxLOD);
    if (S->Kind == SamplerKind::SamplerComparison)
      Desc->setCompareFunction(metalCompare(S->ComparisonOp));
    MTL::SamplerState *Samp = Device->newSamplerState(Desc);
    Desc->release();
    GPUSamplers[S] = Samp;
    return Samp;
  };
  auto FindBinding = [](const Resource &Res, const crucible::Reflection &R)
      -> const crucible::ResourceBinding * {
    auto Class = classify(Res.Kind);
    if (!Class || !Res.DXBinding)
      return nullptr;
    for (const crucible::ResourceBinding &RB : R.Resources)
      if (RB.Class == *Class && RB.Space == Res.DXBinding->Space &&
          Res.DXBinding->Register >= RB.LowerBound &&
          Res.DXBinding->Register - RB.LowerBound < RB.RangeSize)
        return &RB;
    return nullptr;
  };
  auto GetGPUBuffer = [&](CPUBuffer *CPU, uint32_t Slice = 0) -> MTL::Buffer * {
    BufKey Key{CPU, Slice};
    auto It = GPUBuffers.find(Key);
    if (It != GPUBuffers.end())
      return It->second;
    MTL::Buffer *B = Device->newBuffer(CPU->Data[Slice].get(),
                                       std::max<size_t>(CPU->Size, 16),
                                       MTL::ResourceStorageModeShared);
    GPUBuffers[Key] = B;
    return B;
  };
  auto GetGPUCounter = [&](CPUBuffer *CPU, uint32_t Slice) -> MTL::Buffer * {
    BufKey Key{CPU, Slice};
    auto It = GPUCounters.find(Key);
    if (It != GPUCounters.end())
      return It->second;
    if (CPU->Counters.size() <= Slice)
      CPU->Counters.resize(Slice + 1, 0);
    uint32_t InitVal = CPU->Counters[Slice];
    MTL::Buffer *B = Device->newBuffer(&InitVal, 16,
                                       MTL::ResourceStorageModeShared);
    GPUCounters[Key] = B;
    return B;
  };
  auto GetGPUTexture = [&](CPUBuffer *CPU,
                           uint32_t ArrIdx = 0) -> Expected<MTL::Texture *> {
    BufKey Key{CPU, ArrIdx};
    auto It = GPUTextures.find(Key);
    if (It != GPUTextures.end())
      return It->second;
    auto PF = pixelFormatFor(CPU->Format, CPU->Channels);
    if (!PF)
      return PF.takeError();
    int W = std::max(1, CPU->OutputProps.Width);
    int H = std::max(1, CPU->OutputProps.Height);
    int D = std::max(1, CPU->OutputProps.Depth);
    int Mips = CPU->OutputProps.MipLevels;
    int Slices = std::max(1, CPU->OutputProps.ArraySlices);
    MTL::TextureType TT = metalTextureType(TextureKinds.at(CPU));
    if (TT == MTL::TextureType1D || TT == MTL::TextureType1DArray) {
      H = 1;
      D = 1;
    } else if (TT == MTL::TextureTypeCube) {
      Slices = 6;
      D = 1;
    } else if (TT == MTL::TextureTypeCubeArray) {
      Slices = std::max(6, Slices);
      D = 1;
    } else if (TT != MTL::TextureType3D) {
      D = 1;
    } else {
      Slices = 1;
    }
    if (Mips <= 0)
      return createStringError(
          std::errc::invalid_argument,
          "Texture resource '%s' requires at least one mip level (got %d).",
          CPU->Name.c_str(), Mips);
    auto Fmt = toFormat(CPU->Format, CPU->Channels);
    if (!Fmt)
      return Fmt.takeError();
    TextureCreateDesc CreateDesc;
    CreateDesc.Fmt = *Fmt;
    CreateDesc.Dim =
        TT == MTL::TextureType1D || TT == MTL::TextureType1DArray
            ? ResourceDimension::Dim1D
        : TT == MTL::TextureType3D ? ResourceDimension::Dim3D
        : TT == MTL::TextureTypeCube || TT == MTL::TextureTypeCubeArray
            ? ResourceDimension::Cube
            : ResourceDimension::Dim2D;
    CreateDesc.IsArray = TT == MTL::TextureType1DArray ||
                         TT == MTL::TextureType2DArray ||
                         TT == MTL::TextureTypeCubeArray;
    CreateDesc.Width = W;
    CreateDesc.Height = H;
    CreateDesc.Depth = D;
    CreateDesc.ArraySlices = Slices;
    CreateDesc.MipLevels = Mips;
    if (auto Err = validateTextureCreateDesc(CreateDesc))
      return Err;
    MTL::TextureDescriptor *Desc = MTL::TextureDescriptor::alloc()->init();
    Desc->setTextureType(TT);
    Desc->setPixelFormat(*PF);
    Desc->setWidth(W);
    Desc->setHeight(H);
    Desc->setDepth(D);
    Desc->setMipmapLevelCount(Mips);
    Desc->setArrayLength(TT == MTL::TextureTypeCube        ? 1
                         : TT == MTL::TextureTypeCubeArray ? Slices / 6
                                                           : Slices);
    MTL::TextureUsage Usage = TextureUsages.at(CPU);
    if (Usage & MTL::TextureUsageShaderAtomic) {
      if (*PF == MTL::PixelFormatR32Float)
        *PF = MTL::PixelFormatR32Uint;
      if (*PF != MTL::PixelFormatR32Uint && *PF != MTL::PixelFormatR32Sint)
        return createStringError(
            inconvertibleErrorCode(),
            "texture '%s': atomics require R32Uint, R32Sint, or R32Float",
            CPU->Name.c_str());
    }
    Desc->setPixelFormat(*PF);
    Desc->setUsage(Usage);
    Desc->setStorageMode(MTL::StorageModeShared);
    MTL::Texture *Tex = Device->newTexture(Desc);
    Desc->release();
    if (!Tex)
      return createStringError(inconvertibleErrorCode(),
                               "failed to create Metal texture '%s'",
                               CPU->Name.c_str());
    size_t BytesPerPixel = CPU->getSingleElementSize() * CPU->Channels;
    size_t Offset = 0;
    const char *Base = CPU->Data[ArrIdx].get();
    for (int Slice = 0; Slice < Slices; ++Slice) {
      for (int Mip = 0; Mip < Mips; ++Mip) {
        int MW = std::max(1, W >> Mip);
        int MH = std::max(1, H >> Mip);
        int MD = std::max(1, D >> Mip);
        size_t BytesPerRow = static_cast<size_t>(MW) * BytesPerPixel;
        size_t BytesPerImage = BytesPerRow * static_cast<size_t>(MH);
        size_t MipBytes = BytesPerImage * static_cast<size_t>(MD);
        if (Offset + MipBytes <= CPU->Size)
          Tex->replaceRegion(MTL::Region::Make3D(0, 0, 0, MW, MH, MD), Mip,
                             Slice, Base + Offset, BytesPerRow,
                             TT == MTL::TextureType3D ? BytesPerImage : 0);
        Offset += MipBytes;
      }
    }
    GPUTextures[Key] = Tex;
    return Tex;
  };

  // A texture shared by multiple shaders must be created with their combined
  // usage, even when its first binding is read-only.
  SmallVector<crucible::CompileResult, 4> CompiledShaders;
  for (Shader &S : P.Shaders) {
    if (S.Stage != Stages::Compute)
      return createStringError(inconvertibleErrorCode(),
                               "non-compute shader stage in compute pipeline");

    crucible::Options Opts;
    Opts.EntryPoint = S.Entry;
    Opts.FlushDenormsF32 = true;
    for (const DescriptorSet &Set : P.Sets) {
      for (const Resource &Res : Set.Resources) {
        auto Class = classify(Res.Kind);
        if (Class && Res.DXBinding && Res.getArraySize() > 1)
          Opts.ArraySizeOverrides.push_back(
              {*Class, Res.DXBinding->Space, Res.DXBinding->Register,
               Res.getArraySize()});
      }
    }
    auto Compiled =
        crucible::compileDXILToMSL(S.Shader->getMemBufferRef(), Opts);
    if (!Compiled)
      return Compiled.takeError();
    for (DescriptorSet &Set : P.Sets) {
      for (Resource &Res : Set.Resources) {
        if (!isTextureResource(Res.Kind) || !Res.BufferPtr)
          continue;
        const auto *Binding = FindBinding(Res, Compiled->Reflect);
        if (!Binding)
          continue;
        TextureKinds[Res.BufferPtr] = Res.Kind;
        auto &Usage = TextureUsages[Res.BufferPtr];
        Usage = static_cast<MTL::TextureUsage>(
            Usage | MTL::TextureUsageShaderRead |
            (isWritableTexture(Res.Kind)
                 ? MTL::TextureUsageShaderWrite
                 : MTL::TextureUsageUnknown) |
            (Binding->RequiresTextureAtomics ? MTL::TextureUsageShaderAtomic
                                             : MTL::TextureUsageUnknown));
      }
    }
    CompiledShaders.push_back(std::move(*Compiled));
  }
  for (const auto &Compiled : CompiledShaders) {
    const crucible::Reflection &R = Compiled.Reflect;

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
        NS::String::string(Compiled.MSL.c_str(), NS::UTF8StringEncoding), CO,
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
    SmallVector<std::pair<uint32_t, MTL::Buffer *>, 8> BufferBinds;
    SmallVector<std::pair<uint32_t, MTL::Texture *>, 8> TextureBinds;
    SmallVector<std::pair<uint32_t, MTL::SamplerState *>, 8> SamplerBinds;
    std::vector<uint32_t> BufferDims;
    if (R.BufferSizesMetalIndex != UINT32_MAX)
      BufferDims.resize(R.BufferSizesMetalIndex + 1, 0);
    for (DescriptorSet &Set : P.Sets) {
      for (Resource &Res : Set.Resources) {
        auto Class = classify(Res.Kind);
        if (!Class)
          return createStringError(
              inconvertibleErrorCode(),
              "resource '%s': kind is not supported yet (accel)",
              Res.Name.c_str());
        if (!Res.DXBinding)
          return createStringError(inconvertibleErrorCode(),
                                   "resource '%s' has no DirectXBinding",
                                   Res.Name.c_str());

        const auto *Match = FindBinding(Res, R);
        if (!Match)
          continue; // not referenced by this shader
        uint32_t RegOffset = Res.DXBinding->Register - Match->LowerBound;
        uint32_t BaseSlot = Match->MetalIndex + RegOffset;
        if (Res.Kind == ResourceKind::Sampler) {
          if (!Res.SamplerPtr)
            return createStringError(inconvertibleErrorCode(),
                                     "sampler '%s' has no descriptor",
                                     Res.Name.c_str());
          for (uint32_t I = 0; I < Res.getArraySize(); ++I)
            SamplerBinds.push_back(
                {BaseSlot + I, GetGPUSampler(Res.SamplerPtr)});
          continue;
        }
        if (!Res.BufferPtr)
          return createStringError(inconvertibleErrorCode(),
                                   "resource '%s' has no backing buffer",
                                   Res.Name.c_str());
        for (uint32_t I = 0; I < Res.getArraySize(); ++I) {
          if (isTextureResource(Res.Kind)) {
            auto Tex = GetGPUTexture(Res.BufferPtr, I);
            if (!Tex)
              return Tex.takeError();
            TextureBinds.push_back({BaseSlot + I, *Tex});
          } else {
            BufferBinds.push_back(
                {BaseSlot + I, GetGPUBuffer(Res.BufferPtr, I)});
            if (Match->HasCounter)
              BufferBinds.push_back(
                  {Match->CounterMetalIndex + RegOffset + I,
                   GetGPUCounter(Res.BufferPtr, I)});
            if (R.BufferSizesMetalIndex != UINT32_MAX &&
                BaseSlot + I < BufferDims.size()) {
              uint32_t ElemCount = 0;
              if (Match->Kind == crucible::ResourceKind::StructuredBuffer &&
                  Match->ElementStride > 0)
                ElemCount = static_cast<uint32_t>(Res.BufferPtr->Size /
                                                  Match->ElementStride);
              else if (Match->Kind == crucible::ResourceKind::RawBuffer)
                ElemCount = static_cast<uint32_t>(Res.BufferPtr->Size);
              else if (Match->Kind == crucible::ResourceKind::TypedBuffer) {
                uint32_t ElemBytes = Res.BufferPtr->getSingleElementSize() *
                                     Res.BufferPtr->Channels;
                if (ElemBytes > 0)
                  ElemCount =
                      static_cast<uint32_t>(Res.BufferPtr->Size / ElemBytes);
              }
              BufferDims[BaseSlot + I] = ElemCount;
            }
          }
        }
      }
    }
    if (R.BufferSizesMetalIndex != UINT32_MAX) {
      MTL::Buffer *DimsBuf = Device->newBuffer(
          BufferDims.data(),
          std::max<size_t>(BufferDims.size() * sizeof(uint32_t), 16),
          MTL::ResourceStorageModeShared);
      BufferBinds.push_back({R.BufferSizesMetalIndex, DimsBuf});
    }

    MTL::CommandBuffer *Cmd = Queue->commandBuffer();
    MTL::ComputeCommandEncoder *Enc = Cmd->computeCommandEncoder();
    Enc->setComputePipelineState(PSO);
    for (auto &[Index, Buf] : BufferBinds)
      Enc->setBuffer(Buf, 0, Index);
    for (auto &[Index, Tex] : TextureBinds)
      Enc->setTexture(Tex, Index);
    for (auto &[Index, Samp] : SamplerBinds)
      Enc->setSamplerState(Samp, Index);
    if (!P.Settings.DX.RootParams.empty()) {
      auto Container = object::DXContainer::create(
          P.Shaders[&Compiled - CompiledShaders.data()].Shader->getMemBufferRef());
      if (!Container)
        return Container.takeError();
      const auto &RSOpt = Container->getRootSignature();
      if (!RSOpt)
        return createStringError(inconvertibleErrorCode(),
                                 "pipeline specifies RootParameters but DXIL "
                                 "container has no RTS0 root signature");
      const auto &RS = *RSOpt;
      auto Headers = RS.param_headers();
      if (P.Settings.DX.RootParams.size() != RS.getNumRootParameters())
        return createStringError(
            inconvertibleErrorCode(),
            "RootParameters count (%zu) does not match RTS0 (%u)",
            P.Settings.DX.RootParams.size(), RS.getNumRootParameters());
      auto FindReg = [&](crucible::ResourceClass Class, uint32_t Space,
                         uint32_t Reg) -> const crucible::ResourceBinding * {
        for (const crucible::ResourceBinding &RB : R.Resources)
          if (RB.Class == Class && RB.Space == Space &&
              Reg >= RB.LowerBound && Reg - RB.LowerBound < RB.RangeSize)
            return &RB;
        return nullptr;
      };
      size_t Idx = 0;
      for (const auto &Header : Headers) {
        const auto &RP = P.Settings.DX.RootParams[Idx++];
        if (RP.Kind == dx::RootParamKind::DescriptorTable)
          continue;
        auto ParamOrErr = RS.getParameter(Header);
        if (!ParamOrErr)
          return ParamOrErr.takeError();
        if (RP.Kind == dx::RootParamKind::Constant) {
          const auto &RC = std::get<dx::RootConstant>(RP.Data);
          auto *CV = dyn_cast<object::DirectX::RootConstantView>(&*ParamOrErr);
          if (!CV)
            return createStringError(inconvertibleErrorCode(),
                                     "RootParameter '%s' is not Constants32Bit",
                                     RC.Name.c_str());
          auto Info = CV->read();
          if (!Info)
            return Info.takeError();
          if (const auto *Match = FindReg(crucible::ResourceClass::CBV,
                                          Info->RegisterSpace,
                                          Info->ShaderRegister)) {
            if (!RC.BufferPtr)
              return createStringError(inconvertibleErrorCode(),
                                       "root constant '%s' has no buffer",
                                       RC.Name.c_str());
            Enc->setBuffer(GetGPUBuffer(RC.BufferPtr), 0, Match->MetalIndex);
          }
        } else if (RP.Kind == dx::RootParamKind::RootDescriptor) {
          const auto &RR = std::get<dx::RootResource>(RP.Data);
          auto Class = classify(RR.Kind);
          if (!Class)
            return createStringError(inconvertibleErrorCode(),
                                     "root descriptor '%s': unsupported kind",
                                     RR.Name.c_str());
          auto *DV =
              dyn_cast<object::DirectX::RootDescriptorView>(&*ParamOrErr);
          if (!DV)
            return createStringError(
                inconvertibleErrorCode(),
                "RootParameter '%s' is not a RootDescriptor", RR.Name.c_str());
          auto Info = DV->read(RS.getVersion());
          if (!Info)
            return Info.takeError();
          if (const auto *Match = FindReg(*Class, Info->RegisterSpace,
                                          Info->ShaderRegister)) {
            if (!RR.BufferPtr)
              return createStringError(inconvertibleErrorCode(),
                                       "root descriptor '%s' has no buffer",
                                       RR.Name.c_str());
            Enc->setBuffer(GetGPUBuffer(RR.BufferPtr), 0, Match->MetalIndex);
          }
        }
      }
    }
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
  for (auto &[Key, GPU] : GPUBuffers)
    memcpy(Key.first->Data[Key.second].get(), GPU->contents(), Key.first->Size);
  for (auto &[Key, GPU] : GPUCounters)
    Key.first->Counters[Key.second] =
        *reinterpret_cast<const uint32_t *>(GPU->contents());
  for (auto &[Key, Tex] : GPUTextures) {
    CPUBuffer *CPU = Key.first;
    uint32_t ArrIdx = Key.second;
    if (!(TextureUsages.at(CPU) & MTL::TextureUsageShaderWrite))
      continue;
    int W = std::max(1, CPU->OutputProps.Width);
    int H = std::max(1, CPU->OutputProps.Height);
    int D = std::max(1, CPU->OutputProps.Depth);
    int Mips = std::max(1, CPU->OutputProps.MipLevels);
    int Slices = std::max(1, CPU->OutputProps.ArraySlices);
    MTL::TextureType TT = metalTextureType(TextureKinds.at(CPU));
    if (TT == MTL::TextureType1D || TT == MTL::TextureType1DArray) {
      H = 1;
      D = 1;
    } else if (TT == MTL::TextureTypeCube) {
      Slices = 6;
      D = 1;
    } else if (TT == MTL::TextureTypeCubeArray) {
      Slices = std::max(6, Slices);
      D = 1;
    } else if (TT != MTL::TextureType3D) {
      D = 1;
    } else {
      Slices = 1;
    }
    size_t BytesPerPixel = CPU->getSingleElementSize() * CPU->Channels;
    size_t Offset = 0;
    char *Base = CPU->Data[ArrIdx].get();
    for (int Slice = 0; Slice < Slices; ++Slice) {
      for (int Mip = 0; Mip < Mips; ++Mip) {
        int MW = std::max(1, W >> Mip);
        int MH = std::max(1, H >> Mip);
        int MD = std::max(1, D >> Mip);
        size_t BytesPerRow = static_cast<size_t>(MW) * BytesPerPixel;
        size_t BytesPerImage = BytesPerRow * static_cast<size_t>(MH);
        size_t MipBytes = BytesPerImage * static_cast<size_t>(MD);
        if (Offset + MipBytes <= CPU->Size)
          Tex->getBytes(Base + Offset, BytesPerRow,
                        TT == MTL::TextureType3D ? BytesPerImage : 0,
                        MTL::Region::Make3D(0, 0, 0, MW, MH, MD), Mip, Slice);
        Offset += MipBytes;
      }
    }
  }
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

  for (const auto &Param : PipelineDesc.Settings.DX.RootParams) {
    if (Param.Kind == dx::RootParamKind::Constant) {
      const auto &RC = std::get<dx::RootConstant>(Param.Data);
      if (RC.BufferPtr && RC.BufferPtr->ArraySize != 1)
        ExitOnErr(createStringError(
            std::errc::invalid_argument,
            "Root constant cannot refer to resource arrays."));
    } else if (Param.Kind == dx::RootParamKind::RootDescriptor) {
      const auto &RR = std::get<dx::RootResource>(Param.Data);
      if (RR.getArraySize() != 1)
        ExitOnErr(createStringError(
            std::errc::invalid_argument,
            "Root descriptor cannot refer to resource arrays."));
    }
  }

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

  if (Validation && setenv("MTL_DEBUG_LAYER", "1", 1) != 0)
    ExitOnErr(errorCodeToError(std::error_code(errno, std::generic_category())));

  NS::AutoreleasePool *Pool = NS::AutoreleasePool::alloc()->init();
  Error ExecErr = PipelineDesc.isTraditionalRaster()
                      ? executeRaster(PipelineDesc)
                      : executeCompute(PipelineDesc);
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

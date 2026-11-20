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

#include "Common.h"
#include "API/Device.h"
#include "API/FormatConversion.h"
#include "Support/Pipeline.h"

#include "crucible/Compiler.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Object/DXContainer.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
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

namespace {

Error mtlError(const Twine &Msg, NS::Error *Err) {
  std::string Detail =
      Err ? std::string(Err->localizedDescription()->utf8String()) : "unknown";
  return createStringError(inconvertibleErrorCode(),
                           (Msg + ": " + Detail).str().c_str());
}

std::optional<crucible::ResourceClass> classify(ResourceKind Kind) {
  return crucible_offload::classifyResource(Kind);
}

std::optional<uint32_t>
resourceBindingOffset(const Resource &Res,
                      const crucible::ResourceBinding &RB,
                      bool UseVulkanBinding) {
  auto OffsetInRange = [&](uint32_t Register) -> std::optional<uint32_t> {
    if (Register < RB.LowerBound)
      return std::nullopt;
    uint32_t Offset = Register - RB.LowerBound;
    if (RB.RangeSize != 0xFFFFFFFFu && Offset >= RB.RangeSize)
      return std::nullopt;
    return Offset;
  };
  if (UseVulkanBinding) {
    if (Res.VKBinding)
      return RB.Space == 0 ? OffsetInRange(Res.VKBinding->Binding)
                           : std::nullopt;
    if (Res.DXBinding && RB.Space == Res.DXBinding->Space)
      return OffsetInRange(Res.DXBinding->Register);
    return std::nullopt;
  }
  if (Res.DXBinding)
    return RB.Space == Res.DXBinding->Space
               ? OffsetInRange(Res.DXBinding->Register)
               : std::nullopt;
  if (Res.VKBinding && RB.Space == 0)
    return OffsetInRange(Res.VKBinding->Binding);
  return std::nullopt;
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

MTL::SamplerState *createMetalSampler(MTL::Device *Device,
                                      const YAMLSampler *S) {
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
  return Samp;
}

static bool isCrucibleTextureKind(crucible::ResourceKind K) {
  switch (K) {
  case crucible::ResourceKind::Texture1D:
  case crucible::ResourceKind::Texture2D:
  case crucible::ResourceKind::Texture2DMS:
  case crucible::ResourceKind::Texture3D:
  case crucible::ResourceKind::TextureCube:
  case crucible::ResourceKind::Texture1DArray:
  case crucible::ResourceKind::Texture2DArray:
  case crucible::ResourceKind::Texture2DMSArray:
  case crucible::ResourceKind::TextureCubeArray:
    return true;
  default:
    return false;
  }
}

Expected<MTL::Texture *> createMetalTexture(MTL::Device *Device, CPUBuffer *CPU,
                                            ResourceKind Kind,
                                            MTL::TextureUsage Usage,
                                            uint32_t ArrIdx = 0) {
  auto PF = pixelFormatFor(CPU->Format, CPU->Channels);
  if (!PF)
    return PF.takeError();
  int W = std::max(1, CPU->OutputProps.Width);
  int H = std::max(1, CPU->OutputProps.Height);
  int D = std::max(1, CPU->OutputProps.Depth);
  int Mips = std::max(1, CPU->OutputProps.MipLevels);
  int Slices = std::max(1, CPU->OutputProps.ArraySlices);
  MTL::TextureType TT = metalTextureType(Kind);
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
    return std::move(Err);
  if (Usage & MTL::TextureUsageShaderAtomic) {
    if (*PF == MTL::PixelFormatR32Float)
      *PF = MTL::PixelFormatR32Uint;
    if (*PF != MTL::PixelFormatR32Uint && *PF != MTL::PixelFormatR32Sint)
      return createStringError(
          inconvertibleErrorCode(),
          "texture '%s': atomics require R32Uint, R32Sint, or R32Float",
          CPU->Name.c_str());
  }
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
  return Tex;
}

void readbackMetalTexture(MTL::Texture *Tex, CPUBuffer *CPU, ResourceKind Kind,
                          uint32_t ArrIdx = 0) {
  int W = std::max(1, CPU->OutputProps.Width);
  int H = std::max(1, CPU->OutputProps.Height);
  int D = std::max(1, CPU->OutputProps.Depth);
  int Mips = std::max(1, CPU->OutputProps.MipLevels);
  int Slices = std::max(1, CPU->OutputProps.ArraySlices);
  MTL::TextureType TT = metalTextureType(Kind);
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

static unsigned signatureWords(ArrayRef<crucible::StageParameter> Params) {
  unsigned W = 0;
  for (const auto &P : Params)
    W += P.Rows * P.Columns;
  return W;
}

static std::string buildPassThroughVS(ArrayRef<crucible::StageParameter> Outputs) {
  std::string Out;
  raw_string_ostream OS(Out);
  OS << "#include <metal_stdlib>\nusing namespace metal;\n"
     << "struct CrucibleStageOut {\n";
  for (const auto &P : Outputs) {
    const char *Scalar =
        P.Type == crucible::ComponentType::I32   ? "int"
        : P.Type == crucible::ComponentType::U32 ? "uint"
                                                 : "float";
    std::string Ty = P.Columns > 1 ? std::string(Scalar) + std::to_string(P.Columns)
                                   : std::string(Scalar);
    for (uint32_t RIdx = 0; RIdx < P.Rows; ++RIdx) {
      std::string Field = "sig" + std::to_string(P.ID) +
                          (P.Rows > 1 ? "_" + std::to_string(RIdx) : "");
      OS << "  " << Ty << " " << Field;
      if (P.Kind == 3) {
        OS << " [[position]]";
      } else {
        std::string Hex = toHex(StringRef(P.Semantic).lower(), true);
        uint32_t SemIdx = RIdx < P.RowSemanticIndices.size()
                              ? P.RowSemanticIndices[RIdx]
                              : RIdx;
        OS << " [[user(crucible_" << Hex << "_" << SemIdx << ")]]";
      }
      OS << ";\n";
    }
  }
  unsigned OutWords = signatureWords(Outputs);
  OS << "};\nvertex CrucibleStageOut crucible_passthrough_vs(\n"
     << "    uint vid [[vertex_id]],\n"
     << "    const device uint* buf [[buffer(30)]]) {\n"
     << "  CrucibleStageOut out = {};\n"
     << "  uint base = vid * " << OutWords << "u;\n";
  unsigned WordOff = 0;
  for (const auto &P : Outputs) {
    const char *Scalar =
        P.Type == crucible::ComponentType::I32   ? "int"
        : P.Type == crucible::ComponentType::U32 ? "uint"
                                                 : "float";
    for (uint32_t RIdx = 0; RIdx < P.Rows; ++RIdx) {
      std::string Field = "out.sig" + std::to_string(P.ID) +
                          (P.Rows > 1 ? "_" + std::to_string(RIdx) : "");
      for (uint32_t CIdx = 0; CIdx < P.Columns; ++CIdx) {
        std::string Dst = P.Columns > 1 ? Field + "[" + std::to_string(CIdx) + "]"
                                        : Field;
        OS << "  " << Dst << " = as_type<" << Scalar << ">(buf[base + "
           << WordOff << "u]);\n";
        ++WordOff;
      }
    }
  }
  OS << "  return out;\n}\n";
  return Out;
}

Error executeRaster(Pipeline &P, MTL::Device *Device, bool UseVulkanBinding) {
  auto Fail = [](const char *Message) {
    return createStringError(inconvertibleErrorCode(), "%s", Message);
  };
  if (!P.isTraditionalRaster() || P.Shaders.size() < 2)
    return Fail("raster execution requires one vertex and one pixel shader");
  auto &B = P.Bindings;
  bool HasStencil = P.Stencil && P.Stencil->Enabled;
  if (B.SampleCount < 1 ||
      (B.SampleCount > 1 && !Device->supportsTextureSampleCount(B.SampleCount)) ||
      B.getViewportCount() < 1 || !P.PushConstants.empty() ||
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

  auto Desc = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
  Desc->colorAttachments()->object(0)->setPixelFormat(*PF);
  auto VertexDesc = MTL::VertexDescriptor::vertexDescriptor();
  NS::Error *NErr = nullptr;
  auto CO = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
  CO->setFastMathEnabled(false);
  bool HasVertex = false, HasPixel = false, UsesVertexBuffer = false;
  constexpr unsigned VertexDataSlot = 30;
  std::map<CPUBuffer *, NS::SharedPtr<MTL::Buffer>> RasterBuffers;
  std::map<CPUBuffer *, NS::SharedPtr<MTL::Texture>> RasterTextures;
  std::map<CPUBuffer *, ResourceKind> RasterTextureKinds;
  std::map<CPUBuffer *, MTL::TextureUsage> RasterTextureUsages;
  std::map<YAMLSampler *, NS::SharedPtr<MTL::SamplerState>> RasterSamplers;
  SmallVector<std::tuple<bool, unsigned, MTL::Buffer *>, 8> BufferBindings;
  SmallVector<std::tuple<bool, unsigned, MTL::Texture *>, 8> TextureBindings;
  SmallVector<std::tuple<bool, unsigned, MTL::SamplerState *>, 8> SamplerBindings;
  size_t Stride = B.VertexBufferPtr ? B.VertexBufferPtr->Stride : 0;
  if (!Stride)
    Stride = B.getVertexStride();
  const Shader *VS = nullptr, *HS = nullptr, *DS = nullptr, *GS = nullptr, *PS = nullptr;
  for (const Shader &S : P.Shaders) {
    switch (S.Stage) {
    case Stages::Vertex: if (VS) return Fail("duplicate vertex shader"); VS = &S; break;
    case Stages::Hull: if (HS) return Fail("duplicate hull shader"); HS = &S; break;
    case Stages::Domain: if (DS) return Fail("duplicate domain shader"); DS = &S; break;
    case Stages::Geometry: if (GS) return Fail("duplicate geometry shader"); GS = &S; break;
    case Stages::Pixel: if (PS) return Fail("duplicate pixel shader"); PS = &S; break;
    default: return Fail("unsupported raster shader stage");
    }
  }
  if (!VS || !PS || (HS != nullptr) != (DS != nullptr))
    return Fail("raster execution requires one vertex and one pixel shader");

  auto CollectResources = [&](const crucible::Reflection &R,
                              SmallVectorImpl<std::tuple<unsigned, MTL::Buffer *>> &Bufs,
                              SmallVectorImpl<std::tuple<unsigned, MTL::Texture *>> &Texs,
                              SmallVectorImpl<std::tuple<unsigned, MTL::SamplerState *>> &Samps) -> Error {
    for (const auto &RB : R.Resources) {
      if (RB.RangeSize != 1 || RB.HasCounter)
        return Fail("unsupported raster shader resource array or counter");
      if (RB.Class == crucible::ResourceClass::Sampler) {
        YAMLSampler *Samp = nullptr;
        for (auto &Set : P.Sets)
          for (auto &Res : Set.Resources)
            if (Res.Kind == ResourceKind::Sampler && !Res.IsReserved &&
                resourceBindingOffset(Res, RB, UseVulkanBinding) == 0u &&
                Res.getArraySize() == 1)
              Samp = Res.SamplerPtr;
        if (!Samp)
          return Fail("missing sampler for raster shader resource");
        auto &GPU = RasterSamplers[Samp];
        if (!GPU)
          GPU = NS::TransferPtr(createMetalSampler(Device, Samp));
        if (!GPU)
          return Fail("cannot allocate raster sampler");
        Samps.emplace_back(RB.MetalIndex, GPU.get());
        continue;
      }
      if (isCrucibleTextureKind(RB.Kind)) {
        CPUBuffer *CPU = nullptr;
        ResourceKind ResKind = ResourceKind::Texture2D;
        for (auto &Set : P.Sets)
          for (auto &Res : Set.Resources)
            if (classify(Res.Kind) == RB.Class &&
                isTextureResource(Res.Kind) && !Res.HasCounter &&
                !Res.IsReserved && resourceBindingOffset(Res, RB, UseVulkanBinding) == 0u &&
                Res.getArraySize() == 1) {
              CPU = Res.BufferPtr;
              ResKind = Res.Kind;
            }
        if (!CPU || CPU == Output || !CPU->Size || CPU->Data.size() != 1 ||
            !CPU->Data[0])
          return Fail("missing backing texture for raster shader resource");
        RasterTextureKinds[CPU] = ResKind;
        MTL::TextureUsage Usage = MTL::TextureUsageShaderRead;
        if (isWritableTexture(ResKind))
          Usage = static_cast<MTL::TextureUsage>(Usage | MTL::TextureUsageShaderWrite);
        if (RB.RequiresTextureAtomics)
          Usage = static_cast<MTL::TextureUsage>(Usage | MTL::TextureUsageShaderAtomic);
        RasterTextureUsages[CPU] = Usage;
        auto &GPU = RasterTextures[CPU];
        if (!GPU) {
          auto TexOrErr = createMetalTexture(Device, CPU, ResKind, Usage);
          if (!TexOrErr)
            return TexOrErr.takeError();
          GPU = NS::TransferPtr(*TexOrErr);
        }
        Texs.emplace_back(RB.MetalIndex, GPU.get());
        continue;
      }
      if (RB.Kind != crucible::ResourceKind::StructuredBuffer &&
          RB.Kind != crucible::ResourceKind::RawBuffer &&
          RB.Kind != crucible::ResourceKind::TypedBuffer &&
          RB.Kind != crucible::ResourceKind::CBuffer)
        return Fail("unsupported raster shader resource kind");
      if (RB.MetalIndex >= 28)
        return Fail("raster buffer binding exceeds the available slots");
      CPUBuffer *CPU = nullptr;
      for (auto &Set : P.Sets)
        for (auto &Res : Set.Resources)
          if (classify(Res.Kind) == RB.Class &&
              !isTextureResource(Res.Kind) && !Res.HasCounter &&
              !Res.IsReserved && resourceBindingOffset(Res, RB, UseVulkanBinding) == 0u &&
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
      Bufs.emplace_back(RB.MetalIndex, GPU.get());
    }
    return Error::success();
  };

  auto CompileMSL = [&](const crucible::CompileResult &CR) -> Expected<NS::SharedPtr<MTL::Function>> {
    auto Lib = NS::TransferPtr(Device->newLibrary(
        NS::String::string(CR.MSL.c_str(), NS::UTF8StringEncoding), CO.get(), &NErr));
    if (!Lib)
      return mtlError("MSL compilation failed", NErr);
    auto Fn = NS::TransferPtr(Lib->newFunction(
        NS::String::string(CR.Reflect.MSLEntryName.c_str(), NS::UTF8StringEncoding)));
    if (!Fn)
      return Fail("compiled raster entry point was not found");
    return Fn;
  };

  // --- Tessellation and/or Geometry pipeline (VS -> [HS -> DS ->] [GS ->] PS) ---
  if ((HS && DS) || GS) {
    crucible::Options VSOpts;
    VSOpts.EntryPoint = VS->Entry;
    VSOpts.StageAsCompute = true;
    VSOpts.VertexInputStride = static_cast<uint32_t>(Stride);
    for (const auto &Attr : B.VertexAttributes) {
      using ET = crucible::Options::VertexInputAttribute::ElementType;
      ET T = Attr.Format == DataFormat::Int32 ? ET::Int32
           : Attr.Format == DataFormat::UInt32 ? ET::UInt32
                                               : ET::Float32;
      VSOpts.VertexInputLayout.push_back(
          {Attr.Name, static_cast<uint32_t>(Attr.Offset),
           static_cast<uint32_t>(Attr.Channels), T});
    }
    auto VSComp = crucible::compileDXILToMSL(VS->Shader->getMemBufferRef(), VSOpts);
    if (!VSComp) return VSComp.takeError();
    auto VSFn = CompileMSL(*VSComp); if (!VSFn) return VSFn.takeError();

    crucible::Options PSOpts;
    PSOpts.EntryPoint = PS->Entry;
    auto PSComp = crucible::compileDXILToMSL(PS->Shader->getMemBufferRef(), PSOpts);
    if (!PSComp) return PSComp.takeError();
    auto PSFn = CompileMSL(*PSComp); if (!PSFn) return PSFn.takeError();

    SmallVector<std::tuple<unsigned, MTL::Buffer *>, 4> VSBufs, HSBufs, DSBufs, GSBufs, PSBufs;
    SmallVector<std::tuple<unsigned, MTL::Texture *>, 4> VSTexs, HSTexs, DSTexs, GSTexs, PSTexs;
    SmallVector<std::tuple<unsigned, MTL::SamplerState *>, 4> VSSamps, HSSamps, DSSamps, GSSamps, PSSamps;
    if (Error E = CollectResources(VSComp->Reflect, VSBufs, VSTexs, VSSamps)) return E;
    if (Error E = CollectResources(PSComp->Reflect, PSBufs, PSTexs, PSSamps)) return E;

    size_t VertexCount = P.DispatchParameters.VertexCount.value_or(0);
    if (!P.DispatchParameters.VertexCount && B.VertexBufferPtr && Stride &&
        B.VertexBufferPtr->Size % Stride == 0)
      VertexCount = B.VertexBufferPtr->Size / Stride;
    if (!VertexCount || !B.VertexBufferPtr)
      return Fail("invalid raster vertex count or buffer");

    auto Queue = NS::TransferPtr(Device->newCommandQueue());
    auto Vertices = NS::TransferPtr(Device->newBuffer(
        B.VertexBufferPtr->Data[0].get(), B.VertexBufferPtr->Size,
        MTL::ResourceStorageModeShared));
    unsigned VSOutWords = signatureWords(VSComp->Reflect.Outputs);
    auto VSOutBuf = NS::TransferPtr(Device->newBuffer(
        VertexCount * std::max(1u, VSOutWords) * 4, MTL::ResourceStorageModeShared));

    auto DispatchCompute = [&](MTL::Function *Fn, size_t Count,
                               ArrayRef<std::tuple<unsigned, MTL::Buffer *>> Bufs,
                               ArrayRef<std::tuple<unsigned, MTL::Texture *>> Texs,
                               ArrayRef<std::tuple<unsigned, MTL::SamplerState *>> Samps) -> Error {
      if (!Count) return Error::success();
      auto CPSO = NS::TransferPtr(Device->newComputePipelineState(Fn, &NErr));
      if (!CPSO) return mtlError("compute pipeline creation failed", NErr);
      auto *Cmd = Queue->commandBuffer();
      auto *Enc = Cmd->computeCommandEncoder();
      Enc->setComputePipelineState(CPSO.get());
      for (auto [Slot, Buf] : Bufs) Enc->setBuffer(Buf, 0, Slot);
      for (auto [Slot, Tex] : Texs) Enc->setTexture(Tex, Slot);
      for (auto [Slot, Samp] : Samps) Enc->setSamplerState(Samp, Slot);
      Enc->dispatchThreads(MTL::Size(Count, 1, 1), MTL::Size(1, 1, 1));
      Enc->endEncoding();
      Cmd->commit();
      Cmd->waitUntilCompleted();
      if (Cmd->status() == MTL::CommandBufferStatusError)
        return mtlError("compute dispatch failed", Cmd->error());
      return Error::success();
    };

    VSBufs.emplace_back(28, Vertices.get());
    VSBufs.emplace_back(30, VSOutBuf.get());
    if (Error E = DispatchCompute(VSFn->get(), VertexCount, VSBufs, VSTexs, VSSamps))
      return E;

    NS::SharedPtr<MTL::Buffer> StageBuf = VSOutBuf;
    size_t StageVertCount = VertexCount;
    bool UseHardwareTess = false;
    size_t PatchCount = 0;
    uint32_t DSControlPoints = 0;
    NS::SharedPtr<MTL::Buffer> TessFactorBuf, PatchDataBuf;
    NS::SharedPtr<MTL::Function> DSFn;
    MTL::PrimitiveType DrawPrim = MTL::PrimitiveTypeTriangle;

    if (HS && DS) {
      crucible::Options HSOpts;
      HSOpts.EntryPoint = HS->Entry;
      auto HSComp = crucible::compileDXILToMSL(HS->Shader->getMemBufferRef(), HSOpts);
      if (!HSComp) return HSComp.takeError();
      if (!HSComp->Reflect.Hull) return Fail("missing hull reflection");
      const auto &HI = *HSComp->Reflect.Hull;
      bool DSAsCompute = (GS != nullptr) ||
                         HI.Domain == crucible::TessellationDomain::Isoline ||
                         HI.OutputPrimitive == crucible::TessellationOutputPrimitive::Point;
      crucible::Options DSOpts;
      DSOpts.EntryPoint = DS->Entry;
      DSOpts.StageAsCompute = DSAsCompute;
      auto DSComp = crucible::compileDXILToMSL(DS->Shader->getMemBufferRef(), DSOpts);
      if (!DSComp) return DSComp.takeError();
      auto HSFn = CompileMSL(*HSComp); if (!HSFn) return HSFn.takeError();
      auto DSFnRes = CompileMSL(*DSComp); if (!DSFnRes) return DSFnRes.takeError();
      DSFn = *DSFnRes;
      if (Error E = CollectResources(HSComp->Reflect, HSBufs, HSTexs, HSSamps)) return E;
      if (Error E = CollectResources(DSComp->Reflect, DSBufs, DSTexs, DSSamps)) return E;

      PatchCount = VertexCount / HI.InputControlPointCount;
      if (!PatchCount) return Fail("invalid tessellation patch count");
      unsigned HSCPWords = signatureWords(HSComp->Reflect.Outputs);
      unsigned PatchWords = HSCPWords * HI.OutputControlPointCount +
                            signatureWords(HI.PatchConstants);
      PatchDataBuf = NS::TransferPtr(Device->newBuffer(
          PatchCount * std::max(1u, PatchWords) * 4, MTL::ResourceStorageModeShared));
      unsigned TFStride = HI.Domain == crucible::TessellationDomain::Triangle ? 4u
                        : HI.Domain == crucible::TessellationDomain::Quad ? 6u : 2u;
      size_t TFElemSize = DSAsCompute ? sizeof(float) : 2;
      TessFactorBuf = NS::TransferPtr(Device->newBuffer(
          PatchCount * TFStride * TFElemSize, MTL::ResourceStorageModeShared));

      HSBufs.emplace_back(28, VSOutBuf.get());
      HSBufs.emplace_back(29, TessFactorBuf.get());
      HSBufs.emplace_back(30, PatchDataBuf.get());
      if (Error E = DispatchCompute(HSFn->get(), PatchCount, HSBufs, HSTexs, HSSamps))
        return E;

      if (DSAsCompute) {
        const float *TF = static_cast<const float *>(TessFactorBuf->contents());
        struct DomainItem { float u, v, w; uint32_t patchID; };
        SmallVector<DomainItem, 64> Items;
        for (size_t Patch = 0; Patch < PatchCount; ++Patch) {
          if (HI.Domain == crucible::TessellationDomain::Isoline) {
            int NumLines = std::max(1, static_cast<int>(std::round(TF[Patch * 2 + 0])));
            int NumSegs = std::max(1, static_cast<int>(std::round(TF[Patch * 2 + 1])));
            for (int J = 0; J < NumLines; ++J) {
              float V = static_cast<float>(J) / static_cast<float>(NumLines);
              for (int S = 0; S < NumSegs; ++S) {
                float U0 = static_cast<float>(S) / static_cast<float>(NumSegs);
                float U1 = static_cast<float>(S + 1) / static_cast<float>(NumSegs);
                Items.push_back({U0, V, 0.0f, static_cast<uint32_t>(Patch)});
                Items.push_back({U1, V, 0.0f, static_cast<uint32_t>(Patch)});
              }
            }
            DrawPrim = MTL::PrimitiveTypeLine;
          } else if (HI.Domain == crucible::TessellationDomain::Quad) {
            int NU = std::max(1, static_cast<int>(std::round(TF[Patch * 6 + 4])));
            int NV = std::max(1, static_cast<int>(std::round(TF[Patch * 6 + 5])));
            for (int J = 0; J <= NV; ++J)
              for (int I = 0; I <= NU; ++I)
                Items.push_back({static_cast<float>(I) / NU, static_cast<float>(J) / NV,
                                 0.0f, static_cast<uint32_t>(Patch)});
            DrawPrim = MTL::PrimitiveTypePoint;
          }
        }
        StageVertCount = Items.size();
        auto ItemsBuf = NS::TransferPtr(Device->newBuffer(
            Items.data(), Items.size() * sizeof(DomainItem), MTL::ResourceStorageModeShared));
        unsigned DSOutWords = signatureWords(DSComp->Reflect.Outputs);
        auto DSOutBuf = NS::TransferPtr(Device->newBuffer(
            std::max<size_t>(1, StageVertCount) * std::max(1u, DSOutWords) * 4,
            MTL::ResourceStorageModeShared));
        DSBufs.emplace_back(28, DSOutBuf.get());
        DSBufs.emplace_back(29, ItemsBuf.get());
        DSBufs.emplace_back(30, PatchDataBuf.get());
        if (Error E = DispatchCompute(DSFn.get(), StageVertCount, DSBufs, DSTexs, DSSamps))
          return E;
        StageBuf = DSOutBuf;
        if (!GS) {
          std::string PTVS = buildPassThroughVS(DSComp->Reflect.Outputs);
          auto PTLib = NS::TransferPtr(Device->newLibrary(
              NS::String::string(PTVS.c_str(), NS::UTF8StringEncoding), CO.get(), &NErr));
          if (!PTLib) return mtlError("passthrough VS compilation failed", NErr);
          auto PTFn = NS::TransferPtr(PTLib->newFunction(
              NS::String::string("crucible_passthrough_vs", NS::UTF8StringEncoding)));
          Desc->setVertexFunction(PTFn.get());
        }
      } else {
        UseHardwareTess = true;
        DSControlPoints = DSComp->Reflect.Domain->ControlPointCount;
        Desc->setVertexFunction(DSFn.get());
        MTL::TessellationPartitionMode PartMode = MTL::TessellationPartitionModeInteger;
        switch (HI.Partitioning) {
        case crucible::TessellationPartitioning::Integer: PartMode = MTL::TessellationPartitionModeInteger; break;
        case crucible::TessellationPartitioning::Pow2: PartMode = MTL::TessellationPartitionModePow2; break;
        case crucible::TessellationPartitioning::FractionalOdd: PartMode = MTL::TessellationPartitionModeFractionalOdd; break;
        case crucible::TessellationPartitioning::FractionalEven: PartMode = MTL::TessellationPartitionModeFractionalEven; break;
        }
        Desc->setTessellationPartitionMode(PartMode);
        Desc->setTessellationFactorStepFunction(MTL::TessellationFactorStepFunctionPerPatch);
        Desc->setTessellationOutputWindingOrder(
            HI.OutputPrimitive == crucible::TessellationOutputPrimitive::TriangleCCW
                ? MTL::WindingCounterClockwise
                : MTL::WindingClockwise);
        Desc->setMaxTessellationFactor(std::min(64.0f, HI.MaxTessFactor));
        Desc->setTessellationFactorScaleEnabled(false);
        Desc->setTessellationControlPointIndexType(MTL::TessellationControlPointIndexTypeNone);
      }
    }

    if (GS) {
      crucible::Options GSOpts;
      GSOpts.EntryPoint = GS->Entry;
      auto GSComp = crucible::compileDXILToMSL(GS->Shader->getMemBufferRef(), GSOpts);
      if (!GSComp) return GSComp.takeError();
      if (!GSComp->Reflect.Geometry) return Fail("missing geometry reflection");
      const auto &GI = *GSComp->Reflect.Geometry;
      auto GSFn = CompileMSL(*GSComp); if (!GSFn) return GSFn.takeError();
      if (Error E = CollectResources(GSComp->Reflect, GSBufs, GSTexs, GSSamps)) return E;

      unsigned InputVerts = GI.InputPrimitive == 1 ? 1u : GI.InputPrimitive == 2 ? 2u : 3u;
      size_t PrimCount = StageVertCount / InputVerts;
      unsigned MaxVertsPerPrim =
          GI.OutputTopology == 1 ? GI.MaxVertexCount
          : GI.OutputTopology == 3
              ? (GI.MaxVertexCount >= 2 ? (GI.MaxVertexCount - 1) * 2 : 0)
              : (GI.MaxVertexCount >= 3 ? (GI.MaxVertexCount - 2) * 3 : 0);
      unsigned GSOutWords = signatureWords(GSComp->Reflect.Outputs);
      uint32_t InitCounter = 0;
      auto GSCounterBuf = NS::TransferPtr(Device->newBuffer(
          &InitCounter, sizeof(uint32_t), MTL::ResourceStorageModeShared));
      auto GSOutBuf = NS::TransferPtr(Device->newBuffer(
          std::max<size_t>(1, PrimCount * MaxVertsPerPrim) * std::max(1u, GSOutWords) * 4,
          MTL::ResourceStorageModeShared));
      GSBufs.emplace_back(28, StageBuf.get());
      GSBufs.emplace_back(29, GSCounterBuf.get());
      GSBufs.emplace_back(30, GSOutBuf.get());
      if (Error E = DispatchCompute(GSFn->get(), PrimCount, GSBufs, GSTexs, GSSamps))
        return E;

      StageVertCount = *static_cast<const uint32_t *>(GSCounterBuf->contents());
      StageBuf = GSOutBuf;
      DrawPrim = GI.OutputTopology == 1 ? MTL::PrimitiveTypePoint
               : GI.OutputTopology == 3 ? MTL::PrimitiveTypeLine
                                        : MTL::PrimitiveTypeTriangle;
      std::string PTVS = buildPassThroughVS(GSComp->Reflect.Outputs);
      auto PTLib = NS::TransferPtr(Device->newLibrary(
          NS::String::string(PTVS.c_str(), NS::UTF8StringEncoding), CO.get(), &NErr));
      if (!PTLib) return mtlError("passthrough VS compilation failed", NErr);
      auto PTFn = NS::TransferPtr(PTLib->newFunction(
          NS::String::string("crucible_passthrough_vs", NS::UTF8StringEncoding)));
      Desc->setVertexFunction(PTFn.get());
    }

    Desc->setFragmentFunction(PSFn->get());
    auto *TexDesc = MTL::TextureDescriptor::texture2DDescriptor(*PF, Width, Height, false);
    TexDesc->setStorageMode(MTL::StorageModeShared);
    TexDesc->setUsage(MTL::TextureUsageRenderTarget);
    auto Target = NS::TransferPtr(Device->newTexture(TexDesc));
    NS::SharedPtr<MTL::Texture> MSAATarget;
    if (B.SampleCount > 1) {
      auto *MSAADesc = MTL::TextureDescriptor::texture2DDescriptor(*PF, Width, Height, false);
      MSAADesc->setTextureType(MTL::TextureType2DMultisample);
      MSAADesc->setSampleCount(B.SampleCount);
      MSAADesc->setStorageMode(MTL::StorageModePrivate);
      MSAADesc->setUsage(MTL::TextureUsageRenderTarget);
      MSAATarget = NS::TransferPtr(Device->newTexture(MSAADesc));
    }
    if (B.SampleCount > 1)
      Desc->setRasterSampleCount(B.SampleCount);
    MTL::PixelFormat DepthStencilFormat =
        HasStencil ? MTL::PixelFormatDepth32Float_Stencil8
                   : MTL::PixelFormatDepth32Float;
    Desc->setDepthAttachmentPixelFormat(DepthStencilFormat);
    if (HasStencil)
      Desc->setStencilAttachmentPixelFormat(DepthStencilFormat);

    auto PSO = NS::TransferPtr(Device->newRenderPipelineState(Desc.get(), &NErr));
    if (!PSO) return mtlError("raster render pipeline creation failed", NErr);

    auto VP = B.getViewports().front();
    auto Sc = B.getScissors().front();
    auto *Pass = MTL::RenderPassDescriptor::renderPassDescriptor();
    auto *DSDesc = MTL::TextureDescriptor::texture2DDescriptor(DepthStencilFormat, Width, Height, false);
    if (B.SampleCount > 1) {
      DSDesc->setTextureType(MTL::TextureType2DMultisample);
      DSDesc->setSampleCount(B.SampleCount);
    }
    DSDesc->setStorageMode(MTL::StorageModePrivate);
    DSDesc->setUsage(MTL::TextureUsageRenderTarget);
    auto DepthStencilTarget = NS::TransferPtr(Device->newTexture(DSDesc));
    auto DSStateDesc = NS::TransferPtr(MTL::DepthStencilDescriptor::alloc()->init());
    DSStateDesc->setDepthCompareFunction(MTL::CompareFunctionLessEqual);
    DSStateDesc->setDepthWriteEnabled(true);
    auto DepthStencilState = NS::TransferPtr(Device->newDepthStencilState(DSStateDesc.get()));
    auto *DepthAttachment = Pass->depthAttachment();
    DepthAttachment->setTexture(DepthStencilTarget.get());
    DepthAttachment->setLoadAction(MTL::LoadActionClear);
    DepthAttachment->setClearDepth(1.0);
    DepthAttachment->setStoreAction(MTL::StoreActionDontCare);
    auto *Color = Pass->colorAttachments()->object(0);
    if (MSAATarget) {
      Color->setTexture(MSAATarget.get());
      Color->setResolveTexture(Target.get());
      Color->setStoreAction(MTL::StoreActionMultisampleResolve);
    } else {
      Color->setTexture(Target.get());
      Color->setStoreAction(MTL::StoreActionStore);
    }
    Color->setLoadAction(MTL::LoadActionClear);
    Color->setClearColor(MTL::ClearColor(0, 0, 0, 0));

    auto *Cmd = Queue->commandBuffer();
    auto *Enc = Cmd->renderCommandEncoder(Pass);
    Enc->setRenderPipelineState(PSO.get());
    Enc->setDepthStencilState(DepthStencilState.get());
    Enc->setFrontFacingWinding(MTL::WindingCounterClockwise);
    Enc->setViewport({VP.X, VP.Y, VP.Width, VP.Height, VP.MinDepth, VP.MaxDepth});
    Enc->setScissorRect({static_cast<size_t>(Sc.X), static_cast<size_t>(Sc.Y), Sc.Width, Sc.Height});
    for (auto [Slot, Buf] : PSBufs) Enc->setFragmentBuffer(Buf, 0, Slot);
    for (auto [Slot, Tex] : PSTexs) Enc->setFragmentTexture(Tex, Slot);
    for (auto [Slot, Samp] : PSSamps) Enc->setFragmentSamplerState(Samp, Slot);
    if (UseHardwareTess) {
      Enc->setTessellationFactorBuffer(TessFactorBuf.get(), 0, 0);
      Enc->setVertexBuffer(PatchDataBuf.get(), 0, crucible::DomainInterface::DataMetalIndex);
      for (auto [Slot, Buf] : DSBufs) Enc->setVertexBuffer(Buf, 0, Slot);
      for (auto [Slot, Tex] : DSTexs) Enc->setVertexTexture(Tex, Slot);
      for (auto [Slot, Samp] : DSSamps) Enc->setVertexSamplerState(Samp, Slot);
      Enc->drawPatches(DSControlPoints, 0, PatchCount, nullptr, 0, 1, 0);
    } else if (StageVertCount > 0) {
      Enc->setVertexBuffer(StageBuf.get(), 0, 30);
      Enc->drawPrimitives(DrawPrim, NS::UInteger{0}, StageVertCount);
    }
    Enc->endEncoding();
    Cmd->commit();
    Cmd->waitUntilCompleted();
    if (Cmd->status() == MTL::CommandBufferStatusError)
      return mtlError("GPU raster rendering failed", Cmd->error());
    Target->getBytes(Output->Data[0].get(), Pitch, MTL::Region::Make2D(0, 0, Width, Height), 0);
    for (auto &[CPU, GPU] : RasterBuffers)
      memcpy(CPU->Data[0].get(), GPU->contents(), CPU->Size);
    for (auto &[CPU, Tex] : RasterTextures)
      if (RasterTextureUsages.at(CPU) & MTL::TextureUsageShaderWrite)
        readbackMetalTexture(Tex.get(), CPU, RasterTextureKinds.at(CPU));
    return Error::success();
  }

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
      if (RB.RangeSize != 1 || RB.HasCounter)
        return Fail("unsupported raster shader resource array or counter");
      if (RB.Class == crucible::ResourceClass::Sampler) {
        YAMLSampler *Samp = nullptr;
        for (auto &Set : P.Sets)
          for (auto &Res : Set.Resources)
            if (Res.Kind == ResourceKind::Sampler && !Res.IsReserved &&
                resourceBindingOffset(Res, RB, UseVulkanBinding) == 0u &&
                Res.getArraySize() == 1)
              Samp = Res.SamplerPtr;
        if (!Samp)
          return Fail("missing sampler for raster shader resource");
        auto &GPU = RasterSamplers[Samp];
        if (!GPU)
          GPU = NS::TransferPtr(createMetalSampler(Device, Samp));
        if (!GPU)
          return Fail("cannot allocate raster sampler");
        SamplerBindings.emplace_back(Vertex, RB.MetalIndex, GPU.get());
        continue;
      }
      if (isCrucibleTextureKind(RB.Kind)) {
        CPUBuffer *CPU = nullptr;
        ResourceKind ResKind = ResourceKind::Texture2D;
        for (auto &Set : P.Sets)
          for (auto &Res : Set.Resources)
            if (classify(Res.Kind) == RB.Class &&
                isTextureResource(Res.Kind) && !Res.HasCounter &&
                !Res.IsReserved && resourceBindingOffset(Res, RB, UseVulkanBinding) == 0u &&
                Res.getArraySize() == 1) {
              CPU = Res.BufferPtr;
              ResKind = Res.Kind;
            }
        if (!CPU || CPU == Output || !CPU->Size || CPU->Data.size() != 1 ||
            !CPU->Data[0])
          return Fail("missing backing texture for raster shader resource");
        RasterTextureKinds[CPU] = ResKind;
        MTL::TextureUsage Usage = MTL::TextureUsageShaderRead;
        if (isWritableTexture(ResKind))
          Usage = static_cast<MTL::TextureUsage>(Usage |
                                                 MTL::TextureUsageShaderWrite);
        if (RB.RequiresTextureAtomics)
          Usage = static_cast<MTL::TextureUsage>(Usage |
                                                 MTL::TextureUsageShaderAtomic);
        RasterTextureUsages[CPU] = Usage;
        auto &GPU = RasterTextures[CPU];
        if (!GPU) {
          auto TexOrErr = createMetalTexture(Device, CPU, ResKind, Usage);
          if (!TexOrErr)
            return TexOrErr.takeError();
          GPU = NS::TransferPtr(*TexOrErr);
        }
        TextureBindings.emplace_back(Vertex, RB.MetalIndex, GPU.get());
        continue;
      }
      if (RB.Kind != crucible::ResourceKind::StructuredBuffer &&
          RB.Kind != crucible::ResourceKind::RawBuffer &&
          RB.Kind != crucible::ResourceKind::TypedBuffer &&
          RB.Kind != crucible::ResourceKind::CBuffer)
        return Fail("unsupported raster shader resource kind");
      if (RB.MetalIndex >= VertexDataSlot)
        return Fail("raster buffer binding exceeds the available slots");
      CPUBuffer *CPU = nullptr;
      for (auto &Set : P.Sets)
        for (auto &Res : Set.Resources)
          if (classify(Res.Kind) == RB.Class &&
              !isTextureResource(Res.Kind) && !Res.HasCounter &&
              !Res.IsReserved && resourceBindingOffset(Res, RB, UseVulkanBinding) == 0u &&
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
  if (B.SampleCount > 1)
    Desc->setRasterSampleCount(B.SampleCount);
  MTL::PixelFormat DepthStencilFormat =
      HasStencil ? MTL::PixelFormatDepth32Float_Stencil8
                 : MTL::PixelFormatDepth32Float;
  Desc->setDepthAttachmentPixelFormat(DepthStencilFormat);
  if (HasStencil)
    Desc->setStencilAttachmentPixelFormat(DepthStencilFormat);
  auto PSO = NS::TransferPtr(Device->newRenderPipelineState(Desc.get(), &NErr));
  if (!PSO)
    return mtlError("render pipeline state creation failed", NErr);
  auto *TexDesc = MTL::TextureDescriptor::texture2DDescriptor(*PF, Width, Height, false);
  TexDesc->setStorageMode(MTL::StorageModeShared);
  TexDesc->setUsage(MTL::TextureUsageRenderTarget);
  auto Target = NS::TransferPtr(Device->newTexture(TexDesc));
  NS::SharedPtr<MTL::Texture> MSAATarget;
  if (B.SampleCount > 1) {
    auto *MSAADesc = MTL::TextureDescriptor::texture2DDescriptor(*PF, Width, Height, false);
    MSAADesc->setTextureType(MTL::TextureType2DMultisample);
    MSAADesc->setSampleCount(B.SampleCount);
    MSAADesc->setStorageMode(MTL::StorageModePrivate);
    MSAADesc->setUsage(MTL::TextureUsageRenderTarget);
    MSAATarget = NS::TransferPtr(Device->newTexture(MSAADesc));
    if (!MSAATarget)
      return Fail("cannot allocate Metal MSAA render target");
  }
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
  std::vector<MTL::Viewport> VPs;
  for (const auto &VP : B.getViewports())
    VPs.push_back({VP.X, VP.Y, VP.Width, VP.Height, VP.MinDepth, VP.MaxDepth});
  std::vector<MTL::ScissorRect> Scs;
  for (const auto &Sc : B.getScissors()) {
    if (Sc.X < 0 || Sc.Y < 0 || static_cast<size_t>(Sc.X) + Sc.Width > Width ||
        static_cast<size_t>(Sc.Y) + Sc.Height > Height)
      return Fail("scissor exceeds the render target");
    Scs.push_back({static_cast<size_t>(Sc.X), static_cast<size_t>(Sc.Y),
                   Sc.Width, Sc.Height});
  }
  while (Scs.size() < VPs.size())
    Scs.push_back(Scs.empty() ? MTL::ScissorRect{0, 0, Width, Height} : Scs.back());
  auto *Pass = MTL::RenderPassDescriptor::renderPassDescriptor();
  NS::SharedPtr<MTL::Texture> DepthStencilTarget;
  NS::SharedPtr<MTL::DepthStencilState> DepthStencilState;
  {
    auto *DSDesc = MTL::TextureDescriptor::texture2DDescriptor(
        DepthStencilFormat, Width, Height, false);
    if (B.SampleCount > 1) {
      DSDesc->setTextureType(MTL::TextureType2DMultisample);
      DSDesc->setSampleCount(B.SampleCount);
    }
    DSDesc->setStorageMode(MTL::StorageModePrivate);
    DSDesc->setUsage(MTL::TextureUsageRenderTarget);
    DepthStencilTarget = NS::TransferPtr(Device->newTexture(DSDesc));
    auto DS = NS::TransferPtr(MTL::DepthStencilDescriptor::alloc()->init());
    DS->setDepthCompareFunction(MTL::CompareFunctionLessEqual);
    DS->setDepthWriteEnabled(true);
    if (HasStencil) {
      auto StencilDesc = NS::TransferPtr(MTL::StencilDescriptor::alloc()->init());
      StencilDesc->setStencilCompareFunction(metalCompare(P.Stencil->ComparisonOp));
      StencilDesc->setStencilFailureOperation(MTL::StencilOperationKeep);
      StencilDesc->setDepthFailureOperation(MTL::StencilOperationKeep);
      StencilDesc->setDepthStencilPassOperation(MTL::StencilOperationKeep);
      StencilDesc->setReadMask(0xff);
      StencilDesc->setWriteMask(0xff);
      DS->setFrontFaceStencil(StencilDesc.get());
      DS->setBackFaceStencil(StencilDesc.get());
    }
    DepthStencilState = NS::TransferPtr(Device->newDepthStencilState(DS.get()));
    if (!DepthStencilTarget || !DepthStencilState)
      return Fail("cannot allocate Metal depth/stencil target or state");
    auto *DepthAttachment = Pass->depthAttachment();
    DepthAttachment->setTexture(DepthStencilTarget.get());
    DepthAttachment->setLoadAction(MTL::LoadActionClear);
    DepthAttachment->setClearDepth(1.0);
    DepthAttachment->setStoreAction(MTL::StoreActionDontCare);
    if (HasStencil) {
      auto *StencilAttachment = Pass->stencilAttachment();
      StencilAttachment->setTexture(DepthStencilTarget.get());
      StencilAttachment->setLoadAction(MTL::LoadActionClear);
      StencilAttachment->setClearStencil(P.Stencil->ClearValue);
      StencilAttachment->setStoreAction(MTL::StoreActionDontCare);
    }
  }
  auto *Color = Pass->colorAttachments()->object(0);
  if (MSAATarget) {
    Color->setTexture(MSAATarget.get());
    Color->setResolveTexture(Target.get());
    Color->setStoreAction(MTL::StoreActionMultisampleResolve);
  } else {
    Color->setTexture(Target.get());
    Color->setStoreAction(MTL::StoreActionStore);
  }
  Color->setLoadAction(MTL::LoadActionClear);
  Color->setClearColor(MTL::ClearColor(0, 0, 0, 0));
  auto *Cmd = Queue->commandBuffer();
  if (!Cmd)
    return Fail("cannot allocate Metal render command buffer");
  auto *Enc = Cmd->renderCommandEncoder(Pass);
  if (!Enc)
    return Fail("cannot allocate Metal render command encoder");
  Enc->setRenderPipelineState(PSO.get());
  if (DepthStencilState)
    Enc->setDepthStencilState(DepthStencilState.get());
  if (HasStencil)
    Enc->setStencilReferenceValue(0);
  Enc->setFrontFacingWinding(MTL::WindingCounterClockwise);
  Enc->setViewports(VPs.data(), VPs.size());
  Enc->setScissorRects(Scs.data(), Scs.size());
  if (Vertices)
    Enc->setVertexBuffer(Vertices.get(), 0, VertexDataSlot);
  for (auto [Vertex, Slot, GPU] : BufferBindings)
    if (Vertex)
      Enc->setVertexBuffer(GPU, 0, Slot);
    else
      Enc->setFragmentBuffer(GPU, 0, Slot);
  for (auto [Vertex, Slot, Tex] : TextureBindings)
    if (Vertex)
      Enc->setVertexTexture(Tex, Slot);
    else
      Enc->setFragmentTexture(Tex, Slot);
  for (auto [Vertex, Slot, Samp] : SamplerBindings)
    if (Vertex)
      Enc->setVertexSamplerState(Samp, Slot);
    else
      Enc->setFragmentSamplerState(Samp, Slot);
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
  for (auto &[CPU, Tex] : RasterTextures)
    if (RasterTextureUsages.at(CPU) & MTL::TextureUsageShaderWrite)
      readbackMetalTexture(Tex.get(), CPU, RasterTextureKinds.at(CPU));
  return Error::success();
}

Error executeMeshRaster(Pipeline &P, MTL::Device *Device) {
  auto Fail = [](StringRef Msg) {
    return createStringError(inconvertibleErrorCode(), Msg);
  };
  if (!P.isMeshShaderRaster())
    return Fail("expected a mesh-shader raster pipeline");
  const auto &B = P.Bindings;
  if (!B.RTargetBufferPtr || B.RTargetBufferPtr->Data.empty())
    return Fail("mesh raster pipeline has no RenderTarget");
  CPUBuffer *Output = B.RTargetBufferPtr;
  const auto &Props = Output->OutputProps;
  size_t Width = Props.Width ? Props.Width : 1;
  size_t Height = Props.Height ? Props.Height : 1;
  size_t Depth = Props.Depth ? Props.Depth : 1;
  auto PF = pixelFormatFor(Output->Format, Output->Channels);
  if (!PF)
    return PF.takeError();
  size_t BPP = Output->getSingleElementSize() * Output->Channels;
  size_t Pitch = Width * BPP;
  if (!BPP || Depth != 1 || Output->Size < Pitch * Height)
    return Fail("unsupported mesh raster RenderTarget layout");
  bool HasStencil = P.Stencil && P.Stencil->Enabled;

  const Shader *AS = nullptr, *MS = nullptr, *PS = nullptr;
  for (const Shader &S : P.Shaders) {
    if (S.Stage == Stages::Amplification)
      AS = &S;
    else if (S.Stage == Stages::Mesh)
      MS = &S;
    else if (S.Stage == Stages::Pixel)
      PS = &S;
  }
  if (!MS || !PS)
    return Fail("mesh raster pipeline requires Mesh and Pixel shaders");

  auto CO = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
  CO->setFastMathEnabled(false);

  auto CompileShader = [&](const Shader &S)
      -> Expected<std::pair<NS::SharedPtr<MTL::Function>, crucible::Reflection>> {
    crucible::Options Opts;
    Opts.EmulateF64 = true;
    auto Compiled =
        crucible::compileDXILToMSL(S.Shader->getMemBufferRef(), Opts);
    if (!Compiled)
      return Compiled.takeError();
    NS::Error *NErr = nullptr;
    auto *Src =
        NS::String::string(Compiled->MSL.c_str(), NS::UTF8StringEncoding);
    auto Lib = NS::TransferPtr(Device->newLibrary(Src, CO.get(), &NErr));
    if (!Lib)
      return mtlError("MSL compilation failed", NErr);
    auto *FnName = NS::String::string(Compiled->Reflect.MSLEntryName.c_str(),
                                      NS::UTF8StringEncoding);
    auto Fn = NS::TransferPtr(Lib->newFunction(FnName));
    if (!Fn)
      return Fail("compiled Metal library has no entry point");
    return std::make_pair(std::move(Fn), std::move(Compiled->Reflect));
  };

  NS::SharedPtr<MTL::Function> ObjFn, MeshFn, FragFn;
  crucible::Reflection ObjRefl, MeshRefl, FragRefl;
  if (AS) {
    auto R = CompileShader(*AS);
    if (!R)
      return R.takeError();
    ObjFn = std::move(R->first);
    ObjRefl = std::move(R->second);
  }
  {
    auto R = CompileShader(*MS);
    if (!R)
      return R.takeError();
    MeshFn = std::move(R->first);
    MeshRefl = std::move(R->second);
  }
  {
    auto R = CompileShader(*PS);
    if (!R)
      return R.takeError();
    FragFn = std::move(R->first);
    FragRefl = std::move(R->second);
  }

  auto Desc = NS::TransferPtr(MTL::MeshRenderPipelineDescriptor::alloc()->init());
  if (ObjFn)
    Desc->setObjectFunction(ObjFn.get());
  Desc->setMeshFunction(MeshFn.get());
  Desc->setFragmentFunction(FragFn.get());
  Desc->colorAttachments()->object(0)->setPixelFormat(*PF);
  if (B.SampleCount > 1)
    Desc->setRasterSampleCount(B.SampleCount);
  MTL::PixelFormat DepthStencilFormat =
      HasStencil ? MTL::PixelFormatDepth32Float_Stencil8
                 : MTL::PixelFormatDepth32Float;
  Desc->setDepthAttachmentPixelFormat(DepthStencilFormat);
  if (HasStencil)
    Desc->setStencilAttachmentPixelFormat(DepthStencilFormat);

  NS::Error *NErr = nullptr;
  auto PSO = NS::TransferPtr(Device->newRenderPipelineState(
      Desc.get(), MTL::PipelineOptionNone, nullptr, &NErr));
  if (!PSO)
    return mtlError("mesh render pipeline state creation failed", NErr);

  auto *TexDesc =
      MTL::TextureDescriptor::texture2DDescriptor(*PF, Width, Height, false);
  TexDesc->setStorageMode(MTL::StorageModeShared);
  TexDesc->setUsage(MTL::TextureUsageRenderTarget);
  auto Target = NS::TransferPtr(Device->newTexture(TexDesc));
  NS::SharedPtr<MTL::Texture> MSAATarget;
  if (B.SampleCount > 1) {
    auto *MSAADesc =
        MTL::TextureDescriptor::texture2DDescriptor(*PF, Width, Height, false);
    MSAADesc->setTextureType(MTL::TextureType2DMultisample);
    MSAADesc->setSampleCount(B.SampleCount);
    MSAADesc->setStorageMode(MTL::StorageModePrivate);
    MSAADesc->setUsage(MTL::TextureUsageRenderTarget);
    MSAATarget = NS::TransferPtr(Device->newTexture(MSAADesc));
    if (!MSAATarget)
      return Fail("cannot allocate Metal MSAA render target");
  }
  auto Queue = NS::TransferPtr(Device->newCommandQueue());
  if (!Target || !Queue)
    return Fail("cannot allocate Metal render target or queue");

  std::vector<MTL::Viewport> VPs;
  for (const auto &VP : B.getViewports())
    VPs.push_back({VP.X, VP.Y, VP.Width, VP.Height, VP.MinDepth, VP.MaxDepth});
  std::vector<MTL::ScissorRect> Scs;
  for (const auto &Sc : B.getScissors()) {
    if (Sc.X < 0 || Sc.Y < 0 || static_cast<size_t>(Sc.X) + Sc.Width > Width ||
        static_cast<size_t>(Sc.Y) + Sc.Height > Height)
      return Fail("scissor exceeds the render target");
    Scs.push_back({static_cast<size_t>(Sc.X), static_cast<size_t>(Sc.Y),
                   Sc.Width, Sc.Height});
  }
  while (Scs.size() < VPs.size())
    Scs.push_back(Scs.empty() ? MTL::ScissorRect{0, 0, Width, Height} : Scs.back());

  auto *Pass = MTL::RenderPassDescriptor::renderPassDescriptor();
  auto *DSDesc = MTL::TextureDescriptor::texture2DDescriptor(
      DepthStencilFormat, Width, Height, false);
  if (B.SampleCount > 1) {
    DSDesc->setTextureType(MTL::TextureType2DMultisample);
    DSDesc->setSampleCount(B.SampleCount);
  }
  DSDesc->setStorageMode(MTL::StorageModePrivate);
  DSDesc->setUsage(MTL::TextureUsageRenderTarget);
  auto DepthStencilTarget = NS::TransferPtr(Device->newTexture(DSDesc));
  auto DS = NS::TransferPtr(MTL::DepthStencilDescriptor::alloc()->init());
  DS->setDepthCompareFunction(MTL::CompareFunctionLessEqual);
  DS->setDepthWriteEnabled(true);
  if (HasStencil) {
    auto StencilDesc = NS::TransferPtr(MTL::StencilDescriptor::alloc()->init());
    StencilDesc->setStencilCompareFunction(metalCompare(P.Stencil->ComparisonOp));
    StencilDesc->setStencilFailureOperation(MTL::StencilOperationKeep);
    StencilDesc->setDepthFailureOperation(MTL::StencilOperationKeep);
    StencilDesc->setDepthStencilPassOperation(MTL::StencilOperationKeep);
    StencilDesc->setReadMask(0xff);
    StencilDesc->setWriteMask(0xff);
    DS->setFrontFaceStencil(StencilDesc.get());
    DS->setBackFaceStencil(StencilDesc.get());
  }
  auto DepthStencilState =
      NS::TransferPtr(Device->newDepthStencilState(DS.get()));
  if (!DepthStencilTarget || !DepthStencilState)
    return Fail("cannot allocate Metal depth/stencil target or state");
  auto *DepthAttachment = Pass->depthAttachment();
  DepthAttachment->setTexture(DepthStencilTarget.get());
  DepthAttachment->setLoadAction(MTL::LoadActionClear);
  DepthAttachment->setClearDepth(1.0);
  DepthAttachment->setStoreAction(MTL::StoreActionDontCare);
  if (HasStencil) {
    auto *StencilAttachment = Pass->stencilAttachment();
    StencilAttachment->setTexture(DepthStencilTarget.get());
    StencilAttachment->setLoadAction(MTL::LoadActionClear);
    StencilAttachment->setClearStencil(P.Stencil->ClearValue);
    StencilAttachment->setStoreAction(MTL::StoreActionDontCare);
  }

  auto *Color = Pass->colorAttachments()->object(0);
  if (MSAATarget) {
    Color->setTexture(MSAATarget.get());
    Color->setResolveTexture(Target.get());
    Color->setStoreAction(MTL::StoreActionMultisampleResolve);
  } else {
    Color->setTexture(Target.get());
    Color->setStoreAction(MTL::StoreActionStore);
  }
  Color->setLoadAction(MTL::LoadActionClear);
  Color->setClearColor(MTL::ClearColor(0, 0, 0, 0));

  auto *Cmd = Queue->commandBuffer();
  if (!Cmd)
    return Fail("cannot allocate Metal render command buffer");
  auto *Enc = Cmd->renderCommandEncoder(Pass);
  if (!Enc)
    return Fail("cannot allocate Metal render command encoder");
  Enc->setRenderPipelineState(PSO.get());
  Enc->setDepthStencilState(DepthStencilState.get());
  if (HasStencil)
    Enc->setStencilReferenceValue(0);
  Enc->setFrontFacingWinding(MTL::WindingCounterClockwise);
  Enc->setViewports(VPs.data(), VPs.size());
  Enc->setScissorRects(Scs.data(), Scs.size());

  const auto &Grid = P.DispatchParameters.DispatchGroupCount;
  MTL::Size Threadgroups = MTL::Size::Make(
      Grid[0] ? Grid[0] : 1, Grid[1] ? Grid[1] : 1, Grid[2] ? Grid[2] : 1);
  MTL::Size ObjTG = ObjFn
      ? MTL::Size::Make(ObjRefl.NumThreads[0], ObjRefl.NumThreads[1],
                        ObjRefl.NumThreads[2])
      : MTL::Size::Make(1, 1, 1);
  MTL::Size MeshTG = MTL::Size::Make(
      MeshRefl.NumThreads[0], MeshRefl.NumThreads[1], MeshRefl.NumThreads[2]);
  Enc->drawMeshThreadgroups(Threadgroups, ObjTG, MeshTG);
  Enc->endEncoding();
  Cmd->commit();
  Cmd->waitUntilCompleted();
  if (Cmd->status() == MTL::CommandBufferStatusError)
    return mtlError("GPU mesh rendering failed", Cmd->error());
  Target->getBytes(Output->Data[0].get(), Pitch,
                   MTL::Region::Make2D(0, 0, Width, Height), 0);
  return Error::success();
}

Error executeCompute(Pipeline &P, MTL::Device *Device, bool UseVulkanBinding) {
  if (!P.isCompute() && !P.isRayTracing())
    return createStringError(
        inconvertibleErrorCode(),
        "only compute and vertex/pixel raster pipelines are supported");

  MTL::CommandQueue *Queue = Device->newCommandQueue();

  // One MTLBuffer / MTLTexture per (CPUBuffer, ArraySlice), shared across shaders.
  using BufKey = std::pair<CPUBuffer *, uint32_t>;
  std::map<BufKey, MTL::Buffer *> GPUBuffers;
  std::map<BufKey, MTL::Buffer *> GPUCounters;
  std::map<BufKey, MTL::Texture *> GPUTextures;
  std::map<CPUBuffer *, MTL::TextureUsage> TextureUsages;
  std::map<CPUBuffer *, ResourceKind> TextureKinds;
  std::map<YAMLSampler *, MTL::SamplerState *> GPUSamplers;
  std::vector<MTL::AccelerationStructure *> BuiltBLAS;
  std::map<std::pair<TLASDesc *, uint32_t>, MTL::AccelerationStructure *>
      BuiltTLAS;
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
  auto FindBinding = [UseVulkanBinding](const Resource &Res,
                                        const crucible::Reflection &R)
      -> const crucible::ResourceBinding * {
    auto Class = classify(Res.Kind);
    if (!Class)
      return nullptr;
    for (const crucible::ResourceBinding &RB : R.Resources)
      if (RB.Class == *Class && resourceBindingOffset(Res, RB, UseVulkanBinding))
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

  if (!P.AccelStructs.BLAS.empty() || !P.AccelStructs.TLAS.empty()) {
    MTL::CommandBuffer *ASBuf = Queue->commandBuffer();
    MTL::AccelerationStructureCommandEncoder *ASEnc =
        ASBuf->accelerationStructureCommandEncoder();
    auto BuildOneAS = [&](MTL::AccelerationStructureDescriptor *Desc)
        -> Expected<MTL::AccelerationStructure *> {
      MTL::AccelerationStructureSizes Sizes =
          Device->accelerationStructureSizes(Desc);
      MTL::AccelerationStructure *AS =
          Device->newAccelerationStructure(Sizes.accelerationStructureSize);
      MTL::Buffer *Scratch = Device->newBuffer(Sizes.buildScratchBufferSize,
                                               MTL::ResourceStorageModePrivate);
      if (!AS || !Scratch)
        return createStringError(inconvertibleErrorCode(),
                                 "failed to allocate acceleration structure");
      ASEnc->buildAccelerationStructure(AS, Desc, Scratch, 0);
      return AS;
    };

    for (const BLASDesc &BD : P.AccelStructs.BLAS) {
      std::vector<NS::Object *> GeomDescs;
      for (const TriangleGeometry &TG : BD.Triangles) {
        auto *GD = MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()
                       ->init();
        GD->setVertexBuffer(GetGPUBuffer(TG.VertexBufferPtr));
        GD->setVertexBufferOffset(0);
        GD->setVertexStride(TG.VertexStride);
        MTL::AttributeFormat VF = MTL::AttributeFormatFloat3;
        if (TG.VertexFormat == Format::RG32Float)
          VF = MTL::AttributeFormatFloat2;
        else if (TG.VertexFormat == Format::RGBA32Float)
          VF = MTL::AttributeFormatFloat4;
        GD->setVertexFormat(VF);
        if (TG.IndexBufferPtr) {
          GD->setIndexBuffer(GetGPUBuffer(TG.IndexBufferPtr));
          GD->setIndexType(TG.IdxFormat == IndexFormat::Uint16
                               ? MTL::IndexTypeUInt16
                               : MTL::IndexTypeUInt32);
          GD->setTriangleCount(TG.IndexCount / 3);
        } else {
          GD->setTriangleCount(TG.VertexCount / 3);
        }
        if (TG.Transform) {
          const auto &M = *TG.Transform;
          MTL::PackedFloat4x3 PM;
          PM.columns[0] = MTL::PackedFloat3(M[0], M[4], M[8]);
          PM.columns[1] = MTL::PackedFloat3(M[1], M[5], M[9]);
          PM.columns[2] = MTL::PackedFloat3(M[2], M[6], M[10]);
          PM.columns[3] = MTL::PackedFloat3(M[3], M[7], M[11]);
          MTL::Buffer *TB =
              Device->newBuffer(&PM, sizeof(PM), MTL::ResourceStorageModeShared);
          GD->setTransformationMatrixBuffer(TB);
        }
        GD->setOpaque(TG.Opaque);
        GeomDescs.push_back(GD);
      }
      for (const AABBGeometry &AG : BD.AABBs) {
        auto *GD =
            MTL::AccelerationStructureBoundingBoxGeometryDescriptor::alloc()
                ->init();
        GD->setBoundingBoxBuffer(GetGPUBuffer(AG.AABBBufferPtr));
        GD->setBoundingBoxStride(AG.AABBStride);
        GD->setBoundingBoxCount(AG.AABBCount);
        GD->setOpaque(AG.Opaque);
        GeomDescs.push_back(GD);
      }
      auto *PDesc = MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
      PDesc->setGeometryDescriptors(
          NS::Array::array(GeomDescs.data(), GeomDescs.size()));
      auto AS = BuildOneAS(PDesc);
      PDesc->release();
      if (!AS)
        return AS.takeError();
      BuiltBLAS.push_back(*AS);
    }

    NS::Array *BLASArray = NS::Array::array(
        reinterpret_cast<const NS::Object *const *>(BuiltBLAS.data()),
        BuiltBLAS.size());
    for (TLASDesc &TD : P.AccelStructs.TLAS) {
      for (uint32_t ArrIdx = 0; ArrIdx < TD.ArraySize; ++ArrIdx) {
        const auto &Insts = TD.Instances[ArrIdx];
        std::vector<MTL::AccelerationStructureUserIDInstanceDescriptor> Descs(
            Insts.size());
        for (size_t I = 0; I < Insts.size(); ++I) {
          const InstanceDesc &ID = Insts[I];
          auto &D = Descs[I];
          const float *M = ID.Transform;
          D.transformationMatrix.columns[0] = MTL::PackedFloat3(M[0], M[4], M[8]);
          D.transformationMatrix.columns[1] = MTL::PackedFloat3(M[1], M[5], M[9]);
          D.transformationMatrix.columns[2] = MTL::PackedFloat3(M[2], M[6], M[10]);
          D.transformationMatrix.columns[3] = MTL::PackedFloat3(M[3], M[7], M[11]);
          uint32_t Opts = MTL::AccelerationStructureInstanceOptionNone;
          if (ID.Flags & TriangleCullDisable)
            Opts |= MTL::AccelerationStructureInstanceOptionDisableTriangleCulling;
          if (ID.Flags & TriangleFrontCounterclockwise)
            Opts |= MTL::AccelerationStructureInstanceOptionTriangleFrontFacingWindingCounterClockwise;
          if (ID.Flags & ForceOpaque)
            Opts |= MTL::AccelerationStructureInstanceOptionOpaque;
          if (ID.Flags & ForceNonOpaque)
            Opts |= MTL::AccelerationStructureInstanceOptionNonOpaque;
          D.options = static_cast<MTL::AccelerationStructureInstanceOptions>(Opts);
          D.mask = ID.InstanceMask;
          D.intersectionFunctionTableOffset =
              ID.InstanceContributionToHitGroupIndex;
          D.accelerationStructureIndex = static_cast<uint32_t>(ID.BLASIdx);
          D.userID = ID.InstanceID;
        }
        MTL::Buffer *InstBuf = Device->newBuffer(
            Descs.data(),
            std::max<size_t>(
                Descs.size() *
                    sizeof(MTL::AccelerationStructureUserIDInstanceDescriptor),
                16),
            MTL::ResourceStorageModeShared);
        auto *IDesc =
            MTL::InstanceAccelerationStructureDescriptor::alloc()->init();
        IDesc->setInstanceDescriptorType(
            MTL::AccelerationStructureInstanceDescriptorTypeUserID);
        IDesc->setInstanceDescriptorBuffer(InstBuf);
        IDesc->setInstanceCount(Insts.size());
        IDesc->setInstancedAccelerationStructures(BLASArray);
        auto TLAS = BuildOneAS(IDesc);
        IDesc->release();
        if (!TLAS)
          return TLAS.takeError();
        BuiltTLAS[{&TD, ArrIdx}] = *TLAS;
      }
    }
    ASEnc->endEncoding();
    ASBuf->commit();
    ASBuf->waitUntilCompleted();
    if (ASBuf->status() == MTL::CommandBufferStatusError)
      return mtlError("acceleration structure build failed", ASBuf->error());
  }
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
    if (isRayTracingStage(S.Stage) && S.Stage != Stages::RayGeneration)
      continue;
    if (S.Stage != Stages::Compute && S.Stage != Stages::RayGeneration)
      return createStringError(inconvertibleErrorCode(),
                               "non-compute shader stage in compute pipeline");

    crucible::Options Opts;
    Opts.EntryPoint = S.Entry;
    if (S.Stage == Stages::RayGeneration && P.SBT) {
      for (const SBTEntry &ME : P.SBT->Miss)
        Opts.MissShaders.push_back(ME.ShaderName);
      for (const SBTEntry &HE : P.SBT->HitGroup) {
        std::string CH;
        for (const HitGroup &HG : P.HitGroups) {
          if (HG.Name == HE.ShaderName) {
            CH = HG.ClosestHit;
            break;
          }
        }
        Opts.HitGroupClosestHits.push_back(CH);
      }
    }
    Opts.FlushDenormsF32 = true;
    Opts.BoundsCheckBuffers = true;
    Opts.EmulateF64 = true;
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
    SmallVector<std::pair<uint32_t, MTL::AccelerationStructure *>, 4> ASBinds;
    std::vector<uint32_t> BufferDims;
    if (R.BufferSizesMetalIndex != UINT32_MAX)
      BufferDims.resize(R.BufferSizesMetalIndex + 1, 0);
    for (DescriptorSet &Set : P.Sets) {
      for (Resource &Res : Set.Resources) {
        if (Res.HeapIndex)
          continue;
        auto Class = classify(Res.Kind);
        if (!Class)
          return createStringError(
              inconvertibleErrorCode(),
              "resource '%s': kind is not supported yet (accel)",
              Res.Name.c_str());
        const auto *Match = FindBinding(Res, R);
        if (!Match)
          continue; // not referenced by this shader
        auto RegOffsetOrErr = resourceBindingOffset(Res, *Match, UseVulkanBinding);
        if (!RegOffsetOrErr)
          continue;
        uint32_t RegOffset = *RegOffsetOrErr;
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
        if (Res.isAccelerationStructure()) {
          if (!Res.TLASPtr)
            return createStringError(inconvertibleErrorCode(),
                                     "acceleration structure '%s' has no TLAS",
                                     Res.Name.c_str());
          for (uint32_t I = 0; I < Res.getArraySize(); ++I)
            ASBinds.push_back({BaseSlot + I, BuiltTLAS.at({Res.TLASPtr, I})});
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
              size_t EffectiveBytes = Res.BufferPtr->Size;
              if (Res.IsReserved)
                EffectiveBytes = std::min(
                    EffectiveBytes,
                    static_cast<size_t>(Res.TilesMapped.value_or(0)) * 65536u);
              uint32_t ElemCount = 0;
              if (Match->Kind == crucible::ResourceKind::StructuredBuffer &&
                  Match->ElementStride > 0)
                ElemCount = static_cast<uint32_t>(EffectiveBytes /
                                                  Match->ElementStride);
              else if (Match->Kind == crucible::ResourceKind::RawBuffer)
                ElemCount = static_cast<uint32_t>(EffectiveBytes);
              else if (Match->Kind == crucible::ResourceKind::TypedBuffer) {
                uint32_t ElemBytes = Res.BufferPtr->getSingleElementSize() *
                                     Res.BufferPtr->Channels;
                if (ElemBytes > 0)
                  ElemCount =
                      static_cast<uint32_t>(EffectiveBytes / ElemBytes);
              }
              BufferDims[BaseSlot + I] = ElemCount;
            }
          }
        }
      }
    }
    // Resolve root parameters (root constants and root descriptors) before
    // the buffer-sizes buffer is created so their element counts participate
    // in bounds checking, exactly like descriptor-set bindings.
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
      auto BindRootBuffer = [&](const crucible::ResourceBinding *Match,
                                uint32_t Reg, CPUBuffer *Buf) {
        uint32_t Slot = Match->MetalIndex + (Reg - Match->LowerBound);
        BufferBinds.push_back({Slot, GetGPUBuffer(Buf)});
        if (R.BufferSizesMetalIndex != UINT32_MAX &&
            Slot < BufferDims.size()) {
          uint32_t ElemCount = 0;
          if (Match->Kind == crucible::ResourceKind::StructuredBuffer &&
              Match->ElementStride > 0)
            ElemCount =
                static_cast<uint32_t>(Buf->Size / Match->ElementStride);
          else if (Match->Kind == crucible::ResourceKind::RawBuffer)
            ElemCount = static_cast<uint32_t>(Buf->Size);
          else if (Match->Kind == crucible::ResourceKind::TypedBuffer) {
            uint32_t ElemBytes =
                Buf->getSingleElementSize() * Buf->Channels;
            if (ElemBytes > 0)
              ElemCount = static_cast<uint32_t>(Buf->Size / ElemBytes);
          }
          BufferDims[Slot] = ElemCount;
        }
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
            BindRootBuffer(Match, Info->ShaderRegister, RC.BufferPtr);
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
            BindRootBuffer(Match, Info->ShaderRegister, RR.BufferPtr);
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
    if (R.InstanceContributionMetalIndex != UINT32_MAX) {
      std::vector<uint32_t> Contribs;
      if (!P.AccelStructs.TLAS.empty() &&
          !P.AccelStructs.TLAS[0].Instances.empty()) {
        for (const InstanceDesc &ID : P.AccelStructs.TLAS[0].Instances[0])
          Contribs.push_back(ID.InstanceContributionToHitGroupIndex & 0xFFFFFFu);
      }
      MTL::Buffer *ContribBuf = Device->newBuffer(
          Contribs.data(),
          std::max<size_t>(Contribs.size() * sizeof(uint32_t), 16),
          MTL::ResourceStorageModeShared);
      BufferBinds.push_back({R.InstanceContributionMetalIndex, ContribBuf});
    }
    std::vector<std::pair<MTL::Resource *, MTL::ResourceUsage>> HeapUseRes;
    if (R.ResourceHeapMetalIndex != UINT32_MAX ||
        R.CounterHeapMetalIndex != UINT32_MAX ||
        R.SamplerHeapMetalIndex != UINT32_MAX) {
      std::vector<uint64_t> ResHeapAddrs, CounterHeapAddrs, SamplerHeapAddrs;
      for (const DescriptorSet &DS : P.Sets) {
        for (const offloadtest::Resource &Res : DS.Resources) {
          if (!Res.HeapIndex)
            continue;
          uint32_t Idx = *Res.HeapIndex;
          if (Res.Kind == offloadtest::ResourceKind::Sampler) {
            if (SamplerHeapAddrs.size() <= Idx)
              SamplerHeapAddrs.resize(Idx + 1, 0);
            SamplerHeapAddrs[Idx] =
                GetGPUSampler(Res.SamplerPtr)->gpuResourceID()._impl;
          } else if (isTextureResource(Res.Kind)) {
            TextureKinds[Res.BufferPtr] = Res.Kind;
            TextureUsages[Res.BufferPtr] |= Res.isReadWrite()
                ? (MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite)
                : MTL::TextureUsageShaderRead;
            auto Tex = GetGPUTexture(Res.BufferPtr, 0);
            if (!Tex)
              return Tex.takeError();
            if (ResHeapAddrs.size() <= Idx)
              ResHeapAddrs.resize(Idx + 1, 0);
            ResHeapAddrs[Idx] = (*Tex)->gpuResourceID()._impl;
            HeapUseRes.push_back(
                {*Tex, Res.isReadWrite()
                           ? (MTL::ResourceUsageRead | MTL::ResourceUsageWrite)
                           : MTL::ResourceUsageRead});
          } else if (Res.BufferPtr) {
            MTL::Buffer *Buf = GetGPUBuffer(Res.BufferPtr, 0);
            if (ResHeapAddrs.size() <= Idx)
              ResHeapAddrs.resize(Idx + 1, 0);
            ResHeapAddrs[Idx] = Buf->gpuAddress();
            HeapUseRes.push_back(
                {Buf, Res.isReadWrite()
                          ? (MTL::ResourceUsageRead | MTL::ResourceUsageWrite)
                          : MTL::ResourceUsageRead});
            if (Res.HasCounter) {
              MTL::Buffer *CBuf = GetGPUCounter(Res.BufferPtr, 0);
              if (CounterHeapAddrs.size() <= Idx)
                CounterHeapAddrs.resize(Idx + 1, 0);
              CounterHeapAddrs[Idx] = CBuf->gpuAddress();
              HeapUseRes.push_back(
                  {CBuf, MTL::ResourceUsageRead | MTL::ResourceUsageWrite});
            }
          }
        }
      }
      if (R.ResourceHeapMetalIndex != UINT32_MAX) {
        MTL::Buffer *HBuf = Device->newBuffer(
            ResHeapAddrs.data(),
            std::max<size_t>(ResHeapAddrs.size() * sizeof(uint64_t), 16),
            MTL::ResourceStorageModeShared);
        BufferBinds.push_back({R.ResourceHeapMetalIndex, HBuf});
      }
      if (R.CounterHeapMetalIndex != UINT32_MAX) {
        MTL::Buffer *CBuf = Device->newBuffer(
            CounterHeapAddrs.data(),
            std::max<size_t>(CounterHeapAddrs.size() * sizeof(uint64_t), 16),
            MTL::ResourceStorageModeShared);
        BufferBinds.push_back({R.CounterHeapMetalIndex, CBuf});
      }
      if (R.SamplerHeapMetalIndex != UINT32_MAX) {
        MTL::Buffer *SBuf = Device->newBuffer(
            SamplerHeapAddrs.data(),
            std::max<size_t>(SamplerHeapAddrs.size() * sizeof(uint64_t), 16),
            MTL::ResourceStorageModeShared);
        BufferBinds.push_back({R.SamplerHeapMetalIndex, SBuf});
      }
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
    for (auto &[Index, AS] : ASBinds)
      Enc->setAccelerationStructure(AS, Index);
    if (!ASBinds.empty())
      for (MTL::AccelerationStructure *BLAS : BuiltBLAS)
        Enc->useResource(BLAS, MTL::ResourceUsageRead);
    for (auto &[ResPtr, Usage] : HeapUseRes)
      Enc->useResource(ResPtr, Usage);
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

// The suite tools use executeProgram; the lower-level device API is not
// implemented by this provider.
using crucible_offload::CapabilityValueString;
using crucible_offload::validateRootParameters;

class CrucibleDevice final : public offloadtest::Device {
  NS::SharedPtr<MTL::Device> Metal;
  Capabilities Caps;
  GPUAPI API;
  std::string Extra;

  static Error unsupported() {
    return createStringError(std::errc::not_supported,
                             "Crucible supports only pipeline execution");
  }

public:
  explicit CrucibleDevice(NS::SharedPtr<MTL::Device> Metal, GPUAPI API,
                          offloadtest::Device *VulkanInfo)
      : Metal(std::move(Metal)), API(API) {
    Description = this->Metal->name()->utf8String();
    DriverName = "Crucible";
    const auto Version =
        NS::ProcessInfo::processInfo()->operatingSystemVersion();
    DriverVersion = formatv("{0}.{1}.{2}", Version.majorVersion,
                            Version.minorVersion, Version.patchVersion);
    if (StringRef(Description).starts_with("Apple "))
      GPUGeneration = StringRef(Description).split(' ').second.split(' ').first.str();
    else
      GPUGeneration = "Unknown";
    auto MakeStr = [](StringRef Name, std::string V) {
      return Capability(
          Name, std::make_shared<CapabilityValueString>(std::move(V)));
    };
    Caps.try_emplace("WaveOps", makeCapability("WaveOps", true));
    Caps.try_emplace("WaveLaneCountMin",
                     makeCapability("WaveLaneCountMin", uint32_t(32)));
    Caps.try_emplace("WaveLaneCountMax",
                     makeCapability("WaveLaneCountMax", uint32_t(32)));
    Caps.try_emplace("TotalLaneCount",
                     makeCapability("TotalLaneCount", uint32_t(32)));
    Caps.try_emplace("HighestShaderModel",
                     MakeStr("HighestShaderModel", "6.9"));
    Caps.try_emplace("Native16BitShaderOpsSupported",
                     makeCapability("Native16BitShaderOpsSupported", true));
    Caps.try_emplace("Int64ShaderOps", makeCapability("Int64ShaderOps", true));
    Caps.try_emplace("DoublePrecisionFloatShaderOps",
                     makeCapability("DoublePrecisionFloatShaderOps", true));
    Caps.try_emplace("AtomicInt64OnGroupSharedSupported",
                     makeCapability("AtomicInt64OnGroupSharedSupported", false));
    Caps.try_emplace("AtomicInt64OnTypedResourceSupported",
                     makeCapability("AtomicInt64OnTypedResourceSupported", false));
    Caps.try_emplace("ResourceBindingTier",
                     makeCapability("ResourceBindingTier", uint32_t(3)));
    Caps.try_emplace("TiledResourcesTier",
                     makeCapability("TiledResourcesTier", uint32_t(2)));
    Caps.try_emplace("MeshShaderTier",
                     MakeStr("MeshShaderTier",
                             this->Metal->supportsFamily(MTL::GPUFamilyMetal3)
                                 ? "Tier1"
                                 : "NotSupported"));
    Caps.try_emplace("BarycentricsSupported",
                     makeCapability("BarycentricsSupported", false));
    Caps.try_emplace("PSSpecifiedStencilRefSupported",
                     makeCapability("PSSpecifiedStencilRefSupported", true));
    Caps.try_emplace(
        "VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation",
        makeCapability(
            "VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation",
            true));
    Caps.try_emplace("RaytracingTier",
                     MakeStr("RaytracingTier", this->Metal->supportsRaytracing()
                                                   ? "1.1"
                                                   : "NotSupported"));
    if (API == GPUAPI::Vulkan) {
      if (VulkanInfo) {
        Description = VulkanInfo->getDescription();
        GPUGeneration = VulkanInfo->getGPUGeneration();
        FamilyPrefix = VulkanInfo->getFamilyPrefix();
        Caps = VulkanInfo->getCapabilities();
        raw_string_ostream OS(Extra);
        VulkanInfo->printExtra(OS);
        OS.flush();
      } else {
        // Populate Vulkan feature capabilities from Metal when no native
        // Vulkan ICD is present (e.g. macOS CI runners without MoltenVK).
        Caps.clear();
        Caps.try_emplace("shaderFloat16", makeCapability("shaderFloat16", true));
        Caps.try_emplace("shaderInt16", makeCapability("shaderInt16", true));
        Caps.try_emplace("shaderInt64", makeCapability("shaderInt64", true));
        Caps.try_emplace("shaderFloat64", makeCapability("shaderFloat64", true));
        Caps.try_emplace("shaderResourceMinLod",
                         makeCapability("shaderResourceMinLod", true));
        Caps.try_emplace("runtimeDescriptorArray",
                         makeCapability("runtimeDescriptorArray", true));
        Caps.try_emplace("descriptorBindingPartiallyBound",
                         makeCapability("descriptorBindingPartiallyBound", true));
        Caps.try_emplace("multiViewport", makeCapability("multiViewport", true));
        Caps.try_emplace("shaderOutputViewportIndex",
                         makeCapability("shaderOutputViewportIndex", true));
      }
    }
  }

  const Capabilities &getCapabilities() override { return Caps; }
  StringRef getAPIName() const override {
    return API == GPUAPI::Vulkan ? "Vulkan" : "DirectX";
  }
  GPUAPI getAPI() const override { return API; }
  void printExtra(raw_ostream &OS) override {
    if (API == GPUAPI::Vulkan && !Extra.empty()) {
      OS << Extra;
      return;
    }
    OS << "  Backend: Metal\n"
       << "  RegistryID: " << Metal->registryID() << '\n'
       << "  UnifiedMemory: " << (Metal->hasUnifiedMemory() ? "true" : "false")
       << '\n';
  }
  Error executeProgram(Pipeline &P) override {
    auto Pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    if (auto Err = validateRootParameters(P))
      return Err;
    bool UseVulkanBinding = (API == GPUAPI::Vulkan);
    return P.isTraditionalRaster()
               ? executeRaster(P, Metal.get(), UseVulkanBinding)
           : P.isMeshShaderRaster()
               ? executeMeshRaster(P, Metal.get())
               : executeCompute(P, Metal.get(), UseVulkanBinding);
  }

  Queue &getGraphicsQueue() override {
    report_fatal_error("Crucible does not implement the low-level graphics queue");
  }
  Expected<std::unique_ptr<PipelineState>>
  createPipelineCs(StringRef, const BindingsDesc &, ShaderContainer) override {
    return unsupported();
  }
  Expected<std::unique_ptr<PipelineState>> createTraditionalRasterPipeline(
      StringRef, const BindingsDesc &,
      const TraditionalRasterPipelineCreateDesc &) override {
    return unsupported();
  }
  Expected<std::unique_ptr<PipelineState>> createMeshShaderRasterPipeline(
      StringRef, const BindingsDesc &,
      const MeshShaderRasterPipelineCreateDesc &) override {
    return unsupported();
  }
  Expected<std::unique_ptr<PipelineState>> createPipelineRT(
      StringRef, const BindingsDesc &,
      const RayTracingPipelineCreateDesc &) override {
    return unsupported();
  }
  Expected<std::unique_ptr<ShaderBindingTable>> createShaderBindingTable(
      const PipelineState &, const ShaderBindingTableDesc &) override {
    return unsupported();
  }
  Expected<std::unique_ptr<Fence>> createFence(StringRef) override {
    return unsupported();
  }
  Expected<std::unique_ptr<Buffer>>
  createBuffer(std::string, const BufferCreateDesc &, size_t) override {
    return unsupported();
  }
  Expected<std::unique_ptr<Texture>>
  createTexture(std::string, const TextureCreateDesc &) override {
    return unsupported();
  }
  Expected<std::unique_ptr<Sampler>>
  createSampler(std::string, const SamplerCreateDesc &) override {
    return unsupported();
  }
  Expected<std::unique_ptr<MemoryHeap>>
  createMemoryHeap(std::string, size_t) override {
    return unsupported();
  }
  uint32_t getTextureUploadRowStrideInBytes(
      const TextureCreateDesc &) const override {
    report_fatal_error("Crucible does not implement low-level texture upload");
  }
  TextureUploadLayout getTextureUploadLayout(
      const TextureCreateDesc &) const override {
    report_fatal_error("Crucible does not implement low-level texture upload");
  }
  Expected<std::unique_ptr<RenderPass>>
  createRenderPass(const RenderPassDesc &) override { return unsupported(); }
  Expected<std::unique_ptr<CommandBuffer>> createCommandBuffer() override {
    return unsupported();
  }
  Expected<AccelerationStructureSizes>
  getBLASBuildSizes(ArrayRef<TriangleGeometryDesc>) override {
    return unsupported();
  }
  Expected<AccelerationStructureSizes>
  getBLASBuildSizes(ArrayRef<AABBGeometryDesc>) override {
    return unsupported();
  }
  Expected<AccelerationStructureSizes> getTLASBuildSizes(uint32_t) override {
    return unsupported();
  }
  Expected<std::unique_ptr<AccelerationStructure>>
  createBLAS(const AccelerationStructureSizes &) override { return unsupported(); }
  Expected<std::unique_ptr<AccelerationStructure>>
  createTLAS(const AccelerationStructureSizes &) override { return unsupported(); }
};

} // namespace

namespace crucible_offload {

Expected<SmallVector<std::unique_ptr<offloadtest::Device>>>
createMetalDevices(const DeviceConfig &Config,
                   offloadtest::Device *VulkanInfo) {
  if ((Config.EnableValidationLayer || Config.EnableDebugLayer) &&
      setenv("MTL_DEBUG_LAYER", "1", 1) != 0)
    return errorCodeToError(std::error_code(errno, std::generic_category()));
  auto Pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto Metal = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!Metal)
    return createStringError(std::errc::no_such_device,
                             "no Metal device available for Crucible");
  SmallVector<std::unique_ptr<offloadtest::Device>> Devices;
  Devices.push_back(
      std::make_unique<CrucibleDevice>(Metal, GPUAPI::DirectX, nullptr));
  Devices.push_back(std::make_unique<CrucibleDevice>(
      std::move(Metal), GPUAPI::Vulkan, VulkanInfo));
  return Devices;
}

} // namespace crucible_offload

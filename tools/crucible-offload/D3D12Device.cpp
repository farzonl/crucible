//===- D3D12Device.cpp - Crucible DirectX 12 offload proxy -------*- C++ -*-===//
//
// Part of Crucible, an open-source multi-frontend, multi-backend shader compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements offloadtest::Device with a Vulkan (SPIR-V) API personality backed
// by Crucible's SPIR-V -> DXIL compiler (LLVM 24 DirectX backend) proxying any
// underlying DirectX 12 device:
//   - On Linux / macOS: Nacelle (libnacelle_d3d12) or CRUCIBLE_D3D12_LIB
//   - On Windows: hardware D3D12 adapters (NVIDIA, AMD, Intel) and WARP via
//     d3d12.dll + dxgi.dll, or an explicit CRUCIBLE_D3D12_LIB override.
//
// Capabilities are probed directly from the underlying ID3D12Device via
// CheckFeatureSupport() and projected into Vulkan/SPIR-V capabilities.
//
//===----------------------------------------------------------------------===//

#include "Capabilities.h"
#include "Common.h"
#include "crucible/Compiler.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#include "../WindowsDXILSign.h"
#else
#include <dlfcn.h>
#include <wsl/winadapter.h>
#endif
#include <directx/d3d12.h>
#include <dxguids/dxguids.h>

#ifdef interface
#undef interface
#endif
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace llvm;
using namespace offloadtest;
using namespace crucible_offload;

namespace {

#ifdef _WIN32
using PFN_D3D12CreateDevice = HRESULT(WINAPI *)(IUnknown *, D3D_FEATURE_LEVEL,
                                                REFIID, void **);
using PFN_D3D12SerializeRootSignature =
    HRESULT(WINAPI *)(const D3D12_ROOT_SIGNATURE_DESC *,
                      D3D_ROOT_SIGNATURE_VERSION, ID3DBlob **, ID3DBlob **);
#else
using PFN_D3D12CreateDevice = HRESULT (*)(IUnknown *, D3D_FEATURE_LEVEL, REFIID,
                                          void **);
using PFN_D3D12SerializeRootSignature =
    HRESULT (*)(const D3D12_ROOT_SIGNATURE_DESC *, D3D_ROOT_SIGNATURE_VERSION,
                ID3DBlob **, ID3DBlob **);
#endif

struct D3D12Loader {
  void *Handle = nullptr;
  PFN_D3D12CreateDevice CreateDevice = nullptr;
  PFN_D3D12SerializeRootSignature SerializeRS = nullptr;
  std::string RuntimePath;
};

D3D12Loader &getD3D12Loader() {
  static D3D12Loader Loader;
  return Loader;
}

Error hrErr(HRESULT Hr, const char *What) {
  return createStringError(inconvertibleErrorCode(),
                           "D3D12 %s failed (HRESULT=0x%08x)", What,
                           static_cast<uint32_t>(Hr));
}

Error ensureD3D12Loaded() {
  D3D12Loader &L = getD3D12Loader();
  if (L.CreateDevice && L.SerializeRS)
    return Error::success();

  const char *LibPath = std::getenv("CRUCIBLE_D3D12_LIB");
  if (!LibPath || !*LibPath)
    LibPath = std::getenv("CRUCIBLE_NACELLE_LIB");
#ifdef _WIN32
  if (!LibPath || !*LibPath)
    LibPath = "d3d12.dll";
  HMODULE Mod = LoadLibraryA(LibPath);
  if (!Mod)
    return createStringError(inconvertibleErrorCode(),
                             "failed to LoadLibrary '%s'", LibPath);
  crucible_win_dxil::enableExperimentalShaderModels(Mod);
  L.Handle = Mod;
  L.RuntimePath = LibPath;
  L.CreateDevice = reinterpret_cast<PFN_D3D12CreateDevice>(
      GetProcAddress(Mod, "D3D12CreateDevice"));
  L.SerializeRS = reinterpret_cast<PFN_D3D12SerializeRootSignature>(
      GetProcAddress(Mod, "D3D12SerializeRootSignature"));
#else
  if (!LibPath || !*LibPath) {
#ifdef CRUCIBLE_NACELLE_LIB_PATH
    LibPath = CRUCIBLE_NACELLE_LIB_PATH;
#else
    return createStringError(
        inconvertibleErrorCode(),
        "CRUCIBLE_D3D12_LIB not set and no default D3D12 runtime configured");
#endif
  }
  L.Handle = dlopen(LibPath, RTLD_NOW | RTLD_LOCAL);
  if (!L.Handle)
    return createStringError(inconvertibleErrorCode(),
                             "failed to dlopen D3D12 runtime '%s': %s", LibPath,
                             dlerror() ? dlerror() : "unknown error");
  L.RuntimePath = LibPath;
  L.CreateDevice = reinterpret_cast<PFN_D3D12CreateDevice>(
      dlsym(L.Handle, "D3D12CreateDevice"));
  L.SerializeRS = reinterpret_cast<PFN_D3D12SerializeRootSignature>(
      dlsym(L.Handle, "D3D12SerializeRootSignature"));
#endif

  if (!L.CreateDevice || !L.SerializeRS)
    return createStringError(
        inconvertibleErrorCode(),
        "D3D12 runtime '%s' missing D3D12CreateDevice / "
        "D3D12SerializeRootSignature",
        L.RuntimePath.c_str());

  return Error::success();
}

std::string shaderModelToString(D3D_SHADER_MODEL SM) {
  switch (SM) {
  case D3D_SHADER_MODEL_6_8: return "6.8";
  case D3D_SHADER_MODEL_6_7: return "6.7";
  case D3D_SHADER_MODEL_6_6: return "6.6";
  case D3D_SHADER_MODEL_6_5: return "6.5";
  case D3D_SHADER_MODEL_6_4: return "6.4";
  case D3D_SHADER_MODEL_6_3: return "6.3";
  case D3D_SHADER_MODEL_6_2: return "6.2";
  case D3D_SHADER_MODEL_6_1: return "6.1";
  case D3D_SHADER_MODEL_6_0:
  default:
    return "6.0";
  }
}

D3D12BackendCaps probeD3D12Capabilities(ID3D12Device *Dev,
                                        StringRef AdapterName,
                                        uint32_t VendorID, uint32_t DeviceID) {
  D3D12BackendCaps Caps;
  Caps.AdapterName = AdapterName.str();
  Caps.VendorID = VendorID;
  Caps.DeviceID = DeviceID;

  D3D12_FEATURE_DATA_D3D12_OPTIONS Opts0{};
  if (SUCCEEDED(Dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &Opts0,
                                         sizeof(Opts0)))) {
    Caps.DoublePrecisionFloatShaderOps =
        Opts0.DoublePrecisionFloatShaderOps != 0;
    Caps.ResourceBindingTier = static_cast<uint32_t>(Opts0.ResourceBindingTier);
    Caps.TiledResourcesTier = static_cast<uint32_t>(Opts0.TiledResourcesTier);
  }

  D3D12_FEATURE_DATA_D3D12_OPTIONS1 Opts1{};
  if (SUCCEEDED(Dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &Opts1,
                                         sizeof(Opts1)))) {
    Caps.WaveOps = Opts1.WaveOps != 0;
    Caps.WaveLaneCountMin = Opts1.WaveLaneCountMin;
    Caps.WaveLaneCountMax = Opts1.WaveLaneCountMax;
    Caps.Int64ShaderOps = Opts1.Int64ShaderOps != 0;
  }

  D3D12_FEATURE_DATA_D3D12_OPTIONS4 Opts4{};
  if (SUCCEEDED(Dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &Opts4,
                                         sizeof(Opts4)))) {
    Caps.Native16BitShaderOpsSupported =
        Opts4.Native16BitShaderOpsSupported != 0;
  }

  D3D12_FEATURE_DATA_D3D12_OPTIONS9 Opts9{};
  if (SUCCEEDED(Dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS9, &Opts9,
                                         sizeof(Opts9)))) {
    Caps.AtomicInt64OnTypedResourceSupported =
        Opts9.AtomicInt64OnTypedResourceSupported != 0;
    Caps.AtomicInt64OnGroupSharedSupported =
        Opts9.AtomicInt64OnGroupSharedSupported != 0;
  }

  static constexpr D3D_SHADER_MODEL Models[] = {
      D3D_SHADER_MODEL_6_8, D3D_SHADER_MODEL_6_7, D3D_SHADER_MODEL_6_6,
      D3D_SHADER_MODEL_6_5, D3D_SHADER_MODEL_6_4, D3D_SHADER_MODEL_6_3,
      D3D_SHADER_MODEL_6_2, D3D_SHADER_MODEL_6_1, D3D_SHADER_MODEL_6_0,
  };
  for (D3D_SHADER_MODEL Candidate : Models) {
    D3D12_FEATURE_DATA_SHADER_MODEL SMData{Candidate};
    if (SUCCEEDED(Dev->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &SMData,
                                           sizeof(SMData)))) {
      Caps.HighestShaderModel = shaderModelToString(SMData.HighestShaderModel);
      break;
    }
  }

  return Caps;
}

DXGI_FORMAT dxgiFormatFor(DataFormat Format, int Channels) {
  unsigned Ch = Channels ? static_cast<unsigned>(Channels) : 1u;
  switch (Format) {
  case DataFormat::Float32:
    if (Ch == 1) return DXGI_FORMAT_R32_FLOAT;
    if (Ch == 2) return DXGI_FORMAT_R32G32_FLOAT;
    if (Ch == 3) return DXGI_FORMAT_R32G32B32_FLOAT;
    return DXGI_FORMAT_R32G32B32A32_FLOAT;
  case DataFormat::Int32:
    if (Ch == 1) return DXGI_FORMAT_R32_SINT;
    if (Ch == 2) return DXGI_FORMAT_R32G32_SINT;
    if (Ch == 3) return DXGI_FORMAT_R32G32B32_SINT;
    return DXGI_FORMAT_R32G32B32A32_SINT;
  case DataFormat::Hex32:
  case DataFormat::UInt32:
  case DataFormat::Bool:
    if (Ch == 1) return DXGI_FORMAT_R32_UINT;
    if (Ch == 2) return DXGI_FORMAT_R32G32_UINT;
    if (Ch == 3) return DXGI_FORMAT_R32G32B32_UINT;
    return DXGI_FORMAT_R32G32B32A32_UINT;
  case DataFormat::Float16:
    if (Ch == 1) return DXGI_FORMAT_R16_FLOAT;
    if (Ch == 2) return DXGI_FORMAT_R16G16_FLOAT;
    return DXGI_FORMAT_R16G16B16A16_FLOAT;
  case DataFormat::Int16:
    if (Ch == 1) return DXGI_FORMAT_R16_SINT;
    if (Ch == 2) return DXGI_FORMAT_R16G16_SINT;
    return DXGI_FORMAT_R16G16B16A16_SINT;
  case DataFormat::Hex16:
  case DataFormat::UInt16:
    if (Ch == 1) return DXGI_FORMAT_R16_UINT;
    if (Ch == 2) return DXGI_FORMAT_R16G16_UINT;
    return DXGI_FORMAT_R16G16B16A16_UINT;
  case DataFormat::Hex8:
    if (Ch == 1) return DXGI_FORMAT_R8_UINT;
    if (Ch == 2) return DXGI_FORMAT_R8G8_UINT;
    return DXGI_FORMAT_R8G8B8A8_UINT;
  case DataFormat::Int64:
  case DataFormat::UInt64:
  case DataFormat::Hex64:
  case DataFormat::Float64:
    if (Ch == 1) return DXGI_FORMAT_R32G32_UINT;
    return DXGI_FORMAT_R32G32B32A32_UINT;
  default:
    return DXGI_FORMAT_R32_UINT;
  }
}

DXGI_FORMAT dxgiFormatForBuffer(const CPUBuffer &Buf) {
  return dxgiFormatFor(Buf.Format, Buf.Channels);
}

Expected<ID3D12Resource *> createCommittedBuffer(ID3D12Device *Dev,
                                                 D3D12_HEAP_TYPE HeapType,
                                                 uint64_t SizeBytes,
                                                 D3D12_RESOURCE_FLAGS Flags,
                                                 D3D12_RESOURCE_STATES State) {
  D3D12_HEAP_PROPERTIES Heap{};
  Heap.Type = HeapType;
  Heap.CreationNodeMask = 1;
  Heap.VisibleNodeMask = 1;

  D3D12_RESOURCE_DESC Desc{};
  Desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  Desc.Width = std::max<uint64_t>(SizeBytes, 4ull);
  Desc.Height = 1;
  Desc.DepthOrArraySize = 1;
  Desc.MipLevels = 1;
  Desc.Format = DXGI_FORMAT_UNKNOWN;
  Desc.SampleDesc = {1, 0};
  Desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  Desc.Flags = Flags;

  ID3D12Resource *Res = nullptr;
  HRESULT Hr = Dev->CreateCommittedResource(
      &Heap, D3D12_HEAP_FLAG_NONE, &Desc, State, nullptr,
      __uuidof(ID3D12Resource), reinterpret_cast<void **>(&Res));
  if (FAILED(Hr))
    return hrErr(Hr, "CreateCommittedResource");
  return Res;
}

D3D12_TEXTURE_ADDRESS_MODE d3d12AddressMode(AddressMode A) {
  switch (A) {
  case AddressMode::Clamp:
    return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
  case AddressMode::Repeat:
    return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
  case AddressMode::Mirror:
    return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
  case AddressMode::Border:
    return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
  case AddressMode::MirrorOnce:
    return D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
  }
  return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
}

D3D12_COMPARISON_FUNC d3d12CompareFunc(CompareFunction C) {
  switch (C) {
  case CompareFunction::Never:
    return D3D12_COMPARISON_FUNC_NEVER;
  case CompareFunction::Less:
    return D3D12_COMPARISON_FUNC_LESS;
  case CompareFunction::Equal:
    return D3D12_COMPARISON_FUNC_EQUAL;
  case CompareFunction::LessEqual:
    return D3D12_COMPARISON_FUNC_LESS_EQUAL;
  case CompareFunction::Greater:
    return D3D12_COMPARISON_FUNC_GREATER;
  case CompareFunction::NotEqual:
    return D3D12_COMPARISON_FUNC_NOT_EQUAL;
  case CompareFunction::GreaterEqual:
    return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
  case CompareFunction::Always:
    return D3D12_COMPARISON_FUNC_ALWAYS;
  }
  return D3D12_COMPARISON_FUNC_NEVER;
}

D3D12_FILTER d3d12Filter(const YAMLSampler &S) {
  bool MinLin = (S.MinFilter != FilterMode::Nearest);
  bool MagLin = (S.MagFilter != FilterMode::Nearest);
  bool MipLin = MinLin;
  bool Cmp = (S.Kind == SamplerKind::SamplerComparison);
  auto Type = Cmp ? D3D12_FILTER_REDUCTION_TYPE_COMPARISON
                  : D3D12_FILTER_REDUCTION_TYPE_STANDARD;
  return D3D12_ENCODE_BASIC_FILTER(
      MinLin ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT,
      MagLin ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT,
      MipLin ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT, Type);
}

struct BoundSampler {
  const offloadtest::Resource *Res = nullptr;
  uint32_t Space = 0;
  uint32_t Register = 0;
  bool IsUnbounded = false;
};

struct TexSubresourceLayout {
  uint32_t Subresource = 0;
  uint32_t Width = 1;
  uint32_t Height = 1;
  uint32_t Depth = 1;
  uint32_t RowPitch = 256;
  uint64_t SrcOffset = 0;
  uint64_t DstOffset = 0;
};

struct BoundResource {
  const offloadtest::Resource *Res = nullptr;
  uint32_t ArrIdx = 0;
  crucible::ResourceClass Class = crucible::ResourceClass::SRV;
  uint32_t Space = 0;
  uint32_t Register = 0;
  ID3D12Resource *DefaultBuf = nullptr;
  ID3D12Resource *UploadBuf = nullptr;
  ID3D12Resource *ReadbackBuf = nullptr;
  ID3D12Resource *CounterDefaultBuf = nullptr;
  ID3D12Resource *CounterUploadBuf = nullptr;
  ID3D12Resource *CounterReadbackBuf = nullptr;
  uint64_t SizeBytes = 0;
  uint64_t AllocBytes = 0;
  uint32_t ReflectedStride = 0;
  std::vector<uint8_t> InlineData;
  bool IsStandaloneCounter = false;
  bool IsUnbounded = false;
  bool IsTexture = false;
  uint32_t TexW = 1, TexH = 1, TexD = 1, TexMips = 1, TexSlices = 1;
  uint32_t TexBytesPerPixel = 4;
  DXGI_FORMAT TexFmt = DXGI_FORMAT_R32G32B32A32_FLOAT;
  std::vector<TexSubresourceLayout> TexLayouts;
};

Error executeComputeD3D12(Pipeline &P, ID3D12Device *Dev,
                          ID3D12CommandQueue *Queue,
                          const D3D12BackendCaps &BackendCaps) {
  D3D12Loader &L = getD3D12Loader();
  if (P.Shaders.empty() || !P.Shaders[0].Shader)
    return createStringError(std::errc::invalid_argument,
                             "compute pipeline has no shader");

  const offloadtest::Shader &CS = P.Shaders[0];
  crucible::Options Opts;
  Opts.Target = crucible::TargetBackend::DXIL;
  Opts.EntryPoint = CS.Entry;
  // Cap emitted DXIL Shader Model to what the proxied D3D12 device supports
  // (up to 6.6).
  Opts.ShaderModel =
      (!BackendCaps.HighestShaderModel.empty() &&
       BackendCaps.HighestShaderModel < "6.6")
          ? BackendCaps.HighestShaderModel
          : "6.6";

  for (const auto &SC : CS.SpecializationConstants) {
    uint64_t Bits = 0;
    if (SC.Value.size() <= sizeof(Bits))
      std::memcpy(&Bits, SC.Value.data(), SC.Value.size());
    Opts.SpecConstants[SC.ConstantID] = Bits;
  }

  auto Compiled =
      crucible::compileToDXIL(CS.Shader->getMemBufferRef(), Opts);
  if (!Compiled)
    return Compiled.takeError();

  // Map pipeline resources to their DXIL (Class, Space, Register) bindings.
  std::vector<BoundResource> Bound;
  std::vector<BoundSampler> BoundSamplers;
  for (size_t SetIdx = 0; SetIdx < P.Sets.size(); ++SetIdx) {
    for (const offloadtest::Resource &Res : P.Sets[SetIdx].Resources) {
      if (Res.isAccelerationStructure())
        return createStringError(
            std::errc::not_supported,
            "D3D12 offload proxy does not yet support texture/sampler/AS "
            "resource '%s'",
            Res.Name.c_str());

      auto findReflected =
          [&](crucible::ResourceClass C, uint32_t SrcSet, uint32_t SrcBinding,
              const std::string &Name,
              bool HasVK) -> const crucible::ResourceBinding * {
        if (HasVK) {
          for (const crucible::ResourceBinding &RB : Compiled->Reflect.Resources) {
            if (RB.Class == C && !RB.IsPushConstant &&
                RB.SPIRVSet == SrcSet && RB.SPIRVBinding == SrcBinding)
              return &RB;
          }
        }
        if (!Name.empty()) {
          for (const crucible::ResourceBinding &RB : Compiled->Reflect.Resources) {
            if (RB.Class == C && !RB.IsPushConstant && !RB.Name.empty() &&
                RB.Name == Name)
              return &RB;
          }
        }
        for (const crucible::ResourceBinding &RB : Compiled->Reflect.Resources) {
          if (RB.Class == C && !RB.IsPushConstant && RB.Space == SrcSet &&
              RB.LowerBound == SrcBinding)
            return &RB;
        }
        return nullptr;
      };

      if (Res.isSampler()) {
        if (!Res.SamplerPtr)
          return createStringError(std::errc::invalid_argument,
                                   "sampler '%s' has no descriptor",
                                   Res.Name.c_str());
        uint32_t Space = static_cast<uint32_t>(SetIdx);
        uint32_t BaseReg = 0;
        if (Res.VKBinding) {
          BaseReg = Res.VKBinding->Binding;
        } else if (Res.DXBinding) {
          Space = Res.DXBinding->Space;
          BaseReg = Res.DXBinding->Register;
        }
        const crucible::ResourceBinding *Matched =
            findReflected(crucible::ResourceClass::Sampler, Space, BaseReg,
                          Res.Name, static_cast<bool>(Res.VKBinding));
        if (!Matched)
          continue;
        BoundSampler BS;
        BS.Res = &Res;
        BS.Space = Matched->Space;
        BS.Register = Matched->LowerBound;
        BS.IsUnbounded = (Matched->RangeSize == 0xFFFFFFFFu);
        BoundSamplers.push_back(BS);
        continue;
      }

      if (!Res.BufferPtr)
        return createStringError(std::errc::invalid_argument,
                                 "resource '%s' has no backing buffer",
                                 Res.Name.c_str());

      auto ClassOpt = classifyResource(Res.Kind);
      if (!ClassOpt)
        return createStringError(std::errc::not_supported,
                                 "unsupported resource kind for '%s'",
                                 Res.Name.c_str());

      uint32_t Space = static_cast<uint32_t>(SetIdx);
      uint32_t BaseReg = 0;
      if (Res.VKBinding) {
        BaseReg = Res.VKBinding->Binding;
      } else if (Res.DXBinding) {
        Space = Res.DXBinding->Space;
        BaseReg = Res.DXBinding->Register;
      }

      crucible::ResourceClass Class = *ClassOpt;
      const crucible::ResourceBinding *Matched = findReflected(
          Class, Space, BaseReg, Res.Name, static_cast<bool>(Res.VKBinding));
      uint32_t ArrCount = std::max<uint32_t>(1u, Res.BufferPtr->ArraySize);
      if (Matched) {
        Space = Matched->Space;
        BaseReg = Matched->LowerBound;
        bool IsUnbounded = (Matched->RangeSize == 0xFFFFFFFFu);
        for (uint32_t A = 0; A < ArrCount; ++A) {
          BoundResource BR;
          BR.Res = &Res;
          BR.ArrIdx = A;
          BR.Class = Class;
          BR.Space = Space;
          BR.Register = BaseReg + A;
          BR.IsUnbounded = IsUnbounded;
          BR.ReflectedStride = Matched->ElementStride;
          BR.SizeBytes = Res.BufferPtr->Size;
          if (Res.isTexture()) {
            BR.IsTexture = true;
            const CPUBuffer &CPU = *Res.BufferPtr;
            BR.TexW = static_cast<uint32_t>(std::max(1, CPU.OutputProps.Width));
            BR.TexH = static_cast<uint32_t>(std::max(1, CPU.OutputProps.Height));
            BR.TexD = static_cast<uint32_t>(std::max(1, CPU.OutputProps.Depth));
            BR.TexMips =
                static_cast<uint32_t>(std::max(1, CPU.OutputProps.MipLevels));
            BR.TexSlices =
                static_cast<uint32_t>(std::max(1, CPU.OutputProps.ArraySlices));
            if (Res.Kind == ResourceKind::TextureCube)
              BR.TexSlices = 6;
            else if (Res.Kind == ResourceKind::TextureCubeArray)
              BR.TexSlices = std::max(6u, BR.TexSlices);
            BR.TexFmt = dxgiFormatForBuffer(CPU);
            unsigned Ch = CPU.Channels ? static_cast<unsigned>(CPU.Channels) : 1u;
            BR.TexBytesPerPixel =
                static_cast<uint32_t>(CPU.getSingleElementSize() * Ch);
            uint64_t SrcOff = 0, DstOff = 0;
            for (uint32_t S = 0; S < BR.TexSlices; ++S) {
              for (uint32_t M = 0; M < BR.TexMips; ++M) {
                uint32_t MW = std::max(1u, BR.TexW >> M);
                uint32_t MH = std::max(1u, BR.TexH >> M);
                uint32_t MD = std::max(1u, BR.TexD >> M);
                uint32_t RawPitch = MW * BR.TexBytesPerPixel;
                uint32_t AlignedPitch =
                    (RawPitch + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1u) &
                    ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1u);
                DstOff = (DstOff + D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1ull) &
                         ~(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1ull);
                TexSubresourceLayout L{};
                L.Subresource = S * BR.TexMips + M;
                L.Width = MW;
                L.Height = MH;
                L.Depth = MD;
                L.RowPitch = AlignedPitch;
                L.SrcOffset = SrcOff;
                L.DstOffset = DstOff;
                BR.TexLayouts.push_back(L);
                SrcOff += static_cast<uint64_t>(RawPitch) * MH * MD;
                DstOff += static_cast<uint64_t>(AlignedPitch) * MH * MD;
              }
            }
            BR.AllocBytes = std::max<uint64_t>(DstOff, 256ull);
          } else {
            if (Class == crucible::ResourceClass::CBV) {
              BR.AllocBytes = (BR.SizeBytes + 255ull) & ~255ull;
            } else if (Res.Kind == ResourceKind::ByteAddressBuffer ||
                       Res.Kind == ResourceKind::RWByteAddressBuffer) {
              BR.AllocBytes = std::max<uint64_t>(4ull, (BR.SizeBytes + 3ull) & ~3ull);
            } else if (Res.Kind == ResourceKind::StructuredBuffer ||
                       Res.Kind == ResourceKind::RWStructuredBuffer) {
              uint32_t S = BR.ReflectedStride
                               ? BR.ReflectedStride
                               : std::max<uint32_t>(2u, Res.BufferPtr->getElementSize());
              BR.AllocBytes =
                  std::max<uint64_t>(S, ((BR.SizeBytes + S - 1ull) / S) * S);
            } else {
              BR.AllocBytes = std::max<uint64_t>(BR.SizeBytes, 4ull);
            }
          }
          Bound.push_back(std::move(BR));
        }
      }

      // Vulkan SPIR-V represents RWStructuredBuffer counters as a separate UAV
      // buffer at CounterBinding (counter.var.<Name>).
      if (Res.HasCounter && Res.VKBinding && Res.VKBinding->CounterBinding) {
        uint32_t CSpace = static_cast<uint32_t>(SetIdx);
        uint32_t CBaseReg = *Res.VKBinding->CounterBinding;
        std::string CName = "counter.var." + Res.Name;
        const crucible::ResourceBinding *CMatched = findReflected(
            crucible::ResourceClass::UAV, CSpace, CBaseReg, CName, true);
        if (CMatched) {
          CSpace = CMatched->Space;
          CBaseReg = CMatched->LowerBound;
          bool CUnbounded = (CMatched->RangeSize == 0xFFFFFFFFu);
          for (uint32_t A = 0; A < ArrCount; ++A) {
            BoundResource CBR;
            CBR.Res = &Res;
            CBR.ArrIdx = A;
            CBR.Class = crucible::ResourceClass::UAV;
            CBR.Space = CSpace;
            CBR.Register = CBaseReg + A;
            CBR.IsUnbounded = CUnbounded;
            CBR.SizeBytes = 4;
            CBR.AllocBytes = 4;
            CBR.IsStandaloneCounter = true;
            Bound.push_back(CBR);
          }
        }
      }
    }
  }

  // Bind Vulkan PushConstants (remapped to a valid CBV space by
  // assignDXILBindings) if referenced by the compiled shader.
  if (!P.PushConstants.empty()) {
    for (const crucible::ResourceBinding &RB : Compiled->Reflect.Resources) {
      if (RB.Class == crucible::ResourceClass::CBV && RB.IsPushConstant) {
        llvm::SmallVector<uint8_t, 64> PCBytes;
        for (const auto &Block : P.PushConstants) {
          if (Block.Stage == Stages::Compute || P.PushConstants.size() == 1) {
            Block.getContent(PCBytes);
            break;
          }
        }
        uint64_t RawSize = std::max<uint64_t>(
            {static_cast<uint64_t>(PCBytes.size()), RB.ElementStride, 16ull});
        BoundResource BR;
        BR.Res = nullptr;
        BR.Class = crucible::ResourceClass::CBV;
        BR.Space = RB.Space;
        BR.Register = RB.LowerBound;
        BR.InlineData.assign(PCBytes.begin(), PCBytes.end());
        BR.SizeBytes = BR.InlineData.size();
        BR.AllocBytes = (RawSize + 255ull) & ~255ull;
        Bound.push_back(std::move(BR));
      }
    }
  }

  uint32_t NumDescs = std::max<uint32_t>(1u, static_cast<uint32_t>(Bound.size()));
  D3D12_DESCRIPTOR_HEAP_DESC HeapDesc{};
  HeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  HeapDesc.NumDescriptors = NumDescs;
  HeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  ID3D12DescriptorHeap *DescHeap = nullptr;
  HRESULT Hr = Dev->CreateDescriptorHeap(&HeapDesc,
                                         __uuidof(ID3D12DescriptorHeap),
                                         reinterpret_cast<void **>(&DescHeap));
  if (FAILED(Hr))
    return hrErr(Hr, "CreateDescriptorHeap");

  std::vector<D3D12_DESCRIPTOR_RANGE> Ranges;
  Ranges.reserve(Bound.size());
  for (uint32_t I = 0; I < Bound.size();) {
    uint32_t Start = I;
    bool AnyUnbounded = Bound[I].IsUnbounded;
    uint32_t RunLen = 1;
    while (I + RunLen < Bound.size() &&
           Bound[I + RunLen].Class == Bound[Start].Class &&
           Bound[I + RunLen].Space == Bound[Start].Space &&
           Bound[I + RunLen].Register == Bound[Start].Register + RunLen) {
      AnyUnbounded |= Bound[I + RunLen].IsUnbounded;
      ++RunLen;
    }
    D3D12_DESCRIPTOR_RANGE R{};
    switch (Bound[Start].Class) {
    case crucible::ResourceClass::CBV:
      R.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
      break;
    case crucible::ResourceClass::SRV:
      R.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
      break;
    case crucible::ResourceClass::UAV:
      R.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
      break;
    case crucible::ResourceClass::Sampler:
      R.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
      break;
    }
    R.NumDescriptors = AnyUnbounded ? UINT_MAX : RunLen;
    R.BaseShaderRegister = Bound[Start].Register;
    R.RegisterSpace = Bound[Start].Space;
    R.OffsetInDescriptorsFromTableStart = Start;
    Ranges.push_back(R);
    I += RunLen;
  }

  ID3D12DescriptorHeap *SamplerHeap = nullptr;
  std::vector<D3D12_DESCRIPTOR_RANGE> SamplerRanges;
  if (!BoundSamplers.empty()) {
    D3D12_DESCRIPTOR_HEAP_DESC SHeapDesc{};
    SHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    SHeapDesc.NumDescriptors = static_cast<UINT>(BoundSamplers.size());
    SHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    Hr = Dev->CreateDescriptorHeap(&SHeapDesc, __uuidof(ID3D12DescriptorHeap),
                                   reinterpret_cast<void **>(&SamplerHeap));
    if (FAILED(Hr))
      return hrErr(Hr, "CreateDescriptorHeap(Sampler)");
    SamplerRanges.reserve(BoundSamplers.size());
    for (uint32_t I = 0; I < BoundSamplers.size();) {
      uint32_t Start = I;
      bool AnyUnbounded = BoundSamplers[I].IsUnbounded;
      uint32_t RunLen = 1;
      while (I + RunLen < BoundSamplers.size() &&
             BoundSamplers[I + RunLen].Space == BoundSamplers[Start].Space &&
             BoundSamplers[I + RunLen].Register ==
                 BoundSamplers[Start].Register + RunLen) {
        AnyUnbounded |= BoundSamplers[I + RunLen].IsUnbounded;
        ++RunLen;
      }
      D3D12_DESCRIPTOR_RANGE R{};
      R.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
      R.NumDescriptors = AnyUnbounded ? UINT_MAX : RunLen;
      R.BaseShaderRegister = BoundSamplers[Start].Register;
      R.RegisterSpace = BoundSamplers[Start].Space;
      R.OffsetInDescriptorsFromTableStart = Start;
      SamplerRanges.push_back(R);
      I += RunLen;
    }
  }

  std::vector<D3D12_ROOT_PARAMETER> RootParams;
  int CbvSrvUavParamIdx = -1;
  int SamplerParamIdx = -1;
  if (!Ranges.empty()) {
    CbvSrvUavParamIdx = static_cast<int>(RootParams.size());
    D3D12_ROOT_PARAMETER P0{};
    P0.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    P0.DescriptorTable.NumDescriptorRanges = static_cast<UINT>(Ranges.size());
    P0.DescriptorTable.pDescriptorRanges = Ranges.data();
    P0.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    RootParams.push_back(P0);
  }
  if (!SamplerRanges.empty()) {
    SamplerParamIdx = static_cast<int>(RootParams.size());
    D3D12_ROOT_PARAMETER P1{};
    P1.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    P1.DescriptorTable.NumDescriptorRanges =
        static_cast<UINT>(SamplerRanges.size());
    P1.DescriptorTable.pDescriptorRanges = SamplerRanges.data();
    P1.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    RootParams.push_back(P1);
  }

  D3D12_ROOT_SIGNATURE_DESC RSDesc{};
  RSDesc.NumParameters = static_cast<UINT>(RootParams.size());
  RSDesc.pParameters = RootParams.empty() ? nullptr : RootParams.data();
  RSDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

  ID3DBlob *RSBlob = nullptr;
  ID3DBlob *ErrBlob = nullptr;
  Hr = L.SerializeRS(&RSDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &RSBlob,
                     &ErrBlob);
  if (FAILED(Hr)) {
    std::string Msg = "D3D12SerializeRootSignature failed";
    if (ErrBlob) {
      Msg += ": ";
      Msg.append(static_cast<const char *>(ErrBlob->GetBufferPointer()),
                 ErrBlob->GetBufferSize());
      ErrBlob->Release();
    }
    return createStringError(inconvertibleErrorCode(), "%s", Msg.c_str());
  }

  ID3D12RootSignature *RootSig = nullptr;
  Hr = Dev->CreateRootSignature(0, RSBlob->GetBufferPointer(),
                                RSBlob->GetBufferSize(),
                                __uuidof(ID3D12RootSignature),
                                reinterpret_cast<void **>(&RootSig));
  RSBlob->Release();
  if (FAILED(Hr))
    return hrErr(Hr, "CreateRootSignature");

#ifdef _WIN32
  std::string SignErr;
  if (!crucible_win_dxil::signDXILInPlace(Compiled->Output.data(),
                                          Compiled->Output.size(), SignErr))
    return createStringError(inconvertibleErrorCode(),
                             "DXIL validation/signing failed: %s",
                             SignErr.c_str());
#endif

  D3D12_COMPUTE_PIPELINE_STATE_DESC PSODesc{};
  PSODesc.pRootSignature = RootSig;
  PSODesc.CS.pShaderBytecode = Compiled->Output.data();
  PSODesc.CS.BytecodeLength = Compiled->Output.size();
  ID3D12PipelineState *PSO = nullptr;
  Hr = Dev->CreateComputePipelineState(&PSODesc, __uuidof(ID3D12PipelineState),
                                       reinterpret_cast<void **>(&PSO));
  if (FAILED(Hr))
    return hrErr(Hr, "CreateComputePipelineState");

  ID3D12CommandAllocator *Alloc = nullptr;
  Hr = Dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                   __uuidof(ID3D12CommandAllocator),
                                   reinterpret_cast<void **>(&Alloc));
  if (FAILED(Hr))
    return hrErr(Hr, "CreateCommandAllocator");

  ID3D12GraphicsCommandList *Cmd = nullptr;
  Hr = Dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, Alloc, PSO,
                              __uuidof(ID3D12GraphicsCommandList),
                              reinterpret_cast<void **>(&Cmd));
  if (FAILED(Hr))
    return hrErr(Hr, "CreateCommandList");

  UINT IncSize = Dev->GetDescriptorHandleIncrementSize(
      D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  D3D12_CPU_DESCRIPTOR_HANDLE CpuStart =
      DescHeap->GetCPUDescriptorHandleForHeapStart();

  for (uint32_t I = 0; I < Bound.size(); ++I) {
    BoundResource &BR = Bound[I];
    D3D12_CPU_DESCRIPTOR_HANDLE Slot{CpuStart.ptr +
                                     static_cast<SIZE_T>(I) * IncSize};

    if (BR.Class == crucible::ResourceClass::CBV) {
      auto UpOrErr = createCommittedBuffer(
          Dev, D3D12_HEAP_TYPE_UPLOAD, BR.AllocBytes, D3D12_RESOURCE_FLAG_NONE,
          D3D12_RESOURCE_STATE_GENERIC_READ);
      if (!UpOrErr)
        return UpOrErr.takeError();
      BR.UploadBuf = *UpOrErr;
      void *Mapped = nullptr;
      Hr = BR.UploadBuf->Map(0, nullptr, &Mapped);
      if (FAILED(Hr))
        return hrErr(Hr, "Map CBV");
      std::memset(Mapped, 0, BR.AllocBytes);
      if (!BR.InlineData.empty()) {
        std::memcpy(Mapped, BR.InlineData.data(), BR.InlineData.size());
      } else if (BR.Res && BR.Res->BufferPtr) {
        const CPUBuffer &CPU = *BR.Res->BufferPtr;
        if (BR.ArrIdx < CPU.Data.size() && CPU.Data[BR.ArrIdx])
          std::memcpy(Mapped, CPU.Data[BR.ArrIdx].get(), BR.SizeBytes);
      }
      BR.UploadBuf->Unmap(0, nullptr);

      D3D12_CONSTANT_BUFFER_VIEW_DESC CBVDesc{};
      CBVDesc.BufferLocation = BR.UploadBuf->GetGPUVirtualAddress();
      CBVDesc.SizeInBytes = static_cast<UINT>(BR.AllocBytes);
      Dev->CreateConstantBufferView(&CBVDesc, Slot);
    } else if (BR.IsTexture) {
      const CPUBuffer &CPU = *BR.Res->BufferPtr;
      bool IsUAV = (BR.Class == crucible::ResourceClass::UAV);
      D3D12_RESOURCE_FLAGS Flags =
          IsUAV ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                : D3D12_RESOURCE_FLAG_NONE;
      bool Is1D = (BR.Res->Kind == ResourceKind::Texture1D ||
                   BR.Res->Kind == ResourceKind::Texture1DArray ||
                   BR.Res->Kind == ResourceKind::RWTexture1D ||
                   BR.Res->Kind == ResourceKind::RWTexture1DArray);
      bool Is3D = (BR.Res->Kind == ResourceKind::Texture3D ||
                   BR.Res->Kind == ResourceKind::RWTexture3D);
      bool IsArray = (BR.Res->Kind == ResourceKind::Texture1DArray ||
                      BR.Res->Kind == ResourceKind::Texture2DArray ||
                      BR.Res->Kind == ResourceKind::RWTexture1DArray ||
                      BR.Res->Kind == ResourceKind::RWTexture2DArray ||
                      BR.TexSlices > 1);

      D3D12_HEAP_PROPERTIES Heap{};
      Heap.Type = D3D12_HEAP_TYPE_DEFAULT;
      Heap.CreationNodeMask = 1;
      Heap.VisibleNodeMask = 1;
      D3D12_RESOURCE_DESC TDesc{};
      TDesc.Dimension = Is1D   ? D3D12_RESOURCE_DIMENSION_TEXTURE1D
                        : Is3D ? D3D12_RESOURCE_DIMENSION_TEXTURE3D
                               : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      TDesc.Width = BR.TexW;
      TDesc.Height = Is1D ? 1 : BR.TexH;
      TDesc.DepthOrArraySize =
          static_cast<UINT16>(Is3D ? BR.TexD : BR.TexSlices);
      TDesc.MipLevels = static_cast<UINT16>(BR.TexMips);
      TDesc.Format = BR.TexFmt;
      TDesc.SampleDesc = {1, 0};
      TDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
      TDesc.Flags = Flags;
      Hr = Dev->CreateCommittedResource(
          &Heap, D3D12_HEAP_FLAG_NONE, &TDesc, D3D12_RESOURCE_STATE_COPY_DEST,
          nullptr, __uuidof(ID3D12Resource),
          reinterpret_cast<void **>(&BR.DefaultBuf));
      if (FAILED(Hr))
        return hrErr(Hr, "CreateCommittedResource(Texture)");

      auto UpOrErr = createCommittedBuffer(
          Dev, D3D12_HEAP_TYPE_UPLOAD, BR.AllocBytes, D3D12_RESOURCE_FLAG_NONE,
          D3D12_RESOURCE_STATE_GENERIC_READ);
      if (!UpOrErr)
        return UpOrErr.takeError();
      BR.UploadBuf = *UpOrErr;

      void *Mapped = nullptr;
      Hr = BR.UploadBuf->Map(0, nullptr, &Mapped);
      if (FAILED(Hr))
        return hrErr(Hr, "Map texture upload");
      std::memset(Mapped, 0, BR.AllocBytes);
      const char *SrcBase =
          (BR.ArrIdx < CPU.Data.size() && CPU.Data[BR.ArrIdx])
              ? CPU.Data[BR.ArrIdx].get()
              : nullptr;
      for (const TexSubresourceLayout &L : BR.TexLayouts) {
        uint32_t RawPitch = L.Width * BR.TexBytesPerPixel;
        if (SrcBase) {
          for (uint32_t Z = 0; Z < L.Depth; ++Z) {
            for (uint32_t Y = 0; Y < L.Height; ++Y) {
              uint64_t SOff =
                  L.SrcOffset + (static_cast<uint64_t>(Z) * L.Height + Y) * RawPitch;
              uint64_t DOff =
                  L.DstOffset + (static_cast<uint64_t>(Z) * L.Height + Y) * L.RowPitch;
              if (SOff + RawPitch <= BR.SizeBytes)
                std::memcpy(static_cast<char *>(Mapped) + DOff, SrcBase + SOff,
                            RawPitch);
            }
          }
        }
        D3D12_TEXTURE_COPY_LOCATION DstLoc{};
        DstLoc.pResource = BR.DefaultBuf;
        DstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        DstLoc.SubresourceIndex = L.Subresource;

        D3D12_TEXTURE_COPY_LOCATION SrcLoc{};
        SrcLoc.pResource = BR.UploadBuf;
        SrcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        SrcLoc.PlacedFootprint.Offset = L.DstOffset;
        SrcLoc.PlacedFootprint.Footprint.Format = BR.TexFmt;
        SrcLoc.PlacedFootprint.Footprint.Width = L.Width;
        SrcLoc.PlacedFootprint.Footprint.Height = L.Height;
        SrcLoc.PlacedFootprint.Footprint.Depth = L.Depth;
        SrcLoc.PlacedFootprint.Footprint.RowPitch = L.RowPitch;
        Cmd->CopyTextureRegion(&DstLoc, 0, 0, 0, &SrcLoc, nullptr);
      }
      BR.UploadBuf->Unmap(0, nullptr);

      D3D12_RESOURCE_BARRIER Barrier{};
      Barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      Barrier.Transition.pResource = BR.DefaultBuf;
      Barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      Barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
      Barrier.Transition.StateAfter =
          IsUAV ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
      Cmd->ResourceBarrier(1, &Barrier);

      if (IsUAV) {
        auto RbOrErr = createCommittedBuffer(
            Dev, D3D12_HEAP_TYPE_READBACK, BR.AllocBytes,
            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!RbOrErr)
          return RbOrErr.takeError();
        BR.ReadbackBuf = *RbOrErr;

        D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc{};
        UAVDesc.Format = BR.TexFmt;
        if (Is1D) {
          UAVDesc.ViewDimension = IsArray ? D3D12_UAV_DIMENSION_TEXTURE1DARRAY
                                          : D3D12_UAV_DIMENSION_TEXTURE1D;
          if (IsArray) {
            UAVDesc.Texture1DArray.MipSlice = 0;
            UAVDesc.Texture1DArray.FirstArraySlice = 0;
            UAVDesc.Texture1DArray.ArraySize = BR.TexSlices;
          } else {
            UAVDesc.Texture1D.MipSlice = 0;
          }
        } else if (Is3D) {
          UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
          UAVDesc.Texture3D.MipSlice = 0;
          UAVDesc.Texture3D.FirstWSlice = 0;
          UAVDesc.Texture3D.WSize = BR.TexD;
        } else if (IsArray) {
          UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
          UAVDesc.Texture2DArray.MipSlice = 0;
          UAVDesc.Texture2DArray.FirstArraySlice = 0;
          UAVDesc.Texture2DArray.ArraySize = BR.TexSlices;
        } else {
          UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
          UAVDesc.Texture2D.MipSlice = 0;
        }
        Dev->CreateUnorderedAccessView(BR.DefaultBuf, nullptr, &UAVDesc, Slot);
      } else {
        bool IsCube = (BR.Res->Kind == ResourceKind::TextureCube);
        bool IsCubeArray = (BR.Res->Kind == ResourceKind::TextureCubeArray);
        D3D12_SHADER_RESOURCE_VIEW_DESC SRVDesc{};
        SRVDesc.Format = BR.TexFmt;
        SRVDesc.Shader4ComponentMapping =
            D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (Is1D) {
          SRVDesc.ViewDimension = IsArray ? D3D12_SRV_DIMENSION_TEXTURE1DARRAY
                                          : D3D12_SRV_DIMENSION_TEXTURE1D;
          if (IsArray) {
            SRVDesc.Texture1DArray.MipLevels = BR.TexMips;
            SRVDesc.Texture1DArray.ArraySize = BR.TexSlices;
          } else {
            SRVDesc.Texture1D.MipLevels = BR.TexMips;
          }
        } else if (Is3D) {
          SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
          SRVDesc.Texture3D.MipLevels = BR.TexMips;
        } else if (IsCube) {
          SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
          SRVDesc.TextureCube.MipLevels = BR.TexMips;
        } else if (IsCubeArray) {
          SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
          SRVDesc.TextureCubeArray.MipLevels = BR.TexMips;
          SRVDesc.TextureCubeArray.NumCubes = std::max(1u, BR.TexSlices / 6u);
        } else if (IsArray) {
          SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
          SRVDesc.Texture2DArray.MipLevels = BR.TexMips;
          SRVDesc.Texture2DArray.ArraySize = BR.TexSlices;
        } else {
          SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
          SRVDesc.Texture2D.MipLevels = BR.TexMips;
        }
        Dev->CreateShaderResourceView(BR.DefaultBuf, &SRVDesc, Slot);
      }
    } else {
      const CPUBuffer &CPU = *BR.Res->BufferPtr;
      bool IsUAV = (BR.Class == crucible::ResourceClass::UAV);
      D3D12_RESOURCE_FLAGS Flags = IsUAV
                                       ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                                       : D3D12_RESOURCE_FLAG_NONE;
      auto DefOrErr = createCommittedBuffer(Dev, D3D12_HEAP_TYPE_DEFAULT,
                                            BR.AllocBytes, Flags,
                                            D3D12_RESOURCE_STATE_COPY_DEST);
      if (!DefOrErr)
        return DefOrErr.takeError();
      BR.DefaultBuf = *DefOrErr;

      auto UpOrErr = createCommittedBuffer(
          Dev, D3D12_HEAP_TYPE_UPLOAD, BR.AllocBytes, D3D12_RESOURCE_FLAG_NONE,
          D3D12_RESOURCE_STATE_GENERIC_READ);
      if (!UpOrErr)
        return UpOrErr.takeError();
      BR.UploadBuf = *UpOrErr;

      void *Mapped = nullptr;
      Hr = BR.UploadBuf->Map(0, nullptr, &Mapped);
      if (FAILED(Hr))
        return hrErr(Hr, "Map upload");
      std::memset(Mapped, 0, BR.AllocBytes);
      if (BR.IsStandaloneCounter) {
        uint32_t InitVal =
            (BR.ArrIdx < CPU.Counters.size()) ? CPU.Counters[BR.ArrIdx] : 0u;
        std::memcpy(Mapped, &InitVal, 4);
      } else if (BR.ArrIdx < CPU.Data.size() && CPU.Data[BR.ArrIdx]) {
        std::memcpy(Mapped, CPU.Data[BR.ArrIdx].get(), BR.SizeBytes);
      }
      BR.UploadBuf->Unmap(0, nullptr);

      Cmd->CopyBufferRegion(BR.DefaultBuf, 0, BR.UploadBuf, 0, BR.AllocBytes);

      D3D12_RESOURCE_BARRIER Barrier{};
      Barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      Barrier.Transition.pResource = BR.DefaultBuf;
      Barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      Barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
      Barrier.Transition.StateAfter =
          IsUAV ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
      Cmd->ResourceBarrier(1, &Barrier);

      if (IsUAV) {
        auto RbOrErr = createCommittedBuffer(
            Dev, D3D12_HEAP_TYPE_READBACK, BR.AllocBytes,
            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!RbOrErr)
          return RbOrErr.takeError();
        BR.ReadbackBuf = *RbOrErr;

        if (BR.IsStandaloneCounter) {
          D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc{};
          UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
          UAVDesc.Format = DXGI_FORMAT_UNKNOWN;
          UAVDesc.Buffer.FirstElement = 0;
          UAVDesc.Buffer.StructureByteStride = 4;
          UAVDesc.Buffer.NumElements = 1;
          Dev->CreateUnorderedAccessView(BR.DefaultBuf, nullptr, &UAVDesc,
                                         Slot);
          continue;
        }

        if (BR.Res->HasCounter &&
            !(BR.Res->VKBinding && BR.Res->VKBinding->CounterBinding)) {
          auto CDef = createCommittedBuffer(
              Dev, D3D12_HEAP_TYPE_DEFAULT, 4,
              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
              D3D12_RESOURCE_STATE_COPY_DEST);
          auto CUp = createCommittedBuffer(
              Dev, D3D12_HEAP_TYPE_UPLOAD, 4, D3D12_RESOURCE_FLAG_NONE,
              D3D12_RESOURCE_STATE_GENERIC_READ);
          auto CRb = createCommittedBuffer(
              Dev, D3D12_HEAP_TYPE_READBACK, 4, D3D12_RESOURCE_FLAG_NONE,
              D3D12_RESOURCE_STATE_COPY_DEST);
          if (!CDef) return CDef.takeError();
          if (!CUp) return CUp.takeError();
          if (!CRb) return CRb.takeError();
          BR.CounterDefaultBuf = *CDef;
          BR.CounterUploadBuf = *CUp;
          BR.CounterReadbackBuf = *CRb;
          uint32_t InitCount =
              (BR.ArrIdx < CPU.Counters.size()) ? CPU.Counters[BR.ArrIdx] : 0u;
          void *CMapped = nullptr;
          BR.CounterUploadBuf->Map(0, nullptr, &CMapped);
          std::memcpy(CMapped, &InitCount, 4);
          BR.CounterUploadBuf->Unmap(0, nullptr);
          Cmd->CopyBufferRegion(BR.CounterDefaultBuf, 0, BR.CounterUploadBuf, 0,
                                4);
          D3D12_RESOURCE_BARRIER CB{};
          CB.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
          CB.Transition.pResource = BR.CounterDefaultBuf;
          CB.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
          CB.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
          CB.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
          Cmd->ResourceBarrier(1, &CB);
        }

        D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc{};
        UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        if (BR.Res->Kind == ResourceKind::RWByteAddressBuffer) {
          UAVDesc.Format = DXGI_FORMAT_R32_TYPELESS;
          UAVDesc.Buffer.NumElements =
              static_cast<UINT>(std::max<uint64_t>(1, BR.AllocBytes / 4));
          UAVDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        } else if (BR.Res->Kind == ResourceKind::RWStructuredBuffer) {
          uint32_t Stride = BR.ReflectedStride
                                ? BR.ReflectedStride
                                : std::max<uint32_t>(2u, CPU.getElementSize());
          UAVDesc.Format = DXGI_FORMAT_UNKNOWN;
          UAVDesc.Buffer.StructureByteStride = Stride;
          UAVDesc.Buffer.NumElements =
              static_cast<UINT>(std::max<uint64_t>(1, BR.AllocBytes / Stride));
        } else {
          uint32_t ElemSz = std::max<uint32_t>(1u, CPU.getElementSize());
          UAVDesc.Format = dxgiFormatForBuffer(CPU);
          UAVDesc.Buffer.NumElements =
              static_cast<UINT>(std::max<uint64_t>(1, BR.AllocBytes / ElemSz));
        }
        Dev->CreateUnorderedAccessView(BR.DefaultBuf, BR.CounterDefaultBuf,
                                       &UAVDesc, Slot);
      } else {
        D3D12_SHADER_RESOURCE_VIEW_DESC SRVDesc{};
        SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        SRVDesc.Shader4ComponentMapping =
            D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (BR.Res->Kind == ResourceKind::ByteAddressBuffer) {
          SRVDesc.Format = DXGI_FORMAT_R32_TYPELESS;
          SRVDesc.Buffer.NumElements =
              static_cast<UINT>(std::max<uint64_t>(1, BR.AllocBytes / 4));
          SRVDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        } else if (BR.Res->Kind == ResourceKind::StructuredBuffer) {
          uint32_t Stride = BR.ReflectedStride
                                ? BR.ReflectedStride
                                : std::max<uint32_t>(2u, CPU.getElementSize());
          SRVDesc.Format = DXGI_FORMAT_UNKNOWN;
          SRVDesc.Buffer.StructureByteStride = Stride;
          SRVDesc.Buffer.NumElements =
              static_cast<UINT>(std::max<uint64_t>(1, BR.AllocBytes / Stride));
        } else {
          uint32_t ElemSz = std::max<uint32_t>(1u, CPU.getElementSize());
          SRVDesc.Format = dxgiFormatForBuffer(CPU);
          SRVDesc.Buffer.NumElements =
              static_cast<UINT>(std::max<uint64_t>(1, BR.AllocBytes / ElemSz));
        }
        Dev->CreateShaderResourceView(BR.DefaultBuf, &SRVDesc, Slot);
      }
    }
  }

  if (SamplerHeap) {
    UINT SInc =
        Dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    D3D12_CPU_DESCRIPTOR_HANDLE SStart =
        SamplerHeap->GetCPUDescriptorHandleForHeapStart();
    for (uint32_t I = 0; I < BoundSamplers.size(); ++I) {
      const YAMLSampler &S = *BoundSamplers[I].Res->SamplerPtr;
      D3D12_SAMPLER_DESC SDesc{};
      SDesc.Filter = d3d12Filter(S);
      D3D12_TEXTURE_ADDRESS_MODE Addr = d3d12AddressMode(S.Address);
      SDesc.AddressU = Addr;
      SDesc.AddressV = Addr;
      SDesc.AddressW = Addr;
      SDesc.MaxAnisotropy = 1;
      SDesc.ComparisonFunc = (S.Kind == SamplerKind::SamplerComparison)
                                 ? d3d12CompareFunc(S.ComparisonOp)
                                 : D3D12_COMPARISON_FUNC_NEVER;
      SDesc.MinLOD = S.MinLOD;
      SDesc.MaxLOD = S.MaxLOD;
      D3D12_CPU_DESCRIPTOR_HANDLE Slot{SStart.ptr +
                                       static_cast<SIZE_T>(I) * SInc};
      Dev->CreateSampler(&SDesc, Slot);
    }
  }

  ID3D12DescriptorHeap *Heaps[2] = {DescHeap, SamplerHeap};
  Cmd->SetDescriptorHeaps(SamplerHeap ? 2 : 1, Heaps);
  Cmd->SetComputeRootSignature(RootSig);
  Cmd->SetPipelineState(PSO);
  if (CbvSrvUavParamIdx >= 0)
    Cmd->SetComputeRootDescriptorTable(
        static_cast<UINT>(CbvSrvUavParamIdx),
        DescHeap->GetGPUDescriptorHandleForHeapStart());
  if (SamplerParamIdx >= 0 && SamplerHeap)
    Cmd->SetComputeRootDescriptorTable(
        static_cast<UINT>(SamplerParamIdx),
        SamplerHeap->GetGPUDescriptorHandleForHeapStart());

  const auto &Grid = P.DispatchParameters.DispatchGroupCount;
  Cmd->Dispatch(Grid[0], Grid[1], Grid[2]);

  for (BoundResource &BR : Bound) {
    if (BR.Class != crucible::ResourceClass::UAV)
      continue;
    D3D12_RESOURCE_BARRIER Barrier{};
    Barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    Barrier.Transition.pResource = BR.DefaultBuf;
    Barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    Barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    Barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    Cmd->ResourceBarrier(1, &Barrier);
    if (BR.IsTexture) {
      for (const TexSubresourceLayout &L : BR.TexLayouts) {
        D3D12_TEXTURE_COPY_LOCATION SrcLoc{};
        SrcLoc.pResource = BR.DefaultBuf;
        SrcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        SrcLoc.SubresourceIndex = L.Subresource;

        D3D12_TEXTURE_COPY_LOCATION DstLoc{};
        DstLoc.pResource = BR.ReadbackBuf;
        DstLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        DstLoc.PlacedFootprint.Offset = L.DstOffset;
        DstLoc.PlacedFootprint.Footprint.Format = BR.TexFmt;
        DstLoc.PlacedFootprint.Footprint.Width = L.Width;
        DstLoc.PlacedFootprint.Footprint.Height = L.Height;
        DstLoc.PlacedFootprint.Footprint.Depth = L.Depth;
        DstLoc.PlacedFootprint.Footprint.RowPitch = L.RowPitch;
        Cmd->CopyTextureRegion(&DstLoc, 0, 0, 0, &SrcLoc, nullptr);
      }
    } else {
      Cmd->CopyBufferRegion(BR.ReadbackBuf, 0, BR.DefaultBuf, 0, BR.AllocBytes);
    }

    if (BR.CounterDefaultBuf && BR.CounterReadbackBuf) {
      D3D12_RESOURCE_BARRIER CB{};
      CB.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      CB.Transition.pResource = BR.CounterDefaultBuf;
      CB.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      CB.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      CB.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
      Cmd->ResourceBarrier(1, &CB);
      Cmd->CopyBufferRegion(BR.CounterReadbackBuf, 0, BR.CounterDefaultBuf, 0,
                            4);
    }
  }

  Hr = Cmd->Close();
  if (FAILED(Hr))
    return hrErr(Hr, "CommandList::Close");

  ID3D12CommandList *Lists[] = {Cmd};
  Queue->ExecuteCommandLists(1, Lists);

  ID3D12Fence *Fence = nullptr;
  Hr = Dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence),
                        reinterpret_cast<void **>(&Fence));
  if (FAILED(Hr))
    return hrErr(Hr, "CreateFence");
  Hr = Queue->Signal(Fence, 1);
  if (FAILED(Hr))
    return hrErr(Hr, "Queue::Signal");
  if (Fence->GetCompletedValue() < 1) {
#ifdef _WIN32
    HANDLE Ev = CreateEventA(nullptr, FALSE, FALSE, nullptr);
    Hr = Fence->SetEventOnCompletion(1, Ev);
    if (SUCCEEDED(Hr) && Ev) {
      WaitForSingleObject(Ev, INFINITE);
      CloseHandle(Ev);
    }
#else
    Hr = Fence->SetEventOnCompletion(1, nullptr);
#endif
    if (FAILED(Hr))
      return hrErr(Hr, "SetEventOnCompletion");
  }

  for (BoundResource &BR : Bound) {
    if (BR.Class != crucible::ResourceClass::UAV || !BR.ReadbackBuf)
      continue;
    CPUBuffer &CPU = *BR.Res->BufferPtr;
    void *Mapped = nullptr;
    D3D12_RANGE ReadRange{
        0, static_cast<SIZE_T>(BR.IsTexture ? BR.AllocBytes : BR.SizeBytes)};
    Hr = BR.ReadbackBuf->Map(0, &ReadRange, &Mapped);
    if (FAILED(Hr))
      return hrErr(Hr, "Map readback");
    if (BR.IsStandaloneCounter) {
      if (CPU.Counters.size() <= BR.ArrIdx)
        CPU.Counters.resize(BR.ArrIdx + 1, 0);
      std::memcpy(&CPU.Counters[BR.ArrIdx], Mapped, 4);
    } else if (BR.IsTexture) {
      if (BR.ArrIdx < CPU.Data.size() && CPU.Data[BR.ArrIdx]) {
        char *DstBase = CPU.Data[BR.ArrIdx].get();
        for (const TexSubresourceLayout &L : BR.TexLayouts) {
          uint32_t RawPitch = L.Width * BR.TexBytesPerPixel;
          for (uint32_t Z = 0; Z < L.Depth; ++Z) {
            for (uint32_t Y = 0; Y < L.Height; ++Y) {
              uint64_t SOff =
                  L.DstOffset + (static_cast<uint64_t>(Z) * L.Height + Y) * L.RowPitch;
              uint64_t DOff =
                  L.SrcOffset + (static_cast<uint64_t>(Z) * L.Height + Y) * RawPitch;
              if (DOff + RawPitch <= BR.SizeBytes)
                std::memcpy(DstBase + DOff, static_cast<const char *>(Mapped) + SOff,
                            RawPitch);
            }
          }
        }
      }
    } else if (BR.ArrIdx < CPU.Data.size() && CPU.Data[BR.ArrIdx]) {
      std::memcpy(CPU.Data[BR.ArrIdx].get(), Mapped, BR.SizeBytes);
    }
    D3D12_RANGE WriteRange{0, 0};
    BR.ReadbackBuf->Unmap(0, &WriteRange);

    if (BR.CounterReadbackBuf && BR.ArrIdx < CPU.Counters.size()) {
      void *CMapped = nullptr;
      D3D12_RANGE CR{0, 4};
      if (SUCCEEDED(BR.CounterReadbackBuf->Map(0, &CR, &CMapped))) {
        std::memcpy(&CPU.Counters[BR.ArrIdx], CMapped, 4);
        BR.CounterReadbackBuf->Unmap(0, &WriteRange);
      }
    }
  }

  return Error::success();
}

Error executeGraphicsD3D12(Pipeline &P, ID3D12Device *Dev,
                           ID3D12CommandQueue *Queue,
                           const D3D12BackendCaps &BackendCaps) {
  D3D12Loader &L = getD3D12Loader();
  const offloadtest::Shader *VS = nullptr;
  const offloadtest::Shader *PS = nullptr;
  for (const auto &S : P.Shaders) {
    if (S.Stage == Stages::Vertex)
      VS = &S;
    else if (S.Stage == Stages::Pixel)
      PS = &S;
    else
      return createStringError(
          std::errc::not_supported,
          "Crucible D3D12 offload proxy currently supports Vertex+Pixel "
          "graphics pipelines only");
  }
  if (!VS || !VS->Shader || !PS || !PS->Shader)
    return createStringError(std::errc::not_supported,
                             "graphics pipeline requires both Vertex and Pixel "
                             "shaders");

  auto compileStage = [&](const offloadtest::Shader &S)
      -> Expected<crucible::DXILCompileResult> {
    crucible::Options Opts;
    Opts.Target = crucible::TargetBackend::DXIL;
    Opts.EntryPoint = S.Entry;
    Opts.ShaderModel =
        (!BackendCaps.HighestShaderModel.empty() &&
         BackendCaps.HighestShaderModel < "6.6")
            ? BackendCaps.HighestShaderModel
            : "6.6";
    for (const auto &SC : S.SpecializationConstants) {
      uint64_t Bits = 0;
      if (SC.Value.size() <= sizeof(Bits))
        std::memcpy(&Bits, SC.Value.data(), SC.Value.size());
      Opts.SpecConstants[SC.ConstantID] = Bits;
    }
    auto Res = crucible::compileToDXIL(S.Shader->getMemBufferRef(), Opts);
    if (!Res)
      return Res.takeError();
#ifdef _WIN32
    std::string SignErr;
    if (!crucible_win_dxil::signDXILInPlace(Res->Output.data(),
                                            Res->Output.size(), SignErr))
      return createStringError(inconvertibleErrorCode(),
                               "DXIL validation/signing failed: %s",
                               SignErr.c_str());
#endif
    return Res;
  };

  auto VSCompiled = compileStage(*VS);
  if (!VSCompiled)
    return VSCompiled.takeError();
  auto PSCompiled = compileStage(*PS);
  if (!PSCompiled)
    return PSCompiled.takeError();

  const IOBindings &B = P.Bindings;
  CPUBuffer *RTBuf = B.RTargetBufferPtr;
  if (!RTBuf || RTBuf->OutputProps.Width < 1 || RTBuf->OutputProps.Height < 1)
    return createStringError(std::errc::invalid_argument,
                             "graphics pipeline requires a 2D RenderTarget");
  uint32_t RTW = static_cast<uint32_t>(RTBuf->OutputProps.Width);
  uint32_t RTH = static_cast<uint32_t>(RTBuf->OutputProps.Height);
  DXGI_FORMAT RTFmt = dxgiFormatForBuffer(*RTBuf);
  unsigned RTCh = RTBuf->Channels ? static_cast<unsigned>(RTBuf->Channels) : 1u;
  uint32_t RTBpp = static_cast<uint32_t>(RTBuf->getSingleElementSize() * RTCh);
  uint32_t RTRawPitch = RTW * RTBpp;
  uint32_t RTAlignedPitch =
      (RTRawPitch + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1u) &
      ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1u);
  uint64_t RTReadbackBytes = static_cast<uint64_t>(RTAlignedPitch) * RTH;

  // Combine reflected resources from VS and PS for binding lookup.
  std::vector<crucible::ResourceBinding> AllReflect;
  AllReflect.insert(AllReflect.end(), VSCompiled->Reflect.Resources.begin(),
                    VSCompiled->Reflect.Resources.end());
  AllReflect.insert(AllReflect.end(), PSCompiled->Reflect.Resources.begin(),
                    PSCompiled->Reflect.Resources.end());

  std::vector<BoundResource> Bound;
  std::vector<BoundSampler> BoundSamplers;
  for (size_t SetIdx = 0; SetIdx < P.Sets.size(); ++SetIdx) {
    for (const offloadtest::Resource &Res : P.Sets[SetIdx].Resources) {
      if (Res.isAccelerationStructure())
        return createStringError(std::errc::not_supported,
                                 "graphics pipeline does not support AS '%s'",
                                 Res.Name.c_str());
      if (Res.isSampler()) {
        if (!Res.SamplerPtr)
          return createStringError(std::errc::invalid_argument,
                                   "sampler '%s' has no descriptor",
                                   Res.Name.c_str());
        uint32_t Space = static_cast<uint32_t>(SetIdx);
        uint32_t BaseReg = Res.VKBinding   ? Res.VKBinding->Binding
                           : Res.DXBinding ? Res.DXBinding->Register
                                           : 0u;
        if (Res.DXBinding)
          Space = Res.DXBinding->Space;
        for (const auto &RB : AllReflect) {
          if (RB.Class == crucible::ResourceClass::Sampler &&
              !RB.Name.empty() && RB.Name == Res.Name) {
            Space = RB.Space;
            BaseReg = RB.LowerBound;
            break;
          }
        }
        BoundSamplers.push_back({&Res, Space, BaseReg});
        continue;
      }
      if (!Res.BufferPtr)
        return createStringError(std::errc::invalid_argument,
                                 "resource '%s' has no backing buffer",
                                 Res.Name.c_str());
      auto ClassOpt = classifyResource(Res.Kind);
      if (!ClassOpt)
        return createStringError(std::errc::not_supported,
                                 "unsupported resource kind for '%s'",
                                 Res.Name.c_str());
      uint32_t Space = static_cast<uint32_t>(SetIdx);
      uint32_t BaseReg = Res.VKBinding   ? Res.VKBinding->Binding
                         : Res.DXBinding ? Res.DXBinding->Register
                                         : 0u;
      if (Res.DXBinding)
        Space = Res.DXBinding->Space;
      crucible::ResourceClass Class = *ClassOpt;
      for (const auto &RB : AllReflect) {
        if (RB.Class == Class && !RB.Name.empty() && RB.Name == Res.Name) {
          Space = RB.Space;
          BaseReg = RB.LowerBound;
          break;
        }
      }
      uint32_t ArrCount = std::max<uint32_t>(1u, Res.BufferPtr->ArraySize);
      for (uint32_t A = 0; A < ArrCount; ++A) {
        BoundResource BR;
        BR.Res = &Res;
        BR.ArrIdx = A;
        BR.Class = Class;
        BR.Space = Space;
        BR.Register = BaseReg + A;
        BR.SizeBytes = Res.BufferPtr->Size;
        if (Res.isTexture()) {
          BR.IsTexture = true;
          const CPUBuffer &CPU = *Res.BufferPtr;
          BR.TexW = static_cast<uint32_t>(std::max(1, CPU.OutputProps.Width));
          BR.TexH = static_cast<uint32_t>(std::max(1, CPU.OutputProps.Height));
          BR.TexD = static_cast<uint32_t>(std::max(1, CPU.OutputProps.Depth));
          BR.TexMips =
              static_cast<uint32_t>(std::max(1, CPU.OutputProps.MipLevels));
          BR.TexSlices =
              static_cast<uint32_t>(std::max(1, CPU.OutputProps.ArraySlices));
          if (Res.Kind == ResourceKind::TextureCube)
            BR.TexSlices = 6;
          else if (Res.Kind == ResourceKind::TextureCubeArray)
            BR.TexSlices = std::max(6u, BR.TexSlices);
          BR.TexFmt = dxgiFormatForBuffer(CPU);
          unsigned Ch = CPU.Channels ? static_cast<unsigned>(CPU.Channels) : 1u;
          BR.TexBytesPerPixel =
              static_cast<uint32_t>(CPU.getSingleElementSize() * Ch);
          uint64_t SrcOff = 0, DstOff = 0;
          for (uint32_t S = 0; S < BR.TexSlices; ++S) {
            for (uint32_t M = 0; M < BR.TexMips; ++M) {
              uint32_t MW = std::max(1u, BR.TexW >> M);
              uint32_t MH = std::max(1u, BR.TexH >> M);
              uint32_t MD = std::max(1u, BR.TexD >> M);
              uint32_t RawPitch = MW * BR.TexBytesPerPixel;
              uint32_t AlignedPitch =
                  (RawPitch + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1u) &
                  ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1u);
              DstOff = (DstOff + D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1ull) &
                       ~(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1ull);
              BR.TexLayouts.push_back(
                  {S * BR.TexMips + M, MW, MH, MD, AlignedPitch, SrcOff, DstOff});
              SrcOff += static_cast<uint64_t>(RawPitch) * MH * MD;
              DstOff += static_cast<uint64_t>(AlignedPitch) * MH * MD;
            }
          }
          BR.AllocBytes = std::max<uint64_t>(DstOff, 256ull);
        } else {
          BR.AllocBytes = (Class == crucible::ResourceClass::CBV)
                              ? ((BR.SizeBytes + 255ull) & ~255ull)
                              : std::max<uint64_t>(BR.SizeBytes, 4ull);
        }
        Bound.push_back(std::move(BR));
      }
    }
  }

  // Descriptor heaps
  uint32_t NumDescs = std::max<uint32_t>(1u, static_cast<uint32_t>(Bound.size()));
  D3D12_DESCRIPTOR_HEAP_DESC HeapDesc{};
  HeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  HeapDesc.NumDescriptors = NumDescs;
  HeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  ID3D12DescriptorHeap *DescHeap = nullptr;
  HRESULT Hr = Dev->CreateDescriptorHeap(&HeapDesc,
                                         __uuidof(ID3D12DescriptorHeap),
                                         reinterpret_cast<void **>(&DescHeap));
  if (FAILED(Hr))
    return hrErr(Hr, "CreateDescriptorHeap(Graphics)");

  std::vector<D3D12_DESCRIPTOR_RANGE> Ranges;
  for (uint32_t I = 0; I < Bound.size(); ++I) {
    D3D12_DESCRIPTOR_RANGE R{};
    R.RangeType = Bound[I].Class == crucible::ResourceClass::CBV
                      ? D3D12_DESCRIPTOR_RANGE_TYPE_CBV
                  : Bound[I].Class == crucible::ResourceClass::UAV
                      ? D3D12_DESCRIPTOR_RANGE_TYPE_UAV
                      : D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    R.NumDescriptors = 1;
    R.BaseShaderRegister = Bound[I].Register;
    R.RegisterSpace = Bound[I].Space;
    R.OffsetInDescriptorsFromTableStart = I;
    Ranges.push_back(R);
  }

  ID3D12DescriptorHeap *SamplerHeap = nullptr;
  std::vector<D3D12_DESCRIPTOR_RANGE> SamplerRanges;
  if (!BoundSamplers.empty()) {
    D3D12_DESCRIPTOR_HEAP_DESC SHD{};
    SHD.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    SHD.NumDescriptors = static_cast<UINT>(BoundSamplers.size());
    SHD.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    Hr = Dev->CreateDescriptorHeap(&SHD, __uuidof(ID3D12DescriptorHeap),
                                   reinterpret_cast<void **>(&SamplerHeap));
    if (FAILED(Hr))
      return hrErr(Hr, "CreateDescriptorHeap(GraphicsSampler)");
    for (uint32_t I = 0; I < BoundSamplers.size(); ++I) {
      D3D12_DESCRIPTOR_RANGE R{};
      R.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
      R.NumDescriptors = 1;
      R.BaseShaderRegister = BoundSamplers[I].Register;
      R.RegisterSpace = BoundSamplers[I].Space;
      R.OffsetInDescriptorsFromTableStart = I;
      SamplerRanges.push_back(R);
    }
  }

  std::vector<D3D12_ROOT_PARAMETER> RootParams;
  int CbvSrvUavParamIdx = -1;
  int SamplerParamIdx = -1;
  if (!Ranges.empty()) {
    CbvSrvUavParamIdx = static_cast<int>(RootParams.size());
    D3D12_ROOT_PARAMETER P0{};
    P0.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    P0.DescriptorTable.NumDescriptorRanges = static_cast<UINT>(Ranges.size());
    P0.DescriptorTable.pDescriptorRanges = Ranges.data();
    P0.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    RootParams.push_back(P0);
  }
  if (!SamplerRanges.empty()) {
    SamplerParamIdx = static_cast<int>(RootParams.size());
    D3D12_ROOT_PARAMETER P1{};
    P1.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    P1.DescriptorTable.NumDescriptorRanges =
        static_cast<UINT>(SamplerRanges.size());
    P1.DescriptorTable.pDescriptorRanges = SamplerRanges.data();
    P1.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    RootParams.push_back(P1);
  }

  D3D12_ROOT_SIGNATURE_DESC RSDesc{};
  RSDesc.NumParameters = static_cast<UINT>(RootParams.size());
  RSDesc.pParameters = RootParams.empty() ? nullptr : RootParams.data();
  RSDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

  ID3DBlob *RSBlob = nullptr;
  ID3DBlob *ErrBlob = nullptr;
  Hr = L.SerializeRS(&RSDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &RSBlob,
                     &ErrBlob);
  if (FAILED(Hr)) {
    std::string Msg = "D3D12SerializeRootSignature(Graphics) failed";
    if (ErrBlob) {
      Msg += ": ";
      Msg.append(static_cast<const char *>(ErrBlob->GetBufferPointer()),
                 ErrBlob->GetBufferSize());
      ErrBlob->Release();
    }
    return createStringError(inconvertibleErrorCode(), "%s", Msg.c_str());
  }
  ID3D12RootSignature *RootSig = nullptr;
  Hr = Dev->CreateRootSignature(0, RSBlob->GetBufferPointer(),
                                RSBlob->GetBufferSize(),
                                __uuidof(ID3D12RootSignature),
                                reinterpret_cast<void **>(&RootSig));
  RSBlob->Release();
  if (FAILED(Hr))
    return hrErr(Hr, "CreateRootSignature(Graphics)");

  // Build vertex input layout
  std::vector<std::string> SemNames;
  std::vector<D3D12_INPUT_ELEMENT_DESC> InputElems;
  SemNames.reserve(B.VertexAttributes.size());
  InputElems.reserve(B.VertexAttributes.size());
  for (const auto &Attr : B.VertexAttributes) {
    // Split trailing digits from semantic name (e.g., TEXCOORD0 -> TEXCOORD, 0).
    std::string Base = Attr.Name;
    UINT SemIdx = 0;
    size_t Pos = Base.size();
    while (Pos > 0 && std::isdigit(static_cast<unsigned char>(Base[Pos - 1])))
      --Pos;
    if (Pos < Base.size()) {
      SemIdx = static_cast<UINT>(std::stoul(Base.substr(Pos)));
      Base.resize(Pos);
    }
    SemNames.push_back(Base);
    D3D12_INPUT_ELEMENT_DESC E{};
    E.SemanticName = SemNames.back().c_str();
    E.SemanticIndex = SemIdx;
    E.Format = dxgiFormatFor(Attr.Format, Attr.Channels);
    E.InputSlot = 0;
    E.AlignedByteOffset = static_cast<UINT>(Attr.Offset);
    E.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
    E.InstanceDataStepRate = 0;
    InputElems.push_back(E);
  }

  D3D12_GRAPHICS_PIPELINE_STATE_DESC PSODesc{};
  PSODesc.pRootSignature = RootSig;
  PSODesc.VS = {VSCompiled->Output.data(), VSCompiled->Output.size()};
  PSODesc.PS = {PSCompiled->Output.data(), PSCompiled->Output.size()};
  PSODesc.BlendState.AlphaToCoverageEnable = FALSE;
  PSODesc.BlendState.IndependentBlendEnable = FALSE;
  PSODesc.BlendState.RenderTarget[0].RenderTargetWriteMask =
      D3D12_COLOR_WRITE_ENABLE_ALL;
  PSODesc.SampleMask = UINT_MAX;
  PSODesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
  PSODesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
  PSODesc.RasterizerState.FrontCounterClockwise = FALSE;
  PSODesc.RasterizerState.DepthClipEnable = TRUE;
  PSODesc.DepthStencilState.DepthEnable = FALSE;
  PSODesc.DepthStencilState.StencilEnable = FALSE;
  PSODesc.InputLayout = {InputElems.empty() ? nullptr : InputElems.data(),
                         static_cast<UINT>(InputElems.size())};
  uint32_t SampleCount = std::max<uint32_t>(1u, B.SampleCount);
  PSODesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
  PSODesc.NumRenderTargets = 1;
  PSODesc.RTVFormats[0] = RTFmt;
  PSODesc.SampleDesc = {SampleCount, 0};

  ID3D12PipelineState *PSO = nullptr;
  Hr = Dev->CreateGraphicsPipelineState(&PSODesc, __uuidof(ID3D12PipelineState),
                                        reinterpret_cast<void **>(&PSO));
  if (FAILED(Hr))
    return hrErr(Hr, "CreateGraphicsPipelineState");

  // Render target texture + RTV heap
  D3D12_HEAP_PROPERTIES DefHeap{};
  DefHeap.Type = D3D12_HEAP_TYPE_DEFAULT;
  DefHeap.CreationNodeMask = 1;
  DefHeap.VisibleNodeMask = 1;
  D3D12_RESOURCE_DESC RTDesc{};
  RTDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  RTDesc.Width = RTW;
  RTDesc.Height = RTH;
  RTDesc.DepthOrArraySize = 1;
  RTDesc.MipLevels = 1;
  RTDesc.Format = RTFmt;
  RTDesc.SampleDesc = {SampleCount, 0};
  RTDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  RTDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
  ID3D12Resource *RTTex = nullptr;
  Hr = Dev->CreateCommittedResource(&DefHeap, D3D12_HEAP_FLAG_NONE, &RTDesc,
                                    D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                    __uuidof(ID3D12Resource),
                                    reinterpret_cast<void **>(&RTTex));
  if (FAILED(Hr))
    return hrErr(Hr, "CreateCommittedResource(RenderTarget)");

  ID3D12Resource *ResolveTex = nullptr;
  if (SampleCount > 1) {
    D3D12_RESOURCE_DESC ResDesc = RTDesc;
    ResDesc.SampleDesc = {1, 0};
    ResDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    Hr = Dev->CreateCommittedResource(&DefHeap, D3D12_HEAP_FLAG_NONE, &ResDesc,
                                      D3D12_RESOURCE_STATE_RESOLVE_DEST,
                                      nullptr, __uuidof(ID3D12Resource),
                                      reinterpret_cast<void **>(&ResolveTex));
    if (FAILED(Hr))
      return hrErr(Hr, "CreateCommittedResource(ResolveTarget)");
  }

  D3D12_DESCRIPTOR_HEAP_DESC RTVHeapDesc{};
  RTVHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
  RTVHeapDesc.NumDescriptors = 1;
  ID3D12DescriptorHeap *RTVHeap = nullptr;
  Hr = Dev->CreateDescriptorHeap(&RTVHeapDesc, __uuidof(ID3D12DescriptorHeap),
                                 reinterpret_cast<void **>(&RTVHeap));
  if (FAILED(Hr))
    return hrErr(Hr, "CreateDescriptorHeap(RTV)");
  D3D12_CPU_DESCRIPTOR_HANDLE RTVHandle =
      RTVHeap->GetCPUDescriptorHandleForHeapStart();
  Dev->CreateRenderTargetView(RTTex, nullptr, RTVHandle);

  auto RTRbOrErr = createCommittedBuffer(
      Dev, D3D12_HEAP_TYPE_READBACK, RTReadbackBytes, D3D12_RESOURCE_FLAG_NONE,
      D3D12_RESOURCE_STATE_COPY_DEST);
  if (!RTRbOrErr)
    return RTRbOrErr.takeError();
  ID3D12Resource *RTReadback = *RTRbOrErr;

  ID3D12CommandAllocator *Alloc = nullptr;
  Hr = Dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                   __uuidof(ID3D12CommandAllocator),
                                   reinterpret_cast<void **>(&Alloc));
  if (FAILED(Hr))
    return hrErr(Hr, "CreateCommandAllocator(Graphics)");
  ID3D12GraphicsCommandList *Cmd = nullptr;
  Hr = Dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, Alloc, PSO,
                              __uuidof(ID3D12GraphicsCommandList),
                              reinterpret_cast<void **>(&Cmd));
  if (FAILED(Hr))
    return hrErr(Hr, "CreateCommandList(Graphics)");

  // Upload vertex buffer if present
  ID3D12Resource *VBBuf = nullptr;
  D3D12_VERTEX_BUFFER_VIEW VBView{};
  uint32_t Stride = static_cast<uint32_t>(
      B.VertexBufferPtr && B.VertexBufferPtr->Stride ? B.VertexBufferPtr->Stride
                                                     : B.getVertexStride());
  if (B.VertexBufferPtr && !B.VertexBufferPtr->Data.empty() &&
      B.VertexBufferPtr->Data[0]) {
    auto VBOrErr = createCommittedBuffer(
        Dev, D3D12_HEAP_TYPE_UPLOAD, B.VertexBufferPtr->Size,
        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    if (!VBOrErr)
      return VBOrErr.takeError();
    VBBuf = *VBOrErr;
    void *Mapped = nullptr;
    Hr = VBBuf->Map(0, nullptr, &Mapped);
    if (FAILED(Hr))
      return hrErr(Hr, "Map VB");
    std::memcpy(Mapped, B.VertexBufferPtr->Data[0].get(),
                B.VertexBufferPtr->Size);
    VBBuf->Unmap(0, nullptr);
    VBView.BufferLocation = VBBuf->GetGPUVirtualAddress();
    VBView.SizeInBytes = static_cast<UINT>(B.VertexBufferPtr->Size);
    VBView.StrideInBytes = Stride;
  }

  // Upload bound resources (CBV/SRV/UAV buffers and textures)
  UINT IncSize = Dev->GetDescriptorHandleIncrementSize(
      D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  D3D12_CPU_DESCRIPTOR_HANDLE CpuStart =
      DescHeap->GetCPUDescriptorHandleForHeapStart();
  for (uint32_t I = 0; I < Bound.size(); ++I) {
    BoundResource &BR = Bound[I];
    D3D12_CPU_DESCRIPTOR_HANDLE Slot{CpuStart.ptr +
                                     static_cast<SIZE_T>(I) * IncSize};
    if (BR.Class == crucible::ResourceClass::CBV) {
      auto UpOrErr = createCommittedBuffer(
          Dev, D3D12_HEAP_TYPE_UPLOAD, BR.AllocBytes, D3D12_RESOURCE_FLAG_NONE,
          D3D12_RESOURCE_STATE_GENERIC_READ);
      if (!UpOrErr) return UpOrErr.takeError();
      BR.UploadBuf = *UpOrErr;
      void *Mapped = nullptr;
      Hr = BR.UploadBuf->Map(0, nullptr, &Mapped);
      if (FAILED(Hr)) return hrErr(Hr, "Map CBV(Graphics)");
      std::memset(Mapped, 0, BR.AllocBytes);
      if (BR.Res && BR.Res->BufferPtr &&
          BR.ArrIdx < BR.Res->BufferPtr->Data.size() &&
          BR.Res->BufferPtr->Data[BR.ArrIdx])
        std::memcpy(Mapped, BR.Res->BufferPtr->Data[BR.ArrIdx].get(),
                    BR.SizeBytes);
      BR.UploadBuf->Unmap(0, nullptr);
      D3D12_CONSTANT_BUFFER_VIEW_DESC CBVDesc{};
      CBVDesc.BufferLocation = BR.UploadBuf->GetGPUVirtualAddress();
      CBVDesc.SizeInBytes = static_cast<UINT>(BR.AllocBytes);
      Dev->CreateConstantBufferView(&CBVDesc, Slot);
    } else if (BR.IsTexture) {
      const CPUBuffer &CPU = *BR.Res->BufferPtr;
      bool IsUAV = (BR.Class == crucible::ResourceClass::UAV);
      bool Is1D = (BR.Res->Kind == ResourceKind::Texture1D ||
                   BR.Res->Kind == ResourceKind::Texture1DArray ||
                   BR.Res->Kind == ResourceKind::RWTexture1D ||
                   BR.Res->Kind == ResourceKind::RWTexture1DArray);
      bool Is3D = (BR.Res->Kind == ResourceKind::Texture3D ||
                   BR.Res->Kind == ResourceKind::RWTexture3D);
      bool IsArray = (BR.Res->Kind == ResourceKind::Texture1DArray ||
                      BR.Res->Kind == ResourceKind::Texture2DArray ||
                      BR.Res->Kind == ResourceKind::RWTexture1DArray ||
                      BR.Res->Kind == ResourceKind::RWTexture2DArray ||
                      BR.TexSlices > 1);
      bool IsCube = (BR.Res->Kind == ResourceKind::TextureCube);
      bool IsCubeArray = (BR.Res->Kind == ResourceKind::TextureCubeArray);
      D3D12_RESOURCE_DESC TDesc{};
      TDesc.Dimension = Is1D   ? D3D12_RESOURCE_DIMENSION_TEXTURE1D
                        : Is3D ? D3D12_RESOURCE_DIMENSION_TEXTURE3D
                               : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      TDesc.Width = BR.TexW;
      TDesc.Height = Is1D ? 1 : BR.TexH;
      TDesc.DepthOrArraySize =
          static_cast<UINT16>(Is3D ? BR.TexD : BR.TexSlices);
      TDesc.MipLevels = static_cast<UINT16>(BR.TexMips);
      TDesc.Format = BR.TexFmt;
      TDesc.SampleDesc = {1, 0};
      TDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
      TDesc.Flags = IsUAV ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                          : D3D12_RESOURCE_FLAG_NONE;
      Hr = Dev->CreateCommittedResource(
          &DefHeap, D3D12_HEAP_FLAG_NONE, &TDesc,
          D3D12_RESOURCE_STATE_COPY_DEST, nullptr, __uuidof(ID3D12Resource),
          reinterpret_cast<void **>(&BR.DefaultBuf));
      if (FAILED(Hr)) return hrErr(Hr, "CreateCommittedResource(GraphicsTex)");
      auto UpOrErr = createCommittedBuffer(
          Dev, D3D12_HEAP_TYPE_UPLOAD, BR.AllocBytes, D3D12_RESOURCE_FLAG_NONE,
          D3D12_RESOURCE_STATE_GENERIC_READ);
      if (!UpOrErr) return UpOrErr.takeError();
      BR.UploadBuf = *UpOrErr;
      void *Mapped = nullptr;
      Hr = BR.UploadBuf->Map(0, nullptr, &Mapped);
      if (FAILED(Hr)) return hrErr(Hr, "Map GraphicsTex");
      std::memset(Mapped, 0, BR.AllocBytes);
      const char *SrcBase =
          (BR.ArrIdx < CPU.Data.size() && CPU.Data[BR.ArrIdx])
              ? CPU.Data[BR.ArrIdx].get()
              : nullptr;
      for (const TexSubresourceLayout &TL : BR.TexLayouts) {
        uint32_t RawPitch = TL.Width * BR.TexBytesPerPixel;
        if (SrcBase) {
          for (uint32_t Z = 0; Z < TL.Depth; ++Z)
            for (uint32_t Y = 0; Y < TL.Height; ++Y) {
              uint64_t SOff = TL.SrcOffset +
                              (static_cast<uint64_t>(Z) * TL.Height + Y) * RawPitch;
              uint64_t DOff = TL.DstOffset +
                              (static_cast<uint64_t>(Z) * TL.Height + Y) * TL.RowPitch;
              if (SOff + RawPitch <= BR.SizeBytes)
                std::memcpy(static_cast<char *>(Mapped) + DOff, SrcBase + SOff,
                            RawPitch);
            }
        }
        D3D12_TEXTURE_COPY_LOCATION DstLoc{
            BR.DefaultBuf, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        DstLoc.SubresourceIndex = TL.Subresource;
        D3D12_TEXTURE_COPY_LOCATION SrcLoc{
            BR.UploadBuf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        SrcLoc.PlacedFootprint.Offset = TL.DstOffset;
        SrcLoc.PlacedFootprint.Footprint = {BR.TexFmt, TL.Width, TL.Height,
                                            TL.Depth, TL.RowPitch};
        Cmd->CopyTextureRegion(&DstLoc, 0, 0, 0, &SrcLoc, nullptr);
      }
      BR.UploadBuf->Unmap(0, nullptr);
      D3D12_RESOURCE_BARRIER Bar{};
      Bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      Bar.Transition.pResource = BR.DefaultBuf;
      Bar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      Bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
      Bar.Transition.StateAfter =
          IsUAV ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                : D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
      Cmd->ResourceBarrier(1, &Bar);

      D3D12_SHADER_RESOURCE_VIEW_DESC SRVDesc{};
      SRVDesc.Format = BR.TexFmt;
      SRVDesc.Shader4ComponentMapping =
          D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      if (Is1D) {
        SRVDesc.ViewDimension = IsArray ? D3D12_SRV_DIMENSION_TEXTURE1DARRAY
                                        : D3D12_SRV_DIMENSION_TEXTURE1D;
        if (IsArray) {
          SRVDesc.Texture1DArray.MipLevels = BR.TexMips;
          SRVDesc.Texture1DArray.ArraySize = BR.TexSlices;
        } else {
          SRVDesc.Texture1D.MipLevels = BR.TexMips;
        }
      } else if (Is3D) {
        SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
        SRVDesc.Texture3D.MipLevels = BR.TexMips;
      } else if (IsCube) {
        SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
        SRVDesc.TextureCube.MipLevels = BR.TexMips;
      } else if (IsCubeArray) {
        SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
        SRVDesc.TextureCubeArray.MipLevels = BR.TexMips;
        SRVDesc.TextureCubeArray.NumCubes = std::max(1u, BR.TexSlices / 6u);
      } else if (IsArray) {
        SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        SRVDesc.Texture2DArray.MipLevels = BR.TexMips;
        SRVDesc.Texture2DArray.ArraySize = BR.TexSlices;
      } else {
        SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        SRVDesc.Texture2D.MipLevels = BR.TexMips;
      }
      Dev->CreateShaderResourceView(BR.DefaultBuf, &SRVDesc, Slot);
    } else {
      const CPUBuffer &CPU = *BR.Res->BufferPtr;
      bool IsUAV = (BR.Class == crucible::ResourceClass::UAV);
      auto DefOrErr = createCommittedBuffer(
          Dev, D3D12_HEAP_TYPE_DEFAULT, BR.AllocBytes,
          IsUAV ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                : D3D12_RESOURCE_FLAG_NONE,
          D3D12_RESOURCE_STATE_COPY_DEST);
      if (!DefOrErr) return DefOrErr.takeError();
      BR.DefaultBuf = *DefOrErr;
      auto UpOrErr = createCommittedBuffer(
          Dev, D3D12_HEAP_TYPE_UPLOAD, BR.AllocBytes, D3D12_RESOURCE_FLAG_NONE,
          D3D12_RESOURCE_STATE_GENERIC_READ);
      if (!UpOrErr) return UpOrErr.takeError();
      BR.UploadBuf = *UpOrErr;
      void *Mapped = nullptr;
      Hr = BR.UploadBuf->Map(0, nullptr, &Mapped);
      if (FAILED(Hr)) return hrErr(Hr, "Map GraphicsBuf");
      std::memset(Mapped, 0, BR.AllocBytes);
      if (BR.ArrIdx < CPU.Data.size() && CPU.Data[BR.ArrIdx])
        std::memcpy(Mapped, CPU.Data[BR.ArrIdx].get(), BR.SizeBytes);
      BR.UploadBuf->Unmap(0, nullptr);
      Cmd->CopyBufferRegion(BR.DefaultBuf, 0, BR.UploadBuf, 0, BR.AllocBytes);
      D3D12_RESOURCE_BARRIER Bar{};
      Bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      Bar.Transition.pResource = BR.DefaultBuf;
      Bar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      Bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
      Bar.Transition.StateAfter =
          IsUAV ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                : D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
      Cmd->ResourceBarrier(1, &Bar);
      if (IsUAV) {
        auto RbOrErr = createCommittedBuffer(
            Dev, D3D12_HEAP_TYPE_READBACK, BR.AllocBytes,
            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!RbOrErr) return RbOrErr.takeError();
        BR.ReadbackBuf = *RbOrErr;
        D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc{};
        UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        if (BR.Res->Kind == ResourceKind::RWStructuredBuffer) {
          uint32_t S = std::max<uint32_t>(4u, CPU.getElementSize());
          UAVDesc.Format = DXGI_FORMAT_UNKNOWN;
          UAVDesc.Buffer.StructureByteStride = S;
          UAVDesc.Buffer.NumElements =
              static_cast<UINT>(std::max<uint64_t>(1, BR.AllocBytes / S));
        } else {
          uint32_t E = std::max<uint32_t>(1u, CPU.getElementSize());
          UAVDesc.Format = dxgiFormatForBuffer(CPU);
          UAVDesc.Buffer.NumElements =
              static_cast<UINT>(std::max<uint64_t>(1, BR.AllocBytes / E));
        }
        Dev->CreateUnorderedAccessView(BR.DefaultBuf, nullptr, &UAVDesc, Slot);
      } else {
        D3D12_SHADER_RESOURCE_VIEW_DESC SRVDesc{};
        SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        SRVDesc.Shader4ComponentMapping =
            D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (BR.Res->Kind == ResourceKind::StructuredBuffer) {
          uint32_t S = std::max<uint32_t>(4u, CPU.getElementSize());
          SRVDesc.Format = DXGI_FORMAT_UNKNOWN;
          SRVDesc.Buffer.StructureByteStride = S;
          SRVDesc.Buffer.NumElements =
              static_cast<UINT>(std::max<uint64_t>(1, BR.AllocBytes / S));
        } else {
          uint32_t E = std::max<uint32_t>(1u, CPU.getElementSize());
          SRVDesc.Format = dxgiFormatForBuffer(CPU);
          SRVDesc.Buffer.NumElements =
              static_cast<UINT>(std::max<uint64_t>(1, BR.AllocBytes / E));
        }
        Dev->CreateShaderResourceView(BR.DefaultBuf, &SRVDesc, Slot);
      }
    }
  }

  if (SamplerHeap) {
    UINT SInc =
        Dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    D3D12_CPU_DESCRIPTOR_HANDLE SStart =
        SamplerHeap->GetCPUDescriptorHandleForHeapStart();
    for (uint32_t I = 0; I < BoundSamplers.size(); ++I) {
      const YAMLSampler &S = *BoundSamplers[I].Res->SamplerPtr;
      D3D12_SAMPLER_DESC SDesc{};
      SDesc.Filter = d3d12Filter(S);
      D3D12_TEXTURE_ADDRESS_MODE Addr = d3d12AddressMode(S.Address);
      SDesc.AddressU = Addr;
      SDesc.AddressV = Addr;
      SDesc.AddressW = Addr;
      SDesc.MaxAnisotropy = 1;
      SDesc.ComparisonFunc = (S.Kind == SamplerKind::SamplerComparison)
                                 ? d3d12CompareFunc(S.ComparisonOp)
                                 : D3D12_COMPARISON_FUNC_NEVER;
      SDesc.MinLOD = S.MinLOD;
      SDesc.MaxLOD = S.MaxLOD;
      Dev->CreateSampler(&SDesc, {SStart.ptr + static_cast<SIZE_T>(I) * SInc});
    }
  }

  const float ClearColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  Cmd->ClearRenderTargetView(RTVHandle, ClearColor, 0, nullptr);
  Cmd->OMSetRenderTargets(1, &RTVHandle, FALSE, nullptr);

  D3D12_VIEWPORT VP{};
  if (B.getViewportCount() > 0) {
    auto V = B.getViewports().front();
    VP = {V.X, V.Y, V.Width, V.Height, V.MinDepth, V.MaxDepth};
  } else {
    VP = {0.0f, 0.0f, static_cast<float>(RTW), static_cast<float>(RTH),
          0.0f, 1.0f};
  }
  D3D12_RECT Scissor{0, 0, static_cast<LONG>(RTW), static_cast<LONG>(RTH)};
  Cmd->RSSetViewports(1, &VP);
  Cmd->RSSetScissorRects(1, &Scissor);
  Cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  if (VBBuf)
    Cmd->IASetVertexBuffers(0, 1, &VBView);

  ID3D12DescriptorHeap *Heaps[2] = {DescHeap, SamplerHeap};
  Cmd->SetDescriptorHeaps(SamplerHeap ? 2 : 1, Heaps);
  Cmd->SetGraphicsRootSignature(RootSig);
  Cmd->SetPipelineState(PSO);
  if (CbvSrvUavParamIdx >= 0)
    Cmd->SetGraphicsRootDescriptorTable(
        static_cast<UINT>(CbvSrvUavParamIdx),
        DescHeap->GetGPUDescriptorHandleForHeapStart());
  if (SamplerParamIdx >= 0 && SamplerHeap)
    Cmd->SetGraphicsRootDescriptorTable(
        static_cast<UINT>(SamplerParamIdx),
        SamplerHeap->GetGPUDescriptorHandleForHeapStart());

  UINT VertexCount = P.DispatchParameters.VertexCount.value_or(0);
  if (!VertexCount && B.VertexBufferPtr && Stride)
    VertexCount = static_cast<UINT>(B.VertexBufferPtr->Size / Stride);
  if (!VertexCount)
    VertexCount = 3;
  Cmd->DrawInstanced(VertexCount, 1, 0, 0);

  // Read back RenderTarget (resolve MSAA first if SampleCount > 1)
  ID3D12Resource *CopySrcTex = RTTex;
  if (SampleCount > 1 && ResolveTex) {
    D3D12_RESOURCE_BARRIER RTBar{};
    RTBar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    RTBar.Transition.pResource = RTTex;
    RTBar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    RTBar.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    RTBar.Transition.StateAfter = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
    Cmd->ResourceBarrier(1, &RTBar);
    Cmd->ResolveSubresource(ResolveTex, 0, RTTex, 0, RTFmt);
    D3D12_RESOURCE_BARRIER ResBar{};
    ResBar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    ResBar.Transition.pResource = ResolveTex;
    ResBar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    ResBar.Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_DEST;
    ResBar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    Cmd->ResourceBarrier(1, &ResBar);
    CopySrcTex = ResolveTex;
  } else {
    D3D12_RESOURCE_BARRIER RTBar{};
    RTBar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    RTBar.Transition.pResource = RTTex;
    RTBar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    RTBar.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    RTBar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    Cmd->ResourceBarrier(1, &RTBar);
  }
  D3D12_TEXTURE_COPY_LOCATION RTSrc{
      CopySrcTex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
  RTSrc.SubresourceIndex = 0;
  D3D12_TEXTURE_COPY_LOCATION RTDst{
      RTReadback, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
  RTDst.PlacedFootprint.Offset = 0;
  RTDst.PlacedFootprint.Footprint = {RTFmt, RTW, RTH, 1, RTAlignedPitch};
  Cmd->CopyTextureRegion(&RTDst, 0, 0, 0, &RTSrc, nullptr);

  // Read back any UAV buffers
  for (BoundResource &BR : Bound) {
    if (BR.Class != crucible::ResourceClass::UAV || !BR.ReadbackBuf)
      continue;
    D3D12_RESOURCE_BARRIER UB{};
    UB.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    UB.Transition.pResource = BR.DefaultBuf;
    UB.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    UB.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    UB.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    Cmd->ResourceBarrier(1, &UB);
    Cmd->CopyBufferRegion(BR.ReadbackBuf, 0, BR.DefaultBuf, 0, BR.AllocBytes);
  }

  Hr = Cmd->Close();
  if (FAILED(Hr))
    return hrErr(Hr, "CommandList::Close(Graphics)");
  ID3D12CommandList *Lists[] = {Cmd};
  Queue->ExecuteCommandLists(1, Lists);

  ID3D12Fence *Fence = nullptr;
  Hr = Dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence),
                        reinterpret_cast<void **>(&Fence));
  if (FAILED(Hr)) return hrErr(Hr, "CreateFence(Graphics)");
  Hr = Queue->Signal(Fence, 1);
  if (FAILED(Hr)) return hrErr(Hr, "Queue::Signal(Graphics)");
  if (Fence->GetCompletedValue() < 1) {
#ifdef _WIN32
    HANDLE Ev = CreateEventA(nullptr, FALSE, FALSE, nullptr);
    Hr = Fence->SetEventOnCompletion(1, Ev);
    if (SUCCEEDED(Hr) && Ev) {
      WaitForSingleObject(Ev, INFINITE);
      CloseHandle(Ev);
    }
#else
    Hr = Fence->SetEventOnCompletion(1, nullptr);
#endif
    if (FAILED(Hr)) return hrErr(Hr, "SetEventOnCompletion(Graphics)");
  }

  // Unpack RenderTarget into RTBuf->Data[0]
  if (!RTBuf->Data.empty() && RTBuf->Data[0]) {
    void *Mapped = nullptr;
    D3D12_RANGE RR{0, static_cast<SIZE_T>(RTReadbackBytes)};
    Hr = RTReadback->Map(0, &RR, &Mapped);
    if (FAILED(Hr)) return hrErr(Hr, "Map RT readback");
    char *Dst = RTBuf->Data[0].get();
    for (uint32_t Y = 0; Y < RTH; ++Y) {
      uint64_t DOff = static_cast<uint64_t>(Y) * RTRawPitch;
      uint64_t SOff = static_cast<uint64_t>(Y) * RTAlignedPitch;
      if (DOff + RTRawPitch <= RTBuf->Size)
        std::memcpy(Dst + DOff, static_cast<const char *>(Mapped) + SOff,
                    RTRawPitch);
    }
    D3D12_RANGE WR{0, 0};
    RTReadback->Unmap(0, &WR);
  }

  for (BoundResource &BR : Bound) {
    if (BR.Class != crucible::ResourceClass::UAV || !BR.ReadbackBuf)
      continue;
    CPUBuffer &CPU = *BR.Res->BufferPtr;
    void *Mapped = nullptr;
    D3D12_RANGE RR{0, static_cast<SIZE_T>(BR.SizeBytes)};
    Hr = BR.ReadbackBuf->Map(0, &RR, &Mapped);
    if (FAILED(Hr)) return hrErr(Hr, "Map UAV readback(Graphics)");
    if (BR.ArrIdx < CPU.Data.size() && CPU.Data[BR.ArrIdx])
      std::memcpy(CPU.Data[BR.ArrIdx].get(), Mapped, BR.SizeBytes);
    D3D12_RANGE WR{0, 0};
    BR.ReadbackBuf->Unmap(0, &WR);
  }

  return Error::success();
}

Error unsupportedD3D12() {
  return createStringError(
      std::errc::not_supported,
      "Crucible D3D12 offload proxy does not implement low-level API");
}

class CrucibleD3D12Device : public offloadtest::Device {
  ID3D12Device *D3DDev = nullptr;
  ID3D12CommandQueue *D3DQueue = nullptr;
  D3D12BackendCaps BackendCaps;
  Capabilities Caps;

public:
  CrucibleD3D12Device(ID3D12Device *Dev, ID3D12CommandQueue *Q,
                      D3D12BackendCaps BC)
      : D3DDev(Dev), D3DQueue(Q), BackendCaps(std::move(BC)) {
    Description = "Crucible (SPIR-V -> DXIL / " +
                  (BackendCaps.AdapterName.empty() ? std::string("D3D12")
                                                   : BackendCaps.AdapterName) +
                  ")";
    DriverName = "Crucible";
    DriverVersion = "0.0.1";
    GPUGeneration = BackendCaps.AdapterName.empty() ? "D3D12"
                                                    : BackendCaps.AdapterName;

    CrucibleEmulationCaps Emul;
    Emul.HighestShaderModel = BackendCaps.HighestShaderModel;
    Caps = buildVulkanCapsFromDirectX(BackendCaps, Emul);
  }

  const Capabilities &getCapabilities() override { return Caps; }
  GPUAPI getAPI() const override { return GPUAPI::Vulkan; }
  StringRef getAPIName() const override { return "Vulkan"; }
  void printExtra(raw_ostream &OS) override {
    OS << "  Backend: DirectX 12 (" << BackendCaps.AdapterName << ")\n"
       << "  HighestShaderModel: " << BackendCaps.HighestShaderModel << "\n";
    if (BackendCaps.VendorID || BackendCaps.DeviceID)
      OS << formatv("  VendorID: 0x{0:x}\n  DeviceID: 0x{1:x}\n",
                    BackendCaps.VendorID, BackendCaps.DeviceID);
    if (BackendCaps.WaveOps)
      OS << "  WaveLaneCount: " << BackendCaps.WaveLaneCountMin << ".."
         << BackendCaps.WaveLaneCountMax << "\n";
  }

  Error executeProgram(Pipeline &P) override {
    if (P.isCompute())
      return executeComputeD3D12(P, D3DDev, D3DQueue, BackendCaps);
    return executeGraphicsD3D12(P, D3DDev, D3DQueue, BackendCaps);
  }

  Queue &getGraphicsQueue() override {
    report_fatal_error("Crucible D3D12 does not implement low-level queue");
  }
  Expected<std::unique_ptr<PipelineState>>
  createPipelineCs(StringRef, const BindingsDesc &, ShaderContainer) override {
    return unsupportedD3D12();
  }
  Expected<std::unique_ptr<PipelineState>> createTraditionalRasterPipeline(
      StringRef, const BindingsDesc &,
      const TraditionalRasterPipelineCreateDesc &) override {
    return unsupportedD3D12();
  }
  Expected<std::unique_ptr<PipelineState>> createMeshShaderRasterPipeline(
      StringRef, const BindingsDesc &,
      const MeshShaderRasterPipelineCreateDesc &) override {
    return unsupportedD3D12();
  }
  Expected<std::unique_ptr<PipelineState>> createPipelineRT(
      StringRef, const BindingsDesc &,
      const RayTracingPipelineCreateDesc &) override {
    return unsupportedD3D12();
  }
  Expected<std::unique_ptr<ShaderBindingTable>> createShaderBindingTable(
      const PipelineState &, const ShaderBindingTableDesc &) override {
    return unsupportedD3D12();
  }
  Expected<std::unique_ptr<Fence>> createFence(StringRef) override {
    return unsupportedD3D12();
  }
  Expected<std::unique_ptr<Buffer>>
  createBuffer(std::string, const BufferCreateDesc &, size_t) override {
    return unsupportedD3D12();
  }
  Expected<std::unique_ptr<Texture>>
  createTexture(std::string, const TextureCreateDesc &) override {
    return unsupportedD3D12();
  }
  Expected<std::unique_ptr<Sampler>>
  createSampler(std::string, const SamplerCreateDesc &) override {
    return unsupportedD3D12();
  }
  Expected<std::unique_ptr<MemoryHeap>>
  createMemoryHeap(std::string, size_t) override {
    return unsupportedD3D12();
  }
  uint32_t getTextureUploadRowStrideInBytes(
      const TextureCreateDesc &) const override {
    report_fatal_error("Crucible D3D12 does not implement low-level texture");
  }
  TextureUploadLayout getTextureUploadLayout(
      const TextureCreateDesc &) const override {
    report_fatal_error("Crucible D3D12 does not implement low-level texture");
  }
  Expected<std::unique_ptr<RenderPass>>
  createRenderPass(const RenderPassDesc &) override {
    return unsupportedD3D12();
  }
  Expected<std::unique_ptr<CommandBuffer>> createCommandBuffer() override {
    return unsupportedD3D12();
  }
  Expected<AccelerationStructureSizes>
  getBLASBuildSizes(ArrayRef<TriangleGeometryDesc>) override {
    return unsupportedD3D12();
  }
  Expected<AccelerationStructureSizes>
  getBLASBuildSizes(ArrayRef<AABBGeometryDesc>) override {
    return unsupportedD3D12();
  }
  Expected<AccelerationStructureSizes> getTLASBuildSizes(uint32_t) override {
    return unsupportedD3D12();
  }
  Expected<std::unique_ptr<AccelerationStructure>>
  createBLAS(const AccelerationStructureSizes &) override {
    return unsupportedD3D12();
  }
  Expected<std::unique_ptr<AccelerationStructure>>
  createTLAS(const AccelerationStructureSizes &) override {
    return unsupportedD3D12();
  }
};

Expected<std::unique_ptr<offloadtest::Device>>
createProxiedDeviceForAdapter(IUnknown *Adapter, StringRef AdapterName,
                              uint32_t VendorID, uint32_t DeviceID) {
  D3D12Loader &L = getD3D12Loader();
  ID3D12Device *Dev = nullptr;
  HRESULT Hr = L.CreateDevice(Adapter, D3D_FEATURE_LEVEL_11_0,
                              __uuidof(ID3D12Device),
                              reinterpret_cast<void **>(&Dev));
  if (FAILED(Hr))
    return hrErr(Hr, "D3D12CreateDevice");

  D3D12_COMMAND_QUEUE_DESC QDesc{};
  QDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  ID3D12CommandQueue *Queue = nullptr;
  Hr = Dev->CreateCommandQueue(&QDesc, __uuidof(ID3D12CommandQueue),
                               reinterpret_cast<void **>(&Queue));
  if (FAILED(Hr)) {
    Dev->Release();
    return hrErr(Hr, "CreateCommandQueue");
  }

  D3D12BackendCaps BC =
      probeD3D12Capabilities(Dev, AdapterName, VendorID, DeviceID);
  return std::make_unique<CrucibleD3D12Device>(Dev, Queue, std::move(BC));
}

} // namespace

namespace crucible_offload {

Expected<SmallVector<std::unique_ptr<offloadtest::Device>>>
createD3D12Devices(const DeviceConfig &) {
  if (Error E = ensureD3D12Loaded())
    return std::move(E);

  SmallVector<std::unique_ptr<offloadtest::Device>> Devices;

#ifdef _WIN32
  // On Windows (when not overriding with a custom D3D12 DLL), enumerate real
  // DXGI adapters (NVIDIA, AMD, Intel, WARP) or select WARP explicitly when
  // CRUCIBLE_OFFLOAD_RUNTIME=warp or CRUCIBLE_D3D12_WARP=1.
  const char *OverrideLib = std::getenv("CRUCIBLE_D3D12_LIB");
  if (!OverrideLib || !*OverrideLib) {
    bool WantWarp = false;
    if (const char *Rt = std::getenv("CRUCIBLE_OFFLOAD_RUNTIME")) {
      if (StringRef(Rt).equals_insensitive("warp"))
        WantWarp = true;
    }
    if (const char *W = std::getenv("CRUCIBLE_D3D12_WARP")) {
      if (W[0] && W[0] != '0')
        WantWarp = true;
    }

    auto adapterNameFromDesc = [](const DXGI_ADAPTER_DESC1 &Desc) {
      std::string Name;
      for (wchar_t WC : Desc.Description) {
        if (!WC)
          break;
        Name.push_back(WC < 128 ? static_cast<char>(WC) : '?');
      }
      return Name;
    };

    if (HMODULE DXGI = LoadLibraryA("dxgi.dll")) {
      using PFN_CreateDXGIFactory1 = HRESULT(WINAPI *)(REFIID, void **);
      auto CreateFactory = reinterpret_cast<PFN_CreateDXGIFactory1>(
          GetProcAddress(DXGI, "CreateDXGIFactory1"));
      IDXGIFactory4 *Factory = nullptr;
      if (CreateFactory &&
          SUCCEEDED(CreateFactory(__uuidof(IDXGIFactory4),
                                  reinterpret_cast<void **>(&Factory)))) {
        if (WantWarp) {
          IDXGIAdapter1 *WarpAdapter = nullptr;
          HRESULT Hr = Factory->EnumWarpAdapter(
              __uuidof(IDXGIAdapter1),
              reinterpret_cast<void **>(&WarpAdapter));
          Factory->Release();
          if (FAILED(Hr))
            return hrErr(Hr, "EnumWarpAdapter");
          DXGI_ADAPTER_DESC1 Desc{};
          WarpAdapter->GetDesc1(&Desc);
          auto DevOrErr = createProxiedDeviceForAdapter(
              WarpAdapter, adapterNameFromDesc(Desc), Desc.VendorId,
              Desc.DeviceId);
          WarpAdapter->Release();
          if (!DevOrErr)
            return DevOrErr.takeError();
          Devices.push_back(std::move(*DevOrErr));
          return Devices;
        }
        for (UINT Idx = 0;; ++Idx) {
          IDXGIAdapter1 *Adapter = nullptr;
          if (Factory->EnumAdapters1(Idx, &Adapter) == DXGI_ERROR_NOT_FOUND)
            break;
          DXGI_ADAPTER_DESC1 Desc{};
          Adapter->GetDesc1(&Desc);
          if (auto DevOrErr = createProxiedDeviceForAdapter(
                  Adapter, adapterNameFromDesc(Desc), Desc.VendorId,
                  Desc.DeviceId))
            Devices.push_back(std::move(*DevOrErr));
          else
            consumeError(DevOrErr.takeError());
          Adapter->Release();
        }
        Factory->Release();
      }
    }
    if (WantWarp)
      return createStringError(
          inconvertibleErrorCode(),
          "WARP adapter requested but dxgi.dll / IDXGIFactory4 unavailable");
  }
#endif

  if (Devices.empty()) {
    // Default adapter (Nacelle on macOS/Linux, or custom CRUCIBLE_D3D12_LIB).
    D3D12Loader &L = getD3D12Loader();
    std::string DefaultName =
        StringRef(L.RuntimePath).contains_insensitive("nacelle")
            ? "Nacelle D3D12"
            : "D3D12 Default Adapter";
    auto DevOrErr = createProxiedDeviceForAdapter(nullptr, DefaultName, 0, 0);
    if (!DevOrErr)
      return DevOrErr.takeError();
    Devices.push_back(std::move(*DevOrErr));
  }

  return Devices;
}

} // namespace crucible_offload

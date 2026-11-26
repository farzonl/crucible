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

DXGI_FORMAT dxgiFormatForBuffer(const CPUBuffer &Buf) {
  unsigned Ch = Buf.Channels ? static_cast<unsigned>(Buf.Channels) : 1u;
  switch (Buf.Format) {
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
  default:
    return DXGI_FORMAT_R32_UINT;
  }
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
  for (size_t SetIdx = 0; SetIdx < P.Sets.size(); ++SetIdx) {
    for (const offloadtest::Resource &Res : P.Sets[SetIdx].Resources) {
      if (Res.isSampler() || Res.isTexture() || Res.isAccelerationStructure())
        return createStringError(
            std::errc::not_supported,
            "D3D12 offload proxy does not yet support texture/sampler/AS "
            "resource '%s'",
            Res.Name.c_str());
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
      for (const crucible::ResourceBinding &RB : Compiled->Reflect.Resources) {
        if (RB.Space == Space && RB.LowerBound == BaseReg &&
            RB.Class == Class) {
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
        BR.AllocBytes = (Class == crucible::ResourceClass::CBV)
                            ? ((BR.SizeBytes + 255ull) & ~255ull)
                            : std::max<uint64_t>(BR.SizeBytes, 4ull);
        Bound.push_back(BR);
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
  for (uint32_t I = 0; I < Bound.size(); ++I) {
    D3D12_DESCRIPTOR_RANGE R{};
    switch (Bound[I].Class) {
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
    R.NumDescriptors = 1;
    R.BaseShaderRegister = Bound[I].Register;
    R.RegisterSpace = Bound[I].Space;
    R.OffsetInDescriptorsFromTableStart = I;
    Ranges.push_back(R);
  }

  D3D12_ROOT_PARAMETER RootParam{};
  RootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  RootParam.DescriptorTable.NumDescriptorRanges =
      static_cast<UINT>(Ranges.size());
  RootParam.DescriptorTable.pDescriptorRanges =
      Ranges.empty() ? nullptr : Ranges.data();
  RootParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

  D3D12_ROOT_SIGNATURE_DESC RSDesc{};
  RSDesc.NumParameters = Ranges.empty() ? 0 : 1;
  RSDesc.pParameters = Ranges.empty() ? nullptr : &RootParam;
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
    const CPUBuffer &CPU = *BR.Res->BufferPtr;
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
      if (BR.ArrIdx < CPU.Data.size() && CPU.Data[BR.ArrIdx])
        std::memcpy(Mapped, CPU.Data[BR.ArrIdx].get(), BR.SizeBytes);
      BR.UploadBuf->Unmap(0, nullptr);

      D3D12_CONSTANT_BUFFER_VIEW_DESC CBVDesc{};
      CBVDesc.BufferLocation = BR.UploadBuf->GetGPUVirtualAddress();
      CBVDesc.SizeInBytes = static_cast<UINT>(BR.AllocBytes);
      Dev->CreateConstantBufferView(&CBVDesc, Slot);
    } else {
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
      if (BR.ArrIdx < CPU.Data.size() && CPU.Data[BR.ArrIdx])
        std::memcpy(Mapped, CPU.Data[BR.ArrIdx].get(), BR.SizeBytes);
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

        if (BR.Res->HasCounter) {
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
          uint32_t Stride = std::max<uint32_t>(4u, CPU.getElementSize());
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
          uint32_t Stride = std::max<uint32_t>(4u, CPU.getElementSize());
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

  ID3D12DescriptorHeap *Heaps[] = {DescHeap};
  Cmd->SetDescriptorHeaps(1, Heaps);
  Cmd->SetComputeRootSignature(RootSig);
  Cmd->SetPipelineState(PSO);
  if (!Ranges.empty())
    Cmd->SetComputeRootDescriptorTable(
        0, DescHeap->GetGPUDescriptorHandleForHeapStart());

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
    Cmd->CopyBufferRegion(BR.ReadbackBuf, 0, BR.DefaultBuf, 0, BR.AllocBytes);

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
    D3D12_RANGE ReadRange{0, static_cast<SIZE_T>(BR.SizeBytes)};
    Hr = BR.ReadbackBuf->Map(0, &ReadRange, &Mapped);
    if (FAILED(Hr))
      return hrErr(Hr, "Map readback");
    if (BR.ArrIdx < CPU.Data.size() && CPU.Data[BR.ArrIdx])
      std::memcpy(CPU.Data[BR.ArrIdx].get(), Mapped, BR.SizeBytes);
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
    if (!P.isCompute())
      return createStringError(
          std::errc::not_supported,
          "Crucible D3D12 offload proxy currently supports compute pipelines "
          "only");
    return executeComputeD3D12(P, D3DDev, D3DQueue, BackendCaps);
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

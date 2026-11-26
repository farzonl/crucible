//===- D3D12Backend.cpp - Nacelle D3D12 compute backend for crucible-run -===//
//
// Part of Crucible, an open-source multi-frontend, multi-backend shader compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// D3D12 compute dispatch harness that dynamically loads Nacelle's
// libnacelle_d3d12 shared library (macOS/Linux) so DXIL shaders emitted by
// Crucible's DirectX backend can be executed end-to-end.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

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
#include <unistd.h>
#include <wsl/winadapter.h>
#endif
#include <directx/d3d12.h>
#include <dxguids/dxguids.h>

#ifdef interface
#undef interface
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace crucible_run {
namespace {

using PFN_D3D12CreateDevice = HRESULT(WINAPI *)(IUnknown *, D3D_FEATURE_LEVEL,
                                                REFIID, void **);
using PFN_D3D12SerializeRootSignature =
    HRESULT(WINAPI *)(const D3D12_ROOT_SIGNATURE_DESC *,
                      D3D_ROOT_SIGNATURE_VERSION, ID3DBlob **, ID3DBlob **);

void checkHr(HRESULT Hr, const char *What) {
  if (FAILED(Hr)) {
    char Buf[64];
    snprintf(Buf, sizeof(Buf), "0x%08x", static_cast<uint32_t>(Hr));
    fail(std::string(What) + " failed (HRESULT=" + Buf + ")");
  }
}

std::vector<uint8_t> readBinary(const std::string &Path) {
  std::ifstream F(Path, std::ios::binary | std::ios::ate);
  if (!F)
    fail("cannot open DXIL file '" + Path + "'");
  std::streamsize Size = F.tellg();
  if (Size <= 0)
    fail("empty DXIL file '" + Path + "'");
  F.seekg(0, std::ios::beg);
  std::vector<uint8_t> Data(static_cast<size_t>(Size));
  if (!F.read(reinterpret_cast<char *>(Data.data()), Size))
    fail("failed reading '" + Path + "'");
  return Data;
}

ID3D12Resource *createBuffer(ID3D12Device *Dev, D3D12_HEAP_TYPE HeapType,
                             uint64_t SizeBytes, D3D12_RESOURCE_FLAGS Flags,
                             D3D12_RESOURCE_STATES InitState) {
  D3D12_HEAP_PROPERTIES Heap{};
  Heap.Type = HeapType;
  Heap.CreationNodeMask = 1;
  Heap.VisibleNodeMask = 1;

  D3D12_RESOURCE_DESC Desc{};
  Desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  Desc.Width = SizeBytes;
  Desc.Height = 1;
  Desc.DepthOrArraySize = 1;
  Desc.MipLevels = 1;
  Desc.Format = DXGI_FORMAT_UNKNOWN;
  Desc.SampleDesc = {1, 0};
  Desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  Desc.Flags = Flags;

  ID3D12Resource *Res = nullptr;
  checkHr(Dev->CreateCommittedResource(&Heap, D3D12_HEAP_FLAG_NONE, &Desc,
                                       InitState, nullptr, IID_PPV_ARGS(&Res)),
          "CreateCommittedResource");
  return Res;
}

struct AllocatedBuffer {
  BufferSpec Spec;
  ID3D12Resource *DefaultBuf = nullptr;
  ID3D12Resource *UploadBuf = nullptr;
  ID3D12Resource *ReadbackBuf = nullptr;
  uint64_t SizeBytes = 0;
  uint64_t AllocBytes = 0;
};

} // namespace

int runD3D12(const RunnerOptions &Opts) {
  if (Opts.DxilPath.empty())
    fail("--dxil <file.dxil> is required for d3d12 backend");

  const char *LibPath = std::getenv("CRUCIBLE_D3D12_LIB");
  if (!LibPath || !*LibPath)
    LibPath = std::getenv("CRUCIBLE_NACELLE_LIB");

  PFN_D3D12CreateDevice CreateDeviceFn = nullptr;
  PFN_D3D12SerializeRootSignature SerializeRSFn = nullptr;
#ifdef _WIN32
  if (!LibPath || !*LibPath)
    LibPath = "d3d12.dll";
  HMODULE Lib = LoadLibraryA(LibPath);
  if (!Lib)
    fail(std::string("failed to LoadLibrary D3D12 runtime '") + LibPath + "'");
  crucible_win_dxil::enableExperimentalShaderModels(Lib);
  CreateDeviceFn = reinterpret_cast<PFN_D3D12CreateDevice>(
      GetProcAddress(Lib, "D3D12CreateDevice"));
  SerializeRSFn = reinterpret_cast<PFN_D3D12SerializeRootSignature>(
      GetProcAddress(Lib, "D3D12SerializeRootSignature"));
#else
  if (!LibPath || !*LibPath) {
#ifdef CRUCIBLE_NACELLE_LIB_PATH
    LibPath = CRUCIBLE_NACELLE_LIB_PATH;
#else
    fail("CRUCIBLE_D3D12_LIB / CRUCIBLE_NACELLE_LIB not set");
#endif
  }
  void *Lib = dlopen(LibPath, RTLD_NOW | RTLD_LOCAL);
  if (!Lib)
    fail(std::string("failed to dlopen Nacelle D3D12 runtime '") + LibPath +
         "': " + (dlerror() ? dlerror() : "unknown error"));
  CreateDeviceFn = reinterpret_cast<PFN_D3D12CreateDevice>(
      dlsym(Lib, "D3D12CreateDevice"));
  SerializeRSFn = reinterpret_cast<PFN_D3D12SerializeRootSignature>(
      dlsym(Lib, "D3D12SerializeRootSignature"));
#endif
  if (!CreateDeviceFn || !SerializeRSFn)
    fail("D3D12 runtime missing D3D12CreateDevice or "
         "D3D12SerializeRootSignature");

  std::vector<uint8_t> Bytecode = readBinary(Opts.DxilPath);
#ifdef _WIN32
  std::string SignErr;
  if (!crucible_win_dxil::signDXILInPlace(Bytecode.data(), Bytecode.size(),
                                          SignErr))
    fail("DXIL validation/signing failed: " + SignErr);
#endif

  IUnknown *Adapter = nullptr;
#ifdef _WIN32
  if (const char *WarpEnv = std::getenv("CRUCIBLE_D3D12_WARP");
      WarpEnv && WarpEnv[0] && WarpEnv[0] != '0') {
    HMODULE DXGI = LoadLibraryA("dxgi.dll");
    if (!DXGI)
      fail("failed to LoadLibrary dxgi.dll for WARP");
    using PFN_CreateDXGIFactory1 = HRESULT(WINAPI *)(REFIID, void **);
    auto CreateFactory = reinterpret_cast<PFN_CreateDXGIFactory1>(
        GetProcAddress(DXGI, "CreateDXGIFactory1"));
    IDXGIFactory4 *Factory = nullptr;
    if (!CreateFactory)
      fail("dxgi.dll missing CreateDXGIFactory1");
    checkHr(CreateFactory(__uuidof(IDXGIFactory4),
                          reinterpret_cast<void **>(&Factory)),
            "CreateDXGIFactory1");
    IDXGIAdapter1 *WarpAdapter = nullptr;
    checkHr(Factory->EnumWarpAdapter(__uuidof(IDXGIAdapter1),
                                     reinterpret_cast<void **>(&WarpAdapter)),
            "EnumWarpAdapter");
    Factory->Release();
    Adapter = WarpAdapter;
  }
#endif

  ID3D12Device *Dev = nullptr;
  checkHr(CreateDeviceFn(Adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&Dev)),
          "D3D12CreateDevice");

  D3D12_COMMAND_QUEUE_DESC QDesc{};
  QDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  ID3D12CommandQueue *Queue = nullptr;
  checkHr(Dev->CreateCommandQueue(&QDesc, IID_PPV_ARGS(&Queue)),
          "CreateCommandQueue");

  ID3D12CommandAllocator *Alloc = nullptr;
  checkHr(Dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                      IID_PPV_ARGS(&Alloc)),
          "CreateCommandAllocator");

  // Build root signature with one root descriptor per --buffer (UAV u#) and
  // --cbuffer (CBV b#).
  std::vector<D3D12_ROOT_PARAMETER> RootParams;
  RootParams.reserve(Opts.Buffers.size());
  for (const BufferSpec &Spec : Opts.Buffers) {
    D3D12_ROOT_PARAMETER P{};
    P.ParameterType = Spec.IsUniform ? D3D12_ROOT_PARAMETER_TYPE_CBV
                                     : D3D12_ROOT_PARAMETER_TYPE_UAV;
    P.Descriptor.ShaderRegister = Spec.Index;
    P.Descriptor.RegisterSpace = 0;
    P.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    RootParams.push_back(P);
  }

  D3D12_ROOT_SIGNATURE_DESC RSDesc{};
  RSDesc.NumParameters = static_cast<UINT>(RootParams.size());
  RSDesc.pParameters = RootParams.empty() ? nullptr : RootParams.data();
  RSDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

  ID3DBlob *RSBlob = nullptr;
  ID3DBlob *ErrBlob = nullptr;
  HRESULT RSHr = SerializeRSFn(&RSDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &RSBlob,
                               &ErrBlob);
  if (FAILED(RSHr)) {
    std::string ErrMsg = "D3D12SerializeRootSignature failed";
    if (ErrBlob) {
      ErrMsg += ": ";
      ErrMsg.append(static_cast<const char *>(ErrBlob->GetBufferPointer()),
                    ErrBlob->GetBufferSize());
      ErrBlob->Release();
    }
    fail(ErrMsg);
  }

  ID3D12RootSignature *RootSig = nullptr;
  checkHr(Dev->CreateRootSignature(0, RSBlob->GetBufferPointer(),
                                   RSBlob->GetBufferSize(),
                                   IID_PPV_ARGS(&RootSig)),
          "CreateRootSignature");
  RSBlob->Release();

  D3D12_COMPUTE_PIPELINE_STATE_DESC PSODesc{};
  PSODesc.pRootSignature = RootSig;
  PSODesc.CS.pShaderBytecode = Bytecode.data();
  PSODesc.CS.BytecodeLength = Bytecode.size();

  ID3D12PipelineState *PSO = nullptr;
  checkHr(Dev->CreateComputePipelineState(&PSODesc, IID_PPV_ARGS(&PSO)),
          "CreateComputePipelineState");

  ID3D12GraphicsCommandList *Cmd = nullptr;
  checkHr(Dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, Alloc, PSO,
                                 IID_PPV_ARGS(&Cmd)),
          "CreateCommandList");

  // Allocate and populate buffers.
  std::vector<AllocatedBuffer> Bufs;
  Bufs.reserve(Opts.Buffers.size());
  for (const BufferSpec &Spec : Opts.Buffers) {
    AllocatedBuffer AB;
    AB.Spec = Spec;
    AB.SizeBytes = Spec.Count * 4ull;
    // CBVs require 256-byte alignment in D3D12.
    AB.AllocBytes = Spec.IsUniform ? ((AB.SizeBytes + 255ull) & ~255ull)
                                   : std::max<uint64_t>(AB.SizeBytes, 4ull);
    if (Spec.IsUniform) {
      AB.UploadBuf = createBuffer(Dev, D3D12_HEAP_TYPE_UPLOAD, AB.AllocBytes,
                                  D3D12_RESOURCE_FLAG_NONE,
                                  D3D12_RESOURCE_STATE_GENERIC_READ);
      void *Mapped = nullptr;
      checkHr(AB.UploadBuf->Map(0, nullptr, &Mapped), "Map CBV upload");
      std::memset(Mapped, 0, AB.AllocBytes);
      fillBuffer(Spec, Mapped);
      AB.UploadBuf->Unmap(0, nullptr);
    } else {
      AB.DefaultBuf = createBuffer(Dev, D3D12_HEAP_TYPE_DEFAULT, AB.AllocBytes,
                                   D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_COPY_DEST);
      AB.UploadBuf = createBuffer(Dev, D3D12_HEAP_TYPE_UPLOAD, AB.AllocBytes,
                                  D3D12_RESOURCE_FLAG_NONE,
                                  D3D12_RESOURCE_STATE_GENERIC_READ);
      AB.ReadbackBuf = createBuffer(Dev, D3D12_HEAP_TYPE_READBACK,
                                    AB.AllocBytes, D3D12_RESOURCE_FLAG_NONE,
                                    D3D12_RESOURCE_STATE_COPY_DEST);
      void *Mapped = nullptr;
      checkHr(AB.UploadBuf->Map(0, nullptr, &Mapped), "Map UAV upload");
      std::memset(Mapped, 0, AB.AllocBytes);
      fillBuffer(Spec, Mapped);
      AB.UploadBuf->Unmap(0, nullptr);
      Cmd->CopyBufferRegion(AB.DefaultBuf, 0, AB.UploadBuf, 0, AB.AllocBytes);

      D3D12_RESOURCE_BARRIER Barrier{};
      Barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      Barrier.Transition.pResource = AB.DefaultBuf;
      Barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      Barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
      Barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      Cmd->ResourceBarrier(1, &Barrier);
    }
    Bufs.push_back(AB);
  }

  Cmd->SetComputeRootSignature(RootSig);
  Cmd->SetPipelineState(PSO);
  for (size_t I = 0; I < Bufs.size(); ++I) {
    if (Bufs[I].Spec.IsUniform) {
      Cmd->SetComputeRootConstantBufferView(
          static_cast<UINT>(I), Bufs[I].UploadBuf->GetGPUVirtualAddress());
    } else {
      Cmd->SetComputeRootUnorderedAccessView(
          static_cast<UINT>(I), Bufs[I].DefaultBuf->GetGPUVirtualAddress());
    }
  }

  Cmd->Dispatch(Opts.Groups[0], Opts.Groups[1], Opts.Groups[2]);

  for (AllocatedBuffer &AB : Bufs) {
    if (AB.Spec.IsUniform)
      continue;
    D3D12_RESOURCE_BARRIER Barrier{};
    Barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    Barrier.Transition.pResource = AB.DefaultBuf;
    Barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    Barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    Barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    Cmd->ResourceBarrier(1, &Barrier);
    Cmd->CopyBufferRegion(AB.ReadbackBuf, 0, AB.DefaultBuf, 0, AB.AllocBytes);
  }

  checkHr(Cmd->Close(), "CommandList::Close");

  ID3D12CommandList *Lists[] = {Cmd};
  Queue->ExecuteCommandLists(1, Lists);

  ID3D12Fence *Fence = nullptr;
  checkHr(Dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&Fence)),
          "CreateFence");
  checkHr(Queue->Signal(Fence, 1), "Queue::Signal");
  if (Fence->GetCompletedValue() < 1)
    checkHr(Fence->SetEventOnCompletion(1, nullptr), "SetEventOnCompletion");

  for (AllocatedBuffer &AB : Bufs) {
    if (AB.Spec.IsUniform)
      continue;
    void *Mapped = nullptr;
    D3D12_RANGE ReadRange{0, static_cast<SIZE_T>(AB.SizeBytes)};
    checkHr(AB.ReadbackBuf->Map(0, &ReadRange, &Mapped), "Map readback");
    printBuffer(AB.Spec, Mapped);
    D3D12_RANGE WriteRange{0, 0};
    AB.ReadbackBuf->Unmap(0, &WriteRange);
  }

  std::fflush(stdout);
  std::fflush(stderr);
  // Nacelle's background -O3 JIT promotion thread may still be referencing the
  // PSO/device after a short dispatch; exit immediately after flushing output
  // rather than racing with device teardown.
  _exit(0);
}

} // namespace crucible_run

//===- D3D12Backend.cpp - Nacelle D3D12 compute backend for crucible-run -===//
//
// Part of Crucible, an open-source multi-frontend, multi-backend shader compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Self-contained D3D12 compute dispatch harness that dynamically loads
// Nacelle's libnacelle_d3d12 shared library (macOS/Linux) so DXIL shaders
// emitted by Crucible's DirectX backend can be executed end-to-end without
// requiring Windows or DirectX-Headers at build time.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace crucible_run {
// Keep COM interface types in a named namespace with external linkage. Placing
// pure-virtual interfaces in an anonymous namespace makes GCC -fdevirtualize
// treat the hierarchy as closed to this TU and devirtualize calls directly to
// __cxa_pure_virtual.
namespace d3d12_abi {

using HRESULT = int32_t;
using UINT = uint32_t;
using UINT64 = uint64_t;
using SIZE_T = size_t;
using D3D12_GPU_VIRTUAL_ADDRESS = uint64_t;

struct GUID {
  uint32_t Data1;
  uint16_t Data2;
  uint16_t Data3;
  uint8_t Data4[8];
};
using REFIID = const GUID &;

static constexpr GUID IID_ID3D12Device = {
    0x189819f1, 0x1db6, 0x4b57, {0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7}};
static constexpr GUID IID_ID3D12CommandQueue = {
    0x0ec870a6, 0x5d7e, 0x4c22, {0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed}};
static constexpr GUID IID_ID3D12CommandAllocator = {
    0x6102dee4, 0xaf59, 0x4b09, {0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24}};
static constexpr GUID IID_ID3D12GraphicsCommandList = {
    0x5b160d0f, 0xac1b, 0x4185, {0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55}};
static constexpr GUID IID_ID3D12RootSignature = {
    0xc54a6b66, 0x72df, 0x4ee8, {0x8b, 0xe5, 0xa9, 0x46, 0xa1, 0x42, 0x92, 0x14}};
static constexpr GUID IID_ID3D12PipelineState = {
    0x765a30f3, 0xf624, 0x4c6f, {0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45}};
static constexpr GUID IID_ID3D12Resource = {
    0x696442be, 0xa72e, 0x4059, {0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad}};
static constexpr GUID IID_ID3D12Fence = {
    0x0a753dcf, 0xc4d8, 0x4b91, {0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76}};

// Minimal D3D12 ABI structs matching DirectX-Headers on LP64.
struct D3D12_COMMAND_QUEUE_DESC {
  int32_t Type;
  int32_t Priority;
  int32_t Flags;
  UINT NodeMask;
};

struct D3D12_ROOT_DESCRIPTOR {
  UINT ShaderRegister;
  UINT RegisterSpace;
};

struct D3D12_ROOT_PARAMETER {
  int32_t ParameterType; // 2=CBV, 3=SRV, 4=UAV
  union {
    D3D12_ROOT_DESCRIPTOR Descriptor;
    uint64_t Padding[2];
  };
  int32_t ShaderVisibility; // 0=ALL
};

struct D3D12_ROOT_SIGNATURE_DESC {
  UINT NumParameters;
  const D3D12_ROOT_PARAMETER *pParameters;
  UINT NumStaticSamplers;
  const void *pStaticSamplers;
  int32_t Flags;
};

struct D3D12_SHADER_BYTECODE {
  const void *pShaderBytecode;
  SIZE_T BytecodeLength;
};

struct D3D12_CACHED_PIPELINE_STATE {
  const void *pCachedBlob;
  SIZE_T CachedBlobSizeInBytes;
};

struct ID3D12RootSignature;
struct ID3D12PipelineState;
struct ID3D12CommandAllocator;
struct ID3D12CommandList;
struct ID3D12GraphicsCommandList;
struct ID3D12CommandQueue;
struct ID3D12Resource;
struct ID3D12Fence;

struct D3D12_COMPUTE_PIPELINE_STATE_DESC {
  ID3D12RootSignature *pRootSignature;
  D3D12_SHADER_BYTECODE CS;
  UINT NodeMask;
  D3D12_CACHED_PIPELINE_STATE CachedPSO;
  int32_t Flags;
};

struct D3D12_HEAP_PROPERTIES {
  int32_t Type; // 1=DEFAULT, 2=UPLOAD, 3=READBACK
  int32_t CPUPageProperty;
  int32_t MemoryPoolPreference;
  UINT CreationNodeMask;
  UINT VisibleNodeMask;
};

struct DXGI_SAMPLE_DESC {
  UINT Count;
  UINT Quality;
};

struct D3D12_RESOURCE_DESC {
  int32_t Dimension; // 1=BUFFER
  UINT64 Alignment;
  UINT64 Width;
  UINT Height;
  uint16_t DepthOrArraySize;
  uint16_t MipLevels;
  int32_t Format; // 0=UNKNOWN
  DXGI_SAMPLE_DESC SampleDesc;
  int32_t Layout; // 1=ROW_MAJOR
  int32_t Flags;  // 4=ALLOW_UNORDERED_ACCESS
};

struct D3D12_RANGE {
  SIZE_T Begin;
  SIZE_T End;
};

struct D3D12_RESOURCE_TRANSITION_BARRIER {
  ID3D12Resource *pResource;
  UINT Subresource;
  int32_t StateBefore;
  int32_t StateAfter;
};

struct D3D12_RESOURCE_BARRIER {
  int32_t Type; // 0=TRANSITION, 2=UAV
  int32_t Flags;
  union {
    D3D12_RESOURCE_TRANSITION_BARRIER Transition;
    ID3D12Resource *UAVResource;
  };
};

// Minimal COM interfaces with exact D3D12 vtable slot layout.
struct IUnknown {
  virtual HRESULT QueryInterface(REFIID, void **) = 0;
  virtual UINT AddRef() = 0;
  virtual UINT Release() = 0;
};

struct ID3D10Blob : IUnknown {
  virtual void *GetBufferPointer() = 0;
  virtual SIZE_T GetBufferSize() = 0;
};

struct ID3D12Object : IUnknown {
  virtual HRESULT GetPrivateData(REFIID, UINT *, void *) = 0;
  virtual HRESULT SetPrivateData(REFIID, UINT, const void *) = 0;
  virtual HRESULT SetPrivateDataInterface(REFIID, const IUnknown *) = 0;
  virtual HRESULT SetName(const wchar_t *) = 0;
};

struct ID3D12DeviceChild : ID3D12Object {
  virtual HRESULT GetDevice(REFIID, void **) = 0;
};

struct ID3D12Pageable : ID3D12DeviceChild {};
struct ID3D12RootSignature : ID3D12DeviceChild {};
struct ID3D12PipelineState : ID3D12Pageable {};

struct ID3D12CommandAllocator : ID3D12Pageable {
  virtual HRESULT Reset() = 0;
};

struct ID3D12Fence : ID3D12Pageable {
  virtual UINT64 GetCompletedValue() = 0;
  virtual HRESULT SetEventOnCompletion(UINT64 Value, void *hEvent) = 0;
  virtual HRESULT Signal(UINT64 Value) = 0;
};

struct ID3D12Resource : ID3D12Pageable {
  virtual HRESULT Map(UINT Subresource, const D3D12_RANGE *pReadRange,
                      void **ppData) = 0;
  virtual void Unmap(UINT Subresource, const D3D12_RANGE *pWrittenRange) = 0;
  virtual D3D12_RESOURCE_DESC GetDesc() = 0;
  virtual D3D12_GPU_VIRTUAL_ADDRESS GetGPUVirtualAddress() = 0;
};

struct ID3D12CommandList : ID3D12DeviceChild {
  virtual int32_t GetType() = 0;
};

struct ID3D12GraphicsCommandList : ID3D12CommandList {
  virtual HRESULT Close() = 0;
  virtual HRESULT Reset(ID3D12CommandAllocator *, ID3D12PipelineState *) = 0;
  virtual void ClearState(ID3D12PipelineState *) = 0;
  virtual void DrawInstanced(UINT, UINT, UINT, UINT) = 0;
  virtual void DrawIndexedInstanced(UINT, UINT, UINT, int32_t, UINT) = 0;
  virtual void Dispatch(UINT X, UINT Y, UINT Z) = 0;
  virtual void CopyBufferRegion(ID3D12Resource *pDst, UINT64 DstOff,
                                ID3D12Resource *pSrc, UINT64 SrcOff,
                                UINT64 NumBytes) = 0;
  virtual void CopyTextureRegion(const void *, UINT, UINT, UINT, const void *,
                                 const void *) = 0;
  virtual void CopyResource(ID3D12Resource *, ID3D12Resource *) = 0;
  virtual void CopyTiles(ID3D12Resource *, const void *, const void *,
                         ID3D12Resource *, UINT64, int32_t) = 0;
  virtual void ResolveSubresource(ID3D12Resource *, UINT, ID3D12Resource *,
                                  UINT, int32_t) = 0;
  virtual void IASetPrimitiveTopology(int32_t) = 0;
  virtual void RSSetViewports(UINT, const void *) = 0;
  virtual void RSSetScissorRects(UINT, const void *) = 0;
  virtual void OMSetBlendFactor(const float[4]) = 0;
  virtual void OMSetStencilRef(UINT) = 0;
  virtual void SetPipelineState(ID3D12PipelineState *) = 0;
  virtual void ResourceBarrier(UINT, const D3D12_RESOURCE_BARRIER *) = 0;
  virtual void ExecuteBundle(ID3D12GraphicsCommandList *) = 0;
  virtual void SetDescriptorHeaps(UINT, void *const *) = 0;
  virtual void SetComputeRootSignature(ID3D12RootSignature *) = 0;
  virtual void SetGraphicsRootSignature(ID3D12RootSignature *) = 0;
  virtual void SetComputeRootDescriptorTable(UINT, uint64_t) = 0;
  virtual void SetGraphicsRootDescriptorTable(UINT, uint64_t) = 0;
  virtual void SetComputeRoot32BitConstant(UINT, UINT, UINT) = 0;
  virtual void SetGraphicsRoot32BitConstant(UINT, UINT, UINT) = 0;
  virtual void SetComputeRoot32BitConstants(UINT, UINT, const void *, UINT) = 0;
  virtual void SetGraphicsRoot32BitConstants(UINT, UINT, const void *,
                                             UINT) = 0;
  virtual void SetComputeRootConstantBufferView(UINT,
                                                D3D12_GPU_VIRTUAL_ADDRESS) = 0;
  virtual void SetGraphicsRootConstantBufferView(UINT,
                                                 D3D12_GPU_VIRTUAL_ADDRESS) = 0;
  virtual void SetComputeRootShaderResourceView(UINT,
                                                D3D12_GPU_VIRTUAL_ADDRESS) = 0;
  virtual void SetGraphicsRootShaderResourceView(UINT,
                                                 D3D12_GPU_VIRTUAL_ADDRESS) = 0;
  virtual void SetComputeRootUnorderedAccessView(UINT,
                                                 D3D12_GPU_VIRTUAL_ADDRESS) = 0;
};

struct ID3D12CommandQueue : ID3D12Pageable {
  virtual void UpdateTileMappings(ID3D12Resource *, UINT, const void *,
                                  const void *, void *, UINT, const int32_t *,
                                  const UINT *, const UINT *, int32_t) = 0;
  virtual void CopyTileMappings(ID3D12Resource *, const void *,
                                ID3D12Resource *, const void *, const void *,
                                int32_t) = 0;
  virtual void ExecuteCommandLists(UINT NumCommandLists,
                                   ID3D12CommandList *const *ppCommandLists) = 0;
  virtual void SetMarker(UINT, const void *, UINT) = 0;
  virtual void BeginEvent(UINT, const void *, UINT) = 0;
  virtual void EndEvent() = 0;
  virtual HRESULT Signal(ID3D12Fence *, UINT64) = 0;
  virtual HRESULT Wait(ID3D12Fence *, UINT64) = 0;
};

struct ID3D12Device : ID3D12Object {
  virtual UINT GetNodeCount() = 0;
  virtual HRESULT CreateCommandQueue(const D3D12_COMMAND_QUEUE_DESC *, REFIID,
                                     void **) = 0;
  virtual HRESULT CreateCommandAllocator(int32_t, REFIID, void **) = 0;
  virtual HRESULT CreateGraphicsPipelineState(const void *, REFIID,
                                              void **) = 0;
  virtual HRESULT CreateComputePipelineState(
      const D3D12_COMPUTE_PIPELINE_STATE_DESC *, REFIID, void **) = 0;
  virtual HRESULT CreateCommandList(UINT, int32_t, ID3D12CommandAllocator *,
                                    ID3D12PipelineState *, REFIID, void **) = 0;
  virtual HRESULT CheckFeatureSupport(int32_t, void *, UINT) = 0;
  virtual HRESULT CreateDescriptorHeap(const void *, REFIID, void **) = 0;
  virtual UINT GetDescriptorHandleIncrementSize(int32_t) = 0;
  virtual HRESULT CreateRootSignature(UINT, const void *, SIZE_T, REFIID,
                                      void **) = 0;
  virtual void CreateConstantBufferView(const void *, uint64_t) = 0;
  virtual void CreateShaderResourceView(ID3D12Resource *, const void *,
                                        uint64_t) = 0;
  virtual void CreateUnorderedAccessView(ID3D12Resource *, ID3D12Resource *,
                                         const void *, uint64_t) = 0;
  virtual void CreateRenderTargetView(ID3D12Resource *, const void *,
                                      uint64_t) = 0;
  virtual void CreateDepthStencilView(ID3D12Resource *, const void *,
                                      uint64_t) = 0;
  virtual void CreateSampler(const void *, uint64_t) = 0;
  virtual void CopyDescriptors(UINT, const uint64_t *, const UINT *, UINT,
                               const uint64_t *, const UINT *, int32_t) = 0;
  virtual void CopyDescriptorsSimple(UINT, uint64_t, uint64_t, int32_t) = 0;
  virtual void GetResourceAllocationInfo(UINT, UINT, const D3D12_RESOURCE_DESC *) = 0;
  virtual D3D12_HEAP_PROPERTIES GetCustomHeapProperties(UINT, int32_t) = 0;
  virtual HRESULT CreateCommittedResource(const D3D12_HEAP_PROPERTIES *,
                                          int32_t, const D3D12_RESOURCE_DESC *,
                                          int32_t, const void *, REFIID,
                                          void **) = 0;
  virtual HRESULT CreateHeap(const void *, REFIID, void **) = 0;
  virtual HRESULT CreatePlacedResource(void *, UINT64,
                                       const D3D12_RESOURCE_DESC *, int32_t,
                                       const void *, REFIID, void **) = 0;
  virtual HRESULT CreateReservedResource(const D3D12_RESOURCE_DESC *, int32_t,
                                         const void *, REFIID, void **) = 0;
  virtual HRESULT CreateSharedHandle(ID3D12DeviceChild *, const void *, UINT,
                                     const wchar_t *, void **) = 0;
  virtual HRESULT OpenSharedHandle(void *, REFIID, void **) = 0;
  virtual HRESULT OpenSharedHandleByName(const wchar_t *, UINT, void **) = 0;
  virtual HRESULT MakeResident(UINT, ID3D12Pageable *const *) = 0;
  virtual HRESULT Evict(UINT, ID3D12Pageable *const *) = 0;
  virtual HRESULT CreateFence(UINT64, int32_t, REFIID, void **) = 0;
};

using PFN_D3D12CreateDevice = HRESULT (*)(IUnknown *, uint32_t, REFIID,
                                          void **);
using PFN_D3D12SerializeRootSignature =
    HRESULT (*)(const D3D12_ROOT_SIGNATURE_DESC *, uint32_t, ID3D10Blob **,
                ID3D10Blob **);

} // namespace d3d12_abi

namespace {
using namespace d3d12_abi;

void checkHr(HRESULT Hr, const char *What) {
  if (Hr < 0) {
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

ID3D12Resource *createBuffer(ID3D12Device *Dev, int32_t HeapType,
                             uint64_t SizeBytes, int32_t Flags,
                             int32_t InitState) {
  D3D12_HEAP_PROPERTIES Heap{};
  Heap.Type = HeapType;
  Heap.CreationNodeMask = 1;
  Heap.VisibleNodeMask = 1;

  D3D12_RESOURCE_DESC Desc{};
  Desc.Dimension = 1; // BUFFER
  Desc.Width = SizeBytes;
  Desc.Height = 1;
  Desc.DepthOrArraySize = 1;
  Desc.MipLevels = 1;
  Desc.Format = 0; // UNKNOWN
  Desc.SampleDesc = {1, 0};
  Desc.Layout = 1; // ROW_MAJOR
  Desc.Flags = Flags;

  ID3D12Resource *Res = nullptr;
  checkHr(Dev->CreateCommittedResource(&Heap, 0, &Desc, InitState, nullptr,
                                       IID_ID3D12Resource,
                                       reinterpret_cast<void **>(&Res)),
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

  const char *LibPath = std::getenv("CRUCIBLE_NACELLE_LIB");
  if (!LibPath || !*LibPath)
    LibPath = CRUCIBLE_NACELLE_LIB_PATH;

  void *Lib = dlopen(LibPath, RTLD_NOW | RTLD_LOCAL);
  if (!Lib)
    fail(std::string("failed to dlopen Nacelle D3D12 runtime '") + LibPath +
         "': " + (dlerror() ? dlerror() : "unknown error"));

  auto CreateDeviceFn = reinterpret_cast<PFN_D3D12CreateDevice>(
      dlsym(Lib, "D3D12CreateDevice"));
  auto SerializeRSFn = reinterpret_cast<PFN_D3D12SerializeRootSignature>(
      dlsym(Lib, "D3D12SerializeRootSignature"));
  if (!CreateDeviceFn || !SerializeRSFn)
    fail("Nacelle D3D12 runtime missing D3D12CreateDevice or "
         "D3D12SerializeRootSignature");

  std::vector<uint8_t> Bytecode = readBinary(Opts.DxilPath);

  ID3D12Device *Dev = nullptr;
  checkHr(CreateDeviceFn(nullptr, 0xb000 /* D3D_FEATURE_LEVEL_11_0 */,
                         IID_ID3D12Device, reinterpret_cast<void **>(&Dev)),
          "D3D12CreateDevice");

  D3D12_COMMAND_QUEUE_DESC QDesc{};
  QDesc.Type = 0; // DIRECT
  ID3D12CommandQueue *Queue = nullptr;
  checkHr(Dev->CreateCommandQueue(&QDesc, IID_ID3D12CommandQueue,
                                  reinterpret_cast<void **>(&Queue)),
          "CreateCommandQueue");

  ID3D12CommandAllocator *Alloc = nullptr;
  checkHr(Dev->CreateCommandAllocator(0, IID_ID3D12CommandAllocator,
                                      reinterpret_cast<void **>(&Alloc)),
          "CreateCommandAllocator");

  // Build root signature with one root descriptor per --buffer (UAV u#) and
  // --cbuffer (CBV b#).
  std::vector<D3D12_ROOT_PARAMETER> RootParams;
  RootParams.reserve(Opts.Buffers.size());
  for (const BufferSpec &Spec : Opts.Buffers) {
    D3D12_ROOT_PARAMETER P{};
    P.ParameterType = Spec.IsUniform ? 2 /* CBV */ : 4 /* UAV */;
    P.Descriptor.ShaderRegister = Spec.Index;
    P.Descriptor.RegisterSpace = 0;
    P.ShaderVisibility = 0; // ALL
    RootParams.push_back(P);
  }

  D3D12_ROOT_SIGNATURE_DESC RSDesc{};
  RSDesc.NumParameters = static_cast<UINT>(RootParams.size());
  RSDesc.pParameters = RootParams.empty() ? nullptr : RootParams.data();
  RSDesc.Flags = 0;

  ID3D10Blob *RSBlob = nullptr;
  ID3D10Blob *ErrBlob = nullptr;
  HRESULT RSHr = SerializeRSFn(&RSDesc, 1 /* VERSION_1_0 */, &RSBlob, &ErrBlob);
  if (RSHr < 0) {
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
                                   IID_ID3D12RootSignature,
                                   reinterpret_cast<void **>(&RootSig)),
          "CreateRootSignature");
  RSBlob->Release();

  D3D12_COMPUTE_PIPELINE_STATE_DESC PSODesc{};
  PSODesc.pRootSignature = RootSig;
  PSODesc.CS.pShaderBytecode = Bytecode.data();
  PSODesc.CS.BytecodeLength = Bytecode.size();

  ID3D12PipelineState *PSO = nullptr;
  checkHr(Dev->CreateComputePipelineState(&PSODesc, IID_ID3D12PipelineState,
                                          reinterpret_cast<void **>(&PSO)),
          "CreateComputePipelineState");

  ID3D12GraphicsCommandList *Cmd = nullptr;
  checkHr(Dev->CreateCommandList(0, 0, Alloc, PSO, IID_ID3D12GraphicsCommandList,
                                 reinterpret_cast<void **>(&Cmd)),
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
      AB.UploadBuf = createBuffer(Dev, 2 /* UPLOAD */, AB.AllocBytes, 0,
                                  0x00000A83 /* GENERIC_READ */);
      void *Mapped = nullptr;
      checkHr(AB.UploadBuf->Map(0, nullptr, &Mapped), "Map CBV upload");
      std::memset(Mapped, 0, AB.AllocBytes);
      fillBuffer(Spec, Mapped);
      AB.UploadBuf->Unmap(0, nullptr);
    } else {
      AB.DefaultBuf = createBuffer(Dev, 1 /* DEFAULT */, AB.AllocBytes,
                                   4 /* ALLOW_UNORDERED_ACCESS */,
                                   0x00000400 /* COPY_DEST */);
      AB.UploadBuf = createBuffer(Dev, 2 /* UPLOAD */, AB.AllocBytes, 0,
                                  0x00000A83 /* GENERIC_READ */);
      AB.ReadbackBuf = createBuffer(Dev, 3 /* READBACK */, AB.AllocBytes, 0,
                                    0x00000400 /* COPY_DEST */);
      void *Mapped = nullptr;
      checkHr(AB.UploadBuf->Map(0, nullptr, &Mapped), "Map UAV upload");
      std::memset(Mapped, 0, AB.AllocBytes);
      fillBuffer(Spec, Mapped);
      AB.UploadBuf->Unmap(0, nullptr);
      Cmd->CopyBufferRegion(AB.DefaultBuf, 0, AB.UploadBuf, 0, AB.AllocBytes);

      D3D12_RESOURCE_BARRIER Barrier{};
      Barrier.Type = 0; // TRANSITION
      Barrier.Transition.pResource = AB.DefaultBuf;
      Barrier.Transition.Subresource = 0xFFFFFFFFu;
      Barrier.Transition.StateBefore = 0x00000400; // COPY_DEST
      Barrier.Transition.StateAfter = 0x00000008;  // UNORDERED_ACCESS
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
    Barrier.Type = 0; // TRANSITION
    Barrier.Transition.pResource = AB.DefaultBuf;
    Barrier.Transition.Subresource = 0xFFFFFFFFu;
    Barrier.Transition.StateBefore = 0x00000008; // UNORDERED_ACCESS
    Barrier.Transition.StateAfter = 0x00000800;  // COPY_SOURCE
    Cmd->ResourceBarrier(1, &Barrier);
    Cmd->CopyBufferRegion(AB.ReadbackBuf, 0, AB.DefaultBuf, 0, AB.AllocBytes);
  }

  checkHr(Cmd->Close(), "CommandList::Close");

  ID3D12CommandList *Lists[] = {Cmd};
  Queue->ExecuteCommandLists(1, Lists);

  ID3D12Fence *Fence = nullptr;
  checkHr(Dev->CreateFence(0, 0, IID_ID3D12Fence,
                           reinterpret_cast<void **>(&Fence)),
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

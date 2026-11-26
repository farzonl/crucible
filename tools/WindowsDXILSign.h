//===- WindowsDXILSign.h - Windows DXIL signing & experimental mode -*- C++ -*-===//
//
// Part of Crucible, an open-source multi-frontend, multi-backend shader compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Helpers for running LLVM-emitted (unsigned) DXIL containers on Windows D3D12
// (hardware adapters and WARP):
//   1. Best-effort D3D12EnableExperimentalFeatures(D3D12ExperimentalShaderModels)
//      before D3D12CreateDevice (allows unsigned DXIL when Developer Mode is on).
//   2. In-place retail signing/validation via dxil.dll (IDxcValidator::Validate
//      with DxcValidatorFlags_InPlaceEdit) so WARP and retail drivers accept the
//      DXContainer even when Developer Mode is off.
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_TOOLS_WINDOWS_DXIL_SIGN_H
#define CRUCIBLE_TOOLS_WINDOWS_DXIL_SIGN_H

#ifdef _WIN32

#include <windows.h>
#include <unknwn.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace crucible_win_dxil {

// {76f5573e-f13a-40f5-b297-81ce9e18933f}
static constexpr GUID kD3D12ExperimentalShaderModels = {
    0x76f5573e,
    0xf13a,
    0x40f5,
    {0xb2, 0x97, 0x81, 0xce, 0x9e, 0x18, 0x93, 0x3f}};

// {8ca3e215-f728-4cf3-8cdd-88af917587a1}
static constexpr GUID kCLSID_DxcValidator = {
    0x8ca3e215,
    0xf728,
    0x4cf3,
    {0x8c, 0xdd, 0x88, 0xaf, 0x91, 0x75, 0x87, 0xa1}};

// {a6e82bd2-1fd7-4826-9811-2857e797f49a}
static constexpr GUID kIID_IDxcValidator = {
    0xa6e82bd2,
    0x1fd7,
    0x4826,
    {0x98, 0x11, 0x28, 0x57, 0xe7, 0x97, 0xf4, 0x9a}};

// {8ba5fb08-5195-40e2-ac58-0d989c3a0102}
static constexpr GUID kIID_IDxcBlob = {
    0x8ba5fb08,
    0x5195,
    0x40e2,
    {0xac, 0x58, 0x0d, 0x98, 0x9c, 0x3a, 0x01, 0x02}};

struct MinimalDxcBlob : public IUnknown {
  virtual LPVOID STDMETHODCALLTYPE GetBufferPointer() = 0;
  virtual SIZE_T STDMETHODCALLTYPE GetBufferSize() = 0;
};

struct MinimalDxcOperationResult : public IUnknown {
  virtual HRESULT STDMETHODCALLTYPE GetStatus(HRESULT *pStatus) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetResult(MinimalDxcBlob **ppResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE
  GetErrorBuffer(MinimalDxcBlob **ppErrors) = 0;
};

struct MinimalDxcValidator : public IUnknown {
  virtual HRESULT STDMETHODCALLTYPE
  Validate(MinimalDxcBlob *pShader, UINT32 Flags,
           MinimalDxcOperationResult **ppResult) = 0;
};

class InPlaceMemoryBlob final : public MinimalDxcBlob {
  void *Ptr;
  size_t Size;
  std::atomic<ULONG> RefCount{1};

public:
  InPlaceMemoryBlob(void *Ptr, size_t Size) : Ptr(Ptr), Size(Size) {}

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID Riid,
                                           void **PpvObject) override {
    if (!PpvObject)
      return E_POINTER;
    if (IsEqualIID(Riid, IID_IUnknown) || IsEqualIID(Riid, kIID_IDxcBlob)) {
      *PpvObject = static_cast<MinimalDxcBlob *>(this);
      AddRef();
      return S_OK;
    }
    *PpvObject = nullptr;
    return E_NOINTERFACE;
  }
  ULONG STDMETHODCALLTYPE AddRef() override { return ++RefCount; }
  ULONG STDMETHODCALLTYPE Release() override { return --RefCount; }

  LPVOID STDMETHODCALLTYPE GetBufferPointer() override { return Ptr; }
  SIZE_T STDMETHODCALLTYPE GetBufferSize() override { return Size; }
};

inline void enableExperimentalShaderModels(HMODULE D3D12Mod) {
  if (!D3D12Mod)
    return;
  using PFN_D3D12EnableExperimentalFeatures =
      HRESULT(WINAPI *)(UINT, const IID *, void *, UINT *);
  auto Fn = reinterpret_cast<PFN_D3D12EnableExperimentalFeatures>(
      GetProcAddress(D3D12Mod, "D3D12EnableExperimentalFeatures"));
  if (!Fn)
    return;
  GUID Feat = kD3D12ExperimentalShaderModels;
  (void)Fn(1, &Feat, nullptr, nullptr);
}

/// Validates and signs a DXContainer buffer in-place using dxil.dll if
/// available. Returns true if signed (or if dxil.dll is not found, allowing
/// Developer Mode fallback). Returns false and populates ErrOut if dxil.dll
/// reports a validation error.
inline bool signDXILInPlace(void *Data, size_t Size, std::string &ErrOut) {
  using PFN_DxcCreateInstance =
      HRESULT(WINAPI *)(REFCLSID, REFIID, LPVOID *);
  static PFN_DxcCreateInstance CreateInstanceFn = nullptr;
  static bool Probed = false;
  if (!Probed) {
    Probed = true;
    const char *DllPath = std::getenv("CRUCIBLE_DXIL_DLL");
    HMODULE Mod = nullptr;
    if (DllPath && *DllPath)
      Mod = LoadLibraryA(DllPath);
    if (!Mod)
      Mod = LoadLibraryA("dxil.dll");
    if (Mod) {
      CreateInstanceFn = reinterpret_cast<PFN_DxcCreateInstance>(
          GetProcAddress(Mod, "DxcCreateInstance"));
    }
  }
  if (!CreateInstanceFn)
    return true;

  MinimalDxcValidator *Validator = nullptr;
  HRESULT Hr = CreateInstanceFn(kCLSID_DxcValidator, kIID_IDxcValidator,
                                reinterpret_cast<void **>(&Validator));
  if (FAILED(Hr) || !Validator)
    return true;

  InPlaceMemoryBlob Blob(Data, Size);
  MinimalDxcOperationResult *Result = nullptr;
  constexpr UINT32 kDxcValidatorFlags_InPlaceEdit = 1;
  Hr = Validator->Validate(&Blob, kDxcValidatorFlags_InPlaceEdit, &Result);
  Validator->Release();
  if (FAILED(Hr) || !Result) {
    char Buf[64];
    snprintf(Buf, sizeof(Buf), "dxil.dll Validate call failed (HRESULT=0x%08x)",
             static_cast<uint32_t>(Hr));
    ErrOut = Buf;
    return false;
  }

  HRESULT Status = S_OK;
  Result->GetStatus(&Status);
  if (FAILED(Status)) {
    MinimalDxcBlob *ErrBlob = nullptr;
    if (SUCCEEDED(Result->GetErrorBuffer(&ErrBlob)) && ErrBlob) {
      if (ErrBlob->GetBufferPointer() && ErrBlob->GetBufferSize() > 0) {
        ErrOut.assign(static_cast<const char *>(ErrBlob->GetBufferPointer()),
                      ErrBlob->GetBufferSize());
      }
      ErrBlob->Release();
    }
    if (ErrOut.empty()) {
      char Buf[64];
      snprintf(Buf, sizeof(Buf), "dxil.dll validation failed (HRESULT=0x%08x)",
               static_cast<uint32_t>(Status));
      ErrOut = Buf;
    }
    Result->Release();
    return false;
  }

  Result->Release();
  return true;
}

} // namespace crucible_win_dxil

#endif // _WIN32

#endif // CRUCIBLE_TOOLS_WINDOWS_DXIL_SIGN_H

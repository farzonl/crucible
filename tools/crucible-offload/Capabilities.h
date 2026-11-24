//===- Capabilities.h - Unified Crucible offload capabilities ----*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Projects probed execution-backend capabilities (Metal or active Vulkan ICD)
// through runtime dispatchability masks and Crucible compiler emulation
// overlays into offload-test-suite Frontend API capabilities:
//
//   Effective Caps = (Backend Caps & Runtime Dispatchable Caps) | Emulation
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_OFFLOAD_CAPABILITIES_H
#define CRUCIBLE_OFFLOAD_CAPABILITIES_H

#include "Common.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace crucible_offload {

struct MetalBackendCaps {
  std::string DeviceName;
  std::string OSVersion;
  std::string GPUGeneration;
  bool SupportsMetal3 = false;
  bool SupportsRaytracing = false;
  bool HasUnifiedMemory = false;
  uint64_t RegistryID = 0;
};

struct VulkanBackendCaps {
  std::string DeviceName;
  std::string DriverName; // VkPhysicalDeviceDriverProperties::driverName
  std::string DriverInfo; // VkPhysicalDeviceDriverProperties::driverInfo
  uint32_t VendorID = 0;
  uint32_t DeviceID = 0;
  uint32_t ApiVersion = 0;
  uint32_t DriverVersion = 0;
  uint32_t SubgroupSize = 0;
  bool SubgroupCompute = false;
  bool ShaderFloat64 = false;
  bool ShaderInt64 = false;
  bool ShaderInt16 = false;
  bool ShaderFloat16 = false;
  bool ShaderBufferInt64Atomics = false;
  bool ShaderSharedInt64Atomics = false;
  bool ShaderBufferFloat32Atomics = false;
  bool ShaderSharedFloat32Atomics = false;
  bool ShaderResourceMinLod = false;
  bool MultiViewport = false;
  bool ShaderOutputViewportIndex = false;
  bool FragmentShaderBarycentric = false;
  bool ShaderStencilExport = false;
  bool MeshShader = false;
  bool RayTracing = false;
  bool DescriptorIndexing = false;
};

struct VulkanRuntimeCaps {
  bool SupportsMeshShader = false;
  bool SupportsRaytracing = false;
};

struct D3D12BackendCaps {
  std::string AdapterName;
  std::string DriverName;
  uint32_t VendorID = 0;
  uint32_t DeviceID = 0;
  std::string HighestShaderModel = "6.0";
  bool DoublePrecisionFloatShaderOps = false;
  bool Int64ShaderOps = false;
  bool Native16BitShaderOpsSupported = false;
  bool WaveOps = false;
  uint32_t WaveLaneCountMin = 0;
  uint32_t WaveLaneCountMax = 0;
  bool AtomicInt64OnTypedResourceSupported = false;
  bool AtomicInt64OnGroupSharedSupported = false;
  uint32_t ResourceBindingTier = 1;
  uint32_t TiledResourcesTier = 0;
  bool MeshShaderSupported = false;
  bool RaytracingSupported = false;
};

struct CrucibleEmulationCaps {
  bool EmulateF64 = false;
  bool EmulateInt64Atomics = false;
  std::string HighestShaderModel = "6.6";
};

offloadtest::Capabilities
buildDirectXCapsFromMetal(const MetalBackendCaps &Metal,
                          const CrucibleEmulationCaps &Emul);

offloadtest::Capabilities
buildVulkanCapsFromMetal(const MetalBackendCaps &Metal,
                         const CrucibleEmulationCaps &Emul);

void applyVulkanEmulationOverrides(offloadtest::Capabilities &Caps,
                                   const CrucibleEmulationCaps &Emul);

offloadtest::Capabilities
buildDirectXCapsFromVulkan(const VulkanBackendCaps &VK,
                           const VulkanRuntimeCaps &Runtime,
                           const CrucibleEmulationCaps &Emul);

offloadtest::Capabilities
buildVulkanCapsFromDirectX(const D3D12BackendCaps &DX,
                           const CrucibleEmulationCaps &Emul);

std::string formatVulkanDeviceDescription(llvm::StringRef DeviceName,
                                          llvm::StringRef DriverName);

} // namespace crucible_offload

#endif // CRUCIBLE_OFFLOAD_CAPABILITIES_H

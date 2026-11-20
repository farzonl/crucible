//===- Capabilities.cpp - Unified Crucible offload capabilities --*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Capabilities.h"

using namespace llvm;
using namespace offloadtest;

namespace crucible_offload {

Capabilities buildDirectXCapsFromMetal(const MetalBackendCaps &Metal,
                                       const CrucibleEmulationCaps &Emul) {
  Capabilities Caps;
  Caps.try_emplace("WaveOps", makeCapability("WaveOps", true));
  Caps.try_emplace("WaveLaneCountMin",
                   makeCapability("WaveLaneCountMin", uint32_t(32)));
  Caps.try_emplace("WaveLaneCountMax",
                   makeCapability("WaveLaneCountMax", uint32_t(32)));
  Caps.try_emplace("TotalLaneCount",
                   makeCapability("TotalLaneCount", uint32_t(32)));
  Caps.try_emplace(
      "HighestShaderModel",
      makeStringCapability("HighestShaderModel", Emul.HighestShaderModel));
  Caps.try_emplace("Native16BitShaderOpsSupported",
                   makeCapability("Native16BitShaderOpsSupported", true));
  Caps.try_emplace("Int64ShaderOps", makeCapability("Int64ShaderOps", true));
  Caps.try_emplace(
      "DoublePrecisionFloatShaderOps",
      makeCapability("DoublePrecisionFloatShaderOps", Emul.EmulateF64));
  Caps.try_emplace(
      "AtomicInt64OnGroupSharedSupported",
      makeCapability("AtomicInt64OnGroupSharedSupported",
                     Emul.EmulateInt64Atomics));
  Caps.try_emplace(
      "AtomicInt64OnTypedResourceSupported",
      makeCapability("AtomicInt64OnTypedResourceSupported",
                     Emul.EmulateInt64Atomics));
  Caps.try_emplace("ResourceBindingTier",
                   makeCapability("ResourceBindingTier", uint32_t(3)));
  Caps.try_emplace("TiledResourcesTier",
                   makeCapability("TiledResourcesTier", uint32_t(2)));
  Caps.try_emplace(
      "MeshShaderTier",
      makeStringCapability("MeshShaderTier",
                           Metal.SupportsMetal3 ? "Tier1" : "NotSupported"));
  Caps.try_emplace("BarycentricsSupported",
                   makeCapability("BarycentricsSupported", false));
  Caps.try_emplace("PSSpecifiedStencilRefSupported",
                   makeCapability("PSSpecifiedStencilRefSupported", true));
  Caps.try_emplace(
      "VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation",
      makeCapability(
          "VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation",
          true));
  Caps.try_emplace(
      "RaytracingTier",
      makeStringCapability("RaytracingTier",
                           Metal.SupportsRaytracing ? "1.1" : "NotSupported"));
  return Caps;
}

Capabilities buildVulkanCapsFromMetal(const MetalBackendCaps &Metal,
                                      const CrucibleEmulationCaps &Emul) {
  (void)Metal;
  Capabilities Caps;
  Caps.try_emplace("shaderFloat16", makeCapability("shaderFloat16", true));
  Caps.try_emplace("shaderInt16", makeCapability("shaderInt16", true));
  Caps.try_emplace("shaderInt64", makeCapability("shaderInt64", true));
  Caps.try_emplace("shaderFloat64",
                   makeCapability("shaderFloat64", Emul.EmulateF64));
  Caps.try_emplace("shaderResourceMinLod",
                   makeCapability("shaderResourceMinLod", true));
  Caps.try_emplace("runtimeDescriptorArray",
                   makeCapability("runtimeDescriptorArray", true));
  Caps.try_emplace("descriptorBindingPartiallyBound",
                   makeCapability("descriptorBindingPartiallyBound", true));
  Caps.try_emplace("multiViewport", makeCapability("multiViewport", true));
  Caps.try_emplace("shaderOutputViewportIndex",
                   makeCapability("shaderOutputViewportIndex", true));
  Caps.try_emplace("shaderSubgroupExtendedTypes",
                   makeCapability("shaderSubgroupExtendedTypes", true));
  Caps.try_emplace("shaderBufferFloat32Atomics",
                   makeCapability("shaderBufferFloat32Atomics", true));
  Caps.try_emplace("shaderSharedFloat32Atomics",
                   makeCapability("shaderSharedFloat32Atomics", true));
  Caps.try_emplace("shaderBufferInt64Atomics",
                   makeCapability("shaderBufferInt64Atomics",
                                  Emul.EmulateInt64Atomics));
  Caps.try_emplace("shaderSharedInt64Atomics",
                   makeCapability("shaderSharedInt64Atomics",
                                  Emul.EmulateInt64Atomics));
  return Caps;
}

void applyVulkanEmulationOverrides(Capabilities &Caps,
                                   const CrucibleEmulationCaps &Emul) {
  if (Emul.EmulateF64)
    Caps.insert_or_assign("shaderFloat64",
                          makeCapability("shaderFloat64", true));
  if (Emul.EmulateInt64Atomics) {
    Caps.insert_or_assign("shaderBufferInt64Atomics",
                          makeCapability("shaderBufferInt64Atomics", true));
    Caps.insert_or_assign("shaderSharedInt64Atomics",
                          makeCapability("shaderSharedInt64Atomics", true));
  }
}

Capabilities buildDirectXCapsFromVulkan(const VulkanBackendCaps &VK,
                                        const VulkanRuntimeCaps &Runtime,
                                        const CrucibleEmulationCaps &Emul) {
  Capabilities Caps;
  const bool HasWave = VK.SubgroupCompute && VK.SubgroupSize > 0;
  const bool HasNative16 = VK.ShaderInt16 && VK.ShaderFloat16;
  const bool HasF64 = VK.ShaderFloat64 || Emul.EmulateF64;
  const bool HasBufInt64Atomics =
      VK.ShaderBufferInt64Atomics || Emul.EmulateInt64Atomics;
  const bool HasSharedInt64Atomics =
      VK.ShaderSharedInt64Atomics || Emul.EmulateInt64Atomics;
  const bool HasMesh = VK.MeshShader && Runtime.SupportsMeshShader;
  const bool HasRT = VK.RayTracing && Runtime.SupportsRaytracing;

  std::string HighestSM =
      (HasBufInt64Atomics || VK.DescriptorIndexing) ? "6.6" : "6.5";
  if (!Emul.HighestShaderModel.empty() && Emul.HighestShaderModel < HighestSM)
    HighestSM = Emul.HighestShaderModel;

  Caps.try_emplace("HighestShaderModel",
                   makeStringCapability("HighestShaderModel", HighestSM));
  Caps.try_emplace("DoublePrecisionFloatShaderOps",
                   makeCapability("DoublePrecisionFloatShaderOps", HasF64));
  Caps.try_emplace("Int64ShaderOps",
                   makeCapability("Int64ShaderOps", VK.ShaderInt64));
  Caps.try_emplace("Native16BitShaderOpsSupported",
                   makeCapability("Native16BitShaderOpsSupported", HasNative16));
  Caps.try_emplace("AtomicInt64OnTypedResourceSupported",
                   makeCapability("AtomicInt64OnTypedResourceSupported",
                                  HasBufInt64Atomics));
  Caps.try_emplace("AtomicInt64OnGroupSharedSupported",
                   makeCapability("AtomicInt64OnGroupSharedSupported",
                                  HasSharedInt64Atomics));
  Caps.try_emplace("WaveOps", makeCapability("WaveOps", HasWave));
  Caps.try_emplace("WaveLaneCountMin",
                   makeCapability<uint32_t>("WaveLaneCountMin",
                                            VK.SubgroupSize));
  Caps.try_emplace("WaveLaneCountMax",
                   makeCapability<uint32_t>("WaveLaneCountMax",
                                            VK.SubgroupSize));
  Caps.try_emplace("TotalLaneCount",
                   makeCapability<uint32_t>("TotalLaneCount", VK.SubgroupSize));
  Caps.try_emplace("ResourceBindingTier",
                   makeCapability<uint32_t>("ResourceBindingTier",
                                            VK.DescriptorIndexing ? 3u : 2u));
  Caps.try_emplace("TiledResourcesTier",
                   makeCapability<uint32_t>("TiledResourcesTier",
                                            VK.ShaderResourceMinLod ? 2u : 0u));
  Caps.try_emplace("PSSpecifiedStencilRefSupported",
                   makeCapability("PSSpecifiedStencilRefSupported",
                                  VK.ShaderStencilExport));
  Caps.try_emplace(
      "VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation",
      makeCapability(
          "VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation",
          VK.MultiViewport && VK.ShaderOutputViewportIndex));
  Caps.try_emplace("BarycentricsSupported",
                   makeCapability("BarycentricsSupported",
                                  VK.FragmentShaderBarycentric));
  Caps.try_emplace("MeshShaderTier",
                   makeStringCapability("MeshShaderTier",
                                        HasMesh ? "Tier1" : "NotSupported"));
  Caps.try_emplace("RaytracingTier",
                   makeStringCapability("RaytracingTier",
                                        HasRT ? "1.1" : "NotSupported"));
  return Caps;
}

std::string formatVulkanDeviceDescription(StringRef DeviceName,
                                          StringRef DriverName) {
  if (DriverName.empty() || DeviceName.contains_insensitive(DriverName))
    return DeviceName.str();
  return (DeviceName + " (" + DriverName + ")").str();
}

} // namespace crucible_offload

//===- DeviceInit.cpp - Runtime dispatcher for crucible-offload --*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Selects between MetalDevice (dxil-to-metal / spirv-to-metal) and
// VulkanDevice (dxil-to-spirv) at runtime via CRUCIBLE_OFFLOAD_RUNTIME.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include "llvm/ADT/StringRef.h"

#include <cstdlib>

using namespace llvm;
using namespace offloadtest;

Expected<SmallVector<std::unique_ptr<offloadtest::Device>>>
offloadtest::initializeDevices(const DeviceConfig Config) {
  StringRef Runtime;
  if (const char *Env = std::getenv("CRUCIBLE_OFFLOAD_RUNTIME"))
    Runtime = Env;

  if (Runtime.empty()) {
#ifdef CRUCIBLE_HAS_METAL_OFFLOAD
    Runtime = "metal";
#elif defined(CRUCIBLE_HAS_VULKAN_OFFLOAD)
    Runtime = "vulkan";
#else
    return createStringError(
        std::errc::not_supported,
        "no crucible-offload execution runtime was built");
#endif
  }

  SmallVector<std::unique_ptr<offloadtest::Device>> Devices;
  SmallVector<std::unique_ptr<offloadtest::Device>> NativeVulkanDevices;
#ifdef CRUCIBLE_HAS_VULKAN_OFFLOAD
  if (Error E = initializeVulkanDevices(Config, NativeVulkanDevices))
    logAllUnhandledErrors(std::move(E), errs(),
                          "crucible-offload: native Vulkan discovery: ");
#endif
  if (Runtime.equals_insensitive("metal")) {
#ifdef CRUCIBLE_HAS_METAL_OFFLOAD
    offloadtest::Device *VulkanInfo =
        NativeVulkanDevices.empty() ? nullptr
                                    : NativeVulkanDevices.front().get();
    auto MetalDevices =
        crucible_offload::createMetalDevices(Config, VulkanInfo);
    if (!MetalDevices)
      return MetalDevices.takeError();
    for (auto &D : *MetalDevices)
      Devices.push_back(std::move(D));
#else
    return createStringError(
        std::errc::not_supported,
        "CRUCIBLE_OFFLOAD_RUNTIME=metal requested, but Metal runtime is not "
        "built on this platform");
#endif
  } else if (Runtime.equals_insensitive("vulkan")) {
#ifdef CRUCIBLE_HAS_VULKAN_OFFLOAD
    auto VulkanDevices = crucible_offload::createVulkanDevices(Config);
    if (!VulkanDevices)
      return VulkanDevices.takeError();
    for (auto &D : *VulkanDevices)
      Devices.push_back(std::move(D));
#else
    return createStringError(
        std::errc::not_supported,
        "CRUCIBLE_OFFLOAD_RUNTIME=vulkan requested, but Vulkan runtime is not "
        "built");
#endif
  }

  else {
    return createStringError(
        std::errc::invalid_argument,
        "unknown CRUCIBLE_OFFLOAD_RUNTIME='%s' (expected 'metal' or 'vulkan')",
        Runtime.str().c_str());
  }

  for (auto &D : NativeVulkanDevices)
    Devices.push_back(std::move(D));

  if (const char *Driver = std::getenv("CRUCIBLE_OFFLOAD_DRIVER")) {
    StringRef Filter(Driver);
    llvm::erase_if(Devices, [Filter](const auto &D) {
      return !D->getDriverName().equals_insensitive(Filter);
    });
    if (Devices.empty())
      return createStringError(
          std::errc::no_such_device,
          "no Crucible offload device found for driver '%s'", Driver);
  }

  return Devices;
}

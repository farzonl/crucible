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

offloadtest::Device::~Device() = default;

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

  if (Runtime.equals_insensitive("metal")) {
#ifdef CRUCIBLE_HAS_METAL_OFFLOAD
    return crucible_offload::createMetalDevices(Config);
#else
    return createStringError(
        std::errc::not_supported,
        "CRUCIBLE_OFFLOAD_RUNTIME=metal requested, but Metal runtime is not "
        "built on this platform");
#endif
  }

  if (Runtime.equals_insensitive("vulkan")) {
#ifdef CRUCIBLE_HAS_VULKAN_OFFLOAD
    return crucible_offload::createVulkanDevices(Config);
#else
    return createStringError(
        std::errc::not_supported,
        "CRUCIBLE_OFFLOAD_RUNTIME=vulkan requested, but Vulkan runtime is not "
        "built");
#endif
  }

  return createStringError(
      std::errc::invalid_argument,
      "unknown CRUCIBLE_OFFLOAD_RUNTIME='%s' (expected 'metal' or 'vulkan')",
      Runtime.str().c_str());
}

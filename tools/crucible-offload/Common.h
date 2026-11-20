//===- Common.h - Shared crucible-offload helpers ----------------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_OFFLOAD_COMMON_H
#define CRUCIBLE_OFFLOAD_COMMON_H

#include "API/Capabilities.h"
#include "API/Device.h"
#include "Support/Pipeline.h"
#include "crucible/Reflection.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <string>

namespace crucible_offload {

struct CapabilityValueString
    : offloadtest::CapabilityValue<CapabilityValueString, std::string> {
  CapabilityValueString(std::string V)
      : offloadtest::CapabilityValue<CapabilityValueString, std::string>(
            std::move(V)) {}
  static char ID;
  static std::string valueToString(const std::string &V) { return V; }
};

inline offloadtest::Capability makeStringCapability(std::string Name,
                                                    std::string Val) {
  return offloadtest::Capability(
      std::move(Name),
      std::make_unique<CapabilityValueString>(std::move(Val)));
}

std::optional<crucible::ResourceClass>
classifyResource(offloadtest::ResourceKind Kind);

llvm::Error validateRootParameters(const offloadtest::Pipeline &P);

#ifdef CRUCIBLE_HAS_METAL_OFFLOAD
llvm::Expected<llvm::SmallVector<std::unique_ptr<offloadtest::Device>>>
createMetalDevices(const offloadtest::DeviceConfig &Config,
                   offloadtest::Device *VulkanInfo);
#endif

#ifdef CRUCIBLE_HAS_VULKAN_OFFLOAD
llvm::Expected<llvm::SmallVector<std::unique_ptr<offloadtest::Device>>>
createVulkanDevices(const offloadtest::DeviceConfig &Config);
#endif

} // namespace crucible_offload

#endif // CRUCIBLE_OFFLOAD_COMMON_H

//===- Common.cpp - Shared crucible-offload helpers --------------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Common.h"

using namespace llvm;
using namespace offloadtest;

namespace crucible_offload {

char CapabilityValueString::ID = 0;

std::optional<crucible::ResourceClass> classifyResource(ResourceKind Kind) {
  switch (Kind) {
  case ResourceKind::Buffer:
  case ResourceKind::StructuredBuffer:
  case ResourceKind::ByteAddressBuffer:
  case ResourceKind::Texture1D:
  case ResourceKind::Texture1DArray:
  case ResourceKind::Texture2D:
  case ResourceKind::Texture2DArray:
  case ResourceKind::Texture3D:
  case ResourceKind::TextureCube:
  case ResourceKind::TextureCubeArray:
  case ResourceKind::AccelerationStructure:
    return crucible::ResourceClass::SRV;
  case ResourceKind::RWBuffer:
  case ResourceKind::RWStructuredBuffer:
  case ResourceKind::RWByteAddressBuffer:
  case ResourceKind::RWTexture1D:
  case ResourceKind::RWTexture1DArray:
  case ResourceKind::RWTexture2D:
  case ResourceKind::RWTexture2DArray:
  case ResourceKind::RWTexture3D:
    return crucible::ResourceClass::UAV;
  case ResourceKind::ConstantBuffer:
    return crucible::ResourceClass::CBV;
  case ResourceKind::Sampler:
    return crucible::ResourceClass::Sampler;
  default:
    return std::nullopt;
  }
}

Error validateRootParameters(const Pipeline &P) {
  for (const auto &Param : P.Settings.DX.RootParams) {
    if (Param.Kind == dx::RootParamKind::Constant) {
      const auto &RC = std::get<dx::RootConstant>(Param.Data);
      if (RC.BufferPtr && RC.BufferPtr->ArraySize != 1)
        return createStringError(
            std::errc::invalid_argument,
            "Root constant cannot refer to resource arrays.");
    } else if (Param.Kind == dx::RootParamKind::RootDescriptor) {
      const auto &RR = std::get<dx::RootResource>(Param.Data);
      if (RR.getArraySize() != 1)
        return createStringError(
            std::errc::invalid_argument,
            "Root descriptor cannot refer to resource arrays.");
    }
  }
  return Error::success();
}

} // namespace crucible_offload

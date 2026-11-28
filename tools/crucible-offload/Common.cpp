//===- Common.cpp - Shared crucible-offload helpers --------------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "llvm/ADT/StringExtras.h"
#include <cstring>

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
  case ResourceKind::SampledTexture2D:
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

Error parseSpecConstants(const Shader &S, crucible::Options &Opts) {
  for (const SpecializationConstant &SC : S.SpecializationConstants) {
    if (Opts.SpecConstants.count(SC.ConstantID))
      return createStringError(
          inconvertibleErrorCode(),
          "Test configuration contains multiple entries for specialization "
          "constant ID %u.",
          SC.ConstantID);
    StringRef ValStr = StringRef(SC.Value).trim();
    uint64_t Bits = 0;
    auto failInvalid = [&](const char *TypeName) {
      return createStringError(
          inconvertibleErrorCode(),
          "Invalid %s value for specialization constant '%s'", TypeName,
          SC.Value.c_str());
    };
    switch (SC.Type) {
    case DataFormat::Bool: {
      if (ValStr == "1" || ValStr.equals_insensitive("true"))
        Bits = 1;
      else if (ValStr == "0" || ValStr.equals_insensitive("false"))
        Bits = 0;
      else
        return failInvalid("bool");
      break;
    }
    case DataFormat::Int16: {
      int16_t V = 0;
      if (ValStr.getAsInteger(0, V))
        return failInvalid("int16");
      Bits = static_cast<uint16_t>(V);
      break;
    }
    case DataFormat::UInt16: {
      uint16_t V = 0;
      if (ValStr.getAsInteger(0, V))
        return failInvalid("uint16");
      Bits = V;
      break;
    }
    case DataFormat::Int32: {
      int32_t V = 0;
      if (ValStr.getAsInteger(0, V))
        return failInvalid("int32");
      Bits = static_cast<uint32_t>(V);
      break;
    }
    case DataFormat::UInt32: {
      uint32_t V = 0;
      if (ValStr.getAsInteger(0, V))
        return failInvalid("uint32");
      Bits = V;
      break;
    }
    case DataFormat::Float32: {
      float V = 0.0f;
      if (!to_float(ValStr, V))
        return failInvalid("float");
      uint32_t U = 0;
      std::memcpy(&U, &V, sizeof(U));
      Bits = U;
      break;
    }
    case DataFormat::Float64: {
      double V = 0.0;
      if (!to_float(ValStr, V))
        return failInvalid("double");
      std::memcpy(&Bits, &V, sizeof(Bits));
      break;
    }
    default:
      return failInvalid("unknown");
    }
    Opts.SpecConstants[SC.ConstantID] = Bits;
  }
  return Error::success();
}

} // namespace crucible_offload

//===- IntrinsicMap.cpp - DXIL opcode -> MSL mapping -------------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IntrinsicMap.h"

namespace crucible {

llvm::StringRef mslUnaryFunction(DXILOp Op, bool IsFloat) {
  switch (Op) {
  case DXILOp::FAbs:
    return "fabs";
  case DXILOp::Saturate:
    return "saturate";
  case DXILOp::Cos:
    return "cos";
  case DXILOp::Sin:
    return "sin";
  case DXILOp::Tan:
    return "tan";
  case DXILOp::Exp:
    return "exp2"; // DXIL Exp is 2^x
  case DXILOp::Frc:
    return "fract";
  case DXILOp::Log:
    return "log2"; // DXIL Log is log2(x)
  case DXILOp::Sqrt:
    return "sqrt";
  case DXILOp::Rsqrt:
    return "rsqrt";
  case DXILOp::RoundNE:
    return "rint";
  case DXILOp::RoundNI:
    return "floor";
  case DXILOp::RoundPI:
    return "ceil";
  case DXILOp::RoundZ:
    return "trunc";
  case DXILOp::Bfrev:
    return "reverse_bits";
  case DXILOp::Countbits:
    return "popcount";
  default:
    return {};
  }
}

llvm::StringRef mslBinaryFunction(DXILOp Op) {
  switch (Op) {
  case DXILOp::FMax:
  case DXILOp::IMax:
  case DXILOp::UMax:
    return "max";
  case DXILOp::FMin:
  case DXILOp::IMin:
  case DXILOp::UMin:
    return "min";
  default:
    return {};
  }
}

} // namespace crucible

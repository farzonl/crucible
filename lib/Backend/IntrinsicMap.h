//===- IntrinsicMap.h - DXIL opcode -> MSL mapping ---------------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_LIB_BACKEND_INTRINSICMAP_H
#define CRUCIBLE_LIB_BACKEND_INTRINSICMAP_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace crucible {

/// DXIL operation opcodes (matches hlsl::DXIL::OpCode). Only the subset the
/// backend currently understands is listed; extend alongside IntrinsicMap.cpp.
enum class DXILOp : uint32_t {
  // Unary float
  FAbs = 6,
  Saturate = 7,
  Cos = 12,
  Sin = 13,
  Tan = 14,
  Exp = 21, // 2^x in DXIL
  Frc = 22,
  Log = 23, // log2(x) in DXIL
  Sqrt = 24,
  Rsqrt = 25,
  RoundNE = 26,
  RoundNI = 27, // floor
  RoundPI = 28, // ceil
  RoundZ = 29,  // trunc
  // Unary int
  Bfrev = 30,
  Countbits = 31,
  // Binary
  FMax = 35,
  FMin = 36,
  IMax = 37,
  IMin = 38,
  UMax = 39,
  UMin = 40,
  // Tertiary
  FMad = 46,
  Fma = 47,
  IMad = 48,
  UMad = 49,
  // Dot
  Dot2 = 54,
  Dot3 = 55,
  Dot4 = 56,
  // Resources
  CreateHandle = 57,
  CBufferLoadLegacy = 59,
  BufferLoad = 68,
  BufferStore = 69,
  // Atomics
  AtomicBinOp = 78,
  AtomicCompareExchange = 79,
  // Synchronization
  Barrier = 80,
  // Compute IDs
  ThreadId = 93,
  GroupId = 94,
  ThreadIdInGroup = 95,
  FlattenedThreadIdInGroup = 96,
  // SM 6.6 resource handles
  AnnotateHandle = 216,
  CreateHandleFromBinding = 217,
};

/// Returns the metal_stdlib function implementing a unary DXIL op on the
/// given operand class, or empty if unmapped.
llvm::StringRef mslUnaryFunction(DXILOp Op, bool IsFloat);

/// Returns the metal_stdlib function implementing a binary DXIL op
/// (min/max family), or empty if unmapped.
llvm::StringRef mslBinaryFunction(DXILOp Op);

} // namespace crucible

#endif // CRUCIBLE_LIB_BACKEND_INTRINSICMAP_H

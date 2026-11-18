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

/// DXIL operation opcodes (matches hlsl::DXIL::OpCode in
/// DirectXShaderCompiler's DxilConstants.h). Only the subset the backend
/// currently understands is listed; extend alongside IntrinsicMap.cpp.
enum class DXILOp : uint32_t {
  // Unary float
  FAbs = 6,
  Saturate = 7,
  IsNaN = 8,
  IsInf = 9,
  IsFinite = 10,
  Cos = 12,
  Sin = 13,
  Tan = 14,
  Acos = 15,
  Asin = 16,
  Atan = 17,
  Hcos = 18,
  Hsin = 19,
  Htan = 20,
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
  FirstbitLo = 32,
  FirstbitHi = 33,
  FirstbitSHi = 34,
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
  // Bitfield
  Ibfe = 51,
  Ubfe = 52,
  Bfi = 53,
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
  // Wave ops
  WaveIsFirstLane = 110,
  WaveGetLaneIndex = 111,
  WaveGetLaneCount = 112,
  WaveAnyTrue = 113,
  WaveAllTrue = 114,
  WaveActiveAllEqual = 115,
  WaveActiveBallot = 116,
  WaveReadLaneAt = 117,
  WaveReadLaneFirst = 118,
  WaveActiveOp = 119,
  WaveActiveBit = 120,
  WavePrefixOp = 121,
  // SM 6.2+ raw/structured buffer access
  RawBufferLoad = 139,
  RawBufferStore = 140,
  // SM 6.4 packed dot products
  Dot2AddHalf = 162,
  Dot4AddI8Packed = 163,
  Dot4AddU8Packed = 164,
  // SM 6.6 resource handles
  AnnotateHandle = 216,
  CreateHandleFromBinding = 217,
};

/// hlsl::DXIL::AtomicBinOpCode (operand 2 of dx.op.atomicBinOp).
enum class DXILAtomicBinOp : uint32_t {
  Add = 0,
  And = 1,
  Or = 2,
  Xor = 3,
  IMin = 4,
  IMax = 5,
  UMin = 6,
  UMax = 7,
  Exchange = 8,
};

/// hlsl::DXIL::WaveOpKind (operand 2 of dx.op.waveActiveOp / wavePrefixOp).
enum class DXILWaveOpKind : uint32_t { Sum = 0, Product = 1, Min = 2, Max = 3 };

/// hlsl::DXIL::WaveBitOpKind (operand 2 of dx.op.waveActiveBit).
enum class DXILWaveBitOpKind : uint32_t { And = 0, Or = 1, Xor = 2 };

/// Returns the metal_stdlib function implementing a unary DXIL op on the
/// given operand class, or empty if unmapped.
llvm::StringRef mslUnaryFunction(DXILOp Op, bool IsFloat);

/// Returns the metal_stdlib function implementing a binary DXIL op
/// (min/max family), or empty if unmapped.
llvm::StringRef mslBinaryFunction(DXILOp Op);

} // namespace crucible

#endif // CRUCIBLE_LIB_BACKEND_INTRINSICMAP_H

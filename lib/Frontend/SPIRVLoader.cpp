//===- SPIRVLoader.cpp - MLIR SPIR-V frontend -------------------*- C++ -*-===//
//
// Part of Crucible, an open-source shader-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Deserializes SPIR-V binaries into MLIR's spirv dialect, extracts binding and
// stage reflection, rewrites shader-ABI ops (built-ins, descriptor-bound
// buffers) into Crucible's layered MLIR representation (gpu.* + llvm.*), and
// lowers remaining SPIR-V ops via MLIR's ConvertSPIRVToLLVM pass.
//
//===----------------------------------------------------------------------===//

#include "Frontend.h"
#include "../Transforms/MiddleEnd.h"
#include "crucible/Diagnostic.h"
#include "crucible/Dialect/CrucibleDialect.h"

#include "spirv/unified1/GLSL.std.450.h"
#define SPV_ENABLE_UTILITY_CODE
#include "spirv/unified1/spirv.hpp"
#undef SPV_ENABLE_UTILITY_CODE
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Endian.h"

#include "mlir/Conversion/SPIRVToLLVM/SPIRVToLLVM.h"
#include "mlir/Conversion/SPIRVToLLVM/SPIRVToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/SPIRV/Deserialization.h"
#include "mlir/Transforms/Passes.h"

using namespace llvm;

namespace crucible {

static constexpr uint32_t kSPIRVMagic = 0x07230203u;

bool isSPIRVBinary(MemoryBufferRef Buf) {
  if (Buf.getBufferSize() < 4)
    return false;
  uint32_t Magic = support::endian::read32le(Buf.getBufferStart());
  return Magic == kSPIRVMagic;
}

namespace {

static Error expandFloatControls2(SmallVectorImpl<uint32_t> &Words,
                                  bool &RequiresStrictFP) {
  if (Words.size() < 5)
    return makeError("truncated SPIR-V header");
  bool HasControls = false;
  DenseMap<uint32_t, SmallVector<uint32_t>> Types;
  DenseMap<uint32_t, uint32_t> ValueTypes, Constants, UIntTypes;
  DenseSet<uint32_t> GLSLImports, Capabilities;
  uint32_t BoolTy = 0;
  auto IsString = [](ArrayRef<uint32_t> W, StringRef S) {
    StringRef Bytes(reinterpret_cast<const char *>(W.data()), W.size() * 4);
    return Bytes.size() > S.size() && Bytes.starts_with(S) &&
           Bytes[S.size()] == '\0';
  };
  for (size_t I = 5; I < Words.size();) {
    uint32_t N = Words[I] >> 16, Op = Words[I] & 0xffff;
    if (!N || N > Words.size() - I)
      return makeError("malformed SPIR-V instruction");
    if (Op == spv::OpCapability && N == 2) {
      Capabilities.insert(Words[I + 1]);
      HasControls |= Words[I + 1] == spv::CapabilityFloatControls2;
    }
    if ((Op == spv::OpTypeFloat && N == 3) ||
        (Op == spv::OpTypeInt && N == 4) ||
        (Op == spv::OpTypeVector && N == 4) ||
        (Op == spv::OpTypeMatrix && N == 4) ||
        (Op == spv::OpTypeStruct && N >= 2))
      Types[Words[I + 1]] =
          SmallVector<uint32_t>(Words.begin() + I, Words.begin() + I + N);
    if (Op == spv::OpTypeBool && N == 2)
      BoolTy = Words[I + 1];
    if (Op == spv::OpTypeInt && N == 4 && Words[I + 3] == 0)
      UIntTypes[Words[I + 2]] = Words[I + 1];
    if (Op == spv::OpConstant && N == 4)
      Constants[Words[I + 2]] = Words[I + 3];
    if (Op == spv::OpExtInstImport && N >= 3 &&
        IsString(ArrayRef(Words).slice(I + 2, N - 2), "GLSL.std.450"))
      GLSLImports.insert(Words[I + 1]);
    bool HasResult, HasType;
    spv::HasResultAndType(static_cast<spv::Op>(Op), &HasResult, &HasType);
    if (HasResult && HasType && N >= 3)
      ValueTypes[Words[I + 2]] = Words[I + 1];
    I += N;
  }
  if (!HasControls)
    return Error::success();
  RequiresStrictFP = true;
  auto IsSupportedFloat = [&](uint32_t Ty) {
    auto T = Types.find(Ty);
    if (T == Types.end())
      return false;
    if ((T->second[0] & 0xffff) == spv::OpTypeVector) {
      if (T->second[3] < 2 || T->second[3] > 4)
        return false;
      T = Types.find(T->second[2]);
      if (T == Types.end())
        return false;
    }
    return (T->second[0] & 0xffff) == spv::OpTypeFloat &&
           (T->second[2] == 16 || T->second[2] == 32);
  };

  SmallVector<uint32_t> NewCapabilities, Declarations, Body;
  uint32_t NextID = Words[3];
  auto Emit = [](SmallVectorImpl<uint32_t> &Out, spv::Op Op,
                 ArrayRef<uint32_t> Args) {
    Out.push_back(((Args.size() + 1) << 16) | Op);
    Out.append(Args.begin(), Args.end());
  };
  struct BitTypes {
    uint32_t UInt, Bool, Zero, AbsMask, Infinity;
  };
  DenseMap<uint32_t, BitTypes> Cached;
  auto ConstantFor = [&](uint32_t UInt, uint32_t Width, uint32_t Lanes,
                         uint32_t Bits) {
    uint32_t ID = NextID++;
    Emit(Declarations, spv::OpConstant, {UIntTypes[Width], ID, Bits});
    if (Lanes != 1) {
      uint32_t Vector = NextID++;
      SmallVector<uint32_t> Args{UInt, Vector};
      Args.append(Lanes, ID);
      Emit(Declarations, spv::OpConstantComposite, Args);
      return Vector;
    }
    return ID;
  };
  constexpr uint32_t ValidFlags = 0x7000f;
  // Relaxations are permissions, not requirements. These operations implement
  // the strictest mode using integer bit patterns or explicitly safe Metal
  // arithmetic. Other FP operations remain rejected, not silently relaxed.
  for (size_t I = 5; I < Words.size();) {
    uint32_t N = Words[I] >> 16;
    auto Op = static_cast<spv::Op>(Words[I] & 0xffff);
    if (Op == spv::OpExecutionMode && N >= 3 &&
        (Words[I + 2] == spv::ExecutionModeDenormPreserve ||
         Words[I + 2] == spv::ExecutionModeDenormFlushToZero ||
         Words[I + 2] == spv::ExecutionModeRoundingModeRTZ))
      return makeError(
          "unsupported floating-point execution mode in FloatControls2");
    if (Op == spv::OpDecorate && N == 4 &&
        Words[I + 2] == spv::DecorationFPRoundingMode &&
        Words[I + 3] != spv::FPRoundingModeRTE)
      return makeError(
          "unsupported floating-point rounding decoration in FloatControls2");
    if (Op == spv::OpCapability && N == 2 &&
        Words[I + 1] == spv::CapabilityFloatControls2) {
      I += N;
      continue;
    }
    if (Op == spv::OpExtension && IsString(ArrayRef(Words).slice(I + 1, N - 1),
                                           "SPV_KHR_float_controls2")) {
      I += N;
      continue;
    }
    if (Op == spv::OpExecutionModeId && N >= 3 &&
        Words[I + 2] == spv::ExecutionModeFPFastMathDefault) {
      if (N != 5 || !Types.contains(Words[I + 3]) ||
          (Types[Words[I + 3]][0] & 0xffff) != spv::OpTypeFloat ||
          !Constants.contains(Words[I + 4]) ||
          (Constants[Words[I + 4]] & ~ValidFlags))
        return makeError("invalid FloatControls2 default fast-math mode");
      auto C = Types.find(ValueTypes.lookup(Words[I + 4]));
      if (C == Types.end() || (C->second[0] & 0xffff) != spv::OpTypeInt ||
          C->second[2] != 32)
        return makeError(
            "FloatControls2 default flags require an integer32 constant");
      I += N;
      continue;
    }
    if (Op == spv::OpDecorate && N >= 3 &&
        Words[I + 2] == spv::DecorationFPFastMathMode) {
      if (N != 4 || (Words[I + 3] & ~ValidFlags))
        return makeError("invalid FloatControls2 fast-math decoration");
      I += N;
      continue;
    }
    bool Abs = Op == spv::OpExtInst && N == 6 &&
               GLSLImports.contains(Words[I + 3]) &&
               Words[I + 4] == GLSLstd450FAbs;
    bool Rounding = Op == spv::OpExtInst && N == 6 &&
                    GLSLImports.contains(Words[I + 3]) &&
                    (Words[I + 4] == GLSLstd450RoundEven ||
                     Words[I + 4] == GLSLstd450Floor);
    bool Min = Op == spv::OpExtInst && N == 7 &&
               GLSLImports.contains(Words[I + 3]) &&
               Words[I + 4] == GLSLstd450NMin;
    bool Compare = Op == spv::OpFOrdEqual || Op == spv::OpFUnordEqual ||
                   Op == spv::OpFOrdNotEqual || Op == spv::OpFUnordNotEqual;
    bool Determinant = Op == spv::OpExtInst && N == 6 &&
                       GLSLImports.contains(Words[I + 3]) &&
                       Words[I + 4] == GLSLstd450Determinant;
    if (Determinant) {
      uint32_t Ty = Words[I + 1], Matrix = Words[I + 5];
      auto M = Types.find(ValueTypes.lookup(Matrix));
      if (M == Types.end() || (M->second[0] & 0xffff) != spv::OpTypeMatrix ||
          M->second[3] != 2)
        return makeError("FloatControls2 determinant requires a 2x2 matrix");
      auto V = Types.find(M->second[2]);
      if (V == Types.end() || (V->second[0] & 0xffff) != spv::OpTypeVector ||
          V->second[3] != 2 || V->second[2] != Ty || !IsSupportedFloat(Ty))
        return makeError("FloatControls2 determinant requires float16/32");
      if (NextID > UINT32_MAX - 8)
        return makeError("oversized FloatControls2 determinant");
      SmallVector<uint32_t, 4> Elements;
      for (uint32_t Column = 0; Column != 2; ++Column)
        for (uint32_t Row = 0; Row != 2; ++Row) {
          uint32_t ID = NextID++;
          Emit(Body, spv::OpCompositeExtract, {Ty, ID, Matrix, Column, Row});
          Elements.push_back(ID);
        }
      uint32_t Diagonal = NextID++, Cross = NextID++;
      Emit(Body, spv::OpFMul, {Ty, Diagonal, Elements[0], Elements[3]});
      Emit(Body, spv::OpFMul, {Ty, Cross, Elements[1], Elements[2]});
      Emit(Body, spv::OpFSub, {Ty, Words[I + 2], Diagonal, Cross});
    } else if (Abs || Rounding || Min || Compare) {
      if ((Compare && N != 5) || NextID > UINT32_MAX - 256)
        return makeError("malformed or oversized FloatControls2 operation");
      uint32_t Ty = Compare ? ValueTypes.lookup(Words[I + 3]) : Words[I + 1];
      auto T = Types.find(Ty);
      if (T == Types.end())
        return makeError("unknown FloatControls2 floating-point type");
      uint32_t Scalar = Ty, Lanes = 1;
      if ((T->second[0] & 0xffff) == spv::OpTypeVector) {
        Scalar = T->second[2];
        Lanes = T->second[3];
      }
      auto S = Types.find(Scalar);
      if (S == Types.end() || (S->second[0] & 0xffff) != spv::OpTypeFloat ||
          (S->second[2] != 16 && S->second[2] != 32) || Lanes > 4)
        return makeError(
            "FloatControls2 bitwise operations require float16/32");
      if (!Cached.contains(Ty)) {
        uint32_t Width = S->second[2], UInt = UIntTypes.lookup(Width);
        if (!UInt) {
          UInt = NextID++;
          Emit(Declarations, spv::OpTypeInt, {UInt, Width, 0});
          UIntTypes[Width] = UInt;
        }
        if (Width == 16 && Capabilities.insert(spv::CapabilityInt16).second)
          Emit(NewCapabilities, spv::OpCapability, {spv::CapabilityInt16});
        if (!BoolTy) {
          BoolTy = NextID++;
          Emit(Declarations, spv::OpTypeBool, {BoolTy});
        }
        uint32_t Bool = BoolTy;
        if (Lanes != 1) {
          auto FindVector = [&](uint32_t Elem) {
            for (const auto &KV : Types)
              if ((KV.second[0] & 0xffff) == spv::OpTypeVector &&
                  KV.second[2] == Elem && KV.second[3] == Lanes)
                return KV.first;
            uint32_t ID = NextID++;
            Emit(Declarations, spv::OpTypeVector, {ID, Elem, Lanes});
            Types[ID] = {(4u << 16) | spv::OpTypeVector, ID, Elem, Lanes};
            return ID;
          };
          UInt = FindVector(UInt);
          Bool = FindVector(BoolTy);
        }
        auto Constant = [&](uint32_t Bits) {
          return ConstantFor(UInt, Width, Lanes, Bits);
        };
        Cached[Ty] = {UInt, Bool, Constant(0),
                      Constant(Width == 16 ? 0x7fff : 0x7fffffff),
                      Constant(Width == 16 ? 0x7c00 : 0x7f800000)};
      }
      auto C = Cached[Ty];
      auto Unary = [&](spv::Op O, uint32_t Type, uint32_t A) {
        uint32_t ID = NextID++;
        Emit(Body, O, {Type, ID, A});
        return ID;
      };
      auto Binary = [&](spv::Op O, uint32_t Type, uint32_t A, uint32_t B) {
        uint32_t ID = NextID++;
        Emit(Body, O, {Type, ID, A, B});
        return ID;
      };
      auto Select = [&](uint32_t Cond, uint32_t A, uint32_t B) {
        uint32_t ID = NextID++;
        Emit(Body, spv::OpSelect, {C.UInt, ID, Cond, A, B});
        return ID;
      };
      auto Constant = [&](uint32_t Bits) {
        return ConstantFor(C.UInt, S->second[2], Lanes, Bits);
      };
      uint32_t A = Unary(spv::OpBitcast, C.UInt, Words[I + (Compare ? 3 : 5)]);
      uint32_t MagA = Binary(spv::OpBitwiseAnd, C.UInt, A, C.AbsMask);
      if (Abs) {
        Emit(Body, spv::OpBitcast, {Ty, Words[I + 2], MagA});
      } else if (Rounding) {
        uint32_t Mantissa = S->second[2] == 16 ? 10 : 23;
        uint32_t Bias = S->second[2] == 16 ? 15 : 127;
        uint32_t One = Constant(1), Mant = Constant(Mantissa);
        uint32_t Exp = Binary(spv::OpShiftRightLogical, C.UInt, MagA, Mant);
        uint32_t BiasC = Constant(Bias), Limit = Constant(Bias + Mantissa);
        uint32_t Small = Binary(spv::OpULessThan, C.Bool, Exp, BiasC);
        uint32_t Fractional = Binary(spv::OpULessThan, C.Bool, Exp, Limit);
        uint32_t Range = Binary(spv::OpLogicalAnd, C.Bool, Fractional,
                                Unary(spv::OpLogicalNot, C.Bool, Small));
        // Select a safe shift even for zeros, subnormals, NaNs and infinities.
        uint32_t Shift =
            Select(Range, Binary(spv::OpISub, C.UInt, Limit, Exp), One);
        uint32_t Unit = Binary(spv::OpShiftLeftLogical, C.UInt, One, Shift);
        uint32_t Mask = Binary(spv::OpISub, C.UInt, Unit, One);
        uint32_t Fraction = Binary(spv::OpBitwiseAnd, C.UInt, MagA, Mask);
        uint32_t Whole = Binary(spv::OpBitwiseAnd, C.UInt, A,
                                Unary(spv::OpNot, C.UInt, Mask));
        uint32_t Sign =
            Binary(spv::OpBitwiseAnd, C.UInt, A,
                   Constant(S->second[2] == 16 ? 0x8000 : 0x80000000));
        uint32_t OneBits = Constant(Bias << Mantissa);
        uint32_t SmallBits, Increment;
        if (Words[I + 4] == GLSLstd450RoundEven) {
          uint32_t Half = Binary(spv::OpShiftRightLogical, C.UInt, Unit, One);
          uint32_t Greater =
              Binary(spv::OpUGreaterThan, C.Bool, Fraction, Half);
          uint32_t Tie = Binary(spv::OpIEqual, C.Bool, Fraction, Half);
          uint32_t OddBits = Binary(
              spv::OpBitwiseAnd, C.UInt,
              Binary(spv::OpShiftRightLogical, C.UInt, Whole, Shift), One);
          uint32_t Odd = Binary(spv::OpINotEqual, C.Bool, OddBits, C.Zero);
          Increment = Binary(spv::OpLogicalOr, C.Bool, Greater,
                             Binary(spv::OpLogicalAnd, C.Bool, Tie, Odd));
          uint32_t AboveHalf = Binary(spv::OpUGreaterThan, C.Bool, MagA,
                                      Constant((Bias - 1) << Mantissa));
          SmallBits = Binary(spv::OpBitwiseOr, C.UInt, Sign,
                             Select(AboveHalf, OneBits, C.Zero));
        } else {
          uint32_t Negative = Binary(spv::OpINotEqual, C.Bool, Sign, C.Zero);
          uint32_t HasFraction =
              Binary(spv::OpINotEqual, C.Bool, Fraction, C.Zero);
          Increment = Binary(spv::OpLogicalAnd, C.Bool, Negative, HasFraction);
          uint32_t Nonzero = Binary(spv::OpINotEqual, C.Bool, MagA, C.Zero);
          SmallBits = Binary(
              spv::OpBitwiseOr, C.UInt, Sign,
              Select(Binary(spv::OpLogicalAnd, C.Bool, Negative, Nonzero),
                     OneBits, C.Zero));
        }
        uint32_t Rounded =
            Binary(spv::OpIAdd, C.UInt, Whole, Select(Increment, Unit, C.Zero));
        uint32_t Bits =
            Select(Small, SmallBits, Select(Fractional, Rounded, A));
        Emit(Body, spv::OpBitcast, {Ty, Words[I + 2], Bits});
      } else if (Min) {
        uint32_t B = Unary(spv::OpBitcast, C.UInt, Words[I + 6]);
        uint32_t MagB = Binary(spv::OpBitwiseAnd, C.UInt, B, C.AbsMask);
        uint32_t NegA = Binary(spv::OpUGreaterThan, C.Bool, A, C.AbsMask);
        uint32_t NegB = Binary(spv::OpUGreaterThan, C.Bool, B, C.AbsMask);
        uint32_t SameSign = Binary(spv::OpLogicalEqual, C.Bool, NegA, NegB);
        uint32_t Asc = Binary(spv::OpULessThan, C.Bool, MagA, MagB);
        uint32_t Desc = Binary(spv::OpUGreaterThan, C.Bool, MagA, MagB);
        uint32_t OrderedLess = NextID++;
        Emit(Body, spv::OpSelect, {C.Bool, OrderedLess, NegA, Desc, Asc});
        uint32_t Less = NextID++;
        Emit(Body, spv::OpSelect, {C.Bool, Less, SameSign, OrderedLess, NegA});
        uint32_t Zeros = Binary(spv::OpLogicalAnd, C.Bool,
                                Binary(spv::OpIEqual, C.Bool, MagA, C.Zero),
                                Binary(spv::OpIEqual, C.Bool, MagB, C.Zero));
        uint32_t Bits = Select(Zeros, Binary(spv::OpBitwiseOr, C.UInt, A, B),
                               Select(Less, A, B));
        uint32_t NanA = Binary(spv::OpUGreaterThan, C.Bool, MagA, C.Infinity);
        uint32_t NanB = Binary(spv::OpUGreaterThan, C.Bool, MagB, C.Infinity);
        Bits = Select(NanB, A, Select(NanA, B, Bits));
        Emit(Body, spv::OpBitcast, {Ty, Words[I + 2], Bits});
      } else {
        uint32_t B = Unary(spv::OpBitcast, C.UInt, Words[I + 4]);
        uint32_t MagB = Binary(spv::OpBitwiseAnd, C.UInt, B, C.AbsMask);
        uint32_t NanA = Binary(spv::OpUGreaterThan, C.Bool, MagA, C.Infinity);
        uint32_t NanB = Binary(spv::OpUGreaterThan, C.Bool, MagB, C.Infinity);
        uint32_t Nan = Binary(spv::OpLogicalOr, C.Bool, NanA, NanB);
        uint32_t ZeroA = Binary(spv::OpIEqual, C.Bool, MagA, C.Zero);
        uint32_t ZeroB = Binary(spv::OpIEqual, C.Bool, MagB, C.Zero);
        uint32_t Zeros = Binary(spv::OpLogicalAnd, C.Bool, ZeroA, ZeroB);
        uint32_t Same = Binary(spv::OpIEqual, C.Bool, A, B);
        uint32_t Equal = Binary(spv::OpLogicalOr, C.Bool, Same, Zeros);
        if (Op == spv::OpFOrdNotEqual || Op == spv::OpFUnordNotEqual)
          Equal = Unary(spv::OpLogicalNot, C.Bool, Equal);
        bool Ordered = Op == spv::OpFOrdEqual || Op == spv::OpFOrdNotEqual;
        if (Ordered)
          Nan = Unary(spv::OpLogicalNot, C.Bool, Nan);
        Emit(Body, Ordered ? spv::OpLogicalAnd : spv::OpLogicalOr,
             {Words[I + 1], Words[I + 2], Equal, Nan});
      }
    } else if (Op == spv::OpExtInst && N >= 6 &&
               GLSLImports.contains(Words[I + 3]) &&
               ((N == 8 && Words[I + 4] == GLSLstd450Fma) ||
                (N == 7 && Words[I + 4] == GLSLstd450Modf) ||
                (N == 7 && Words[I + 4] == GLSLstd450Atan2) ||
                (N == 7 && Words[I + 4] == GLSLstd450Reflect) ||
                (N == 6 && (Words[I + 4] == GLSLstd450ModfStruct ||
                            Words[I + 4] == GLSLstd450Acosh ||
                            Words[I + 4] == GLSLstd450Cosh ||
                            Words[I + 4] == GLSLstd450Exp ||
                            Words[I + 4] == GLSLstd450Log2)))) {
      uint32_t Ty = Words[I + 1];
      if (Words[I + 4] == GLSLstd450ModfStruct) {
        auto T = Types.find(Ty);
        if (T == Types.end() || (T->second[0] & 0xffff) != spv::OpTypeStruct ||
            T->second.size() != 4 || T->second[2] != T->second[3])
          return makeError("invalid FloatControls2 ModfStruct result type");
        Ty = T->second[2];
      }
      if (!IsSupportedFloat(Ty))
        return makeError(
            "FloatControls2 math requires scalar/vector float16/32");
      Body.append(Words.begin() + I, Words.begin() + I + N);
    } else {
      if (Op == spv::OpFSub && !IsSupportedFloat(Words[I + 1]))
        return makeError("FloatControls2 subtraction requires float16/32");
      if (Op == spv::OpFConvert &&
          (!IsSupportedFloat(Words[I + 1]) ||
           !IsSupportedFloat(ValueTypes.lookup(Words[I + 3]))))
        return makeError("FloatControls2 conversion requires float16/32");
      switch (Op) {
      case spv::OpCapability:
      case spv::OpExtension:
      case spv::OpExtInstImport:
      case spv::OpMemoryModel:
      case spv::OpEntryPoint:
      case spv::OpExecutionMode:
      case spv::OpName:
      case spv::OpMemberName:
      case spv::OpString:
      case spv::OpSource:
      case spv::OpSourceExtension:
      case spv::OpSourceContinued:
      case spv::OpLine:
      case spv::OpNoLine:
      case spv::OpDecorate:
      case spv::OpMemberDecorate:
      case spv::OpTypeVoid:
      case spv::OpTypeBool:
      case spv::OpTypeInt:
      case spv::OpTypeFloat:
      case spv::OpTypeVector:
      case spv::OpTypeMatrix:
      case spv::OpTypeArray:
      case spv::OpTypeRuntimeArray:
      case spv::OpTypeStruct:
      case spv::OpTypePointer:
      case spv::OpTypeFunction:
      case spv::OpConstant:
      case spv::OpConstantTrue:
      case spv::OpConstantFalse:
      case spv::OpConstantComposite:
      case spv::OpConstantNull:
      case spv::OpUndef:
      case spv::OpVariable:
      case spv::OpLoad:
      case spv::OpStore:
      case spv::OpAccessChain:
      case spv::OpInBoundsAccessChain:
      case spv::OpFunction:
      case spv::OpFunctionParameter:
      case spv::OpFunctionEnd:
      case spv::OpFunctionCall:
      case spv::OpReturn:
      case spv::OpReturnValue:
      case spv::OpLabel:
      case spv::OpSelectionMerge:
      case spv::OpLoopMerge:
      case spv::OpBranch:
      case spv::OpBranchConditional:
      case spv::OpSwitch:
      case spv::OpPhi:
      case spv::OpSelect:
      case spv::OpCopyObject:
      case spv::OpCompositeConstruct:
      case spv::OpCompositeExtract:
      case spv::OpCompositeInsert:
      case spv::OpVectorShuffle:
      case spv::OpVectorExtractDynamic:
      case spv::OpVectorInsertDynamic:
      case spv::OpBitcast:
      case spv::OpUConvert:
      case spv::OpSConvert:
      case spv::OpIAdd:
      case spv::OpFSub:
      case spv::OpFConvert:
      case spv::OpISub:
      case spv::OpIMul:
      case spv::OpIEqual:
      case spv::OpINotEqual:
      case spv::OpULessThan:
      case spv::OpUGreaterThan:
      case spv::OpSLessThan:
      case spv::OpSGreaterThan:
      case spv::OpLogicalAnd:
      case spv::OpLogicalOr:
      case spv::OpLogicalNot:
      case spv::OpBitwiseAnd:
      case spv::OpBitwiseOr:
      case spv::OpBitwiseXor:
      case spv::OpShiftLeftLogical:
      case spv::OpShiftRightLogical:
        Body.append(Words.begin() + I, Words.begin() + I + N);
        break;
      default:
        return makeError(
            "unsupported operation {0} in FloatControls2 strict subset",
            static_cast<uint32_t>(Op));
      }
    }
    I += N;
  }
  SmallVector<uint32_t> Normalized(Words.begin(), Words.begin() + 5);
  Normalized[3] = NextID;
  Normalized.append(NewCapabilities);
  bool Inserted = false;
  for (size_t I = 0; I < Body.size();) {
    uint32_t N = Body[I] >> 16;
    if (!Inserted && (Body[I] & 0xffff) == spv::OpFunction) {
      Normalized.append(Declarations);
      Inserted = true;
    }
    Normalized.append(Body.begin() + I, Body.begin() + I + N);
    I += N;
  }
  if (!Inserted && !Declarations.empty())
    return makeError("missing function for FloatControls2 declarations");
  Words.assign(Normalized.begin(), Normalized.end());
  return Error::success();
}

static Error expandFmaKHR(SmallVectorImpl<uint32_t> &Words) {
  if (Words.size() < 5)
    return makeError("truncated SPIR-V header");
  bool HasFma = false, HasLdexp = false;
  uint32_t GLSL = 0;
  DenseSet<uint32_t> GLSLImports;
  DenseSet<uint32_t> Float32Types;
  auto IsString = [](ArrayRef<uint32_t> Words, StringRef Text) {
    StringRef Bytes(reinterpret_cast<const char *>(Words.data()),
                    Words.size() * sizeof(uint32_t));
    return Bytes.size() > Text.size() && Bytes.starts_with(Text) &&
           Bytes[Text.size()] == '\0';
  };
  for (size_t I = 5; I < Words.size();) {
    uint32_t Count = Words[I] >> 16, Op = Words[I] & 0xffff;
    if (!Count || Count > Words.size() - I)
      return makeError("malformed SPIR-V instruction");
    if (Op == spv::OpFmaKHR)
      HasFma = true;
    if (Op == spv::OpExtInstImport && Count >= 3 &&
        IsString(ArrayRef(Words).slice(I + 2, Count - 2), "GLSL.std.450")) {
      GLSL = Words[I + 1];
      GLSLImports.insert(GLSL);
    }
    if (Op == spv::OpExtInst && Count == 7 &&
        GLSLImports.contains(Words[I + 3]) && Words[I + 4] == GLSLstd450Ldexp)
      HasLdexp = true;
    if (Op == spv::OpTypeFloat && Count == 3 && Words[I + 2] == 32)
      Float32Types.insert(Words[I + 1]);
    if (Op == spv::OpTypeVector && Count == 4 &&
        Float32Types.contains(Words[I + 2]) && Words[I + 3] >= 2 &&
        Words[I + 3] <= 4)
      Float32Types.insert(Words[I + 1]);
    I += Count;
  }
  if (!HasFma && !HasLdexp)
    return Error::success();

  // Both intrinsics support the same float32 execution-mode subset.
  StringRef Instruction = HasFma ? "FmaKHR" : "ldexp";
  for (size_t I = 5; I < Words.size();) {
    uint32_t Count = Words[I] >> 16, Op = Words[I] & 0xffff;
    if (Op == spv::OpExecutionMode && Count >= 3) {
      uint32_t Mode = Words[I + 2];
      if (Mode == spv::ExecutionModeRoundingModeRTZ ||
          Mode == spv::ExecutionModeDenormPreserve ||
          Mode == spv::ExecutionModeDenormFlushToZero ||
          Mode == spv::ExecutionModeSignedZeroInfNanPreserve ||
          (Mode == spv::ExecutionModeRoundingModeRTE &&
           (Count != 4 || Words[I + 3] != 32)))
        return makeError("unsupported floating-point execution mode for {0}",
                         Instruction);
    }
    if (Op == spv::OpDecorate && Count == 4 &&
        Words[I + 2] == spv::DecorationFPRoundingMode &&
        Words[I + 3] != spv::FPRoundingModeRTE)
      return makeError("unsupported floating-point rounding decoration for {0}",
                       Instruction);
    I += Count;
  }
  if (!HasFma)
    return Error::success();

  SmallVector<uint32_t> Normalized(Words.begin(), Words.begin() + 5);
  bool AddImport = !GLSL;
  if (AddImport) {
    if (Words[3] == UINT32_MAX)
      return makeError("SPIR-V ID bound is too large for FmaKHR normalization");
    GLSL = Normalized[3]++;
  }
  for (size_t I = 5; I < Words.size();) {
    uint32_t Count = Words[I] >> 16, Op = Words[I] & 0xffff;
    if (Op == spv::OpCapability && Count == 2 &&
        Words[I + 1] == spv::CapabilityFMAKHR) {
      // The core operation is implemented as LLVM fma via GLSL Fma, not
      // discarded as an unknown capability. LLVM/Metal retain fused rounding.
      I += Count;
      continue;
    }
    if (Op == spv::OpExtension && Count >= 2 &&
        IsString(ArrayRef(Words).slice(I + 1, Count - 1), "SPV_KHR_fma")) {
      I += Count;
      continue;
    }
    if (Op == spv::OpMemoryModel && AddImport) {
      Normalized.append({(6u << 16) | spv::OpExtInstImport, GLSL, 0x4c534c47,
                         0x6474732e, 0x3035342e, 0});
      AddImport = false;
    }
    if (Op == spv::OpFmaKHR) {
      if (Count != 6 || !Float32Types.contains(Words[I + 1]))
        return makeError("FmaKHR supports only scalar/vector float32");
      Normalized.append({(8u << 16) | spv::OpExtInst, Words[I + 1],
                         Words[I + 2], GLSL, GLSLstd450Fma, Words[I + 3],
                         Words[I + 4], Words[I + 5]});
    } else {
      Normalized.append(Words.begin() + I, Words.begin() + I + Count);
    }
    I += Count;
  }
  if (AddImport)
    return makeError("missing SPIR-V memory model for FmaKHR normalization");
  Words.assign(Normalized.begin(), Normalized.end());
  return Error::success();
}

// Normalize the GLSL aggregate operation missing from MLIR's deserializer.
static Error expandModfStruct(SmallVectorImpl<uint32_t> &Words) {
  if (Words.size() < 5)
    return makeError("truncated SPIR-V header");
  DenseMap<uint32_t, SmallVector<uint32_t, 4>> Types;
  DenseMap<uint32_t, uint32_t> TypeKinds;
  DenseSet<uint32_t> GLSLImports;
  DenseSet<uint32_t> Capabilities;
  DenseMap<uint32_t, uint32_t> UnsignedTypes;
  uint32_t BoolTy = 0;
  bool HasModf = false;
  for (size_t I = 5; I < Words.size();) {
    uint32_t Count = Words[I] >> 16, Op = Words[I] & 0xffff;
    if (!Count || Count > Words.size() - I)
      return makeError("malformed SPIR-V instruction");
    if (Op == spv::OpExtInstImport && Count >= 3) {
      StringRef Name(reinterpret_cast<const char *>(&Words[I + 2]),
                     (Count - 2) * sizeof(uint32_t));
      if (Name.starts_with(StringRef("GLSL.std.450\0", 13)))
        GLSLImports.insert(Words[I + 1]);
    }
    if (Op == spv::OpTypeBool && Count == 2)
      BoolTy = Words[I + 1];
    if (Op == spv::OpCapability && Count == 2)
      Capabilities.insert(Words[I + 1]);
    if (Op == spv::OpTypeInt && Count == 4 && Words[I + 3] == 0)
      UnsignedTypes[Words[I + 2]] = Words[I + 1];
    if ((Op == spv::OpTypeFloat && Count == 3) ||
        (Op == spv::OpTypeVector && Count == 4) ||
        (Op == spv::OpTypeStruct && Count >= 2)) {
      Types[Words[I + 1]] = SmallVector<uint32_t, 4>(Words.begin() + I + 1,
                                                     Words.begin() + I + Count);
      TypeKinds[Words[I + 1]] = Op;
    }
    if (Op == spv::OpExtInst && Count >= 5 &&
        GLSLImports.contains(Words[I + 3]) &&
        (Words[I + 4] == GLSLstd450ModfStruct ||
         Words[I + 4] == GLSLstd450Modf))
      HasModf = true;
    I += Count;
  }
  if (!HasModf)
    return Error::success();

  SmallVector<uint32_t> NewCapabilities, Declarations, Body;
  uint32_t NextID = Words[3];
  auto Emit = [](SmallVectorImpl<uint32_t> &Out, spv::Op Op,
                 ArrayRef<uint32_t> Args) {
    Out.push_back(((Args.size() + 1) << 16) | Op);
    Out.append(Args.begin(), Args.end());
  };
  struct Constants {
    uint32_t Bool, UInt, Zero, SignMask, AbsMask, Infinity, One;
  };
  DenseMap<uint32_t, Constants> Cached;
  DenseMap<uint32_t, uint32_t> BoolVectors;
  for (size_t I = 5; I < Words.size();) {
    uint32_t Count = Words[I] >> 16, Op = Words[I] & 0xffff;
    if (Op != spv::OpExtInst || Count < 5 ||
        !GLSLImports.contains(Words[I + 3]) ||
        (Words[I + 4] != GLSLstd450ModfStruct &&
         Words[I + 4] != GLSLstd450Modf)) {
      Body.append(Words.begin() + I, Words.begin() + I + Count);
      I += Count;
      continue;
    }
    bool PointerResult = Words[I + 4] == GLSLstd450Modf;
    if (Count != (PointerResult ? 7 : 6) || NextID > UINT32_MAX - 64)
      return makeError("malformed or oversized SPIR-V Modf");
    uint32_t Ty = Words[I + 1];
    if (!PointerResult) {
      auto Struct = Types.find(Ty);
      if (Struct == Types.end() ||
          TypeKinds[Struct->first] != spv::OpTypeStruct ||
          Struct->second.size() != 3 || Struct->second[1] != Struct->second[2])
        return makeError(
            "ModfStruct requires two matching floating-point members");
      Ty = Struct->second[1];
    }
    uint32_t ScalarTy = Ty, Lanes = 1;
    auto Type = Types.find(Ty);
    if (Type == Types.end())
      return makeError("unknown ModfStruct member type");
    if (TypeKinds[Ty] == spv::OpTypeVector) {
      ScalarTy = Type->second[1];
      Lanes = Type->second[2];
    }
    auto Scalar = Types.find(ScalarTy);
    if (Scalar == Types.end() || TypeKinds[ScalarTy] != spv::OpTypeFloat ||
        Scalar->second.size() != 2 ||
        (Scalar->second[1] != 16 && Scalar->second[1] != 32 &&
         Scalar->second[1] != 64) ||
        Lanes < 1 || Lanes > 4)
      return makeError("unsupported ModfStruct floating-point type");
    if (!Cached.contains(Ty)) {
      if (!BoolTy) {
        BoolTy = NextID++;
        Emit(Declarations, spv::OpTypeBool, {BoolTy});
      }
      uint32_t Bool = BoolTy;
      if (Lanes != 1) {
        auto It = llvm::find_if(Types, [&](const auto &KV) {
          return TypeKinds[KV.first] == spv::OpTypeVector &&
                 KV.second[1] == BoolTy && KV.second[2] == Lanes;
        });
        if (BoolVectors.contains(Lanes)) {
          Bool = BoolVectors[Lanes];
        } else if (It != Types.end()) {
          Bool = It->first;
        } else {
          Bool = NextID++;
          Emit(Declarations, spv::OpTypeVector, {Bool, BoolTy, Lanes});
        }
        BoolVectors[Lanes] = Bool;
      }
      uint32_t Width = Scalar->second[1];
      uint32_t UIntScalar = UnsignedTypes.lookup(Width);
      if (!UIntScalar) {
        UIntScalar = NextID++;
        Emit(Declarations, spv::OpTypeInt, {UIntScalar, Width, 0});
        UnsignedTypes[Width] = UIntScalar;
        auto Capability = Width == 16   ? spv::CapabilityInt16
                          : Width == 64 ? spv::CapabilityInt64
                                        : spv::CapabilityShader;
        if (Capabilities.insert(Capability).second)
          Emit(NewCapabilities, spv::OpCapability,
               {static_cast<uint32_t>(Capability)});
      }
      uint32_t UInt = UIntScalar;
      if (Lanes != 1) {
        auto It = llvm::find_if(Types, [&](const auto &KV) {
          return TypeKinds[KV.first] == spv::OpTypeVector &&
                 KV.second[1] == UIntScalar && KV.second[2] == Lanes;
        });
        if (It != Types.end()) {
          UInt = It->first;
        } else {
          UInt = NextID++;
          Emit(Declarations, spv::OpTypeVector, {UInt, UIntScalar, Lanes});
        }
      }
      auto Constant = [&](uint32_t ScalarType, uint32_t ValueType,
                          uint64_t Bits) {
        uint32_t ID = NextID++;
        SmallVector<uint32_t, 4> Args{ScalarType, ID,
                                      static_cast<uint32_t>(Bits)};
        if (Width == 64)
          Args.push_back(Bits >> 32);
        Emit(Declarations, spv::OpConstant, Args);
        if (Lanes == 1)
          return ID;
        uint32_t VectorID = NextID++;
        SmallVector<uint32_t, 6> Elements{ValueType, VectorID};
        Elements.append(Lanes, ID);
        Emit(Declarations, spv::OpConstantComposite, Elements);
        return VectorID;
      };
      uint64_t One = Width == 16   ? 0x3c00ull
                     : Width == 32 ? 0x3f800000ull
                                   : 0x3ff0000000000000ull;
      uint64_t SignMask = uint64_t(1) << (Width - 1);
      uint64_t Infinity = Width == 16   ? 0x7c00ull
                          : Width == 32 ? 0x7f800000ull
                                        : 0x7ff0000000000000ull;
      Cached[Ty] = {Bool,
                    UInt,
                    Constant(UIntScalar, UInt, 0),
                    Constant(UIntScalar, UInt, SignMask),
                    Constant(UIntScalar, UInt, SignMask - 1),
                    Constant(UIntScalar, UInt, Infinity),
                    Constant(ScalarTy, Ty, One)};
    }
    auto C = Cached[Ty];
    uint32_t X = Words[I + 5], Whole = NextID++, Fraction = NextID++,
             Bits = NextID++, Sign = NextID++, Abs = NextID++,
             Infinite = NextID++, FractionBits = NextID++,
             FiniteFraction = NextID++;
    Emit(Body, spv::OpExtInst, {Ty, Whole, Words[I + 3], GLSLstd450Trunc, X});
    Emit(Body, spv::OpFRem, {Ty, Fraction, X, C.One});
    Emit(Body, spv::OpBitcast, {C.UInt, Bits, X});
    Emit(Body, spv::OpBitwiseAnd, {C.UInt, Sign, Bits, C.SignMask});
    Emit(Body, spv::OpBitwiseAnd, {C.UInt, Abs, Bits, C.AbsMask});
    Emit(Body, spv::OpIEqual, {C.Bool, Infinite, Abs, C.Infinity});
    Emit(Body, spv::OpBitcast, {C.UInt, FractionBits, Fraction});
    Emit(Body, spv::OpSelect,
         {C.UInt, FiniteFraction, Infinite, C.Zero, FractionBits});
    // Metal fmod can lose the sign of zero. Restore signs using input bits,
    // including the whole-number result for negative zero and |x| < 1.
    auto CopySign = [&](uint32_t ValueBits) {
      uint32_t Magnitude = NextID++, Signed = NextID++, Result = NextID++;
      Emit(Body, spv::OpBitwiseAnd, {C.UInt, Magnitude, ValueBits, C.AbsMask});
      Emit(Body, spv::OpBitwiseOr, {C.UInt, Signed, Magnitude, Sign});
      Emit(Body, spv::OpBitcast, {Ty, Result, Signed});
      return Result;
    };
    uint32_t ResultFraction = CopySign(FiniteFraction);
    uint32_t WholeBits = NextID++;
    Emit(Body, spv::OpBitcast, {C.UInt, WholeBits, Whole});
    uint32_t ResultWhole = CopySign(WholeBits);
    if (PointerResult) {
      Emit(Body, spv::OpStore, {Words[I + 6], ResultWhole});
      Emit(Body, spv::OpCopyObject, {Ty, Words[I + 2], ResultFraction});
    } else {
      Emit(Body, spv::OpCompositeConstruct,
           {Words[I + 1], Words[I + 2], ResultFraction, ResultWhole});
    }
    I += Count;
  }
  // Generated types/constants must precede all function definitions.
  size_t Function = 0;
  while (Function < Body.size() && (Body[Function] & 0xffff) != spv::OpFunction)
    Function += Body[Function] >> 16;
  Words.resize(5);
  Words[3] = NextID;
  Words.append(NewCapabilities.begin(), NewCapabilities.end());
  Words.append(Body.begin(), Body.begin() + Function);
  Words.append(Declarations.begin(), Declarations.end());
  Words.append(Body.begin() + Function, Body.end());
  return Error::success();
}

static Error preserveBufferMemoryAccess(
    mlir::Operation *Dst, mlir::spirv::MemoryAccessAttr AccessAttr,
    mlir::IntegerAttr Alignment) {
  using MA = mlir::spirv::MemoryAccess;
  auto Access = AccessAttr ? AccessAttr.getValue() : MA::None;
  if (mlir::spirv::bitEnumContainsAny(
          Access, ~(MA::Volatile | MA::Aligned | MA::Nontemporal)))
    return makeError("unsupported SPIR-V buffer memory operands");
  if (mlir::spirv::bitEnumContainsAny(Access, MA::Volatile))
    Dst->setAttr("crucible.volatile", mlir::UnitAttr::get(Dst->getContext()));
  if (Alignment) {
    auto ScalarAlignment = Dst->getAttrOfType<mlir::IntegerAttr>("alignment");
    Dst->setAttr("alignment",
                 mlir::IntegerAttr::get(
                     Alignment.getType(),
                     std::min(Alignment.getInt(), ScalarAlignment.getInt())));
  }
  return Error::success();
}

struct BoundBufferInfo {
  size_t ReflectIndex = 0;
  mlir::Type ElemType;
  uint32_t ElemStride = 4;
  bool IsArray = false;
  uint32_t MatrixStride = 16;
  bool MatrixRowMajor = false;
  uint32_t BaseByteOffset = 0;
  bool IsAddressTable = false;
};

struct BoundCBufferInfo {
  size_t ReflectIndex = 0;
  mlir::spirv::StructType StructTy;
  bool IsArray = false;
};

struct WorkgroupGlobalInfo {
  std::string Name;
  mlir::Type LLVMType;
};

struct BoundImageInfo {
  size_t ReflectIndex = 0;
  size_t SamplerReflectIndex = SIZE_MAX;
  ResourceKind Kind = ResourceKind::Invalid;
  mlir::Type ElemType;
  bool IsUAV = false;
  bool IsArray = false;
};

struct StageIOInfo {
  uint32_t ID = 0;
  bool IsInput = true;
  mlir::Type ValueType;
  uint32_t Columns = 1;
  bool IsMultiRow = false;
};

// Split a DXC SPIR-V stage variable name like "in.var.COLOR2" or
// "out.var.SV_Target0" into (semantic, index).
std::pair<std::string, uint32_t> parseStageVarSemantic(StringRef SymName,
                                                       uint32_t DefaultIdx) {
  StringRef S = SymName;
  bool HasStagePrefix = false;
  if (S.starts_with("in.var.")) {
    S = S.drop_front(7);
    HasStagePrefix = true;
  } else if (S.starts_with("out.var.")) {
    S = S.drop_front(8);
    HasStagePrefix = true;
  }
  size_t End = S.size();
  while (End > 0 && std::isdigit(static_cast<unsigned char>(S[End - 1])))
    --End;
  if (End < S.size() && End > 0) {
    uint32_t Idx = 0;
    if (!S.substr(End).getAsInteger(10, Idx))
      return {S.substr(0, End).str(), Idx};
  }
  return {S.str(), HasStagePrefix ? 0u : DefaultIdx};
}

ResourceKind mapImageKind(mlir::spirv::ImageType ImgTy) {
  bool Arrayed = ImgTy.getArrayedInfo() == mlir::spirv::ImageArrayedInfo::Arrayed;
  switch (ImgTy.getDim()) {
  case mlir::spirv::Dim::Buffer:
    return ResourceKind::TypedBuffer;
  case mlir::spirv::Dim::Dim1D:
    return Arrayed ? ResourceKind::Texture1DArray : ResourceKind::Texture1D;
  case mlir::spirv::Dim::Dim2D:
    return Arrayed ? ResourceKind::Texture2DArray : ResourceKind::Texture2D;
  case mlir::spirv::Dim::Dim3D:
    return ResourceKind::Texture3D;
  case mlir::spirv::Dim::Cube:
    return Arrayed ? ResourceKind::TextureCubeArray : ResourceKind::TextureCube;
  default:
    return ResourceKind::Invalid;
  }
}

uint32_t imageFormatComponents(mlir::spirv::ImageFormat Fmt) {
  switch (Fmt) {
  case mlir::spirv::ImageFormat::R32f:
  case mlir::spirv::ImageFormat::R16f:
  case mlir::spirv::ImageFormat::R32i:
  case mlir::spirv::ImageFormat::R32ui:
  case mlir::spirv::ImageFormat::R16i:
  case mlir::spirv::ImageFormat::R16ui:
  case mlir::spirv::ImageFormat::R8i:
  case mlir::spirv::ImageFormat::R8ui:
  case mlir::spirv::ImageFormat::R8:
  case mlir::spirv::ImageFormat::R16:
  case mlir::spirv::ImageFormat::R8Snorm:
  case mlir::spirv::ImageFormat::R16Snorm:
  case mlir::spirv::ImageFormat::R64i:
  case mlir::spirv::ImageFormat::R64ui:
    return 1;
  case mlir::spirv::ImageFormat::Rg32f:
  case mlir::spirv::ImageFormat::Rg16f:
  case mlir::spirv::ImageFormat::Rg32i:
  case mlir::spirv::ImageFormat::Rg32ui:
  case mlir::spirv::ImageFormat::Rg16i:
  case mlir::spirv::ImageFormat::Rg16ui:
  case mlir::spirv::ImageFormat::Rg8i:
  case mlir::spirv::ImageFormat::Rg8ui:
  case mlir::spirv::ImageFormat::Rg8:
  case mlir::spirv::ImageFormat::Rg16:
  case mlir::spirv::ImageFormat::Rg8Snorm:
  case mlir::spirv::ImageFormat::Rg16Snorm:
    return 2;
  default:
    return 4;
  }
}

mlir::Type convertSpirvDataType(mlir::Type T, mlir::OpBuilder &B) {
  if (auto IT = mlir::dyn_cast<mlir::IntegerType>(T))
    return B.getIntegerType(IT.getWidth());
  if (auto Vec = mlir::dyn_cast<mlir::VectorType>(T))
    return mlir::VectorType::get(Vec.getShape(),
                                 convertSpirvDataType(Vec.getElementType(), B));
  if (auto Mat = mlir::dyn_cast<mlir::spirv::MatrixType>(T))
    return mlir::LLVM::LLVMArrayType::get(
        convertSpirvDataType(Mat.getColumnType(), B), Mat.getNumColumns());
  if (auto Arr = mlir::dyn_cast<mlir::spirv::ArrayType>(T))
    return mlir::LLVM::LLVMArrayType::get(
        convertSpirvDataType(Arr.getElementType(), B), Arr.getNumElements());
  if (auto ST = mlir::dyn_cast<mlir::spirv::StructType>(T)) {
    SmallVector<mlir::Type, 4> Elems;
    for (unsigned I = 0; I < ST.getNumElements(); ++I)
      Elems.push_back(convertSpirvDataType(ST.getElementType(I), B));
    return mlir::LLVM::LLVMStructType::getLiteral(B.getContext(), Elems);
  }
  return T;
}

uint32_t scalarTypeBytes(mlir::Type T) {
  if (auto P = mlir::dyn_cast<mlir::spirv::PointerType>(T);
      P &&
      P.getStorageClass() == mlir::spirv::StorageClass::PhysicalStorageBuffer)
    return 8;
  if (T.isF16() || T.isInteger(16))
    return 2;
  if (T.isF64() || T.isInteger(64))
    return 8;
  return 4;
}

uint32_t spirvTypeBytes(mlir::Type T) {
  if (auto Vec = mlir::dyn_cast<mlir::VectorType>(T))
    return static_cast<uint32_t>(Vec.getNumElements()) *
           scalarTypeBytes(Vec.getElementType());
  if (auto Mat = mlir::dyn_cast<mlir::spirv::MatrixType>(T))
    return Mat.getNumColumns() * 16u;
  if (auto Arr = mlir::dyn_cast<mlir::spirv::ArrayType>(T)) {
    uint32_t Stride = Arr.getArrayStride();
    if (Stride == 0)
      Stride = spirvTypeBytes(Arr.getElementType());
    return Arr.getNumElements() * Stride;
  }
  if (auto ST = mlir::dyn_cast<mlir::spirv::StructType>(T)) {
    unsigned N = ST.getNumElements();
    if (N == 0)
      return 0;
    uint32_t LastOff = ST.getMemberOffset(N - 1);
    return LastOff + spirvTypeBytes(ST.getElementType(N - 1));
  }
  return scalarTypeBytes(T);
}

std::optional<int64_t> getConstSpirvInt(mlir::Value V) {
  if (auto C = V.getDefiningOp<mlir::spirv::ConstantOp>()) {
    if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue()))
      return IA.getInt();
  }
  return std::nullopt;
}

ComponentType toComponentType(mlir::Type T) {
  if (T.isF32())
    return ComponentType::F32;
  if (T.isF16())
    return ComponentType::F16;
  if (T.isF64())
    return ComponentType::F64;
  if (auto IT = mlir::dyn_cast<mlir::IntegerType>(T)) {
    if (IT.getWidth() == 32)
      return IT.isSigned() ? ComponentType::I32 : ComponentType::U32;
    if (IT.getWidth() == 16)
      return IT.isSigned() ? ComponentType::I16 : ComponentType::U16;
    if (IT.getWidth() == 64)
      return IT.isSigned() ? ComponentType::I64 : ComponentType::U64;
    if (IT.getWidth() == 1)
      return ComponentType::I1;
  }
  return ComponentType::Invalid;
}

std::string typeSuffix(mlir::Type T) {
  if (T.isF32())
    return "f32";
  if (T.isF16())
    return "f16";
  if (T.isF64())
    return "f64";
  if (auto IT = mlir::dyn_cast<mlir::IntegerType>(T))
    return "i" + std::to_string(IT.getWidth());
  return "i32";
}

mlir::Type makeSignless(mlir::Type T, mlir::OpBuilder &B) {
  if (auto IT = mlir::dyn_cast<mlir::IntegerType>(T)) {
    if (!IT.isSignless())
      return B.getIntegerType(IT.getWidth());
  }
  if (auto Vec = mlir::dyn_cast<mlir::VectorType>(T)) {
    mlir::Type E = makeSignless(Vec.getElementType(), B);
    if (E != Vec.getElementType())
      return mlir::VectorType::get(Vec.getShape(), E);
  }
  return T;
}

mlir::Value castToSignless(mlir::OpBuilder &B, mlir::Location Loc,
                           mlir::Value V) {
  mlir::Type DstTy = makeSignless(V.getType(), B);
  if (DstTy == V.getType())
    return V;
  return mlir::spirv::BitcastOp::create(B, Loc, DstTy, V);
}

mlir::Value castFromSignless(mlir::OpBuilder &B, mlir::Location Loc,
                             mlir::Value V, mlir::Type OrigTy) {
  if (OrigTy == V.getType())
    return V;
  return mlir::spirv::BitcastOp::create(B, Loc, OrigTy, V);
}

mlir::LLVM::LLVMStructType getHandleType(mlir::MLIRContext *Ctx) {
  auto PtrTy = mlir::LLVM::LLVMPointerType::get(Ctx);
  return mlir::LLVM::LLVMStructType::getIdentified(Ctx, "dx.types.Handle");
}

mlir::LLVM::LLVMStructType getOrInitHandleType(mlir::MLIRContext *Ctx) {
  auto ST = getHandleType(Ctx);
  if (!ST.isInitialized()) {
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(Ctx);
    (void)ST.setBody({PtrTy}, /*isPacked=*/false);
  }
  return ST;
}

mlir::LLVM::LLVMStructType getOrInitResRetType(mlir::MLIRContext *Ctx,
                                               mlir::Type ElemTy) {
  std::string Name = "dx.types.ResRet." + typeSuffix(ElemTy);
  auto ST = mlir::LLVM::LLVMStructType::getIdentified(Ctx, Name);
  if (!ST.isInitialized()) {
    auto I32 = mlir::IntegerType::get(Ctx, 32);
    (void)ST.setBody({ElemTy, ElemTy, ElemTy, ElemTy, I32}, /*isPacked=*/false);
  }
  return ST;
}

mlir::LLVM::LLVMStructType getOrInitCBufRetType(mlir::MLIRContext *Ctx,
                                                mlir::Type ElemTy) {
  std::string Name = "dx.types.CBufRet." + typeSuffix(ElemTy);
  auto ST = mlir::LLVM::LLVMStructType::getIdentified(Ctx, Name);
  if (!ST.isInitialized()) {
    if (ElemTy.isF16() || ElemTy.isInteger(16))
      (void)ST.setBody(
          {ElemTy, ElemTy, ElemTy, ElemTy, ElemTy, ElemTy, ElemTy, ElemTy},
          /*isPacked=*/false);
    else if (ElemTy.isF64() || ElemTy.isInteger(64))
      (void)ST.setBody({ElemTy, ElemTy}, /*isPacked=*/false);
    else
      (void)ST.setBody({ElemTy, ElemTy, ElemTy, ElemTy}, /*isPacked=*/false);
  }
  return ST;
}

mlir::LLVM::LLVMFuncOp getOrCreateDecl(mlir::spirv::ModuleOp SpvMod,
                                       StringRef Name,
                                       mlir::LLVM::LLVMFunctionType FnTy) {
  if (auto Existing = SpvMod.lookupSymbol<mlir::LLVM::LLVMFuncOp>(Name))
    return Existing;
  mlir::OpBuilder B = mlir::OpBuilder::atBlockBegin(&SpvMod.getBodyRegion().front());
  return mlir::LLVM::LLVMFuncOp::create(B, SpvMod.getLoc(), Name, FnTy);
}

mlir::Value emitGPUIdComponent(mlir::OpBuilder &B, mlir::Location Loc,
                               mlir::spirv::BuiltIn BI, uint32_t Comp,
                               mlir::Type ResultTy) {
  mlir::gpu::Dimension Dim = mlir::gpu::Dimension::x;
  if (Comp == 1)
    Dim = mlir::gpu::Dimension::y;
  else if (Comp == 2)
    Dim = mlir::gpu::Dimension::z;

  mlir::Value IdxVal;
  switch (BI) {
  case mlir::spirv::BuiltIn::GlobalInvocationId:
    IdxVal = mlir::gpu::GlobalIdOp::create(B, Loc, Dim);
    break;
  case mlir::spirv::BuiltIn::WorkgroupId:
    IdxVal = mlir::gpu::BlockIdOp::create(B, Loc, Dim);
    break;
  case mlir::spirv::BuiltIn::LocalInvocationId:
    IdxVal = mlir::gpu::ThreadIdOp::create(B, Loc, Dim);
    break;
  default:
    return nullptr;
  }

  mlir::Type SignlessTy = makeSignless(ResultTy, B);
  return mlir::arith::IndexCastUIOp::create(B, Loc, SignlessTy, IdxVal);
}

} // namespace

Expected<LoadedMLIRShader>
loadSPIRV(MemoryBufferRef Buf, mlir::MLIRContext &Ctx,
          StringRef EntryOverride,
          const std::unordered_map<uint32_t, uint64_t> &SpecConstants) {
  if (!isSPIRVBinary(Buf))
    return makeError("input buffer is not a valid SPIR-V binary");
  if (Buf.getBufferSize() % 4 != 0)
    return makeError("SPIR-V binary size ({0}) is not a multiple of 4",
                     Buf.getBufferSize());

  mlir::DialectRegistry Registry;
  Registry.insert<mlir::spirv::SPIRVDialect, mlir::LLVM::LLVMDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect,
                  mlir::cf::ControlFlowDialect, mlir::scf::SCFDialect,
                  mlir::arith::ArithDialect, mlir::gpu::GPUDialect,
                  mlir::math::MathDialect, mlir::ub::UBDialect,
                  crucible::ir::CrucibleDialect>();
  Ctx.appendDialectRegistry(Registry);
  Ctx.loadAllAvailableDialects();

  size_t NumWords = Buf.getBufferSize() / 4;
  SmallVector<uint32_t> Words(NumWords);
  const char *Raw = Buf.getBufferStart();
  for (size_t I = 0; I < NumWords; ++I)
    Words[I] = support::endian::read32le(Raw + I * 4);
  bool RequiresStrictFP = false;
  if (Error E = expandFloatControls2(Words, RequiresStrictFP))
    return std::move(E);
  if (Error E = expandFmaKHR(Words))
    return std::move(E);
  if (Error E = expandModfStruct(Words))
    return std::move(E);

  // Strip SPV_KHR_compute_shader_derivatives metadata (extension/capability/
  // execution-mode) that newer DXC emits for compute quad ops, which MLIR's
  // SPIR-V deserializer does not yet recognize even though it handles
  // GroupNonUniformQuadSwap natively.
  {
    bool HasDerivExt = false;
    auto IsExtName = [](ArrayRef<uint32_t> W, StringRef Name) {
      std::string S;
      S.reserve(W.size() * 4);
      for (uint32_t Word : W) {
        for (int B = 0; B < 4; ++B) {
          char C = static_cast<char>((Word >> (B * 8)) & 0xff);
          if (!C)
            return S == Name;
          S.push_back(C);
        }
      }
      return S == Name;
    };
    for (size_t I = 5; I < Words.size();) {
      uint32_t Count = Words[I] >> 16, Op = Words[I] & 0xffff;
      if (Count == 0 || I + Count > Words.size())
        break;
      if (Op == spv::OpExtension && Count >= 2 &&
          IsExtName(ArrayRef(Words).slice(I + 1, Count - 1),
                    "SPV_KHR_compute_shader_derivatives")) {
        HasDerivExt = true;
        break;
      }
      I += Count;
    }
    if (HasDerivExt) {
      SmallVector<uint32_t> Out(Words.begin(), Words.begin() + 5);
      for (size_t I = 5; I < Words.size();) {
        uint32_t Count = Words[I] >> 16, Op = Words[I] & 0xffff;
        if (Count == 0 || I + Count > Words.size())
          break;
        bool Skip = false;
        if (Op == spv::OpExtension && Count >= 2 &&
            IsExtName(ArrayRef(Words).slice(I + 1, Count - 1),
                      "SPV_KHR_compute_shader_derivatives"))
          Skip = true;
        else if (Op == spv::OpCapability && Count == 2 &&
                 (Words[I + 1] == 5288u || Words[I + 1] == 5350u))
          Skip = true;
        else if (Op == spv::OpExecutionMode && Count >= 3 &&
                 (Words[I + 2] == 5289u || Words[I + 2] == 5290u))
          Skip = true;
        if (!Skip)
          Out.append(Words.begin() + I, Words.begin() + I + Count);
        I += Count;
      }
      Words.assign(Out.begin(), Out.end());
    }
  }

  // MLIR's SPIR-V deserializer drops the value attribute on OpConstantComposite
  // of OpTypeMatrix, producing an invalid null-value spirv.Constant. Rewrite
  // constant matrices into OpCompositeConstruct at the start of each function
  // that references them.
  {
    DenseSet<uint32_t> MatrixTypes;
    struct ConstMatrixDef {
      uint32_t TypeId;
      uint32_t ResultId;
      SmallVector<uint32_t, 4> ColIds;
    };
    DenseMap<uint32_t, ConstMatrixDef> ConstMatrices;
    for (size_t I = 5; I < Words.size();) {
      uint32_t Count = Words[I] >> 16, Op = Words[I] & 0xffff;
      if (Count == 0 || I + Count > Words.size())
        break;
      if (Op == spv::OpTypeMatrix && Count >= 4)
        MatrixTypes.insert(Words[I + 1]);
      else if (Op == spv::OpConstantComposite && Count >= 4 &&
               MatrixTypes.contains(Words[I + 1])) {
        ConstMatrixDef Def;
        Def.TypeId = Words[I + 1];
        Def.ResultId = Words[I + 2];
        for (uint32_t K = 3; K < Count; ++K)
          Def.ColIds.push_back(Words[I + K]);
        ConstMatrices[Def.ResultId] = std::move(Def);
      }
      I += Count;
    }
    if (!ConstMatrices.empty()) {
      SmallVector<uint32_t> Out(Words.begin(), Words.begin() + 5);
      uint32_t NextId = Words[3];
      bool InFunction = false;
      bool InsertedInFn = false;
      DenseMap<uint32_t, uint32_t> FnIdMap;
      for (size_t I = 5; I < Words.size();) {
        uint32_t Count = Words[I] >> 16, Op = Words[I] & 0xffff;
        if (Count == 0 || I + Count > Words.size())
          break;
        if (Op == spv::OpConstantComposite && Count >= 4 &&
            ConstMatrices.contains(Words[I + 2])) {
          I += Count;
          continue;
        }
        if (Op == spv::OpFunction) {
          InFunction = true;
          InsertedInFn = false;
          FnIdMap.clear();
        } else if (Op == spv::OpFunctionEnd) {
          InFunction = false;
          FnIdMap.clear();
        }
        if (InFunction && Op == spv::OpLabel && !InsertedInFn) {
          Out.append(Words.begin() + I, Words.begin() + I + Count);
          InsertedInFn = true;
          for (const auto &KV : ConstMatrices) {
            uint32_t NewId = NextId++;
            FnIdMap[KV.first] = NewId;
            const ConstMatrixDef &Def = KV.second;
            uint32_t CCWords = 3u + static_cast<uint32_t>(Def.ColIds.size());
            Out.push_back((CCWords << 16) | spv::OpCompositeConstruct);
            Out.push_back(Def.TypeId);
            Out.push_back(NewId);
            for (uint32_t ColId : Def.ColIds)
              Out.push_back(ColId);
          }
          I += Count;
          continue;
        }
        if (InFunction && !FnIdMap.empty()) {
          size_t Start = Out.size();
          Out.append(Words.begin() + I, Words.begin() + I + Count);
          // Remap operand IDs (skip opcode word and any result type/id words
          // only where not a reference; since ConstMatrices IDs were global
          // constant IDs, they only ever appear as operand references inside
          // function bodies).
          for (size_t K = 1; K < Count; ++K) {
            auto It = FnIdMap.find(Out[Start + K]);
            if (It != FnIdMap.end())
              Out[Start + K] = It->second;
          }
        } else {
          Out.append(Words.begin() + I, Words.begin() + I + Count);
        }
        I += Count;
      }
      Out[3] = NextId;
      Words.assign(Out.begin(), Out.end());
    }
  }

  // Rewrite SPIR-V derivative opcodes (OpDPdx..OpFwidthCoarse, 207..215),
  // OpDemoteToHelperInvocation (5380), GLSL.std.450 FaceForward (70) and
  // Refract (72), and strip NonUniform decoration (5300) — all of which MLIR's
  // SPIR-V deserializer does not yet recognize — into OpFunctionCall calls to
  // dummy helper functions named "__crucible_spv_*".
  {
    constexpr uint32_t kOpDemoteToHelperInvocation = 5380u;
    constexpr uint32_t kCapDemoteToHelperInvocation = 5379u;
    constexpr uint32_t kDecNonUniform = 5300u;
    constexpr uint32_t kGlslFaceForward = 70u;
    constexpr uint32_t kGlslRefract = 72u;
    // Synthetic op tags in HelperFnInfo::Op for GLSL.std.450 rewrites.
    constexpr uint32_t kTagFaceForward = 0x10070u;
    constexpr uint32_t kTagRefract = 0x10072u;
    auto isDerivOp = [](uint32_t Op) { return Op >= 207u && Op <= 215u; };
    auto isExtName = [](ArrayRef<uint32_t> W, StringRef Name) {
      std::string S;
      S.reserve(W.size() * 4);
      for (uint32_t Word : W) {
        for (int B = 0; B < 4; ++B) {
          char C = static_cast<char>((Word >> (B * 8)) & 0xff);
          if (!C)
            return S == Name;
          S.push_back(C);
        }
      }
      return S == Name;
    };

    uint32_t VoidTyId = 0;
    uint32_t GlslExtId = 0;
    bool HasDemote = false;
    bool HasDemoteExtOrCap = false;
    bool HasNonUniformDec = false;
    DenseMap<uint32_t, uint32_t> VecElemType;
    // Key: (Op << 32) | ResultTypeId -> HelperFnInfo
    struct HelperFnInfo {
      uint32_t Op = 0;
      uint32_t TypeId = 0;
      uint32_t ScalarTypeId = 0; // For Refract's eta parameter
      uint32_t FnTyId = 0;
      uint32_t FnId = 0;
      uint32_t ParamIds[3] = {0, 0, 0};
      uint32_t NumParams = 1;
      uint32_t LabelId = 0;
      std::string Name;
    };
    DenseMap<uint64_t, HelperFnInfo> Helpers;
    SmallVector<uint64_t, 8> HelperOrder;

    for (size_t I = 5; I < Words.size();) {
      uint32_t Count = Words[I] >> 16, Op = Words[I] & 0xffff;
      if (Count == 0 || I + Count > Words.size())
        break;
      if (Op == spv::OpExtInstImport && Count >= 3 &&
          isExtName(ArrayRef(Words).slice(I + 2, Count - 2), "GLSL.std.450")) {
        GlslExtId = Words[I + 1];
      } else if (Op == spv::OpTypeVoid && Count == 2) {
        VoidTyId = Words[I + 1];
      } else if (Op == spv::OpTypeVector && Count == 4) {
        VecElemType[Words[I + 1]] = Words[I + 2];
      } else if (Op == spv::OpExtension && Count >= 2 &&
                 isExtName(ArrayRef(Words).slice(I + 1, Count - 1),
                           "SPV_EXT_demote_to_helper_invocation")) {
        HasDemoteExtOrCap = true;
      } else if (Op == spv::OpCapability && Count == 2 &&
                 Words[I + 1] == kCapDemoteToHelperInvocation) {
        HasDemoteExtOrCap = true;
      } else if ((Op == spv::OpDecorate && Count >= 3 &&
                  Words[I + 2] == kDecNonUniform) ||
                 (Op == spv::OpMemberDecorate && Count >= 4 &&
                  Words[I + 3] == kDecNonUniform)) {
        HasNonUniformDec = true;
      } else if (isDerivOp(Op) && Count == 4) {
        uint32_t ResTy = Words[I + 1];
        uint64_t Key = (static_cast<uint64_t>(Op) << 32) | ResTy;
        if (!Helpers.contains(Key)) {
          HelperFnInfo Info;
          Info.Op = Op;
          Info.TypeId = ResTy;
          Info.NumParams = 1;
          Helpers[Key] = Info;
          HelperOrder.push_back(Key);
        }
      } else if (Op == spv::OpExtInst && Count == 8 && GlslExtId != 0 &&
                 Words[I + 3] == GlslExtId &&
                 (Words[I + 4] == kGlslFaceForward ||
                  Words[I + 4] == kGlslRefract)) {
        uint32_t ResTy = Words[I + 1];
        uint32_t Tag = (Words[I + 4] == kGlslFaceForward) ? kTagFaceForward
                                                          : kTagRefract;
        uint64_t Key = (static_cast<uint64_t>(Tag) << 32) | ResTy;
        if (!Helpers.contains(Key)) {
          HelperFnInfo Info;
          Info.Op = Tag;
          Info.TypeId = ResTy;
          auto ElemIt = VecElemType.find(ResTy);
          Info.ScalarTypeId = (ElemIt != VecElemType.end()) ? ElemIt->second : ResTy;
          Info.NumParams = 3;
          Helpers[Key] = Info;
          HelperOrder.push_back(Key);
        }
      } else if (Op == kOpDemoteToHelperInvocation && Count == 1) {
        HasDemote = true;
      }
      I += Count;
    }

    if (!Helpers.empty() || HasDemote || HasDemoteExtOrCap || HasNonUniformDec) {
      uint32_t NextId = Words[3];
      bool NeedVoidTy = HasDemote && (VoidTyId == 0);
      if (NeedVoidTy)
        VoidTyId = NextId++;

      auto helperPrefix = [](uint32_t Op) -> const char * {
        switch (Op) {
        case 207: return "__crucible_spv_dpdx_";
        case 208: return "__crucible_spv_dpdy_";
        case 209: return "__crucible_spv_fwidth_";
        case 210: return "__crucible_spv_dpdx_fine_";
        case 211: return "__crucible_spv_dpdy_fine_";
        case 212: return "__crucible_spv_fwidth_fine_";
        case 213: return "__crucible_spv_dpdx_coarse_";
        case 214: return "__crucible_spv_dpdy_coarse_";
        case 215: return "__crucible_spv_fwidth_coarse_";
        case kTagFaceForward: return "__crucible_spv_faceforward_";
        case kTagRefract: return "__crucible_spv_refract_";
        default: return "__crucible_spv_deriv_";
        }
      };

      for (uint64_t Key : HelperOrder) {
        HelperFnInfo &Info = Helpers[Key];
        Info.FnTyId = NextId++;
        Info.FnId = NextId++;
        for (uint32_t P = 0; P < Info.NumParams; ++P)
          Info.ParamIds[P] = NextId++;
        Info.LabelId = NextId++;
        Info.Name = std::string(helperPrefix(Info.Op)) + std::to_string(Info.TypeId);
      }

      HelperFnInfo DemoteInfo;
      if (HasDemote) {
        DemoteInfo.Op = kOpDemoteToHelperInvocation;
        DemoteInfo.TypeId = VoidTyId;
        DemoteInfo.FnTyId = NextId++;
        DemoteInfo.FnId = NextId++;
        DemoteInfo.LabelId = NextId++;
        DemoteInfo.Name = "__crucible_spv_demote";
      }

      auto emitOpName = [](SmallVectorImpl<uint32_t> &Out, uint32_t TargetId,
                           StringRef Name) {
        size_t StrWords = (Name.size() + 4) / 4;
        uint32_t Count = static_cast<uint32_t>(2 + StrWords);
        Out.push_back((Count << 16) | spv::OpName);
        Out.push_back(TargetId);
        for (size_t W = 0; W < StrWords; ++W) {
          uint32_t Word = 0;
          for (size_t B = 0; B < 4; ++B) {
            size_t Idx = W * 4 + B;
            if (Idx < Name.size())
              Word |= static_cast<uint32_t>(static_cast<uint8_t>(Name[Idx]))
                      << (B * 8);
          }
          Out.push_back(Word);
        }
      };

      auto isBeforeNames = [](uint32_t Op) {
        return Op == spv::OpCapability || Op == spv::OpExtension ||
               Op == spv::OpExtInstImport || Op == spv::OpMemoryModel ||
               Op == spv::OpEntryPoint || Op == spv::OpExecutionMode ||
               Op == spv::OpExecutionModeId || Op == spv::OpString ||
               Op == spv::OpSourceExtension || Op == spv::OpSource ||
               Op == spv::OpSourceContinued;
      };

      SmallVector<uint32_t> Out(Words.begin(), Words.begin() + 5);
      bool EmittedNames = false;
      bool EmittedHelpers = false;
      for (size_t I = 5; I < Words.size();) {
        uint32_t Count = Words[I] >> 16, Op = Words[I] & 0xffff;
        if (Count == 0 || I + Count > Words.size())
          break;
        // Strip demote extension/capability and NonUniform decorations
        if ((Op == spv::OpExtension && Count >= 2 &&
             isExtName(ArrayRef(Words).slice(I + 1, Count - 1),
                       "SPV_EXT_demote_to_helper_invocation")) ||
            (Op == spv::OpCapability && Count == 2 &&
             Words[I + 1] == kCapDemoteToHelperInvocation) ||
            (Op == spv::OpDecorate && Count >= 3 &&
             Words[I + 2] == kDecNonUniform) ||
            (Op == spv::OpMemberDecorate && Count >= 4 &&
             Words[I + 3] == kDecNonUniform)) {
          I += Count;
          continue;
        }
        // Emit OpName declarations right after header/source section
        if (!EmittedNames && !isBeforeNames(Op)) {
          EmittedNames = true;
          for (uint64_t Key : HelperOrder)
            emitOpName(Out, Helpers[Key].FnId, Helpers[Key].Name);
          if (HasDemote)
            emitOpName(Out, DemoteInfo.FnId, DemoteInfo.Name);
        }
        // Emit helper OpTypeFunction + dummy helper function definitions
        // right before the first OpFunction.
        if (!EmittedHelpers && Op == spv::OpFunction) {
          EmittedHelpers = true;
          if (NeedVoidTy) {
            Out.push_back((2u << 16) | spv::OpTypeVoid);
            Out.push_back(VoidTyId);
          }
          for (uint64_t Key : HelperOrder) {
            const HelperFnInfo &Info = Helpers[Key];
            uint32_t FTWords = 3u + Info.NumParams;
            Out.push_back((FTWords << 16) | spv::OpTypeFunction);
            Out.push_back(Info.FnTyId);
            Out.push_back(Info.TypeId);
            for (uint32_t P = 0; P < Info.NumParams; ++P) {
              uint32_t PTy = (Info.Op == kTagRefract && P == 2)
                                 ? Info.ScalarTypeId
                                 : Info.TypeId;
              Out.push_back(PTy);
            }
          }
          if (HasDemote) {
            Out.push_back((3u << 16) | spv::OpTypeFunction);
            Out.push_back(DemoteInfo.FnTyId);
            Out.push_back(VoidTyId);
          }
          for (uint64_t Key : HelperOrder) {
            const HelperFnInfo &Info = Helpers[Key];
            Out.push_back((5u << 16) | spv::OpFunction);
            Out.push_back(Info.TypeId);
            Out.push_back(Info.FnId);
            Out.push_back(0u);
            Out.push_back(Info.FnTyId);
            for (uint32_t P = 0; P < Info.NumParams; ++P) {
              uint32_t PTy = (Info.Op == kTagRefract && P == 2)
                                 ? Info.ScalarTypeId
                                 : Info.TypeId;
              Out.push_back((3u << 16) | spv::OpFunctionParameter);
              Out.push_back(PTy);
              Out.push_back(Info.ParamIds[P]);
            }
            Out.push_back((2u << 16) | spv::OpLabel);
            Out.push_back(Info.LabelId);
            Out.push_back((2u << 16) | spv::OpReturnValue);
            Out.push_back(Info.ParamIds[0]);
            Out.push_back((1u << 16) | spv::OpFunctionEnd);
          }
          if (HasDemote) {
            Out.push_back((5u << 16) | spv::OpFunction);
            Out.push_back(VoidTyId);
            Out.push_back(DemoteInfo.FnId);
            Out.push_back(0u);
            Out.push_back(DemoteInfo.FnTyId);
            Out.push_back((2u << 16) | spv::OpLabel);
            Out.push_back(DemoteInfo.LabelId);
            Out.push_back((1u << 16) | spv::OpReturn);
            Out.push_back((1u << 16) | spv::OpFunctionEnd);
          }
        }
        // Rewrite derivative instructions -> OpFunctionCall
        if (isDerivOp(Op) && Count == 4) {
          uint32_t ResTy = Words[I + 1];
          uint32_t ResId = Words[I + 2];
          uint32_t ValId = Words[I + 3];
          uint64_t Key = (static_cast<uint64_t>(Op) << 32) | ResTy;
          const HelperFnInfo &Info = Helpers[Key];
          Out.push_back((5u << 16) | spv::OpFunctionCall);
          Out.push_back(ResTy);
          Out.push_back(ResId);
          Out.push_back(Info.FnId);
          Out.push_back(ValId);
          I += Count;
          continue;
        }
        // Rewrite GLSL.std.450 FaceForward / Refract -> OpFunctionCall
        if (Op == spv::OpExtInst && Count == 8 && GlslExtId != 0 &&
            Words[I + 3] == GlslExtId &&
            (Words[I + 4] == kGlslFaceForward ||
             Words[I + 4] == kGlslRefract)) {
          uint32_t ResTy = Words[I + 1];
          uint32_t ResId = Words[I + 2];
          uint32_t Tag = (Words[I + 4] == kGlslFaceForward) ? kTagFaceForward
                                                            : kTagRefract;
          uint64_t Key = (static_cast<uint64_t>(Tag) << 32) | ResTy;
          const HelperFnInfo &Info = Helpers[Key];
          Out.push_back((7u << 16) | spv::OpFunctionCall);
          Out.push_back(ResTy);
          Out.push_back(ResId);
          Out.push_back(Info.FnId);
          Out.push_back(Words[I + 5]);
          Out.push_back(Words[I + 6]);
          Out.push_back(Words[I + 7]);
          I += Count;
          continue;
        }
        // Rewrite OpDemoteToHelperInvocation -> OpFunctionCall __crucible_spv_demote
        if (Op == kOpDemoteToHelperInvocation && Count == 1) {
          uint32_t CallResId = NextId++;
          Out.push_back((4u << 16) | spv::OpFunctionCall);
          Out.push_back(VoidTyId);
          Out.push_back(CallResId);
          Out.push_back(DemoteInfo.FnId);
          I += Count;
          continue;
        }
        Out.append(Words.begin() + I, Words.begin() + I + Count);
        I += Count;
      }
      Out[3] = NextId;
      Words.assign(Out.begin(), Out.end());
    }
  }

  // Keep CFG as plain spirv.Branch / spirv.BranchConditional so
  // ConvertSPIRVToLLVM + LiftControlFlowToSCF handle all control flow.
  mlir::spirv::DeserializationOptions DeserOpts;
  DeserOpts.enableControlFlowStructurization = false;
  auto SpvMod = mlir::spirv::deserialize(Words, &Ctx, DeserOpts);
  if (!SpvMod)
    return makeError("MLIR SPIR-V deserialization failed");

  // Wrap inside a top-level builtin.module so ConvertSPIRVToLLVM can run on it.
  mlir::OwningOpRef<mlir::ModuleOp> TopMod =
      mlir::ModuleOp::create(SpvMod->getLoc());
  TopMod->getBody()->push_back(SpvMod.release());
  auto SpvModule = *TopMod->getOps<mlir::spirv::ModuleOp>().begin();

  // Lower synthetic __crucible_spv_* calls (from pre-deserialization SPIR-V
  // derivative/demote rewrites) BEFORE helper inlining so their dummy bodies
  // are never inlined into callers.
  {
    auto I1TyLocal = mlir::IntegerType::get(&Ctx, 1);
    SmallVector<mlir::spirv::FunctionCallOp, 8> SynthCalls;
    SpvModule.walk([&](mlir::spirv::FunctionCallOp FC) {
      if (FC.getCallee().starts_with("__crucible_spv_"))
        SynthCalls.push_back(FC);
    });
    for (auto FC : SynthCalls) {
      mlir::OpBuilder B(FC);
      mlir::Location Loc = FC.getLoc();
      StringRef Callee = FC.getCallee();
      if (Callee == "__crucible_spv_demote") {
        auto TrueC = mlir::LLVM::ConstantOp::create(B, Loc, I1TyLocal, true);
        ir::DiscardOp::create(B, Loc, TrueC);
        FC.erase();
        continue;
      }
      if (FC.getNumOperands() == 1 && FC.getNumResults() == 1) {
        mlir::Value Arg = FC.getOperand(0);
        mlir::Type ResTy = FC.getResult(0).getType();
        ir::DerivQuality Qual =
            Callee.contains("_fine_") ? ir::DerivQuality::Fine
                                      : ir::DerivQuality::Coarse;
        mlir::Value Res;
        if (Callee.starts_with("__crucible_spv_fwidth")) {
          auto Dx = ir::DerivOp::create(
              B, Loc, ResTy, Arg,
              ir::DerivAxisAttr::get(&Ctx, ir::DerivAxis::X),
              ir::DerivQualityAttr::get(&Ctx, Qual));
          auto Dy = ir::DerivOp::create(
              B, Loc, ResTy, Arg,
              ir::DerivAxisAttr::get(&Ctx, ir::DerivAxis::Y),
              ir::DerivQualityAttr::get(&Ctx, Qual));
          auto AbsDx = mlir::LLVM::FAbsOp::create(B, Loc, ResTy, Dx.getResult());
          auto AbsDy = mlir::LLVM::FAbsOp::create(B, Loc, ResTy, Dy.getResult());
          Res = mlir::LLVM::FAddOp::create(B, Loc, ResTy, AbsDx, AbsDy);
        } else {
          ir::DerivAxis Axis = Callee.starts_with("__crucible_spv_dpdy")
                                   ? ir::DerivAxis::Y
                                   : ir::DerivAxis::X;
          Res = ir::DerivOp::create(B, Loc, ResTy, Arg,
                                    ir::DerivAxisAttr::get(&Ctx, Axis),
                                    ir::DerivQualityAttr::get(&Ctx, Qual))
                    .getResult();
        }
        FC.getResult(0).replaceAllUsesWith(Res);
        FC.erase();
        continue;
      }
      if (FC.getNumOperands() == 3 && FC.getNumResults() == 1 &&
          (Callee.starts_with("__crucible_spv_faceforward_") ||
           Callee.starts_with("__crucible_spv_refract_"))) {
        mlir::Type ResTy = FC.getResult(0).getType();
        auto VecTy = mlir::dyn_cast<mlir::VectorType>(ResTy);
        auto ElemTy = mlir::cast<mlir::FloatType>(
            VecTy ? VecTy.getElementType() : ResTy);
        auto I32TyLocal = mlir::IntegerType::get(&Ctx, 32);

        auto buildDot = [&](mlir::Value A, mlir::Value C) -> mlir::Value {
          if (!VecTy)
            return mlir::LLVM::FMulOp::create(B, Loc, ElemTy, A, C);
          unsigned N = static_cast<unsigned>(VecTy.getNumElements());
          mlir::Value Acc;
          for (unsigned K = 0; K < N; ++K) {
            auto IdxC = mlir::LLVM::ConstantOp::create(
                B, Loc, I32TyLocal, static_cast<int32_t>(K));
            mlir::Value AK = mlir::LLVM::ExtractElementOp::create(B, Loc, A, IdxC);
            mlir::Value CK = mlir::LLVM::ExtractElementOp::create(B, Loc, C, IdxC);
            mlir::Value Prod = mlir::LLVM::FMulOp::create(B, Loc, ElemTy, AK, CK);
            Acc = Acc ? mlir::Value(mlir::LLVM::FAddOp::create(B, Loc, ElemTy, Acc, Prod))
                      : Prod;
          }
          return Acc;
        };
        auto splatScalar = [&](mlir::Value S) -> mlir::Value {
          if (!VecTy)
            return S;
          unsigned N = static_cast<unsigned>(VecTy.getNumElements());
          mlir::Value V = mlir::LLVM::PoisonOp::create(B, Loc, VecTy);
          for (unsigned K = 0; K < N; ++K) {
            auto IdxC = mlir::LLVM::ConstantOp::create(
                B, Loc, I32TyLocal, static_cast<int32_t>(K));
            V = mlir::LLVM::InsertElementOp::create(B, Loc, V, S, IdxC);
          }
          return V;
        };
        auto fConst = [&](double Val) -> mlir::Value {
          return mlir::LLVM::ConstantOp::create(
              B, Loc, ElemTy, B.getFloatAttr(ElemTy, Val));
        };

        mlir::Value ZeroS = fConst(0.0);
        mlir::Value Res;
        if (Callee.starts_with("__crucible_spv_faceforward_")) {
          // faceforward(N, I, Nref) = dot(Nref, I) < 0 ? N : -N
          mlir::Value N = FC.getOperand(0);
          mlir::Value I = FC.getOperand(1);
          mlir::Value Nref = FC.getOperand(2);
          mlir::Value D = buildDot(Nref, I);
          mlir::Value Lt0 = mlir::LLVM::FCmpOp::create(
              B, Loc, mlir::LLVM::FCmpPredicate::olt, D, ZeroS);
          mlir::Value NegN = mlir::LLVM::FNegOp::create(B, Loc, ResTy, N);
          Res = mlir::LLVM::SelectOp::create(B, Loc, ResTy, Lt0, N, NegN);
        } else {
          // refract(I, N, eta):
          //   d = dot(N, I)
          //   k = 1.0 - eta * eta * (1.0 - d * d)
          //   res = (k < 0.0) ? 0.0 : eta * I - (eta * d + sqrt(max(k, 0.0))) * N
          mlir::Value I = FC.getOperand(0);
          mlir::Value N = FC.getOperand(1);
          mlir::Value Eta = FC.getOperand(2);
          mlir::Value OneS = fConst(1.0);
          mlir::Value D = buildDot(N, I);
          mlir::Value DD = mlir::LLVM::FMulOp::create(B, Loc, ElemTy, D, D);
          mlir::Value OneMinusDD = mlir::LLVM::FSubOp::create(B, Loc, ElemTy, OneS, DD);
          mlir::Value Eta2 = mlir::LLVM::FMulOp::create(B, Loc, ElemTy, Eta, Eta);
          mlir::Value Term = mlir::LLVM::FMulOp::create(B, Loc, ElemTy, Eta2, OneMinusDD);
          mlir::Value K = mlir::LLVM::FSubOp::create(B, Loc, ElemTy, OneS, Term);
          mlir::Value Klt0 = mlir::LLVM::FCmpOp::create(
              B, Loc, mlir::LLVM::FCmpPredicate::olt, K, ZeroS);
          mlir::Value SafeK = mlir::LLVM::SelectOp::create(
              B, Loc, ElemTy, Klt0, ZeroS, K);
          mlir::Value SqrtK = mlir::LLVM::SqrtOp::create(B, Loc, ElemTy, SafeK);
          mlir::Value EtaD = mlir::LLVM::FMulOp::create(B, Loc, ElemTy, Eta, D);
          mlir::Value ScaleN = mlir::LLVM::FAddOp::create(B, Loc, ElemTy, EtaD, SqrtK);
          mlir::Value EtaI = mlir::LLVM::FMulOp::create(
              B, Loc, ResTy, splatScalar(Eta), I);
          mlir::Value ScaledN = mlir::LLVM::FMulOp::create(
              B, Loc, ResTy, splatScalar(ScaleN), N);
          mlir::Value Diff = mlir::LLVM::FSubOp::create(B, Loc, ResTy, EtaI, ScaledN);
          Res = mlir::LLVM::SelectOp::create(
              B, Loc, ResTy, Klt0, splatScalar(ZeroS), Diff);
        }
        FC.getResult(0).replaceAllUsesWith(Res);
        FC.erase();
      }
    }
    SmallVector<mlir::spirv::FuncOp, 8> SynthFuncs;
    SpvModule.walk([&](mlir::spirv::FuncOp F) {
      if (F.getSymName().starts_with("__crucible_spv_"))
        SynthFuncs.push_back(F);
    });
    for (auto F : SynthFuncs)
      F.erase();
  }

  // Shader ABI rewriting below operates on entry bodies, including helpers.
  llvm::StringSet<> EntryNames;
  for (auto EP : SpvModule.getOps<mlir::spirv::EntryPointOp>())
    EntryNames.insert(EP.getFn());
  for (auto Fn : SpvModule.getOps<mlir::spirv::FuncOp>())
    if (!EntryNames.contains(Fn.getSymName()))
      mlir::SymbolTable::setSymbolVisibility(
          Fn, mlir::SymbolTable::Visibility::Private);
  mlir::PassManager InlinePM(&Ctx);
  // Narrow dot ops have an upstream verifier restriction removed by our
  // lowering below. Verify the fully lowered module, not this intermediate IR.
  InlinePM.enableVerifier(false);
  InlinePM.addNestedPass<mlir::spirv::ModuleOp>(mlir::createInlinerPass());
  InlinePM.addNestedPass<mlir::spirv::ModuleOp>(mlir::createSymbolDCEPass());
  if (mlir::failed(InlinePM.run(*TopMod)))
    return makeError("SPIR-V helper inlining failed");

  // 1. Extract entry point & execution mode into Reflection.
  Reflection Reflect;
  Reflect.RequiresStrictFP = RequiresStrictFP;
  mlir::spirv::EntryPointOp SelectedEP;
  for (auto EP : SpvModule.getOps<mlir::spirv::EntryPointOp>()) {
    if (EntryOverride.empty() || EP.getFn() == EntryOverride) {
      SelectedEP = EP;
      break;
    }
  }
  if (!SelectedEP)
    return makeError("no matching SPIR-V entry point found");

  Reflect.EntryName = SelectedEP.getFn().str();
  switch (SelectedEP.getExecutionModel()) {
  case mlir::spirv::ExecutionModel::GLCompute:
    Reflect.Stage = ShaderStage::Compute;
    break;
  case mlir::spirv::ExecutionModel::Vertex:
    Reflect.Stage = ShaderStage::Vertex;
    break;
  case mlir::spirv::ExecutionModel::Fragment:
    Reflect.Stage = ShaderStage::Pixel;
    break;
  default:
    return makeError("unsupported SPIR-V execution model");
  }

  for (auto EM : SpvModule.getOps<mlir::spirv::ExecutionModeOp>()) {
    if (EM.getFn() != Reflect.EntryName)
      continue;
    if (EM.getExecutionMode() == mlir::spirv::ExecutionMode::LocalSize) {
      auto Vals = EM.getValues();
      if (Vals.size() == 3) {
        for (unsigned I = 0; I < 3; ++I)
          Reflect.NumThreads[I] =
              static_cast<uint32_t>(mlir::cast<mlir::IntegerAttr>(Vals[I]).getInt());
      }
    }
  }

  // 2. Inspect spirv.GlobalVariables for built-ins, workgroup memory, stage
  // I/O, textures/samplers, and descriptor bindings.
  StringMap<mlir::spirv::BuiltIn> BuiltInGlobals;
  StringMap<BoundBufferInfo> BufferGlobals;
  StringMap<BoundCBufferInfo> CBufferGlobals;
  StringMap<WorkgroupGlobalInfo> WorkgroupGlobals;
  StringMap<BoundImageInfo> ImageGlobals;
  StringMap<size_t> SamplerGlobals;
  StringMap<StageIOInfo> StageIOGlobals;
  SmallVector<mlir::spirv::GlobalVariableOp, 8> GlobalsToErase;
  uint32_t NextSRVId = 0, NextUAVId = 0, NextCBVId = 0, NextSamplerId = 0;
  uint32_t NextInputID = 0, NextOutputID = 0;
  bool IsGraphicsStage = Reflect.Stage == ShaderStage::Vertex ||
                         Reflect.Stage == ShaderStage::Pixel;

  for (auto GV : SpvModule.getOps<mlir::spirv::GlobalVariableOp>()) {
    auto PtrTy = mlir::dyn_cast<mlir::spirv::PointerType>(GV.getType());
    if (!PtrTy)
      continue;
    auto SC = PtrTy.getStorageClass();

    if (auto BIName = GV.getBuiltIn()) {
      auto BI = mlir::spirv::symbolizeBuiltIn(*BIName);
      if (!BI)
        return makeError("unknown SPIR-V built_in '{0}'", *BIName);

      // Graphics stage built-ins map to StageParameters (Input / Output).
      if (IsGraphicsStage &&
          (SC == mlir::spirv::StorageClass::Input ||
           SC == mlir::spirv::StorageClass::Output)) {
        bool IsInput = SC == mlir::spirv::StorageClass::Input;
        mlir::Type ValTy = PtrTy.getPointeeType();
        mlir::Type ElemTy = ValTy;
        uint32_t Cols = 1;
        if (auto Vec = mlir::dyn_cast<mlir::VectorType>(ValTy)) {
          ElemTy = Vec.getElementType();
          Cols = static_cast<uint32_t>(Vec.getNumElements());
        } else if (auto Arr = mlir::dyn_cast<mlir::spirv::ArrayType>(ValTy)) {
          ElemTy = Arr.getElementType();
          Cols = static_cast<uint32_t>(Arr.getNumElements());
        }
        StageParameter SP{};
        SP.ID = IsInput ? NextInputID++ : NextOutputID++;
        SP.Type = toComponentType(ElemTy);
        SP.Rows = 1;
        SP.Columns = std::min(Cols, 4u);
        SP.Interpolation = 2; // Linear
        switch (*BI) {
        case mlir::spirv::BuiltIn::Position:
        case mlir::spirv::BuiltIn::FragCoord:
          SP.Semantic = "SV_Position";
          SP.Kind = 3; // Position
          break;
        case mlir::spirv::BuiltIn::ClipDistance:
          SP.Semantic = "SV_ClipDistance";
          SP.Kind = 6; // ClipDistance
          break;
        case mlir::spirv::BuiltIn::CullDistance:
          SP.Semantic = "SV_CullDistance";
          SP.Kind = 7; // CullDistance
          break;
        case mlir::spirv::BuiltIn::VertexIndex:
          SP.Semantic = "SV_VertexID";
          SP.Kind = 1;
          break;
        case mlir::spirv::BuiltIn::InstanceIndex:
          SP.Semantic = "SV_InstanceID";
          SP.Kind = 2;
          break;
        case mlir::spirv::BuiltIn::PrimitiveId:
          SP.Semantic = "SV_PrimitiveID";
          SP.Kind = 7;
          break;
        case mlir::spirv::BuiltIn::FrontFacing:
          SP.Semantic = "SV_IsFrontFace";
          SP.Kind = 13;
          SP.Type = ComponentType::U32;
          break;
        case mlir::spirv::BuiltIn::FragDepth:
          SP.Semantic = "SV_Depth";
          SP.Kind = 17;
          break;
        default:
          BuiltInGlobals[GV.getSymName()] = *BI;
          GlobalsToErase.push_back(GV);
          continue;
        }
        StageIOInfo IO{SP.ID, IsInput, ValTy, Cols};
        StageIOGlobals[GV.getSymName()] = IO;
        if (IsInput)
          Reflect.Inputs.push_back(std::move(SP));
        else
          Reflect.Outputs.push_back(std::move(SP));
        GlobalsToErase.push_back(GV);
        continue;
      }

      BuiltInGlobals[GV.getSymName()] = *BI;
      GlobalsToErase.push_back(GV);
      continue;
    }

    // Graphics stage user varyings (Input / Output with location attribute).
    if (IsGraphicsStage &&
        (SC == mlir::spirv::StorageClass::Input ||
         SC == mlir::spirv::StorageClass::Output)) {
      bool IsInput = SC == mlir::spirv::StorageClass::Input;
      uint32_t LocAttr = 0;
      if (auto LA = GV->getAttrOfType<mlir::IntegerAttr>("location"))
        LocAttr = static_cast<uint32_t>(LA.getInt());
      mlir::Type ValTy = PtrTy.getPointeeType();
      mlir::Type RowTy = ValTy;
      uint32_t Rows = 1;
      if (auto Arr = mlir::dyn_cast<mlir::spirv::ArrayType>(ValTy)) {
        Rows = static_cast<uint32_t>(Arr.getNumElements());
        RowTy = Arr.getElementType();
      } else if (auto Mat = mlir::dyn_cast<mlir::spirv::MatrixType>(ValTy)) {
        Rows = static_cast<uint32_t>(Mat.getNumColumns());
        RowTy = Mat.getColumnType();
      }
      mlir::Type ElemTy = RowTy;
      uint32_t Cols = 1;
      if (auto Vec = mlir::dyn_cast<mlir::VectorType>(RowTy)) {
        ElemTy = Vec.getElementType();
        Cols = static_cast<uint32_t>(Vec.getNumElements());
      }
      auto [Sem, SemIdx] = parseStageVarSemantic(GV.getSymName(), LocAttr);
      StageParameter SP{};
      SP.ID = IsInput ? NextInputID++ : NextOutputID++;
      SP.Semantic = Sem;
      SP.SemanticIndex = SemIdx;
      SP.Type = toComponentType(ElemTy);
      SP.Rows = Rows;
      SP.Columns = Cols;
      SP.Location = LocAttr;
      SP.Interpolation = 2; // Linear
      for (uint32_t R = 0; R < Rows; ++R)
        SP.RowSemanticIndices.push_back(SemIdx + R);
      if (!IsInput && Reflect.Stage == ShaderStage::Pixel) {
        SP.Semantic = "SV_Target";
        SP.SemanticIndex = LocAttr;
        SP.Kind = 16; // Target
      } else {
        SP.Kind = 0; // Arbitrary
      }
      StageIOInfo IO{SP.ID, IsInput, ValTy, Cols, Rows > 1};
      StageIOGlobals[GV.getSymName()] = IO;
      if (IsInput)
        Reflect.Inputs.push_back(std::move(SP));
      else
        Reflect.Outputs.push_back(std::move(SP));
      GlobalsToErase.push_back(GV);
      continue;
    }

    if (SC == mlir::spirv::StorageClass::Workgroup) {
      mlir::OpBuilder B(&Ctx);
      WorkgroupGlobalInfo Info;
      Info.Name = GV.getSymName().str();
      Info.LLVMType = convertSpirvDataType(PtrTy.getPointeeType(), B);
      WorkgroupGlobals[GV.getSymName()] = std::move(Info);
      GlobalsToErase.push_back(GV);
      continue;
    }

    auto DS = GV.getDescriptorSet();
    // PushConstant block (no DescriptorSet/Binding decorations).
    if (SC == mlir::spirv::StorageClass::PushConstant) {
      auto StructTy =
          mlir::dyn_cast<mlir::spirv::StructType>(PtrTy.getPointeeType());
      if (!StructTy || StructTy.getNumElements() == 0)
        continue;
      uint32_t RawBytes = spirvTypeBytes(StructTy);
      uint32_t PaddedBytes = ((RawBytes + 15u) / 16u) * 16u;
      if (PaddedBytes == 0)
        PaddedBytes = 16;

      ResourceBinding RB;
      RB.Class = ResourceClass::CBV;
      RB.Kind = ResourceKind::CBuffer;
      RB.RecordID = NextCBVId++;
      RB.Name = GV.getSymName().str();
      RB.Space = 0xFFFFFFFEu; // Sentinel space for Vulkan PushConstants.
      RB.LowerBound = 0;
      RB.RangeSize = 1;
      RB.ElementStride = PaddedBytes;
      RB.IsPushConstant = true;

      BoundCBufferInfo Info;
      Info.ReflectIndex = Reflect.Resources.size();
      Info.StructTy = StructTy;
      CBufferGlobals[GV.getSymName()] = Info;
      Reflect.Resources.push_back(std::move(RB));
      GlobalsToErase.push_back(GV);
      continue;
    }

    auto Binding = GV.getBinding();
    if (!DS || !Binding)
      continue;

    mlir::Type Pointee = PtrTy.getPointeeType();
    uint32_t RangeSize = 1;
    bool IsResourceArray = false;
    if (auto Arr = mlir::dyn_cast<mlir::spirv::ArrayType>(Pointee)) {
      RangeSize = Arr.getNumElements();
      Pointee = Arr.getElementType();
      IsResourceArray = true;
    } else if (auto RTArr =
                   mlir::dyn_cast<mlir::spirv::RuntimeArrayType>(Pointee)) {
      RangeSize = 0xFFFFFFFFu;
      Pointee = RTArr.getElementType();
      IsResourceArray = true;
    }

    // UniformConstant images (typed buffers & textures) and samplers.
    if (SC == mlir::spirv::StorageClass::UniformConstant) {
      if (auto ImgTy = mlir::dyn_cast<mlir::spirv::ImageType>(Pointee)) {
        ResourceKind Kind = mapImageKind(ImgTy);
        if (Kind == ResourceKind::Invalid)
          continue;
        bool IsUAV = ImgTy.getSamplerUseInfo() ==
                     mlir::spirv::ImageSamplerUseInfo::NoSampler;
        ResourceBinding RB;
        RB.Class = IsUAV ? ResourceClass::UAV : ResourceClass::SRV;
        RB.Kind = Kind;
        RB.ElementType = toComponentType(ImgTy.getElementType());
        RB.RecordID = IsUAV ? NextUAVId++ : NextSRVId++;
        RB.Name = GV.getSymName().str();
        RB.Space = *DS;
        RB.LowerBound = *Binding;
        RB.RangeSize = RangeSize;
        if (Kind == ResourceKind::TypedBuffer)
          RB.TypedComponents = imageFormatComponents(ImgTy.getImageFormat());

        BoundImageInfo Info;
        Info.ReflectIndex = Reflect.Resources.size();
        Info.Kind = Kind;
        Info.ElemType = ImgTy.getElementType();
        Info.IsUAV = IsUAV;
        Info.IsArray = IsResourceArray;
        ImageGlobals[GV.getSymName()] = Info;
        Reflect.Resources.push_back(std::move(RB));
        GlobalsToErase.push_back(GV);
        continue;
      }
      if (auto SITy = mlir::dyn_cast<mlir::spirv::SampledImageType>(Pointee)) {
        auto ImgTy = mlir::dyn_cast<mlir::spirv::ImageType>(SITy.getImageType());
        if (!ImgTy)
          continue;
        ResourceKind Kind = mapImageKind(ImgTy);
        if (Kind == ResourceKind::Invalid)
          continue;
        ResourceBinding TexRB;
        TexRB.Class = ResourceClass::SRV;
        TexRB.Kind = Kind;
        TexRB.ElementType = toComponentType(ImgTy.getElementType());
        TexRB.RecordID = NextSRVId++;
        TexRB.Name = GV.getSymName().str();
        TexRB.Space = *DS;
        TexRB.LowerBound = *Binding;
        TexRB.RangeSize = RangeSize;

        ResourceBinding SampRB;
        SampRB.Class = ResourceClass::Sampler;
        SampRB.Kind = ResourceKind::Sampler;
        SampRB.RecordID = NextSamplerId++;
        SampRB.Name = GV.getSymName().str() + "_sampler";
        SampRB.Space = *DS;
        SampRB.LowerBound = *Binding;
        SampRB.RangeSize = RangeSize;

        BoundImageInfo Info;
        Info.ReflectIndex = Reflect.Resources.size();
        Reflect.Resources.push_back(std::move(TexRB));
        Info.SamplerReflectIndex = Reflect.Resources.size();
        Reflect.Resources.push_back(std::move(SampRB));
        Info.Kind = Kind;
        Info.ElemType = ImgTy.getElementType();
        Info.IsUAV = false;
        Info.IsArray = IsResourceArray;
        ImageGlobals[GV.getSymName()] = Info;
        GlobalsToErase.push_back(GV);
        continue;
      }
      if (mlir::isa<mlir::spirv::SamplerType>(Pointee)) {
        ResourceBinding RB;
        RB.Class = ResourceClass::Sampler;
        RB.Kind = ResourceKind::Sampler;
        RB.RecordID = NextSamplerId++;
        RB.Name = GV.getSymName().str();
        RB.Space = *DS;
        RB.LowerBound = *Binding;
        RB.RangeSize = RangeSize;
        SamplerGlobals[GV.getSymName()] = Reflect.Resources.size();
        Reflect.Resources.push_back(std::move(RB));
        GlobalsToErase.push_back(GV);
        continue;
      }
      continue;
    }

    if (SC != mlir::spirv::StorageClass::Uniform &&
        SC != mlir::spirv::StorageClass::StorageBuffer)
      continue;

    auto StructTy = mlir::dyn_cast<mlir::spirv::StructType>(Pointee);
    if (!StructTy || StructTy.getNumElements() == 0)
      continue;

    bool IsStorageBlock =
        SC == mlir::spirv::StorageClass::StorageBuffer ||
        StructTy.hasDecoration(mlir::spirv::Decoration::BufferBlock);
    bool IsAddressTable =
        IsStorageBlock &&
        llvm::all_of(StructTy.getElementTypes(), [](mlir::Type T) {
          auto P = mlir::dyn_cast<mlir::spirv::PointerType>(T);
          return P && P.getStorageClass() ==
                          mlir::spirv::StorageClass::PhysicalStorageBuffer;
        });
    if (IsAddressTable) {
      ResourceBinding RB;
      RB.Class = ResourceClass::UAV;
      RB.Kind = ResourceKind::StructuredBuffer;
      RB.ElementType = ComponentType::U64;
      RB.RecordID = NextUAVId++;
      RB.Name = GV.getSymName().str();
      RB.Space = *DS;
      RB.LowerBound = *Binding;
      RB.RangeSize = RangeSize;
      RB.ElementStride = 8;
      BoundBufferInfo Info;
      Info.ReflectIndex = Reflect.Resources.size();
      Info.ElemType = StructTy;
      Info.ElemStride = 8;
      Info.IsArray = IsResourceArray;
      Info.IsAddressTable = true;
      BufferGlobals[GV.getSymName()] = Info;
      Reflect.Resources.push_back(std::move(RB));
      GlobalsToErase.push_back(GV);
      continue;
    }
    mlir::Type ArrayElemTy;
    uint32_t ArrayStride = 0;
    if (auto Arr = mlir::dyn_cast<mlir::spirv::RuntimeArrayType>(
            StructTy.getElementType(0))) {
      ArrayElemTy = Arr.getElementType();
      ArrayStride = Arr.getArrayStride();
    } else if (auto Arr = mlir::dyn_cast<mlir::spirv::ArrayType>(
                   StructTy.getElementType(0));
               Arr && IsStorageBlock && StructTy.getNumElements() == 1) {
      ArrayElemTy = Arr.getElementType();
      ArrayStride = Arr.getArrayStride();
    }
    if (ArrayElemTy) {
      mlir::Type ElemTy = ArrayElemTy;
      uint32_t Stride = ArrayStride;
      if (Stride == 0)
        Stride = 4;

      bool IsReadOnly = false;
      uint32_t MatStride = 16;
      bool MatRowMajor = false;
      SmallVector<mlir::spirv::StructType::MemberDecorationInfo> MemDecs;
      StructTy.getMemberDecorations(0, MemDecs);
      for (const auto &D : MemDecs) {
        if (D.decoration == mlir::spirv::Decoration::NonWritable)
          IsReadOnly = true;
        else if (D.decoration == mlir::spirv::Decoration::MatrixStride) {
          if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(D.decorationValue))
            MatStride = static_cast<uint32_t>(IA.getInt());
        } else if (D.decoration == mlir::spirv::Decoration::RowMajor) {
          MatRowMajor = true;
        }
      }

      ResourceBinding RB;
      RB.Class = IsReadOnly ? ResourceClass::SRV : ResourceClass::UAV;
      RB.Kind = ResourceKind::StructuredBuffer;
      RB.ElementType = toComponentType(ElemTy);
      RB.RecordID = IsReadOnly ? NextSRVId++ : NextUAVId++;
      RB.Name = GV.getSymName().str();
      RB.Space = *DS;
      RB.LowerBound = *Binding;
      RB.RangeSize = RangeSize;
      RB.ElementStride = Stride;

      BoundBufferInfo Info;
      Info.ReflectIndex = Reflect.Resources.size();
      Info.ElemType = ElemTy;
      Info.ElemStride = Stride;
      Info.IsArray = IsResourceArray;
      Info.MatrixStride = MatStride;
      Info.MatrixRowMajor = MatRowMajor;
      Info.BaseByteOffset = StructTy.getMemberOffset(0);
      BufferGlobals[GV.getSymName()] = Info;
      Reflect.Resources.push_back(std::move(RB));
      GlobalsToErase.push_back(GV);
      continue;
    }

    // Single-element storage block (e.g., UAV counter.var.* with
    // type.ACSBuffer.counter = struct<(si32 [0])>).
    if (IsStorageBlock && StructTy.getNumElements() == 1) {
      mlir::Type ElemTy = StructTy.getElementType(0);
      uint32_t Stride = spirvTypeBytes(ElemTy);
      if (Stride == 0)
        Stride = 4;

      ResourceBinding RB;
      RB.Class = ResourceClass::UAV;
      RB.Kind = ResourceKind::StructuredBuffer;
      RB.ElementType = toComponentType(ElemTy);
      RB.RecordID = NextUAVId++;
      RB.Name = GV.getSymName().str();
      RB.Space = *DS;
      RB.LowerBound = *Binding;
      RB.RangeSize = RangeSize;
      RB.ElementStride = Stride;

      BoundBufferInfo Info;
      Info.ReflectIndex = Reflect.Resources.size();
      Info.ElemType = ElemTy;
      Info.ElemStride = Stride;
      Info.IsArray = IsResourceArray;
      Info.BaseByteOffset = StructTy.getMemberOffset(0);
      BufferGlobals[GV.getSymName()] = Info;
      Reflect.Resources.push_back(std::move(RB));
      GlobalsToErase.push_back(GV);
      continue;
    }

    // Uniform Block struct without RuntimeArray -> Constant Buffer (CBV).
    if (SC == mlir::spirv::StorageClass::Uniform) {
      uint32_t RawBytes = spirvTypeBytes(StructTy);
      uint32_t PaddedBytes = ((RawBytes + 15u) / 16u) * 16u;
      if (PaddedBytes == 0)
        PaddedBytes = 16;

      ResourceBinding RB;
      RB.Class = ResourceClass::CBV;
      RB.Kind = ResourceKind::CBuffer;
      RB.RecordID = NextCBVId++;
      RB.Name = GV.getSymName().str();
      RB.Space = *DS;
      RB.LowerBound = *Binding;
      RB.RangeSize = RangeSize;
      RB.ElementStride = PaddedBytes;

      BoundCBufferInfo Info;
      Info.ReflectIndex = Reflect.Resources.size();
      Info.StructTy = StructTy;
      Info.IsArray = IsResourceArray;
      CBufferGlobals[GV.getSymName()] = Info;
      Reflect.Resources.push_back(std::move(RB));
      GlobalsToErase.push_back(GV);
      continue;
    }
  }

  auto EntryFn = SpvModule.lookupSymbol<mlir::spirv::FuncOp>(Reflect.EntryName);
  if (!EntryFn)
    return makeError("SPIR-V entry function '{0}' not found", Reflect.EntryName);

  SmallVector<mlir::spirv::InBoundsAccessChainOp, 8> InBoundsChains;
  EntryFn.walk([&](mlir::spirv::InBoundsAccessChainOp Chain) {
    InBoundsChains.push_back(Chain);
  });
  for (auto Chain : InBoundsChains) {
    mlir::OpBuilder B(Chain);
    auto Replacement = mlir::spirv::AccessChainOp::create(
        B, Chain.getLoc(), Chain.getBasePtr(), Chain.getIndices());
    Replacement->setAttrs(Chain->getAttrs());
    Chain.replaceAllUsesWith(Replacement.getResult());
    Chain.erase();
  }

  SmallVector<mlir::spirv::VectorExtractDynamicOp, 8> DynamicExtracts;
  EntryFn.walk([&](mlir::spirv::VectorExtractDynamicOp Extract) {
    DynamicExtracts.push_back(Extract);
  });
  for (auto Extract : DynamicExtracts) {
    mlir::OpBuilder B(Extract);
    auto Loc = Extract.getLoc();
    auto Value = mlir::LLVM::ExtractElementOp::create(
        B, Loc, castToSignless(B, Loc, Extract.getVector()),
        castToSignless(B, Loc, Extract.getIndex()));
    Extract.replaceAllUsesWith(
        castFromSignless(B, Loc, Value, Extract.getType()));
    Extract.erase();
  }

  SmallVector<mlir::spirv::CopyMemoryOp, 8> Copies;
  EntryFn.walk([&](mlir::spirv::CopyMemoryOp Copy) { Copies.push_back(Copy); });
  for (auto Copy : Copies) {
    auto Ty = mlir::cast<mlir::spirv::PointerType>(Copy.getSource().getType())
                  .getPointeeType();
    if (!mlir::isa<mlir::IntegerType, mlir::FloatType, mlir::VectorType>(Ty))
      return makeError("SPIR-V buffer copy requires a scalar or vector pointee");
    mlir::OpBuilder B(Copy);
    auto SourceAccess = Copy.getSourceMemoryAccessAttr();
    auto SourceAlignment = Copy.getSourceAlignmentAttr();
    if (!SourceAccess) {
      SourceAccess = Copy.getMemoryAccessAttr();
      SourceAlignment = Copy.getAlignmentAttr();
    }
    auto Load = mlir::spirv::LoadOp::create(
        B, Copy.getLoc(), Copy.getSource(), SourceAccess, SourceAlignment);
    mlir::spirv::StoreOp::create(
        B, Copy.getLoc(), Copy.getTarget(), Load.getResult(),
        Copy.getMemoryAccessAttr(), Copy.getAlignmentAttr());
    Copy.erase();
  }

  // 2b. Fold spirv.SpecConstant + spirv.mlir.referenceof -> spirv.Constant.
  struct SpecConstInfo {
    uint32_t SpecId = 0;
    mlir::Attribute DefaultVal;
  };
  llvm::StringMap<SpecConstInfo> SpecConstMap;
  SmallVector<mlir::spirv::SpecConstantOp, 8> SpecConstsToErase;
  for (mlir::spirv::SpecConstantOp SC :
       SpvModule.getOps<mlir::spirv::SpecConstantOp>()) {
    uint32_t Id = 0;
    if (auto SpecIdAttr = SC->getAttrOfType<mlir::IntegerAttr>("spec_id"))
      Id = static_cast<uint32_t>(SpecIdAttr.getInt());
    SpecConstMap[SC.getSymName()] = {Id, SC.getDefaultValue()};
    SpecConstsToErase.push_back(SC);
  }
  if (!SpecConstMap.empty()) {
    SmallVector<mlir::spirv::ReferenceOfOp, 8> RefOps;
    EntryFn.walk([&](mlir::spirv::ReferenceOfOp R) { RefOps.push_back(R); });
    for (mlir::spirv::ReferenceOfOp Ref : RefOps) {
      auto It = SpecConstMap.find(Ref.getSpecConst());
      if (It == SpecConstMap.end())
        continue;
      mlir::OpBuilder B(Ref);
      mlir::Type Ty = Ref.getType();
      mlir::Attribute ValAttr = It->second.DefaultVal;
      auto OverIt = SpecConstants.find(It->second.SpecId);
      if (OverIt != SpecConstants.end()) {
        uint64_t Bits = OverIt->second;
        if (Ty.isInteger(1)) {
          ValAttr = B.getBoolAttr(Bits != 0);
        } else if (auto IT = mlir::dyn_cast<mlir::IntegerType>(Ty)) {
          ValAttr = mlir::IntegerAttr::get(IT, llvm::APInt(IT.getWidth(), Bits));
        } else if (Ty.isF16()) {
          ValAttr = mlir::FloatAttr::get(
              Ty, llvm::APFloat(llvm::APFloat::IEEEhalf(),
                                llvm::APInt(16, Bits & 0xFFFFu)));
        } else if (Ty.isF32()) {
          ValAttr = mlir::FloatAttr::get(
              Ty, llvm::APFloat(llvm::APFloat::IEEEsingle(),
                                llvm::APInt(32, Bits & 0xFFFFFFFFu)));
        } else if (Ty.isF64()) {
          ValAttr = mlir::FloatAttr::get(
              Ty, llvm::APFloat(llvm::APFloat::IEEEdouble(),
                                llvm::APInt(64, Bits)));
        }
      }
      auto Cst =
          mlir::spirv::ConstantOp::create(B, Ref.getLoc(), Ty, ValAttr);
      Ref.replaceAllUsesWith(Cst.getResult());
      Ref.erase();
    }
    for (auto SC : SpecConstsToErase)
      SC.erase();
  }

  // 3. Rewrite built-in & bound-buffer accesses inside the entry function.
  mlir::OpBuilder EntryBuilder =
      mlir::OpBuilder::atBlockBegin(&EntryFn.getBody().front());
  mlir::Location FnLoc = EntryFn.getLoc();
  mlir::LLVM::LLVMStructType HandleTy = getOrInitHandleType(&Ctx);
  auto I1Ty = EntryBuilder.getI1Type();
  auto I8Ty = EntryBuilder.getI8Type();
  auto I32Ty = EntryBuilder.getI32Type();
  auto VoidTy = mlir::LLVM::LLVMVoidType::get(&Ctx);

  DenseMap<size_t, mlir::Value> HandleCache;
  auto getHandleForIndex = [&](size_t ReflectIndex) -> mlir::Value {
    auto It = HandleCache.find(ReflectIndex);
    if (It != HandleCache.end())
      return It->second;

    EntryBuilder.setInsertionPointToStart(&EntryFn.getBody().front());
    const ResourceBinding &RB = Reflect.Resources[ReflectIndex];
    auto ResTy = ir::ResourceType::get(&Ctx);
    auto IdxC = mlir::LLVM::ConstantOp::create(
        EntryBuilder, FnLoc, I32Ty, static_cast<int32_t>(RB.LowerBound));
    auto H = ir::BindResourceOp::create(
        EntryBuilder, FnLoc, ResTy,
        ir::ResourceClassAttr::get(
            &Ctx, static_cast<ir::ResourceClass>(RB.Class)),
        EntryBuilder.getI32IntegerAttr(static_cast<int32_t>(RB.RecordID)),
        /*space=*/nullptr, /*binding=*/nullptr,
        EntryBuilder.getBoolAttr(false), IdxC);
    HandleCache[ReflectIndex] = H.getResult();
    return H.getResult();
  };
  auto getHandleForIndexAt = [&](size_t ReflectIndex, mlir::OpBuilder &B,
                                 mlir::Location Loc,
                                 mlir::Value ArrIdx) -> mlir::Value {
    if (!ArrIdx)
      return getHandleForIndex(ReflectIndex);
    const ResourceBinding &RB = Reflect.Resources[ReflectIndex];
    auto ResTy = ir::ResourceType::get(&Ctx);
    mlir::Value IdxVal;
    if (auto C = getConstSpirvInt(ArrIdx)) {
      IdxVal = mlir::LLVM::ConstantOp::create(
          B, Loc, I32Ty, static_cast<int32_t>(RB.LowerBound + *C));
    } else {
      mlir::Value Arr32 = castToSignless(B, Loc, ArrIdx);
      if (RB.LowerBound != 0) {
        auto BaseC = mlir::LLVM::ConstantOp::create(
            B, Loc, I32Ty, static_cast<int32_t>(RB.LowerBound));
        IdxVal = mlir::LLVM::AddOp::create(B, Loc, BaseC, Arr32);
      } else {
        IdxVal = Arr32;
      }
    }
    return ir::BindResourceOp::create(
               B, Loc, ResTy,
               ir::ResourceClassAttr::get(
                   &Ctx, static_cast<ir::ResourceClass>(RB.Class)),
               B.getI32IntegerAttr(static_cast<int32_t>(RB.RecordID)),
               /*space=*/nullptr, /*binding=*/nullptr, B.getBoolAttr(false),
               IdxVal)
        .getResult();
  };

  SmallVector<mlir::spirv::AddressOfOp, 8> AddrOps;
  EntryFn.walk([&](mlir::spirv::AddressOfOp AO) { AddrOps.push_back(AO); });
  DenseSet<mlir::Operation *> ErasedAddrOps;

  for (auto AO : AddrOps) {
    if (ErasedAddrOps.contains(AO))
      continue;
    StringRef Sym = AO.getVariable();

    // 3a. Built-in globals (vector IDs -> gpu.*, LocalInvocationIndex -> gidx).
    auto BIIt = BuiltInGlobals.find(Sym);
    if (BIIt != BuiltInGlobals.end()) {
      mlir::spirv::BuiltIn BI = BIIt->second;
      SmallVector<mlir::Operation *, 4> Loads(AO->getUsers().begin(),
                                              AO->getUsers().end());
      for (mlir::Operation *U : Loads) {
        if (auto Chain = mlir::dyn_cast<mlir::spirv::AccessChainOp>(U)) {
          if (Chain.getIndices().size() != 1)
            return makeError("unsupported access chain on built-in '{0}'", Sym);
          auto ConstComp = getConstSpirvInt(Chain.getIndices()[0]);
          if (!ConstComp)
            return makeError("dynamic index on built-in '{0}'", Sym);
          uint32_t Comp = static_cast<uint32_t>(*ConstComp);
          SmallVector<mlir::Operation *, 4> CLoads(Chain->getUsers().begin(),
                                                   Chain->getUsers().end());
          for (mlir::Operation *CLU : CLoads) {
            auto CLoad = mlir::dyn_cast<mlir::spirv::LoadOp>(CLU);
            if (!CLoad)
              return makeError("unsupported user of built-in chain '{0}'", Sym);
            mlir::OpBuilder B(CLoad);
            mlir::Value Rep =
                emitGPUIdComponent(B, CLoad.getLoc(), BI, Comp, CLoad.getType());
            if (!Rep)
              return makeError("unsupported SPIR-V built-in '{0}'", Sym);
            Rep = castFromSignless(B, CLoad.getLoc(), Rep, CLoad.getType());
            CLoad.replaceAllUsesWith(Rep);
            CLoad.erase();
          }
          Chain.erase();
          continue;
        }
        auto Load = mlir::dyn_cast<mlir::spirv::LoadOp>(U);
        if (!Load)
          return makeError("unsupported user of built-in global '{0}'", Sym);
        if (BI == mlir::spirv::BuiltIn::SubgroupLocalInvocationId ||
            BI == mlir::spirv::BuiltIn::SubgroupSize) {
          mlir::OpBuilder B(Load);
          mlir::Location Loc = Load.getLoc();
          mlir::Value IdxVal =
              BI == mlir::spirv::BuiltIn::SubgroupLocalInvocationId
                  ? mlir::Value(mlir::gpu::LaneIdOp::create(B, Loc, /*upper_bound=*/nullptr))
                  : mlir::Value(mlir::gpu::SubgroupSizeOp::create(B, Loc, /*upper_bound=*/nullptr));
          mlir::Type DstTy = makeSignless(Load.getType(), B);
          auto Cast =
              mlir::arith::IndexCastUIOp::create(B, Loc, DstTy, IdxVal);
          Load.replaceAllUsesWith(Cast.getResult());
          Load.erase();
          continue;
        }
        if (BI == mlir::spirv::BuiltIn::LocalInvocationIndex) {
          mlir::OpBuilder B(Load);
          mlir::Location Loc = Load.getLoc();
          auto SV = ir::SysValOp::create(
              B, Loc, I32Ty,
              ir::SysValKindAttr::get(B.getContext(),
                                      ir::SysValKind::FlattenedThreadIdInGroup),
              /*component=*/nullptr);
          Load.replaceAllUsesWith(SV.getResult());
          Load.erase();
          continue;
        }
        SmallVector<mlir::Operation *, 4> Extracts(Load->getUsers().begin(),
                                                   Load->getUsers().end());
        bool AllSingleExtracts = !Extracts.empty();
        for (mlir::Operation *EU : Extracts) {
          auto Extract = mlir::dyn_cast<mlir::spirv::CompositeExtractOp>(EU);
          if (!Extract || Extract.getIndices().size() != 1) {
            AllSingleExtracts = false;
            break;
          }
        }
        if (AllSingleExtracts) {
          for (mlir::Operation *EU : Extracts) {
            auto Extract = mlir::cast<mlir::spirv::CompositeExtractOp>(EU);
            uint32_t Comp = static_cast<uint32_t>(
                mlir::cast<mlir::IntegerAttr>(Extract.getIndices()[0]).getInt());
            mlir::OpBuilder B(Extract);
            mlir::Value Rep = emitGPUIdComponent(B, Extract.getLoc(), BI, Comp,
                                                 Extract.getType());
            if (!Rep)
              return makeError("unsupported SPIR-V built-in '{0}'", Sym);
            Extract.replaceAllUsesWith(Rep);
            Extract.erase();
          }
        } else if (auto VecTy = mlir::dyn_cast<mlir::VectorType>(Load.getType())) {
          mlir::OpBuilder B(Load);
          mlir::Location Loc = Load.getLoc();
          mlir::Type SignlessVecTy = makeSignless(VecTy, B);
          mlir::Value Vec = mlir::LLVM::UndefOp::create(B, Loc, SignlessVecTy);
          for (int64_t C = 0; C < VecTy.getNumElements(); ++C) {
            mlir::Value CompVal = emitGPUIdComponent(
                B, Loc, BI, static_cast<uint32_t>(C), VecTy.getElementType());
            if (!CompVal)
              return makeError("unsupported SPIR-V built-in '{0}'", Sym);
            CompVal = castToSignless(B, Loc, CompVal);
            auto IdxC = mlir::LLVM::ConstantOp::create(
                B, Loc, I32Ty, static_cast<int32_t>(C));
            Vec = mlir::LLVM::InsertElementOp::create(B, Loc, Vec, CompVal, IdxC);
          }
          Load.replaceAllUsesWith(castFromSignless(B, Loc, Vec, Load.getType()));
        } else {
          return makeError("unsupported extract on built-in '{0}'", Sym);
        }
        if (!Load->use_empty())
          return makeError("unhandled vector use of built-in '{0}'", Sym);
        Load.erase();
      }
      AO.erase();
      continue;
    }

    // 3b. Descriptor-bound structured buffers -> crucible.buffer_load / store.
    auto BufIt = BufferGlobals.find(Sym);
    if (BufIt != BufferGlobals.end()) {
      const BoundBufferInfo &Info = BufIt->second;

      std::function<Error(mlir::spirv::AccessChainOp,
                          SmallVector<mlir::Value, 8>)>
          lowerBufferChain = [&](mlir::spirv::AccessChainOp Chain,
                                 SmallVector<mlir::Value, 8> AllIndices) -> Error {
        for (mlir::Value Idx : Chain.getIndices())
          AllIndices.push_back(Idx);
        size_t BaseIdx = Info.IsArray ? 1 : 0;
        if (AllIndices.size() <= BaseIdx)
          return makeError("unsupported access pattern on buffer '{0}'", Sym);

        mlir::Value ArrIdx = Info.IsArray ? AllIndices[0] : mlir::Value();
        mlir::Value ElemIdx =
            !Info.IsAddressTable && AllIndices.size() >= BaseIdx + 2
                ? AllIndices[BaseIdx + 1]
                : mlir::Value();
        auto getElemIdx = [&](mlir::OpBuilder &B, mlir::Location Loc) -> mlir::Value {
          if (ElemIdx)
            return castToSignless(B, Loc, ElemIdx);
          return mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 0);
        };
        // Compute sub-element byte offset (constant or dynamic) for struct/
        // array/matrix/vector fields.
        uint32_t CurMatrixStride = Info.MatrixStride;
        bool CurMatrixRowMajor = Info.MatrixRowMajor;
        uint32_t NextVecElemStride = 0;
        auto computeByteOff = [&](mlir::OpBuilder &B,
                                  mlir::Location Loc) -> Expected<mlir::Value> {
          uint32_t ConstOff = Info.BaseByteOffset;
          mlir::Value DynOff;
          mlir::Type CurTy = Info.ElemType;
          CurMatrixStride = Info.MatrixStride;
          CurMatrixRowMajor = Info.MatrixRowMajor;
          NextVecElemStride = 0;
          auto addDynTerm = [&](mlir::Value IdxV, uint32_t S) {
            mlir::Value Idx32 = castToSignless(B, Loc, IdxV);
            auto StrideC = mlir::LLVM::ConstantOp::create(
                B, Loc, I32Ty, static_cast<int32_t>(S));
            mlir::Value Term =
                mlir::LLVM::MulOp::create(B, Loc, Idx32, StrideC);
            DynOff = DynOff
                         ? mlir::Value(
                               mlir::LLVM::AddOp::create(B, Loc, DynOff, Term))
                         : Term;
          };
          for (size_t K = BaseIdx + (Info.IsAddressTable ? 0 : 2);
               K < AllIndices.size(); ++K) {
            mlir::Value IdxV = AllIndices[K];
            auto CIdx = getConstSpirvInt(IdxV);
            if (auto ST = mlir::dyn_cast<mlir::spirv::StructType>(CurTy)) {
              if (!CIdx)
                return makeError("dynamic struct index on buffer '{0}'", Sym);
              uint32_t I = static_cast<uint32_t>(*CIdx);
              ConstOff += ST.getMemberOffset(I);
              CurMatrixStride = 16u;
              CurMatrixRowMajor = false;
              NextVecElemStride = 0;
              SmallVector<mlir::spirv::StructType::MemberDecorationInfo> MDecs;
              ST.getMemberDecorations(I, MDecs);
              for (const auto &D : MDecs) {
                if (D.decoration == mlir::spirv::Decoration::MatrixStride) {
                  if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(D.decorationValue))
                    CurMatrixStride = static_cast<uint32_t>(IA.getInt());
                } else if (D.decoration == mlir::spirv::Decoration::RowMajor) {
                  CurMatrixRowMajor = true;
                }
              }
              CurTy = ST.getElementType(I);
            } else if (auto Arr =
                           mlir::dyn_cast<mlir::spirv::ArrayType>(CurTy)) {
              uint32_t S = Arr.getArrayStride();
              if (S == 0)
                S = spirvTypeBytes(Arr.getElementType());
              if (CIdx)
                ConstOff += static_cast<uint32_t>(*CIdx) * S;
              else
                addDynTerm(IdxV, S);
              CurTy = Arr.getElementType();
            } else if (auto Mat =
                           mlir::dyn_cast<mlir::spirv::MatrixType>(CurTy)) {
              auto ColVec = mlir::cast<mlir::VectorType>(Mat.getColumnType());
              uint32_t ElemBytes = scalarTypeBytes(ColVec.getElementType());
              uint32_t Step = CurMatrixRowMajor ? ElemBytes : CurMatrixStride;
              NextVecElemStride = CurMatrixRowMajor ? CurMatrixStride : ElemBytes;
              if (CIdx)
                ConstOff += static_cast<uint32_t>(*CIdx) * Step;
              else
                addDynTerm(IdxV, Step);
              CurTy = ColVec;
            } else if (auto Vec = mlir::dyn_cast<mlir::VectorType>(CurTy)) {
              uint32_t S = NextVecElemStride
                               ? NextVecElemStride
                               : scalarTypeBytes(Vec.getElementType());
              if (CIdx)
                ConstOff += static_cast<uint32_t>(*CIdx) * S;
              else
                addDynTerm(IdxV, S);
              NextVecElemStride = 0;
              CurTy = Vec.getElementType();
            }
          }
          mlir::Value Base = mlir::LLVM::ConstantOp::create(
              B, Loc, I32Ty, static_cast<int32_t>(ConstOff));
          if (DynOff) {
            if (ConstOff == 0)
              return DynOff;
            return mlir::Value(mlir::LLVM::AddOp::create(B, Loc, Base, DynOff));
          }
          return Base;
        };
        SmallVector<mlir::Operation *, 8> MemUsers(Chain->getUsers().begin(),
                                                   Chain->getUsers().end());
        for (mlir::Operation *MU : MemUsers) {
          if (auto SubChain = mlir::dyn_cast<mlir::spirv::AccessChainOp>(MU)) {
            if (auto Err = lowerBufferChain(SubChain, AllIndices))
              return Err;
          } else if (auto Load = mlir::dyn_cast<mlir::spirv::LoadOp>(MU)) {
            mlir::OpBuilder B(Load);
            mlir::Location Loc = Load.getLoc();
            mlir::Value Handle =
                getHandleForIndexAt(Info.ReflectIndex, B, Loc, ArrIdx);
            auto OffValOrErr = computeByteOff(B, Loc);
            if (!OffValOrErr)
              return OffValOrErr.takeError();
            mlir::Value OffC = *OffValOrErr;
            mlir::Type OrigTy = Load.getType();
            mlir::Value IdxVal = getElemIdx(B, Loc);
            auto addByteOff = [&](uint32_t Delta) -> mlir::Value {
              if (Delta == 0)
                return OffC;
              auto DeltaC = mlir::LLVM::ConstantOp::create(
                  B, Loc, I32Ty, static_cast<int32_t>(Delta));
              return mlir::LLVM::AddOp::create(B, Loc, OffC, DeltaC);
            };
            if (auto MatTy = mlir::dyn_cast<mlir::spirv::MatrixType>(OrigTy)) {
              auto ColVec = mlir::cast<mlir::VectorType>(MatTy.getColumnType());
              mlir::Type ScalarTy = makeSignless(ColVec.getElementType(), B);
              mlir::Type SignlessColTy = makeSignless(ColVec, B);
              uint32_t ElemBytes = scalarTypeBytes(ScalarTy);
              uint32_t ColStep = CurMatrixRowMajor ? ElemBytes : CurMatrixStride;
              uint32_t RowStep = CurMatrixRowMajor ? CurMatrixStride : ElemBytes;
              auto RetVecTy = mlir::VectorType::get({4}, ScalarTy);
              auto ZeroC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 0);
              SmallVector<mlir::Value, 4> Cols;
              for (uint32_t Col = 0; Col < MatTy.getNumColumns(); ++Col) {
                mlir::Value Vec = mlir::LLVM::UndefOp::create(B, Loc, SignlessColTy);
                for (int64_t Row = 0; Row < ColVec.getNumElements(); ++Row) {
                  mlir::Value ElemOff = addByteOff(
                      Col * ColStep + static_cast<uint32_t>(Row) * RowStep);
                  auto LOp = ir::BufferLoadOp::create(
                      B, Loc, RetVecTy, mlir::Type(), Handle,
                      ir::BufferKindAttr::get(&Ctx, ir::BufferKind::Raw), IdxVal,
                      ElemOff, B.getI32IntegerAttr(1),
                      B.getI32IntegerAttr(static_cast<int32_t>(ElemBytes)));
                  if (auto E = preserveBufferMemoryAccess(
                          LOp, Load.getMemoryAccessAttr(),
                          Load.getAlignmentAttr()))
                    return E;
                  auto Lane = mlir::LLVM::ExtractElementOp::create(
                      B, Loc, LOp.getResult(), ZeroC);
                  auto IdxC = mlir::LLVM::ConstantOp::create(
                      B, Loc, I32Ty, static_cast<int32_t>(Row));
                  Vec = mlir::LLVM::InsertElementOp::create(B, Loc, Vec, Lane,
                                                            IdxC);
                }
                Cols.push_back(castFromSignless(B, Loc, Vec, ColVec));
              }
              auto MatCC = mlir::spirv::CompositeConstructOp::create(
                  B, Loc, OrigTy, Cols);
              Load.replaceAllUsesWith(MatCC.getResult());
              Load.erase();
              continue;
            }
            auto VecTy = mlir::dyn_cast<mlir::VectorType>(OrigTy);
            mlir::Type ScalarTy = makeSignless(
                VecTy ? VecTy.getElementType() : OrigTy, B);
            if (Info.IsAddressTable) {
              auto P = mlir::dyn_cast<mlir::spirv::PointerType>(OrigTy);
              if (!P || P.getStorageClass() !=
                            mlir::spirv::StorageClass::PhysicalStorageBuffer)
                return makeError(
                    "address-table loads require physical pointers");
              ScalarTy = B.getI64Type();
            }
            uint32_t ElemBytes = scalarTypeBytes(ScalarTy);
            unsigned NumComps =
                VecTy ? static_cast<unsigned>(VecTy.getNumElements()) : 1u;
            auto RetVecTy = mlir::VectorType::get({4}, ScalarTy);
            mlir::Value ResultVal;
            if (VecTy && NextVecElemStride > ElemBytes) {
              auto ZeroC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 0);
              mlir::Type SignlessVecTy = makeSignless(VecTy, B);
              mlir::Value Vec = mlir::LLVM::UndefOp::create(B, Loc, SignlessVecTy);
              for (unsigned C = 0; C < NumComps && C < 4; ++C) {
                auto LOp = ir::BufferLoadOp::create(
                    B, Loc, RetVecTy, mlir::Type(), Handle,
                    ir::BufferKindAttr::get(&Ctx, ir::BufferKind::Raw), IdxVal,
                    addByteOff(C * NextVecElemStride), B.getI32IntegerAttr(1),
                    B.getI32IntegerAttr(static_cast<int32_t>(ElemBytes)));
                if (auto E = preserveBufferMemoryAccess(
                        LOp, Load.getMemoryAccessAttr(),
                        Load.getAlignmentAttr()))
                  return E;
                auto VC = mlir::LLVM::ExtractElementOp::create(
                    B, Loc, LOp.getResult(), ZeroC);
                auto IdxC = mlir::LLVM::ConstantOp::create(
                    B, Loc, I32Ty, static_cast<int32_t>(C));
                Vec = mlir::LLVM::InsertElementOp::create(B, Loc, Vec, VC, IdxC);
              }
              ResultVal = Vec;
            } else {
              uint8_t Mask = static_cast<uint8_t>((1u << std::min(NumComps, 4u)) - 1u);
              auto LoadOp = ir::BufferLoadOp::create(
                  B, Loc, RetVecTy, mlir::Type(), Handle,
                  ir::BufferKindAttr::get(&Ctx, ir::BufferKind::Raw), IdxVal,
                  OffC, B.getI32IntegerAttr(static_cast<int32_t>(Mask)),
                  B.getI32IntegerAttr(static_cast<int32_t>(ElemBytes)));
              if (auto E = preserveBufferMemoryAccess(
                      LoadOp, Load.getMemoryAccessAttr(),
                      Load.getAlignmentAttr()))
                return E;
              if (!VecTy) {
                auto ZeroC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 0);
                ResultVal = mlir::LLVM::ExtractElementOp::create(
                    B, Loc, LoadOp.getResult(), ZeroC);
                if (Info.IsAddressTable)
                  ResultVal.getDefiningOp()->setAttr("crucible.address_table",
                                                     B.getUnitAttr());
              } else {
                mlir::Type SignlessVecTy = makeSignless(VecTy, B);
                mlir::Value Vec = mlir::LLVM::UndefOp::create(B, Loc, SignlessVecTy);
                for (unsigned C = 0; C < NumComps && C < 4; ++C) {
                  auto IdxC = mlir::LLVM::ConstantOp::create(
                      B, Loc, I32Ty, static_cast<int32_t>(C));
                  auto VC = mlir::LLVM::ExtractElementOp::create(
                      B, Loc, LoadOp.getResult(), IdxC);
                  Vec = mlir::LLVM::InsertElementOp::create(B, Loc, Vec, VC, IdxC);
                }
                ResultVal = Vec;
              }
            }
            ResultVal = Info.IsAddressTable
                            ? mlir::UnrealizedConversionCastOp::create(
                                  B, Loc, OrigTy, ResultVal)
                                  .getResult(0)
                            : castFromSignless(B, Loc, ResultVal, OrigTy);
            Load.replaceAllUsesWith(ResultVal);
            Load.erase();
          } else if (auto Store = mlir::dyn_cast<mlir::spirv::StoreOp>(MU)) {
            if (Info.IsAddressTable)
              return makeError(
                  "stores to physical address tables are unsupported");
            mlir::OpBuilder B(Store);
            mlir::Location Loc = Store.getLoc();
            mlir::Value Handle =
                getHandleForIndexAt(Info.ReflectIndex, B, Loc, ArrIdx);
            auto OffValOrErr = computeByteOff(B, Loc);
            if (!OffValOrErr)
              return OffValOrErr.takeError();
            mlir::Value OffC = *OffValOrErr;
            mlir::Value IdxVal = getElemIdx(B, Loc);
            auto addByteOff = [&](uint32_t Delta) -> mlir::Value {
              if (Delta == 0)
                return OffC;
              auto DeltaC = mlir::LLVM::ConstantOp::create(
                  B, Loc, I32Ty, static_cast<int32_t>(Delta));
              return mlir::LLVM::AddOp::create(B, Loc, OffC, DeltaC);
            };
            if (auto MatTy = mlir::dyn_cast<mlir::spirv::MatrixType>(
                    Store.getValue().getType())) {
              auto ColVec = mlir::cast<mlir::VectorType>(MatTy.getColumnType());
              mlir::Type ScalarTy = makeSignless(ColVec.getElementType(), B);
              uint32_t ElemBytes = scalarTypeBytes(ScalarTy);
              uint32_t ColStep = CurMatrixRowMajor ? ElemBytes : CurMatrixStride;
              uint32_t RowStep = CurMatrixRowMajor ? CurMatrixStride : ElemBytes;
              auto Undef = mlir::LLVM::UndefOp::create(B, Loc, ScalarTy);
              for (uint32_t Col = 0; Col < MatTy.getNumColumns(); ++Col) {
                mlir::Value ColV = mlir::spirv::CompositeExtractOp::create(
                    B, Loc, ColVec, Store.getValue(),
                    B.getI32ArrayAttr({static_cast<int32_t>(Col)}));
                ColV = castToSignless(B, Loc, ColV);
                for (int64_t Row = 0; Row < ColVec.getNumElements(); ++Row) {
                  auto IdxC = mlir::LLVM::ConstantOp::create(
                      B, Loc, I32Ty, static_cast<int32_t>(Row));
                  mlir::Value Lane =
                      mlir::LLVM::ExtractElementOp::create(B, Loc, ColV, IdxC);
                  SmallVector<mlir::Value, 4> Vals(4, Undef.getResult());
                  Vals[0] = Lane;
                  auto SOp = ir::BufferStoreOp::create(
                      B, Loc, Handle,
                      ir::BufferKindAttr::get(&Ctx, ir::BufferKind::Raw), IdxVal,
                      addByteOff(Col * ColStep +
                                 static_cast<uint32_t>(Row) * RowStep),
                      Vals, B.getI32IntegerAttr(1),
                      B.getI32IntegerAttr(static_cast<int32_t>(ElemBytes)));
                  if (auto E = preserveBufferMemoryAccess(
                          SOp, Store.getMemoryAccessAttr(),
                          Store.getAlignmentAttr()))
                    return E;
                }
              }
              Store.erase();
              continue;
            }
            mlir::Value StoreVal = castToSignless(B, Loc, Store.getValue());
            auto VecTy = mlir::dyn_cast<mlir::VectorType>(StoreVal.getType());
            mlir::Type ScalarTy =
                VecTy ? VecTy.getElementType() : StoreVal.getType();
            uint32_t ElemBytes = scalarTypeBytes(ScalarTy);
            unsigned NumComps =
                VecTy ? static_cast<unsigned>(VecTy.getNumElements()) : 1u;
            auto Undef = mlir::LLVM::UndefOp::create(B, Loc, ScalarTy);
            if (VecTy && NextVecElemStride > ElemBytes) {
              for (unsigned C = 0; C < NumComps && C < 4; ++C) {
                auto IdxC = mlir::LLVM::ConstantOp::create(
                    B, Loc, I32Ty, static_cast<int32_t>(C));
                SmallVector<mlir::Value, 4> Vals(4, Undef.getResult());
                Vals[0] = mlir::LLVM::ExtractElementOp::create(B, Loc, StoreVal,
                                                               IdxC);
                auto SOp = ir::BufferStoreOp::create(
                    B, Loc, Handle,
                    ir::BufferKindAttr::get(&Ctx, ir::BufferKind::Raw), IdxVal,
                    addByteOff(C * NextVecElemStride), Vals,
                    B.getI32IntegerAttr(1),
                    B.getI32IntegerAttr(static_cast<int32_t>(ElemBytes)));
                if (auto E = preserveBufferMemoryAccess(
                        SOp, Store.getMemoryAccessAttr(),
                        Store.getAlignmentAttr()))
                  return E;
              }
              Store.erase();
              continue;
            }
            uint8_t Mask = static_cast<uint8_t>((1u << std::min(NumComps, 4u)) - 1u);
            SmallVector<mlir::Value, 4> Vals(4, Undef.getResult());
            if (!VecTy) {
              Vals[0] = StoreVal;
            } else {
              for (unsigned C = 0; C < NumComps && C < 4; ++C) {
                auto IdxC = mlir::LLVM::ConstantOp::create(
                    B, Loc, I32Ty, static_cast<int32_t>(C));
                Vals[C] = mlir::LLVM::ExtractElementOp::create(B, Loc, StoreVal,
                                                               IdxC);
              }
            }
            auto SOp = ir::BufferStoreOp::create(
                B, Loc, Handle,
                ir::BufferKindAttr::get(&Ctx, ir::BufferKind::Raw), IdxVal,
                OffC, Vals, B.getI32IntegerAttr(static_cast<int32_t>(Mask)),
                B.getI32IntegerAttr(static_cast<int32_t>(ElemBytes)));
            if (auto E = preserveBufferMemoryAccess(
                    SOp, Store.getMemoryAccessAttr(), Store.getAlignmentAttr()))
              return E;
            Store.erase();
          } else {
            // Structured buffer atomics -> crucible.buffer_atomic.
            std::optional<ir::AtomicOp> ABinOp;
            bool NegateVal = false;
            mlir::Value AVal;
            if (auto A = mlir::dyn_cast<mlir::spirv::AtomicIAddOp>(MU)) {
              ABinOp = ir::AtomicOp::Add;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicISubOp>(MU)) {
              ABinOp = ir::AtomicOp::Add;
              NegateVal = true;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicAndOp>(MU)) {
              ABinOp = ir::AtomicOp::And;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicOrOp>(MU)) {
              ABinOp = ir::AtomicOp::Or;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicXorOp>(MU)) {
              ABinOp = ir::AtomicOp::Xor;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicSMinOp>(MU)) {
              ABinOp = ir::AtomicOp::SMin;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicSMaxOp>(MU)) {
              ABinOp = ir::AtomicOp::SMax;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicUMinOp>(MU)) {
              ABinOp = ir::AtomicOp::UMin;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicUMaxOp>(MU)) {
              ABinOp = ir::AtomicOp::UMax;
              AVal = A.getValue();
            } else if (auto A =
                           mlir::dyn_cast<mlir::spirv::AtomicExchangeOp>(MU)) {
              ABinOp = ir::AtomicOp::Xchg;
              AVal = A.getValue();
            }
            if (!ABinOp)
              return makeError("unsupported memory op '{0}' on buffer '{1}'",
                               MU->getName().getStringRef(), Sym);

            mlir::OpBuilder B(MU);
            mlir::Location Loc = MU->getLoc();
            mlir::Value Handle =
                getHandleForIndexAt(Info.ReflectIndex, B, Loc, ArrIdx);
            mlir::Type OrigTy = AVal.getType();
            mlir::Value SignlessVal = castToSignless(B, Loc, AVal);
            mlir::Value IdxVal = getElemIdx(B, Loc);
            mlir::Type ValTy = SignlessVal.getType();
            if (NegateVal) {
              auto ZeroVal = mlir::LLVM::ConstantOp::create(B, Loc, ValTy, 0);
              SignlessVal =
                  mlir::LLVM::SubOp::create(B, Loc, ZeroVal, SignlessVal);
            }
            auto ZeroC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 0);
            auto UndefI32 = mlir::LLVM::UndefOp::create(B, Loc, I32Ty);
            SmallVector<mlir::Value, 3> Coords = {IdxVal, ZeroC, UndefI32};
            auto Atomic = ir::BufferAtomicOp::create(
                B, Loc, ValTy, Handle, ir::AtomicOpAttr::get(&Ctx, *ABinOp),
                Coords, SignlessVal);
            MU->getResult(0).replaceAllUsesWith(
                castFromSignless(B, Loc, Atomic.getResult(), OrigTy));
            MU->erase();
          }
        }
        Chain.erase();
        return Error::success();
      };
      SmallVector<mlir::Operation *, 8> ChainUsers(AO->getUsers().begin(),
                                                   AO->getUsers().end());
      for (mlir::Operation *CU : ChainUsers) {
        auto Chain = mlir::dyn_cast<mlir::spirv::AccessChainOp>(CU);
        if (!Chain)
          return makeError("unsupported access pattern on buffer '{0}'", Sym);
        if (auto Err = lowerBufferChain(Chain, {}))
          return Err;
      }
      AO.erase();
      continue;
    }

    // 3c. Constant buffers (Uniform Block) -> crucible.cbuffer_load.
    auto CBIt = CBufferGlobals.find(Sym);
    if (CBIt != CBufferGlobals.end()) {
      const BoundCBufferInfo &Info = CBIt->second;

      std::function<Error(mlir::spirv::AccessChainOp,
                          SmallVector<mlir::Value, 8>)>
          lowerCBufferChain = [&](mlir::spirv::AccessChainOp Chain,
                                  SmallVector<mlir::Value, 8> AllIndices) -> Error {
        for (mlir::Value Idx : Chain.getIndices())
          AllIndices.push_back(Idx);
        size_t StartIdx = Info.IsArray ? 1 : 0;

        mlir::Value ArrIdx = Info.IsArray ? AllIndices[0] : mlir::Value();
        SmallVector<mlir::Operation *, 8> MemUsers(Chain->getUsers().begin(),
                                                   Chain->getUsers().end());
        for (mlir::Operation *MU : MemUsers) {
          if (auto SubChain = mlir::dyn_cast<mlir::spirv::AccessChainOp>(MU)) {
            if (auto Err = lowerCBufferChain(SubChain, AllIndices))
              return Err;
            continue;
          }
          if (AllIndices.size() <= StartIdx)
            return makeError("unsupported access pattern on cbuffer '{0}'", Sym);
          auto Load = mlir::dyn_cast<mlir::spirv::LoadOp>(MU);
          if (!Load)
            return makeError("unsupported non-load user of cbuffer '{0}'", Sym);

          mlir::OpBuilder B(Load);
          mlir::Location Loc = Load.getLoc();
          mlir::Value Handle =
              getHandleForIndexAt(Info.ReflectIndex, B, Loc, ArrIdx);

          // Compute static + dynamic byte offset through nested structs/arrays/matrices/vectors.
          mlir::Type CurTy = Info.StructTy;
          uint32_t ByteOffset = 0;
          mlir::Value DynOffset;
          uint32_t CurMatrixStride = 16u;
          bool CurMatrixRowMajor = false;
          uint32_t NextVecElemStride = 0;
          auto addDynTerm = [&](mlir::Value IdxV, uint32_t Stride) {
            mlir::Value Idx32 = castToSignless(B, Loc, IdxV);
            auto StrideC = mlir::LLVM::ConstantOp::create(
                B, Loc, I32Ty, static_cast<int32_t>(Stride));
            mlir::Value Term =
                mlir::LLVM::MulOp::create(B, Loc, Idx32, StrideC);
            DynOffset = DynOffset
                            ? mlir::Value(mlir::LLVM::AddOp::create(
                                  B, Loc, DynOffset, Term))
                            : Term;
          };
          for (size_t IdxI = StartIdx; IdxI < AllIndices.size(); ++IdxI) {
            mlir::Value IdxV = AllIndices[IdxI];
            auto ConstIdx = getConstSpirvInt(IdxV);
            if (auto ST = mlir::dyn_cast<mlir::spirv::StructType>(CurTy)) {
              if (!ConstIdx)
                return makeError("dynamic struct index into cbuffer '{0}'", Sym);
              uint32_t I = static_cast<uint32_t>(*ConstIdx);
              ByteOffset += ST.getMemberOffset(I);
              CurMatrixStride = 16u;
              CurMatrixRowMajor = false;
              NextVecElemStride = 0;
              SmallVector<mlir::spirv::StructType::MemberDecorationInfo> MDecs;
              ST.getMemberDecorations(I, MDecs);
              for (const auto &D : MDecs) {
                if (D.decoration == mlir::spirv::Decoration::MatrixStride) {
                  if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(D.decorationValue))
                    CurMatrixStride = static_cast<uint32_t>(IA.getInt());
                } else if (D.decoration == mlir::spirv::Decoration::RowMajor) {
                  CurMatrixRowMajor = true;
                }
              }
              CurTy = ST.getElementType(I);
            } else if (auto Arr = mlir::dyn_cast<mlir::spirv::ArrayType>(CurTy)) {
              uint32_t Stride = Arr.getArrayStride();
              if (Stride == 0)
                Stride = spirvTypeBytes(Arr.getElementType());
              if (ConstIdx)
                ByteOffset += static_cast<uint32_t>(*ConstIdx) * Stride;
              else
                addDynTerm(IdxV, Stride);
              CurTy = Arr.getElementType();
            } else if (auto Mat = mlir::dyn_cast<mlir::spirv::MatrixType>(CurTy)) {
              auto ColVec = mlir::cast<mlir::VectorType>(Mat.getColumnType());
              uint32_t ElemBytes = scalarTypeBytes(ColVec.getElementType());
              uint32_t Step = CurMatrixRowMajor ? ElemBytes : CurMatrixStride;
              NextVecElemStride = CurMatrixRowMajor ? CurMatrixStride : ElemBytes;
              if (ConstIdx)
                ByteOffset += static_cast<uint32_t>(*ConstIdx) * Step;
              else
                addDynTerm(IdxV, Step);
              CurTy = ColVec;
            } else if (auto Vec = mlir::dyn_cast<mlir::VectorType>(CurTy)) {
              uint32_t Step = NextVecElemStride
                                  ? NextVecElemStride
                                  : scalarTypeBytes(Vec.getElementType());
              if (ConstIdx)
                ByteOffset += static_cast<uint32_t>(*ConstIdx) * Step;
              else
                addDynTerm(IdxV, Step);
              NextVecElemStride = 0;
              CurTy = Vec.getElementType();
            } else {
              return makeError("unsupported cbuffer index step on '{0}'", Sym);
            }
          }

          mlir::Type OrigTy = Load.getType();
          auto loadScalarAt = [&](mlir::Type ScalarTy,
                                  uint32_t OffBytes) -> mlir::Value {
            mlir::Type LoadScalarTy = ScalarTy;
            if (ScalarTy.isInteger(1))
              LoadScalarTy = I32Ty;
            uint32_t ElemBytes = scalarTypeBytes(LoadScalarTy);
            int64_t Lanes = 16 / ElemBytes;
            auto RetVecTy = mlir::VectorType::get({Lanes}, LoadScalarTy);
            mlir::Value RowVal, CompVal;
            if (!DynOffset) {
              uint32_t Row = OffBytes / 16u;
              uint32_t Comp = (OffBytes % 16u) / ElemBytes;
              RowVal = mlir::LLVM::ConstantOp::create(
                  B, Loc, I32Ty, static_cast<int32_t>(Row));
              CompVal = mlir::LLVM::ConstantOp::create(
                  B, Loc, I32Ty, static_cast<int32_t>(Comp));
            } else {
              mlir::Value TotalBytes = DynOffset;
              if (OffBytes != 0) {
                auto OffC = mlir::LLVM::ConstantOp::create(
                    B, Loc, I32Ty, static_cast<int32_t>(OffBytes));
                TotalBytes = mlir::LLVM::AddOp::create(B, Loc, DynOffset, OffC);
              }
              auto Shift4 = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 4);
              RowVal = mlir::LLVM::LShrOp::create(B, Loc, TotalBytes, Shift4);
              auto Mask15 = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 15);
              mlir::Value SubOff =
                  mlir::LLVM::AndOp::create(B, Loc, TotalBytes, Mask15);
              unsigned ElemShift = llvm::Log2_32(ElemBytes);
              if (ElemShift > 0) {
                auto ShiftE = mlir::LLVM::ConstantOp::create(
                    B, Loc, I32Ty, static_cast<int32_t>(ElemShift));
                CompVal = mlir::LLVM::LShrOp::create(B, Loc, SubOff, ShiftE);
              } else {
                CompVal = SubOff;
              }
            }
            auto CBLoad =
                ir::CBufferLoadOp::create(B, Loc, RetVecTy, Handle, RowVal);
            mlir::Value V = mlir::LLVM::ExtractElementOp::create(
                B, Loc, CBLoad.getResult(), CompVal);
            if (ScalarTy.isInteger(1)) {
              auto ZeroC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 0);
              V = mlir::LLVM::ICmpOp::create(B, Loc, mlir::LLVM::ICmpPredicate::ne,
                                             V, ZeroC);
            }
            return V;
          };

          mlir::Value ResultVal;
          if (auto MatTy = mlir::dyn_cast<mlir::spirv::MatrixType>(OrigTy)) {
            auto ColVec = mlir::cast<mlir::VectorType>(MatTy.getColumnType());
            mlir::Type ScalarTy = makeSignless(ColVec.getElementType(), B);
            mlir::Type SignlessColTy = makeSignless(ColVec, B);
            uint32_t ElemBytes = scalarTypeBytes(ScalarTy);
            uint32_t ColStep = CurMatrixRowMajor ? ElemBytes : CurMatrixStride;
            uint32_t RowStep = CurMatrixRowMajor ? CurMatrixStride : ElemBytes;
            SmallVector<mlir::Value, 4> Cols;
            for (uint32_t Col = 0; Col < MatTy.getNumColumns(); ++Col) {
              mlir::Value Vec = mlir::LLVM::UndefOp::create(B, Loc, SignlessColTy);
              for (int64_t Row = 0; Row < ColVec.getNumElements(); ++Row) {
                mlir::Value Lane = loadScalarAt(
                    ScalarTy, ByteOffset + Col * ColStep +
                                  static_cast<uint32_t>(Row) * RowStep);
                auto IdxC = mlir::LLVM::ConstantOp::create(
                    B, Loc, I32Ty, static_cast<int32_t>(Row));
                Vec = mlir::LLVM::InsertElementOp::create(B, Loc, Vec, Lane, IdxC);
              }
              Cols.push_back(castFromSignless(B, Loc, Vec, ColVec));
            }
            ResultVal = mlir::spirv::CompositeConstructOp::create(
                            B, Loc, OrigTy, Cols)
                            .getResult();
            Load.replaceAllUsesWith(ResultVal);
            Load.erase();
            continue;
          } else if (auto VecTy = mlir::dyn_cast<mlir::VectorType>(OrigTy)) {
            mlir::Type ScalarTy = makeSignless(VecTy.getElementType(), B);
            mlir::Type SignlessVecTy = makeSignless(VecTy, B);
            uint32_t ElemBytes =
                ScalarTy.isInteger(1) ? 4u : scalarTypeBytes(ScalarTy);
            uint32_t LaneStride = NextVecElemStride ? NextVecElemStride : ElemBytes;
            mlir::Value Vec = mlir::LLVM::UndefOp::create(B, Loc, SignlessVecTy);
            for (int64_t C = 0; C < VecTy.getNumElements(); ++C) {
              mlir::Value Lane = loadScalarAt(
                  ScalarTy, ByteOffset + static_cast<uint32_t>(C) * LaneStride);
              auto IdxC = mlir::LLVM::ConstantOp::create(
                  B, Loc, I32Ty, static_cast<int32_t>(C));
              Vec = mlir::LLVM::InsertElementOp::create(B, Loc, Vec, Lane, IdxC);
            }
            ResultVal = Vec;
          } else {
            mlir::Type ScalarTy = makeSignless(OrigTy, B);
            ResultVal = loadScalarAt(ScalarTy, ByteOffset);
          }
          Load.replaceAllUsesWith(castFromSignless(B, Loc, ResultVal, OrigTy));
          Load.erase();
        }
        Chain.erase();
        return Error::success();
      };
      SmallVector<mlir::Operation *, 8> ChainUsers(AO->getUsers().begin(),
                                                   AO->getUsers().end());
      for (mlir::Operation *CU : ChainUsers) {
        auto Chain = mlir::dyn_cast<mlir::spirv::AccessChainOp>(CU);
        if (!Chain)
          return makeError("unsupported access pattern on cbuffer '{0}'", Sym);
        if (auto Err = lowerCBufferChain(Chain, {}))
          return Err;
      }
      AO.erase();
      continue;
    }

    // 3c2. Workgroup (groupshared) globals -> addrspace(3) LLVM ops.
    auto WGIt = WorkgroupGlobals.find(Sym);
    if (WGIt != WorkgroupGlobals.end()) {
      const WorkgroupGlobalInfo &Info = WGIt->second;
      mlir::OpBuilder AOB(AO);
      auto Ptr3Ty = mlir::LLVM::LLVMPointerType::get(&Ctx, 3);
      auto LLVMAddr = mlir::LLVM::AddressOfOp::create(
          AOB, AO.getLoc(), Ptr3Ty,
          mlir::FlatSymbolRefAttr::get(&Ctx, Info.Name));

      auto rewritePtrUsers = [&](mlir::Value OldPtr, mlir::Value NewPtr) -> Error {
        SmallVector<mlir::Operation *, 8> Users(OldPtr.getUsers().begin(),
                                                OldPtr.getUsers().end());
        for (mlir::Operation *MU : Users) {
          if (auto Load = mlir::dyn_cast<mlir::spirv::LoadOp>(MU)) {
            mlir::OpBuilder B(Load);
            mlir::Location Loc = Load.getLoc();
            mlir::Type OrigTy = Load.getType();
            mlir::Type ValTy = makeSignless(OrigTy, B);
            auto LLVMLoad = mlir::LLVM::LoadOp::create(B, Loc, ValTy, NewPtr);
            Load.replaceAllUsesWith(
                castFromSignless(B, Loc, LLVMLoad.getResult(), OrigTy));
            Load.erase();
          } else if (auto Store = mlir::dyn_cast<mlir::spirv::StoreOp>(MU)) {
            mlir::OpBuilder B(Store);
            mlir::Location Loc = Store.getLoc();
            mlir::Value Val = castToSignless(B, Loc, Store.getValue());
            mlir::LLVM::StoreOp::create(B, Loc, Val, NewPtr);
            Store.erase();
          } else if (auto CX =
                         mlir::dyn_cast<mlir::spirv::AtomicCompareExchangeOp>(
                             MU)) {
            mlir::OpBuilder B(CX);
            mlir::Location Loc = CX.getLoc();
            mlir::Type OrigTy = CX.getValue().getType();
            mlir::Value Cmp = castToSignless(B, Loc, CX.getComparator());
            mlir::Value Val = castToSignless(B, Loc, CX.getValue());
            auto CmpXchg = mlir::LLVM::AtomicCmpXchgOp::create(
                B, Loc, NewPtr, Cmp, Val,
                mlir::LLVM::AtomicOrdering::monotonic,
                mlir::LLVM::AtomicOrdering::monotonic);
            auto OrigVal = mlir::LLVM::ExtractValueOp::create(
                B, Loc, CmpXchg.getResult(), ArrayRef<int64_t>{0});
            CX.replaceAllUsesWith(
                castFromSignless(B, Loc, OrigVal.getResult(), OrigTy));
            CX.erase();
          } else {
            std::optional<mlir::LLVM::AtomicBinOp> BinOp;
            mlir::Value AVal;
            if (auto A = mlir::dyn_cast<mlir::spirv::AtomicIAddOp>(MU)) {
              BinOp = mlir::LLVM::AtomicBinOp::add;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicISubOp>(MU)) {
              BinOp = mlir::LLVM::AtomicBinOp::sub;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicAndOp>(MU)) {
              BinOp = mlir::LLVM::AtomicBinOp::_and;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicOrOp>(MU)) {
              BinOp = mlir::LLVM::AtomicBinOp::_or;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicXorOp>(MU)) {
              BinOp = mlir::LLVM::AtomicBinOp::_xor;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicSMinOp>(MU)) {
              BinOp = mlir::LLVM::AtomicBinOp::min;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicSMaxOp>(MU)) {
              BinOp = mlir::LLVM::AtomicBinOp::max;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicUMinOp>(MU)) {
              BinOp = mlir::LLVM::AtomicBinOp::umin;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicUMaxOp>(MU)) {
              BinOp = mlir::LLVM::AtomicBinOp::umax;
              AVal = A.getValue();
            } else if (auto A =
                           mlir::dyn_cast<mlir::spirv::AtomicExchangeOp>(MU)) {
              BinOp = mlir::LLVM::AtomicBinOp::xchg;
              AVal = A.getValue();
            }
            if (!BinOp)
              return makeError("unsupported op '{0}' on workgroup global '{1}'",
                               MU->getName().getStringRef(), Sym);
            mlir::OpBuilder B(MU);
            mlir::Location Loc = MU->getLoc();
            mlir::Type OrigTy = AVal.getType();
            mlir::Value SignlessVal = castToSignless(B, Loc, AVal);
            if (SignlessVal.getType().isF32()) {
              SignlessVal =
                  mlir::LLVM::BitcastOp::create(B, Loc, I32Ty, SignlessVal);
              auto RMW = mlir::LLVM::AtomicRMWOp::create(
                  B, Loc, *BinOp, NewPtr, SignlessVal,
                  mlir::LLVM::AtomicOrdering::monotonic);
              auto FRes = mlir::LLVM::BitcastOp::create(B, Loc, OrigTy,
                                                        RMW.getResult());
              MU->getResult(0).replaceAllUsesWith(FRes.getResult());
              MU->erase();
            } else {
              auto RMW = mlir::LLVM::AtomicRMWOp::create(
                  B, Loc, *BinOp, NewPtr, SignlessVal,
                  mlir::LLVM::AtomicOrdering::monotonic);
              MU->getResult(0).replaceAllUsesWith(
                  castFromSignless(B, Loc, RMW.getResult(), OrigTy));
              MU->erase();
            }
          }
        }
        return Error::success();
      };

      SmallVector<mlir::Operation *, 8> AOUsers(AO->getUsers().begin(),
                                                AO->getUsers().end());
      for (mlir::Operation *U : AOUsers) {
        if (auto Chain = mlir::dyn_cast<mlir::spirv::AccessChainOp>(U)) {
          mlir::OpBuilder B(Chain);
          SmallVector<mlir::LLVM::GEPArg, 4> GEPArgs;
          GEPArgs.push_back(int32_t(0));
          for (mlir::Value IdxV : Chain.getIndices()) {
            if (auto C = getConstSpirvInt(IdxV))
              GEPArgs.push_back(static_cast<int32_t>(*C));
            else
              GEPArgs.push_back(castToSignless(B, Chain.getLoc(), IdxV));
          }
          auto GEP = mlir::LLVM::GEPOp::create(
              B, Chain.getLoc(), Ptr3Ty, Info.LLVMType, LLVMAddr.getResult(),
              GEPArgs);
          if (Error E = rewritePtrUsers(Chain.getResult(), GEP.getResult()))
            return std::move(E);
          Chain.erase();
        } else {
          if (Error E = rewritePtrUsers(AO.getResult(), LLVMAddr.getResult()))
            return std::move(E);
          break;
        }
      }
      AO.erase();
      continue;
    }

    // 3c3. Graphics stage I/O globals -> crucible.load_input / crucible.store_output.
    auto IOIt = StageIOGlobals.find(Sym);
    if (IOIt != StageIOGlobals.end()) {
      const StageIOInfo &Info = IOIt->second;
      SmallVector<mlir::Operation *, 8> Users(AO->getUsers().begin(),
                                              AO->getUsers().end());
      for (mlir::Operation *U : Users) {
        if (auto Load = mlir::dyn_cast<mlir::spirv::LoadOp>(U)) {
          mlir::OpBuilder B(Load);
          mlir::Location Loc = Load.getLoc();
          mlir::Type ValTy = Load.getType();

          auto loadRow = [&](mlir::Type RowTy, int32_t RowIdx) -> mlir::Value {
            auto RowC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, RowIdx);
            if (RowTy.isInteger(1)) {
              auto LI = ir::LoadInputOp::create(
                  B, Loc, I32Ty, static_cast<int32_t>(Info.ID), RowC,
                  static_cast<int8_t>(0), /*vertex_id=*/mlir::Value());
              auto ZeroC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 0);
              return mlir::LLVM::ICmpOp::create(B, Loc,
                                                mlir::LLVM::ICmpPredicate::ne,
                                                LI.getResult(), ZeroC)
                  .getResult();
            }
            if (auto VecTy = mlir::dyn_cast<mlir::VectorType>(RowTy)) {
              mlir::Type ElemTy = makeSignless(VecTy.getElementType(), B);
              SmallVector<mlir::Value, 4> Comps;
              for (unsigned C = 0; C < VecTy.getNumElements(); ++C) {
                auto LI = ir::LoadInputOp::create(
                    B, Loc, ElemTy, static_cast<int32_t>(Info.ID), RowC,
                    static_cast<int8_t>(C), /*vertex_id=*/mlir::Value());
                Comps.push_back(castFromSignless(B, Loc, LI.getResult(),
                                                 VecTy.getElementType()));
              }
              return mlir::spirv::CompositeConstructOp::create(B, Loc, RowTy,
                                                               Comps)
                  .getResult();
            }
            mlir::Type ElemTy = makeSignless(RowTy, B);
            auto LI = ir::LoadInputOp::create(
                B, Loc, ElemTy, static_cast<int32_t>(Info.ID), RowC,
                static_cast<int8_t>(0), /*vertex_id=*/mlir::Value());
            return castFromSignless(B, Loc, LI.getResult(), RowTy);
          };

          if (auto ArrTy = mlir::dyn_cast<mlir::spirv::ArrayType>(ValTy)) {
            SmallVector<mlir::Value, 4> Elems;
            for (unsigned R = 0; R < ArrTy.getNumElements(); ++R)
              Elems.push_back(
                  loadRow(ArrTy.getElementType(), static_cast<int32_t>(R)));
            auto Agg =
                mlir::spirv::CompositeConstructOp::create(B, Loc, ValTy, Elems);
            Load.replaceAllUsesWith(Agg.getResult());
          } else if (auto MatTy =
                         mlir::dyn_cast<mlir::spirv::MatrixType>(ValTy)) {
            SmallVector<mlir::Value, 4> Cols;
            for (unsigned R = 0; R < MatTy.getNumColumns(); ++R)
              Cols.push_back(
                  loadRow(MatTy.getColumnType(), static_cast<int32_t>(R)));
            auto Agg =
                mlir::spirv::CompositeConstructOp::create(B, Loc, ValTy, Cols);
            Load.replaceAllUsesWith(Agg.getResult());
          } else {
            Load.replaceAllUsesWith(loadRow(ValTy, 0));
          }
          Load.erase();
        } else if (auto Store = mlir::dyn_cast<mlir::spirv::StoreOp>(U)) {
          mlir::OpBuilder B(Store);
          mlir::Location Loc = Store.getLoc();
          mlir::Value OrigVal = Store.getValue();
          mlir::Type ValTy = OrigVal.getType();

          auto storeRow = [&](mlir::Value RowVal, int32_t RowIdx) {
            mlir::Value SVal = castToSignless(B, Loc, RowVal);
            auto RowC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, RowIdx);
            if (auto VecTy = mlir::dyn_cast<mlir::VectorType>(SVal.getType())) {
              for (unsigned C = 0; C < VecTy.getNumElements(); ++C) {
                auto IdxC = mlir::LLVM::ConstantOp::create(
                    B, Loc, I32Ty, static_cast<int32_t>(C));
                auto Lane = mlir::LLVM::ExtractElementOp::create(
                    B, Loc, SVal, IdxC);
                ir::StoreOutputOp::create(B, Loc, static_cast<int32_t>(Info.ID),
                                          RowC, static_cast<int8_t>(C),
                                          Lane.getResult());
              }
            } else {
              ir::StoreOutputOp::create(B, Loc, static_cast<int32_t>(Info.ID),
                                        RowC, static_cast<int8_t>(0), SVal);
            }
          };

          if (auto ArrTy = mlir::dyn_cast<mlir::spirv::ArrayType>(ValTy)) {
            for (unsigned R = 0; R < ArrTy.getNumElements(); ++R) {
              auto Elem = mlir::spirv::CompositeExtractOp::create(
                  B, Loc, OrigVal, ArrayRef<int32_t>{static_cast<int32_t>(R)});
              storeRow(Elem.getResult(), static_cast<int32_t>(R));
            }
          } else if (auto MatTy = mlir::dyn_cast<mlir::spirv::MatrixType>(ValTy)) {
            for (unsigned R = 0; R < MatTy.getNumColumns(); ++R) {
              auto Col = mlir::spirv::CompositeExtractOp::create(
                  B, Loc, OrigVal, ArrayRef<int32_t>{static_cast<int32_t>(R)});
              storeRow(Col.getResult(), static_cast<int32_t>(R));
            }
          } else {
            storeRow(OrigVal, 0);
          }
          Store.erase();
        } else if (auto AC = mlir::dyn_cast<mlir::spirv::AccessChainOp>(U)) {
          auto Indices = AC.getIndices();
          auto readConstIdx = [](mlir::Value V) -> int64_t {
            if (auto SC = V.getDefiningOp<mlir::spirv::ConstantOp>()) {
              if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(SC.getValue()))
                return IA.getInt();
            } else if (auto LC = V.getDefiningOp<mlir::LLVM::ConstantOp>()) {
              if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(LC.getValue()))
                return IA.getInt();
            }
            return -1;
          };
          int64_t RowIdx = 0;
          int64_t ColIdx = 0;
          if (Info.IsMultiRow) {
            if (Indices.empty() || Indices.size() > 2)
              return makeError(
                  "unsupported AccessChain index on stage I/O global '{0}'", Sym);
            RowIdx = readConstIdx(Indices[0]);
            if (RowIdx < 0)
              return makeError(
                  "unsupported AccessChain row index on stage I/O global '{0}'",
                  Sym);
            if (Indices.size() == 2) {
              ColIdx = readConstIdx(Indices[1]);
              if (ColIdx < 0 || ColIdx >= 4)
                return makeError(
                    "unsupported AccessChain col index on stage I/O global '{0}'",
                    Sym);
            }
          } else {
            if (Indices.size() != 1)
              return makeError(
                  "unsupported AccessChain index on stage I/O global '{0}'", Sym);
            ColIdx = readConstIdx(Indices[0]);
            if (ColIdx < 0 || ColIdx >= 4)
              return makeError(
                  "unsupported AccessChain index on stage I/O global '{0}'", Sym);
          }
          SmallVector<mlir::Operation *, 4> ACUsers(AC->getUsers().begin(),
                                                    AC->getUsers().end());
          for (mlir::Operation *AU : ACUsers) {
            if (auto Load = mlir::dyn_cast<mlir::spirv::LoadOp>(AU)) {
              mlir::OpBuilder B(Load);
              mlir::Location Loc = Load.getLoc();
              auto RowC = mlir::LLVM::ConstantOp::create(
                  B, Loc, I32Ty, static_cast<int32_t>(RowIdx));
              if (auto VecTy = mlir::dyn_cast<mlir::VectorType>(Load.getType())) {
                mlir::Type ElemTy = makeSignless(VecTy.getElementType(), B);
                SmallVector<mlir::Value, 4> Comps;
                for (unsigned C = 0; C < VecTy.getNumElements(); ++C) {
                  auto LI = ir::LoadInputOp::create(
                      B, Loc, ElemTy, static_cast<int32_t>(Info.ID), RowC,
                      static_cast<int8_t>(C), /*vertex_id=*/mlir::Value());
                  Comps.push_back(castFromSignless(B, Loc, LI.getResult(),
                                                   VecTy.getElementType()));
                }
                auto Vec = mlir::spirv::CompositeConstructOp::create(
                    B, Loc, Load.getType(), Comps);
                Load.replaceAllUsesWith(Vec.getResult());
              } else {
                mlir::Type ElemTy = makeSignless(Load.getType(), B);
                auto LI = ir::LoadInputOp::create(
                    B, Loc, ElemTy, static_cast<int32_t>(Info.ID), RowC,
                    static_cast<int8_t>(ColIdx), /*vertex_id=*/mlir::Value());
                Load.replaceAllUsesWith(
                    castFromSignless(B, Loc, LI.getResult(), Load.getType()));
              }
              Load.erase();
            } else if (auto Store = mlir::dyn_cast<mlir::spirv::StoreOp>(AU)) {
              mlir::OpBuilder B(Store);
              mlir::Location Loc = Store.getLoc();
              mlir::Value Val = castToSignless(B, Loc, Store.getValue());
              auto RowC = mlir::LLVM::ConstantOp::create(
                  B, Loc, I32Ty, static_cast<int32_t>(RowIdx));
              if (auto VecTy = mlir::dyn_cast<mlir::VectorType>(Val.getType())) {
                for (unsigned C = 0; C < VecTy.getNumElements(); ++C) {
                  auto IdxC = mlir::LLVM::ConstantOp::create(
                      B, Loc, I32Ty, static_cast<int32_t>(C));
                  auto Lane = mlir::LLVM::ExtractElementOp::create(
                      B, Loc, Val, IdxC);
                  ir::StoreOutputOp::create(B, Loc, static_cast<int32_t>(Info.ID),
                                            RowC, static_cast<int8_t>(C),
                                            Lane.getResult());
                }
              } else {
                ir::StoreOutputOp::create(B, Loc, static_cast<int32_t>(Info.ID),
                                          RowC, static_cast<int8_t>(ColIdx), Val);
              }
              Store.erase();
            } else {
              return makeError(
                  "unsupported user '{0}' of stage I/O AccessChain '{1}'",
                  AU->getName().getStringRef(), Sym);
            }
          }
          AC.erase();
        } else {
          return makeError("unsupported user '{0}' of stage I/O global '{1}'",
                           U->getName().getStringRef(), Sym);
        }
      }
      AO.erase();
      continue;
    }

    // 3c4. UniformConstant images (typed buffers & textures).
    auto ImgIt = ImageGlobals.find(Sym);
    if (ImgIt != ImageGlobals.end()) {
      const BoundImageInfo &Info = ImgIt->second;
      SmallVector<std::pair<mlir::spirv::LoadOp, mlir::Value>, 8> LoadEntries;
      SmallVector<mlir::spirv::AccessChainOp, 8> ChainsToErase;
      SmallVector<mlir::Operation *, 8> Users(AO->getUsers().begin(),
                                              AO->getUsers().end());
      for (mlir::Operation *LU : Users) {
        if (auto ImgLoad = mlir::dyn_cast<mlir::spirv::LoadOp>(LU)) {
          LoadEntries.push_back({ImgLoad, mlir::Value()});
        } else if (Info.IsArray) {
          auto Chain = mlir::dyn_cast<mlir::spirv::AccessChainOp>(LU);
          if (!Chain || Chain.getIndices().size() != 1)
            return makeError("unsupported user of image global '{0}'", Sym);
          mlir::Value ArrIdx = Chain.getIndices()[0];
          for (mlir::Operation *CU : Chain->getUsers()) {
            auto ImgLoad = mlir::dyn_cast<mlir::spirv::LoadOp>(CU);
            if (!ImgLoad)
              return makeError("unsupported user of image global '{0}'", Sym);
            LoadEntries.push_back({ImgLoad, ArrIdx});
          }
          ChainsToErase.push_back(Chain);
        } else {
          return makeError("unsupported user of image global '{0}'", Sym);
        }
      }
      for (auto &[ImgLoad, ArrIdx] : LoadEntries) {
        mlir::OpBuilder LoadB(ImgLoad);
        mlir::Value Handle = getHandleForIndexAt(Info.ReflectIndex, LoadB,
                                                 ImgLoad.getLoc(), ArrIdx);
        SmallVector<mlir::Operation *, 8> ImgUsers;
        SmallVector<mlir::spirv::ImageOp, 4> ImageOpsToErase;
        for (mlir::Operation *U : ImgLoad->getUsers()) {
          if (auto ImgOp = mlir::dyn_cast<mlir::spirv::ImageOp>(U)) {
            ImageOpsToErase.push_back(ImgOp);
            for (mlir::Operation *SubU : ImgOp->getUsers())
              ImgUsers.push_back(SubU);
          } else {
            ImgUsers.push_back(U);
          }
        }
        auto lowerSampledImageUser = [&](mlir::Operation *SU,
                                         mlir::Value SampHandle) -> Error {
          auto unpackFloatVec = [&](mlir::OpBuilder &B, mlir::Location Loc,
                                    mlir::Value V,
                                    SmallVectorImpl<mlir::Value> &Out) {
            auto F32Ty = B.getF32Type();
            if (auto CVec = mlir::dyn_cast<mlir::VectorType>(V.getType())) {
              for (unsigned C = 0; C < CVec.getNumElements() && C < 4; ++C) {
                Out.push_back(mlir::spirv::CompositeExtractOp::create(
                    B, Loc, F32Ty, V,
                    B.getI32ArrayAttr({static_cast<int32_t>(C)})));
              }
            } else {
              Out.push_back(V);
            }
          };
          auto unpackIntVec = [&](mlir::OpBuilder &B, mlir::Location Loc,
                                  mlir::Value V,
                                  SmallVectorImpl<mlir::Value> &Out) {
            mlir::Value SV = castToSignless(B, Loc, V);
            if (auto CVec = mlir::dyn_cast<mlir::VectorType>(SV.getType())) {
              for (unsigned C = 0; C < CVec.getNumElements() && C < 3; ++C) {
                auto IdxC = mlir::LLVM::ConstantOp::create(
                    B, Loc, I32Ty, static_cast<int32_t>(C));
                Out.push_back(
                    mlir::LLVM::ExtractElementOp::create(B, Loc, SV, IdxC));
              }
            } else {
              Out.push_back(SV);
            }
          };
          auto replaceVec4Result = [&](mlir::OpBuilder &B, mlir::Location Loc,
                                       mlir::Operation *TargetOp,
                                       mlir::Value Vec4Res) {
            mlir::Type ResTy = TargetOp->getResult(0).getType();
            auto ResVec = mlir::dyn_cast<mlir::VectorType>(ResTy);
            mlir::Value ResultVal;
            if (!ResVec) {
              auto ZeroC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 0);
              ResultVal =
                  mlir::LLVM::ExtractElementOp::create(B, Loc, Vec4Res, ZeroC);
            } else {
              mlir::Type SignlessVecTy = makeSignless(ResVec, B);
              unsigned N = static_cast<unsigned>(ResVec.getNumElements());
              mlir::Value Vec =
                  mlir::LLVM::UndefOp::create(B, Loc, SignlessVecTy);
              for (unsigned C = 0; C < N && C < 4; ++C) {
                auto IdxC = mlir::LLVM::ConstantOp::create(
                    B, Loc, I32Ty, static_cast<int32_t>(C));
                auto VC =
                    mlir::LLVM::ExtractElementOp::create(B, Loc, Vec4Res, IdxC);
                Vec = mlir::LLVM::InsertElementOp::create(B, Loc, Vec, VC,
                                                          IdxC);
              }
              ResultVal = Vec;
            }
            TargetOp->getResult(0).replaceAllUsesWith(
                castFromSignless(B, Loc, ResultVal, ResTy));
            TargetOp->erase();
          };
          auto lowerSample = [&](mlir::Operation *SampleOp, mlir::Value Coord,
                                 std::optional<mlir::spirv::ImageOperands> ImgOps,
                                 mlir::ValueRange OpArgs,
                                 bool IsExplicit) -> Error {
            mlir::OpBuilder B(SampleOp);
            mlir::Location Loc = SampleOp->getLoc();
            mlir::Type ResTy = SampleOp->getResult(0).getType();
            auto ResVec = mlir::dyn_cast<mlir::VectorType>(ResTy);
            mlir::Type ElemTy = makeSignless(
                ResVec ? ResVec.getElementType() : ResTy, B);
            auto RetVecTy = mlir::VectorType::get({4}, ElemTy);
            SmallVector<mlir::Value, 4> Coords;
            unpackFloatVec(B, Loc, Coord, Coords);
            ir::SampleMode Mode =
                IsExplicit ? ir::SampleMode::Level : ir::SampleMode::Implicit;
            mlir::Value LodOrBias;
            SmallVector<mlir::Value, 3> Ddx, Ddy, Offsets;
            mlir::Value ClampVal;
            uint32_t Mask = ImgOps ? static_cast<uint32_t>(*ImgOps) : 0u;
            size_t ArgIdx = 0;
            if (Mask & static_cast<uint32_t>(mlir::spirv::ImageOperands::Bias)) {
              if (ArgIdx < OpArgs.size()) {
                Mode = ir::SampleMode::Bias;
                LodOrBias = OpArgs[ArgIdx++];
              }
            }
            if (Mask & static_cast<uint32_t>(mlir::spirv::ImageOperands::Lod)) {
              if (ArgIdx < OpArgs.size()) {
                Mode = ir::SampleMode::Level;
                LodOrBias = OpArgs[ArgIdx++];
              }
            }
            if (Mask & static_cast<uint32_t>(mlir::spirv::ImageOperands::Grad)) {
              if (ArgIdx + 1 < OpArgs.size()) {
                Mode = ir::SampleMode::Grad;
                unpackFloatVec(B, Loc, OpArgs[ArgIdx++], Ddx);
                unpackFloatVec(B, Loc, OpArgs[ArgIdx++], Ddy);
              }
            }
            if (Mask &
                static_cast<uint32_t>(mlir::spirv::ImageOperands::ConstOffset)) {
              if (ArgIdx < OpArgs.size())
                unpackIntVec(B, Loc, OpArgs[ArgIdx++], Offsets);
            }
            if (Mask &
                static_cast<uint32_t>(mlir::spirv::ImageOperands::Offset)) {
              if (ArgIdx < OpArgs.size())
                unpackIntVec(B, Loc, OpArgs[ArgIdx++], Offsets);
            }
            if (Mask &
                static_cast<uint32_t>(mlir::spirv::ImageOperands::MinLod)) {
              if (ArgIdx < OpArgs.size())
                ClampVal = OpArgs[ArgIdx++];
            }
            auto SampOp = ir::TexSampleOp::create(
                B, Loc, RetVecTy, mlir::Type(), Handle, SampHandle,
                ir::SampleModeAttr::get(&Ctx, Mode), Coords, Offsets,
                LodOrBias, Ddx, Ddy, /*cmp_val=*/mlir::Value(), ClampVal);
            replaceVec4Result(B, Loc, SampleOp, SampOp.getResult());
            return Error::success();
          };
          if (auto Imp =
                  mlir::dyn_cast<mlir::spirv::ImageSampleImplicitLodOp>(SU)) {
            return lowerSample(Imp, Imp.getCoordinate(), Imp.getImageOperands(),
                               Imp.getOperandArguments(), /*IsExplicit=*/false);
          }
          if (auto Exp =
                  mlir::dyn_cast<mlir::spirv::ImageSampleExplicitLodOp>(SU)) {
            return lowerSample(Exp, Exp.getCoordinate(), Exp.getImageOperands(),
                               Exp.getOperandArguments(), /*IsExplicit=*/true);
          }
          if (auto DG = mlir::dyn_cast<mlir::spirv::ImageDrefGatherOp>(SU)) {
            mlir::OpBuilder B(DG);
            mlir::Location Loc = DG.getLoc();
            mlir::Type ResTy = DG.getType();
            auto ResVec = mlir::dyn_cast<mlir::VectorType>(ResTy);
            mlir::Type ElemTy = makeSignless(
                ResVec ? ResVec.getElementType() : ResTy, B);
            auto RetVecTy = mlir::VectorType::get({4}, ElemTy);
            SmallVector<mlir::Value, 4> Coords;
            unpackFloatVec(B, Loc, DG.getCoordinate(), Coords);
            SmallVector<mlir::Value, 3> Offsets;
            auto ImgOps = DG.getImageOperands();
            uint32_t Mask = ImgOps ? static_cast<uint32_t>(*ImgOps) : 0u;
            mlir::ValueRange OpArgs = DG.getOperandArguments();
            size_t ArgIdx = 0;
            if (Mask &
                static_cast<uint32_t>(mlir::spirv::ImageOperands::ConstOffset)) {
              if (ArgIdx < OpArgs.size())
                unpackIntVec(B, Loc, OpArgs[ArgIdx++], Offsets);
            }
            if (Mask &
                static_cast<uint32_t>(mlir::spirv::ImageOperands::Offset)) {
              if (ArgIdx < OpArgs.size())
                unpackIntVec(B, Loc, OpArgs[ArgIdx++], Offsets);
            }
            auto G = ir::TexGatherOp::create(
                B, Loc, RetVecTy, mlir::Type(), Handle, SampHandle, Coords,
                Offsets, DG.getDref(), B.getI32IntegerAttr(0));
            replaceVec4Result(B, Loc, DG, G.getResult());
            return Error::success();
          }
          return makeError("unsupported sampled-image op '{0}'",
                           SU->getName().getStringRef());
        };
        for (mlir::Operation *IU : ImgUsers) {
          if (auto IW = mlir::dyn_cast<mlir::spirv::ImageWriteOp>(IU)) {
            mlir::OpBuilder B(IW);
            mlir::Location Loc = IW.getLoc();
            mlir::Value Coord = castToSignless(B, Loc, IW.getCoordinate());
            mlir::Value Texel = castToSignless(B, Loc, IW.getTexel());
            auto TexVec = mlir::dyn_cast<mlir::VectorType>(Texel.getType());
            mlir::Type ElemTy =
                TexVec ? TexVec.getElementType() : Texel.getType();
            auto UndefVal = mlir::LLVM::UndefOp::create(B, Loc, ElemTy);
            auto UndefI32 = mlir::LLVM::UndefOp::create(B, Loc, I32Ty);

            SmallVector<mlir::Value, 4> Vals(4, UndefVal.getResult());
            uint8_t Mask = 1;
            if (!TexVec) {
              Vals[0] = Texel;
              Mask = 1;
            } else {
              unsigned N = static_cast<unsigned>(TexVec.getNumElements());
              Mask = static_cast<uint8_t>((1u << std::min(N, 4u)) - 1u);
              for (unsigned C = 0; C < N && C < 4; ++C) {
                auto IdxC = mlir::LLVM::ConstantOp::create(
                    B, Loc, I32Ty, static_cast<int32_t>(C));
                Vals[C] =
                    mlir::LLVM::ExtractElementOp::create(B, Loc, Texel, IdxC);
              }
            }

            if (Info.Kind == ResourceKind::TypedBuffer) {
              ir::BufferStoreOp::create(
                  B, Loc, Handle,
                  ir::BufferKindAttr::get(&Ctx, ir::BufferKind::Typed), Coord,
                  /*element_offset=*/mlir::Value(), Vals,
                  B.getI32IntegerAttr(static_cast<int32_t>(Mask)),
                  /*alignment=*/nullptr);
            } else {
              SmallVector<mlir::Value, 3> Coords;
              if (auto CVec =
                      mlir::dyn_cast<mlir::VectorType>(Coord.getType())) {
                for (unsigned C = 0; C < CVec.getNumElements() && C < 3; ++C) {
                  Coords.push_back(mlir::spirv::CompositeExtractOp::create(
                      B, Loc, I32Ty, Coord,
                      B.getI32ArrayAttr({static_cast<int32_t>(C)})));
                }
              } else {
                Coords.push_back(Coord);
              }
              ir::TexStoreOp::create(
                  B, Loc, Handle, Coords, Vals,
                  B.getI32IntegerAttr(static_cast<int32_t>(Mask)));
            }
            IW.erase();
          } else if (mlir::isa<mlir::spirv::ImageFetchOp,
                               mlir::spirv::ImageReadOp>(IU)) {
            mlir::OpBuilder B(IU);
            mlir::Location Loc = IU->getLoc();
            mlir::Value Coord =
                mlir::isa<mlir::spirv::ImageFetchOp>(IU)
                    ? mlir::cast<mlir::spirv::ImageFetchOp>(IU).getCoordinate()
                    : mlir::cast<mlir::spirv::ImageReadOp>(IU).getCoordinate();
            Coord = castToSignless(B, Loc, Coord);
            mlir::Type OrigResTy = IU->getResult(0).getType();
            auto ResVec = mlir::dyn_cast<mlir::VectorType>(OrigResTy);
            mlir::Type ElemTy = makeSignless(
                ResVec ? ResVec.getElementType() : OrigResTy, B);
            auto RetVecTy = mlir::VectorType::get({4}, ElemTy);
            auto ZeroI32 = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 0);

            mlir::Value LoadVec;
            if (Info.Kind == ResourceKind::TypedBuffer) {
              auto L = ir::BufferLoadOp::create(
                  B, Loc, RetVecTy, mlir::Type(), Handle,
                  ir::BufferKindAttr::get(&Ctx, ir::BufferKind::Typed), Coord,
                  /*element_offset=*/mlir::Value(), /*mask=*/nullptr,
                  /*alignment=*/nullptr);
              LoadVec = L.getResult();
            } else {
              SmallVector<mlir::Value, 3> Coords;
              if (auto CVec =
                      mlir::dyn_cast<mlir::VectorType>(Coord.getType())) {
                for (unsigned C = 0; C < CVec.getNumElements() && C < 3; ++C) {
                  auto IdxC = mlir::LLVM::ConstantOp::create(
                      B, Loc, I32Ty, static_cast<int32_t>(C));
                  Coords.push_back(mlir::LLVM::ExtractElementOp::create(
                      B, Loc, Coord, IdxC));
                }
              } else {
                Coords.push_back(Coord);
              }
              mlir::Value MipVal = ZeroI32;
              SmallVector<mlir::Value, 3> Offsets;
              if (auto FetchOp = mlir::dyn_cast<mlir::spirv::ImageFetchOp>(IU)) {
                auto IO = FetchOp.getImageOperands();
                uint32_t Mask = IO ? static_cast<uint32_t>(*IO) : 0u;
                mlir::ValueRange OpArgs = FetchOp.getOperandArguments();
                size_t ArgIdx = 0;
                if (Mask & static_cast<uint32_t>(mlir::spirv::ImageOperands::Lod)) {
                  if (ArgIdx < OpArgs.size())
                    MipVal = castToSignless(B, Loc, OpArgs[ArgIdx++]);
                }
                if (Mask & static_cast<uint32_t>(
                               mlir::spirv::ImageOperands::ConstOffset)) {
                  if (ArgIdx < OpArgs.size()) {
                    mlir::Value OV = castToSignless(B, Loc, OpArgs[ArgIdx++]);
                    if (auto OVec = mlir::dyn_cast<mlir::VectorType>(OV.getType())) {
                      for (unsigned C = 0; C < OVec.getNumElements() && C < 3; ++C) {
                        auto IdxC = mlir::LLVM::ConstantOp::create(
                            B, Loc, I32Ty, static_cast<int32_t>(C));
                        Offsets.push_back(
                            mlir::LLVM::ExtractElementOp::create(B, Loc, OV, IdxC));
                      }
                    } else {
                      Offsets.push_back(OV);
                    }
                  }
                }
                if (Mask &
                    static_cast<uint32_t>(mlir::spirv::ImageOperands::Offset)) {
                  if (ArgIdx < OpArgs.size()) {
                    mlir::Value OV = castToSignless(B, Loc, OpArgs[ArgIdx++]);
                    if (auto OVec = mlir::dyn_cast<mlir::VectorType>(OV.getType())) {
                      for (unsigned C = 0; C < OVec.getNumElements() && C < 3; ++C) {
                        auto IdxC = mlir::LLVM::ConstantOp::create(
                            B, Loc, I32Ty, static_cast<int32_t>(C));
                        Offsets.push_back(
                            mlir::LLVM::ExtractElementOp::create(B, Loc, OV, IdxC));
                      }
                    } else {
                      Offsets.push_back(OV);
                    }
                  }
                }
              }
              auto L = ir::TexLoadOp::create(B, Loc, RetVecTy, mlir::Type(),
                                             Handle, MipVal, Coords, Offsets);
              LoadVec = L.getResult();
            }
            mlir::Value ResultVal;
            if (!ResVec) {
              auto ZeroC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 0);
              ResultVal =
                  mlir::LLVM::ExtractElementOp::create(B, Loc, LoadVec, ZeroC);
            } else {
              mlir::Type SignlessVecTy = makeSignless(ResVec, B);
              unsigned N = static_cast<unsigned>(ResVec.getNumElements());
              mlir::Value Vec =
                  mlir::LLVM::UndefOp::create(B, Loc, SignlessVecTy);
              for (unsigned C = 0; C < N && C < 4; ++C) {
                auto IdxC = mlir::LLVM::ConstantOp::create(
                    B, Loc, I32Ty, static_cast<int32_t>(C));
                auto VC = mlir::LLVM::ExtractElementOp::create(B, Loc, LoadVec,
                                                               IdxC);
                Vec = mlir::LLVM::InsertElementOp::create(B, Loc, Vec, VC, IdxC);
              }
              ResultVal = Vec;
            }
            ResultVal = castFromSignless(B, Loc, ResultVal, OrigResTy);
            IU->getResult(0).replaceAllUsesWith(ResultVal);
            IU->erase();
          } else if (auto SI = mlir::dyn_cast<mlir::spirv::SampledImageOp>(IU)) {
            // Lower SampledImage + ImageSample{Implicit,Explicit}Lod / Gather.
            auto SampLoad =
                SI.getSampler().getDefiningOp<mlir::spirv::LoadOp>();
            mlir::spirv::AddressOfOp SampAO;
            mlir::spirv::AccessChainOp SampChain;
            mlir::Value SampArrIdx;
            if (SampLoad) {
              SampAO =
                  SampLoad.getPtr().getDefiningOp<mlir::spirv::AddressOfOp>();
              if (!SampAO) {
                SampChain =
                    SampLoad.getPtr().getDefiningOp<mlir::spirv::AccessChainOp>();
                if (SampChain && SampChain.getIndices().size() == 1) {
                  SampAO = SampChain.getBasePtr()
                               .getDefiningOp<mlir::spirv::AddressOfOp>();
                  SampArrIdx = SampChain.getIndices()[0];
                }
              }
            }
            if (!SampAO || !SamplerGlobals.contains(SampAO.getVariable()))
              return makeError("unresolved sampler in spirv.SampledImage");
            mlir::OpBuilder SIB(SI);
            mlir::Value SampHandle = getHandleForIndexAt(
                SamplerGlobals[SampAO.getVariable()], SIB, SI.getLoc(),
                SampArrIdx);
            SmallVector<mlir::Operation *, 4> SIUsers(SI->getUsers().begin(),
                                                      SI->getUsers().end());
            for (mlir::Operation *SU : SIUsers) {
              if (Error E = lowerSampledImageUser(SU, SampHandle))
                return std::move(E);
            }
            SI.erase();
            if (SampLoad && SampLoad->use_empty())
              SampLoad.erase();
            if (SampChain && SampChain->use_empty())
              SampChain.erase();
            if (SampAO && SampAO->use_empty()) {
              ErasedAddrOps.insert(SampAO);
              SampAO.erase();
            }
          } else if (Info.SamplerReflectIndex != SIZE_MAX &&
                     mlir::isa<mlir::spirv::ImageSampleImplicitLodOp,
                               mlir::spirv::ImageSampleExplicitLodOp,
                               mlir::spirv::ImageDrefGatherOp>(IU)) {
            mlir::Value SampHandle = getHandleForIndexAt(
                Info.SamplerReflectIndex, LoadB, ImgLoad.getLoc(), ArrIdx);
            if (Error E = lowerSampledImageUser(IU, SampHandle))
              return std::move(E);
          } else if (auto QS = mlir::dyn_cast<mlir::spirv::ImageQuerySizeOp>(IU)) {
            mlir::OpBuilder B(QS);
            mlir::Location Loc = QS.getLoc();
            auto RetVecTy = mlir::VectorType::get({4}, I32Ty);
            auto D = ir::TexGetDimensionsOp::create(B, Loc, RetVecTy, Handle,
                                                    /*mip=*/mlir::Value());
            mlir::Type OrigTy = QS.getType();
            auto VecTy = mlir::dyn_cast<mlir::VectorType>(OrigTy);
            mlir::Value ResultVal;
            if (!VecTy) {
              auto ZeroC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 0);
              ResultVal = mlir::LLVM::ExtractElementOp::create(
                  B, Loc, D.getResult(), ZeroC);
            } else {
              mlir::Type SignlessVecTy = makeSignless(VecTy, B);
              mlir::Value Vec =
                  mlir::LLVM::UndefOp::create(B, Loc, SignlessVecTy);
              for (int64_t C = 0; C < VecTy.getNumElements() && C < 4; ++C) {
                auto IdxC = mlir::LLVM::ConstantOp::create(
                    B, Loc, I32Ty, static_cast<int32_t>(C));
                auto VC = mlir::LLVM::ExtractElementOp::create(
                    B, Loc, D.getResult(), IdxC);
                Vec =
                    mlir::LLVM::InsertElementOp::create(B, Loc, Vec, VC, IdxC);
              }
              ResultVal = Vec;
            }
            QS.replaceAllUsesWith(castFromSignless(B, Loc, ResultVal, OrigTy));
            QS.erase();
          } else {
            return makeError("unsupported image op '{0}' on '{1}'",
                             IU->getName().getStringRef(), Sym);
          }
        }
        for (auto ImgOp : ImageOpsToErase) {
          if (ImgOp->use_empty())
            ImgOp.erase();
        }
        if (ImgLoad->use_empty())
          ImgLoad.erase();
      }
      for (auto Chain : ChainsToErase) {
        if (Chain->use_empty())
          Chain.erase();
      }
      if (AO->use_empty())
        AO.erase();
      continue;
    }
  }

  // Physical pointers carry GPU addresses, not host or thread-local pointers.
  auto IsPhysical = [](mlir::Type T) {
    auto P = mlir::dyn_cast<mlir::spirv::PointerType>(T);
    return P && P.getStorageClass() ==
                    mlir::spirv::StorageClass::PhysicalStorageBuffer;
  };
  auto AddressOf = [&](mlir::Value V) -> Expected<mlir::Value> {
    auto Cast = V.getDefiningOp<mlir::UnrealizedConversionCastOp>();
    if (!Cast || Cast.getInputs().size() != 1 ||
        !Cast.getInputs()[0].getType().isInteger(64))
      return makeError("unsupported physical pointer producer");
    return Cast.getInputs()[0];
  };
  SmallVector<mlir::Operation *> PhysicalOps;
  EntryFn.walk([&](mlir::Operation *Op) {
    if (llvm::any_of(Op->getOperandTypes(), IsPhysical) ||
        llvm::any_of(Op->getResultTypes(), IsPhysical))
      PhysicalOps.push_back(Op);
  });
  for (auto *Op : PhysicalOps) {
    if (mlir::isa<mlir::UnrealizedConversionCastOp>(Op))
      continue;
    mlir::OpBuilder B(Op);
    auto Loc = Op->getLoc();
    auto Wrap = [&](mlir::Value V, mlir::Type Ty) {
      return mlir::UnrealizedConversionCastOp::create(B, Loc, Ty, V)
          .getResult(0);
    };
    if (auto Chain = mlir::dyn_cast<mlir::spirv::AccessChainOp>(Op)) {
      auto Base = AddressOf(Chain.getBasePtr());
      if (!Base)
        return Base.takeError();
      mlir::Value Address = *Base;
      mlir::Type Ty =
          mlir::cast<mlir::spirv::PointerType>(Chain.getBasePtr().getType())
              .getPointeeType();
      for (mlir::Value Index : Chain.getIndices()) {
        if (auto ST = mlir::dyn_cast<mlir::spirv::StructType>(Ty)) {
          auto I = getConstSpirvInt(Index);
          if (!I || *I >= ST.getNumElements())
            return makeError("invalid physical struct index");
          auto Off = mlir::LLVM::ConstantOp::create(B, Loc, B.getI64Type(),
                                                    ST.getMemberOffset(*I));
          Address = mlir::LLVM::AddOp::create(B, Loc, Address, Off);
          Ty = ST.getElementType(*I);
        } else {
          uint32_t Stride = 0;
          if (auto Arr = mlir::dyn_cast<mlir::spirv::ArrayType>(Ty)) {
            Stride = Arr.getArrayStride();
            Ty = Arr.getElementType();
          } else if (auto Arr =
                         mlir::dyn_cast<mlir::spirv::RuntimeArrayType>(Ty)) {
            Stride = Arr.getArrayStride();
            Ty = Arr.getElementType();
          } else {
            return makeError("unsupported physical access-chain type");
          }
          if (!Stride)
            return makeError("physical arrays require explicit ArrayStride");
          auto Idx = castToSignless(B, Loc, Index);
          if (!Idx.getType().isInteger(32) && !Idx.getType().isInteger(64))
            return makeError("physical array indices require 32 or 64 bits");
          if (Idx.getType().isInteger(32))
            Idx = mlir::LLVM::SExtOp::create(B, Loc, B.getI64Type(), Idx);
          auto Step =
              mlir::LLVM::ConstantOp::create(B, Loc, B.getI64Type(), Stride);
          auto Offset = mlir::LLVM::MulOp::create(B, Loc, Idx, Step);
          Address = mlir::LLVM::AddOp::create(B, Loc, Address, Offset);
        }
      }
      Chain.replaceAllUsesWith(Wrap(Address, Chain.getType()));
    } else if (auto Select = mlir::dyn_cast<mlir::spirv::SelectOp>(Op)) {
      auto T = AddressOf(Select.getTrueValue());
      auto F = AddressOf(Select.getFalseValue());
      if (!T)
        return T.takeError();
      if (!F)
        return F.takeError();
      auto Value =
          mlir::LLVM::SelectOp::create(B, Loc, Select.getCondition(), *T, *F);
      Select.replaceAllUsesWith(Wrap(Value, Select.getType()));
    } else if (mlir::isa<mlir::spirv::LoadOp, mlir::spirv::StoreOp>(Op)) {
      auto Load = mlir::dyn_cast<mlir::spirv::LoadOp>(Op);
      auto Store = mlir::dyn_cast<mlir::spirv::StoreOp>(Op);
      auto Ptr = Load ? Load.getPtr() : Store.getPtr();
      auto Address = AddressOf(Ptr);
      if (!Address)
        return Address.takeError();
      auto Ty = Load ? Load.getType() : Store.getValue().getType();
      if (!mlir::isa<mlir::IntegerType, mlir::FloatType>(Ty))
        return makeError(
            "physical memory accesses currently require scalar values");
      auto Access = Load ? Load.getMemoryAccess() : Store.getMemoryAccess();
      auto Alignment =
          Load ? Load.getAlignmentAttr() : Store.getAlignmentAttr();
      if (!Access || *Access != mlir::spirv::MemoryAccess::Aligned ||
          !Alignment || Alignment.getInt() < scalarTypeBytes(Ty))
        return makeError(
            "physical scalar accesses require natural explicit alignment");
      auto Pointer = mlir::LLVM::IntToPtrOp::create(
          B, Loc, mlir::LLVM::LLVMPointerType::get(&Ctx, 1), *Address);
      Pointer->setAttr("crucible.physical_type",
                       mlir::TypeAttr::get(makeSignless(Ty, B)));
      if (Load) {
        auto Value =
            mlir::LLVM::LoadOp::create(B, Loc, makeSignless(Ty, B), Pointer);
        Load.replaceAllUsesWith(castFromSignless(B, Loc, Value, Ty));
      } else {
        mlir::LLVM::StoreOp::create(
            B, Loc, castToSignless(B, Loc, Store.getValue()), Pointer);
      }
    } else {
      return makeError("unsupported physical pointer operation '{0}'",
                       Op->getName().getStringRef());
    }
    Op->erase();
  }
  SmallVector<mlir::UnrealizedConversionCastOp> DeadAddressCasts;
  EntryFn.walk([&](mlir::UnrealizedConversionCastOp Cast) {
    if (Cast->use_empty() && llvm::any_of(Cast->getResultTypes(), IsPhysical))
      DeadAddressCasts.push_back(Cast);
  });
  for (auto Cast : DeadAddressCasts)
    Cast.erase();

  // 3d. Lower spirv.ControlBarrier & spirv.MemoryBarrier -> Layer 1 gpu.barrier.
  SmallVector<mlir::spirv::ControlBarrierOp, 4> Barriers;
  EntryFn.walk(
      [&](mlir::spirv::ControlBarrierOp CB) { Barriers.push_back(CB); });
  for (auto CB : Barriers) {
    mlir::OpBuilder B(CB);
    auto GPU = mlir::gpu::BarrierOp::create(B, CB.getLoc());
    auto Sem = CB.getMemorySemantics();
    if (mlir::spirv::bitEnumContainsAny(
            Sem, mlir::spirv::MemorySemantics::UniformMemory |
                     mlir::spirv::MemorySemantics::ImageMemory)) {
      GPU->setAttr("crucible.device_fence", B.getUnitAttr());
    }
    CB.erase();
  }
  SmallVector<mlir::spirv::MemoryBarrierOp, 4> MemBarriers;
  EntryFn.walk(
      [&](mlir::spirv::MemoryBarrierOp MB) { MemBarriers.push_back(MB); });
  for (auto MB : MemBarriers) {
    mlir::OpBuilder B(MB);
    auto GPU = mlir::gpu::BarrierOp::create(B, MB.getLoc());
    auto Sem = MB.getMemorySemantics();
    if (mlir::spirv::bitEnumContainsAny(
            Sem, mlir::spirv::MemorySemantics::UniformMemory |
                     mlir::spirv::MemorySemantics::ImageMemory)) {
      GPU->setAttr("crucible.device_fence", B.getUnitAttr());
    }
    if (!mlir::spirv::bitEnumContainsAny(
            Sem, mlir::spirv::MemorySemantics::WorkgroupMemory)) {
      GPU->setAttr("crucible.no_tgsm_fence", B.getUnitAttr());
    }
    MB.erase();
  }

  // 3e. Lower spirv.GroupNonUniform* subgroup ops -> Layer 1 gpu.* ops.
  auto lowerGroupOp = [&](mlir::Operation *Op) {
    mlir::OpBuilder B(Op);
    mlir::Location Loc = Op->getLoc();

    auto emitReduce = [&](mlir::Value RawVal,
                          mlir::gpu::AllReduceOperation ReduceOp,
                          mlir::spirv::GroupOperation GroupOp,
                          mlir::Value ClusterSize = mlir::Value()) {
      mlir::Type OrigTy = RawVal.getType();
      mlir::Value Val = castToSignless(B, Loc, RawVal);
      if (GroupOp == mlir::spirv::GroupOperation::PartitionedExclusiveScanNV &&
          ClusterSize) {
        std::optional<ir::WaveMultiPrefixOpKind> MPKind;
        switch (ReduceOp) {
        case mlir::gpu::AllReduceOperation::ADD:
          MPKind = ir::WaveMultiPrefixOpKind::Sum;
          break;
        case mlir::gpu::AllReduceOperation::MUL:
          MPKind = ir::WaveMultiPrefixOpKind::Product;
          break;
        case mlir::gpu::AllReduceOperation::AND:
          MPKind = ir::WaveMultiPrefixOpKind::And;
          break;
        case mlir::gpu::AllReduceOperation::OR:
          MPKind = ir::WaveMultiPrefixOpKind::Or;
          break;
        case mlir::gpu::AllReduceOperation::XOR:
          MPKind = ir::WaveMultiPrefixOpKind::Xor;
          break;
        default:
          break;
        }
        if (MPKind) {
          mlir::Value MaskVec = castToSignless(B, Loc, ClusterSize);
          SmallVector<mlir::Value, 4> Mask;
          for (int32_t I = 0; I < 4; ++I) {
            auto Idx = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, I);
            Mask.push_back(
                mlir::LLVM::ExtractElementOp::create(B, Loc, MaskVec, Idx));
          }
          mlir::Type OrigElemTy = OrigTy;
          if (auto VT = mlir::dyn_cast<mlir::VectorType>(OrigTy))
            OrigElemTy = VT.getElementType();
          bool IsUnsigned = false;
          if (auto IT = mlir::dyn_cast<mlir::IntegerType>(OrigElemTy))
            IsUnsigned = IT.isUnsigned();
          mlir::Value Res;
          if (auto VT = mlir::dyn_cast<mlir::VectorType>(Val.getType())) {
            mlir::Value Acc = mlir::LLVM::UndefOp::create(B, Loc, VT);
            for (int64_t I = 0; I < VT.getNumElements(); ++I) {
              auto Idx = mlir::LLVM::ConstantOp::create(
                  B, Loc, I32Ty, static_cast<int32_t>(I));
              mlir::Value Lane =
                  mlir::LLVM::ExtractElementOp::create(B, Loc, Val, Idx);
              auto W = ir::WaveMultiPrefixOp::create(
                  B, Loc, VT.getElementType(), Lane, Mask, *MPKind, IsUnsigned);
              Acc = mlir::LLVM::InsertElementOp::create(B, Loc, Acc,
                                                        W.getResult(), Idx);
            }
            Res = Acc;
          } else {
            auto W = ir::WaveMultiPrefixOp::create(B, Loc, Val.getType(), Val,
                                                   Mask, *MPKind, IsUnsigned);
            Res = W.getResult();
          }
          Op->getResult(0).replaceAllUsesWith(
              castFromSignless(B, Loc, Res, OrigTy));
          Op->erase();
          return;
        }
      }
      auto Red = mlir::gpu::SubgroupReduceOp::create(B, Loc, Val, ReduceOp,
                                                     /*uniform=*/false);
      if (GroupOp == mlir::spirv::GroupOperation::ExclusiveScan)
        Red->setAttr("crucible.scan_kind", B.getStringAttr("exclusive"));
      else if (GroupOp == mlir::spirv::GroupOperation::InclusiveScan)
        Red->setAttr("crucible.scan_kind", B.getStringAttr("inclusive"));
      mlir::Value Res = castFromSignless(B, Loc, Red.getResult(), OrigTy);
      Op->getResult(0).replaceAllUsesWith(Res);
      Op->erase();
    };

    if (auto G = mlir::dyn_cast<mlir::spirv::GroupNonUniformIAddOp>(Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::ADD,
                 G.getGroupOperation(), G.getClusterSize());
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformFAddOp>(Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::ADD,
                 G.getGroupOperation(), G.getClusterSize());
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformIMulOp>(Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::MUL,
                 G.getGroupOperation(), G.getClusterSize());
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformFMulOp>(Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::MUL,
                 G.getGroupOperation(), G.getClusterSize());
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformSMinOp>(Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::MINSI,
                 G.getGroupOperation(), G.getClusterSize());
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformUMinOp>(Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::MINUI,
                 G.getGroupOperation(), G.getClusterSize());
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformFMinOp>(Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::MINNUMF,
                 G.getGroupOperation(), G.getClusterSize());
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformSMaxOp>(Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::MAXSI,
                 G.getGroupOperation(), G.getClusterSize());
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformUMaxOp>(Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::MAXUI,
                 G.getGroupOperation(), G.getClusterSize());
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformFMaxOp>(Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::MAXNUMF,
                 G.getGroupOperation(), G.getClusterSize());
    } else if (auto G = mlir::dyn_cast<mlir::spirv::GroupNonUniformBitwiseAndOp>(
                   Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::AND,
                 G.getGroupOperation(), G.getClusterSize());
    } else if (auto G = mlir::dyn_cast<mlir::spirv::GroupNonUniformBitwiseOrOp>(
                   Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::OR,
                 G.getGroupOperation(), G.getClusterSize());
    } else if (auto G = mlir::dyn_cast<mlir::spirv::GroupNonUniformBitwiseXorOp>(
                   Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::XOR,
                 G.getGroupOperation(), G.getClusterSize());
    } else if (auto G = mlir::dyn_cast<
                   mlir::spirv::GroupNonUniformBroadcastFirstOp>(Op)) {
      mlir::Type OrigTy = G.getType();
      mlir::Value Val = castToSignless(B, Loc, G.getValue());
      auto BC = mlir::gpu::SubgroupBroadcastOp::create(
          B, Loc, Val, /*subgroupId=*/nullptr,
          mlir::gpu::BroadcastType::first_active_lane);
      G.replaceAllUsesWith(castFromSignless(B, Loc, BC.getResult(), OrigTy));
      G.erase();
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformBroadcastOp>(Op)) {
      mlir::Type OrigTy = G.getType();
      mlir::Value Val = castToSignless(B, Loc, G.getValue());
      mlir::Value Id = castToSignless(B, Loc, G.getId());
      auto BC = mlir::gpu::SubgroupBroadcastOp::create(
          B, Loc, Val, Id, mlir::gpu::BroadcastType::specific_lane);
      G.replaceAllUsesWith(castFromSignless(B, Loc, BC.getResult(), OrigTy));
      G.erase();
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformShuffleOp>(Op)) {
      mlir::Type OrigTy = G.getType();
      mlir::Value Val = castToSignless(B, Loc, G.getValue());
      mlir::Value Id = castToSignless(B, Loc, G.getId());
      auto Width = mlir::arith::ConstantIntOp::create(B, Loc, 32, 32);
      auto Shuf = mlir::gpu::ShuffleOp::create(
          B, Loc, Val, Id, Width, mlir::gpu::ShuffleMode::IDX);
      G.replaceAllUsesWith(
          castFromSignless(B, Loc, Shuf.getShuffleResult(), OrigTy));
      G.erase();
    } else if (auto G = mlir::dyn_cast<mlir::spirv::GroupNonUniformShuffleXorOp>(
                   Op)) {
      mlir::Type OrigTy = G.getType();
      mlir::Value Val = castToSignless(B, Loc, G.getValue());
      mlir::Value Mask = castToSignless(B, Loc, G.getMask());
      auto Width = mlir::arith::ConstantIntOp::create(B, Loc, 32, 32);
      auto Shuf = mlir::gpu::ShuffleOp::create(
          B, Loc, Val, Mask, Width, mlir::gpu::ShuffleMode::XOR);
      G.replaceAllUsesWith(
          castFromSignless(B, Loc, Shuf.getShuffleResult(), OrigTy));
      G.erase();
    } else if (auto G = mlir::dyn_cast<mlir::spirv::GroupNonUniformShuffleUpOp>(
                   Op)) {
      mlir::Type OrigTy = G.getType();
      mlir::Value Val = castToSignless(B, Loc, G.getValue());
      mlir::Value Delta = castToSignless(B, Loc, G.getDelta());
      auto Width = mlir::arith::ConstantIntOp::create(B, Loc, 32, 32);
      auto Shuf = mlir::gpu::ShuffleOp::create(
          B, Loc, Val, Delta, Width, mlir::gpu::ShuffleMode::UP);
      G.replaceAllUsesWith(
          castFromSignless(B, Loc, Shuf.getShuffleResult(), OrigTy));
      G.erase();
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformShuffleDownOp>(
                       Op)) {
      mlir::Type OrigTy = G.getType();
      mlir::Value Val = castToSignless(B, Loc, G.getValue());
      mlir::Value Delta = castToSignless(B, Loc, G.getDelta());
      auto Width = mlir::arith::ConstantIntOp::create(B, Loc, 32, 32);
      auto Shuf = mlir::gpu::ShuffleOp::create(
          B, Loc, Val, Delta, Width, mlir::gpu::ShuffleMode::DOWN);
      G.replaceAllUsesWith(
          castFromSignless(B, Loc, Shuf.getShuffleResult(), OrigTy));
      G.erase();
    } else if (auto G = mlir::dyn_cast<mlir::spirv::GroupNonUniformAllOp>(Op)) {
      auto W = ir::WaveVoteOp::create(
          B, Loc, I1Ty, G.getPredicate(),
          ir::WaveVoteKindAttr::get(B.getContext(), ir::WaveVoteKind::All));
      G.replaceAllUsesWith(W.getResult());
      G.erase();
    } else if (auto G = mlir::dyn_cast<mlir::spirv::GroupNonUniformAnyOp>(Op)) {
      auto W = ir::WaveVoteOp::create(
          B, Loc, I1Ty, G.getPredicate(),
          ir::WaveVoteKindAttr::get(B.getContext(), ir::WaveVoteKind::Any));
      G.replaceAllUsesWith(W.getResult());
      G.erase();
    } else if (auto G = mlir::dyn_cast<mlir::spirv::GroupNonUniformAllEqualOp>(
                   Op)) {
      mlir::Value Val = castToSignless(B, Loc, G.getValue());
      auto W = ir::WaveVoteOp::create(
          B, Loc, I1Ty, Val,
          ir::WaveVoteKindAttr::get(B.getContext(), ir::WaveVoteKind::AllEqual));
      G.replaceAllUsesWith(W.getResult());
      G.erase();
    } else if (auto G = mlir::dyn_cast<mlir::spirv::GroupNonUniformQuadSwapOp>(
                   Op)) {
      mlir::Type OrigTy = G.getType();
      mlir::Value Val = castToSignless(B, Loc, G.getValue());
      uint32_t Dir = static_cast<uint32_t>(G.getDirection());
      ir::QuadOpKind Kind = ir::QuadOpKind::ReadAcrossX;
      if (Dir == 1)
        Kind = ir::QuadOpKind::ReadAcrossY;
      else if (Dir == 2)
        Kind = ir::QuadOpKind::ReadAcrossDiagonal;
      auto KindAttr = ir::QuadOpKindAttr::get(B.getContext(), Kind);
      mlir::Value Res;
      if (auto VT = mlir::dyn_cast<mlir::VectorType>(Val.getType())) {
        mlir::Value Acc = mlir::LLVM::PoisonOp::create(B, Loc, VT);
        for (int64_t I = 0; I < VT.getNumElements(); ++I) {
          mlir::Value Idx = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, I);
          mlir::Value Lane =
              mlir::LLVM::ExtractElementOp::create(B, Loc, Val, Idx);
          auto Q = ir::QuadOp::create(B, Loc, VT.getElementType(), Lane,
                                      KindAttr);
          Acc = mlir::LLVM::InsertElementOp::create(B, Loc, Acc, Q.getResult(),
                                                    Idx);
        }
        Res = Acc;
      } else {
        auto Q = ir::QuadOp::create(B, Loc, Val.getType(), Val, KindAttr);
        Res = Q.getResult();
      }
      G.replaceAllUsesWith(castFromSignless(B, Loc, Res, OrigTy));
      G.erase();
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformElectOp>(Op)) {
      auto One = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 1);
      auto Zero = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 0);
      auto Scan = mlir::gpu::SubgroupReduceOp::create(
          B, Loc, One.getResult(), mlir::gpu::AllReduceOperation::ADD, false);
      Scan->setAttr("crucible.scan_kind", B.getStringAttr("exclusive"));
      auto Cmp = mlir::LLVM::ICmpOp::create(
          B, Loc, mlir::LLVM::ICmpPredicate::eq, Scan.getResult(),
          Zero.getResult());
      G.replaceAllUsesWith(Cmp.getResult());
      G.erase();
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformBallotOp>(Op)) {
      auto Bal = mlir::gpu::BallotOp::create(B, Loc, I32Ty, G.getPredicate());
      SmallVector<mlir::Operation *, 4> Users(G->getUsers().begin(),
                                              G->getUsers().end());
      for (mlir::Operation *U : Users) {
        if (auto Ext = mlir::dyn_cast<mlir::spirv::CompositeExtractOp>(U)) {
          if (Ext.getIndices().size() == 1) {
            int64_t Idx =
                mlir::cast<mlir::IntegerAttr>(Ext.getIndices()[0]).getInt();
            mlir::OpBuilder EB(Ext);
            mlir::Value Rep =
                Idx == 0
                    ? Bal.getResult()
                    : mlir::LLVM::ConstantOp::create(EB, Ext.getLoc(), I32Ty, 0)
                          .getResult();
            Ext.replaceAllUsesWith(Rep);
            Ext.erase();
          }
        } else if (auto BC = mlir::dyn_cast<
                       mlir::spirv::GroupNonUniformBallotBitCountOp>(U)) {
          mlir::OpBuilder BCB(BC);
          mlir::Location BLoc = BC.getLoc();
          mlir::Value Masked = Bal.getResult();
          if (BC.getGroupOperation() ==
              mlir::spirv::GroupOperation::ExclusiveScan) {
            auto LaneIdx =
                mlir::gpu::LaneIdOp::create(BCB, BLoc, /*upper_bound=*/nullptr);
            auto Lane =
                mlir::arith::IndexCastUIOp::create(BCB, BLoc, I32Ty, LaneIdx);
            auto One = mlir::LLVM::ConstantOp::create(BCB, BLoc, I32Ty, 1);
            auto Shl = mlir::LLVM::ShlOp::create(BCB, BLoc, One, Lane);
            auto Mask = mlir::LLVM::SubOp::create(BCB, BLoc, Shl, One);
            Masked = mlir::LLVM::AndOp::create(BCB, BLoc, Masked, Mask);
          }
          auto Pop = mlir::math::CtPopOp::create(BCB, BLoc, Masked);
          BC.replaceAllUsesWith(Pop.getResult());
          BC.erase();
        }
      }
      if (G->use_empty())
        G.erase();
    }
  };

  SmallVector<mlir::Operation *, 16> SubgroupOps;
  EntryFn.walk([&](mlir::Operation *Op) {
    StringRef Name = Op->getName().getStringRef();
    if (Name.starts_with("spirv.GroupNonUniform") &&
        Name != "spirv.GroupNonUniformBallotBitCount")
      SubgroupOps.push_back(Op);
  });
  for (mlir::Operation *Op : SubgroupOps)
    lowerGroupOp(Op);

  // 3f. Lower spirv.VectorTimesScalar and spirv.CompositeConstruct on vectors
  // to LLVM dialect ops (ConvertSPIRVToLLVM does not cover these directly).
  SmallVector<mlir::spirv::VectorTimesScalarOp, 8> VTSOps;
  EntryFn.walk([&](mlir::spirv::VectorTimesScalarOp VTS) { VTSOps.push_back(VTS); });
  for (auto VTS : VTSOps) {
    mlir::OpBuilder B(VTS);
    mlir::Location Loc = VTS.getLoc();
    auto VecTy = mlir::cast<mlir::VectorType>(VTS.getType());
    mlir::Value Splat = mlir::LLVM::UndefOp::create(B, Loc, VecTy);
    for (unsigned C = 0; C < VecTy.getNumElements(); ++C) {
      auto IdxC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty,
                                                 static_cast<int32_t>(C));
      Splat = mlir::LLVM::InsertElementOp::create(B, Loc, Splat,
                                                  VTS.getScalar(), IdxC);
    }
    auto Mul = mlir::LLVM::FMulOp::create(B, Loc, VTS.getVector(), Splat);
    VTS.replaceAllUsesWith(Mul.getResult());
    VTS.erase();
  }

  SmallVector<mlir::spirv::CompositeConstructOp, 8> CCOps;
  EntryFn.walk(
      [&](mlir::spirv::CompositeConstructOp CC) { CCOps.push_back(CC); });
  for (auto CC : CCOps) {
    auto OrigVecTy = mlir::dyn_cast<mlir::VectorType>(CC.getType());
    if (!OrigVecTy)
      continue;
    mlir::OpBuilder B(CC);
    mlir::Location Loc = CC.getLoc();
    mlir::Type SignlessVecTy = makeSignless(OrigVecTy, B);
    mlir::Value Vec = mlir::LLVM::UndefOp::create(B, Loc, SignlessVecTy);
    unsigned Lane = 0;
    for (mlir::Value RawOpnd : CC.getConstituents()) {
      mlir::Value Opnd = castToSignless(B, Loc, RawOpnd);
      if (auto SubVec = mlir::dyn_cast<mlir::VectorType>(Opnd.getType())) {
        for (unsigned K = 0; K < SubVec.getNumElements(); ++K) {
          auto KIdx = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty,
                                                     static_cast<int32_t>(K));
          auto Elem = mlir::LLVM::ExtractElementOp::create(B, Loc, Opnd, KIdx);
          auto DstIdx = mlir::LLVM::ConstantOp::create(
              B, Loc, I32Ty, static_cast<int32_t>(Lane++));
          Vec = mlir::LLVM::InsertElementOp::create(B, Loc, Vec, Elem, DstIdx);
        }
      } else {
        auto DstIdx = mlir::LLVM::ConstantOp::create(
            B, Loc, I32Ty, static_cast<int32_t>(Lane++));
        Vec = mlir::LLVM::InsertElementOp::create(B, Loc, Vec, Opnd, DstIdx);
      }
    }
    Vec = castFromSignless(B, Loc, Vec, OrigVecTy);
    CC.replaceAllUsesWith(Vec);
    CC.erase();
  }

  // 3g. Lower spirv.Kill -> crucible.discard(true).
  SmallVector<mlir::spirv::KillOp, 4> KillOps;
  SpvModule.walk([&](mlir::spirv::KillOp K) { KillOps.push_back(K); });
  for (auto K : KillOps) {
    mlir::OpBuilder B(K);
    mlir::Location Loc = K.getLoc();
    auto TrueC = mlir::LLVM::ConstantOp::create(B, Loc, I1Ty, true);
    ir::DiscardOp::create(B, Loc, TrueC);
    mlir::spirv::ReturnOp::create(B, Loc);
    K.erase();
  }

  // 3g2. Lower spirv.All, spirv.Any, spirv.IsNan, spirv.IsInf to LLVM ops.
  SmallVector<mlir::Operation *, 8> LogicOps;
  EntryFn.walk([&](mlir::Operation *Op) {
    if (mlir::isa<mlir::spirv::AllOp, mlir::spirv::AnyOp, mlir::spirv::IsNanOp,
                  mlir::spirv::IsInfOp>(Op))
      LogicOps.push_back(Op);
  });
  for (mlir::Operation *Op : LogicOps) {
    mlir::OpBuilder B(Op);
    mlir::Location Loc = Op->getLoc();
    if (auto All = mlir::dyn_cast<mlir::spirv::AllOp>(Op)) {
      mlir::Value Vec = All.getVector();
      auto VTy = mlir::cast<mlir::VectorType>(Vec.getType());
      mlir::Value Acc;
      for (unsigned C = 0; C < VTy.getNumElements(); ++C) {
        auto IdxC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty,
                                                   static_cast<int32_t>(C));
        mlir::Value Lane =
            mlir::LLVM::ExtractElementOp::create(B, Loc, Vec, IdxC);
        Acc = Acc ? mlir::Value(mlir::LLVM::AndOp::create(B, Loc, Acc, Lane))
                  : Lane;
      }
      All.replaceAllUsesWith(Acc);
      All.erase();
    } else if (auto Any = mlir::dyn_cast<mlir::spirv::AnyOp>(Op)) {
      mlir::Value Vec = Any.getVector();
      auto VTy = mlir::cast<mlir::VectorType>(Vec.getType());
      mlir::Value Acc;
      for (unsigned C = 0; C < VTy.getNumElements(); ++C) {
        auto IdxC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty,
                                                   static_cast<int32_t>(C));
        mlir::Value Lane =
            mlir::LLVM::ExtractElementOp::create(B, Loc, Vec, IdxC);
        Acc = Acc ? mlir::Value(mlir::LLVM::OrOp::create(B, Loc, Acc, Lane))
                  : Lane;
      }
      Any.replaceAllUsesWith(Acc);
      Any.erase();
    } else if (auto IsN = mlir::dyn_cast<mlir::spirv::IsNanOp>(Op)) {
      mlir::Value X = IsN.getOperand();
      auto Cmp = mlir::LLVM::FCmpOp::create(
          B, Loc, mlir::LLVM::FCmpPredicate::uno, X, X);
      IsN.replaceAllUsesWith(Cmp.getResult());
      IsN.erase();
    } else if (auto IsI = mlir::dyn_cast<mlir::spirv::IsInfOp>(Op)) {
      mlir::Value X = IsI.getOperand();
      mlir::Type XTy = X.getType();
      mlir::Type ElemTy = XTy;
      if (auto VTy = mlir::dyn_cast<mlir::VectorType>(XTy))
        ElemTy = VTy.getElementType();
      auto FTy = mlir::cast<mlir::FloatType>(ElemTy);
      auto InfVal = APFloat::getInf(FTy.getFloatSemantics());
      mlir::Attribute InfAttr =
          mlir::isa<mlir::VectorType>(XTy)
              ? mlir::Attribute(mlir::DenseElementsAttr::get(
                    mlir::cast<mlir::VectorType>(XTy), InfVal))
              : mlir::Attribute(B.getFloatAttr(FTy, InfVal));
      auto Abs = mlir::LLVM::FAbsOp::create(B, Loc, X);
      auto InfC = mlir::LLVM::ConstantOp::create(B, Loc, XTy, InfAttr);
      auto Cmp = mlir::LLVM::FCmpOp::create(
          B, Loc, mlir::LLVM::FCmpPredicate::oeq, Abs, InfC);
      IsI.replaceAllUsesWith(Cmp.getResult());
      IsI.erase();
    }
  }

  // 3g3. Lower spirv.Switch -> chain of spirv.IEqual + spirv.BranchConditional
  // so ConvertSPIRVToLLVM + LiftControlFlowToSCF handle switches cleanly.
  SmallVector<mlir::spirv::SwitchOp, 4> SwitchOps;
  EntryFn.walk([&](mlir::spirv::SwitchOp SW) { SwitchOps.push_back(SW); });
  for (auto SW : SwitchOps) {
    mlir::Location Loc = SW.getLoc();
    mlir::Value Sel = SW.getSelector();
    mlir::Block *DefaultBB = SW.getDefaultTarget();
    mlir::ValueRange DefaultArgs = SW.getDefaultOperands();
    auto Literals = SW.getLiterals();
    auto Targets = SW.getTargets();

    if (!Literals || Targets.empty()) {
      mlir::OpBuilder B(SW);
      mlir::spirv::BranchOp::create(B, Loc, DefaultBB, DefaultArgs);
      SW.erase();
      continue;
    }

    auto LitVals = Literals->getValues<llvm::APInt>();
    unsigned NumCases = static_cast<unsigned>(Targets.size());
    mlir::Block *CurBB = SW->getBlock();
    mlir::Region *Reg = CurBB->getParent();

    for (unsigned I = 0; I < NumCases; ++I) {
      mlir::OpBuilder B = mlir::OpBuilder::atBlockEnd(CurBB);
      if (I == 0)
        B.setInsertionPoint(SW);
      mlir::Block *NextBB = DefaultBB;
      mlir::ValueRange NextArgs = DefaultArgs;
      if (I + 1 < NumCases) {
        NextBB = new mlir::Block();
        Reg->push_back(NextBB);
        NextArgs = {};
      }
      auto CaseAttr = B.getIntegerAttr(Sel.getType(), LitVals[I]);
      auto CaseC = mlir::spirv::ConstantOp::create(B, Loc, Sel.getType(),
                                                   CaseAttr);
      auto Cmp = mlir::spirv::IEqualOp::create(B, Loc, Sel, CaseC);
      mlir::spirv::BranchConditionalOp::create(
          B, Loc, Cmp, Targets[I], SW.getTargetOperands(I), NextBB, NextArgs);
      CurBB = NextBB;
    }
    SW.erase();
  }

  // Work around upstream MLIR SPIRVToLLVM BranchConditionalConversionPattern
  // passing unconverted block arguments to LLVM::CondBrOp: route any
  // conditional branch edge with block arguments through a spirv.Branch
  // trampoline (BranchConversionPattern properly uses adaptor.getOperands()).
  {
    SmallVector<mlir::spirv::BranchConditionalOp, 8> CondBrs;
    EntryFn.walk([&](mlir::spirv::BranchConditionalOp BC) {
      if (!BC.getTrueBlockArguments().empty() ||
          !BC.getFalseBlockArguments().empty())
        CondBrs.push_back(BC);
    });
    for (auto BC : CondBrs) {
      mlir::Location Loc = BC.getLoc();
      mlir::Region *Reg = BC->getBlock()->getParent();
      mlir::Block *TrueBB = BC.getTrueBlock();
      mlir::Block *FalseBB = BC.getFalseBlock();
      if (!BC.getTrueBlockArguments().empty()) {
        auto *Tramp = new mlir::Block();
        Reg->push_back(Tramp);
        mlir::OpBuilder TB = mlir::OpBuilder::atBlockEnd(Tramp);
        mlir::spirv::BranchOp::create(TB, Loc, TrueBB,
                                      BC.getTrueBlockArguments());
        TrueBB = Tramp;
      }
      if (!BC.getFalseBlockArguments().empty()) {
        auto *Tramp = new mlir::Block();
        Reg->push_back(Tramp);
        mlir::OpBuilder FB = mlir::OpBuilder::atBlockEnd(Tramp);
        mlir::spirv::BranchOp::create(FB, Loc, FalseBB,
                                      BC.getFalseBlockArguments());
        FalseBB = Tramp;
      }
      mlir::OpBuilder B(BC);
      mlir::spirv::BranchConditionalOp::create(
          B, Loc, BC.getCondition(), TrueBB, mlir::ValueRange{}, FalseBB,
          mlir::ValueRange{});
      BC.erase();
    }
  }

  // 3g4. Lower spirv.GL.* and spirv.*Dot ops -> math.* / llvm.*.
  SmallVector<mlir::Operation *, 16> GLMathOps;
  EntryFn.walk([&](mlir::Operation *Op) {
    if (mlir::isa<mlir::spirv::GLAtan2Op, mlir::spirv::GLStepOp,
                  mlir::spirv::GLNormalizeOp, mlir::spirv::DotOp,
                  mlir::spirv::SDotOp, mlir::spirv::UDotOp,
                  mlir::spirv::SUDotOp, mlir::spirv::SDotAccSatOp,
                  mlir::spirv::UDotAccSatOp, mlir::spirv::SUDotAccSatOp,
                  mlir::spirv::GLFSignOp, mlir::spirv::GLSSignOp,
                  mlir::spirv::GLLengthOp, mlir::spirv::GLDistanceOp,
                  mlir::spirv::GLCrossOp, mlir::spirv::GLReflectOp,
                  mlir::spirv::GLSmoothStepOp, mlir::spirv::GLPackHalf2x16Op,
                  mlir::spirv::GLUnpackHalf2x16Op, mlir::spirv::GLFindILsbOp,
                  mlir::spirv::GLFindUMsbOp, mlir::spirv::GLFindSMsbOp>(Op) ||
        (Reflect.RequiresStrictFP && mlir::isa<mlir::spirv::GLAcoshOp>(Op)))
      GLMathOps.push_back(Op);
  });
  for (mlir::Operation *Op : GLMathOps) {
    mlir::OpBuilder B(Op);
    mlir::Location Loc = Op->getLoc();
    auto mkFloatConst = [&](mlir::Type Ty, double V) -> mlir::Value {
      mlir::Type ElemTy = Ty;
      if (auto VTy = mlir::dyn_cast<mlir::VectorType>(Ty))
        ElemTy = VTy.getElementType();
      auto FTy = mlir::cast<mlir::FloatType>(ElemTy);
      mlir::Attribute A =
          mlir::isa<mlir::VectorType>(Ty)
              ? mlir::Attribute(mlir::DenseElementsAttr::get(
                    mlir::cast<mlir::VectorType>(Ty), B.getFloatAttr(FTy, V)))
              : mlir::Attribute(B.getFloatAttr(FTy, V));
      return mlir::LLVM::ConstantOp::create(B, Loc, Ty, A);
    };
    auto mkIntConst = [&](mlir::Type Ty, int64_t V) -> mlir::Value {
      mlir::Type ElemTy = Ty;
      if (auto VTy = mlir::dyn_cast<mlir::VectorType>(Ty))
        ElemTy = VTy.getElementType();
      auto ITy = mlir::cast<mlir::IntegerType>(ElemTy);
      mlir::Attribute A =
          mlir::isa<mlir::VectorType>(Ty)
              ? mlir::Attribute(mlir::DenseElementsAttr::get(
                    mlir::cast<mlir::VectorType>(Ty),
                    B.getIntegerAttr(ITy, V)))
              : mlir::Attribute(B.getIntegerAttr(ITy, V));
      return mlir::LLVM::ConstantOp::create(B, Loc, Ty, A);
    };
    auto emitFloatDot = [&](mlir::Value A, mlir::Value BVal) -> mlir::Value {
      mlir::Value Prod = mlir::LLVM::FMulOp::create(B, Loc, A, BVal);
      if (auto VTy = mlir::dyn_cast<mlir::VectorType>(A.getType())) {
        mlir::Value Sum;
        for (unsigned C = 0; C < VTy.getNumElements(); ++C) {
          auto IdxC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty,
                                                     static_cast<int32_t>(C));
          mlir::Value Lane =
              mlir::LLVM::ExtractElementOp::create(B, Loc, Prod, IdxC);
          Sum = Sum ? mlir::Value(mlir::LLVM::FAddOp::create(B, Loc, Sum, Lane))
                    : Lane;
        }
        return Sum;
      }
      return Prod;
    };
    auto splatScalar = [&](mlir::Value Scalar,
                           mlir::VectorType VTy) -> mlir::Value {
      mlir::Value Splat = mlir::LLVM::UndefOp::create(B, Loc, VTy);
      for (unsigned C = 0; C < VTy.getNumElements(); ++C) {
        auto IdxC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty,
                                                   static_cast<int32_t>(C));
        Splat = mlir::LLVM::InsertElementOp::create(B, Loc, Splat, Scalar, IdxC);
      }
      return Splat;
    };

    if (auto Acosh = mlir::dyn_cast<mlir::spirv::GLAcoshOp>(Op)) {
      // Retain the robust builtin instead of log(x + sqrt(x*x - 1)), which
      // overflows for large finite inputs and mishandles negative infinity.
      auto Rep = mlir::math::AcoshOp::create(B, Loc, Acosh.getOperand());
      Acosh.replaceAllUsesWith(Rep.getResult());
      Acosh.erase();
    } else if (auto At2 = mlir::dyn_cast<mlir::spirv::GLAtan2Op>(Op)) {
      auto Rep = mlir::math::Atan2Op::create(B, Loc, At2->getOperand(0),
                                             At2->getOperand(1));
      At2.replaceAllUsesWith(Rep.getResult());
      At2.erase();
    } else if (auto St = mlir::dyn_cast<mlir::spirv::GLStepOp>(Op)) {
      mlir::Value Edge = St->getOperand(0);
      mlir::Value X = St->getOperand(1);
      mlir::Type Ty = X.getType();
      auto Cmp = mlir::LLVM::FCmpOp::create(
          B, Loc, mlir::LLVM::FCmpPredicate::olt, X, Edge);
      auto Sel = mlir::LLVM::SelectOp::create(
          B, Loc, Cmp, mkFloatConst(Ty, 0.0), mkFloatConst(Ty, 1.0));
      St.replaceAllUsesWith(Sel.getResult());
      St.erase();
    } else if (auto Norm = mlir::dyn_cast<mlir::spirv::GLNormalizeOp>(Op)) {
      mlir::Value X = Norm.getOperand();
      mlir::Type Ty = X.getType();
      if (auto VTy = mlir::dyn_cast<mlir::VectorType>(Ty)) {
        mlir::Value Sum = emitFloatDot(X, X);
        mlir::Value InvLen = mlir::math::RsqrtOp::create(B, Loc, Sum);
        mlir::Value Splat = splatScalar(InvLen, VTy);
        auto Res = mlir::LLVM::FMulOp::create(B, Loc, X, Splat);
        Norm.replaceAllUsesWith(Res.getResult());
        Norm.erase();
      } else {
        auto Abs = mlir::math::AbsFOp::create(B, Loc, X);
        auto Res = mlir::LLVM::FDivOp::create(B, Loc, X, Abs);
        Norm.replaceAllUsesWith(Res.getResult());
        Norm.erase();
      }
    } else if (auto Dot = mlir::dyn_cast<mlir::spirv::DotOp>(Op)) {
      mlir::Value Sum = emitFloatDot(Dot.getVector1(), Dot.getVector2());
      Dot.replaceAllUsesWith(Sum);
      Dot.erase();
    } else if (mlir::isa<mlir::spirv::SDotOp, mlir::spirv::UDotOp,
                         mlir::spirv::SUDotOp, mlir::spirv::SDotAccSatOp,
                         mlir::spirv::UDotAccSatOp, mlir::spirv::SUDotAccSatOp>(
                   Op)) {
      bool Saturating = Op->getNumOperands() == 3;
      bool Signed1 =
          mlir::isa<mlir::spirv::SDotOp, mlir::spirv::SUDotOp,
                    mlir::spirv::SDotAccSatOp,
                    mlir::spirv::SUDotAccSatOp>(Op);
      bool Signed2 =
          mlir::isa<mlir::spirv::SDotOp, mlir::spirv::SDotAccSatOp>(Op);
      mlir::Value V1 = castToSignless(B, Loc, Op->getOperand(0));
      mlir::Value V2 = castToSignless(B, Loc, Op->getOperand(1));
      mlir::Type OrigResTy = Op->getResult(0).getType();
      mlir::Type SignlessResTy =
          B.getIntegerType(OrigResTy.getIntOrFloatBitWidth());
      mlir::Type DotTy = Saturating
                            ? B.getIntegerType(std::max(
                                  32u, OrigResTy.getIntOrFloatBitWidth()))
                            : SignlessResTy;
      auto I8V4Ty = mlir::VectorType::get({4}, B.getI8Type());
      if (V1.getType().isInteger(32))
        V1 = mlir::LLVM::BitcastOp::create(B, Loc, I8V4Ty, V1);
      if (V2.getType().isInteger(32))
        V2 = mlir::LLVM::BitcastOp::create(B, Loc, I8V4Ty, V2);
      auto VTy = mlir::cast<mlir::VectorType>(V1.getType());
      mlir::Value Sum;
      for (unsigned C = 0; C < VTy.getNumElements(); ++C) {
        auto IdxC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty,
                                                   static_cast<int32_t>(C));
        mlir::Value E1 = mlir::LLVM::ExtractElementOp::create(B, Loc, V1, IdxC);
        mlir::Value E2 = mlir::LLVM::ExtractElementOp::create(B, Loc, V2, IdxC);
        if (E1.getType() != DotTy)
          E1 = Signed1
                   ? mlir::Value(mlir::LLVM::SExtOp::create(B, Loc,
                                                            DotTy, E1))
                   : mlir::Value(mlir::LLVM::ZExtOp::create(B, Loc,
                                                            DotTy, E1));
        if (E2.getType() != DotTy)
          E2 = Signed2
                   ? mlir::Value(mlir::LLVM::SExtOp::create(B, Loc,
                                                            DotTy, E2))
                   : mlir::Value(mlir::LLVM::ZExtOp::create(B, Loc,
                                                            DotTy, E2));
        mlir::Value P = mlir::LLVM::MulOp::create(B, Loc, E1, E2);
        Sum = Sum ? mlir::Value(mlir::LLVM::AddOp::create(B, Loc, Sum, P)) : P;
      }
      if (Saturating) {
        // Only the final accumulation saturates. Intermediate dot overflow is
        // undefined by SPIR-V; keep narrow arithmetic out of MSL promotions.
        if (DotTy != SignlessResTy)
          Sum = mlir::LLVM::TruncOp::create(B, Loc, SignlessResTy, Sum);
        mlir::Value Acc = castToSignless(B, Loc, Op->getOperand(2));
        Sum = Signed1
                  ? mlir::Value(mlir::LLVM::SAddSat::create(B, Loc, Sum, Acc))
                  : mlir::Value(mlir::LLVM::UAddSat::create(B, Loc, Sum, Acc));
      }
      Op->getResult(0).replaceAllUsesWith(
          castFromSignless(B, Loc, Sum, OrigResTy));
      Op->erase();
    } else if (auto FS = mlir::dyn_cast<mlir::spirv::GLFSignOp>(Op)) {
      mlir::Value X = FS.getOperand();
      mlir::Type Ty = X.getType();
      mlir::Value Zero = mkFloatConst(Ty, 0.0);
      mlir::Value Gt = mlir::LLVM::FCmpOp::create(
          B, Loc, mlir::LLVM::FCmpPredicate::ogt, X, Zero);
      mlir::Value Lt = mlir::LLVM::FCmpOp::create(
          B, Loc, mlir::LLVM::FCmpPredicate::olt, X, Zero);
      mlir::Value Pos = mlir::LLVM::UIToFPOp::create(B, Loc, Ty, Gt);
      mlir::Value Neg = mlir::LLVM::UIToFPOp::create(B, Loc, Ty, Lt);
      mlir::Value Res = mlir::LLVM::FSubOp::create(B, Loc, Pos, Neg);
      FS.replaceAllUsesWith(Res);
      FS.erase();
    } else if (auto SS = mlir::dyn_cast<mlir::spirv::GLSSignOp>(Op)) {
      mlir::Type OrigTy = SS.getType();
      mlir::Value X = castToSignless(B, Loc, SS.getOperand());
      mlir::Type Ty = X.getType();
      mlir::Value Zero = mkIntConst(Ty, 0);
      mlir::Value Gt = mlir::LLVM::ICmpOp::create(
          B, Loc, mlir::LLVM::ICmpPredicate::sgt, X, Zero);
      mlir::Value Lt = mlir::LLVM::ICmpOp::create(
          B, Loc, mlir::LLVM::ICmpPredicate::slt, X, Zero);
      mlir::Value Pos = mlir::LLVM::ZExtOp::create(B, Loc, Ty, Gt);
      mlir::Value Neg = mlir::LLVM::ZExtOp::create(B, Loc, Ty, Lt);
      mlir::Value Res = mlir::LLVM::SubOp::create(B, Loc, Pos, Neg);
      SS.replaceAllUsesWith(castFromSignless(B, Loc, Res, OrigTy));
      SS.erase();
    } else if (auto Len = mlir::dyn_cast<mlir::spirv::GLLengthOp>(Op)) {
      mlir::Value X = Len.getOperand();
      mlir::Value Res;
      if (mlir::isa<mlir::VectorType>(X.getType())) {
        mlir::Value Sum = emitFloatDot(X, X);
        Res = mlir::math::SqrtOp::create(B, Loc, Sum);
      } else {
        Res = mlir::math::AbsFOp::create(B, Loc, X);
      }
      Len.replaceAllUsesWith(Res);
      Len.erase();
    } else if (auto Dist = mlir::dyn_cast<mlir::spirv::GLDistanceOp>(Op)) {
      mlir::Value Diff =
          mlir::LLVM::FSubOp::create(B, Loc, Dist->getOperand(0),
                                     Dist->getOperand(1));
      mlir::Value Res;
      if (mlir::isa<mlir::VectorType>(Diff.getType())) {
        mlir::Value Sum = emitFloatDot(Diff, Diff);
        Res = mlir::math::SqrtOp::create(B, Loc, Sum);
      } else {
        Res = mlir::math::AbsFOp::create(B, Loc, Diff);
      }
      Dist.replaceAllUsesWith(Res);
      Dist.erase();
    } else if (auto Cross = mlir::dyn_cast<mlir::spirv::GLCrossOp>(Op)) {
      mlir::Value A = Cross->getOperand(0);
      mlir::Value BVal = Cross->getOperand(1);
      auto VTy = mlir::cast<mlir::VectorType>(A.getType());
      auto c0 = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, int32_t(0));
      auto c1 = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, int32_t(1));
      auto c2 = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, int32_t(2));
      mlir::Value ax = mlir::LLVM::ExtractElementOp::create(B, Loc, A, c0);
      mlir::Value ay = mlir::LLVM::ExtractElementOp::create(B, Loc, A, c1);
      mlir::Value az = mlir::LLVM::ExtractElementOp::create(B, Loc, A, c2);
      mlir::Value bx = mlir::LLVM::ExtractElementOp::create(B, Loc, BVal, c0);
      mlir::Value by = mlir::LLVM::ExtractElementOp::create(B, Loc, BVal, c1);
      mlir::Value bz = mlir::LLVM::ExtractElementOp::create(B, Loc, BVal, c2);
      mlir::Value rx = mlir::LLVM::FSubOp::create(
          B, Loc, mlir::LLVM::FMulOp::create(B, Loc, ay, bz),
          mlir::LLVM::FMulOp::create(B, Loc, az, by));
      mlir::Value ry = mlir::LLVM::FSubOp::create(
          B, Loc, mlir::LLVM::FMulOp::create(B, Loc, az, bx),
          mlir::LLVM::FMulOp::create(B, Loc, ax, bz));
      mlir::Value rz = mlir::LLVM::FSubOp::create(
          B, Loc, mlir::LLVM::FMulOp::create(B, Loc, ax, by),
          mlir::LLVM::FMulOp::create(B, Loc, ay, bx));
      mlir::Value Res = mlir::LLVM::UndefOp::create(B, Loc, VTy);
      Res = mlir::LLVM::InsertElementOp::create(B, Loc, Res, rx, c0);
      Res = mlir::LLVM::InsertElementOp::create(B, Loc, Res, ry, c1);
      Res = mlir::LLVM::InsertElementOp::create(B, Loc, Res, rz, c2);
      Cross.replaceAllUsesWith(Res);
      Cross.erase();
    } else if (auto Refl = mlir::dyn_cast<mlir::spirv::GLReflectOp>(Op)) {
      mlir::Value I = Refl->getOperand(0);
      mlir::Value N = Refl->getOperand(1);
      mlir::Value D = emitFloatDot(N, I);
      mlir::Value Two = mkFloatConst(D.getType(), 2.0);
      mlir::Value TwoD = mlir::LLVM::FMulOp::create(B, Loc, Two, D);
      if (auto VTy = mlir::dyn_cast<mlir::VectorType>(N.getType()))
        TwoD = splatScalar(TwoD, VTy);
      mlir::Value Scaled = mlir::LLVM::FMulOp::create(B, Loc, TwoD, N);
      mlir::Value Res = mlir::LLVM::FSubOp::create(B, Loc, I, Scaled);
      Refl.replaceAllUsesWith(Res);
      Refl.erase();
    } else if (auto Sm = mlir::dyn_cast<mlir::spirv::GLSmoothStepOp>(Op)) {
      mlir::Value Edge0 = Sm->getOperand(0);
      mlir::Value Edge1 = Sm->getOperand(1);
      mlir::Value X = Sm->getOperand(2);
      mlir::Type Ty = X.getType();
      mlir::Value Num = mlir::LLVM::FSubOp::create(B, Loc, X, Edge0);
      mlir::Value Den = mlir::LLVM::FSubOp::create(B, Loc, Edge1, Edge0);
      mlir::Value Ratio = mlir::LLVM::FDivOp::create(B, Loc, Num, Den);
      mlir::Value ClampedLo = mlir::LLVM::MaxNumOp::create(
          B, Loc, Ratio, mkFloatConst(Ty, 0.0));
      mlir::Value T = mlir::LLVM::MinNumOp::create(B, Loc, ClampedLo,
                                                   mkFloatConst(Ty, 1.0));
      mlir::Value TT = mlir::LLVM::FMulOp::create(B, Loc, T, T);
      mlir::Value TwoT =
          mlir::LLVM::FMulOp::create(B, Loc, mkFloatConst(Ty, 2.0), T);
      mlir::Value ThreeMinus2T =
          mlir::LLVM::FSubOp::create(B, Loc, mkFloatConst(Ty, 3.0), TwoT);
      mlir::Value Res = mlir::LLVM::FMulOp::create(B, Loc, TT, ThreeMinus2T);
      Sm.replaceAllUsesWith(Res);
      Sm.erase();
    } else if (auto Pack = mlir::dyn_cast<mlir::spirv::GLPackHalf2x16Op>(Op)) {
      auto F16V2Ty = mlir::VectorType::get({2}, B.getF16Type());
      mlir::Value H2 =
          mlir::LLVM::FPTruncOp::create(B, Loc, F16V2Ty, Pack.getOperand());
      mlir::Value Bits = mlir::LLVM::BitcastOp::create(B, Loc, I32Ty, H2);
      Pack.replaceAllUsesWith(
          castFromSignless(B, Loc, Bits, Pack.getType()));
      Pack.erase();
    } else if (auto Unpack =
                   mlir::dyn_cast<mlir::spirv::GLUnpackHalf2x16Op>(Op)) {
      auto F16V2Ty = mlir::VectorType::get({2}, B.getF16Type());
      mlir::Value S = castToSignless(B, Loc, Unpack.getOperand());
      mlir::Value H2 = mlir::LLVM::BitcastOp::create(B, Loc, F16V2Ty, S);
      mlir::Value Res =
          mlir::LLVM::FPExtOp::create(B, Loc, Unpack.getType(), H2);
      Unpack.replaceAllUsesWith(Res);
      Unpack.erase();
    } else if (auto FLsb = mlir::dyn_cast<mlir::spirv::GLFindILsbOp>(Op)) {
      mlir::Type OrigTy = FLsb.getType();
      mlir::Value X = castToSignless(B, Loc, FLsb.getOperand());
      mlir::Type Ty = X.getType();
      mlir::Value Ctz = mlir::LLVM::CountTrailingZerosOp::create(
          B, Loc, X, /*is_zero_poison=*/false);
      mlir::Value IsZero = mlir::LLVM::ICmpOp::create(
          B, Loc, mlir::LLVM::ICmpPredicate::eq, X, mkIntConst(Ty, 0));
      mlir::Value Mask = mlir::LLVM::SExtOp::create(B, Loc, Ty, IsZero);
      mlir::Value Res = mlir::LLVM::OrOp::create(B, Loc, Ctz, Mask);
      FLsb.replaceAllUsesWith(castFromSignless(B, Loc, Res, OrigTy));
      FLsb.erase();
    } else if (auto FUMsb = mlir::dyn_cast<mlir::spirv::GLFindUMsbOp>(Op)) {
      mlir::Type OrigTy = FUMsb.getType();
      mlir::Value X = castToSignless(B, Loc, FUMsb.getOperand());
      mlir::Type Ty = X.getType();
      mlir::Type ElemTy = Ty;
      if (auto VTy = mlir::dyn_cast<mlir::VectorType>(Ty))
        ElemTy = VTy.getElementType();
      int64_t BitsMinus1 =
          static_cast<int64_t>(ElemTy.getIntOrFloatBitWidth()) - 1;
      mlir::Value Clz = mlir::LLVM::CountLeadingZerosOp::create(
          B, Loc, X, /*is_zero_poison=*/false);
      mlir::Value Res = mlir::LLVM::SubOp::create(
          B, Loc, mkIntConst(Ty, BitsMinus1), Clz);
      FUMsb.replaceAllUsesWith(castFromSignless(B, Loc, Res, OrigTy));
      FUMsb.erase();
    } else if (auto FSMsb = mlir::dyn_cast<mlir::spirv::GLFindSMsbOp>(Op)) {
      mlir::Type OrigTy = FSMsb.getType();
      mlir::Value X = castToSignless(B, Loc, FSMsb.getOperand());
      mlir::Type Ty = X.getType();
      mlir::Type ElemTy = Ty;
      if (auto VTy = mlir::dyn_cast<mlir::VectorType>(Ty))
        ElemTy = VTy.getElementType();
      int64_t BitsMinus1 =
          static_cast<int64_t>(ElemTy.getIntOrFloatBitWidth()) - 1;
      mlir::Value SignMask = mlir::LLVM::AShrOp::create(
          B, Loc, X, mkIntConst(Ty, BitsMinus1));
      mlir::Value Norm = mlir::LLVM::XOrOp::create(B, Loc, X, SignMask);
      mlir::Value Clz = mlir::LLVM::CountLeadingZerosOp::create(
          B, Loc, Norm, /*is_zero_poison=*/false);
      mlir::Value Res = mlir::LLVM::SubOp::create(
          B, Loc, mkIntConst(Ty, BitsMinus1), Clz);
      FSMsb.replaceAllUsesWith(castFromSignless(B, Loc, Res, OrigTy));
      FSMsb.erase();
    }
  }

  // 3g5. Lower spirv integer comparison ops with non-signless integer operands
  // (si32/ui32/vectors) directly to llvm.icmp with signless casts, working
  // around ConvertSPIRVToLLVM leaving signed operands on llvm.icmp.
  SmallVector<mlir::Operation *, 16> ICmpOps;
  EntryFn.walk([&](mlir::Operation *Op) {
    if (mlir::isa<mlir::spirv::IEqualOp, mlir::spirv::INotEqualOp,
                  mlir::spirv::SLessThanOp, mlir::spirv::SLessThanEqualOp,
                  mlir::spirv::SGreaterThanOp, mlir::spirv::SGreaterThanEqualOp,
                  mlir::spirv::ULessThanOp, mlir::spirv::ULessThanEqualOp,
                  mlir::spirv::UGreaterThanOp,
                  mlir::spirv::UGreaterThanEqualOp>(Op)) {
      mlir::Type OpTy = Op->getOperand(0).getType();
      mlir::Type ElemTy = OpTy;
      if (auto VTy = mlir::dyn_cast<mlir::VectorType>(OpTy))
        ElemTy = VTy.getElementType();
      if (auto IT = mlir::dyn_cast<mlir::IntegerType>(ElemTy);
          IT && !IT.isSignless())
        ICmpOps.push_back(Op);
    }
  });
  for (mlir::Operation *Op : ICmpOps) {
    mlir::OpBuilder B(Op);
    mlir::Location Loc = Op->getLoc();
    mlir::LLVM::ICmpPredicate Pred;
    if (mlir::isa<mlir::spirv::IEqualOp>(Op))
      Pred = mlir::LLVM::ICmpPredicate::eq;
    else if (mlir::isa<mlir::spirv::INotEqualOp>(Op))
      Pred = mlir::LLVM::ICmpPredicate::ne;
    else if (mlir::isa<mlir::spirv::SLessThanOp>(Op))
      Pred = mlir::LLVM::ICmpPredicate::slt;
    else if (mlir::isa<mlir::spirv::SLessThanEqualOp>(Op))
      Pred = mlir::LLVM::ICmpPredicate::sle;
    else if (mlir::isa<mlir::spirv::SGreaterThanOp>(Op))
      Pred = mlir::LLVM::ICmpPredicate::sgt;
    else if (mlir::isa<mlir::spirv::SGreaterThanEqualOp>(Op))
      Pred = mlir::LLVM::ICmpPredicate::sge;
    else if (mlir::isa<mlir::spirv::ULessThanOp>(Op))
      Pred = mlir::LLVM::ICmpPredicate::ult;
    else if (mlir::isa<mlir::spirv::ULessThanEqualOp>(Op))
      Pred = mlir::LLVM::ICmpPredicate::ule;
    else if (mlir::isa<mlir::spirv::UGreaterThanOp>(Op))
      Pred = mlir::LLVM::ICmpPredicate::ugt;
    else
      Pred = mlir::LLVM::ICmpPredicate::uge;

    mlir::Value LHS = castToSignless(B, Loc, Op->getOperand(0));
    mlir::Value RHS = castToSignless(B, Loc, Op->getOperand(1));
    auto Cmp = mlir::LLVM::ICmpOp::create(B, Loc, Pred, LHS, RHS);
    Op->getResult(0).replaceAllUsesWith(Cmp.getResult());
    Op->erase();
  }

  // 3g6. Fold spirv.CompositeExtract(spirv.CompositeConstruct(...), idx) for
  // array/struct/matrix composites (e.g. matrix construction by sub-matrices).
  bool FoldedComposite = true;
  while (FoldedComposite) {
    FoldedComposite = false;
    SmallVector<mlir::spirv::CompositeExtractOp, 16> Extracts;
    EntryFn.walk(
        [&](mlir::spirv::CompositeExtractOp E) { Extracts.push_back(E); });
    for (auto E : Extracts) {
      auto CC = E.getComposite().getDefiningOp<mlir::spirv::CompositeConstructOp>();
      if (!CC || E.getIndices().empty())
        continue;
      if (mlir::isa<mlir::VectorType>(CC.getType()))
        continue;
      int64_t Idx = mlir::cast<mlir::IntegerAttr>(E.getIndices()[0]).getInt();
      if (Idx < 0 || static_cast<size_t>(Idx) >= CC.getConstituents().size())
        continue;
      mlir::Value Elem = CC.getConstituents()[Idx];
      if (E.getIndices().size() == 1) {
        E.replaceAllUsesWith(Elem);
        E.erase();
        FoldedComposite = true;
      } else {
        mlir::OpBuilder B(E);
        SmallVector<mlir::Attribute, 4> Rest(E.getIndices().begin() + 1,
                                             E.getIndices().end());
        auto NewE = mlir::spirv::CompositeExtractOp::create(
            B, E.getLoc(), E.getType(), Elem, B.getArrayAttr(Rest));
        E.replaceAllUsesWith(NewE.getResult());
        E.erase();
        FoldedComposite = true;
      }
    }
    SmallVector<mlir::spirv::CompositeConstructOp, 16> DeadCCs;
    EntryFn.walk([&](mlir::spirv::CompositeConstructOp CC) {
      if (CC->use_empty())
        DeadCCs.push_back(CC);
    });
    for (auto CC : DeadCCs) {
      CC.erase();
      FoldedComposite = true;
    }
  }

  // 3h. Lower Function-scope array/matrix spirv.Variable (including constant-
  // array & CompositeConstruct initializers) -> llvm.alloca + element stores.
  SmallVector<mlir::spirv::VariableOp, 4> FuncArrayVars;
  EntryFn.walk([&](mlir::spirv::VariableOp V) {
    auto PtrTy = mlir::dyn_cast<mlir::spirv::PointerType>(V.getType());
    if (PtrTy && PtrTy.getStorageClass() == mlir::spirv::StorageClass::Function &&
        mlir::isa<mlir::spirv::ArrayType, mlir::spirv::MatrixType>(
            PtrTy.getPointeeType()))
      FuncArrayVars.push_back(V);
  });
  for (auto V : FuncArrayVars) {
    auto PtrTy = mlir::cast<mlir::spirv::PointerType>(V.getType());
    mlir::Type PointeeTy = PtrTy.getPointeeType();
    mlir::OpBuilder VB(V);
    mlir::Location Loc = V.getLoc();
    mlir::Type LLVMArrTy = convertSpirvDataType(PointeeTy, VB);
    auto Ptr0Ty = mlir::LLVM::LLVMPointerType::get(&Ctx, 0);
    auto OneC = mlir::LLVM::ConstantOp::create(VB, Loc, I32Ty, 1);
    auto Alloca = mlir::LLVM::AllocaOp::create(VB, Loc, Ptr0Ty, LLVMArrTy, OneC);

    SmallVector<mlir::Operation *, 8> Users(V->getUsers().begin(),
                                            V->getUsers().end());
    for (mlir::Operation *U : Users) {
      if (auto Store = mlir::dyn_cast<mlir::spirv::StoreOp>(U)) {
        if (auto ConstArr =
                Store.getValue().getDefiningOp<mlir::spirv::ConstantOp>()) {
          if (auto AA = mlir::dyn_cast<mlir::ArrayAttr>(ConstArr.getValue())) {
            mlir::OpBuilder SB(Store);
            for (unsigned I = 0; I < AA.size(); ++I) {
              SmallVector<mlir::LLVM::GEPArg, 2> Idx{int32_t(0),
                                                     static_cast<int32_t>(I)};
              auto ElemPtr = mlir::LLVM::GEPOp::create(
                  SB, Store.getLoc(), Ptr0Ty, LLVMArrTy, Alloca, Idx);
              auto ArrTy = mlir::cast<mlir::spirv::ArrayType>(PointeeTy);
              auto ElemConst = mlir::LLVM::ConstantOp::create(
                  SB, Store.getLoc(),
                  convertSpirvDataType(ArrTy.getElementType(), SB), AA[I]);
              mlir::LLVM::StoreOp::create(SB, Store.getLoc(), ElemConst,
                                          ElemPtr);
            }
            Store.erase();
            if (ConstArr->use_empty())
              ConstArr.erase();
            continue;
          }
        }
        if (auto CC = Store.getValue()
                          .getDefiningOp<mlir::spirv::CompositeConstructOp>()) {
          mlir::OpBuilder SB(Store);
          auto Elems = CC.getConstituents();
          for (unsigned I = 0; I < Elems.size(); ++I) {
            SmallVector<mlir::LLVM::GEPArg, 2> Idx{int32_t(0),
                                                   static_cast<int32_t>(I)};
            auto ElemPtr = mlir::LLVM::GEPOp::create(
                SB, Store.getLoc(), Ptr0Ty, LLVMArrTy, Alloca, Idx);
            mlir::Value ElemVal = castToSignless(SB, Store.getLoc(), Elems[I]);
            mlir::LLVM::StoreOp::create(SB, Store.getLoc(), ElemVal, ElemPtr);
          }
          Store.erase();
          if (CC->use_empty())
            CC.erase();
          continue;
        }
        if (auto MatTy = mlir::dyn_cast<mlir::spirv::MatrixType>(PointeeTy)) {
          mlir::OpBuilder SB(Store);
          mlir::Type ColTy = MatTy.getColumnType();
          for (uint32_t I = 0; I < MatTy.getNumColumns(); ++I) {
            SmallVector<mlir::LLVM::GEPArg, 2> Idx{int32_t(0),
                                                   static_cast<int32_t>(I)};
            auto ElemPtr = mlir::LLVM::GEPOp::create(
                SB, Store.getLoc(), Ptr0Ty, LLVMArrTy, Alloca, Idx);
            auto ColVal = mlir::spirv::CompositeExtractOp::create(
                SB, Store.getLoc(), ColTy, Store.getValue(),
                SB.getI32ArrayAttr({static_cast<int32_t>(I)}));
            mlir::Value SignlessCol =
                castToSignless(SB, Store.getLoc(), ColVal.getResult());
            mlir::LLVM::StoreOp::create(SB, Store.getLoc(), SignlessCol,
                                        ElemPtr);
          }
          Store.erase();
          continue;
        }
        if (Store.getMemoryAccess() &&
            *Store.getMemoryAccess() != mlir::spirv::MemoryAccess::None)
          return makeError(
              "memory operands on whole-array stores are unsupported");
        mlir::OpBuilder SB(Store);
        mlir::Value Value = mlir::UnrealizedConversionCastOp::create(
                                SB, Store.getLoc(), LLVMArrTy, Store.getValue())
                                .getResult(0);
        mlir::LLVM::StoreOp::create(SB, Store.getLoc(), Value, Alloca);
        Store.erase();
      } else if (auto Chain = mlir::dyn_cast<mlir::spirv::AccessChainOp>(U)) {
        mlir::OpBuilder CB(Chain);
        SmallVector<mlir::LLVM::GEPArg, 4> GEPArgs;
        GEPArgs.push_back(int32_t(0));
        for (mlir::Value IdxV : Chain.getIndices()) {
          if (auto C = getConstSpirvInt(IdxV))
            GEPArgs.push_back(static_cast<int32_t>(*C));
          else
            GEPArgs.push_back(castToSignless(CB, Chain.getLoc(), IdxV));
        }
        auto GEP = mlir::LLVM::GEPOp::create(CB, Chain.getLoc(), Ptr0Ty,
                                             LLVMArrTy, Alloca, GEPArgs);
        SmallVector<mlir::Operation *, 4> CUsers(Chain->getUsers().begin(),
                                                 Chain->getUsers().end());
        for (mlir::Operation *CU : CUsers) {
          if (auto Load = mlir::dyn_cast<mlir::spirv::LoadOp>(CU)) {
            mlir::OpBuilder LB(Load);
            mlir::Type OrigTy = Load.getType();
            auto LLVMLoad = mlir::LLVM::LoadOp::create(
                LB, Load.getLoc(), makeSignless(OrigTy, LB), GEP);
            Load.replaceAllUsesWith(
                castFromSignless(LB, Load.getLoc(), LLVMLoad.getResult(),
                                 OrigTy));
            Load.erase();
          } else if (auto Store = mlir::dyn_cast<mlir::spirv::StoreOp>(CU)) {
            mlir::OpBuilder SB(Store);
            mlir::Value Val = castToSignless(SB, Store.getLoc(), Store.getValue());
            mlir::LLVM::StoreOp::create(SB, Store.getLoc(), Val, GEP);
            Store.erase();
          }
        }
        Chain.erase();
      } else if (auto Load = mlir::dyn_cast<mlir::spirv::LoadOp>(U)) {
        mlir::OpBuilder LB(Load);
        if (auto MatTy = mlir::dyn_cast<mlir::spirv::MatrixType>(PointeeTy)) {
          mlir::Type ColTy = MatTy.getColumnType();
          mlir::Type SignlessColTy = makeSignless(ColTy, LB);
          SmallVector<mlir::Value, 4> Cols;
          for (uint32_t I = 0; I < MatTy.getNumColumns(); ++I) {
            SmallVector<mlir::LLVM::GEPArg, 2> Idx{int32_t(0),
                                                   static_cast<int32_t>(I)};
            auto ElemPtr = mlir::LLVM::GEPOp::create(
                LB, Load.getLoc(), Ptr0Ty, LLVMArrTy, Alloca, Idx);
            auto LLVMLoad = mlir::LLVM::LoadOp::create(
                LB, Load.getLoc(), SignlessColTy, ElemPtr);
            Cols.push_back(castFromSignless(LB, Load.getLoc(),
                                            LLVMLoad.getResult(), ColTy));
          }
          auto MatCC = mlir::spirv::CompositeConstructOp::create(
              LB, Load.getLoc(), Load.getType(), Cols);
          Load.replaceAllUsesWith(MatCC.getResult());
          Load.erase();
          continue;
        }
        auto LLVMLoad =
            mlir::LLVM::LoadOp::create(LB, Load.getLoc(), LLVMArrTy, Alloca);
        mlir::Value Repl = LLVMLoad.getResult();
        if (Repl.getType() != Load.getType()) {
          Repl = mlir::UnrealizedConversionCastOp::create(
                     LB, Load.getLoc(), Load.getType(), Repl)
                     .getResult(0);
        }
        Load.replaceAllUsesWith(Repl);
        Load.erase();
      }
    }
    if (V->use_empty())
      V.erase();
  }

  // 3i. Lower matrix SSA ops (Constant, Transpose, MatrixTimes*, Composite*)
  // to pure vector ops so no !spirv.matrix values survive to ConvertSPIRVToLLVM.
  {
    // Expand constant matrices into CompositeConstruct of column constants.
    SmallVector<mlir::spirv::ConstantOp, 8> MatConsts;
    EntryFn.walk([&](mlir::spirv::ConstantOp C) {
      if (mlir::isa<mlir::spirv::MatrixType>(C.getType()))
        MatConsts.push_back(C);
    });
    for (auto C : MatConsts) {
      if (!C.getValue())
        continue;
      auto MatTy = mlir::cast<mlir::spirv::MatrixType>(C.getType());
      auto ColTy = mlir::cast<mlir::VectorType>(MatTy.getColumnType());
      auto AA = mlir::dyn_cast<mlir::ArrayAttr>(C.getValue());
      if (!AA || AA.size() != MatTy.getNumColumns())
        continue;
      mlir::OpBuilder B(C);
      SmallVector<mlir::Value, 4> Cols;
      for (uint32_t I = 0; I < MatTy.getNumColumns(); ++I) {
        auto ColC = mlir::spirv::ConstantOp::create(B, C.getLoc(), ColTy, AA[I]);
        Cols.push_back(ColC.getResult());
      }
      auto CC = mlir::spirv::CompositeConstructOp::create(
          B, C.getLoc(), MatTy, Cols);
      C.replaceAllUsesWith(CC.getResult());
      C.erase();
    }

    auto buildVec = [&](mlir::OpBuilder &B, mlir::Location Loc,
                        mlir::VectorType VecTy,
                        ArrayRef<mlir::Value> Elems) -> mlir::Value {
      mlir::Type SignlessVecTy = makeSignless(VecTy, B);
      mlir::Value Acc = mlir::LLVM::UndefOp::create(B, Loc, SignlessVecTy);
      for (size_t I = 0; I < Elems.size(); ++I) {
        auto IdxC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty,
                                                   static_cast<int32_t>(I));
        mlir::Value EV = castToSignless(B, Loc, Elems[I]);
        Acc = mlir::LLVM::InsertElementOp::create(B, Loc, Acc, EV, IdxC);
      }
      return castFromSignless(B, Loc, Acc, VecTy);
    };
    auto makeSplat = [&](mlir::OpBuilder &B, mlir::Location Loc,
                         mlir::Value Scalar, mlir::VectorType VecTy) -> mlir::Value {
      SmallVector<mlir::Value, 4> Elems(VecTy.getNumElements(), Scalar);
      return buildVec(B, Loc, VecTy, Elems);
    };
    auto emitMatVecMul = [&](mlir::OpBuilder &B, mlir::Location Loc,
                             mlir::spirv::CompositeConstructOp MatCC,
                             mlir::Value Vec) -> mlir::Value {
      auto Cols = MatCC.getConstituents();
      auto ColTy = mlir::cast<mlir::VectorType>(Cols[0].getType());
      mlir::Type ElemTy = ColTy.getElementType();
      mlir::Value Acc;
      for (unsigned C = 0; C < Cols.size(); ++C) {
        auto VC = mlir::spirv::CompositeExtractOp::create(
            B, Loc, ElemTy, Vec, B.getI32ArrayAttr({static_cast<int32_t>(C)}));
        mlir::Value Splat = makeSplat(B, Loc, VC.getResult(), ColTy);
        mlir::Value Prod = mlir::spirv::FMulOp::create(B, Loc, Cols[C], Splat);
        Acc = Acc ? mlir::Value(mlir::spirv::FAddOp::create(B, Loc, Acc, Prod))
                  : Prod;
      }
      return Acc;
    };

    bool Changed = true;
    while (Changed) {
      Changed = false;

      // Transpose(CompositeConstruct) -> CompositeConstruct of transposed cols.
      SmallVector<mlir::spirv::TransposeOp, 8> Transposes;
      EntryFn.walk([&](mlir::spirv::TransposeOp T) { Transposes.push_back(T); });
      for (auto T : Transposes) {
        auto CC = T.getMatrix().getDefiningOp<mlir::spirv::CompositeConstructOp>();
        if (!CC)
          continue;
        auto ResMatTy = mlir::cast<mlir::spirv::MatrixType>(T.getType());
        auto ResColTy = mlir::cast<mlir::VectorType>(ResMatTy.getColumnType());
        mlir::Type ElemTy = ResColTy.getElementType();
        auto InCols = CC.getConstituents();
        mlir::OpBuilder B(T);
        SmallVector<mlir::Value, 4> OutCols;
        for (uint32_t R = 0; R < ResMatTy.getNumColumns(); ++R) {
          SmallVector<mlir::Value, 4> Lanes;
          for (unsigned C = 0; C < InCols.size(); ++C) {
            Lanes.push_back(mlir::spirv::CompositeExtractOp::create(
                                B, T.getLoc(), ElemTy, InCols[C],
                                B.getI32ArrayAttr({static_cast<int32_t>(R)}))
                                .getResult());
          }
          OutCols.push_back(buildVec(B, T.getLoc(), ResColTy, Lanes));
        }
        auto OutCC = mlir::spirv::CompositeConstructOp::create(
            B, T.getLoc(), ResMatTy, OutCols);
        T.replaceAllUsesWith(OutCC.getResult());
        T.erase();
        Changed = true;
      }

      // MatrixTimesScalar(CompositeConstruct, s) -> CompositeConstruct.
      SmallVector<mlir::spirv::MatrixTimesScalarOp, 8> MTSs;
      EntryFn.walk([&](mlir::spirv::MatrixTimesScalarOp M) { MTSs.push_back(M); });
      for (auto M : MTSs) {
        auto CC = M.getMatrix().getDefiningOp<mlir::spirv::CompositeConstructOp>();
        if (!CC)
          continue;
        mlir::OpBuilder B(M);
        auto MatTy = mlir::cast<mlir::spirv::MatrixType>(M.getType());
        auto ColTy = mlir::cast<mlir::VectorType>(MatTy.getColumnType());
        mlir::Value Splat = makeSplat(B, M.getLoc(), M.getScalar(), ColTy);
        SmallVector<mlir::Value, 4> OutCols;
        for (mlir::Value Col : CC.getConstituents())
          OutCols.push_back(mlir::spirv::FMulOp::create(B, M.getLoc(), Col, Splat));
        auto OutCC = mlir::spirv::CompositeConstructOp::create(
            B, M.getLoc(), MatTy, OutCols);
        M.replaceAllUsesWith(OutCC.getResult());
        M.erase();
        Changed = true;
      }

      // MatrixTimesVector(CompositeConstruct, v) -> vector.
      SmallVector<mlir::spirv::MatrixTimesVectorOp, 8> MTVs;
      EntryFn.walk([&](mlir::spirv::MatrixTimesVectorOp M) { MTVs.push_back(M); });
      for (auto M : MTVs) {
        auto CC = M.getMatrix().getDefiningOp<mlir::spirv::CompositeConstructOp>();
        if (!CC)
          continue;
        mlir::OpBuilder B(M);
        mlir::Value Res = emitMatVecMul(B, M.getLoc(), CC, M.getVector());
        M.replaceAllUsesWith(Res);
        M.erase();
        Changed = true;
      }

      // VectorTimesMatrix(v, CompositeConstruct) -> vector.
      SmallVector<mlir::spirv::VectorTimesMatrixOp, 8> VTMs;
      EntryFn.walk([&](mlir::spirv::VectorTimesMatrixOp M) { VTMs.push_back(M); });
      for (auto M : VTMs) {
        auto CC = M.getMatrix().getDefiningOp<mlir::spirv::CompositeConstructOp>();
        if (!CC)
          continue;
        mlir::OpBuilder B(M);
        auto ResVecTy = mlir::cast<mlir::VectorType>(M.getType());
        mlir::Type ElemTy = ResVecTy.getElementType();
        mlir::Value V = M.getVector();
        auto VTy = mlir::cast<mlir::VectorType>(V.getType());
        SmallVector<mlir::Value, 4> Lanes;
        for (mlir::Value Col : CC.getConstituents()) {
          mlir::Value Prod = mlir::spirv::FMulOp::create(B, M.getLoc(), V, Col);
          mlir::Value Sum;
          for (int64_t R = 0; R < VTy.getNumElements(); ++R) {
            auto Lane = mlir::spirv::CompositeExtractOp::create(
                B, M.getLoc(), ElemTy, Prod,
                B.getI32ArrayAttr({static_cast<int32_t>(R)}));
            Sum = Sum ? mlir::Value(mlir::spirv::FAddOp::create(
                            B, M.getLoc(), Sum, Lane.getResult()))
                      : Lane.getResult();
          }
          Lanes.push_back(Sum);
        }
        mlir::Value Res = buildVec(B, M.getLoc(), ResVecTy, Lanes);
        M.replaceAllUsesWith(Res);
        M.erase();
        Changed = true;
      }

      // MatrixTimesMatrix(CompositeConstruct, CompositeConstruct) -> CompositeConstruct.
      SmallVector<mlir::spirv::MatrixTimesMatrixOp, 8> MTMs;
      EntryFn.walk([&](mlir::spirv::MatrixTimesMatrixOp M) { MTMs.push_back(M); });
      for (auto M : MTMs) {
        auto LeftCC =
            M->getOperand(0).getDefiningOp<mlir::spirv::CompositeConstructOp>();
        auto RightCC =
            M->getOperand(1).getDefiningOp<mlir::spirv::CompositeConstructOp>();
        if (!LeftCC || !RightCC)
          continue;
        mlir::OpBuilder B(M);
        SmallVector<mlir::Value, 4> OutCols;
        for (mlir::Value RCol : RightCC.getConstituents())
          OutCols.push_back(emitMatVecMul(B, M.getLoc(), LeftCC, RCol));
        auto OutCC = mlir::spirv::CompositeConstructOp::create(
            B, M.getLoc(), M.getType(), OutCols);
        M.replaceAllUsesWith(OutCC.getResult());
        M.erase();
        Changed = true;
      }

      // CompositeInsert into CompositeConstruct (matrix/array) -> new CompositeConstruct.
      SmallVector<mlir::spirv::CompositeInsertOp, 16> Inserts;
      EntryFn.walk(
          [&](mlir::spirv::CompositeInsertOp I) { Inserts.push_back(I); });
      for (auto Ins : Inserts) {
        auto CC =
            Ins.getComposite().getDefiningOp<mlir::spirv::CompositeConstructOp>();
        if (!CC || mlir::isa<mlir::VectorType>(CC.getType()) ||
            Ins.getIndices().empty())
          continue;
        int64_t Idx = mlir::cast<mlir::IntegerAttr>(Ins.getIndices()[0]).getInt();
        if (Idx < 0 || static_cast<size_t>(Idx) >= CC.getConstituents().size())
          continue;
        mlir::OpBuilder B(Ins);
        SmallVector<mlir::Value, 4> NewElems(CC.getConstituents().begin(),
                                             CC.getConstituents().end());
        if (Ins.getIndices().size() == 1) {
          NewElems[Idx] = Ins.getObject();
        } else {
          SmallVector<mlir::Attribute, 4> Rest(Ins.getIndices().begin() + 1,
                                               Ins.getIndices().end());
          NewElems[Idx] = mlir::spirv::CompositeInsertOp::create(
                              B, Ins.getLoc(), NewElems[Idx].getType(),
                              Ins.getObject(), NewElems[Idx],
                              B.getArrayAttr(Rest))
                              .getResult();
        }
        auto NewCC = mlir::spirv::CompositeConstructOp::create(
            B, Ins.getLoc(), CC.getType(), NewElems);
        Ins.replaceAllUsesWith(NewCC.getResult());
        Ins.erase();
        Changed = true;
      }

      // CompositeExtract from CompositeConstruct (matrix/array).
      SmallVector<mlir::spirv::CompositeExtractOp, 16> Extracts;
      EntryFn.walk(
          [&](mlir::spirv::CompositeExtractOp E) { Extracts.push_back(E); });
      for (auto E : Extracts) {
        auto CC =
            E.getComposite().getDefiningOp<mlir::spirv::CompositeConstructOp>();
        if (!CC || mlir::isa<mlir::VectorType>(CC.getType()) ||
            E.getIndices().empty())
          continue;
        int64_t Idx = mlir::cast<mlir::IntegerAttr>(E.getIndices()[0]).getInt();
        if (Idx < 0 || static_cast<size_t>(Idx) >= CC.getConstituents().size())
          continue;
        mlir::Value Elem = CC.getConstituents()[Idx];
        if (E.getIndices().size() == 1) {
          E.replaceAllUsesWith(Elem);
          E.erase();
          Changed = true;
        } else {
          mlir::OpBuilder B(E);
          SmallVector<mlir::Attribute, 4> Rest(E.getIndices().begin() + 1,
                                               E.getIndices().end());
          auto NewE = mlir::spirv::CompositeExtractOp::create(
              B, E.getLoc(), E.getType(), Elem, B.getArrayAttr(Rest));
          E.replaceAllUsesWith(NewE.getResult());
          E.erase();
          Changed = true;
        }
      }

      SmallVector<mlir::spirv::CompositeConstructOp, 16> DeadCCs;
      EntryFn.walk([&](mlir::spirv::CompositeConstructOp CC) {
        if (CC->use_empty())
          DeadCCs.push_back(CC);
      });
      for (auto CC : DeadCCs) {
        CC.erase();
        Changed = true;
      }
    }
  }

  // Upstream handles aggregate insert/extract and memory operations, but not
  // construction. Use its type converter so local aggregate layouts agree,
  // augmented with signless integer/vector conversions for signed/unsigned
  // SPIR-V struct fields (e.g. si32 / vector<3xsi32>).
  mlir::LLVMTypeConverter AggregateTypes(&Ctx);
  mlir::populateSPIRVToLLVMTypeConversion(AggregateTypes);
  // Register signless integer/vector conversions AFTER populateSPIRVToLLVMTypeConversion
  // because MLIR TypeConverter evaluates conversions in LIFO order.
  AggregateTypes.addConversion([](mlir::IntegerType T) -> mlir::Type {
    if (T.isSignless())
      return T;
    return mlir::IntegerType::get(T.getContext(), T.getWidth());
  });
  AggregateTypes.addConversion([&](mlir::VectorType VT) -> mlir::Type {
    mlir::Type E = AggregateTypes.convertType(VT.getElementType());
    if (!E)
      return nullptr;
    return mlir::VectorType::get(VT.getShape(), E);
  });
  SmallVector<mlir::spirv::CompositeConstructOp, 8> AggregateConstructs;
  EntryFn.walk([&](mlir::spirv::CompositeConstructOp CC) {
    if (mlir::isa<mlir::spirv::StructType, mlir::spirv::ArrayType,
                  mlir::spirv::MatrixType>(CC.getType()))
      AggregateConstructs.push_back(CC);
  });
  for (auto CC : AggregateConstructs) {
    mlir::OpBuilder B(CC);
    auto Loc = CC.getLoc();
    auto Ty = AggregateTypes.convertType(CC.getType());
    if (!Ty)
      Ty = convertSpirvDataType(CC.getType(), B);
    if (!Ty)
      return makeError("unsupported SPIR-V aggregate construction type");
    mlir::Value Value = mlir::LLVM::UndefOp::create(B, Loc, Ty);
    for (auto [I, Elem] : llvm::enumerate(CC.getConstituents())) {
      auto ElemTy = AggregateTypes.convertType(Elem.getType());
      if (!ElemTy)
        ElemTy = convertSpirvDataType(Elem.getType(), B);
      if (!ElemTy)
        return makeError("unsupported SPIR-V aggregate element type");
      if (ElemTy != Elem.getType())
        Elem = mlir::UnrealizedConversionCastOp::create(B, Loc, ElemTy, Elem)
                   .getResult(0);
      Value = mlir::LLVM::InsertValueOp::create(
          B, Loc, Value, Elem, ArrayRef<int64_t>{static_cast<int64_t>(I)});
    }
    auto Cast = mlir::UnrealizedConversionCastOp::create(
        B, Loc, CC.getType(), Value);
    CC.replaceAllUsesWith(Cast.getResult(0));
    CC.erase();
  }

  // LLVM extractvalue cannot index vector lanes; split those from aggregates.
  SmallVector<mlir::spirv::CompositeExtractOp, 8> AggregateExtracts;
  EntryFn.walk([&](mlir::spirv::CompositeExtractOp E) {
    if (E.getIndices().size() > 1)
      AggregateExtracts.push_back(E);
  });
  for (auto E : AggregateExtracts) {
    mlir::Type Ty = E.getComposite().getType();
    for (auto [I, Index] : llvm::enumerate(E.getIndices())) {
      if (I && mlir::isa<mlir::VectorType>(Ty)) {
        mlir::OpBuilder B(E);
        auto Indices = E.getIndices().getValue();
        auto Aggregate = mlir::spirv::CompositeExtractOp::create(
            B, E.getLoc(), Ty, E.getComposite(),
            B.getArrayAttr(Indices.take_front(I)));
        auto Lane = mlir::spirv::CompositeExtractOp::create(
            B, E.getLoc(), E.getType(), Aggregate,
            B.getArrayAttr(Indices.drop_front(I)));
        E.replaceAllUsesWith(Lane.getResult());
        E.erase();
        break;
      }
      auto CompositeTy = mlir::dyn_cast<mlir::spirv::CompositeType>(Ty);
      if (!CompositeTy)
        break;
      Ty = CompositeTy.getElementType(
          mlir::cast<mlir::IntegerAttr>(Index).getInt());
    }
  }

  // Erase rewritten shader-ABI globals and entry/execution-mode ops before
  // running ConvertSPIRVToLLVM.
  for (auto GV : GlobalsToErase)
    GV.erase();

  // Emit addrspace(3) LLVM globals for Workgroup variables now that the
  // original spirv.GlobalVariable symbols have been erased.
  {
    mlir::OpBuilder GB =
        mlir::OpBuilder::atBlockBegin(&SpvModule.getBodyRegion().front());
    for (const auto &KV : WorkgroupGlobals) {
      const WorkgroupGlobalInfo &Info = KV.second;
      mlir::LLVM::GlobalOp::create(
          GB, SpvModule.getLoc(), Info.LLVMType, /*isConstant=*/false,
          mlir::LLVM::Linkage::Internal, Info.Name, /*value=*/mlir::Attribute(),
          /*alignment=*/0, /*addrSpace=*/3);
    }
  }
  SmallVector<mlir::Operation *, 4> MetaOps;
  for (auto EP : SpvModule.getOps<mlir::spirv::EntryPointOp>())
    MetaOps.push_back(EP);
  for (auto EM : SpvModule.getOps<mlir::spirv::ExecutionModeOp>())
    MetaOps.push_back(EM);
  for (mlir::Operation *Op : MetaOps)
    Op->erase();

  // 4. Lower remaining SPIR-V ops (arithmetic, constants, casts, branches) to
  // LLVM dialect via MLIR's upstream ConvertSPIRVToLLVM pass.
  mlir::PassManager PM(&Ctx);
  PM.enableVerifier(false);
  PM.addPass(mlir::createConvertSPIRVToLLVMPass());
  if (mlir::failed(PM.run(*TopMod)))
    return makeError("MLIR ConvertSPIRVToLLVM pass failed");

  // Unwrap any remaining spirv.module container by hoisting its llvm.func ops
  // into TopMod so structurizeMLIRModule sees a flat module.
  SmallVector<mlir::spirv::ModuleOp, 1> RemainingSpvMods(
      TopMod->getOps<mlir::spirv::ModuleOp>().begin(),
      TopMod->getOps<mlir::spirv::ModuleOp>().end());
  for (auto SM : RemainingSpvMods) {
    mlir::Block &SMBody = SM.getBodyRegion().front();
    SmallVector<mlir::Operation *, 8> OpsToMove;
    for (mlir::Operation &Op : SMBody.without_terminator())
      OpsToMove.push_back(&Op);
    for (mlir::Operation *Op : OpsToMove)
      Op->moveBefore(SM);
    SM.erase();
  }

  normalizeSignlessLLVMConstants(*TopMod);

  if (mlir::failed(mlir::verify(*TopMod)))
    return makeError("MLIR verification failed after SPIR-V lowering");

  LoadedMLIRShader Out;
  Out.Module = std::move(TopMod);
  Out.Reflect = std::move(Reflect);
  return std::move(Out);
}

} // namespace crucible

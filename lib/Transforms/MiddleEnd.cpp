//===- MiddleEnd.cpp - Crucible shared MLIR middle-end pipeline --*- C++ -*-===//
//
// Part of Crucible, an open-source multi-frontend, multi-backend shader compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MiddleEnd.h"
#include "crucible/Diagnostic.h"
#include "crucible/Dialect/CrucibleDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <optional>

using namespace llvm;

namespace crucible {

void normalizeSignlessLLVMConstants(mlir::ModuleOp Mod) {
  mlir::MLIRContext *Ctx = Mod.getContext();
  Mod.walk([&](mlir::LLVM::ConstantOp ConstOp) {
    mlir::Type ResTy = ConstOp.getType();
    mlir::Attribute ValAttr = ConstOp.getValue();

    if (auto IntResTy = dyn_cast<mlir::IntegerType>(ResTy)) {
      if (auto IntAttr = dyn_cast<mlir::IntegerAttr>(ValAttr)) {
        if (IntAttr.getType() != IntResTy) {
          auto AttrIntTy = dyn_cast<mlir::IntegerType>(IntAttr.getType());
          if (AttrIntTy && AttrIntTy.getWidth() == IntResTy.getWidth()) {
            APInt Bits = IntAttr.getValue();
            ConstOp.setValueAttr(mlir::IntegerAttr::get(IntResTy, Bits));
          }
        }
      }
      return;
    }

    if (auto VecTy = dyn_cast<mlir::VectorType>(ResTy)) {
      auto ElemIntTy = dyn_cast<mlir::IntegerType>(VecTy.getElementType());
      auto DenseAttr = dyn_cast<mlir::DenseIntElementsAttr>(ValAttr);
      if (!ElemIntTy || !DenseAttr)
        return;
      auto AttrElemTy =
          dyn_cast<mlir::IntegerType>(DenseAttr.getElementType());
      if (!AttrElemTy || AttrElemTy == ElemIntTy ||
          AttrElemTy.getWidth() != ElemIntTy.getWidth())
        return;
      auto TargetShapedTy =
          mlir::RankedTensorType::get(VecTy.getShape(), ElemIntTy);
      if (DenseAttr.isSplat()) {
        APInt SplatVal = DenseAttr.getSplatValue<APInt>();
        ConstOp.setValueAttr(
            mlir::DenseElementsAttr::get(TargetShapedTy, SplatVal));
      } else {
        SmallVector<APInt, 8> Values;
        Values.reserve(DenseAttr.getNumElements());
        for (const APInt &V : DenseAttr.getValues<APInt>())
          Values.push_back(V);
        ConstOp.setValueAttr(
            mlir::DenseElementsAttr::get(TargetShapedTy, Values));
      }
    }
  });
  (void)Ctx;
}

namespace {

std::optional<int64_t> getConstantIntValue(mlir::Value V) {
  if (!V)
    return std::nullopt;
  if (auto C = V.getDefiningOp<mlir::LLVM::ConstantOp>()) {
    if (auto IA = dyn_cast<mlir::IntegerAttr>(C.getValue()))
      return IA.getValue().getSExtValue();
  }
  if (auto C = V.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto IA = dyn_cast<mlir::IntegerAttr>(C.getValue()))
      return IA.getValue().getSExtValue();
  }
  return std::nullopt;
}

ResourceBinding *findCBufferBinding(Reflection &Reflect,
                                    ir::BindResourceOp BindOp) {
  if (BindOp.getResourceClass() != ir::ResourceClass::CBV)
    return nullptr;

  if (auto RecId = BindOp.getRecordId()) {
    uint32_t Id = static_cast<uint32_t>(*RecId);
    for (ResourceBinding &RB : Reflect.Resources) {
      if (RB.Class == ResourceClass::CBV && RB.RecordID == Id)
        return &RB;
    }
  }
  if (auto Space = BindOp.getSpace(); auto Binding = BindOp.getBinding()) {
    uint32_t Sp = static_cast<uint32_t>(*Space);
    uint32_t Reg = static_cast<uint32_t>(*Binding);
    for (ResourceBinding &RB : Reflect.Resources) {
      if (RB.Class == ResourceClass::CBV && RB.Space == Sp &&
          RB.LowerBound <= Reg &&
          (RB.RangeSize == 0xFFFFFFFFu || Reg < RB.LowerBound + RB.RangeSize))
        return &RB;
    }
  }
  return nullptr;
}

Error normalizeAndValidateReflection(mlir::ModuleOp Mod, Reflection &Reflect) {
  for (const ResourceBinding &RB : Reflect.Resources) {
    if (RB.RangeSize == 0)
      return makeError(
          "middle-end reflection error: resource '{0}' has zero RangeSize",
          RB.Name);
  }

  // Ensure every CBuffer's ElementStride (byte size) covers all constant
  // 16-byte register indices accessed by crucible.cbuffer_load ops.
  Mod.walk([&](ir::CBufferLoadOp LoadOp) {
    auto BindOp = LoadOp.getHandle().getDefiningOp<ir::BindResourceOp>();
    if (!BindOp)
      return;
    ResourceBinding *RB = findCBufferBinding(Reflect, BindOp);
    if (!RB || RB->Kind != ResourceKind::CBuffer)
      return;
    auto RegIdx = getConstantIntValue(LoadOp.getRegIndex());
    if (!RegIdx || *RegIdx < 0)
      return;
    uint64_t MinBytes = (static_cast<uint64_t>(*RegIdx) + 1u) * 16u;
    if (MinBytes <= UINT32_MAX && MinBytes > RB->ElementStride)
      RB->ElementStride = static_cast<uint32_t>(MinBytes);
  });

  return Error::success();
}

Error verifyCrucibleContract(mlir::ModuleOp Mod) {
  std::string LeakedOp;
  Mod.walk([&](mlir::Operation *Op) {
    if (!LeakedOp.empty())
      return;
    if (Op->getName().getDialectNamespace() == "spirv")
      LeakedOp = Op->getName().getStringRef().str();
  });
  if (!LeakedOp.empty())
    return makeError("middle-end contract violation: leaked SPIR-V op '{0}' "
                     "must be lowered before backend emission",
                     LeakedOp);

  if (mlir::failed(mlir::verify(Mod)))
    return makeError("middle-end contract violation: MLIR verification failed");

  return Error::success();
}

mlir::Value makeFloatConst(mlir::OpBuilder &B, mlir::Location Loc,
                           mlir::Type Ty, double Val) {
  if (auto FTy = dyn_cast<mlir::FloatType>(Ty)) {
    APFloat F(Val);
    bool LosesInfo = false;
    F.convert(FTy.getFloatSemantics(), APFloat::rmNearestTiesToEven,
              &LosesInfo);
    return mlir::LLVM::ConstantOp::create(B, Loc, FTy,
                                          B.getFloatAttr(FTy, F));
  }
  if (auto VecTy = dyn_cast<mlir::VectorType>(Ty)) {
    auto FTy = cast<mlir::FloatType>(VecTy.getElementType());
    APFloat F(Val);
    bool LosesInfo = false;
    F.convert(FTy.getFloatSemantics(), APFloat::rmNearestTiesToEven,
              &LosesInfo);
    auto ShapedTy = mlir::RankedTensorType::get(VecTy.getShape(), FTy);
    auto Attr = mlir::DenseElementsAttr::get(ShapedTy, F);
    return mlir::LLVM::ConstantOp::create(B, Loc, VecTy, Attr);
  }
  return nullptr;
}

/// Decompose math.atan2(y, x) into math.atan + quadrant adjustment for targets
/// without a native atan2 instruction (e.g., DXIL).
void decomposeAtan2(mlir::ModuleOp Mod) {
  SmallVector<mlir::math::Atan2Op, 8> Ops;
  Mod.walk([&](mlir::math::Atan2Op Op) { Ops.push_back(Op); });
  if (Ops.empty())
    return;

  constexpr double kPi = 3.14159265358979323846;
  constexpr double kHalfPi = 1.57079632679489661923;

  for (mlir::math::Atan2Op Op : Ops) {
    mlir::Type Ty = Op.getType();
    mlir::Type ElemTy = Ty;
    if (auto VecTy = dyn_cast<mlir::VectorType>(Ty))
      ElemTy = VecTy.getElementType();
    if (!isa<mlir::FloatType>(ElemTy))
      continue;

    mlir::OpBuilder B(Op);
    mlir::Location Loc = Op.getLoc();
    mlir::Value Y = Op.getLhs();
    mlir::Value X = Op.getRhs();

    mlir::Value Zero = makeFloatConst(B, Loc, Ty, 0.0);
    mlir::Value One = makeFloatConst(B, Loc, Ty, 1.0);
    mlir::Value Pi = makeFloatConst(B, Loc, Ty, kPi);
    mlir::Value HalfPi = makeFloatConst(B, Loc, Ty, kHalfPi);
    mlir::Value NegHalfPi = makeFloatConst(B, Loc, Ty, -kHalfPi);

    mlir::Value XEq0 = mlir::LLVM::FCmpOp::create(
        B, Loc, mlir::LLVM::FCmpPredicate::oeq, X, Zero);
    mlir::Value SafeX = mlir::LLVM::SelectOp::create(B, Loc, XEq0, One, X);
    mlir::Value Ratio = mlir::LLVM::FDivOp::create(B, Loc, Y, SafeX);
    mlir::Value Base = mlir::math::AtanOp::create(B, Loc, Ratio);

    mlir::Value XLt0 = mlir::LLVM::FCmpOp::create(
        B, Loc, mlir::LLVM::FCmpPredicate::olt, X, Zero);
    mlir::Value YGe0 = mlir::LLVM::FCmpOp::create(
        B, Loc, mlir::LLVM::FCmpPredicate::oge, Y, Zero);
    mlir::Value BasePlusPi = mlir::LLVM::FAddOp::create(B, Loc, Base, Pi);
    mlir::Value BaseMinusPi = mlir::LLVM::FSubOp::create(B, Loc, Base, Pi);
    mlir::Value NegXRes =
        mlir::LLVM::SelectOp::create(B, Loc, YGe0, BasePlusPi, BaseMinusPi);
    mlir::Value NonZeroXRes =
        mlir::LLVM::SelectOp::create(B, Loc, XLt0, NegXRes, Base);

    mlir::Value YGt0 = mlir::LLVM::FCmpOp::create(
        B, Loc, mlir::LLVM::FCmpPredicate::ogt, Y, Zero);
    mlir::Value YLt0 = mlir::LLVM::FCmpOp::create(
        B, Loc, mlir::LLVM::FCmpPredicate::olt, Y, Zero);
    mlir::Value ZeroXNegOrZero =
        mlir::LLVM::SelectOp::create(B, Loc, YLt0, NegHalfPi, Zero);
    mlir::Value ZeroXRes =
        mlir::LLVM::SelectOp::create(B, Loc, YGt0, HalfPi, ZeroXNegOrZero);

    mlir::Value Res =
        mlir::LLVM::SelectOp::create(B, Loc, XEq0, ZeroXRes, NonZeroXRes);

    Op.getResult().replaceAllUsesWith(Res);
    Op.erase();
  }
}

/// Decompose math.ctlz / math.cttz and llvm.intr.ctlz / llvm.intr.cttz on
/// scalar i32 into crucible.first_bit (DXIL firstbit_hi / firstbit_lo) with
/// explicit zero handling.
void decomposeCtlzCttz(mlir::ModuleOp Mod) {
  SmallVector<std::pair<mlir::Operation *, ir::FirstBitKind>, 8> Work;
  Mod.walk([&](mlir::Operation *Op) {
    if (isa<mlir::math::CountLeadingZerosOp, mlir::LLVM::CountLeadingZerosOp>(
            Op))
      Work.push_back({Op, ir::FirstBitKind::Hi});
    else if (isa<mlir::math::CountTrailingZerosOp,
                 mlir::LLVM::CountTrailingZerosOp>(Op))
      Work.push_back({Op, ir::FirstBitKind::Lo});
  });

  for (auto [Op, Kind] : Work) {
    mlir::Value In = Op->getOperand(0);
    mlir::Type Ty = In.getType();
    auto emitScalar = [&](mlir::OpBuilder &B, mlir::Location Loc,
                          mlir::Value Val,
                          mlir::IntegerType IntTy) -> mlir::Value {
      mlir::Value Zero =
          mlir::LLVM::ConstantOp::create(B, Loc, IntTy, B.getI32IntegerAttr(0));
      mlir::Value Width = mlir::LLVM::ConstantOp::create(
          B, Loc, IntTy, B.getI32IntegerAttr(32));
      mlir::Value First = ir::FirstBitOp::create(
          B, Loc, IntTy, Val, ir::FirstBitKindAttr::get(B.getContext(), Kind));
      mlir::Value IsZero = mlir::LLVM::ICmpOp::create(
          B, Loc, mlir::LLVM::ICmpPredicate::eq, Val, Zero);
      return mlir::LLVM::SelectOp::create(B, Loc, IsZero, Width, First);
    };

    if (auto IntTy = dyn_cast<mlir::IntegerType>(Ty)) {
      if (IntTy.getWidth() != 32)
        continue;
      mlir::OpBuilder B(Op);
      mlir::Value Res = emitScalar(B, Op->getLoc(), In, IntTy);
      Op->getResult(0).replaceAllUsesWith(Res);
      Op->erase();
    } else if (auto VecTy = dyn_cast<mlir::VectorType>(Ty)) {
      auto ElemTy = dyn_cast<mlir::IntegerType>(VecTy.getElementType());
      if (!ElemTy || ElemTy.getWidth() != 32 || VecTy.getRank() != 1)
        continue;
      mlir::OpBuilder B(Op);
      mlir::Location Loc = Op->getLoc();
      mlir::Value Acc = mlir::LLVM::PoisonOp::create(B, Loc, VecTy);
      for (int64_t I = 0; I < VecTy.getNumElements(); ++I) {
        mlir::Value Idx = mlir::LLVM::ConstantOp::create(
            B, Loc, ElemTy, B.getI32IntegerAttr(static_cast<int32_t>(I)));
        mlir::Value Lane =
            mlir::LLVM::ExtractElementOp::create(B, Loc, In, Idx);
        mlir::Value OutLane = emitScalar(B, Loc, Lane, ElemTy);
        Acc = mlir::LLVM::InsertElementOp::create(B, Loc, Acc, OutLane, Idx);
      }
      Op->getResult(0).replaceAllUsesWith(Acc);
      Op->erase();
    }
  }
}

/// Decompose math.fma / llvm.intr.fma on unsupported float widths (e.g.
/// f16/f32 on DXIL, where dx.op.fma only supports f64) into llvm.fmul +
/// llvm.fadd.
void decomposeFma(mlir::ModuleOp Mod, bool LowerF16, bool LowerF32) {
  SmallVector<mlir::Operation *, 8> Ops;
  Mod.walk([&](mlir::Operation *Op) {
    if (isa<mlir::math::FmaOp, mlir::LLVM::FMAOp>(Op))
      Ops.push_back(Op);
  });
  for (mlir::Operation *Op : Ops) {
    mlir::Type Ty = Op->getResult(0).getType();
    mlir::Type ElemTy = Ty;
    if (auto VecTy = dyn_cast<mlir::VectorType>(Ty))
      ElemTy = VecTy.getElementType();
    bool ShouldLower =
        (LowerF16 && ElemTy.isF16()) || (LowerF32 && ElemTy.isF32());
    if (!ShouldLower)
      continue;
    mlir::OpBuilder B(Op);
    mlir::Location Loc = Op->getLoc();
    mlir::Value Prod = mlir::LLVM::FMulOp::create(B, Loc, Op->getOperand(0),
                                                  Op->getOperand(1));
    mlir::Value Sum =
        mlir::LLVM::FAddOp::create(B, Loc, Prod, Op->getOperand(2));
    Op->getResult(0).replaceAllUsesWith(Sum);
    Op->erase();
  }
}

} // namespace

Error runMiddleEnd(mlir::ModuleOp Mod, Reflection &Reflect,
                   const TargetCaps &Caps) {
  // 1. Normalize frontend IR quirks once before verification or emission.
  normalizeSignlessLLVMConstants(Mod);

  // 2. Normalize and validate Reflection invariants once for all backends.
  if (Error E = normalizeAndValidateReflection(Mod, Reflect))
    return E;

  // 3. Enforce the Layer 0/1/2 MLIR contract once for all backends.
  if (Error E = verifyCrucibleContract(Mod))
    return E;

  // 4. Capability-driven legalizations.
  if (!Caps.HasNativeAtan2)
    decomposeAtan2(Mod);
  if (!Caps.HasNativeCtlzCttz)
    decomposeCtlzCttz(Mod);
  if (!Caps.HasNativeF16Fma || !Caps.HasNativeF32Fma)
    decomposeFma(Mod, !Caps.HasNativeF16Fma, !Caps.HasNativeF32Fma);

  return Error::success();
}

} // namespace crucible

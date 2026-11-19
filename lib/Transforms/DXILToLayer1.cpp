//===- DXILToLayer1.cpp - Lower DXIL ops to Layer 1 gpu/math MLIR -*- C++ -*-=//
//
// Part of Crucible, an open-source shader-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Converges the DXIL frontend onto Crucible's frontend-agnostic Layer 1 MLIR
// contract (gpu.* + math.* + llvm.intr.*) so DXIL, SPIR-V, and future compute
// frontends (e.g. PTX) share a single MSL emission path.
//
//===----------------------------------------------------------------------===//

#include "Transforms.h"
#include "../Backend/IntrinsicMap.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"

#include <optional>

using namespace llvm;

namespace crucible {

namespace {

std::optional<int64_t> getConstInt(mlir::Value V) {
  if (auto *Def = V.getDefiningOp()) {
    if (auto C = mlir::dyn_cast<mlir::LLVM::ConstantOp>(Def)) {
      if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue()))
        return IA.getInt();
    } else if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(Def)) {
      if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue()))
        return IA.getInt();
    }
  }
  return std::nullopt;
}

std::optional<DXILOp> getDXILOpcode(mlir::LLVM::CallOp Call) {
  auto Callee = Call.getCallee();
  if (!Callee || !Callee->starts_with("dx.op.") || Call.getArgOperands().empty())
    return std::nullopt;
  auto C = getConstInt(Call.getArgOperands()[0]);
  if (!C)
    return std::nullopt;
  return static_cast<DXILOp>(*C);
}

mlir::gpu::Dimension toGPUDim(int64_t Comp) {
  if (Comp == 1)
    return mlir::gpu::Dimension::y;
  if (Comp == 2)
    return mlir::gpu::Dimension::z;
  return mlir::gpu::Dimension::x;
}

} // namespace

void lowerDXILToLayer1(mlir::ModuleOp Mod) {
  SmallVector<mlir::LLVM::CallOp, 32> Calls;
  Mod.walk([&](mlir::LLVM::CallOp Call) {
    if (getDXILOpcode(Call))
      Calls.push_back(Call);
  });

  for (mlir::LLVM::CallOp Call : Calls) {
    auto MaybeOp = getDXILOpcode(Call);
    if (!MaybeOp)
      continue;
    DXILOp Op = *MaybeOp;
    auto Args = Call.getArgOperands();
    mlir::OpBuilder B(Call);
    mlir::Location Loc = Call.getLoc();
    auto I32Ty = B.getI32Type();

    // Compute thread/group/lane IDs -> gpu.* + arith.index_castui.
    if (Op == DXILOp::ThreadId || Op == DXILOp::GroupId ||
        Op == DXILOp::ThreadIdInGroup) {
      auto Comp = getConstInt(Args[1]);
      if (!Comp || *Comp < 0 || *Comp > 2)
        continue;
      mlir::gpu::Dimension Dim = toGPUDim(*Comp);
      mlir::Value IdxVal =
          Op == DXILOp::ThreadId
              ? mlir::Value(mlir::gpu::GlobalIdOp::create(B, Loc, Dim))
          : Op == DXILOp::GroupId
              ? mlir::Value(mlir::gpu::BlockIdOp::create(B, Loc, Dim))
              : mlir::Value(mlir::gpu::ThreadIdOp::create(B, Loc, Dim));
      auto Cast = mlir::arith::IndexCastUIOp::create(
          B, Loc, Call.getResult().getType(), IdxVal);
      Call.getResult().replaceAllUsesWith(Cast.getResult());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::WaveGetLaneIndex || Op == DXILOp::WaveGetLaneCount) {
      mlir::Value IdxVal =
          Op == DXILOp::WaveGetLaneIndex
              ? mlir::Value(
                    mlir::gpu::LaneIdOp::create(B, Loc, /*upper_bound=*/nullptr))
              : mlir::Value(mlir::gpu::SubgroupSizeOp::create(
                    B, Loc, /*upper_bound=*/nullptr));
      auto Cast = mlir::arith::IndexCastUIOp::create(
          B, Loc, Call.getResult().getType(), IdxVal);
      Call.getResult().replaceAllUsesWith(Cast.getResult());
      Call.erase();
      continue;
    }

    // Barriers -> gpu.barrier (preserving DXIL mode flags via attributes).
    if (Op == DXILOp::Barrier) {
      auto Mode = getConstInt(Args[1]);
      if (!Mode || (*Mode & 1) == 0)
        continue;
      auto Bar = mlir::gpu::BarrierOp::create(B, Loc);
      if (*Mode & (2 | 4))
        Bar->setAttr("crucible.device_fence", B.getUnitAttr());
      if ((*Mode & 8) == 0)
        Bar->setAttr("crucible.no_tgsm_fence", B.getUnitAttr());
      Call.erase();
      continue;
    }

    // Wave / subgroup ops -> gpu.* (skip 64-bit/f64 where MSLEmitter uses
    // specialized emulation helpers).
    if (Op == DXILOp::WaveReadLaneFirst) {
      auto BC = mlir::gpu::SubgroupBroadcastOp::create(
          B, Loc, Args[1], /*subgroupId=*/nullptr,
          mlir::gpu::BroadcastType::first_active_lane);
      Call.getResult().replaceAllUsesWith(BC.getResult());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::WaveReadLaneAt) {
      auto Width = mlir::arith::ConstantIntOp::create(B, Loc, 32, 32);
      auto Shuf = mlir::gpu::ShuffleOp::create(B, Loc, Args[1], Args[2], Width,
                                               mlir::gpu::ShuffleMode::IDX);
      Call.getResult().replaceAllUsesWith(Shuf.getShuffleResult());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::WaveActiveOp || Op == DXILOp::WavePrefixOp) {
      mlir::Type ValTy = Args[1].getType();
      // Keep 64-bit wave ops on MSLEmitter's specialized emulation path.
      if (ValTy.isInteger(64) || ValTy.isF64())
        continue;
      auto Kind = getConstInt(Args[2]);
      auto SignedOp = getConstInt(Args[3]);
      if (!Kind || !SignedOp)
        continue;
      bool IsFloat = mlir::isa<mlir::FloatType>(ValTy);
      bool IsSigned = !IsFloat && (*SignedOp == 0);
      bool IsPrefix = Op == DXILOp::WavePrefixOp;

      std::optional<mlir::gpu::AllReduceOperation> RedOp;
      switch (static_cast<DXILWaveOpKind>(*Kind)) {
      case DXILWaveOpKind::Sum:
        RedOp = mlir::gpu::AllReduceOperation::ADD;
        break;
      case DXILWaveOpKind::Product:
        RedOp = mlir::gpu::AllReduceOperation::MUL;
        break;
      case DXILWaveOpKind::Min:
        if (!IsPrefix)
          RedOp = IsFloat    ? mlir::gpu::AllReduceOperation::MINNUMF
                  : IsSigned ? mlir::gpu::AllReduceOperation::MINSI
                             : mlir::gpu::AllReduceOperation::MINUI;
        break;
      case DXILWaveOpKind::Max:
        if (!IsPrefix)
          RedOp = IsFloat    ? mlir::gpu::AllReduceOperation::MAXNUMF
                  : IsSigned ? mlir::gpu::AllReduceOperation::MAXSI
                             : mlir::gpu::AllReduceOperation::MAXUI;
        break;
      }
      if (!RedOp)
        continue;
      auto Red = mlir::gpu::SubgroupReduceOp::create(B, Loc, Args[1], *RedOp,
                                                     /*uniform=*/false);
      if (IsPrefix)
        Red->setAttr("crucible.scan_kind", B.getStringAttr("exclusive"));
      Call.getResult().replaceAllUsesWith(Red.getResult());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::WaveActiveBit) {
      mlir::Type ValTy = Args[1].getType();
      if (ValTy.isInteger(64))
        continue;
      auto Kind = getConstInt(Args[2]);
      if (!Kind)
        continue;
      std::optional<mlir::gpu::AllReduceOperation> RedOp;
      switch (static_cast<DXILWaveBitOpKind>(*Kind)) {
      case DXILWaveBitOpKind::And:
        RedOp = mlir::gpu::AllReduceOperation::AND;
        break;
      case DXILWaveBitOpKind::Or:
        RedOp = mlir::gpu::AllReduceOperation::OR;
        break;
      case DXILWaveBitOpKind::Xor:
        RedOp = mlir::gpu::AllReduceOperation::XOR;
        break;
      }
      if (!RedOp)
        continue;
      auto Red = mlir::gpu::SubgroupReduceOp::create(B, Loc, Args[1], *RedOp,
                                                     /*uniform=*/false);
      Call.getResult().replaceAllUsesWith(Red.getResult());
      Call.erase();
      continue;
    }

    // Scalar math & bit ops -> math.* / llvm.intr.* (skip f64 so --emulate-f64
    // keeps its software IEEE-754 path).
    if (Call.getNumResults() != 1)
      continue;
    mlir::Type ResTy = Call.getResult().getType();
    if (ResTy.isF64())
      continue;

    mlir::Value Rep;
    switch (Op) {
    case DXILOp::FAbs:
      Rep = mlir::math::AbsFOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::Sqrt:
      Rep = mlir::math::SqrtOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::Rsqrt:
      Rep = mlir::math::RsqrtOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::Sin:
      Rep = mlir::math::SinOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::Cos:
      Rep = mlir::math::CosOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::Tan:
      Rep = mlir::math::TanOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::Exp:
      Rep = mlir::math::Exp2Op::create(B, Loc, Args[1]);
      break;
    case DXILOp::Log:
      Rep = mlir::math::Log2Op::create(B, Loc, Args[1]);
      break;
    case DXILOp::RoundNE:
      Rep = mlir::math::RoundEvenOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::RoundNI:
      Rep = mlir::math::FloorOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::RoundPI:
      Rep = mlir::math::CeilOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::RoundZ:
      Rep = mlir::math::TruncOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::Fma:
      Rep = mlir::math::FmaOp::create(B, Loc, Args[1], Args[2], Args[3]);
      break;
    case DXILOp::Countbits:
      if (Args[1].getType() == I32Ty)
        Rep = mlir::math::CtPopOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::Bfrev:
      Rep = mlir::LLVM::BitReverseOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::FMin:
      Rep = mlir::LLVM::MinNumOp::create(B, Loc, Args[1], Args[2]);
      break;
    case DXILOp::FMax:
      Rep = mlir::LLVM::MaxNumOp::create(B, Loc, Args[1], Args[2]);
      break;
    case DXILOp::IMin:
      Rep = mlir::LLVM::SMinOp::create(B, Loc, Args[1], Args[2]);
      break;
    case DXILOp::IMax:
      Rep = mlir::LLVM::SMaxOp::create(B, Loc, Args[1], Args[2]);
      break;
    case DXILOp::UMin:
      Rep = mlir::LLVM::UMinOp::create(B, Loc, Args[1], Args[2]);
      break;
    case DXILOp::UMax:
      Rep = mlir::LLVM::UMaxOp::create(B, Loc, Args[1], Args[2]);
      break;
    default:
      break;
    }

    if (Rep) {
      Call.getResult().replaceAllUsesWith(Rep);
      Call.erase();
    }
  }
}

} // namespace crucible

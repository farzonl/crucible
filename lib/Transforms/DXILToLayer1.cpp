//===- DXILToLayer1.cpp - Lower DXIL ops to Layer 1 + crucible MLIR -*- C++ -*-===//
//
// Part of Crucible, an open-source shader-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Converges the DXIL frontend onto Crucible's frontend-agnostic MLIR contract
// (gpu.* + math.* + llvm.intr.* + crucible.*) so DXIL, SPIR-V, and future
// frontends share a single backend representation without DXILOp leakage.
//
//===----------------------------------------------------------------------===//

#include "Transforms.h"
#include "DXILOp.h"
#include "crucible/Dialect/CrucibleDialect.h"

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

/// Returns true if all users of `V` are single-index `llvm.extractvalue` ops.
bool allSingleIndexExtractValues(mlir::Value V) {
  for (mlir::Operation *User : V.getUsers()) {
    auto EV = mlir::dyn_cast<mlir::LLVM::ExtractValueOp>(User);
    if (!EV || EV.getPosition().size() != 1)
      return false;
  }
  return true;
}

/// Replace all `llvm.extractvalue %agg[k]` users with `llvm.extractelement %vec[%k]`.
void replaceStructUsersWithVectorExtracts(mlir::Value Agg, mlir::Value Vec) {
  SmallVector<mlir::LLVM::ExtractValueOp, 4> Users;
  for (mlir::Operation *U : Agg.getUsers())
    Users.push_back(mlir::cast<mlir::LLVM::ExtractValueOp>(U));
  for (auto EV : Users) {
    mlir::OpBuilder B(EV);
    int32_t Idx = static_cast<int32_t>(EV.getPosition()[0]);
    auto IdxC = mlir::LLVM::ConstantOp::create(B, EV.getLoc(), B.getI32Type(),
                                               B.getI32IntegerAttr(Idx));
    auto EE = mlir::LLVM::ExtractElementOp::create(B, EV.getLoc(), Vec, IdxC);
    EV.getResult().replaceAllUsesWith(EE.getResult());
    EV.erase();
  }
}

/// Replace all `llvm.extractvalue %agg[0/1]` users with two direct values.
void replacePairStructUsers(mlir::Value Agg, mlir::Value R0, mlir::Value R1) {
  SmallVector<mlir::LLVM::ExtractValueOp, 4> Users;
  for (mlir::Operation *U : Agg.getUsers())
    Users.push_back(mlir::cast<mlir::LLVM::ExtractValueOp>(U));
  for (auto EV : Users) {
    int64_t Idx = EV.getPosition()[0];
    EV.getResult().replaceAllUsesWith(Idx == 0 ? R0 : R1);
    EV.erase();
  }
}

bool isUndefOrPoison(mlir::Value V) {
  if (!V)
    return true;
  return mlir::isa_and_nonnull<mlir::LLVM::UndefOp, mlir::LLVM::PoisonOp>(
      V.getDefiningOp());
}

bool hasStatusExtract(mlir::Value Agg) {
  for (mlir::Operation *U : Agg.getUsers()) {
    if (auto EV = mlir::dyn_cast<mlir::LLVM::ExtractValueOp>(U))
      if (EV.getPosition().size() == 1 && EV.getPosition()[0] == 4)
        return true;
  }
  return false;
}

/// Replace `dx.types.ResRet` extractvalues: 0..3 -> extractelement on Vec,
/// 4 -> Status.
void replaceResRetUsers(mlir::Value Agg, mlir::Value Vec, mlir::Value Status) {
  SmallVector<mlir::LLVM::ExtractValueOp, 8> Users;
  for (mlir::Operation *U : Agg.getUsers())
    Users.push_back(mlir::cast<mlir::LLVM::ExtractValueOp>(U));
  for (auto EV : Users) {
    int64_t Idx = EV.getPosition()[0];
    if (Idx == 4) {
      EV.getResult().replaceAllUsesWith(Status);
      EV.erase();
      continue;
    }
    mlir::OpBuilder B(EV);
    auto IdxC = mlir::LLVM::ConstantOp::create(
        B, EV.getLoc(), B.getI32Type(),
        B.getI32IntegerAttr(static_cast<int32_t>(Idx)));
    auto EE = mlir::LLVM::ExtractElementOp::create(B, EV.getLoc(), Vec, IdxC);
    EV.getResult().replaceAllUsesWith(EE.getResult());
    EV.erase();
  }
}

SmallVector<mlir::Value, 4> collectDefinedValues(mlir::ValueRange Range) {
  SmallVector<mlir::Value, 4> Out;
  for (mlir::Value V : Range) {
    if (!isUndefOrPoison(V))
      Out.push_back(V);
  }
  return Out;
}

} // namespace

void lowerDXILToLayer1(mlir::ModuleOp Mod) {
  SmallVector<mlir::LLVM::CallOp, 64> Calls;
  Mod.walk([&](mlir::LLVM::CallOp Call) {
    if (getDXILOpcode(Call))
      Calls.push_back(Call);
  });

  // Pass 1: Lower resource handle producers and RayQuery allocators into
  // crucible dialect handles, keeping old ops alive until Pass 3 cleanup.
  DenseMap<mlir::Value, mlir::Value> HandleMap;
  DenseMap<mlir::Value, mlir::Value> RayQueryMap;
  SmallVector<mlir::LLVM::CallOp, 32> DeferredErase;

  for (mlir::LLVM::CallOp Call : Calls) {
    auto MaybeOp = getDXILOpcode(Call);
    if (!MaybeOp)
      continue;
    DXILOp Op = *MaybeOp;
    auto Args = Call.getArgOperands();
    mlir::OpBuilder B(Call);
    mlir::Location Loc = Call.getLoc();
    auto ResTy = ir::ResourceType::get(B.getContext());

    if (Op == DXILOp::CreateHandle) {
      auto ClassC = getConstInt(Args[1]);
      auto RangeC = getConstInt(Args[2]);
      if (!ClassC || !RangeC)
        continue;
      bool NonUniform = getConstInt(Args[4]).value_or(0) != 0;
      auto H = ir::BindResourceOp::create(
          B, Loc, ResTy,
          ir::ResourceClassAttr::get(B.getContext(),
                                     static_cast<ir::ResourceClass>(*ClassC)),
          B.getI32IntegerAttr(static_cast<int32_t>(*RangeC)),
          /*space=*/nullptr, /*binding=*/nullptr, B.getBoolAttr(NonUniform),
          Args[3]);
      HandleMap[Call.getResult()] = H.getResult();
      DeferredErase.push_back(Call);
      continue;
    }

    if (Op == DXILOp::CreateHandleFromBinding) {
      uint32_t LB = 0, Space = 0, Class = 0;
      mlir::Operation *Cur = Args[1].getDefiningOp();
      while (auto IV = mlir::dyn_cast_or_null<mlir::LLVM::InsertValueOp>(Cur)) {
        auto Pos = IV.getPosition();
        if (Pos.size() == 1) {
          if (auto CV = getConstInt(IV.getValue())) {
            if (Pos[0] == 0)
              LB = static_cast<uint32_t>(*CV);
            else if (Pos[0] == 2)
              Space = static_cast<uint32_t>(*CV);
            else if (Pos[0] == 3)
              Class = static_cast<uint32_t>(*CV);
          }
        }
        Cur = IV.getContainer().getDefiningOp();
      }
      bool NonUniform = getConstInt(Args[3]).value_or(0) != 0;
      auto H = ir::BindResourceOp::create(
          B, Loc, ResTy,
          ir::ResourceClassAttr::get(B.getContext(),
                                     static_cast<ir::ResourceClass>(Class)),
          /*record_id=*/nullptr,
          B.getI32IntegerAttr(static_cast<int32_t>(Space)),
          B.getI32IntegerAttr(static_cast<int32_t>(LB)),
          B.getBoolAttr(NonUniform), Args[2]);
      HandleMap[Call.getResult()] = H.getResult();
      DeferredErase.push_back(Call);
      continue;
    }

    if (Op == DXILOp::CreateHandleFromHeap) {
      auto IsSamp = getConstInt(Args[2]);
      if (!IsSamp)
        continue;
      bool NonUniform = getConstInt(Args[3]).value_or(0) != 0;
      uint32_t PropsW0 = 0, PropsW1 = 0;
      for (mlir::Operation *U : Call.getResult().getUsers()) {
        auto AC = mlir::dyn_cast<mlir::LLVM::CallOp>(U);
        if (!AC || getDXILOpcode(AC) != DXILOp::AnnotateHandle)
          continue;
        mlir::Operation *Cur = AC.getArgOperands()[2].getDefiningOp();
        while (auto IV =
                   mlir::dyn_cast_or_null<mlir::LLVM::InsertValueOp>(Cur)) {
          auto Pos = IV.getPosition();
          if (Pos.size() == 1) {
            if (auto CV = getConstInt(IV.getValue())) {
              if (Pos[0] == 0)
                PropsW0 = static_cast<uint32_t>(*CV);
              else if (Pos[0] == 1)
                PropsW1 = static_cast<uint32_t>(*CV);
            }
          }
          Cur = IV.getContainer().getDefiningOp();
        }
        break;
      }
      uint32_t Kind = PropsW0 & 0xFFu;
      bool IsUAV = (PropsW0 & 0x1000u) != 0;
      bool HasCounter = (PropsW0 & 0x8000u) != 0;
      ir::ResourceClass RC = *IsSamp       ? ir::ResourceClass::Sampler
                             : Kind == 13  ? ir::ResourceClass::CBV
                             : IsUAV       ? ir::ResourceClass::UAV
                                           : ir::ResourceClass::SRV;
      auto H = ir::HeapResourceOp::create(
          B, Loc, ResTy, Args[1], B.getBoolAttr(*IsSamp != 0),
          B.getBoolAttr(NonUniform),
          ir::ResourceClassAttr::get(B.getContext(), RC),
          Kind ? B.getI32IntegerAttr(static_cast<int32_t>(Kind)) : nullptr,
          PropsW1 ? B.getI32IntegerAttr(static_cast<int32_t>(PropsW1))
                  : nullptr,
          B.getBoolAttr(HasCounter));
      HandleMap[Call.getResult()] = H.getResult();
      DeferredErase.push_back(Call);
      continue;
    }

    if (Op == DXILOp::AllocateRayQuery) {
      auto RQTy = ir::RayQueryType::get(B.getContext());
      auto RQ = ir::RayQueryAllocOp::create(B, Loc, RQTy, Args[1]);
      RayQueryMap[Call.getResult()] = RQ.getResult();
      DeferredErase.push_back(Call);
      continue;
    }
  }

  // Propagate AnnotateHandle chains into HandleMap.
  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (mlir::LLVM::CallOp Call : Calls) {
      if (getDXILOpcode(Call) != DXILOp::AnnotateHandle)
        continue;
      if (HandleMap.count(Call.getResult()))
        continue;
      auto It = HandleMap.find(Call.getArgOperands()[1]);
      if (It != HandleMap.end()) {
        HandleMap[Call.getResult()] = It->second;
        DeferredErase.push_back(Call);
        Changed = true;
      }
    }
  }

  auto lookupHandle = [&](mlir::Value V) -> mlir::Value {
    auto It = HandleMap.find(V);
    return It != HandleMap.end() ? It->second : mlir::Value();
  };
  auto lookupRQ = [&](mlir::Value V) -> mlir::Value {
    auto It = RayQueryMap.find(V);
    return It != RayQueryMap.end() ? It->second : mlir::Value();
  };

  for (mlir::LLVM::CallOp Call : Calls) {
    auto MaybeOp = getDXILOpcode(Call);
    if (!MaybeOp)
      continue;
    DXILOp Op = *MaybeOp;
    auto Args = Call.getArgOperands();
    mlir::OpBuilder B(Call);
    mlir::Location Loc = Call.getLoc();
    auto I32Ty = B.getI32Type();

    //===------------------------------------------------------------------===//
    // Compute thread/group/lane IDs -> gpu.* / crucible.sysval
    //===------------------------------------------------------------------===//
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

    if (Op == DXILOp::FlattenedThreadIdInGroup) {
      auto SV = ir::SysValOp::create(
          B, Loc, Call.getResult().getType(),
          ir::SysValKindAttr::get(B.getContext(),
                                  ir::SysValKind::FlattenedThreadIdInGroup),
          /*component=*/nullptr);
      Call.getResult().replaceAllUsesWith(SV.getResult());
      Call.erase();
      continue;
    }

    //===------------------------------------------------------------------===//
    // Stage I/O, Pixel, Tessellation, Geometry & RT System Values -> crucible.*
    //===------------------------------------------------------------------===//
    if (Op == DXILOp::LoadInput) {
      auto ID = getConstInt(Args[1]);
      auto Col = getConstInt(Args[3]);
      if (!ID || !Col)
        continue;
      mlir::Value VtxId;
      if (Args.size() > 4 &&
          !mlir::isa_and_nonnull<mlir::LLVM::UndefOp, mlir::LLVM::PoisonOp>(
              Args[4].getDefiningOp()))
        VtxId = Args[4];
      auto LI = ir::LoadInputOp::create(B, Loc, Call.getResult().getType(),
                                        static_cast<int32_t>(*ID), Args[2],
                                        static_cast<int8_t>(*Col), VtxId);
      Call.getResult().replaceAllUsesWith(LI.getResult());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::StoreOutput) {
      auto ID = getConstInt(Args[1]);
      auto Col = getConstInt(Args[3]);
      if (!ID || !Col)
        continue;
      ir::StoreOutputOp::create(B, Loc, static_cast<int32_t>(*ID), Args[2],
                                static_cast<int8_t>(*Col), Args[4]);
      Call.erase();
      continue;
    }

    if (Op == DXILOp::Discard) {
      ir::DiscardOp::create(B, Loc, Args[1]);
      Call.erase();
      continue;
    }

    if (Op == DXILOp::DerivCoarseX || Op == DXILOp::DerivCoarseY ||
        Op == DXILOp::DerivFineX || Op == DXILOp::DerivFineY) {
      auto Axis = (Op == DXILOp::DerivCoarseX || Op == DXILOp::DerivFineX)
                      ? ir::DerivAxis::X
                      : ir::DerivAxis::Y;
      auto Qual = (Op == DXILOp::DerivCoarseX || Op == DXILOp::DerivCoarseY)
                      ? ir::DerivQuality::Coarse
                      : ir::DerivQuality::Fine;
      auto D = ir::DerivOp::create(B, Loc, Call.getResult().getType(), Args[1],
                                   Axis, Qual);
      Call.getResult().replaceAllUsesWith(D.getResult());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::LoadOutputControlPoint) {
      auto ID = getConstInt(Args[1]);
      auto Col = getConstInt(Args[3]);
      if (!ID || !Col)
        continue;
      auto L = ir::LoadControlPointOp::create(
          B, Loc, Call.getResult().getType(), static_cast<int32_t>(*ID),
          Args[2], static_cast<int8_t>(*Col), Args[4]);
      Call.getResult().replaceAllUsesWith(L.getResult());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::LoadPatchConstant) {
      auto ID = getConstInt(Args[1]);
      auto Col = getConstInt(Args[3]);
      if (!ID || !Col)
        continue;
      auto L = ir::LoadPatchConstantOp::create(
          B, Loc, Call.getResult().getType(), static_cast<int32_t>(*ID),
          Args[2], static_cast<int8_t>(*Col));
      Call.getResult().replaceAllUsesWith(L.getResult());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::StorePatchConstant) {
      auto ID = getConstInt(Args[1]);
      auto Col = getConstInt(Args[3]);
      if (!ID || !Col)
        continue;
      ir::StorePatchConstantOp::create(B, Loc, static_cast<int32_t>(*ID),
                                       Args[2], static_cast<int8_t>(*Col),
                                       Args[4]);
      Call.erase();
      continue;
    }

    if (Op == DXILOp::EmitStream || Op == DXILOp::CutStream ||
        Op == DXILOp::EmitThenCutStream) {
      int8_t StreamId = 0;
      if (Args.size() > 1)
        if (auto S = getConstInt(Args[1]))
          StreamId = static_cast<int8_t>(*S);
      if (Op == DXILOp::EmitStream || Op == DXILOp::EmitThenCutStream)
        ir::EmitStreamOp::create(B, Loc, StreamId);
      if (Op == DXILOp::CutStream || Op == DXILOp::EmitThenCutStream)
        ir::CutStreamOp::create(B, Loc, StreamId);
      Call.erase();
      continue;
    }

    if (Op == DXILOp::PrimitiveID || Op == DXILOp::OutputControlPointID ||
        Op == DXILOp::RayTCurrent || Op == DXILOp::PrimitiveIndex) {
      auto Kind = Op == DXILOp::PrimitiveID ? ir::SysValKind::PrimitiveId
                  : Op == DXILOp::OutputControlPointID
                      ? ir::SysValKind::OutputControlPointId
                  : Op == DXILOp::RayTCurrent ? ir::SysValKind::RayTCurrent
                                              : ir::SysValKind::PrimitiveIndex;
      auto SV = ir::SysValOp::create(
          B, Loc, Call.getResult().getType(),
          ir::SysValKindAttr::get(B.getContext(), Kind), /*component=*/nullptr);
      Call.getResult().replaceAllUsesWith(SV.getResult());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::DomainLocation || Op == DXILOp::DispatchRaysIndex ||
        Op == DXILOp::DispatchRaysDimensions || Op == DXILOp::WorldRayOrigin ||
        Op == DXILOp::WorldRayDirection) {
      auto Comp = getConstInt(Args[1]);
      if (!Comp)
        continue;
      auto Kind = Op == DXILOp::DomainLocation ? ir::SysValKind::DomainLocation
                  : Op == DXILOp::DispatchRaysIndex
                      ? ir::SysValKind::DispatchRaysIndex
                  : Op == DXILOp::DispatchRaysDimensions
                      ? ir::SysValKind::DispatchRaysDimensions
                  : Op == DXILOp::WorldRayOrigin
                      ? ir::SysValKind::WorldRayOrigin
                      : ir::SysValKind::WorldRayDirection;
      auto SV = ir::SysValOp::create(
          B, Loc, Call.getResult().getType(),
          ir::SysValKindAttr::get(B.getContext(), Kind),
          B.getI32IntegerAttr(static_cast<int32_t>(*Comp)));
      Call.getResult().replaceAllUsesWith(SV.getResult());
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

    //===------------------------------------------------------------------===//
    // Barriers -> gpu.barrier / crucible.barrier
    //===------------------------------------------------------------------===//
    if (Op == DXILOp::Barrier) {
      auto Mode = getConstInt(Args[1]);
      if (!Mode)
        continue;
      bool ExecSync = (*Mode & 1) != 0;
      bool DeviceFence = (*Mode & (2 | 4)) != 0;
      bool TgsmFence = (*Mode & 8) != 0;
      if (ExecSync) {
        auto Bar = mlir::gpu::BarrierOp::create(B, Loc);
        if (DeviceFence)
          Bar->setAttr("crucible.device_fence", B.getUnitAttr());
        if (!TgsmFence)
          Bar->setAttr("crucible.no_tgsm_fence", B.getUnitAttr());
      } else {
        ir::BarrierOp::create(B, Loc, /*exec_sync=*/false, TgsmFence,
                              DeviceFence, /*handle=*/mlir::Value());
      }
      Call.erase();
      continue;
    }

    if (Op == DXILOp::BarrierByMemoryType) {
      auto Mem = getConstInt(Args[1]);
      auto Sem = getConstInt(Args[2]);
      if (!Mem || !Sem)
        continue;
      bool ExecSync = (*Sem & 1) != 0;
      bool DeviceFence = (*Mem & 1) != 0;
      bool TgsmFence = (*Mem & 2) != 0;
      ir::BarrierOp::create(B, Loc, ExecSync, TgsmFence, DeviceFence,
                            /*handle=*/mlir::Value());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::BarrierByMemoryHandle) {
      mlir::Value H = lookupHandle(Args[1]);
      auto Sem = getConstInt(Args[2]);
      if (!H || !Sem)
        continue;
      bool ExecSync = (*Sem & 1) != 0;
      ir::BarrierOp::create(B, Loc, ExecSync, /*tgsm_fence=*/false,
                            /*device_fence=*/true, H);
      Call.erase();
      continue;
    }

    //===------------------------------------------------------------------===//
    // Buffers, Constant Buffers & Atomics -> crucible.*
    //===------------------------------------------------------------------===//
    if (Op == DXILOp::CBufferLoadLegacy) {
      mlir::Value H = lookupHandle(Args[1]);
      auto RetStruct =
          mlir::dyn_cast<mlir::LLVM::LLVMStructType>(Call.getResult().getType());
      if (!H || !RetStruct || RetStruct.getBody().empty() ||
          !allSingleIndexExtractValues(Call.getResult()))
        continue;
      auto ElemTy = RetStruct.getBody()[0];
      auto VecTy = mlir::VectorType::get(
          {static_cast<int64_t>(RetStruct.getBody().size())}, ElemTy);
      auto L = ir::CBufferLoadOp::create(B, Loc, VecTy, H, Args[2]);
      replaceStructUsersWithVectorExtracts(Call.getResult(), L.getResult());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::BufferLoad || Op == DXILOp::RawBufferLoad) {
      mlir::Value H = lookupHandle(Args[1]);
      auto RetStruct =
          mlir::dyn_cast<mlir::LLVM::LLVMStructType>(Call.getResult().getType());
      if (!H || !RetStruct || RetStruct.getBody().empty() ||
          !allSingleIndexExtractValues(Call.getResult()))
        continue;
      auto ElemTy = RetStruct.getBody()[0];
      auto VecTy = mlir::VectorType::get({4}, ElemTy);
      bool NeedStatus = hasStatusExtract(Call.getResult());
      mlir::Value ElemOff = isUndefOrPoison(Args[3]) ? mlir::Value() : Args[3];
      ir::BufferKind Kind =
          Op == DXILOp::BufferLoad ? ir::BufferKind::Typed : ir::BufferKind::Raw;
      mlir::IntegerAttr MaskAttr, AlignAttr;
      if (Op == DXILOp::RawBufferLoad) {
        if (auto M = getConstInt(Args[4]))
          MaskAttr = B.getI32IntegerAttr(static_cast<int32_t>(*M));
        if (auto A = getConstInt(Args[5]))
          AlignAttr = B.getI32IntegerAttr(static_cast<int32_t>(*A));
      }
      auto L = ir::BufferLoadOp::create(
          B, Loc, VecTy, NeedStatus ? mlir::Type(I32Ty) : mlir::Type(), H,
          ir::BufferKindAttr::get(B.getContext(), Kind), Args[2], ElemOff,
          MaskAttr, AlignAttr);
      replaceResRetUsers(Call.getResult(), L.getResult(), L.getStatus());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::BufferStore || Op == DXILOp::RawBufferStore) {
      mlir::Value H = lookupHandle(Args[1]);
      auto Mask = getConstInt(Args[8]);
      if (!H || !Mask)
        continue;
      mlir::Value ElemOff = isUndefOrPoison(Args[3]) ? mlir::Value() : Args[3];
      SmallVector<mlir::Value, 4> Vals = {Args[4], Args[5], Args[6], Args[7]};
      ir::BufferKind Kind = Op == DXILOp::BufferStore ? ir::BufferKind::Typed
                                                      : ir::BufferKind::Raw;
      mlir::IntegerAttr AlignAttr;
      if (Op == DXILOp::RawBufferStore && Args.size() > 9)
        if (auto A = getConstInt(Args[9]))
          AlignAttr = B.getI32IntegerAttr(static_cast<int32_t>(*A));
      ir::BufferStoreOp::create(
          B, Loc, H, ir::BufferKindAttr::get(B.getContext(), Kind), Args[2],
          ElemOff, Vals, B.getI32IntegerAttr(static_cast<int32_t>(*Mask)),
          AlignAttr);
      Call.erase();
      continue;
    }

    if (Op == DXILOp::BufferUpdateCounter) {
      mlir::Value H = lookupHandle(Args[1]);
      auto Inc = getConstInt(Args[2]);
      if (!H || !Inc)
        continue;
      auto U = ir::BufferUpdateCounterOp::create(B, Loc, I32Ty, H,
                                                 static_cast<int8_t>(*Inc));
      Call.getResult().replaceAllUsesWith(U.getResult());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::AtomicBinOp) {
      mlir::Value H = lookupHandle(Args[1]);
      auto BinOp = getConstInt(Args[2]);
      if (!H || !BinOp || *BinOp < 0 || *BinOp > 8)
        continue;
      SmallVector<mlir::Value, 3> Coords = {Args[3], Args[4], Args[5]};
      auto A = ir::BufferAtomicOp::create(
          B, Loc, Call.getResult().getType(), H,
          ir::AtomicOpAttr::get(B.getContext(),
                                static_cast<ir::AtomicOp>(*BinOp)),
          Coords, Args[6]);
      Call.getResult().replaceAllUsesWith(A.getResult());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::AtomicCompareExchange) {
      mlir::Value H = lookupHandle(Args[1]);
      if (!H)
        continue;
      SmallVector<mlir::Value, 3> Coords = {Args[2], Args[3], Args[4]};
      auto A = ir::BufferAtomicCmpXchgOp::create(
          B, Loc, Call.getResult().getType(), H, Coords, Args[5], Args[6]);
      Call.getResult().replaceAllUsesWith(A.getResult());
      Call.erase();
      continue;
    }

    //===------------------------------------------------------------------===//
    // Textures & Sampling -> crucible.*
    //===------------------------------------------------------------------===//
    if (Op == DXILOp::TextureLoad) {
      mlir::Value H = lookupHandle(Args[1]);
      auto RetStruct =
          mlir::dyn_cast<mlir::LLVM::LLVMStructType>(Call.getResult().getType());
      if (!H || !RetStruct || RetStruct.getBody().empty() ||
          !allSingleIndexExtractValues(Call.getResult()))
        continue;
      auto ElemTy = RetStruct.getBody()[0];
      auto VecTy = mlir::VectorType::get({4}, ElemTy);
      bool NeedStatus = hasStatusExtract(Call.getResult());
      mlir::Value Mip = isUndefOrPoison(Args[2]) ? mlir::Value() : Args[2];
      SmallVector<mlir::Value, 3> Coords =
          collectDefinedValues(Args.slice(3, 3));
      SmallVector<mlir::Value, 4> Offsets;
      if (Args.size() >= 9)
        Offsets = collectDefinedValues(Args.slice(6, 3));
      auto L = ir::TexLoadOp::create(
          B, Loc, VecTy, NeedStatus ? mlir::Type(I32Ty) : mlir::Type(), H, Mip,
          Coords, Offsets);
      replaceResRetUsers(Call.getResult(), L.getResult(), L.getStatus());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::TextureStore) {
      mlir::Value H = lookupHandle(Args[1]);
      auto Mask = getConstInt(Args[9]);
      if (!H || !Mask)
        continue;
      SmallVector<mlir::Value, 3> Coords =
          collectDefinedValues(Args.slice(2, 3));
      SmallVector<mlir::Value, 4> Vals = {Args[5], Args[6], Args[7], Args[8]};
      ir::TexStoreOp::create(B, Loc, H, Coords, Vals,
                             B.getI32IntegerAttr(static_cast<int32_t>(*Mask)));
      Call.erase();
      continue;
    }

    if (Op == DXILOp::Sample || Op == DXILOp::SampleBias ||
        Op == DXILOp::SampleLevel || Op == DXILOp::SampleGrad ||
        Op == DXILOp::SampleCmp || Op == DXILOp::SampleCmpLevelZero) {
      mlir::Value Tex = lookupHandle(Args[1]);
      mlir::Value Samp = lookupHandle(Args[2]);
      auto RetStruct =
          mlir::dyn_cast<mlir::LLVM::LLVMStructType>(Call.getResult().getType());
      if (!Tex || !Samp || !RetStruct || RetStruct.getBody().empty() ||
          !allSingleIndexExtractValues(Call.getResult()))
        continue;
      auto ElemTy = RetStruct.getBody()[0];
      auto VecTy = mlir::VectorType::get({4}, ElemTy);
      bool NeedStatus = hasStatusExtract(Call.getResult());
      ir::SampleMode Mode =
          Op == DXILOp::Sample       ? ir::SampleMode::Implicit
          : Op == DXILOp::SampleBias ? ir::SampleMode::Bias
          : Op == DXILOp::SampleLevel ? ir::SampleMode::Level
          : Op == DXILOp::SampleGrad  ? ir::SampleMode::Grad
          : Op == DXILOp::SampleCmp   ? ir::SampleMode::Cmp
                                      : ir::SampleMode::CmpLevelZero;
      SmallVector<mlir::Value, 4> Coords =
          collectDefinedValues(Args.slice(3, 4));
      SmallVector<mlir::Value, 3> Offsets =
          collectDefinedValues(Args.slice(7, 3));
      mlir::Value LodOrBias, CmpVal, Clamp;
      SmallVector<mlir::Value, 3> Ddx, Ddy;
      if (Op == DXILOp::Sample) {
        if (Args.size() > 10 && !isUndefOrPoison(Args[10]))
          Clamp = Args[10];
      } else if (Op == DXILOp::SampleBias) {
        LodOrBias = Args[10];
        if (Args.size() > 11 && !isUndefOrPoison(Args[11]))
          Clamp = Args[11];
      } else if (Op == DXILOp::SampleLevel) {
        LodOrBias = Args[10];
      } else if (Op == DXILOp::SampleGrad) {
        Ddx = collectDefinedValues(Args.slice(10, 3));
        Ddy = collectDefinedValues(Args.slice(13, 3));
        if (Args.size() > 16 && !isUndefOrPoison(Args[16]))
          Clamp = Args[16];
      } else if (Op == DXILOp::SampleCmp) {
        CmpVal = Args[10];
        if (Args.size() > 11 && !isUndefOrPoison(Args[11]))
          Clamp = Args[11];
      } else if (Op == DXILOp::SampleCmpLevelZero) {
        CmpVal = Args[10];
      }
      auto S = ir::TexSampleOp::create(
          B, Loc, VecTy, NeedStatus ? mlir::Type(I32Ty) : mlir::Type(), Tex,
          Samp, ir::SampleModeAttr::get(B.getContext(), Mode), Coords, Offsets,
          LodOrBias, Ddx, Ddy, CmpVal, Clamp);
      replaceResRetUsers(Call.getResult(), S.getResult(), S.getStatus());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::TextureGather || Op == DXILOp::TextureGatherCmp) {
      mlir::Value Tex = lookupHandle(Args[1]);
      mlir::Value Samp = lookupHandle(Args[2]);
      auto Chan = getConstInt(Args[9]);
      auto RetStruct =
          mlir::dyn_cast<mlir::LLVM::LLVMStructType>(Call.getResult().getType());
      if (!Tex || !Samp || !Chan || !RetStruct || RetStruct.getBody().empty() ||
          !allSingleIndexExtractValues(Call.getResult()))
        continue;
      auto ElemTy = RetStruct.getBody()[0];
      auto VecTy = mlir::VectorType::get({4}, ElemTy);
      bool NeedStatus = hasStatusExtract(Call.getResult());
      SmallVector<mlir::Value, 4> Coords =
          collectDefinedValues(Args.slice(3, 4));
      SmallVector<mlir::Value, 2> Offsets =
          collectDefinedValues(Args.slice(7, 2));
      mlir::Value CmpVal =
          (Op == DXILOp::TextureGatherCmp && Args.size() > 10) ? Args[10]
                                                               : mlir::Value();
      auto G = ir::TexGatherOp::create(
          B, Loc, VecTy, NeedStatus ? mlir::Type(I32Ty) : mlir::Type(), Tex,
          Samp, Coords, Offsets, CmpVal,
          B.getI32IntegerAttr(static_cast<int32_t>(*Chan)));
      replaceResRetUsers(Call.getResult(), G.getResult(), G.getStatus());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::GetDimensions) {
      mlir::Value H = lookupHandle(Args[1]);
      if (!H || !allSingleIndexExtractValues(Call.getResult()))
        continue;
      auto VecTy = mlir::VectorType::get({4}, I32Ty);
      mlir::Value Mip =
          (Args.size() > 2 && !isUndefOrPoison(Args[2])) ? Args[2]
                                                         : mlir::Value();
      auto D = ir::TexGetDimensionsOp::create(B, Loc, VecTy, H, Mip);
      replaceStructUsersWithVectorExtracts(Call.getResult(), D.getResult());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::CalculateLOD) {
      mlir::Value Tex = lookupHandle(Args[1]);
      mlir::Value Samp = lookupHandle(Args[2]);
      auto Clamped = getConstInt(Args[6]);
      if (!Tex || !Samp || !Clamped)
        continue;
      SmallVector<mlir::Value, 3> Coords =
          collectDefinedValues(Args.slice(3, 3));
      auto L = ir::TexCalcLodOp::create(B, Loc, B.getF32Type(), Tex, Samp,
                                        Coords, B.getBoolAttr(*Clamped != 0));
      Call.getResult().replaceAllUsesWith(L.getResult());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::CheckAccessFullyMapped) {
      auto Zero = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 0);
      auto Cmp = mlir::LLVM::ICmpOp::create(B, Loc, mlir::LLVM::ICmpPredicate::ne,
                                            Args[1], Zero);
      Call.getResult().replaceAllUsesWith(Cmp.getResult());
      Call.erase();
      continue;
    }

    //===------------------------------------------------------------------===//
    // Mesh & Amplification Shader ops -> crucible.*
    //===------------------------------------------------------------------===//
    if (Op == DXILOp::GetMeshPayload) {
      auto P =
          ir::GetMeshPayloadOp::create(B, Loc, Call.getResult().getType());
      Call.getResult().replaceAllUsesWith(P.getResult());
      Call.erase();
      continue;
    }
    if (Op == DXILOp::SetMeshOutputCounts) {
      ir::SetMeshOutputCountsOp::create(B, Loc, Args[1], Args[2]);
      Call.erase();
      continue;
    }
    if (Op == DXILOp::EmitIndices) {
      SmallVector<mlir::Value, 3> Idxs =
          collectDefinedValues(Args.slice(2, Args.size() - 2));
      ir::EmitIndicesOp::create(B, Loc, Args[1], Idxs);
      Call.erase();
      continue;
    }
    if (Op == DXILOp::StoreVertexOutput) {
      auto ID = getConstInt(Args[1]);
      auto Col = getConstInt(Args[3]);
      if (!ID || !Col)
        continue;
      ir::StoreVertexOutputOp::create(B, Loc, static_cast<int32_t>(*ID),
                                      Args[2], static_cast<int8_t>(*Col),
                                      Args[4], Args[5]);
      Call.erase();
      continue;
    }
    if (Op == DXILOp::StorePrimitiveOutput) {
      auto ID = getConstInt(Args[1]);
      auto Col = getConstInt(Args[3]);
      if (!ID || !Col)
        continue;
      ir::StorePrimitiveOutputOp::create(B, Loc, static_cast<int32_t>(*ID),
                                         Args[2], static_cast<int8_t>(*Col),
                                         Args[4], Args[5]);
      Call.erase();
      continue;
    }
    if (Op == DXILOp::DispatchMesh) {
      ir::DispatchMeshOp::create(B, Loc, Args[1], Args[2], Args[3], Args[4]);
      Call.erase();
      continue;
    }

    //===------------------------------------------------------------------===//
    // Raytracing & RayQuery ops -> crucible.*
    //===------------------------------------------------------------------===//
    if (Op == DXILOp::TraceRay) {
      mlir::Value Accel = lookupHandle(Args[1]);
      if (!Accel)
        continue;
      ir::TraceRayOp::create(B, Loc, Accel, Args[2], Args[3], Args[4], Args[5],
                             Args[6], Args[7], Args[8], Args[9], Args[10],
                             Args[11], Args[12], Args[13], Args[14], Args[15]);
      Call.erase();
      continue;
    }
    if (Op == DXILOp::RayQuery_TraceRayInline) {
      mlir::Value RQ = lookupRQ(Args[1]);
      mlir::Value Accel = lookupHandle(Args[2]);
      if (!RQ || !Accel)
        continue;
      ir::RayQueryTraceInlineOp::create(B, Loc, RQ, Accel, Args[3], Args[4],
                                        Args[5], Args[6], Args[7], Args[8],
                                        Args[9], Args[10], Args[11], Args[12]);
      Call.erase();
      continue;
    }
    if (Op == DXILOp::RayQuery_Proceed) {
      mlir::Value RQ = lookupRQ(Args[1]);
      if (!RQ)
        continue;
      auto P = ir::RayQueryProceedOp::create(B, Loc, B.getI1Type(), RQ);
      Call.getResult().replaceAllUsesWith(P.getResult());
      Call.erase();
      continue;
    }
    if (Op == DXILOp::RayQuery_Abort) {
      mlir::Value RQ = lookupRQ(Args[1]);
      if (!RQ)
        continue;
      ir::RayQueryAbortOp::create(B, Loc, RQ);
      Call.erase();
      continue;
    }
    if (Op == DXILOp::RayQuery_CommitNonOpaqueTriangleHit ||
        Op == DXILOp::RayQuery_CommitProceduralPrimitiveHit) {
      mlir::Value RQ = lookupRQ(Args[1]);
      if (!RQ)
        continue;
      bool IsTri = Op == DXILOp::RayQuery_CommitNonOpaqueTriangleHit;
      ir::RayQueryCommitOp::create(
          B, Loc, RQ,
          ir::RayQueryCommitKindAttr::get(
              B.getContext(), IsTri ? ir::RayQueryCommitKind::Triangle
                                    : ir::RayQueryCommitKind::Procedural),
          IsTri ? mlir::Value() : Args[2]);
      Call.erase();
      continue;
    }
    if (Op == DXILOp::RayQuery_CommittedStatus ||
        Op == DXILOp::RayQuery_CandidateType ||
        Op == DXILOp::RayQuery_CommittedTriangleFrontFace ||
        Op == DXILOp::RayQuery_CommittedTriangleBarycentrics ||
        Op == DXILOp::RayQuery_RayFlags ||
        Op == DXILOp::RayQuery_WorldRayOrigin ||
        Op == DXILOp::RayQuery_WorldRayDirection ||
        Op == DXILOp::RayQuery_RayTMin ||
        Op == DXILOp::RayQuery_CommittedRayT ||
        Op == DXILOp::RayQuery_CommittedInstanceID ||
        Op == DXILOp::RayQuery_CommittedInstanceIndex ||
        Op == DXILOp::RayQuery_CommittedPrimitiveIndex ||
        Op == DXILOp::RayQuery_CommittedGeometryIndex ||
        Op == DXILOp::RayQuery_CommittedInstanceContributionToHitGroupIndex) {
      mlir::Value RQ = lookupRQ(Args[1]);
      if (!RQ)
        continue;
      ir::RayQueryField Field = ir::RayQueryField::CommittedStatus;
      mlir::Value Comp;
      switch (Op) {
      case DXILOp::RayQuery_CommittedStatus:
        Field = ir::RayQueryField::CommittedStatus;
        break;
      case DXILOp::RayQuery_CandidateType:
        Field = ir::RayQueryField::CandidateType;
        break;
      case DXILOp::RayQuery_CommittedTriangleFrontFace:
        Field = ir::RayQueryField::CommittedTriangleFrontFace;
        break;
      case DXILOp::RayQuery_CommittedTriangleBarycentrics:
        Field = ir::RayQueryField::CommittedTriangleBarycentrics;
        Comp = Args[2];
        break;
      case DXILOp::RayQuery_RayFlags:
        Field = ir::RayQueryField::RayFlags;
        break;
      case DXILOp::RayQuery_WorldRayOrigin:
        Field = ir::RayQueryField::WorldRayOrigin;
        Comp = Args[2];
        break;
      case DXILOp::RayQuery_WorldRayDirection:
        Field = ir::RayQueryField::WorldRayDirection;
        Comp = Args[2];
        break;
      case DXILOp::RayQuery_RayTMin:
        Field = ir::RayQueryField::RayTMin;
        break;
      case DXILOp::RayQuery_CommittedRayT:
        Field = ir::RayQueryField::CommittedRayT;
        break;
      case DXILOp::RayQuery_CommittedInstanceID:
        Field = ir::RayQueryField::CommittedInstanceId;
        break;
      case DXILOp::RayQuery_CommittedInstanceIndex:
        Field = ir::RayQueryField::CommittedInstanceIndex;
        break;
      case DXILOp::RayQuery_CommittedPrimitiveIndex:
        Field = ir::RayQueryField::CommittedPrimitiveIndex;
        break;
      case DXILOp::RayQuery_CommittedGeometryIndex:
        Field = ir::RayQueryField::CommittedGeometryIndex;
        break;
      case DXILOp::RayQuery_CommittedInstanceContributionToHitGroupIndex:
        Field = ir::RayQueryField::CommittedInstanceContrib;
        break;
      default:
        break;
      }
      if (Comp && !Comp.getType().isInteger(32)) {
        if (auto C = getConstInt(Comp))
          Comp = mlir::LLVM::ConstantOp::create(
              B, Loc, I32Ty, B.getI32IntegerAttr(static_cast<int32_t>(*C)));
        else
          Comp = mlir::LLVM::ZExtOp::create(B, Loc, I32Ty, Comp);
      }
      auto Q = ir::RayQueryGetOp::create(
          B, Loc, Call.getResult().getType(), RQ,
          ir::RayQueryFieldAttr::get(B.getContext(), Field), Comp);
      Call.getResult().replaceAllUsesWith(Q.getResult());
      Call.erase();
      continue;
    }

    //===------------------------------------------------------------------===//
    // Wave / subgroup & quad ops -> gpu.* / crucible.*
    //===------------------------------------------------------------------===//
    if (Op == DXILOp::WaveIsFirstLane) {
      auto W = ir::WaveIsFirstLaneOp::create(B, Loc, B.getI1Type());
      Call.getResult().replaceAllUsesWith(W.getResult());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::WaveAnyTrue || Op == DXILOp::WaveAllTrue ||
        Op == DXILOp::WaveActiveAllEqual) {
      ir::WaveVoteKind Kind = Op == DXILOp::WaveAnyTrue
                                  ? ir::WaveVoteKind::Any
                              : Op == DXILOp::WaveAllTrue
                                  ? ir::WaveVoteKind::All
                                  : ir::WaveVoteKind::AllEqual;
      auto W = ir::WaveVoteOp::create(
          B, Loc, B.getI1Type(), Args[1],
          ir::WaveVoteKindAttr::get(B.getContext(), Kind));
      Call.getResult().replaceAllUsesWith(W.getResult());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::WaveActiveBallot) {
      if (!allSingleIndexExtractValues(Call.getResult()))
        continue;
      auto VecTy = mlir::VectorType::get({4}, I32Ty);
      auto W = ir::WaveBallotOp::create(B, Loc, VecTy, Args[1]);
      replaceStructUsersWithVectorExtracts(Call.getResult(), W.getResult());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::WaveAllBitCount || Op == DXILOp::WavePrefixBitCount ||
        Op == DXILOp::WaveMultiPrefixBitCount) {
      ir::WaveBitCountKind Kind = Op == DXILOp::WaveAllBitCount
                                      ? ir::WaveBitCountKind::All
                                  : Op == DXILOp::WavePrefixBitCount
                                      ? ir::WaveBitCountKind::Prefix
                                      : ir::WaveBitCountKind::MultiPrefix;
      SmallVector<mlir::Value, 4> Mask;
      if (Op == DXILOp::WaveMultiPrefixBitCount) {
        Mask.push_back(Args[2]);
        Mask.push_back(Args[3]);
        if (Args.size() > 5) {
          Mask.push_back(Args[4]);
          Mask.push_back(Args[5]);
        }
      }
      auto W = ir::WaveBitCountOp::create(
          B, Loc, I32Ty, Args[1],
          ir::WaveBitCountKindAttr::get(B.getContext(), Kind), Mask);
      Call.getResult().replaceAllUsesWith(W.getResult());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::WaveMultiPrefixOp) {
      auto Kind = getConstInt(Args[6]);
      auto SignedOp = getConstInt(Args[7]);
      if (!Kind || *Kind < 0 || *Kind > 4 || !SignedOp)
        continue;
      ir::WaveMultiPrefixOpKind OpKind =
          static_cast<ir::WaveMultiPrefixOpKind>(*Kind);
      SmallVector<mlir::Value, 4> Mask = {Args[2], Args[3], Args[4], Args[5]};
      auto W = ir::WaveMultiPrefixOp::create(
          B, Loc, Call.getResult().getType(), Args[1], Mask, OpKind,
          /*is_unsigned=*/(*SignedOp != 0));
      Call.getResult().replaceAllUsesWith(W.getResult());
      Call.erase();
      continue;
    }

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

    if (Op == DXILOp::QuadReadLaneAt) {
      auto Q = ir::QuadReadLaneAtOp::create(B, Loc, Call.getResult().getType(),
                                            Args[1], Args[2]);
      Call.getResult().replaceAllUsesWith(Q.getResult());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::QuadOp) {
      auto Kind = getConstInt(Args[2]);
      if (!Kind || *Kind < 0 || *Kind > 2)
        continue;
      auto Q = ir::QuadOp::create(
          B, Loc, Call.getResult().getType(), Args[1],
          ir::QuadOpKindAttr::get(B.getContext(),
                                  static_cast<ir::QuadOpKind>(*Kind)));
      Call.getResult().replaceAllUsesWith(Q.getResult());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::WaveActiveOp || Op == DXILOp::WavePrefixOp) {
      mlir::Type ValTy = Args[1].getType();
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

    //===------------------------------------------------------------------===//
    // Aggregate-returning scalar/bit ops (SplitDouble, UAddc, Unpack4x8)
    //===------------------------------------------------------------------===//
    if (Op == DXILOp::SplitDouble) {
      if (!allSingleIndexExtractValues(Call.getResult()))
        continue;
      auto SD = ir::SplitDoubleOp::create(B, Loc, I32Ty, I32Ty, Args[1]);
      replacePairStructUsers(Call.getResult(), SD.getLo(), SD.getHi());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::UAddc) {
      if (!allSingleIndexExtractValues(Call.getResult()))
        continue;
      auto UA = ir::UAddcOp::create(B, Loc, Args[1].getType(), B.getI1Type(),
                                    Args[1], Args[2]);
      replacePairStructUsers(Call.getResult(), UA.getSum(), UA.getCarry());
      Call.erase();
      continue;
    }

    if (Op == DXILOp::Unpack4x8) {
      auto Mode = getConstInt(Args[1]);
      auto RetStruct =
          mlir::dyn_cast<mlir::LLVM::LLVMStructType>(Call.getResult().getType());
      if (!Mode || !RetStruct || RetStruct.getBody().empty() ||
          !allSingleIndexExtractValues(Call.getResult()))
        continue;
      auto ElemTy = RetStruct.getBody()[0];
      auto VecTy = mlir::VectorType::get({4}, ElemTy);
      auto UP = ir::Unpack4x8Op::create(B, Loc, VecTy,
                                        static_cast<int8_t>(*Mode), Args[2]);
      replaceStructUsersWithVectorExtracts(Call.getResult(), UP.getResult());
      Call.erase();
      continue;
    }

    //===------------------------------------------------------------------===//
    // Scalar math, bit, dot, pack, and conversion ops -> math.* / llvm.* / crucible.*
    //===------------------------------------------------------------------===//
    if (Call.getNumResults() != 1)
      continue;
    mlir::Type ResTy = Call.getResult().getType();

    mlir::Value Rep;
    switch (Op) {
    case DXILOp::FAbs:
      Rep = mlir::math::AbsFOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::IsNaN:
      Rep = mlir::math::IsNaNOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::IsInf:
      Rep = mlir::math::IsInfOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::IsFinite:
      Rep = mlir::math::IsFiniteOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::Saturate:
      Rep = ir::SaturateOp::create(B, Loc, ResTy, Args[1]);
      break;
    case DXILOp::Frc:
      Rep = ir::FrcOp::create(B, Loc, ResTy, Args[1]);
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
    case DXILOp::Asin:
      Rep = mlir::math::AsinOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::Acos:
      Rep = mlir::math::AcosOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::Atan:
      Rep = mlir::math::AtanOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::Hsin:
      Rep = mlir::math::SinhOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::Hcos:
      Rep = mlir::math::CoshOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::Htan:
      Rep = mlir::math::TanhOp::create(B, Loc, Args[1]);
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
    case DXILOp::FMad:
      Rep = mlir::math::FmaOp::create(B, Loc, Args[1], Args[2], Args[3]);
      break;
    case DXILOp::IMad:
    case DXILOp::UMad: {
      auto Mul = mlir::LLVM::MulOp::create(B, Loc, Args[1], Args[2]);
      Rep = mlir::LLVM::AddOp::create(B, Loc, Mul, Args[3]);
      break;
    }
    case DXILOp::Countbits: {
      mlir::Type ArgTy = Args[1].getType();
      if (ArgTy == I32Ty) {
        Rep = mlir::math::CtPopOp::create(B, Loc, Args[1]);
      } else {
        mlir::Value Pop = mlir::LLVM::CtPopOp::create(B, Loc, Args[1]);
        unsigned Bits = ArgTy.getIntOrFloatBitWidth();
        if (Bits < 32)
          Rep = mlir::LLVM::ZExtOp::create(B, Loc, I32Ty, Pop);
        else if (Bits > 32)
          Rep = mlir::LLVM::TruncOp::create(B, Loc, I32Ty, Pop);
        else
          Rep = Pop;
      }
      break;
    }
    case DXILOp::Bfrev:
      Rep = mlir::LLVM::BitReverseOp::create(B, Loc, Args[1]);
      break;
    case DXILOp::FirstbitLo:
    case DXILOp::FirstbitHi:
    case DXILOp::FirstbitSHi: {
      ir::FirstBitKind Kind = Op == DXILOp::FirstbitLo   ? ir::FirstBitKind::Lo
                              : Op == DXILOp::FirstbitHi ? ir::FirstBitKind::Hi
                                                         : ir::FirstBitKind::SHi;
      Rep = ir::FirstBitOp::create(
          B, Loc, I32Ty, Args[1],
          ir::FirstBitKindAttr::get(B.getContext(), Kind));
      break;
    }
    case DXILOp::Ibfe:
    case DXILOp::Ubfe:
      Rep = ir::BfeOp::create(B, Loc, ResTy, Args[1], Args[2], Args[3],
                              /*is_signed=*/(Op == DXILOp::Ibfe));
      break;
    case DXILOp::Bfi:
      Rep = ir::BfiOp::create(B, Loc, ResTy, Args[1], Args[2], Args[3], Args[4]);
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
    case DXILOp::Dot2:
    case DXILOp::Dot3:
    case DXILOp::Dot4: {
      unsigned N = Op == DXILOp::Dot2 ? 2 : Op == DXILOp::Dot3 ? 3 : 4;
      SmallVector<mlir::Value, 4> LHS(Args.begin() + 1, Args.begin() + 1 + N);
      SmallVector<mlir::Value, 4> RHS(Args.begin() + 1 + N,
                                      Args.begin() + 1 + 2 * N);
      Rep = ir::DotOp::create(B, Loc, ResTy, LHS, RHS);
      break;
    }
    case DXILOp::Dot2AddHalf: {
      SmallVector<mlir::Value, 4> Inputs = {Args[2], Args[3], Args[4], Args[5]};
      Rep = ir::DotPackedOp::create(
          B, Loc, ResTy, Args[1], Inputs,
          ir::DotPackedKindAttr::get(B.getContext(), ir::DotPackedKind::F16x2));
      break;
    }
    case DXILOp::Dot4AddI8Packed:
    case DXILOp::Dot4AddU8Packed: {
      SmallVector<mlir::Value, 2> Inputs = {Args[2], Args[3]};
      auto Kind = Op == DXILOp::Dot4AddI8Packed ? ir::DotPackedKind::I8x4
                                                : ir::DotPackedKind::U8x4;
      Rep = ir::DotPackedOp::create(
          B, Loc, ResTy, Args[1], Inputs,
          ir::DotPackedKindAttr::get(B.getContext(), Kind));
      break;
    }
    case DXILOp::Pack4x8: {
      auto Mode = getConstInt(Args[1]);
      if (!Mode)
        break;
      Rep = ir::Pack4x8Op::create(B, Loc, I32Ty, static_cast<int8_t>(*Mode),
                                  Args[2], Args[3], Args[4], Args[5]);
      break;
    }
    case DXILOp::LegacyF32ToF16:
      Rep = ir::LegacyF32ToF16Op::create(B, Loc, I32Ty, Args[1]);
      break;
    case DXILOp::LegacyF16ToF32:
      Rep = ir::LegacyF16ToF32Op::create(B, Loc, B.getF32Type(), Args[1]);
      break;
    case DXILOp::MakeDouble:
      Rep = ir::MakeDoubleOp::create(B, Loc, B.getF64Type(), Args[1], Args[2]);
      break;
    default:
      break;
    }

    if (Rep) {
      Call.getResult().replaceAllUsesWith(Rep);
      Call.erase();
    }
  }

  // Pass 3: Erase now-unused handle / RayQuery producer calls in reverse order
  // (AnnotateHandle before CreateHandle*).
  for (auto It = DeferredErase.rbegin(); It != DeferredErase.rend(); ++It) {
    if ((*It)->use_empty())
      (*It).erase();
  }
}

} // namespace crucible

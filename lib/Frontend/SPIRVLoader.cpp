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
#include "../Backend/IntrinsicMap.h"
#include "crucible/Diagnostic.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Endian.h"

#include "mlir/Conversion/SPIRVToLLVM/SPIRVToLLVMPass.h"
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

struct BoundBufferInfo {
  size_t ReflectIndex = 0;
  mlir::Type ElemType;
  uint32_t ElemStride = 4;
};

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
  return T;
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

Expected<LoadedMLIRShader> loadSPIRV(MemoryBufferRef Buf,
                                     mlir::MLIRContext &Ctx,
                                     StringRef EntryOverride) {
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
                  mlir::math::MathDialect, mlir::ub::UBDialect>();
  Ctx.appendDialectRegistry(Registry);
  Ctx.loadAllAvailableDialects();

  size_t NumWords = Buf.getBufferSize() / 4;
  SmallVector<uint32_t> Words(NumWords);
  const char *Raw = Buf.getBufferStart();
  for (size_t I = 0; I < NumWords; ++I)
    Words[I] = support::endian::read32le(Raw + I * 4);

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

  // 1. Extract entry point & execution mode into Reflection.
  Reflection Reflect;
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

  // 2. Inspect spirv.GlobalVariables for built-ins and descriptor bindings.
  StringMap<mlir::spirv::BuiltIn> BuiltInGlobals;
  StringMap<BoundBufferInfo> BufferGlobals;
  SmallVector<mlir::spirv::GlobalVariableOp, 8> GlobalsToErase;
  uint32_t NextSRVId = 0, NextUAVId = 0;

  for (auto GV : SpvModule.getOps<mlir::spirv::GlobalVariableOp>()) {
    if (auto BIName = GV.getBuiltIn()) {
      auto BI = mlir::spirv::symbolizeBuiltIn(*BIName);
      if (!BI)
        return makeError("unknown SPIR-V built_in '{0}'", *BIName);
      BuiltInGlobals[GV.getSymName()] = *BI;
      GlobalsToErase.push_back(GV);
      continue;
    }

    auto DS = GV.getDescriptorSet();
    auto Binding = GV.getBinding();
    if (!DS || !Binding)
      continue;

    auto PtrTy = mlir::dyn_cast<mlir::spirv::PointerType>(GV.getType());
    if (!PtrTy)
      continue;
    auto SC = PtrTy.getStorageClass();
    if (SC != mlir::spirv::StorageClass::Uniform &&
        SC != mlir::spirv::StorageClass::StorageBuffer)
      continue;

    auto StructTy =
        mlir::dyn_cast<mlir::spirv::StructType>(PtrTy.getPointeeType());
    if (!StructTy || StructTy.getNumElements() == 0)
      continue;

    auto RTArr =
        mlir::dyn_cast<mlir::spirv::RuntimeArrayType>(StructTy.getElementType(0));
    if (!RTArr)
      continue;

    mlir::Type ElemTy = RTArr.getElementType();
    uint32_t Stride = RTArr.getArrayStride();
    if (Stride == 0)
      Stride = 4;

    bool IsReadOnly = false;
    SmallVector<mlir::spirv::StructType::MemberDecorationInfo> MemDecs;
    StructTy.getMemberDecorations(0, MemDecs);
    for (const auto &D : MemDecs) {
      if (D.decoration == mlir::spirv::Decoration::NonWritable)
        IsReadOnly = true;
    }

    ResourceBinding RB;
    RB.Class = IsReadOnly ? ResourceClass::SRV : ResourceClass::UAV;
    RB.Kind = ResourceKind::StructuredBuffer;
    RB.ElementType = toComponentType(ElemTy);
    RB.RecordID = IsReadOnly ? NextSRVId++ : NextUAVId++;
    RB.Name = GV.getSymName().str();
    RB.Space = *DS;
    RB.LowerBound = *Binding;
    RB.RangeSize = 1;
    RB.ElementStride = Stride;

    BoundBufferInfo Info;
    Info.ReflectIndex = Reflect.Resources.size();
    Info.ElemType = ElemTy;
    Info.ElemStride = Stride;
    BufferGlobals[GV.getSymName()] = Info;
    Reflect.Resources.push_back(std::move(RB));
    GlobalsToErase.push_back(GV);
  }

  auto EntryFn = SpvModule.lookupSymbol<mlir::spirv::FuncOp>(Reflect.EntryName);
  if (!EntryFn)
    return makeError("SPIR-V entry function '{0}' not found", Reflect.EntryName);

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
  auto getHandleFor = [&](const BoundBufferInfo &Info) -> mlir::Value {
    auto It = HandleCache.find(Info.ReflectIndex);
    if (It != HandleCache.end())
      return It->second;

    EntryBuilder.setInsertionPointToStart(&EntryFn.getBody().front());
    const ResourceBinding &RB = Reflect.Resources[Info.ReflectIndex];
    auto FnTy = mlir::LLVM::LLVMFunctionType::get(
        HandleTy, {I32Ty, I8Ty, I32Ty, I32Ty, I1Ty});
    getOrCreateDecl(SpvModule, "dx.op.createHandle", FnTy);

    auto OpC = mlir::LLVM::ConstantOp::create(
        EntryBuilder, FnLoc, I32Ty, static_cast<int32_t>(DXILOp::CreateHandle));
    auto ClassC = mlir::LLVM::ConstantOp::create(
        EntryBuilder, FnLoc, I8Ty, static_cast<int8_t>(RB.Class));
    auto IdC = mlir::LLVM::ConstantOp::create(
        EntryBuilder, FnLoc, I32Ty, static_cast<int32_t>(RB.RecordID));
    auto IdxC = mlir::LLVM::ConstantOp::create(
        EntryBuilder, FnLoc, I32Ty, static_cast<int32_t>(RB.LowerBound));
    auto NonUnifC =
        mlir::LLVM::ConstantOp::create(EntryBuilder, FnLoc, I1Ty, false);

    auto Call = mlir::LLVM::CallOp::create(
        EntryBuilder, FnLoc, mlir::TypeRange{HandleTy}, "dx.op.createHandle",
        mlir::ValueRange{OpC, ClassC, IdC, IdxC, NonUnifC});
    HandleCache[Info.ReflectIndex] = Call.getResult();
    return Call.getResult();
  };

  SmallVector<mlir::spirv::AddressOfOp, 8> AddrOps;
  EntryFn.walk([&](mlir::spirv::AddressOfOp AO) { AddrOps.push_back(AO); });

  for (auto AO : AddrOps) {
    StringRef Sym = AO.getVariable();

    // 3a. Built-in globals (e.g. GlobalInvocationId -> gpu.global_id).
    auto BIIt = BuiltInGlobals.find(Sym);
    if (BIIt != BuiltInGlobals.end()) {
      mlir::spirv::BuiltIn BI = BIIt->second;
      SmallVector<mlir::Operation *, 4> Loads(AO->getUsers().begin(),
                                              AO->getUsers().end());
      for (mlir::Operation *U : Loads) {
        auto Load = mlir::dyn_cast<mlir::spirv::LoadOp>(U);
        if (!Load)
          return makeError("unsupported user of built-in global '{0}'", Sym);
        SmallVector<mlir::Operation *, 4> Extracts(Load->getUsers().begin(),
                                                   Load->getUsers().end());
        for (mlir::Operation *EU : Extracts) {
          auto Extract = mlir::dyn_cast<mlir::spirv::CompositeExtractOp>(EU);
          if (!Extract || Extract.getIndices().size() != 1)
            return makeError("unsupported extract on built-in '{0}'", Sym);
          uint32_t Comp = static_cast<uint32_t>(
              mlir::cast<mlir::IntegerAttr>(Extract.getIndices()[0]).getInt());
          mlir::OpBuilder B(Extract);
          mlir::Value Rep =
              emitGPUIdComponent(B, Extract.getLoc(), BI, Comp, Extract.getType());
          if (!Rep)
            return makeError("unsupported SPIR-V built-in '{0}'", Sym);
          Extract.replaceAllUsesWith(Rep);
          Extract.erase();
        }
        if (!Load->use_empty())
          return makeError("unhandled vector use of built-in '{0}'", Sym);
        Load.erase();
      }
      AO.erase();
      continue;
    }

    // 3b. Descriptor-bound structured buffers -> dx.op.rawBufferLoad / Store.
    auto BufIt = BufferGlobals.find(Sym);
    if (BufIt != BufferGlobals.end()) {
      const BoundBufferInfo &Info = BufIt->second;
      mlir::Value Handle = getHandleFor(Info);

      SmallVector<mlir::Operation *, 8> ChainUsers(AO->getUsers().begin(),
                                                   AO->getUsers().end());
      for (mlir::Operation *CU : ChainUsers) {
        auto Chain = mlir::dyn_cast<mlir::spirv::AccessChainOp>(CU);
        if (!Chain || Chain.getIndices().size() < 2)
          return makeError("unsupported access pattern on buffer '{0}'", Sym);

        mlir::Value ElemIdx = Chain.getIndices()[1];
        SmallVector<mlir::Operation *, 8> MemUsers(Chain->getUsers().begin(),
                                                   Chain->getUsers().end());
        for (mlir::Operation *MU : MemUsers) {
          if (auto Load = mlir::dyn_cast<mlir::spirv::LoadOp>(MU)) {
            mlir::OpBuilder B(Load);
            mlir::Location Loc = Load.getLoc();
            mlir::Type ValTy = makeSignless(Load.getType(), B);
            auto ResRetTy = getOrInitResRetType(&Ctx, ValTy);
            std::string Callee = "dx.op.rawBufferLoad." + typeSuffix(ValTy);
            auto FnTy = mlir::LLVM::LLVMFunctionType::get(
                ResRetTy, {I32Ty, HandleTy, I32Ty, I32Ty, I8Ty, I32Ty});
            getOrCreateDecl(SpvModule, Callee, FnTy);

            auto OpC = mlir::LLVM::ConstantOp::create(
                B, Loc, I32Ty, static_cast<int32_t>(DXILOp::RawBufferLoad));
            auto OffC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 0);
            auto MaskC = mlir::LLVM::ConstantOp::create(
                B, Loc, I8Ty, static_cast<int8_t>(1));
            auto AlignC = mlir::LLVM::ConstantOp::create(
                B, Loc, I32Ty, static_cast<int32_t>(Info.ElemStride));

            auto Call = mlir::LLVM::CallOp::create(
                B, Loc, mlir::TypeRange{ResRetTy}, Callee,
                mlir::ValueRange{OpC, Handle, ElemIdx, OffC, MaskC, AlignC});
            auto Val = mlir::LLVM::ExtractValueOp::create(
                B, Loc, Call.getResult(), ArrayRef<int64_t>{0});
            Load.replaceAllUsesWith(Val.getResult());
            Load.erase();
          } else if (auto Store = mlir::dyn_cast<mlir::spirv::StoreOp>(MU)) {
            mlir::OpBuilder B(Store);
            mlir::Location Loc = Store.getLoc();
            mlir::Value StoreVal = Store.getValue();
            mlir::Type ValTy = makeSignless(StoreVal.getType(), B);
            std::string Callee = "dx.op.rawBufferStore." + typeSuffix(ValTy);
            auto FnTy = mlir::LLVM::LLVMFunctionType::get(
                VoidTy, {I32Ty, HandleTy, I32Ty, I32Ty, ValTy, ValTy, ValTy,
                         ValTy, I8Ty, I32Ty});
            getOrCreateDecl(SpvModule, Callee, FnTy);

            auto OpC = mlir::LLVM::ConstantOp::create(
                B, Loc, I32Ty, static_cast<int32_t>(DXILOp::RawBufferStore));
            auto OffC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 0);
            auto Undef = mlir::LLVM::UndefOp::create(B, Loc, ValTy);
            auto MaskC = mlir::LLVM::ConstantOp::create(
                B, Loc, I8Ty, static_cast<int8_t>(1));
            auto AlignC = mlir::LLVM::ConstantOp::create(
                B, Loc, I32Ty, static_cast<int32_t>(Info.ElemStride));

            mlir::LLVM::CallOp::create(
                B, Loc, mlir::TypeRange{}, Callee,
                mlir::ValueRange{OpC, Handle, ElemIdx, OffC, StoreVal, Undef,
                                 Undef, Undef, MaskC, AlignC});
            Store.erase();
          } else {
            return makeError("unsupported memory op on buffer '{0}'", Sym);
          }
        }
        Chain.erase();
      }
      AO.erase();
      continue;
    }
  }

  // Erase rewritten shader-ABI globals and entry/execution-mode ops before
  // running ConvertSPIRVToLLVM.
  for (auto GV : GlobalsToErase)
    GV.erase();
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

  if (mlir::failed(mlir::verify(*TopMod)))
    return makeError("MLIR verification failed after SPIR-V lowering");

  LoadedMLIRShader Out;
  Out.Module = std::move(TopMod);
  Out.Reflect = std::move(Reflect);
  return std::move(Out);
}

} // namespace crucible

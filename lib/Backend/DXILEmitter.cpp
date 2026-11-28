//===- DXILEmitter.cpp - Crucible MLIR -> LLVM DirectX backend --*- C++ -*-===//
//
// Part of Crucible, an open-source multi-frontend, multi-backend shader compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Lowers unified Crucible MLIR (gpu.* + math.* + llvm.* + crucible.*, with zero
// DXILOp dependencies) into LLVM 24's DirectX target IR (target("dx.*") types
// and llvm.dx.* / standard LLVM intrinsics), then invokes LLVM's DirectX
// TargetMachine to emit a DXContainer binary or DXIL IR assembly.
//
//===----------------------------------------------------------------------===//

#include "DXILEmitter.h"
#include "crucible/Diagnostic.h"
#include "crucible/Dialect/CrucibleDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#if CRUCIBLE_HAS_DIRECTX
#include "llvm/IR/IntrinsicsDirectX.h"
#endif
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <optional>
#include <string>

using namespace llvm;

namespace crucible {

#if !CRUCIBLE_HAS_DIRECTX

Expected<std::string> emitDXIL(mlir::ModuleOp, Reflection &, const Options &) {
  return makeError(
      "DirectX backend is not enabled in this build (rebuild Crucible against "
      "LLVM 24 with the DirectX target enabled)");
}

#else

namespace {

/// Internal bridge opcodes between Crucible MLIR and LLVM DirectX IR lowering.
enum class DxOp : uint32_t {
  // Resources
  BindResource = 1,
  HeapResource,
  // Buffers & CBuffers
  BufferLoad,
  BufferStatus,
  BufferStore,
  CBufferLoad,
  BufferUpdateCounter,
  BufferAtomic,
  BufferAtomicCmpXchg,
  // Textures & Sampling
  TexLoad,
  TexStore,
  TexSample,
  TexGather,
  TexCalcLod,
  TexGetDimensions,
  // Barriers
  Barrier,
  // Compute / SysVal
  ThreadId,
  GroupId,
  ThreadIdInGroup,
  FlattenedThreadIdInGroup,
  LaneId,
  SubgroupSize,
  SysVal,
  // Stage I/O & Pixel
  LoadInput,
  StoreOutput,
  Discard,
  Deriv,
  // Wave & Quad
  WaveIsFirstLane,
  WaveVote,
  WaveBallot,
  WaveBitCount,
  WaveMultiPrefix,
  WaveReadLaneFirst,
  WaveReadLaneAt,
  WaveReduce,
  QuadReadLaneAt,
  QuadOp,
  // Math & Bit gaps
  Saturate,
  Frc,
  IsNaN,
  IsInf,
  IsFinite,
  Rsqrt,
  FirstBit,
  Bfe,
  Bfi,
  Dot,
  DotPacked,
  LegacyF32ToF16,
  LegacyF16ToF32,
  MakeDouble,
  SplitDouble,
  UAddc,
};

void initializeDirectXTargetOnce() {
  static std::once_flag Once;
  std::call_once(Once, []() {
    LLVMInitializeDirectXTargetInfo();
    LLVMInitializeDirectXTarget();
    LLVMInitializeDirectXTargetMC();
    LLVMInitializeDirectXAsmPrinter();
  });
}

Expected<std::string> stageTripleSuffix(ShaderStage Stage) {
  switch (Stage) {
  case ShaderStage::Compute:
    return "compute";
  case ShaderStage::Vertex:
    return "vertex";
  case ShaderStage::Pixel:
    return "pixel";
  case ShaderStage::Geometry:
    return "geometry";
  case ShaderStage::Hull:
    return "hull";
  case ShaderStage::Domain:
    return "domain";
  case ShaderStage::Mesh:
    return "mesh";
  case ShaderStage::Amplification:
    return "amplification";
  case ShaderStage::Library:
    return "library";
  default:
    return makeError("DirectX backend: unsupported shader stage {0}",
                     static_cast<unsigned>(Stage));
  }
}

struct ResolvedHandle {
  const ResourceBinding *RB = nullptr;
  Value *DxHandle = nullptr;
  Type *DxHandleTy = nullptr;
};

class DXILLowerer {
public:
  DXILLowerer(Module &M, Function *Entry, Reflection &Reflect,
              const Options &Opts)
      : M(M), Ctx(M.getContext()), Entry(Entry), Reflect(Reflect), Opts(Opts) {}

  Error run(StringRef StageName);

private:
  Module &M;
  LLVMContext &Ctx;
  Function *Entry;
  Reflection &Reflect;
  const Options &Opts;

  DenseMap<const Value *, ResolvedHandle> HandleMap;
  DenseMap<const ResourceBinding *, GlobalVariable *> NameGlobals;

  const ResourceBinding *findResourceByID(ResourceClass Class,
                                          uint32_t RecordID) const;
  const ResourceBinding *findResourceByBinding(ResourceClass Class,
                                               uint32_t Space,
                                               uint32_t LowerBound) const;
  GlobalVariable *getOrCreateResourceName(const ResourceBinding &RB);
  Type *getComponentLLVMType(ComponentType CT);
  bool isSignedComponent(ComponentType CT) const;
  Type *getDXILResourceType(const ResourceBinding &RB);
  Expected<ResolvedHandle> resolveHandle(Value *V);

  Error lowerBridgeCall(CallInst *CI);
  Error lowerBindResource(CallInst *CI);
  Error lowerHeapResource(CallInst *CI);
  Error lowerBufferLoad(CallInst *CI);
  Error lowerBufferStore(CallInst *CI);
  Error lowerCBufferLoad(CallInst *CI);
  Error lowerBufferUpdateCounter(CallInst *CI);
  Error lowerBufferAtomic(CallInst *CI);
  Error lowerBufferAtomicCmpXchg(CallInst *CI);
  Error lowerTexLoad(CallInst *CI);
  Error lowerTexStore(CallInst *CI);
  Error lowerTexSample(CallInst *CI);
  Error lowerTexGather(CallInst *CI);
  Error lowerTexCalcLod(CallInst *CI);
  Error lowerTexGetDimensions(CallInst *CI);
  Error lowerBarrier(CallInst *CI);
  Error lowerWaveOp(CallInst *CI, DxOp Op);
  Error lowerMathOrBitOp(CallInst *CI, DxOp Op);
  Error lowerPixelOp(CallInst *CI, DxOp Op);

  void configureEntryAndLinkage(StringRef StageName);
  void emitSemanticSignatures();
  void normalizeGroupsharedAtomics();
  void cleanupBridgeArtifacts();
};

const ResourceBinding *DXILLowerer::findResourceByID(ResourceClass Class,
                                                     uint32_t RecordID) const {
  for (const ResourceBinding &RB : Reflect.Resources) {
    if (RB.Class == Class && RB.RecordID == RecordID)
      return &RB;
  }
  return nullptr;
}

const ResourceBinding *
DXILLowerer::findResourceByBinding(ResourceClass Class, uint32_t Space,
                                   uint32_t LowerBound) const {
  for (const ResourceBinding &RB : Reflect.Resources) {
    if (RB.Class == Class && RB.Space == Space && RB.LowerBound == LowerBound)
      return &RB;
  }
  return nullptr;
}

GlobalVariable *
DXILLowerer::getOrCreateResourceName(const ResourceBinding &RB) {
  auto It = NameGlobals.find(&RB);
  if (It != NameGlobals.end())
    return It->second;

  std::string Name = RB.Name;
  if (Name.empty())
    Name = "res_" + std::to_string(RB.Space) + "_" +
           std::to_string(RB.LowerBound);

  Constant *StrC = ConstantDataArray::getString(Ctx, Name, /*AddNull=*/true);
  auto *GV = new GlobalVariable(M, StrC->getType(), /*isConstant=*/true,
                                GlobalValue::PrivateLinkage, StrC,
                                ".dx.name." + Name, /*InsertBefore=*/nullptr,
                                GlobalValue::NotThreadLocal, /*AddressSpace=*/0);
  GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  NameGlobals[&RB] = GV;
  return GV;
}

Type *DXILLowerer::getComponentLLVMType(ComponentType CT) {
  switch (CT) {
  case ComponentType::I16:
  case ComponentType::U16:
    return Type::getInt16Ty(Ctx);
  case ComponentType::I32:
  case ComponentType::U32:
    return Type::getInt32Ty(Ctx);
  case ComponentType::I64:
  case ComponentType::U64:
    return Type::getInt64Ty(Ctx);
  case ComponentType::F16:
  case ComponentType::SNormF16:
  case ComponentType::UNormF16:
    return Type::getHalfTy(Ctx);
  case ComponentType::F64:
  case ComponentType::SNormF64:
  case ComponentType::UNormF64:
    return Type::getDoubleTy(Ctx);
  case ComponentType::F32:
  case ComponentType::SNormF32:
  case ComponentType::UNormF32:
  default:
    return Type::getFloatTy(Ctx);
  }
}

bool DXILLowerer::isSignedComponent(ComponentType CT) const {
  switch (CT) {
  case ComponentType::I16:
  case ComponentType::I32:
  case ComponentType::I64:
    return true;
  default:
    return false;
  }
}

Type *DXILLowerer::getDXILResourceType(const ResourceBinding &RB) {
  unsigned IsWriteable = (RB.Class == ResourceClass::UAV) ? 1u : 0u;
  unsigned IsROV = 0u;
  unsigned IsSigned = isSignedComponent(RB.ElementType) ? 1u : 0u;

  switch (RB.Kind) {
  case ResourceKind::RawBuffer:
    return TargetExtType::get(Ctx, "dx.RawBuffer", {Type::getInt8Ty(Ctx)},
                              {IsWriteable, IsROV});
  case ResourceKind::StructuredBuffer: {
    uint32_t Stride = RB.ElementStride ? RB.ElementStride : 4u;
    Type *ElemTy = nullptr;
    if (Stride == 4u)
      ElemTy = Type::getInt32Ty(Ctx);
    else if ((Stride % 4u) == 0)
      ElemTy = ArrayType::get(Type::getInt32Ty(Ctx), Stride / 4u);
    else if (Stride == 2u)
      ElemTy = Type::getInt16Ty(Ctx);
    else if ((Stride % 2u) == 0)
      ElemTy = ArrayType::get(Type::getInt16Ty(Ctx), Stride / 2u);
    else
      ElemTy = ArrayType::get(Type::getInt32Ty(Ctx), (Stride + 3u) / 4u);
    return TargetExtType::get(Ctx, "dx.RawBuffer", {ElemTy},
                              {IsWriteable, IsROV});
  }
  case ResourceKind::TypedBuffer: {
    Type *ScalarTy = getComponentLLVMType(RB.ElementType);
    uint32_t Comps = RB.TypedComponents ? RB.TypedComponents : 4u;
    Type *ElemTy =
        (Comps > 1) ? static_cast<Type *>(FixedVectorType::get(ScalarTy, Comps))
                    : ScalarTy;
    return TargetExtType::get(Ctx, "dx.TypedBuffer", {ElemTy},
                              {IsWriteable, IsROV, IsSigned});
  }
  case ResourceKind::CBuffer: {
    uint32_t NumVec4 = std::max(1u, (RB.ElementStride + 15u) / 16u);
    Type *Vec4I32 = FixedVectorType::get(Type::getInt32Ty(Ctx), 4);
    Type *ArrTy = ArrayType::get(Vec4I32, NumVec4);
    Type *LayoutTy = StructType::get(Ctx, {ArrTy}, /*isPacked=*/true);
    return TargetExtType::get(Ctx, "dx.CBuffer", {LayoutTy}, {});
  }
  case ResourceKind::Texture1D:
  case ResourceKind::Texture2D:
  case ResourceKind::Texture3D:
  case ResourceKind::TextureCube:
  case ResourceKind::Texture1DArray:
  case ResourceKind::Texture2DArray:
  case ResourceKind::TextureCubeArray: {
    Type *ScalarTy = getComponentLLVMType(RB.ElementType);
    uint32_t Comps = RB.TypedComponents ? RB.TypedComponents : 4u;
    Type *ElemTy =
        (Comps > 1) ? static_cast<Type *>(FixedVectorType::get(ScalarTy, Comps))
                    : ScalarTy;
    unsigned Dim = static_cast<unsigned>(RB.Kind);
    return TargetExtType::get(Ctx, "dx.Texture", {ElemTy},
                              {IsWriteable, IsROV, IsSigned, Dim});
  }
  case ResourceKind::Texture2DMS:
  case ResourceKind::Texture2DMSArray: {
    Type *ScalarTy = getComponentLLVMType(RB.ElementType);
    uint32_t Comps = RB.TypedComponents ? RB.TypedComponents : 4u;
    Type *ElemTy =
        (Comps > 1) ? static_cast<Type *>(FixedVectorType::get(ScalarTy, Comps))
                    : ScalarTy;
    unsigned Dim = static_cast<unsigned>(RB.Kind);
    return TargetExtType::get(Ctx, "dx.MSTexture", {ElemTy},
                              {IsWriteable, /*Samples=*/0u, IsSigned, Dim});
  }
  case ResourceKind::Sampler:
    return TargetExtType::get(Ctx, "dx.Sampler", {}, {0u /* Default */});
  default:
    return nullptr;
  }
}

Expected<ResolvedHandle> DXILLowerer::resolveHandle(Value *V) {
  auto It = HandleMap.find(V);
  if (It != HandleMap.end())
    return It->second;
  return makeError("DirectX backend: unresolved resource handle");
}

Error DXILLowerer::lowerBindResource(CallInst *CI) {
  auto Class = static_cast<ResourceClass>(
      cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue());
  int32_t RecordID =
      static_cast<int32_t>(cast<ConstantInt>(CI->getArgOperand(2))->getSExtValue());
  int32_t Space =
      static_cast<int32_t>(cast<ConstantInt>(CI->getArgOperand(3))->getSExtValue());
  int32_t Binding =
      static_cast<int32_t>(cast<ConstantInt>(CI->getArgOperand(4))->getSExtValue());
  bool NonUniform = cast<ConstantInt>(CI->getArgOperand(5))->getZExtValue() != 0;
  Value *ArrIdx = CI->getArgOperand(6);

  const ResourceBinding *RB = nullptr;
  if (RecordID >= 0)
    RB = findResourceByID(Class, static_cast<uint32_t>(RecordID));
  if (!RB && Space >= 0 && Binding >= 0)
    RB = findResourceByBinding(Class, static_cast<uint32_t>(Space),
                               static_cast<uint32_t>(Binding));
  if (!RB)
    return makeError("DirectX backend: cannot resolve resource binding "
                     "(class={0}, id={1}, space={2}, reg={3})",
                     static_cast<unsigned>(Class), RecordID, Space, Binding);

  Type *ResTy = getDXILResourceType(*RB);
  if (!ResTy)
    return makeError("DirectX backend: unsupported resource kind {0}",
                     static_cast<unsigned>(RB->Kind));

  IRBuilder<> B(CI);
  // crucible.bind_resource carries the absolute register index (LowerBound +
  // arrayOffset), whereas llvm.dx.resource.handlefrombinding expects the
  // 0-based offset within [LowerBound, LowerBound + RangeSize).
  // ArrIdx was biased by the original pre-remapping lower bound (RB->SPIRVBinding),
  // which assignDXILBindings snapshots before resolving register collisions.
  Value *Idx = nullptr;
  if (RB->RangeSize == 1) {
    Idx = B.getInt32(0);
  } else {
    uint32_t SrcLowerBound = RB->SPIRVBinding;
    Idx = (SrcLowerBound == 0) ? ArrIdx
                               : B.CreateSub(ArrIdx, B.getInt32(SrcLowerBound));
  }
  if (NonUniform) {
    Function *NuriFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_resource_nonuniformindex);
    Idx = B.CreateCall(NuriFn, {Idx});
  }
  uint32_t RangeSize = (RB->RangeSize == 0xFFFFFFFFu) ? 0u : RB->RangeSize;
  Function *HandleFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::dx_resource_handlefrombinding, {ResTy});
  Value *Handle = B.CreateCall(
      HandleFn, {B.getInt32(RB->Space), B.getInt32(RB->LowerBound),
                 B.getInt32(RangeSize), Idx, getOrCreateResourceName(*RB)});

  HandleMap[CI] = ResolvedHandle{RB, Handle, ResTy};
  return Error::success();
}

Error DXILLowerer::lowerHeapResource(CallInst *CI) {
  Value *HeapIdx = CI->getArgOperand(1);
  bool IsSampler = cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue() != 0;
  bool NonUniform = cast<ConstantInt>(CI->getArgOperand(3))->getZExtValue() != 0;
  uint32_t ClassRaw = cast<ConstantInt>(CI->getArgOperand(4))->getZExtValue();
  uint32_t KindRaw = cast<ConstantInt>(CI->getArgOperand(5))->getZExtValue();
  uint32_t Stride = cast<ConstantInt>(CI->getArgOperand(6))->getZExtValue();

  ResourceBinding TempRB;
  TempRB.Class = IsSampler ? ResourceClass::Sampler
                           : static_cast<ResourceClass>(ClassRaw);
  TempRB.Kind = IsSampler ? ResourceKind::Sampler
                          : static_cast<ResourceKind>(KindRaw);
  TempRB.ElementStride = Stride;
  Type *ResTy = getDXILResourceType(TempRB);
  if (!ResTy)
    return makeError("DirectX backend: unsupported heap resource kind {0}",
                     KindRaw);

  IRBuilder<> B(CI);
  Value *Idx = HeapIdx;
  if (NonUniform) {
    Function *NuriFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_resource_nonuniformindex);
    Idx = B.CreateCall(NuriFn, {Idx});
  }
  Function *HeapFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::dx_resource_handlefromheap, {ResTy});
  Value *Handle = B.CreateCall(HeapFn, {Idx});
  HandleMap[CI] = ResolvedHandle{nullptr, Handle, ResTy};
  return Error::success();
}

Error DXILLowerer::lowerBufferLoad(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  Value *Index = CI->getArgOperand(2);
  Value *ElemOffset = CI->getArgOperand(3);
  uint32_t Mask = cast<ConstantInt>(CI->getArgOperand(4))->getZExtValue();
  Type *RetTy = CI->getType();

  IRBuilder<> B(CI);
  bool IsTyped = RH.RB && RH.RB->Kind == ResourceKind::TypedBuffer;
  bool IsRaw = !RH.RB || RH.RB->Kind == ResourceKind::RawBuffer;

  Type *ScalarTy = RetTy->getScalarType();
  unsigned NumVecComps = 1;
  if (auto *VT = dyn_cast<FixedVectorType>(RetTy))
    NumVecComps = VT->getNumElements();

  // Compute how many contiguous components to load based on the mask.
  unsigned LoadComps = NumVecComps;
  if (Mask != 0 && NumVecComps > 1) {
    unsigned Highest = 0;
    for (unsigned I = 0; I < NumVecComps; ++I)
      if (Mask & (1u << I))
        Highest = I + 1;
    if (Highest > 0)
      LoadComps = Highest;
  }

  Type *LoadTy = (LoadComps == 1)
                     ? ScalarTy
                     : static_cast<Type *>(
                           FixedVectorType::get(ScalarTy, LoadComps));

  Value *LoadedVal = nullptr;
  if (IsTyped) {
    Type *TypedLoadTy = FixedVectorType::get(ScalarTy, 4);
    Function *LoadFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_resource_load_typedbuffer, {TypedLoadTy, RH.DxHandleTy});
    Value *Pair = B.CreateCall(LoadFn, {RH.DxHandle, Index});
    LoadedVal = B.CreateExtractValue(Pair, {0});
    if (NumVecComps == 1)
      LoadedVal = B.CreateExtractElement(LoadedVal, B.getInt32(0));
    else if (NumVecComps < 4) {
      Value *Vec = PoisonValue::get(RetTy);
      for (unsigned I = 0; I < NumVecComps; ++I) {
        Value *E = B.CreateExtractElement(LoadedVal, B.getInt32(I));
        Vec = B.CreateInsertElement(Vec, E, B.getInt32(I));
      }
      LoadedVal = Vec;
    }
  } else {
    Value *OffArg = IsRaw ? static_cast<Value *>(PoisonValue::get(B.getInt32Ty()))
                          : ElemOffset;
    if (!IsRaw && isa<UndefValue>(OffArg))
      OffArg = B.getInt32(0);

    // Handle sub-32-bit byte loads (i8) on RawBuffer by loading an aligned i32
    // word and extracting the byte.
    if (ScalarTy->isIntegerTy(8)) {
      Value *AlignedOff = B.CreateAnd(Index, B.getInt32(~3u));
      Value *ByteShift = B.CreateShl(B.CreateAnd(Index, B.getInt32(3u)), B.getInt32(3));
      Function *LoadFn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::dx_resource_load_rawbuffer, {B.getInt32Ty(), RH.DxHandleTy});
      Value *Pair = B.CreateCall(LoadFn, {RH.DxHandle, AlignedOff, OffArg});
      Value *Word = B.CreateExtractValue(Pair, {0});
      Value *ByteVal = B.CreateTrunc(B.CreateLShr(Word, ByteShift), B.getInt8Ty());
      if (NumVecComps == 1) {
        LoadedVal = ByteVal;
      } else {
        Value *Vec = PoisonValue::get(RetTy);
        Vec = B.CreateInsertElement(Vec, ByteVal, B.getInt32(0));
        LoadedVal = Vec;
      }
    } else {
      Function *LoadFn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::dx_resource_load_rawbuffer, {LoadTy, RH.DxHandleTy});
      Value *Pair = B.CreateCall(LoadFn, {RH.DxHandle, Index, OffArg});
      Value *RawVal = B.CreateExtractValue(Pair, {0});
      if (LoadTy == RetTy) {
        LoadedVal = RawVal;
      } else if (NumVecComps > 1 && LoadComps == 1) {
        Value *Vec = PoisonValue::get(RetTy);
        LoadedVal = B.CreateInsertElement(Vec, RawVal, B.getInt32(0));
      } else if (NumVecComps > LoadComps) {
        Value *Vec = PoisonValue::get(RetTy);
        for (unsigned I = 0; I < LoadComps; ++I) {
          Value *E = B.CreateExtractElement(RawVal, B.getInt32(I));
          Vec = B.CreateInsertElement(Vec, E, B.getInt32(I));
        }
        LoadedVal = Vec;
      } else {
        LoadedVal = RawVal;
      }
    }
  }

  CI->replaceAllUsesWith(LoadedVal);
  CI->eraseFromParent();
  return Error::success();
}

Error DXILLowerer::lowerBufferStore(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  Value *Index = CI->getArgOperand(2);
  Value *ElemOffset = CI->getArgOperand(3);
  uint32_t Mask = cast<ConstantInt>(CI->getArgOperand(4))->getZExtValue();
  unsigned NumVals = CI->arg_size() - 5;

  IRBuilder<> B(CI);
  bool IsTyped = RH.RB && RH.RB->Kind == ResourceKind::TypedBuffer;
  bool IsRaw = !RH.RB || RH.RB->Kind == ResourceKind::RawBuffer;

  SmallVector<Value *, 4> Vals;
  for (unsigned I = 0; I < NumVals; ++I)
    Vals.push_back(CI->getArgOperand(5 + I));

  if (Vals.empty()) {
    CI->eraseFromParent();
    return Error::success();
  }

  Type *ScalarTy = Vals[0]->getType();
  unsigned ActiveComps = 0;
  for (unsigned I = 0; I < Vals.size(); ++I)
    if (Mask & (1u << I))
      ActiveComps = I + 1;
  if (ActiveComps == 0)
    ActiveComps = 1;

  if (IsTyped) {
    // A DXIL typed buffer store writes at most 128 bits (4 x 32-bit = 2 x 64-bit).
    // Passing <4 x i64> causes LLVM's DXILOpLowering to split into two overlapping
    // dx.op.bufferStore.i32 calls at the same index, overwriting with zeros.
    unsigned StoreVecLen = (ScalarTy->getPrimitiveSizeInBits() == 64) ? 2u : 4u;
    Type *VecTy = FixedVectorType::get(ScalarTy, StoreVecLen);
    Value *Vec = PoisonValue::get(VecTy);
    for (unsigned I = 0; I < StoreVecLen; ++I) {
      Value *V =
          (I < Vals.size() && (Mask & (1u << I)) && !isa<UndefValue>(Vals[I]))
              ? Vals[I]
              : Constant::getNullValue(ScalarTy);
      Vec = B.CreateInsertElement(Vec, V, B.getInt32(I));
    }
    Function *StoreFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_resource_store_typedbuffer, {RH.DxHandleTy, VecTy});
    B.CreateCall(StoreFn, {RH.DxHandle, Index, Vec});
  } else {
    Value *OffArg = IsRaw ? static_cast<Value *>(PoisonValue::get(B.getInt32Ty()))
                          : ElemOffset;
    if (!IsRaw && isa<UndefValue>(OffArg))
      OffArg = B.getInt32(0);

    Value *StoreVal = nullptr;
    if (ActiveComps == 1) {
      StoreVal = Vals[0];
    } else {
      Type *VecTy = FixedVectorType::get(ScalarTy, ActiveComps);
      Value *Vec = PoisonValue::get(VecTy);
      for (unsigned I = 0; I < ActiveComps; ++I)
        Vec = B.CreateInsertElement(Vec, Vals[I], B.getInt32(I));
      StoreVal = Vec;
    }

    Function *StoreFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_resource_store_rawbuffer,
        {RH.DxHandleTy, StoreVal->getType()});
    B.CreateCall(StoreFn, {RH.DxHandle, Index, OffArg, StoreVal});
  }

  CI->eraseFromParent();
  return Error::success();
}

Error DXILLowerer::lowerCBufferLoad(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  Value *RegIdx = CI->getArgOperand(2);
  Type *RetTy = CI->getType();
  IRBuilder<> B(CI);

  Type *ScalarTy = RetTy->getScalarType();
  unsigned Bits = ScalarTy->getPrimitiveSizeInBits();
  unsigned RetElems = 1;
  if (auto *VT = dyn_cast<FixedVectorType>(RetTy))
    RetElems = VT->getNumElements();

  Value *RowStruct = nullptr;
  unsigned RowCount = 4;
  if (Bits == 64) {
    RowCount = 2;
    Function *LoadFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_resource_load_cbufferrow_2,
        {ScalarTy, ScalarTy, RH.DxHandleTy});
    RowStruct = B.CreateCall(LoadFn, {RH.DxHandle, RegIdx});
  } else if (Bits == 16) {
    RowCount = 8;
    Function *LoadFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_resource_load_cbufferrow_8,
        {ScalarTy, ScalarTy, ScalarTy, ScalarTy, ScalarTy, ScalarTy, ScalarTy,
         ScalarTy, RH.DxHandleTy});
    RowStruct = B.CreateCall(LoadFn, {RH.DxHandle, RegIdx});
  } else {
    RowCount = 4;
    Type *LoadScalar = (Bits == 32) ? ScalarTy : B.getInt32Ty();
    Function *LoadFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_resource_load_cbufferrow_4,
        {LoadScalar, LoadScalar, LoadScalar, LoadScalar, RH.DxHandleTy});
    RowStruct = B.CreateCall(LoadFn, {RH.DxHandle, RegIdx});
  }

  Value *Result = nullptr;
  if (RetElems == 1) {
    Result = B.CreateExtractValue(RowStruct, {0});
    if (Result->getType() != RetTy)
      Result = B.CreateTrunc(Result, RetTy);
  } else {
    Value *Vec = PoisonValue::get(RetTy);
    for (unsigned I = 0; I < std::min(RetElems, RowCount); ++I) {
      Value *E = B.CreateExtractValue(RowStruct, {I});
      if (E->getType() != ScalarTy)
        E = B.CreateTrunc(E, ScalarTy);
      Vec = B.CreateInsertElement(Vec, E, B.getInt32(I));
    }
    Result = Vec;
  }

  CI->replaceAllUsesWith(Result);
  CI->eraseFromParent();
  return Error::success();
}

Error DXILLowerer::lowerBufferUpdateCounter(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;
  Value *Inc = CI->getArgOperand(2);

  IRBuilder<> B(CI);
  Function *Fn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::dx_resource_updatecounter, {RH.DxHandleTy});
  Value *Res = B.CreateCall(Fn, {RH.DxHandle, Inc});
  CI->replaceAllUsesWith(Res);
  CI->eraseFromParent();
  return Error::success();
}

Error DXILLowerer::lowerBufferAtomic(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  uint32_t AtomicOpCode = cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();
  Value *Coord0 = CI->getArgOperand(3);
  Value *Coord1 = CI->getArgOperand(4);
  Value *Coord2 = CI->getArgOperand(5);
  Value *Val = CI->getArgOperand(6);

  IRBuilder<> B(CI);
  bool IsRaw = !RH.RB || RH.RB->Kind == ResourceKind::RawBuffer;
  if (IsRaw) {
    Coord1 = PoisonValue::get(B.getInt32Ty());
    Coord2 = PoisonValue::get(B.getInt32Ty());
  } else if (RH.RB->Kind == ResourceKind::StructuredBuffer) {
    Coord2 = PoisonValue::get(B.getInt32Ty());
  }

  Function *Fn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::dx_resource_atomic_binop, {Val->getType(), RH.DxHandleTy});
  Value *Res = B.CreateCall(Fn, {RH.DxHandle, B.getInt32(AtomicOpCode), Coord0,
                                 Coord1, Coord2, Val});
  CI->replaceAllUsesWith(Res);
  CI->eraseFromParent();
  return Error::success();
}

Error DXILLowerer::lowerBufferAtomicCmpXchg(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  Value *Coord0 = CI->getArgOperand(2);
  Value *Coord1 = CI->getArgOperand(3);
  Value *Coord2 = CI->getArgOperand(4);
  Value *Cmp = CI->getArgOperand(5);
  Value *Val = CI->getArgOperand(6);

  IRBuilder<> B(CI);
  bool IsRaw = !RH.RB || RH.RB->Kind == ResourceKind::RawBuffer;
  if (IsRaw) {
    Coord1 = PoisonValue::get(B.getInt32Ty());
    Coord2 = PoisonValue::get(B.getInt32Ty());
  } else if (RH.RB->Kind == ResourceKind::StructuredBuffer) {
    Coord2 = PoisonValue::get(B.getInt32Ty());
  }

  Function *Fn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::dx_resource_atomic_compare_exchange,
      {Val->getType(), RH.DxHandleTy});
  Value *Res = B.CreateCall(Fn, {RH.DxHandle, Coord0, Coord1, Coord2, Cmp, Val});
  CI->replaceAllUsesWith(Res);
  CI->eraseFromParent();
  return Error::success();
}

Error DXILLowerer::lowerTexLoad(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  Value *MipOrSample = CI->getArgOperand(2);
  uint32_t NumCoords = cast<ConstantInt>(CI->getArgOperand(3))->getZExtValue();
  IRBuilder<> B(CI);

  Value *CoordVal = nullptr;
  if (NumCoords == 1) {
    CoordVal = CI->getArgOperand(4);
  } else {
    Type *CoordTy = FixedVectorType::get(B.getInt32Ty(), NumCoords);
    CoordVal = PoisonValue::get(CoordTy);
    for (unsigned I = 0; I < NumCoords; ++I)
      CoordVal = B.CreateInsertElement(CoordVal, CI->getArgOperand(4 + I),
                                       B.getInt32(I));
  }
  auto spatialDims = [](const ResourceBinding *RB, uint32_t Fallback) -> uint32_t {
    if (!RB)
      return Fallback;
    switch (RB->Kind) {
    case ResourceKind::Texture1D:
    case ResourceKind::Texture1DArray:
      return 1;
    case ResourceKind::Texture2D:
    case ResourceKind::Texture2DArray:
    case ResourceKind::Texture2DMS:
    case ResourceKind::Texture2DMSArray:
      return 2;
    case ResourceKind::Texture3D:
      return 3;
    case ResourceKind::TextureCube:
    case ResourceKind::TextureCubeArray:
      return 0;
    default:
      return Fallback;
    }
  };
  uint32_t SpatialDims = spatialDims(RH.RB, NumCoords);
  Value *OffsetVal = UndefValue::get(CoordVal->getType());
  unsigned OffStart = 4 + NumCoords;
  uint32_t NumOffsets = 0;
  if (OffStart < CI->arg_size())
    NumOffsets = cast<ConstantInt>(CI->getArgOperand(OffStart))->getZExtValue();
  bool IsUAV = RH.RB && RH.RB->Class == ResourceClass::UAV;
  if (SpatialDims > 0 && !IsUAV) {
    if (NumCoords == 1) {
      OffsetVal = (NumOffsets > 0) ? CI->getArgOperand(OffStart + 1)
                                   : static_cast<Value *>(B.getInt32(0));
    } else {
      Type *OffTy = CoordVal->getType();
      OffsetVal = UndefValue::get(OffTy);
      for (unsigned I = 0; I < std::min(SpatialDims, NumCoords); ++I) {
        Value *OV = (I < NumOffsets) ? CI->getArgOperand(OffStart + 1 + I)
                                     : static_cast<Value *>(B.getInt32(0));
        OffsetVal = B.CreateInsertElement(OffsetVal, OV, B.getInt32(I));
      }
    }
  }
  bool IsMS = RH.RB && (RH.RB->Kind == ResourceKind::Texture2DMS ||
                        RH.RB->Kind == ResourceKind::Texture2DMSArray);
  Value *Res = nullptr;
  if (IsMS) {
    Function *MSFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_resource_load_ms,
        {CI->getType(), RH.DxHandleTy, CoordVal->getType(),
         OffsetVal->getType()});
    Res = B.CreateCall(MSFn, {RH.DxHandle, CoordVal, MipOrSample, OffsetVal});
  } else {
    Function *LvlFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_resource_load_level,
        {CI->getType(), RH.DxHandleTy, CoordVal->getType(),
         MipOrSample->getType(), OffsetVal->getType()});
    Res = B.CreateCall(LvlFn, {RH.DxHandle, CoordVal, MipOrSample, OffsetVal});
  }
  CI->replaceAllUsesWith(Res);
  CI->eraseFromParent();
  return Error::success();
}

Error DXILLowerer::lowerTexStore(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  uint32_t NumCoords = cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();
  IRBuilder<> B(CI);

  Value *CoordVal = nullptr;
  if (NumCoords == 1) {
    CoordVal = CI->getArgOperand(3);
  } else {
    Type *CoordTy = FixedVectorType::get(B.getInt32Ty(), NumCoords);
    CoordVal = PoisonValue::get(CoordTy);
    for (unsigned I = 0; I < NumCoords; ++I)
      CoordVal = B.CreateInsertElement(CoordVal, CI->getArgOperand(3 + I),
                                       B.getInt32(I));
  }

  unsigned ValStart = 3 + NumCoords;
  unsigned NumVals = CI->arg_size() - ValStart;
  Type *ElemTy = CI->getArgOperand(ValStart)->getType();
  // Always pad texture UAV stores to 4 channels with zeroes so dx.op.textureStore
  // never receives undef lanes (which fails DXIL validation).
  Type *Vec4Ty = FixedVectorType::get(ElemTy, 4);
  Value *StoreVal = PoisonValue::get(Vec4Ty);
  for (unsigned I = 0; I < 4; ++I) {
    Value *V = (I < NumVals && !isa<UndefValue>(CI->getArgOperand(ValStart + I)))
                   ? CI->getArgOperand(ValStart + I)
                   : Constant::getNullValue(ElemTy);
    StoreVal = B.CreateInsertElement(StoreVal, V, B.getInt32(I));
  }

  Function *Fn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::dx_resource_store_texture,
      {RH.DxHandleTy, CoordVal->getType(), Vec4Ty});
  B.CreateCall(Fn, {RH.DxHandle, CoordVal, StoreVal});
  CI->eraseFromParent();
  return Error::success();
}

Error DXILLowerer::lowerTexSample(CallInst *CI) {
  auto TexOrErr = resolveHandle(CI->getArgOperand(1));
  if (!TexOrErr)
    return TexOrErr.takeError();
  auto SmpOrErr = resolveHandle(CI->getArgOperand(2));
  if (!SmpOrErr)
    return SmpOrErr.takeError();
  const ResolvedHandle &TexRH = *TexOrErr;
  const ResolvedHandle &SmpRH = *SmpOrErr;

  auto Mode = static_cast<ir::SampleMode>(
      cast<ConstantInt>(CI->getArgOperand(3))->getZExtValue());
  uint32_t NumCoords = cast<ConstantInt>(CI->getArgOperand(4))->getZExtValue();
  Value *LodOrBias = CI->getArgOperand(5);
  Value *CmpVal = CI->getArgOperand(6);

  IRBuilder<> B(CI);
  Value *CoordVal = nullptr;
  if (NumCoords == 1) {
    CoordVal = CI->getArgOperand(7);
  } else {
    Type *CoordTy = FixedVectorType::get(B.getFloatTy(), NumCoords);
    CoordVal = PoisonValue::get(CoordTy);
    for (unsigned I = 0; I < NumCoords; ++I)
      CoordVal = B.CreateInsertElement(CoordVal, CI->getArgOperand(7 + I),
                                       B.getInt32(I));
  }
  auto spatialDims = [](const ResourceBinding *RB, uint32_t Fallback) -> uint32_t {
    if (!RB)
      return Fallback;
    switch (RB->Kind) {
    case ResourceKind::Texture1D:
    case ResourceKind::Texture1DArray:
      return 1;
    case ResourceKind::Texture2D:
    case ResourceKind::Texture2DArray:
    case ResourceKind::Texture2DMS:
    case ResourceKind::Texture2DMSArray:
      return 2;
    case ResourceKind::Texture3D:
      return 3;
    case ResourceKind::TextureCube:
    case ResourceKind::TextureCubeArray:
      return 0;
    default:
      return Fallback;
    }
  };
  uint32_t SpatialDims = spatialDims(TexRH.RB, NumCoords);
  Type *OffsetTy = (NumCoords == 1)
                       ? static_cast<Type *>(B.getInt32Ty())
                       : static_cast<Type *>(
                             FixedVectorType::get(B.getInt32Ty(), NumCoords));
  Value *ZeroOffset = UndefValue::get(OffsetTy);
  unsigned OffIdx = 7 + NumCoords;
  if (Mode == ir::SampleMode::Grad && OffIdx < CI->arg_size()) {
    uint32_t NumGrads =
        cast<ConstantInt>(CI->getArgOperand(OffIdx))->getZExtValue();
    OffIdx += 1 + 2 * NumGrads;
  }
  bool HasClamp = false;
  Value *ClampVal = nullptr;
  uint32_t NumOffsets = 0;
  if (OffIdx < CI->arg_size()) {
    NumOffsets = cast<ConstantInt>(CI->getArgOperand(OffIdx))->getZExtValue();
    unsigned ClampIdx = OffIdx + 1 + NumOffsets;
    if (ClampIdx + 1 < CI->arg_size()) {
      HasClamp = cast<ConstantInt>(CI->getArgOperand(ClampIdx))->getZExtValue() != 0;
      if (HasClamp)
        ClampVal = CI->getArgOperand(ClampIdx + 1);
    }
  }
  if (SpatialDims > 0) {
    if (NumCoords == 1) {
      ZeroOffset = (NumOffsets > 0) ? CI->getArgOperand(OffIdx + 1)
                                    : static_cast<Value *>(B.getInt32(0));
    } else {
      Value *OV = UndefValue::get(OffsetTy);
      for (unsigned I = 0; I < std::min(SpatialDims, NumCoords); ++I) {
        Value *Lane = (I < NumOffsets) ? CI->getArgOperand(OffIdx + 1 + I)
                                       : static_cast<Value *>(B.getInt32(0));
        OV = B.CreateInsertElement(OV, Lane, B.getInt32(I));
      }
      ZeroOffset = OV;
    }
  }

  Value *Res = nullptr;
  switch (Mode) {
  case ir::SampleMode::Implicit: {
    if (HasClamp) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::dx_resource_sample_clamp,
          {CI->getType(), TexRH.DxHandleTy, SmpRH.DxHandleTy,
           CoordVal->getType(), ZeroOffset->getType()});
      Res = B.CreateCall(
          Fn, {TexRH.DxHandle, SmpRH.DxHandle, CoordVal, ZeroOffset, ClampVal});
    } else {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::dx_resource_sample,
          {CI->getType(), TexRH.DxHandleTy, SmpRH.DxHandleTy,
           CoordVal->getType(), ZeroOffset->getType()});
      Res = B.CreateCall(Fn, {TexRH.DxHandle, SmpRH.DxHandle, CoordVal, ZeroOffset});
    }
    break;
  }
  case ir::SampleMode::Bias: {
    if (HasClamp) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::dx_resource_samplebias_clamp,
          {CI->getType(), TexRH.DxHandleTy, SmpRH.DxHandleTy,
           CoordVal->getType(), ZeroOffset->getType()});
      Res = B.CreateCall(Fn, {TexRH.DxHandle, SmpRH.DxHandle, CoordVal, LodOrBias,
                              ZeroOffset, ClampVal});
    } else {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::dx_resource_samplebias,
          {CI->getType(), TexRH.DxHandleTy, SmpRH.DxHandleTy,
           CoordVal->getType(), ZeroOffset->getType()});
      Res = B.CreateCall(Fn, {TexRH.DxHandle, SmpRH.DxHandle, CoordVal, LodOrBias,
                              ZeroOffset});
    }
    break;
  }
  case ir::SampleMode::Level: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_resource_samplelevel,
        {CI->getType(), TexRH.DxHandleTy, SmpRH.DxHandleTy,
         CoordVal->getType(), ZeroOffset->getType()});
    Res = B.CreateCall(Fn, {TexRH.DxHandle, SmpRH.DxHandle, CoordVal, LodOrBias,
                            ZeroOffset});
    break;
  }
  case ir::SampleMode::Grad: {
    unsigned GradIdx = 7 + NumCoords;
    uint32_t NumGrads =
        GradIdx < CI->arg_size()
            ? cast<ConstantInt>(CI->getArgOperand(GradIdx))->getZExtValue()
            : 0u;
    auto buildGradVec = [&](unsigned Start) -> Value * {
      if (NumGrads <= 1)
        return Start < CI->arg_size() ? CI->getArgOperand(Start)
                                      : ConstantFP::get(B.getFloatTy(), 0.0);
      Type *GTy = FixedVectorType::get(B.getFloatTy(), NumGrads);
      Value *GV = PoisonValue::get(GTy);
      for (unsigned I = 0; I < NumGrads; ++I)
        GV = B.CreateInsertElement(GV, CI->getArgOperand(Start + I),
                                   B.getInt32(I));
      return GV;
    };
    Value *DDX = buildGradVec(GradIdx + 1);
    Value *DDY = buildGradVec(GradIdx + 1 + NumGrads);
    if (HasClamp) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::dx_resource_samplegrad_clamp,
          {CI->getType(), TexRH.DxHandleTy, SmpRH.DxHandleTy,
           CoordVal->getType(), DDX->getType(), DDY->getType(),
           ZeroOffset->getType()});
      Res = B.CreateCall(Fn, {TexRH.DxHandle, SmpRH.DxHandle, CoordVal, DDX,
                              DDY, ZeroOffset, ClampVal});
    } else {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::dx_resource_samplegrad,
          {CI->getType(), TexRH.DxHandleTy, SmpRH.DxHandleTy,
           CoordVal->getType(), DDX->getType(), DDY->getType(),
           ZeroOffset->getType()});
      Res = B.CreateCall(
          Fn, {TexRH.DxHandle, SmpRH.DxHandle, CoordVal, DDX, DDY, ZeroOffset});
    }
    break;
  }
  case ir::SampleMode::Cmp: {
    if (HasClamp) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::dx_resource_samplecmp_clamp,
          {CI->getType(), TexRH.DxHandleTy, SmpRH.DxHandleTy,
           CoordVal->getType(), ZeroOffset->getType()});
      Res = B.CreateCall(Fn, {TexRH.DxHandle, SmpRH.DxHandle, CoordVal, CmpVal,
                              ZeroOffset, ClampVal});
    } else {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::dx_resource_samplecmp,
          {CI->getType(), TexRH.DxHandleTy, SmpRH.DxHandleTy,
           CoordVal->getType(), ZeroOffset->getType()});
      Res = B.CreateCall(Fn, {TexRH.DxHandle, SmpRH.DxHandle, CoordVal, CmpVal,
                              ZeroOffset});
    }
    break;
  }
  case ir::SampleMode::CmpLevelZero: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_resource_samplecmplevelzero,
        {CI->getType(), TexRH.DxHandleTy, SmpRH.DxHandleTy,
         CoordVal->getType(), ZeroOffset->getType()});
    Res = B.CreateCall(Fn, {TexRH.DxHandle, SmpRH.DxHandle, CoordVal, CmpVal,
                            ZeroOffset});
    break;
  }
  default:
    return makeError("DirectX backend: unsupported texture sample mode {0}",
                     static_cast<unsigned>(Mode));
  }

  CI->replaceAllUsesWith(Res);
  CI->eraseFromParent();
  return Error::success();
}

Error DXILLowerer::lowerTexGather(CallInst *CI) {
  auto TexOrErr = resolveHandle(CI->getArgOperand(1));
  if (!TexOrErr)
    return TexOrErr.takeError();
  auto SmpOrErr = resolveHandle(CI->getArgOperand(2));
  if (!SmpOrErr)
    return SmpOrErr.takeError();
  const ResolvedHandle &TexRH = *TexOrErr;
  const ResolvedHandle &SmpRH = *SmpOrErr;

  Value *Channel = CI->getArgOperand(3);
  uint32_t NumCoords = cast<ConstantInt>(CI->getArgOperand(4))->getZExtValue();
  IRBuilder<> B(CI);

  Type *CoordTy = FixedVectorType::get(B.getFloatTy(), NumCoords);
  Value *CoordVal = PoisonValue::get(CoordTy);
  for (unsigned I = 0; I < NumCoords; ++I)
    CoordVal = B.CreateInsertElement(CoordVal, CI->getArgOperand(5 + I),
                                     B.getInt32(I));
  Type *OffsetTy = FixedVectorType::get(B.getInt32Ty(), NumCoords);
  Value *ZeroOffset = Constant::getNullValue(OffsetTy);

  Function *Fn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::dx_resource_gather,
      {CI->getType(), TexRH.DxHandleTy, SmpRH.DxHandleTy, CoordVal->getType(),
       ZeroOffset->getType()});
  Value *Res = B.CreateCall(
      Fn, {TexRH.DxHandle, SmpRH.DxHandle, CoordVal, Channel, ZeroOffset});
  CI->replaceAllUsesWith(Res);
  CI->eraseFromParent();
  return Error::success();
}

Error DXILLowerer::lowerTexCalcLod(CallInst *CI) {
  auto TexOrErr = resolveHandle(CI->getArgOperand(1));
  if (!TexOrErr)
    return TexOrErr.takeError();
  auto SmpOrErr = resolveHandle(CI->getArgOperand(2));
  if (!SmpOrErr)
    return SmpOrErr.takeError();
  bool Clamped = cast<ConstantInt>(CI->getArgOperand(3))->getZExtValue() != 0;
  uint32_t NumCoords = cast<ConstantInt>(CI->getArgOperand(4))->getZExtValue();

  IRBuilder<> B(CI);
  Value *CoordVal = nullptr;
  if (NumCoords == 1) {
    CoordVal = CI->getArgOperand(5);
  } else {
    Type *CoordTy = FixedVectorType::get(B.getFloatTy(), NumCoords);
    CoordVal = PoisonValue::get(CoordTy);
    for (unsigned I = 0; I < NumCoords; ++I)
      CoordVal = B.CreateInsertElement(CoordVal, CI->getArgOperand(5 + I),
                                       B.getInt32(I));
  }
  Function *Fn = Intrinsic::getOrInsertDeclaration(
      &M,
      Clamped ? Intrinsic::dx_resource_calculate_lod
              : Intrinsic::dx_resource_calculate_lod_unclamped,
      {TexOrErr->DxHandleTy, SmpOrErr->DxHandleTy, CoordVal->getType()});
  Value *Res =
      B.CreateCall(Fn, {TexOrErr->DxHandle, SmpOrErr->DxHandle, CoordVal});
  CI->replaceAllUsesWith(Res);
  CI->eraseFromParent();
  return Error::success();
}

Error DXILLowerer::lowerTexGetDimensions(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;
  Value *Mip = CI->getArgOperand(2);
  IRBuilder<> B(CI);

  Type *RetTy = CI->getType();
  Value *Res = PoisonValue::get(RetTy);
  if (RH.RB && (RH.RB->Kind == ResourceKind::Texture2D ||
                RH.RB->Kind == ResourceKind::Texture2DArray)) {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_resource_getdimensions_levels_xy, {RH.DxHandleTy});
    Value *V3 = B.CreateCall(Fn, {RH.DxHandle, Mip});
    if (auto *VT = dyn_cast<FixedVectorType>(RetTy)) {
      unsigned N = VT->getNumElements();
      for (unsigned I = 0; I < std::min(N, 3u); ++I)
        Res = B.CreateInsertElement(
            Res, B.CreateExtractElement(V3, B.getInt32(I)), B.getInt32(I));
      for (unsigned I = 3; I < N; ++I)
        Res = B.CreateInsertElement(Res, B.getInt32(1), B.getInt32(I));
    } else {
      Res = B.CreateExtractElement(V3, B.getInt32(0));
    }
  } else {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_resource_getdimensions_x, {RH.DxHandleTy});
    Value *W = B.CreateCall(Fn, {RH.DxHandle});
    if (auto *VT = dyn_cast<FixedVectorType>(RetTy)) {
      Res = B.CreateInsertElement(Res, W, B.getInt32(0));
      for (unsigned I = 1; I < VT->getNumElements(); ++I)
        Res = B.CreateInsertElement(Res, B.getInt32(1), B.getInt32(I));
    } else {
      Res = W;
    }
  }
  CI->replaceAllUsesWith(Res);
  CI->eraseFromParent();
  return Error::success();
}

Error DXILLowerer::lowerBarrier(CallInst *CI) {
  bool ExecSync = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue() != 0;
  bool TgsmFence = cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue() != 0;
  bool DeviceFence = cast<ConstantInt>(CI->getArgOperand(3))->getZExtValue() != 0;

  IRBuilder<> B(CI);
  Intrinsic::ID ID = Intrinsic::dx_group_memory_barrier_with_group_sync;
  if (TgsmFence && DeviceFence)
    ID = ExecSync ? Intrinsic::dx_all_memory_barrier_with_group_sync
                  : Intrinsic::dx_all_memory_barrier;
  else if (DeviceFence)
    ID = ExecSync ? Intrinsic::dx_device_memory_barrier_with_group_sync
                  : Intrinsic::dx_device_memory_barrier;
  else
    ID = ExecSync ? Intrinsic::dx_group_memory_barrier_with_group_sync
                  : Intrinsic::dx_group_memory_barrier;

  Function *Fn = Intrinsic::getOrInsertDeclaration(&M, ID);
  B.CreateCall(Fn);
  CI->eraseFromParent();
  return Error::success();
}

Error DXILLowerer::lowerWaveOp(CallInst *CI, DxOp Op) {
  IRBuilder<> B(CI);
  Value *Res = nullptr;
  switch (Op) {
  case DxOp::WaveIsFirstLane: {
    Function *Fn =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::dx_wave_is_first_lane);
    Res = B.CreateCall(Fn);
    break;
  }
  case DxOp::WaveVote: {
    Value *Val = CI->getArgOperand(1);
    auto Kind = static_cast<ir::WaveVoteKind>(
        cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue());
    if (Kind == ir::WaveVoteKind::All) {
      Function *Fn =
          Intrinsic::getOrInsertDeclaration(&M, Intrinsic::dx_wave_all);
      Res = B.CreateCall(Fn, {Val});
    } else if (Kind == ir::WaveVoteKind::Any) {
      Function *Fn =
          Intrinsic::getOrInsertDeclaration(&M, Intrinsic::dx_wave_any);
      Res = B.CreateCall(Fn, {Val});
    } else {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::dx_wave_all_equal, {Val->getType()});
      Res = B.CreateCall(Fn, {Val});
    }
    break;
  }
  case DxOp::WaveBallot: {
    Value *Cond = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_wave_ballot, {B.getInt32Ty()});
    Value *Agg = B.CreateCall(Fn, {Cond});
    if (auto *VT = dyn_cast<FixedVectorType>(CI->getType())) {
      Value *Vec = PoisonValue::get(VT);
      for (unsigned I = 0; I < std::min(VT->getNumElements(), 4u); ++I)
        Vec = B.CreateInsertElement(Vec, B.CreateExtractValue(Agg, {I}),
                                    B.getInt32(I));
      Res = Vec;
    } else {
      Res = B.CreateExtractValue(Agg, {0});
    }
    break;
  }
  case DxOp::WaveBitCount: {
    Value *Cond = CI->getArgOperand(1);
    auto Kind = static_cast<ir::WaveBitCountKind>(
        cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue());
    if (Kind == ir::WaveBitCountKind::All) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::dx_wave_active_countbits);
      Res = B.CreateCall(Fn, {Cond});
    } else if (Kind == ir::WaveBitCountKind::Prefix) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::dx_wave_prefix_bit_count);
      Res = B.CreateCall(Fn, {Cond});
    } else {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::dx_wave_multi_prefix_bit_count);
      Value *M0 = CI->arg_size() > 3 ? CI->getArgOperand(3) : B.getInt32(0);
      Value *M1 = CI->arg_size() > 4 ? CI->getArgOperand(4) : B.getInt32(0);
      Value *M2 = CI->arg_size() > 5 ? CI->getArgOperand(5) : B.getInt32(0);
      Value *M3 = CI->arg_size() > 6 ? CI->getArgOperand(6) : B.getInt32(0);
      Res = B.CreateCall(Fn, {Cond, M0, M1, M2, M3});
    }
    break;
  }
  case DxOp::WaveMultiPrefix: {
    Value *Val = CI->getArgOperand(1);
    auto OpKind = static_cast<ir::WaveMultiPrefixOpKind>(
        cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue());
    bool IsUnsigned =
        cast<ConstantInt>(CI->getArgOperand(3))->getZExtValue() != 0;
    Value *M0 = CI->arg_size() > 4 ? CI->getArgOperand(4) : B.getInt32(0);
    Value *M1 = CI->arg_size() > 5 ? CI->getArgOperand(5) : B.getInt32(0);
    Value *M2 = CI->arg_size() > 6 ? CI->getArgOperand(6) : B.getInt32(0);
    Value *M3 = CI->arg_size() > 7 ? CI->getArgOperand(7) : B.getInt32(0);
    Intrinsic::ID ID = Intrinsic::dx_wave_multi_prefix_sum;
    switch (OpKind) {
    case ir::WaveMultiPrefixOpKind::Sum:
      ID = (IsUnsigned && Val->getType()->isIntOrIntVectorTy())
               ? Intrinsic::dx_wave_multi_prefix_usum
               : Intrinsic::dx_wave_multi_prefix_sum;
      break;
    case ir::WaveMultiPrefixOpKind::Product:
      ID = (IsUnsigned && Val->getType()->isIntOrIntVectorTy())
               ? Intrinsic::dx_wave_multi_prefix_uproduct
               : Intrinsic::dx_wave_multi_prefix_product;
      break;
    case ir::WaveMultiPrefixOpKind::And:
      ID = Intrinsic::dx_wave_multi_prefix_bit_and;
      break;
    case ir::WaveMultiPrefixOpKind::Or:
      ID = Intrinsic::dx_wave_multi_prefix_bit_or;
      break;
    case ir::WaveMultiPrefixOpKind::Xor:
      ID = Intrinsic::dx_wave_multi_prefix_bit_xor;
      break;
    }
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, ID, {Val->getType()});
    Res = B.CreateCall(Fn, {Val, M0, M1, M2, M3});
    break;
  }
  case DxOp::WaveReadLaneFirst: {
    Value *Val = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_wave_readlane_first, {Val->getType()});
    Res = B.CreateCall(Fn, {Val});
    break;
  }
  case DxOp::WaveReadLaneAt: {
    Value *Val = CI->getArgOperand(1);
    Value *Lane = CI->getArgOperand(2);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_wave_readlane, {Val->getType()});
    Res = B.CreateCall(Fn, {Val, Lane});
    break;
  }
  case DxOp::WaveReduce: {
    Value *Val = CI->getArgOperand(1);
    uint32_t Kind = cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();
    // 0=Add, 1=Mul, 2=MinS, 3=MinU, 4=MaxS, 5=MaxU, 6=And, 7=Or, 8=Xor,
    // 9=PrefixAdd, 10=PrefixMul
    Intrinsic::ID ID = Intrinsic::dx_wave_reduce_sum;
    switch (Kind) {
    case 0: ID = Intrinsic::dx_wave_reduce_sum; break;
    case 1: ID = Intrinsic::dx_wave_product; break;
    case 2: ID = Intrinsic::dx_wave_reduce_min; break;
    case 3: ID = Intrinsic::dx_wave_reduce_umin; break;
    case 4: ID = Intrinsic::dx_wave_reduce_max; break;
    case 5: ID = Intrinsic::dx_wave_reduce_umax; break;
    case 6: ID = Intrinsic::dx_wave_reduce_and; break;
    case 7: ID = Intrinsic::dx_wave_reduce_or; break;
    case 8: ID = Intrinsic::dx_wave_reduce_xor; break;
    case 9: ID = Intrinsic::dx_wave_prefix_sum; break;
    case 10: ID = Intrinsic::dx_wave_prefix_product; break;
    default:
      return makeError("DirectX backend: unsupported wave reduce kind {0}", Kind);
    }
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, ID, {Val->getType()});
    Res = B.CreateCall(Fn, {Val});
    break;
  }
  case DxOp::QuadOp: {
    Value *Val = CI->getArgOperand(1);
    auto Kind = static_cast<ir::QuadOpKind>(
        cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue());
    Intrinsic::ID ID = Intrinsic::dx_quad_read_across_x;
    if (Kind == ir::QuadOpKind::ReadAcrossY)
      ID = Intrinsic::dx_quad_read_across_y;
    else if (Kind == ir::QuadOpKind::ReadAcrossDiagonal)
      ID = Intrinsic::dx_quad_read_across_diagonal;
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, ID, {Val->getType()});
    Res = B.CreateCall(Fn, {Val});
    break;
  }
  default:
    return makeError("DirectX backend: unsupported wave/quad op {0}",
                     static_cast<unsigned>(Op));
  }
  CI->replaceAllUsesWith(Res);
  CI->eraseFromParent();
  return Error::success();
}

Error DXILLowerer::lowerMathOrBitOp(CallInst *CI, DxOp Op) {
  IRBuilder<> B(CI);
  Value *Res = nullptr;
  switch (Op) {
  case DxOp::Saturate: {
    Value *X = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::dx_saturate,
                                                     {X->getType()});
    Res = B.CreateCall(Fn, {X});
    break;
  }
  case DxOp::Frc: {
    Value *X = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::dx_frac,
                                                     {X->getType()});
    Res = B.CreateCall(Fn, {X});
    break;
  }
  case DxOp::Rsqrt: {
    Value *X = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::dx_rsqrt,
                                                     {X->getType()});
    Res = B.CreateCall(Fn, {X});
    break;
  }
  case DxOp::IsNaN: {
    Value *X = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::dx_isnan,
                                                     {X->getType()});
    Res = B.CreateCall(Fn, {X});
    break;
  }
  case DxOp::IsInf: {
    Value *X = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::dx_isinf,
                                                     {X->getType()});
    Res = B.CreateCall(Fn, {X});
    break;
  }
  case DxOp::IsFinite: {
    Value *X = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::dx_isfinite,
                                                     {X->getType()});
    Res = B.CreateCall(Fn, {X});
    break;
  }
  case DxOp::FirstBit: {
    Value *X = CI->getArgOperand(1);
    auto Kind = static_cast<ir::FirstBitKind>(
        cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue());
    Intrinsic::ID ID = Intrinsic::dx_firstbitlow;
    if (Kind == ir::FirstBitKind::Hi)
      ID = Intrinsic::dx_firstbituhigh;
    else if (Kind == ir::FirstBitKind::SHi)
      ID = Intrinsic::dx_firstbitshigh;
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, ID, {X->getType()});
    Res = B.CreateCall(Fn, {X});
    break;
  }
  case DxOp::Bfe: {
    Value *Width = CI->getArgOperand(1);
    Value *Offset = CI->getArgOperand(2);
    Value *Val = CI->getArgOperand(3);
    bool IsSigned = cast<ConstantInt>(CI->getArgOperand(4))->getZExtValue() != 0;
    Type *Ty = Val->getType();
    unsigned Bits = Ty->getScalarSizeInBits();
    Value *W = B.CreateZExtOrTrunc(Width, Ty);
    Value *O = B.CreateZExtOrTrunc(Offset, Ty);
    Value *BitsC = ConstantInt::get(Ty, Bits);
    Value *Zero = ConstantInt::get(Ty, 0);
    Value *ShlAmt = B.CreateSub(BitsC, B.CreateAdd(W, O));
    Value *Shl = B.CreateShl(Val, ShlAmt);
    Value *ShrAmt = B.CreateSub(BitsC, W);
    Value *Shifted = IsSigned ? B.CreateAShr(Shl, ShrAmt) : B.CreateLShr(Shl, ShrAmt);
    Res = B.CreateSelect(B.CreateICmpEQ(W, Zero), Zero, Shifted);
    break;
  }
  case DxOp::Bfi: {
    Value *Width = CI->getArgOperand(1);
    Value *Offset = CI->getArgOperand(2);
    Value *Insert = CI->getArgOperand(3);
    Value *Base = CI->getArgOperand(4);
    Type *Ty = Base->getType();
    unsigned Bits = Ty->getScalarSizeInBits();
    Value *W = B.CreateZExtOrTrunc(Width, Ty);
    Value *O = B.CreateZExtOrTrunc(Offset, Ty);
    Value *One = ConstantInt::get(Ty, 1);
    Value *AllOnes = ConstantInt::getAllOnesValue(Ty);
    Value *MaskRaw = B.CreateSub(B.CreateShl(One, W), One);
    Value *Mask = B.CreateSelect(B.CreateICmpUGE(W, ConstantInt::get(Ty, Bits)),
                                 AllOnes, MaskRaw);
    Value *ShiftedMask = B.CreateShl(Mask, O);
    Value *ShiftedIns = B.CreateAnd(B.CreateShl(Insert, O), ShiftedMask);
    Value *Cleared = B.CreateAnd(Base, B.CreateNot(ShiftedMask));
    Res = B.CreateOr(Cleared, ShiftedIns);
    break;
  }
  case DxOp::Dot: {
    unsigned Half = (CI->arg_size() - 1) / 2;
    Type *Ty = CI->getType();
    if (Half == 2) {
      Function *Fn =
          Intrinsic::getOrInsertDeclaration(&M, Intrinsic::dx_dot2, {Ty});
      Res = B.CreateCall(Fn, {CI->getArgOperand(1), CI->getArgOperand(2),
                              CI->getArgOperand(3), CI->getArgOperand(4)});
    } else if (Half == 3) {
      Function *Fn =
          Intrinsic::getOrInsertDeclaration(&M, Intrinsic::dx_dot3, {Ty});
      Res = B.CreateCall(Fn, {CI->getArgOperand(1), CI->getArgOperand(2),
                              CI->getArgOperand(3), CI->getArgOperand(4),
                              CI->getArgOperand(5), CI->getArgOperand(6)});
    } else if (Half == 4) {
      Function *Fn =
          Intrinsic::getOrInsertDeclaration(&M, Intrinsic::dx_dot4, {Ty});
      Res = B.CreateCall(Fn, {CI->getArgOperand(1), CI->getArgOperand(2),
                              CI->getArgOperand(3), CI->getArgOperand(4),
                              CI->getArgOperand(5), CI->getArgOperand(6),
                              CI->getArgOperand(7), CI->getArgOperand(8)});
    } else {
      return makeError("DirectX backend: unsupported dot arity {0}", Half);
    }
    break;
  }
  case DxOp::DotPacked: {
    auto Kind = static_cast<ir::DotPackedKind>(
        cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue());
    Value *Acc = CI->getArgOperand(2);
    if (Kind == ir::DotPackedKind::I8x4) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::dx_dot4add_i8packed);
      Res = B.CreateCall(Fn, {Acc, CI->getArgOperand(3), CI->getArgOperand(4)});
    } else if (Kind == ir::DotPackedKind::U8x4) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::dx_dot4add_u8packed);
      Res = B.CreateCall(Fn, {Acc, CI->getArgOperand(3), CI->getArgOperand(4)});
    } else {
      Function *Fn =
          Intrinsic::getOrInsertDeclaration(&M, Intrinsic::dx_dot2add);
      Res = B.CreateCall(Fn, {Acc, CI->getArgOperand(3), CI->getArgOperand(4),
                              CI->getArgOperand(5), CI->getArgOperand(6)});
    }
    break;
  }
  case DxOp::LegacyF32ToF16: {
    Value *X = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_legacyf32tof16, {X->getType()});
    Res = B.CreateCall(Fn, {X});
    break;
  }
  case DxOp::LegacyF16ToF32: {
    Value *X = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_legacyf16tof32, {X->getType()});
    Res = B.CreateCall(Fn, {X});
    break;
  }
  case DxOp::MakeDouble: {
    Value *Lo = CI->getArgOperand(1);
    Value *Hi = CI->getArgOperand(2);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_asdouble, {Lo->getType()});
    Res = B.CreateCall(Fn, {Lo, Hi});
    break;
  }
  case DxOp::SplitDouble: {
    Value *Val = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_splitdouble, {B.getInt32Ty()});
    Res = B.CreateCall(Fn, {Val});
    break;
  }
  case DxOp::UAddc: {
    Value *L = CI->getArgOperand(1);
    Value *R = CI->getArgOperand(2);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::uadd_with_overflow, {L->getType()});
    Res = B.CreateCall(Fn, {L, R});
    break;
  }
  default:
    return makeError("DirectX backend: unsupported math/bit op {0}",
                     static_cast<unsigned>(Op));
  }
  CI->replaceAllUsesWith(Res);
  CI->eraseFromParent();
  return Error::success();
}

Error DXILLowerer::lowerPixelOp(CallInst *CI, DxOp Op) {
  IRBuilder<> B(CI);
  if (Op == DxOp::LoadInput) {
    Value *InputId = CI->getArgOperand(1);
    Value *Row = CI->getArgOperand(2);
    Value *Col = B.CreateTrunc(CI->getArgOperand(3), B.getInt8Ty());
    Value *VertId = CI->getArgOperand(4);
    if (isa<UndefValue>(VertId))
      VertId = PoisonValue::get(B.getInt32Ty());
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_load_input, {CI->getType()});
    Value *Res = B.CreateCall(Fn, {InputId, Row, Col, VertId});
    CI->replaceAllUsesWith(Res);
    CI->eraseFromParent();
    return Error::success();
  }
  if (Op == DxOp::StoreOutput) {
    Value *OutputId = CI->getArgOperand(1);
    Value *Row = CI->getArgOperand(2);
    Value *Col = B.CreateTrunc(CI->getArgOperand(3), B.getInt8Ty());
    Value *Val = CI->getArgOperand(4);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_store_output, {Val->getType()});
    B.CreateCall(Fn, {OutputId, Row, Col, Val});
    CI->eraseFromParent();
    return Error::success();
  }
  if (Op == DxOp::Discard) {
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::dx_discard);
    B.CreateCall(Fn, {CI->getArgOperand(1)});
    CI->eraseFromParent();
    return Error::success();
  }
  if (Op == DxOp::Deriv) {
    Value *Val = CI->getArgOperand(1);
    auto Axis = static_cast<ir::DerivAxis>(
        cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue());
    auto Quality = static_cast<ir::DerivQuality>(
        cast<ConstantInt>(CI->getArgOperand(3))->getZExtValue());
    Intrinsic::ID ID = Intrinsic::dx_ddx_coarse;
    if (Axis == ir::DerivAxis::X)
      ID = (Quality == ir::DerivQuality::Fine) ? Intrinsic::dx_ddx_fine
                                               : Intrinsic::dx_ddx_coarse;
    else
      ID = (Quality == ir::DerivQuality::Fine) ? Intrinsic::dx_ddy_fine
                                               : Intrinsic::dx_ddy_coarse;
    Value *Res = nullptr;
    if (auto *VTy = dyn_cast<FixedVectorType>(Val->getType())) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, ID, {VTy->getElementType()});
      Res = PoisonValue::get(VTy);
      for (unsigned I = 0; I < VTy->getNumElements(); ++I) {
        Value *Lane = B.CreateExtractElement(Val, B.getInt32(I));
        Value *DLane = B.CreateCall(Fn, {Lane});
        Res = B.CreateInsertElement(Res, DLane, B.getInt32(I));
      }
    } else {
      Function *Fn = Intrinsic::getOrInsertDeclaration(&M, ID, {Val->getType()});
      Res = B.CreateCall(Fn, {Val});
    }
    CI->replaceAllUsesWith(Res);
    CI->eraseFromParent();
    return Error::success();
  }
  return makeError("DirectX backend: unsupported pixel op {0}",
                   static_cast<unsigned>(Op));
}

Error DXILLowerer::lowerBridgeCall(CallInst *CI) {
  auto Op =
      static_cast<DxOp>(cast<ConstantInt>(CI->getArgOperand(0))->getZExtValue());
  IRBuilder<> B(CI);
  switch (Op) {
  case DxOp::BindResource:
    return lowerBindResource(CI);
  case DxOp::HeapResource:
    return lowerHeapResource(CI);
  case DxOp::BufferLoad:
    return lowerBufferLoad(CI);
  case DxOp::BufferStore:
    return lowerBufferStore(CI);
  case DxOp::CBufferLoad:
    return lowerCBufferLoad(CI);
  case DxOp::BufferUpdateCounter:
    return lowerBufferUpdateCounter(CI);
  case DxOp::BufferAtomic:
    return lowerBufferAtomic(CI);
  case DxOp::BufferAtomicCmpXchg:
    return lowerBufferAtomicCmpXchg(CI);
  case DxOp::TexLoad:
    return lowerTexLoad(CI);
  case DxOp::TexStore:
    return lowerTexStore(CI);
  case DxOp::TexSample:
    return lowerTexSample(CI);
  case DxOp::TexGather:
    return lowerTexGather(CI);
  case DxOp::TexCalcLod:
    return lowerTexCalcLod(CI);
  case DxOp::TexGetDimensions:
    return lowerTexGetDimensions(CI);
  case DxOp::Barrier:
    return lowerBarrier(CI);
  case DxOp::ThreadId: {
    Value *Dim = CI->getArgOperand(1);
    Function *Fn =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::dx_thread_id);
    Value *Res = B.CreateCall(Fn, {Dim});
    CI->replaceAllUsesWith(Res);
    CI->eraseFromParent();
    return Error::success();
  }
  case DxOp::GroupId: {
    Value *Dim = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::dx_group_id);
    Value *Res = B.CreateCall(Fn, {Dim});
    CI->replaceAllUsesWith(Res);
    CI->eraseFromParent();
    return Error::success();
  }
  case DxOp::ThreadIdInGroup: {
    Value *Dim = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_thread_id_in_group);
    Value *Res = B.CreateCall(Fn, {Dim});
    CI->replaceAllUsesWith(Res);
    CI->eraseFromParent();
    return Error::success();
  }
  case DxOp::FlattenedThreadIdInGroup: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_flattened_thread_id_in_group);
    Value *Res = B.CreateCall(Fn);
    CI->replaceAllUsesWith(Res);
    CI->eraseFromParent();
    return Error::success();
  }
  case DxOp::LaneId: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_wave_getlaneindex);
    Value *Res = B.CreateCall(Fn);
    CI->replaceAllUsesWith(Res);
    CI->eraseFromParent();
    return Error::success();
  }
  case DxOp::SubgroupSize: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::dx_wave_get_lane_count);
    Value *Res = B.CreateCall(Fn);
    CI->replaceAllUsesWith(Res);
    CI->eraseFromParent();
    return Error::success();
  }
  case DxOp::WaveIsFirstLane:
  case DxOp::WaveVote:
  case DxOp::WaveBallot:
  case DxOp::WaveBitCount:
  case DxOp::WaveMultiPrefix:
  case DxOp::WaveReadLaneFirst:
  case DxOp::WaveReadLaneAt:
  case DxOp::WaveReduce:
  case DxOp::QuadReadLaneAt:
  case DxOp::QuadOp:
    return lowerWaveOp(CI, Op);
  case DxOp::Saturate:
  case DxOp::Frc:
  case DxOp::IsNaN:
  case DxOp::IsInf:
  case DxOp::IsFinite:
  case DxOp::Rsqrt:
  case DxOp::FirstBit:
  case DxOp::Bfe:
  case DxOp::Bfi:
  case DxOp::Dot:
  case DxOp::DotPacked:
  case DxOp::LegacyF32ToF16:
  case DxOp::LegacyF16ToF32:
  case DxOp::MakeDouble:
  case DxOp::SplitDouble:
  case DxOp::UAddc:
    return lowerMathOrBitOp(CI, Op);
  case DxOp::LoadInput:
  case DxOp::StoreOutput:
  case DxOp::Discard:
  case DxOp::Deriv:
    return lowerPixelOp(CI, Op);
  default:
    return makeError("DirectX backend: unsupported bridge op {0}",
                     static_cast<unsigned>(Op));
  }
}

void DXILLowerer::configureEntryAndLinkage(StringRef StageName) {
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    if (&F == Entry) {
      F.setLinkage(GlobalValue::ExternalLinkage);
      F.addFnAttr(Attribute::NoInline);
      F.addFnAttr(Attribute::NoUnwind);
      F.addFnAttr("hlsl.shader", StageName);
      if (Reflect.Stage == ShaderStage::Compute) {
        std::string NT = std::to_string(Reflect.NumThreads[0]) + "," +
                         std::to_string(Reflect.NumThreads[1]) + "," +
                         std::to_string(Reflect.NumThreads[2]);
        F.addFnAttr("hlsl.numthreads", NT);
      }
    } else {
      F.setLinkage(GlobalValue::InternalLinkage);
      F.addFnAttr(Attribute::AlwaysInline);
    }
  }
}

void DXILLowerer::emitSemanticSignatures() {
  if (Reflect.Stage != ShaderStage::Vertex &&
      Reflect.Stage != ShaderStage::Pixel)
    return;
  if (Reflect.Inputs.empty() && Reflect.Outputs.empty())
    return;

  auto I32 = [&](uint32_t V) {
    return ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), V));
  };
  auto I8 = [&](uint8_t V) {
    return ConstantAsMetadata::get(ConstantInt::get(Type::getInt8Ty(Ctx), V));
  };

  auto isIntComp = [](ComponentType T) {
    return T == ComponentType::I1 || T == ComponentType::I16 ||
           T == ComponentType::U16 || T == ComponentType::I32 ||
           T == ComponentType::U32;
  };

  auto buildSigList = [&](const std::vector<StageParameter> &Params,
                          bool IsInput) -> MDNode * {
    if (Params.empty())
      return nullptr;
    SmallVector<Metadata *, 8> Nodes;
    // For VS outputs and PS inputs, assign deterministic stage-independent
    // StartRows so VS OSG1 and PS ISG1 always match on D3D12:
    //   - SV_Position (Kind == 3) -> Row 0
    //   - User varyings with SPIR-V Location L -> Row 1 + L
    //   - Other allocated SGVs (ClipDistance, CullDistance, IsFrontFace, ...)
    //     -> allocated sequentially after the highest user varying row.
    bool IsInterStage = (!IsInput && Reflect.Stage == ShaderStage::Vertex) ||
                        (IsInput && Reflect.Stage == ShaderStage::Pixel);
    uint32_t NextRow = 0;
    if (IsInterStage) {
      NextRow = 1; // Reserve Row 0 for SV_Position
      for (const StageParameter &SP : Params) {
        if (SP.Location != UINT32_MAX) {
          uint32_t Rows = std::max(1u, SP.Rows);
          NextRow = std::max(NextRow, 1u + SP.Location + Rows);
        }
      }
    }
    for (const StageParameter &SP : Params) {
      uint32_t Kind = SP.Kind;
      // Normalize SemanticKind for known system values if Kind == 0.
      if (Kind == 0) {
        if (StringRef(SP.Semantic).equals_insensitive("SV_Position"))
          Kind = 3; // Position
        else if (StringRef(SP.Semantic).equals_insensitive("SV_VertexID"))
          Kind = 1; // VertexID
        else if (StringRef(SP.Semantic).equals_insensitive("SV_InstanceID"))
          Kind = 2; // InstanceID
        else if (StringRef(SP.Semantic).equals_insensitive("SV_Target"))
          Kind = 16; // Target
        else if (StringRef(SP.Semantic).equals_insensitive("SV_IsFrontFace"))
          Kind = 13; // IsFrontFace
      }
      // Only NotPacked depth/stencil outputs omit register allocation;
      // SGV inputs like SV_VertexID (1), SV_InstanceID (2), SV_IsFrontFace (13)
      // require an allocated signature row/column in DXIL.
      bool IsNotAllocated =
          (Kind == 17 || Kind == 26 || Kind == 27 || Kind == 28);
      uint32_t Rows = std::max(1u, SP.Rows);
      uint8_t Cols = static_cast<uint8_t>(std::clamp(SP.Columns, 1u, 4u));
      uint32_t StartRow = ~0u;
      uint8_t StartCol = 0xFFu;
      if (!IsNotAllocated) {
        if (Kind == 16 && !IsInput && Reflect.Stage == ShaderStage::Pixel) {
          StartRow = SP.SemanticIndex;
        } else if (IsInterStage && Kind == 3) {
          StartRow = 0;
        } else if (IsInterStage && SP.Location != UINT32_MAX) {
          StartRow = 1u + SP.Location;
        } else {
          StartRow = NextRow;
          NextRow += Rows;
        }
        StartCol = 0;
      }

      uint32_t Interp = 0; // Undefined
      bool NeedsInterp = (!IsInput && Reflect.Stage == ShaderStage::Vertex) ||
                         (IsInput && Reflect.Stage == ShaderStage::Pixel);
      if (NeedsInterp && !IsNotAllocated) {
        if (Kind == 3)
          Interp = 4; // LinearNoperspective for SV_Position
        else if (isIntComp(SP.Type))
          Interp = 1; // Constant
        else
          Interp = SP.Interpolation ? SP.Interpolation : 2u; // Linear
      }

      SmallVector<Metadata *, 4> Indices;
      if (!SP.RowSemanticIndices.empty()) {
        for (uint32_t Idx : SP.RowSemanticIndices)
          Indices.push_back(I32(Idx));
      } else {
        for (uint32_t R = 0; R < Rows; ++R)
          Indices.push_back(I32(SP.SemanticIndex + R));
      }

      uint32_t CompTy = static_cast<uint32_t>(SP.Type);
      if (CompTy == 0)
        CompTy = static_cast<uint32_t>(ComponentType::F32);

      // 13-operand SemanticSignatureElement metadata node (see
      // DXILSignatureAnalysis.cpp:checkElementNode).
      Nodes.push_back(MDNode::get(
          Ctx, {I32(SP.ID), MDString::get(Ctx, SP.Semantic), I32(CompTy),
                I32(Kind), MDNode::get(Ctx, Indices), I32(Interp), I32(Rows),
                I8(Cols), I32(StartRow), I8(StartCol), I8(0), I8(0), I32(0)}));
    }
    return MDNode::get(Ctx, Nodes);
  };

  MDNode *InMD = buildSigList(Reflect.Inputs, /*IsInput=*/true);
  MDNode *OutMD = buildSigList(Reflect.Outputs, /*IsInput=*/false);
  if (!InMD && !OutMD)
    return;
  NamedMDNode *SigMD = M.getOrInsertNamedMetadata("dx.semantic.signatures");
  SigMD->addOperand(MDNode::get(Ctx, {ValueAsMetadata::get(Entry), InMD, OutMD}));
}

void DXILLowerer::normalizeGroupsharedAtomics() {
  // 1. Promote float / [N x float] groupshared (addrspace(3)) globals touched
  // by atomicrmw / cmpxchg to i32 / [N x i32] so DXIL atomic ops operate on a
  // native i32 groupshared pointer without bitcasting float addrspace(3)*
  // (which causes DXGI_ERROR_DEVICE_REMOVED on WARP).
  auto isFloatOrFloatArray = [](Type *T) {
    while (auto *AT = dyn_cast<ArrayType>(T))
      T = AT->getElementType();
    return T->isFloatTy();
  };
  auto replaceFloatWithI32 = [&](auto &Self, Type *T) -> Type * {
    if (auto *AT = dyn_cast<ArrayType>(T))
      return ArrayType::get(Self(Self, AT->getElementType()),
                            AT->getNumElements());
    if (T->isFloatTy())
      return Type::getInt32Ty(M.getContext());
    return T;
  };
  SmallPtrSet<GlobalVariable *, 8> FloatAtomicGVs;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        Value *Ptr = nullptr;
        if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
          Ptr = RMW->getPointerOperand();
        else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I))
          Ptr = CX->getPointerOperand();
        else
          continue;
        Value *Base = Ptr;
        while (true) {
          if (auto *GEP = dyn_cast<GEPOperator>(Base))
            Base = GEP->getPointerOperand();
          else if (auto *BC = dyn_cast<BitCastOperator>(Base))
            Base = BC->getOperand(0);
          else if (auto *ASC = dyn_cast<AddrSpaceCastOperator>(Base))
            Base = ASC->getOperand(0);
          else
            break;
        }
        if (auto *GV = dyn_cast<GlobalVariable>(Base)) {
          if (GV->getAddressSpace() == 3 &&
              isFloatOrFloatArray(GV->getValueType()))
            FloatAtomicGVs.insert(GV);
        }
      }
    }
  }
  for (GlobalVariable *OldGV : FloatAtomicGVs) {
    Type *NewValTy = replaceFloatWithI32(replaceFloatWithI32, OldGV->getValueType());
    auto *NewGV = new GlobalVariable(
        M, NewValTy, OldGV->isConstant(), OldGV->getLinkage(),
        OldGV->hasInitializer() ? UndefValue::get(NewValTy) : nullptr, "",
        OldGV, OldGV->getThreadLocalMode(), OldGV->getAddressSpace());
    NewGV->takeName(OldGV);
    NewGV->setAlignment(OldGV->getAlign());
    NewGV->setUnnamedAddr(OldGV->getUnnamedAddr());

    // Collect all instructions derived from OldGV (GEPs, bitcasts, loads, stores).
    SmallVector<Instruction *, 16> Worklist;
    SmallPtrSet<Value *, 32> Derived;
    Derived.insert(OldGV);
    for (User *U : OldGV->users())
      if (auto *Inst = dyn_cast<Instruction>(U))
        Worklist.push_back(Inst);
    // BFS to find all GEP/bitcast chains, then rewrite loads/stores/GEPs.
    SmallVector<GetElementPtrInst *, 16> GEPs;
    SmallVector<LoadInst *, 16> Loads;
    SmallVector<StoreInst *, 16> Stores;
    for (size_t Idx = 0; Idx < Worklist.size(); ++Idx) {
      Instruction *Inst = Worklist[Idx];
      if (!Derived.insert(Inst).second)
        continue;
      if (auto *GEP = dyn_cast<GetElementPtrInst>(Inst)) {
        GEPs.push_back(GEP);
        for (User *U : GEP->users())
          if (auto *UI = dyn_cast<Instruction>(U))
            Worklist.push_back(UI);
      } else if (isa<BitCastInst>(Inst) || isa<AddrSpaceCastInst>(Inst)) {
        for (User *U : Inst->users())
          if (auto *UI = dyn_cast<Instruction>(U))
            Worklist.push_back(UI);
      } else if (auto *LI = dyn_cast<LoadInst>(Inst)) {
        if (LI->getType()->isFloatTy())
          Loads.push_back(LI);
      } else if (auto *SI = dyn_cast<StoreInst>(Inst)) {
        if (SI->getValueOperand()->getType()->isFloatTy())
          Stores.push_back(SI);
      }
    }
    OldGV->replaceAllUsesWith(NewGV);
    OldGV->eraseFromParent();
    for (GetElementPtrInst *GEP : GEPs) {
      Type *NewSrcTy =
          replaceFloatWithI32(replaceFloatWithI32, GEP->getSourceElementType());
      SmallVector<Value *, 4> Idxs(GEP->indices());
      auto *NewGEP = GetElementPtrInst::Create(
          NewSrcTy, GEP->getPointerOperand(), Idxs, "", GEP->getIterator());
      NewGEP->setIsInBounds(GEP->isInBounds());
      NewGEP->takeName(GEP);
      GEP->replaceAllUsesWith(NewGEP);
      GEP->eraseFromParent();
    }
    for (LoadInst *LI : Loads) {
      IRBuilder<> B(LI);
      auto *IntLoad = B.CreateAlignedLoad(B.getInt32Ty(), LI->getPointerOperand(),
                                          LI->getAlign(), LI->isVolatile());
      Value *Flt = B.CreateBitCast(IntLoad, B.getFloatTy());
      LI->replaceAllUsesWith(Flt);
      LI->eraseFromParent();
    }
    for (StoreInst *SI : Stores) {
      IRBuilder<> B(SI);
      Value *IntVal = B.CreateBitCast(SI->getValueOperand(), B.getInt32Ty());
      B.CreateAlignedStore(IntVal, SI->getPointerOperand(), SI->getAlign(),
                           SI->isVolatile());
      SI->eraseFromParent();
    }
  }

  // 2. DXIL 3.7 bitcode uses typed pointers: atomicrmw / cmpxchg must operate on a
  // pointer to the scalar element type, not directly on an array global like
  // `[1 x i32] addrspace(3)*`. Insert a zero-index GEP when needed.
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        Value *Ptr = nullptr;
        unsigned PtrOpIdx = 0;
        if (auto *RMW = dyn_cast<AtomicRMWInst>(&I)) {
          Ptr = RMW->getPointerOperand();
          PtrOpIdx = AtomicRMWInst::getPointerOperandIndex();
          RMW->setSyncScopeID(SyncScope::System);
        } else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I)) {
          Ptr = CX->getPointerOperand();
          PtrOpIdx = AtomicCmpXchgInst::getPointerOperandIndex();
          CX->setSyncScopeID(SyncScope::System);
        } else {
          continue;
        }
        auto *GV = dyn_cast<GlobalVariable>(Ptr);
        if (!GV)
          continue;
        Type *ValTy = GV->getValueType();
        if (!isa<ArrayType>(ValTy))
          continue;
        IRBuilder<> B(&I);
        SmallVector<Value *, 4> Idxs = {B.getInt32(0)};
        Type *CurTy = ValTy;
        while (auto *AT = dyn_cast<ArrayType>(CurTy)) {
          Idxs.push_back(B.getInt32(0));
          CurTy = AT->getElementType();
        }
        auto *GEP = GetElementPtrInst::CreateInBounds(ValTy, GV, Idxs, "",
                                                      I.getIterator());
        I.setOperand(PtrOpIdx, GEP);
      }
    }
  }
}

void DXILLowerer::cleanupBridgeArtifacts() {
  SmallVector<CallInst *, 16> HandleCalls;
  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (Function *Callee = CI->getCalledFunction()) {
            if (Callee->getName().starts_with("__crucible_dx."))
              HandleCalls.push_back(CI);
          }
        }
      }
    }
  }
  for (CallInst *CI : HandleCalls) {
    if (!CI->use_empty())
      CI->replaceAllUsesWith(PoisonValue::get(CI->getType()));
    CI->eraseFromParent();
  }

  SmallVector<Function *, 16> BridgeDecls;
  for (Function &F : M) {
    if (F.isDeclaration() && F.getName().starts_with("__crucible_dx."))
      BridgeDecls.push_back(&F);
  }
  for (Function *F : BridgeDecls)
    F->eraseFromParent();
}

Error DXILLowerer::run(StringRef StageName) {
  // First pass: resolve all resource binding calls so handles are in HandleMap
  // before any buffer/texture operations reference them.
  SmallVector<CallInst *, 16> BindCalls;
  SmallVector<CallInst *, 64> OtherCalls;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI)
          continue;
        Function *Callee = CI->getCalledFunction();
        if (!Callee || !Callee->getName().starts_with("__crucible_dx."))
          continue;
        auto Op = static_cast<DxOp>(
            cast<ConstantInt>(CI->getArgOperand(0))->getZExtValue());
        if (Op == DxOp::BindResource || Op == DxOp::HeapResource)
          BindCalls.push_back(CI);
        else
          OtherCalls.push_back(CI);
      }
    }
  }

  for (CallInst *CI : BindCalls) {
    if (Error E = lowerBridgeCall(CI))
      return E;
  }
  for (CallInst *CI : OtherCalls) {
    if (Error E = lowerBridgeCall(CI))
      return E;
  }

  cleanupBridgeArtifacts();
  normalizeGroupsharedAtomics();
  configureEntryAndLinkage(StageName);
  emitSemanticSignatures();
  return Error::success();
}

// Convert any func.func with cf.* terminators (e.g. from standalone MLIR
// inputs) into llvm.func with llvm.br / llvm.cond_br / llvm.return so
// mlir::translateModuleToLLVMIR can export it.
void convertFuncOpsToLLVM(mlir::ModuleOp Mod) {
  mlir::MLIRContext *Ctx = Mod.getContext();
  SmallVector<mlir::func::FuncOp, 4> Funcs(Mod.getOps<mlir::func::FuncOp>().begin(),
                                           Mod.getOps<mlir::func::FuncOp>().end());
  for (mlir::func::FuncOp F : Funcs) {
    mlir::OpBuilder B(F);
    auto FTy = F.getFunctionType();
    mlir::Type RetTy = FTy.getNumResults() == 0
                           ? static_cast<mlir::Type>(mlir::LLVM::LLVMVoidType::get(Ctx))
                           : FTy.getResult(0);
    auto LLVMFTy = mlir::LLVM::LLVMFunctionType::get(RetTy, FTy.getInputs());
    auto LFunc = mlir::LLVM::LLVMFuncOp::create(B, F.getLoc(), F.getName(), LLVMFTy);
    LFunc.getBody().takeBody(F.getBody());
    F.erase();

    SmallVector<mlir::Operation *, 16> Terms;
    LFunc.walk([&](mlir::Operation *Op) {
      if (mlir::isa<mlir::cf::BranchOp, mlir::cf::CondBranchOp,
                    mlir::func::ReturnOp>(Op))
        Terms.push_back(Op);
    });
    for (mlir::Operation *T : Terms) {
      mlir::OpBuilder TB(T);
      mlir::Location Loc = T->getLoc();
      if (auto Br = mlir::dyn_cast<mlir::cf::BranchOp>(T)) {
        mlir::LLVM::BrOp::create(TB, Loc, Br.getDestOperands(), Br.getDest());
        Br.erase();
      } else if (auto CBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(T)) {
        mlir::LLVM::CondBrOp::create(TB, Loc, CBr.getCondition(),
                                     CBr.getTrueDest(), CBr.getTrueDestOperands(),
                                     CBr.getFalseDest(), CBr.getFalseDestOperands());
        CBr.erase();
      } else if (auto Ret = mlir::dyn_cast<mlir::func::ReturnOp>(T)) {
        mlir::LLVM::ReturnOp::create(TB, Loc, Ret.getOperands());
        Ret.erase();
      }
    }
  }
}

Expected<std::unique_ptr<Module>>
lowerMLIRToDXILLLVM(mlir::ModuleOp Mod, LLVMContext &LLVMCtx) {
  // Unwrap any nested builtin.module containers (e.g. produced by
  // ConvertSPIRVToLLVM) so all llvm.func and llvm.global ops live directly
  // inside Mod before translateModuleToLLVMIR.
  SmallVector<mlir::ModuleOp, 2> NestedMods;
  for (auto Inner : Mod.getOps<mlir::ModuleOp>())
    NestedMods.push_back(Inner);
  for (auto Inner : NestedMods) {
    SmallVector<mlir::Operation *, 16> InnerOps;
    for (mlir::Operation &Op : Inner.getBody()->without_terminator())
      InnerOps.push_back(&Op);
    for (mlir::Operation *Op : InnerOps)
      Op->moveBefore(Inner);
    Inner.erase();
  }

  convertFuncOpsToLLVM(Mod);

  mlir::MLIRContext *MCtx = Mod.getContext();
  auto I1Ty = mlir::IntegerType::get(MCtx, 1);
  auto I8Ty = mlir::IntegerType::get(MCtx, 8);
  auto I32Ty = mlir::IntegerType::get(MCtx, 32);
  auto F32Ty = mlir::Float32Type::get(MCtx);
  auto PtrTy = mlir::LLVM::LLVMPointerType::get(MCtx);
  auto VoidTy = mlir::LLVM::LLVMVoidType::get(MCtx);

  unsigned BridgeCounter = 0;
  auto makeBridgeCall = [&](mlir::OpBuilder &B, mlir::Location Loc, DxOp Op,
                            mlir::Type RetTy,
                            ArrayRef<mlir::Value> Args) -> mlir::Value {
    SmallVector<mlir::Value, 8> FullArgs;
    FullArgs.push_back(mlir::LLVM::ConstantOp::create(
        B, Loc, I32Ty, B.getI32IntegerAttr(static_cast<int32_t>(Op))));
    FullArgs.append(Args.begin(), Args.end());

    SmallVector<mlir::Type, 8> ArgTys;
    for (mlir::Value V : FullArgs)
      ArgTys.push_back(V.getType());

    mlir::Type FnRetTy = RetTy ? RetTy : mlir::Type(VoidTy);
    auto FnTy = mlir::LLVM::LLVMFunctionType::get(FnRetTy, ArgTys);
    std::string Name = "__crucible_dx." + std::to_string(BridgeCounter++);
    {
      mlir::OpBuilder ModB(Mod.getBodyRegion());
      mlir::LLVM::LLVMFuncOp::create(ModB, Loc, Name, FnTy);
    }
    SmallVector<mlir::Type, 1> ResTys;
    if (RetTy)
      ResTys.push_back(RetTy);
    auto Call = mlir::LLVM::CallOp::create(
        B, Loc, ResTys, mlir::SymbolRefAttr::get(MCtx, Name), FullArgs);
    return RetTy ? Call.getResult() : mlir::Value();
  };

  auto i32C = [&](mlir::OpBuilder &B, mlir::Location Loc,
                  int32_t V) -> mlir::Value {
    return mlir::LLVM::ConstantOp::create(B, Loc, I32Ty,
                                          B.getI32IntegerAttr(V));
  };
  auto i8C = [&](mlir::OpBuilder &B, mlir::Location Loc,
                 int8_t V) -> mlir::Value {
    return mlir::LLVM::ConstantOp::create(B, Loc, I8Ty, B.getI8IntegerAttr(V));
  };
  auto undefV = [&](mlir::OpBuilder &B, mlir::Location Loc,
                    mlir::Type T) -> mlir::Value {
    return mlir::LLVM::UndefOp::create(B, Loc, T);
  };
  auto dimIndex = [](mlir::gpu::Dimension D) -> int32_t {
    if (D == mlir::gpu::Dimension::y)
      return 1;
    if (D == mlir::gpu::Dimension::z)
      return 2;
    return 0;
  };
  auto castIndexToI32 = [&](mlir::OpBuilder &B, mlir::Location Loc,
                            mlir::Value V, mlir::Type OrigTy) -> mlir::Value {
    if (mlir::isa<mlir::IndexType>(OrigTy))
      return mlir::UnrealizedConversionCastOp::create(B, Loc, OrigTy, V)
          .getResult(0);
    return V;
  };

  SmallVector<mlir::Operation *, 128> Ops;
  Mod.walk([&](mlir::Operation *Op) {
    if (mlir::isa<mlir::ModuleOp, mlir::LLVM::LLVMFuncOp>(Op))
      return;
    StringRef Dialect = Op->getDialect()->getNamespace();
    if (Dialect == "llvm" || Dialect == "builtin")
      return;
    Ops.push_back(Op);
  });

  for (mlir::Operation *Op : Ops) {
    mlir::OpBuilder B(Op);
    mlir::Location Loc = Op->getLoc();

    // --- arith.* -> llvm.* ---
    if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(Op)) {
      mlir::Type Ty = C.getType();
      mlir::Attribute Val = C.getValue();
      if (mlir::isa<mlir::IndexType>(Ty)) {
        int64_t V = mlir::cast<mlir::IntegerAttr>(Val).getInt();
        mlir::Value Rep = i32C(B, Loc, static_cast<int32_t>(V));
        C.getResult().replaceAllUsesWith(castIndexToI32(B, Loc, Rep, Ty));
      } else {
        auto Rep = mlir::LLVM::ConstantOp::create(B, Loc, Ty, Val);
        C.getResult().replaceAllUsesWith(Rep.getResult());
      }
      C.erase();
      continue;
    }
    if (auto IC = mlir::dyn_cast<mlir::arith::IndexCastOp>(Op)) {
      mlir::Value In = IC.getIn();
      if (auto U = In.getDefiningOp<mlir::UnrealizedConversionCastOp>())
        In = U.getInputs()[0];
      mlir::Type DstTy = IC.getType();
      if (mlir::isa<mlir::IndexType>(DstTy)) {
        mlir::Value Res = In;
        if (In.getType() != I32Ty)
          Res = mlir::LLVM::SExtOp::create(B, Loc, I32Ty, In);
        IC.getResult().replaceAllUsesWith(castIndexToI32(B, Loc, Res, DstTy));
      } else if (In.getType() == DstTy) {
        IC.getResult().replaceAllUsesWith(In);
      } else if (In.getType().getIntOrFloatBitWidth() <
                 DstTy.getIntOrFloatBitWidth()) {
        IC.getResult().replaceAllUsesWith(
            mlir::LLVM::SExtOp::create(B, Loc, DstTy, In));
      } else {
        IC.getResult().replaceAllUsesWith(
            mlir::LLVM::TruncOp::create(B, Loc, DstTy, In));
      }
      IC.erase();
      continue;
    }
    if (auto IC = mlir::dyn_cast<mlir::arith::IndexCastUIOp>(Op)) {
      mlir::Value In = IC.getIn();
      if (auto U = In.getDefiningOp<mlir::UnrealizedConversionCastOp>())
        In = U.getInputs()[0];
      mlir::Type DstTy = IC.getType();
      if (mlir::isa<mlir::IndexType>(DstTy)) {
        mlir::Value Res = In;
        if (In.getType() != I32Ty)
          Res = mlir::LLVM::ZExtOp::create(B, Loc, I32Ty, In);
        IC.getResult().replaceAllUsesWith(castIndexToI32(B, Loc, Res, DstTy));
      } else if (In.getType() == DstTy) {
        IC.getResult().replaceAllUsesWith(In);
      } else if (In.getType().getIntOrFloatBitWidth() <
                 DstTy.getIntOrFloatBitWidth()) {
        IC.getResult().replaceAllUsesWith(
            mlir::LLVM::ZExtOp::create(B, Loc, DstTy, In));
      } else {
        IC.getResult().replaceAllUsesWith(
            mlir::LLVM::TruncOp::create(B, Loc, DstTy, In));
      }
      IC.erase();
      continue;
    }
    if (auto Sel = mlir::dyn_cast<mlir::arith::SelectOp>(Op)) {
      auto Rep = mlir::LLVM::SelectOp::create(
          B, Loc, Sel.getType(), Sel.getCondition(), Sel.getTrueValue(),
          Sel.getFalseValue());
      Sel.getResult().replaceAllUsesWith(Rep.getResult());
      Sel.erase();
      continue;
    }

    // --- gpu.* -> bridge calls ---
    if (auto G = mlir::dyn_cast<mlir::gpu::GlobalIdOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::ThreadId, I32Ty,
                                       {i32C(B, Loc, dimIndex(G.getDimension()))});
      G.getResult().replaceAllUsesWith(castIndexToI32(B, Loc, Rep, G.getType()));
      G.erase();
      continue;
    }
    if (auto G = mlir::dyn_cast<mlir::gpu::BlockIdOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::GroupId, I32Ty,
                                       {i32C(B, Loc, dimIndex(G.getDimension()))});
      G.getResult().replaceAllUsesWith(castIndexToI32(B, Loc, Rep, G.getType()));
      G.erase();
      continue;
    }
    if (auto G = mlir::dyn_cast<mlir::gpu::ThreadIdOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::ThreadIdInGroup, I32Ty,
                                       {i32C(B, Loc, dimIndex(G.getDimension()))});
      G.getResult().replaceAllUsesWith(castIndexToI32(B, Loc, Rep, G.getType()));
      G.erase();
      continue;
    }
    if (auto G = mlir::dyn_cast<mlir::gpu::LaneIdOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::LaneId, I32Ty, {});
      G.getResult().replaceAllUsesWith(castIndexToI32(B, Loc, Rep, G.getType()));
      G.erase();
      continue;
    }
    if (auto G = mlir::dyn_cast<mlir::gpu::SubgroupSizeOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::SubgroupSize, I32Ty, {});
      G.getResult().replaceAllUsesWith(castIndexToI32(B, Loc, Rep, G.getType()));
      G.erase();
      continue;
    }
    if (auto G = mlir::dyn_cast<mlir::gpu::BarrierOp>(Op)) {
      makeBridgeCall(B, Loc, DxOp::Barrier, mlir::Type(),
                     {i32C(B, Loc, 1), i32C(B, Loc, 1), i32C(B, Loc, 0)});
      G.erase();
      continue;
    }
    if (auto G = mlir::dyn_cast<mlir::gpu::BallotOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::WaveBallot, G.getType(),
                                       {G.getPredicate()});
      G.getResult().replaceAllUsesWith(Rep);
      G.erase();
      continue;
    }
    if (auto G = mlir::dyn_cast<mlir::gpu::ShuffleOp>(Op)) {
      mlir::Value Offset = G.getOffset();
      mlir::Value TargetLane = Offset;
      if (G.getMode() != mlir::gpu::ShuffleMode::IDX) {
        mlir::Value Lane = makeBridgeCall(B, Loc, DxOp::LaneId, I32Ty, {});
        if (G.getMode() == mlir::gpu::ShuffleMode::XOR)
          TargetLane = mlir::LLVM::XOrOp::create(B, Loc, Lane, Offset);
        else if (G.getMode() == mlir::gpu::ShuffleMode::UP)
          TargetLane = mlir::LLVM::SubOp::create(B, Loc, Lane, Offset);
        else
          TargetLane = mlir::LLVM::AddOp::create(B, Loc, Lane, Offset);
      }
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::WaveReadLaneAt,
                                       G.getShuffleResult().getType(),
                                       {G.getValue(), TargetLane});
      mlir::Value Valid =
          mlir::LLVM::ConstantOp::create(B, Loc, I1Ty, B.getBoolAttr(true));
      G.getShuffleResult().replaceAllUsesWith(Rep);
      G.getValid().replaceAllUsesWith(Valid);
      G.erase();
      continue;
    }
    if (auto G = mlir::dyn_cast<mlir::gpu::SubgroupBroadcastOp>(Op)) {
      mlir::Value Rep;
      if (G.getBroadcastType() == mlir::gpu::BroadcastType::first_active_lane) {
        Rep = makeBridgeCall(B, Loc, DxOp::WaveReadLaneFirst, G.getType(),
                             {G.getSrc()});
      } else {
        Rep = makeBridgeCall(B, Loc, DxOp::WaveReadLaneAt, G.getType(),
                             {G.getSrc(), G.getOperand(1)});
      }
      G.getResult().replaceAllUsesWith(Rep);
      G.erase();
      continue;
    }
    if (auto G = mlir::dyn_cast<mlir::gpu::SubgroupReduceOp>(Op)) {
      StringRef ScanKind;
      if (auto Attr = G->getAttrOfType<mlir::StringAttr>("crucible.scan_kind"))
        ScanKind = Attr.getValue();
      int32_t Kind = -1;
      if (ScanKind == "exclusive" || ScanKind == "inclusive") {
        if (G.getOp() == mlir::gpu::AllReduceOperation::ADD)
          Kind = 9;
        else if (G.getOp() == mlir::gpu::AllReduceOperation::MUL)
          Kind = 10;
      }
      if (Kind < 0) {
        switch (G.getOp()) {
        case mlir::gpu::AllReduceOperation::ADD: Kind = 0; break;
        case mlir::gpu::AllReduceOperation::MUL: Kind = 1; break;
        case mlir::gpu::AllReduceOperation::MINSI:
        case mlir::gpu::AllReduceOperation::MINIMUMF:
        case mlir::gpu::AllReduceOperation::MINNUMF:
        case mlir::gpu::AllReduceOperation::MINIMUMNUMF: Kind = 2; break;
        case mlir::gpu::AllReduceOperation::MINUI: Kind = 3; break;
        case mlir::gpu::AllReduceOperation::MAXSI:
        case mlir::gpu::AllReduceOperation::MAXIMUMF:
        case mlir::gpu::AllReduceOperation::MAXNUMF:
        case mlir::gpu::AllReduceOperation::MAXIMUMNUMF: Kind = 4; break;
        case mlir::gpu::AllReduceOperation::MAXUI: Kind = 5; break;
        case mlir::gpu::AllReduceOperation::AND: Kind = 6; break;
        case mlir::gpu::AllReduceOperation::OR: Kind = 7; break;
        case mlir::gpu::AllReduceOperation::XOR: Kind = 8; break;
        }
      }
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::WaveReduce, G.getType(),
                                       {G.getValue(), i32C(B, Loc, Kind)});
      if (ScanKind == "inclusive" && (Kind == 9 || Kind == 10)) {
        mlir::Type Ty = G.getType();
        mlir::Type ElemTy = Ty;
        if (auto VT = mlir::dyn_cast<mlir::VectorType>(Ty))
          ElemTy = VT.getElementType();
        bool IsFloat = mlir::isa<mlir::FloatType>(ElemTy);
        if (Kind == 9)
          Rep = IsFloat
                    ? mlir::LLVM::FAddOp::create(B, Loc, Rep, G.getValue())
                          .getResult()
                    : mlir::LLVM::AddOp::create(B, Loc, Rep, G.getValue())
                          .getResult();
        else
          Rep = IsFloat
                    ? mlir::LLVM::FMulOp::create(B, Loc, Rep, G.getValue())
                          .getResult()
                    : mlir::LLVM::MulOp::create(B, Loc, Rep, G.getValue())
                          .getResult();
      }
      G.getResult().replaceAllUsesWith(Rep);
      G.erase();
      continue;
    }

    // --- math.* -> llvm.intr.* / bridge ---
    if (auto MOp = mlir::dyn_cast<mlir::math::AbsFOp>(Op)) {
      auto Rep = mlir::LLVM::FAbsOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::AbsIOp>(Op)) {
      auto Rep =
          mlir::LLVM::AbsOp::create(B, Loc, MOp.getType(), MOp.getOperand(), false);
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::SqrtOp>(Op)) {
      auto Rep = mlir::LLVM::SqrtOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::RsqrtOp>(Op)) {
      mlir::Value Rep =
          makeBridgeCall(B, Loc, DxOp::Rsqrt, MOp.getType(), {MOp.getOperand()});
      MOp.getResult().replaceAllUsesWith(Rep);
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::SinOp>(Op)) {
      auto Rep = mlir::LLVM::SinOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::CosOp>(Op)) {
      auto Rep = mlir::LLVM::CosOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::TanOp>(Op)) {
      auto Rep = mlir::LLVM::TanOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::AsinOp>(Op)) {
      auto Rep = mlir::LLVM::ASinOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::AcosOp>(Op)) {
      auto Rep = mlir::LLVM::ACosOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::AtanOp>(Op)) {
      auto Rep = mlir::LLVM::ATanOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::SinhOp>(Op)) {
      auto Rep = mlir::LLVM::SinhOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::CoshOp>(Op)) {
      auto Rep = mlir::LLVM::CoshOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::TanhOp>(Op)) {
      auto Rep = mlir::LLVM::TanhOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::Exp2Op>(Op)) {
      auto Rep = mlir::LLVM::Exp2Op::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::ExpOp>(Op)) {
      auto Rep = mlir::LLVM::ExpOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::Log2Op>(Op)) {
      auto Rep = mlir::LLVM::Log2Op::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::LogOp>(Op)) {
      auto Rep = mlir::LLVM::LogOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::Log10Op>(Op)) {
      auto Rep = mlir::LLVM::Log10Op::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::PowFOp>(Op)) {
      auto Rep = mlir::LLVM::PowOp::create(B, Loc, MOp.getLhs(), MOp.getRhs());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::FloorOp>(Op)) {
      auto Rep = mlir::LLVM::FFloorOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::CeilOp>(Op)) {
      auto Rep = mlir::LLVM::FCeilOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::TruncOp>(Op)) {
      auto Rep = mlir::LLVM::FTruncOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::RoundEvenOp>(Op)) {
      auto Rep = mlir::LLVM::RoundEvenOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::FmaOp>(Op)) {
      auto Rep = mlir::LLVM::FMAOp::create(B, Loc, MOp.getA(), MOp.getB(),
                                           MOp.getC());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::CtPopOp>(Op)) {
      auto Rep = mlir::LLVM::CtPopOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::CountLeadingZerosOp>(Op)) {
      auto Rep = mlir::LLVM::CountLeadingZerosOp::create(
          B, Loc, MOp.getType(), MOp.getOperand(), false);
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::CountTrailingZerosOp>(Op)) {
      auto Rep = mlir::LLVM::CountTrailingZerosOp::create(
          B, Loc, MOp.getType(), MOp.getOperand(), false);
      MOp.getResult().replaceAllUsesWith(Rep.getResult());
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::IsNaNOp>(Op)) {
      mlir::Value Rep =
          makeBridgeCall(B, Loc, DxOp::IsNaN, MOp.getType(), {MOp.getOperand()});
      MOp.getResult().replaceAllUsesWith(Rep);
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::IsInfOp>(Op)) {
      mlir::Value Rep =
          makeBridgeCall(B, Loc, DxOp::IsInf, MOp.getType(), {MOp.getOperand()});
      MOp.getResult().replaceAllUsesWith(Rep);
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::IsFiniteOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::IsFinite, MOp.getType(),
                                       {MOp.getOperand()});
      MOp.getResult().replaceAllUsesWith(Rep);
      MOp.erase();
      continue;
    }

    // --- crucible.* -> bridge calls ---
    if (auto COp = mlir::dyn_cast<ir::BindResourceOp>(Op)) {
      mlir::Value Idx = COp.getIndex() ? COp.getIndex() : i32C(B, Loc, 0);
      mlir::Value Rep = makeBridgeCall(
          B, Loc, DxOp::BindResource, PtrTy,
          {i32C(B, Loc, static_cast<int32_t>(COp.getResourceClass())),
           i32C(B, Loc, COp.getRecordId().value_or(-1)),
           i32C(B, Loc, COp.getSpace().value_or(-1)),
           i32C(B, Loc, COp.getBinding().value_or(-1)),
           i32C(B, Loc, COp.getNonUniform() ? 1 : 0), Idx});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::HeapResourceOp>(Op)) {
      int32_t ClassVal =
          COp.getResourceClass()
              ? static_cast<int32_t>(*COp.getResourceClass())
              : 0;
      mlir::Value Rep = makeBridgeCall(
          B, Loc, DxOp::HeapResource, PtrTy,
          {COp.getIndex(), i32C(B, Loc, COp.getIsSamplerHeap() ? 1 : 0),
           i32C(B, Loc, COp.getNonUniform() ? 1 : 0), i32C(B, Loc, ClassVal),
           i32C(B, Loc, COp.getResourceKind().value_or(0)),
           i32C(B, Loc, COp.getElementStride().value_or(0))});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::BufferLoadOp>(Op)) {
      mlir::Value ElemOff =
          COp.getElementOffset() ? COp.getElementOffset() : undefV(B, Loc, I32Ty);
      mlir::Value Rep = makeBridgeCall(
          B, Loc, DxOp::BufferLoad, COp.getResult().getType(),
          {COp.getHandle(), COp.getIndex(), ElemOff,
           i32C(B, Loc, COp.getMask().value_or(0))});
      COp.getResult().replaceAllUsesWith(Rep);
      if (COp.getStatus())
        COp.getStatus().replaceAllUsesWith(i32C(B, Loc, 0));
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::BufferStoreOp>(Op)) {
      mlir::Value ElemOff =
          COp.getElementOffset() ? COp.getElementOffset() : undefV(B, Loc, I32Ty);
      SmallVector<mlir::Value, 8> Args = {COp.getHandle(), COp.getIndex(),
                                          ElemOff, i32C(B, Loc, COp.getMask())};
      Args.append(COp.getValues().begin(), COp.getValues().end());
      makeBridgeCall(B, Loc, DxOp::BufferStore, mlir::Type(), Args);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::CBufferLoadOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::CBufferLoad,
                                       COp.getResult().getType(),
                                       {COp.getHandle(), COp.getRegIndex()});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::BufferUpdateCounterOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::BufferUpdateCounter, I32Ty,
                                       {COp.getHandle(), i8C(B, Loc, COp.getInc())});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::BufferAtomicOp>(Op)) {
      auto Coords = COp.getCoords();
      mlir::Value C0 = Coords.size() > 0 ? Coords[0] : undefV(B, Loc, I32Ty);
      mlir::Value C1 = Coords.size() > 1 ? Coords[1] : undefV(B, Loc, I32Ty);
      mlir::Value C2 = Coords.size() > 2 ? Coords[2] : undefV(B, Loc, I32Ty);
      mlir::Value Rep = makeBridgeCall(
          B, Loc, DxOp::BufferAtomic, COp.getResult().getType(),
          {COp.getHandle(), i32C(B, Loc, static_cast<int32_t>(COp.getOp())), C0,
           C1, C2, COp.getValue()});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::BufferAtomicCmpXchgOp>(Op)) {
      auto Coords = COp.getCoords();
      mlir::Value C0 = Coords.size() > 0 ? Coords[0] : undefV(B, Loc, I32Ty);
      mlir::Value C1 = Coords.size() > 1 ? Coords[1] : undefV(B, Loc, I32Ty);
      mlir::Value C2 = Coords.size() > 2 ? Coords[2] : undefV(B, Loc, I32Ty);
      mlir::Value Rep = makeBridgeCall(
          B, Loc, DxOp::BufferAtomicCmpXchg, COp.getResult().getType(),
          {COp.getHandle(), C0, C1, C2, COp.getCompare(), COp.getValue()});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::TexLoadOp>(Op)) {
      mlir::Value Mip =
          COp.getMipOrSample() ? COp.getMipOrSample() : i32C(B, Loc, 0);
      SmallVector<mlir::Value, 12> Args = {
          COp.getHandle(), Mip,
          i32C(B, Loc, static_cast<int32_t>(COp.getCoords().size()))};
      Args.append(COp.getCoords().begin(), COp.getCoords().end());
      Args.push_back(
          i32C(B, Loc, static_cast<int32_t>(COp.getOffsets().size())));
      Args.append(COp.getOffsets().begin(), COp.getOffsets().end());
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::TexLoad,
                                       COp.getResult().getType(), Args);
      COp.getResult().replaceAllUsesWith(Rep);
      if (COp.getStatus())
        COp.getStatus().replaceAllUsesWith(i32C(B, Loc, 0));
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::TexStoreOp>(Op)) {
      SmallVector<mlir::Value, 10> Args = {
          COp.getHandle(),
          i32C(B, Loc, static_cast<int32_t>(COp.getCoords().size()))};
      Args.append(COp.getCoords().begin(), COp.getCoords().end());
      Args.append(COp.getValues().begin(), COp.getValues().end());
      makeBridgeCall(B, Loc, DxOp::TexStore, mlir::Type(), Args);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::TexSampleOp>(Op)) {
      mlir::Value Lod =
          COp.getLodOrBias() ? COp.getLodOrBias() : undefV(B, Loc, F32Ty);
      mlir::Value Cmp =
          COp.getCmpVal() ? COp.getCmpVal() : undefV(B, Loc, F32Ty);
      SmallVector<mlir::Value, 14> Args = {
          COp.getTex(), COp.getSampler(),
          i32C(B, Loc, static_cast<int32_t>(COp.getMode())),
          i32C(B, Loc, static_cast<int32_t>(COp.getCoords().size())), Lod, Cmp};
      Args.append(COp.getCoords().begin(), COp.getCoords().end());
      if (COp.getMode() == ir::SampleMode::Grad) {
        Args.push_back(i32C(B, Loc, static_cast<int32_t>(COp.getDdx().size())));
        Args.append(COp.getDdx().begin(), COp.getDdx().end());
        Args.append(COp.getDdy().begin(), COp.getDdy().end());
      }
      Args.push_back(
          i32C(B, Loc, static_cast<int32_t>(COp.getOffsets().size())));
      Args.append(COp.getOffsets().begin(), COp.getOffsets().end());
      Args.push_back(i32C(B, Loc, COp.getClamp() ? 1 : 0));
      Args.push_back(COp.getClamp() ? COp.getClamp() : undefV(B, Loc, F32Ty));
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::TexSample,
                                       COp.getResult().getType(), Args);
      COp.getResult().replaceAllUsesWith(Rep);
      if (COp.getStatus())
        COp.getStatus().replaceAllUsesWith(i32C(B, Loc, 0));
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::TexGatherOp>(Op)) {
      SmallVector<mlir::Value, 8> Args = {
          COp.getTex(), COp.getSampler(), i32C(B, Loc, COp.getChannel()),
          i32C(B, Loc, static_cast<int32_t>(COp.getCoords().size()))};
      Args.append(COp.getCoords().begin(), COp.getCoords().end());
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::TexGather,
                                       COp.getResult().getType(), Args);
      COp.getResult().replaceAllUsesWith(Rep);
      if (COp.getStatus())
        COp.getStatus().replaceAllUsesWith(i32C(B, Loc, 0));
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::TexCalcLodOp>(Op)) {
      SmallVector<mlir::Value, 8> Args = {
          COp.getTex(), COp.getSampler(),
          i32C(B, Loc, COp.getClamped() ? 1 : 0),
          i32C(B, Loc, static_cast<int32_t>(COp.getCoords().size()))};
      Args.append(COp.getCoords().begin(), COp.getCoords().end());
      mlir::Value Rep =
          makeBridgeCall(B, Loc, DxOp::TexCalcLod, F32Ty, Args);
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::TexGetDimensionsOp>(Op)) {
      mlir::Value Mip =
          COp.getMipLevel() ? COp.getMipLevel() : i32C(B, Loc, 0);
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::TexGetDimensions,
                                       COp.getResult().getType(),
                                       {COp.getHandle(), Mip});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::BarrierOp>(Op)) {
      makeBridgeCall(B, Loc, DxOp::Barrier, mlir::Type(),
                     {i32C(B, Loc, COp.getExecSync() ? 1 : 0),
                      i32C(B, Loc, COp.getTgsmFence() ? 1 : 0),
                      i32C(B, Loc, COp.getDeviceFence() ? 1 : 0)});
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::SysValOp>(Op)) {
      if (COp.getKind() == ir::SysValKind::FlattenedThreadIdInGroup) {
        mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::FlattenedThreadIdInGroup,
                                         COp.getResult().getType(), {});
        COp.getResult().replaceAllUsesWith(Rep);
        COp.erase();
        continue;
      }
      return makeError("DirectX backend: unsupported sysval kind {0}",
                       static_cast<unsigned>(COp.getKind()));
    }
    if (auto COp = mlir::dyn_cast<ir::LoadInputOp>(Op)) {
      mlir::Value VertId =
          COp.getVertexId() ? COp.getVertexId() : undefV(B, Loc, I32Ty);
      mlir::Value Rep = makeBridgeCall(
          B, Loc, DxOp::LoadInput, COp.getResult().getType(),
          {i32C(B, Loc, COp.getInputId()), COp.getRow(),
           i32C(B, Loc, static_cast<int32_t>(COp.getCol())), VertId});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::StoreOutputOp>(Op)) {
      makeBridgeCall(B, Loc, DxOp::StoreOutput, mlir::Type(),
                     {i32C(B, Loc, COp.getOutputId()), COp.getRow(),
                      i32C(B, Loc, static_cast<int32_t>(COp.getCol())),
                      COp.getValue()});
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::DiscardOp>(Op)) {
      makeBridgeCall(B, Loc, DxOp::Discard, mlir::Type(), {COp.getCond()});
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::DerivOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, DxOp::Deriv, COp.getResult().getType(),
          {COp.getValue(),
           i32C(B, Loc, static_cast<int32_t>(COp.getAxis())),
           i32C(B, Loc, static_cast<int32_t>(COp.getQuality()))});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::WaveIsFirstLaneOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::WaveIsFirstLane, I1Ty, {});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::WaveVoteOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, DxOp::WaveVote, I1Ty,
          {COp.getValue(), i32C(B, Loc, static_cast<int32_t>(COp.getKind()))});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::WaveBallotOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::WaveBallot,
                                       COp.getResult().getType(),
                                       {COp.getCond()});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::WaveBitCountOp>(Op)) {
      SmallVector<mlir::Value, 6> Args = {
          COp.getCond(), i32C(B, Loc, static_cast<int32_t>(COp.getKind()))};
      Args.append(COp.getMask().begin(), COp.getMask().end());
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::WaveBitCount, I32Ty, Args);
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::WaveMultiPrefixOp>(Op)) {
      SmallVector<mlir::Value, 8> Args = {
          COp.getValue(), i32C(B, Loc, static_cast<int32_t>(COp.getOp())),
          i32C(B, Loc, COp.getIsUnsigned() ? 1 : 0)};
      Args.append(COp.getMask().begin(), COp.getMask().end());
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::WaveMultiPrefix,
                                       COp.getResult().getType(), Args);
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::QuadOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, DxOp::QuadOp, COp.getResult().getType(),
          {COp.getValue(), i32C(B, Loc, static_cast<int32_t>(COp.getKind()))});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::SaturateOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::Saturate,
                                       COp.getResult().getType(),
                                       {COp.getValue()});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::FrcOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::Frc,
                                       COp.getResult().getType(),
                                       {COp.getValue()});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::FirstBitOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, DxOp::FirstBit, I32Ty,
          {COp.getValue(), i32C(B, Loc, static_cast<int32_t>(COp.getKind()))});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::BfeOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, DxOp::Bfe, COp.getResult().getType(),
          {COp.getWidth(), COp.getOffset(), COp.getValue(),
           i32C(B, Loc, COp.getIsSigned() ? 1 : 0)});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::BfiOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, DxOp::Bfi, COp.getResult().getType(),
          {COp.getWidth(), COp.getOffset(), COp.getInsert(), COp.getBase()});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::DotOp>(Op)) {
      SmallVector<mlir::Value, 8> Args;
      Args.append(COp.getLhs().begin(), COp.getLhs().end());
      Args.append(COp.getRhs().begin(), COp.getRhs().end());
      mlir::Value Rep =
          makeBridgeCall(B, Loc, DxOp::Dot, COp.getResult().getType(), Args);
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::DotPackedOp>(Op)) {
      SmallVector<mlir::Value, 8> Args = {
          i32C(B, Loc, static_cast<int32_t>(COp.getKind())), COp.getAcc()};
      Args.append(COp.getInputs().begin(), COp.getInputs().end());
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::DotPacked,
                                       COp.getResult().getType(), Args);
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::LegacyF32ToF16Op>(Op)) {
      mlir::Value Rep =
          makeBridgeCall(B, Loc, DxOp::LegacyF32ToF16, I32Ty, {COp.getValue()});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::LegacyF16ToF32Op>(Op)) {
      mlir::Value Rep =
          makeBridgeCall(B, Loc, DxOp::LegacyF16ToF32, F32Ty, {COp.getValue()});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::MakeDoubleOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(B, Loc, DxOp::MakeDouble,
                                       COp.getResult().getType(),
                                       {COp.getLo(), COp.getHi()});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::SplitDoubleOp>(Op)) {
      auto RetTy = mlir::LLVM::LLVMStructType::getLiteral(MCtx, {I32Ty, I32Ty});
      mlir::Value Pair =
          makeBridgeCall(B, Loc, DxOp::SplitDouble, RetTy, {COp.getValue()});
      COp.getLo().replaceAllUsesWith(
          mlir::LLVM::ExtractValueOp::create(B, Loc, Pair, 0));
      COp.getHi().replaceAllUsesWith(
          mlir::LLVM::ExtractValueOp::create(B, Loc, Pair, 1));
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::UAddcOp>(Op)) {
      mlir::Type SumTy = COp.getSum().getType();
      auto RetTy = mlir::LLVM::LLVMStructType::getLiteral(MCtx, {SumTy, I1Ty});
      mlir::Value Pair = makeBridgeCall(B, Loc, DxOp::UAddc, RetTy,
                                        {COp.getLhs(), COp.getRhs()});
      COp.getSum().replaceAllUsesWith(
          mlir::LLVM::ExtractValueOp::create(B, Loc, Pair, 0));
      COp.getCarry().replaceAllUsesWith(
          mlir::LLVM::ExtractValueOp::create(B, Loc, Pair, 1));
      COp.erase();
      continue;
    }

    return makeError("DirectX backend: unsupported MLIR op '{0}'",
                     Op->getName().getStringRef());
  }

  // Clean up any dead unrealized_conversion_cast ops left from index bridging.
  SmallVector<mlir::UnrealizedConversionCastOp, 16> DeadCasts;
  Mod.walk([&](mlir::UnrealizedConversionCastOp UC) {
    if (UC->use_empty())
      DeadCasts.push_back(UC);
  });
  for (auto UC : DeadCasts)
    UC.erase();

  // Erase unused external declarations left over from frontend import.
  SmallVector<mlir::LLVM::LLVMFuncOp, 16> UnusedDecls;
  for (auto F : Mod.getOps<mlir::LLVM::LLVMFuncOp>()) {
    if (F.isExternal() && !F.getName().starts_with("__crucible_dx.") &&
        mlir::SymbolTable::symbolKnownUseEmpty(F, Mod))
      UnusedDecls.push_back(F);
  }
  for (auto F : UnusedDecls)
    F.erase();

  mlir::DialectRegistry Reg;
  mlir::registerBuiltinDialectTranslation(Reg);
  mlir::registerLLVMDialectTranslation(Reg);
  MCtx->appendDialectRegistry(Reg);

  auto LLVMMod = mlir::translateModuleToLLVMIR(Mod, LLVMCtx);
  if (!LLVMMod)
    return makeError("DirectX backend: failed to export MLIR module to LLVM IR");
  return LLVMMod;
}

} // namespace

Expected<std::string> emitDXIL(mlir::ModuleOp Mod, Reflection &Reflect,
                               const Options &Opts) {
  initializeDirectXTargetOnce();

  auto StageOrErr = stageTripleSuffix(Reflect.Stage);
  if (!StageOrErr)
    return StageOrErr.takeError();

  LLVMContext LLVMCtx;
  auto MOrErr = lowerMLIRToDXILLLVM(Mod, LLVMCtx);
  if (!MOrErr)
    return MOrErr.takeError();
  Module &M = **MOrErr;

  Function *Entry = nullptr;
  if (!Reflect.EntryName.empty())
    Entry = M.getFunction(Reflect.EntryName);
  if (!Entry) {
    for (Function &F : M) {
      if (!F.isDeclaration()) {
        Entry = &F;
        break;
      }
    }
  }
  if (!Entry)
    return makeError("DirectX backend: no entry function found in module");

  std::string SM = Opts.ShaderModel.empty() ? "6.6" : Opts.ShaderModel;
  std::string TripleStr = "dxil-pc-shadermodel" + SM + "-" + *StageOrErr;

  Triple TheTriple(TripleStr);
  std::string LookupErr;
  const Target *T = TargetRegistry::lookupTarget(TheTriple, LookupErr);
  if (!T)
    return makeError("DirectX target lookup failed for '{0}': {1}", TripleStr,
                     LookupErr);

  TargetOptions TOpts;
  std::unique_ptr<TargetMachine> TM(T->createTargetMachine(
      TheTriple, /*CPU=*/"", /*Features=*/"", TOpts, Reloc::Static,
      CodeModel::Small, CodeGenOptLevel::Default));
  if (!TM)
    return makeError("failed to create DirectX TargetMachine for '{0}'",
                     TripleStr);

  M.setTargetTriple(TheTriple);
  M.setDataLayout(TM->createDataLayout());

  // Record the target shader model in reflection metadata.
  if (auto Dot = SM.find('.'); Dot != std::string::npos) {
    uint32_t Maj = 0, Min = 0;
    if (!StringRef(SM).substr(0, Dot).getAsInteger(10, Maj) &&
        !StringRef(SM).substr(Dot + 1).getAsInteger(10, Min)) {
      Reflect.ShaderModelMajor = Maj;
      Reflect.ShaderModelMinor = Min;
    }
  }

  // Set dx.valver = 1.8 to match LLVM DirectX's PSV0 version 3 (52-byte
  // PSVRuntimeInfo3) so dxil.dll validation and Windows D3D12/WARP accept the
  // emitted DXContainer.
  if (!M.getNamedMetadata("dx.valver")) {
    auto *I32Ty = Type::getInt32Ty(LLVMCtx);
    auto *ValVerMD = MDNode::get(
        LLVMCtx, {ConstantAsMetadata::get(ConstantInt::get(I32Ty, 1)),
                  ConstantAsMetadata::get(ConstantInt::get(I32Ty, 8))});
    M.getOrInsertNamedMetadata("dx.valver")->addOperand(ValVerMD);
  }

  // Enable native 16-bit types (UseNativeLowPrecision) on SM 6.2+ so f16/i16
  // CBuffer loads, 16-bit StructuredBuffer strides (2/6 bytes), half<->i16
  // bitcasts, and precise half ops validate cleanly under dxil.dll.
  if (Reflect.ShaderModelMajor > 6 ||
      (Reflect.ShaderModelMajor == 6 && Reflect.ShaderModelMinor >= 2)) {
    M.addModuleFlag(Module::ModFlagBehavior::Override, "dx.nativelowprec", 1u);
  }

  DXILLowerer Lowerer(M, Entry, Reflect, Opts);
  if (Error E = Lowerer.run(*StageOrErr))
    return std::move(E);

  // Scalarize scalar<->vector bitcasts (e.g. i32 <-> <2 x half>, double <->
  // <2 x i32>) which LLVM's DXIL scalarizer does not lower and dxil.dll rejects
  // with "Instructions must be of an allowed type".
  {
    SmallVector<BitCastInst *, 16> VecBitCasts;
    for (Function &F : M)
      for (BasicBlock &BB : F)
        for (Instruction &I : BB)
          if (auto *BC = dyn_cast<BitCastInst>(&I))
            if (isa<FixedVectorType>(BC->getSrcTy()) !=
                isa<FixedVectorType>(BC->getDestTy()))
              VecBitCasts.push_back(BC);

    for (BitCastInst *BC : VecBitCasts) {
      IRBuilder<> B(BC);
      Value *Src = BC->getOperand(0);
      Type *SrcTy = BC->getSrcTy();
      Type *DstTy = BC->getDestTy();
      Value *Rep = nullptr;

      if (auto *DstVT = dyn_cast<FixedVectorType>(DstTy)) {
        // Scalar -> Vector
        unsigned N = DstVT->getNumElements();
        Type *ElemTy = DstVT->getElementType();
        unsigned ElemBits = ElemTy->getPrimitiveSizeInBits();
        unsigned SrcBits = SrcTy->getPrimitiveSizeInBits();
        if (SrcTy->isDoubleTy() && N == 2 && ElemTy->isIntegerTy(32)) {
          Function *SplitFn = Intrinsic::getOrInsertDeclaration(
              &M, Intrinsic::dx_splitdouble, {B.getInt32Ty()});
          Value *Pair = B.CreateCall(SplitFn, {Src});
          Value *Vec = PoisonValue::get(DstVT);
          Vec = B.CreateInsertElement(Vec, B.CreateExtractValue(Pair, {0}),
                                      B.getInt32(0));
          Vec = B.CreateInsertElement(Vec, B.CreateExtractValue(Pair, {1}),
                                      B.getInt32(1));
          Rep = Vec;
        } else if (ElemBits > 0 && N * ElemBits == SrcBits) {
          Type *WideIntTy = Type::getIntNTy(LLVMCtx, SrcBits);
          Type *ElemIntTy = Type::getIntNTy(LLVMCtx, ElemBits);
          Value *IntSrc =
              SrcTy->isIntegerTy() ? Src : B.CreateBitCast(Src, WideIntTy);
          Value *Vec = PoisonValue::get(DstVT);
          for (unsigned I = 0; I < N; ++I) {
            Value *Shifted =
                (I == 0) ? IntSrc
                         : B.CreateLShr(IntSrc,
                                        ConstantInt::get(WideIntTy, I * ElemBits));
            Value *Part = B.CreateTrunc(Shifted, ElemIntTy);
            if (ElemTy != ElemIntTy)
              Part = B.CreateBitCast(Part, ElemTy);
            Vec = B.CreateInsertElement(Vec, Part, B.getInt32(I));
          }
          Rep = Vec;
        }
      } else if (auto *SrcVT = dyn_cast<FixedVectorType>(SrcTy)) {
        // Vector -> Scalar
        unsigned N = SrcVT->getNumElements();
        Type *ElemTy = SrcVT->getElementType();
        unsigned ElemBits = ElemTy->getPrimitiveSizeInBits();
        unsigned DstBits = DstTy->getPrimitiveSizeInBits();
        if (DstTy->isDoubleTy() && N == 2 && ElemTy->isIntegerTy(32)) {
          Function *MakeFn = Intrinsic::getOrInsertDeclaration(
              &M, Intrinsic::dx_asdouble, {B.getInt32Ty()});
          Value *Lo = B.CreateExtractElement(Src, B.getInt32(0));
          Value *Hi = B.CreateExtractElement(Src, B.getInt32(1));
          Rep = B.CreateCall(MakeFn, {Lo, Hi});
        } else if (ElemBits > 0 && N * ElemBits == DstBits) {
          Type *WideIntTy = Type::getIntNTy(LLVMCtx, DstBits);
          Type *ElemIntTy = Type::getIntNTy(LLVMCtx, ElemBits);
          Value *Acc = ConstantInt::get(WideIntTy, 0);
          for (unsigned I = 0; I < N; ++I) {
            Value *Part = B.CreateExtractElement(Src, B.getInt32(I));
            if (ElemTy != ElemIntTy)
              Part = B.CreateBitCast(Part, ElemIntTy);
            Value *Ext = B.CreateZExt(Part, WideIntTy);
            if (I > 0)
              Ext = B.CreateShl(Ext, ConstantInt::get(WideIntTy, I * ElemBits));
            Acc = B.CreateOr(Acc, Ext);
          }
          Rep = DstTy->isIntegerTy() ? Acc : B.CreateBitCast(Acc, DstTy);
        }
      }

      if (Rep) {
        BC->replaceAllUsesWith(Rep);
        BC->eraseFromParent();
      }
    }
  }

  // DXIL forbids extractvalue on LLVM array types ([N x T]). Rewrite
  // `load [N x T]` whose users are all `extractvalue` into per-element GEPs
  // and scalar loads (fixes ByteAddressBuffers.test).
  {
    SmallVector<LoadInst *, 8> ArrLoads;
    for (Function &F : M)
      for (BasicBlock &BB : F)
        for (Instruction &I : BB)
          if (auto *LI = dyn_cast<LoadInst>(&I))
            if (isa<ArrayType>(LI->getType()) && !LI->isVolatile() &&
                !LI->isAtomic())
              ArrLoads.push_back(LI);

    for (LoadInst *LI : ArrLoads) {
      bool AllEV = true;
      for (User *U : LI->users()) {
        if (!isa<ExtractValueInst>(U)) {
          AllEV = false;
          break;
        }
      }
      if (!AllEV || LI->use_empty())
        continue;
      auto *ArrTy = cast<ArrayType>(LI->getType());
      Value *Ptr = LI->getPointerOperand();
      SmallVector<ExtractValueInst *, 8> EVs;
      for (User *U : LI->users())
        EVs.push_back(cast<ExtractValueInst>(U));
      for (ExtractValueInst *EV : EVs) {
        IRBuilder<> B(LI);
        SmallVector<Value *, 4> Idxs = {B.getInt32(0)};
        for (unsigned Idx : EV->indices())
          Idxs.push_back(B.getInt32(Idx));
        Value *ElemPtr = B.CreateInBoundsGEP(ArrTy, Ptr, Idxs);
        Value *ElemLoad = B.CreateLoad(EV->getType(), ElemPtr);
        EV->replaceAllUsesWith(ElemLoad);
        EV->eraseFromParent();
      }
      LI->eraseFromParent();
    }
  }

  // DXIL validation forbids !dx.precise metadata on minprecision (half/f16)
  // instructions. Mark half FP ops fast when strict FP is not requested so
  // LLVM's DirectX backend does not attach !dx.precise to them.
  if (!Reflect.RequiresStrictFP) {
    for (Function &F : M) {
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          if (isa<FPMathOperator>(I) &&
              I.getType()->getScalarType()->isHalfTy())
            I.setFast(true);
        }
      }
    }
  }

  std::string VerifyErr;
  raw_string_ostream VerifyOS(VerifyErr);
  if (verifyModule(M, &VerifyOS))
    return makeError("DirectX lowered LLVM IR failed verification:\n{0}",
                     VerifyErr);

  SmallString<0> OutBuf;
  raw_svector_ostream OutStream(OutBuf);
  legacy::PassManager CodeGenPM;
  CodeGenFileType FileType = Opts.EmitDXILAsm ? CodeGenFileType::AssemblyFile
                                              : CodeGenFileType::ObjectFile;
  if (TM->addPassesToEmitFile(CodeGenPM, OutStream, nullptr, FileType))
    return makeError("DirectX TargetMachine cannot emit requested file type");

  CodeGenPM.run(M);
  return std::string(OutBuf.data(), OutBuf.size());
}

#endif // CRUCIBLE_HAS_DIRECTX

} // namespace crucible

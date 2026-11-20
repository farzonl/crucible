//===- SPIRVEmitter.cpp - DXIL -> LLVM SPIR-V backend -----------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Lowers upgraded DXIL LLVM IR (dx.op.* intrinsics, DXIL resource handles,
// compute/wave/math/atomic/barrier ops) into LLVM's Vulkan SPIR-V IR
// (target("spirv.*") types and llvm.spv.* / standard LLVM intrinsics), then
// invokes LLVM's SPIR-V TargetMachine to emit validated Vulkan SPIR-V.
//
//===----------------------------------------------------------------------===//

#include "SPIRVEmitter.h"
#include "IntrinsicMap.h"
#include "crucible/Diagnostic.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"

#include <algorithm>
#include <mutex>
#include <optional>

using namespace llvm;

namespace crucible {
namespace {

void initializeSPIRVTargetOnce() {
  static std::once_flag Once;
  std::call_once(Once, []() {
    LLVMInitializeSPIRVTargetInfo();
    LLVMInitializeSPIRVTarget();
    LLVMInitializeSPIRVTargetMC();
    LLVMInitializeSPIRVAsmPrinter();
  });
}

Expected<std::string> stageTripleSuffix(ShaderStage Stage, bool StageAsCompute) {
  switch (Stage) {
  case ShaderStage::Compute:
    return "compute";
  case ShaderStage::Vertex:
    return "vertex";
  case ShaderStage::Pixel:
    return "pixel";
  case ShaderStage::RayGeneration:
  case ShaderStage::Library:
    if (StageAsCompute)
      return "compute";
    return makeError(
        "SPIR-V backend: native Raytracing pipeline execution model is not "
        "supported by LLVM 23's SPIR-V backend; pass --stage-as-compute to "
        "lower as a Vulkan compute shader");
  default:
    return makeError("shader stage {0} is not supported by the SPIR-V backend",
                     static_cast<unsigned>(Stage));
  }
}

struct ResolvedHandle {
  const ResourceBinding *RB = nullptr;
  Value *SpvHandle = nullptr;
  Type *SpvHandleTy = nullptr;
};

struct StageVar {
  const StageParameter *Param = nullptr;
  GlobalVariable *GV = nullptr;
  Type *VarTy = nullptr;
  SmallVector<SmallVector<uint32_t, 2>, 3> Decorations;
  bool IsBoolBuiltIn = false;
  bool Referenced = false;
};

struct PendingHeapHandle {
  Value *Index = nullptr;
  bool IsSamplerHeap = false;
  bool NonUniform = false;
};

class SPIRVLowerer {
public:
  SPIRVLowerer(Module &M, Function *Entry, Reflection &Reflect,
               const Options &Opts)
      : M(M), Ctx(M.getContext()), Entry(Entry), Reflect(Reflect), Opts(Opts) {}

  Error run();

private:
  Module &M;
  LLVMContext &Ctx;
  Function *Entry;
  Reflection &Reflect;
  const Options &Opts;

  DenseMap<const Value *, ResolvedHandle> HandleMap;
  DenseMap<const Value *, PendingHeapHandle> PendingHeapHandles;
  std::vector<std::unique_ptr<ResourceBinding>> SyntheticBindings;
  DenseMap<const ResourceBinding *, GlobalVariable *> NameGlobals;
  DenseMap<uint32_t, StageVar> InputVars;
  DenseMap<uint32_t, StageVar> OutputVars;
  struct RayQueryState {
    AllocaInst *Flags = nullptr;
    AllocaInst *Origin[3] = {nullptr, nullptr, nullptr};
    AllocaInst *Dir[3] = {nullptr, nullptr, nullptr};
    AllocaInst *TMin = nullptr;
    AllocaInst *CommittedT = nullptr;
    AllocaInst *CommittedStatus = nullptr;
    AllocaInst *CandidateType = nullptr;
    AllocaInst *FrontFace = nullptr;
    AllocaInst *Bary[2] = {nullptr, nullptr};
    AllocaInst *InstanceID = nullptr;
    AllocaInst *InstanceIndex = nullptr;
    AllocaInst *PrimitiveIndex = nullptr;
    AllocaInst *GeometryIndex = nullptr;
    AllocaInst *InstanceContrib = nullptr;
  };
  DenseMap<const Value *, RayQueryState> RayQueries;

  const ResourceBinding *findResourceByID(ResourceClass Class,
                                          uint32_t RecordID) const;
  const ResourceBinding *findResourceByBinding(ResourceClass Class,
                                               uint32_t Space,
                                               uint32_t LowerBound) const;

  GlobalVariable *getOrCreateResourceName(const ResourceBinding &RB);
  Type *getSPIRVResourceType(const ResourceBinding &RB);
  Type *getComponentLLVMType(ComponentType CT);
  Expected<ResolvedHandle> resolveHandle(Value *V);

  Value *getStorageBufferWordPtr(IRBuilder<> &B, const ResolvedHandle &RH,
                                 Value *WordIdx);
  Value *computeBufferByteOffset(IRBuilder<> &B, const ResourceBinding &RB,
                                 Value *Index, Value *ElemOffset);
  Value *loadTypedValueFromStorageBuffer(IRBuilder<> &B,
                                         const ResolvedHandle &RH,
                                         Value *ByteOffset, Type *ElemTy);
  void storeTypedValueToStorageBuffer(IRBuilder<> &B, const ResolvedHandle &RH,
                                      Value *ByteOffset, Value *Val);

  Error lowerDXILCall(CallInst *CI);
  Error lowerCreateHandle(CallInst *CI);
  Error lowerCreateHandleFromBinding(CallInst *CI);
  Error lowerCreateHandleFromHeap(CallInst *CI);
  Error lowerAnnotateHandle(CallInst *CI);
  Error lowerRawBufferLoad(CallInst *CI);
  Error lowerRawBufferStore(CallInst *CI);
  Error lowerBufferLoad(CallInst *CI);
  Error lowerBufferStore(CallInst *CI);
  Error lowerCBufferLoadLegacy(CallInst *CI);
  Error lowerBufferUpdateCounter(CallInst *CI);
  Error lowerTextureLoad(CallInst *CI);
  Error lowerTextureStore(CallInst *CI);
  Error lowerTextureSample(CallInst *CI, DXILOp Op);
  Error lowerTextureGather(CallInst *CI, DXILOp Op);
  Error lowerCalculateLOD(CallInst *CI);
  Error lowerGetDimensions(CallInst *CI);
  Error lowerAtomicBinOp(CallInst *CI);
  Error lowerAtomicCompareExchange(CallInst *CI);
  Error lowerBarrier(CallInst *CI, DXILOp Op);
  Error lowerWaveOp(CallInst *CI, DXILOp Op);
  Error lowerMathOrBitOp(CallInst *CI, DXILOp Op);
  Error lowerPixelOp(CallInst *CI, DXILOp Op);
  Error lowerLoadInput(CallInst *CI);
  Error lowerStoreOutput(CallInst *CI);
  Error lowerRaytracingOp(CallInst *CI, DXILOp Op);

  void prepareStageInterface();
  void emitStageDecorations();
  void fixupExistingAtomics(Function &F);
  void normalizeArrayGlobalAccesses(Function &F);
  void configureEntryAndLinkage(StringRef StageName);
  void cleanupDXILArtifacts();
};

const ResourceBinding *SPIRVLowerer::findResourceByID(ResourceClass Class,
                                                      uint32_t RecordID) const {
  for (const ResourceBinding &RB : Reflect.Resources) {
    if (RB.Class == Class && RB.RecordID == RecordID)
      return &RB;
  }
  return nullptr;
}

const ResourceBinding *
SPIRVLowerer::findResourceByBinding(ResourceClass Class, uint32_t Space,
                                    uint32_t LowerBound) const {
  for (const ResourceBinding &RB : Reflect.Resources) {
    if (RB.Class == Class && RB.Space == Space && RB.LowerBound == LowerBound)
      return &RB;
  }
  return nullptr;
}

GlobalVariable *
SPIRVLowerer::getOrCreateResourceName(const ResourceBinding &RB) {
  auto It = NameGlobals.find(&RB);
  if (It != NameGlobals.end())
    return It->second;

  std::string Name = RB.Name;
  if (Name.empty())
    Name = "res_" + std::to_string(RB.SPIRVSet) + "_" +
           std::to_string(RB.SPIRVBinding);

  Constant *StrC = ConstantDataArray::getString(Ctx, Name, /*AddNull=*/true);
  auto *GV = new GlobalVariable(M, StrC->getType(), /*isConstant=*/true,
                                GlobalValue::PrivateLinkage, StrC,
                                ".spv.name." + Name, /*InsertBefore=*/nullptr,
                                GlobalValue::NotThreadLocal, /*AddressSpace=*/0);
  GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  NameGlobals[&RB] = GV;
  return GV;
}

Type *SPIRVLowerer::getComponentLLVMType(ComponentType CT) {
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

static unsigned getSPIRVImageDim(ResourceKind K) {
  switch (K) {
  case ResourceKind::Texture1D:
  case ResourceKind::Texture1DArray:
    return 0; // Dim 1D
  case ResourceKind::Texture2D:
  case ResourceKind::Texture2DArray:
  case ResourceKind::Texture2DMS:
  case ResourceKind::Texture2DMSArray:
    return 1; // Dim 2D
  case ResourceKind::Texture3D:
    return 2; // Dim 3D
  case ResourceKind::TextureCube:
  case ResourceKind::TextureCubeArray:
    return 3; // Dim Cube
  case ResourceKind::TypedBuffer:
    return 5; // Dim Buffer
  default:
    return 1;
  }
}

static bool isArrayedResource(ResourceKind K) {
  switch (K) {
  case ResourceKind::Texture1DArray:
  case ResourceKind::Texture2DArray:
  case ResourceKind::Texture2DMSArray:
  case ResourceKind::TextureCubeArray:
    return true;
  default:
    return false;
  }
}

static bool isMSResource(ResourceKind K) {
  return K == ResourceKind::Texture2DMS || K == ResourceKind::Texture2DMSArray;
}

Type *SPIRVLowerer::getSPIRVResourceType(const ResourceBinding &RB) {
  bool IsWritable = (RB.Class == ResourceClass::UAV);
  switch (RB.Kind) {
  case ResourceKind::RawBuffer:
  case ResourceKind::StructuredBuffer:
  case ResourceKind::RTAccelerationStructure: {
    // Model raw/structured buffers as a StorageBuffer runtime array of i32
    // words so arbitrary byte offsets and mixed element types lower cleanly.
    Type *ArrTy = ArrayType::get(Type::getInt32Ty(Ctx), 0);
    return TargetExtType::get(Ctx, "spirv.VulkanBuffer", {ArrTy},
                              {12u /* StorageBuffer */,
                               IsWritable ? 1u : 0u});
  }
  case ResourceKind::CBuffer: {
    uint32_t NumVec4 = std::max(1u, (RB.ElementStride + 15u) / 16u);
    Type *Vec4I32 = FixedVectorType::get(Type::getInt32Ty(Ctx), 4);
    Type *ArrTy = ArrayType::get(Vec4I32, NumVec4);
    StructType *CBStruct = StructType::get(Ctx, ArrayRef<Type *>{ArrTy});
    return TargetExtType::get(Ctx, "spirv.VulkanBuffer", {CBStruct},
                              {2u /* Uniform */, 0u /* NonWritable */});
  }
  case ResourceKind::TypedBuffer:
  case ResourceKind::Texture1D:
  case ResourceKind::Texture2D:
  case ResourceKind::Texture2DMS:
  case ResourceKind::Texture3D:
  case ResourceKind::TextureCube:
  case ResourceKind::Texture1DArray:
  case ResourceKind::Texture2DArray:
  case ResourceKind::Texture2DMSArray:
  case ResourceKind::TextureCubeArray: {
    Type *SampledTy = getComponentLLVMType(RB.ElementType);
    if (SampledTy->isHalfTy())
      SampledTy = Type::getFloatTy(Ctx);
    else if (SampledTy->isIntegerTy(16))
      SampledTy = Type::getInt32Ty(Ctx);
    unsigned Dim = getSPIRVImageDim(RB.Kind);
    unsigned Arrayed = isArrayedResource(RB.Kind) ? 1u : 0u;
    unsigned MS = isMSResource(RB.Kind) ? 1u : 0u;
    unsigned Sampled = IsWritable ? 2u : 1u;
    unsigned Format = 0u; // Unknown
    return TargetExtType::get(Ctx, "spirv.Image", {SampledTy},
                              {Dim, 2u /* Depth unknown */, Arrayed, MS,
                               Sampled, Format});
  }
  case ResourceKind::Sampler:
    return TargetExtType::get(Ctx, "spirv.Sampler", {}, {});
  default:
    return nullptr;
  }
}

Expected<ResolvedHandle> SPIRVLowerer::resolveHandle(Value *V) {
  auto It = HandleMap.find(V);
  if (It != HandleMap.end())
    return It->second;
  return makeError("SPIR-V backend: unresolved DXIL resource handle");
}

Value *SPIRVLowerer::getStorageBufferWordPtr(IRBuilder<> &B,
                                             const ResolvedHandle &RH,
                                             Value *WordIdx) {
  Type *PtrTy = PointerType::get(Ctx, 11); // StorageBuffer address space
  Function *GetPtrFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::spv_resource_getpointer,
      {PtrTy, RH.SpvHandleTy, WordIdx->getType()});
  return B.CreateCall(GetPtrFn, {RH.SpvHandle, WordIdx});
}

Value *SPIRVLowerer::computeBufferByteOffset(IRBuilder<> &B,
                                             const ResourceBinding &RB,
                                             Value *Index, Value *ElemOffset) {
  if (RB.Kind == ResourceKind::RawBuffer)
    return Index;
  uint32_t Stride = RB.ElementStride ? RB.ElementStride : 4u;
  Value *Base = (Stride == 1) ? Index : B.CreateMul(Index, B.getInt32(Stride));
  if (ElemOffset && !isa<UndefValue>(ElemOffset) &&
      !isa<PoisonValue>(ElemOffset)) {
    if (auto *C = dyn_cast<ConstantInt>(ElemOffset); !C || !C->isZero())
      return B.CreateAdd(Base, ElemOffset);
  }
  return Base;
}

Value *SPIRVLowerer::loadTypedValueFromStorageBuffer(IRBuilder<> &B,
                                                     const ResolvedHandle &RH,
                                                     Value *ByteOffset,
                                                     Type *ElemTy) {
  Value *WordIdx = B.CreateLShr(ByteOffset, B.getInt32(2));
  unsigned Bits = ElemTy->getPrimitiveSizeInBits();
  if (Bits == 32) {
    Value *Ptr = getStorageBufferWordPtr(B, RH, WordIdx);
    Value *Word = B.CreateAlignedLoad(B.getInt32Ty(), Ptr, Align(4));
    return ElemTy->isIntegerTy(32) ? Word : B.CreateBitCast(Word, ElemTy);
  }
  if (Bits == 64) {
    Value *LoPtr = getStorageBufferWordPtr(B, RH, WordIdx);
    Value *HiPtr =
        getStorageBufferWordPtr(B, RH, B.CreateAdd(WordIdx, B.getInt32(1)));
    Value *Lo = B.CreateAlignedLoad(B.getInt32Ty(), LoPtr, Align(4));
    Value *Hi = B.CreateAlignedLoad(B.getInt32Ty(), HiPtr, Align(4));
    Value *Lo64 = B.CreateZExt(Lo, B.getInt64Ty());
    Value *Hi64 = B.CreateShl(B.CreateZExt(Hi, B.getInt64Ty()), B.getInt64(32));
    Value *Combined = B.CreateOr(Lo64, Hi64);
    return ElemTy->isIntegerTy(64) ? Combined
                                   : B.CreateBitCast(Combined, ElemTy);
  }
  if (Bits == 16) {
    Value *Ptr = getStorageBufferWordPtr(B, RH, WordIdx);
    Value *Word = B.CreateAlignedLoad(B.getInt32Ty(), Ptr, Align(4));
    Value *Shift = B.CreateShl(B.CreateAnd(ByteOffset, B.getInt32(2)),
                               B.getInt32(3));
    Value *Shifted = B.CreateLShr(Word, Shift);
    Value *Short = B.CreateTrunc(Shifted, B.getInt16Ty());
    return ElemTy->isIntegerTy(16) ? Short : B.CreateBitCast(Short, ElemTy);
  }
  if (Bits == 8) {
    Value *Ptr = getStorageBufferWordPtr(B, RH, WordIdx);
    Value *Word = B.CreateAlignedLoad(B.getInt32Ty(), Ptr, Align(4));
    Value *Shift = B.CreateShl(B.CreateAnd(ByteOffset, B.getInt32(3)),
                               B.getInt32(3));
    return B.CreateTrunc(B.CreateLShr(Word, Shift), ElemTy);
  }
  Value *Ptr = getStorageBufferWordPtr(B, RH, WordIdx);
  Value *Word = B.CreateAlignedLoad(B.getInt32Ty(), Ptr, Align(4));
  return B.CreateTrunc(Word, ElemTy);
}

void SPIRVLowerer::storeTypedValueToStorageBuffer(IRBuilder<> &B,
                                                  const ResolvedHandle &RH,
                                                  Value *ByteOffset,
                                                  Value *Val) {
  Value *WordIdx = B.CreateLShr(ByteOffset, B.getInt32(2));
  Type *ElemTy = Val->getType();
  unsigned Bits = ElemTy->getPrimitiveSizeInBits();
  if (Bits == 32) {
    Value *Word =
        ElemTy->isIntegerTy(32) ? Val : B.CreateBitCast(Val, B.getInt32Ty());
    Value *Ptr = getStorageBufferWordPtr(B, RH, WordIdx);
    B.CreateAlignedStore(Word, Ptr, Align(4));
    return;
  }
  if (Bits == 64) {
    Value *Val64 =
        ElemTy->isIntegerTy(64) ? Val : B.CreateBitCast(Val, B.getInt64Ty());
    Value *Lo = B.CreateTrunc(Val64, B.getInt32Ty());
    Value *Hi = B.CreateTrunc(B.CreateLShr(Val64, B.getInt64(32)),
                              B.getInt32Ty());
    Value *LoPtr = getStorageBufferWordPtr(B, RH, WordIdx);
    Value *HiPtr =
        getStorageBufferWordPtr(B, RH, B.CreateAdd(WordIdx, B.getInt32(1)));
    B.CreateAlignedStore(Lo, LoPtr, Align(4));
    B.CreateAlignedStore(Hi, HiPtr, Align(4));
    return;
  }
  if (Bits == 16) {
    Value *Val16 =
        ElemTy->isIntegerTy(16) ? Val : B.CreateBitCast(Val, B.getInt16Ty());
    Value *Ptr = getStorageBufferWordPtr(B, RH, WordIdx);
    Value *Shift = B.CreateShl(B.CreateAnd(ByteOffset, B.getInt32(2)),
                               B.getInt32(3));
    Value *ClearMask = B.CreateNot(B.CreateShl(B.getInt32(0xFFFFu), Shift));
    Value *SetBits = B.CreateShl(B.CreateZExt(Val16, B.getInt32Ty()), Shift);
    SyncScope::ID DeviceSS = Ctx.getOrInsertSyncScopeID("device");
    B.CreateAtomicRMW(AtomicRMWInst::And, Ptr, ClearMask, Align(4),
                      AtomicOrdering::AcquireRelease, DeviceSS);
    B.CreateAtomicRMW(AtomicRMWInst::Or, Ptr, SetBits, Align(4),
                      AtomicOrdering::AcquireRelease, DeviceSS);
    return;
  }
  if (Bits == 8) {
    Value *Ptr = getStorageBufferWordPtr(B, RH, WordIdx);
    Value *Shift = B.CreateShl(B.CreateAnd(ByteOffset, B.getInt32(3)),
                               B.getInt32(3));
    Value *ClearMask = B.CreateNot(B.CreateShl(B.getInt32(0xFFu), Shift));
    Value *SetBits = B.CreateShl(B.CreateZExt(Val, B.getInt32Ty()), Shift);
    SyncScope::ID DeviceSS = Ctx.getOrInsertSyncScopeID("device");
    B.CreateAtomicRMW(AtomicRMWInst::And, Ptr, ClearMask, Align(4),
                      AtomicOrdering::AcquireRelease, DeviceSS);
    B.CreateAtomicRMW(AtomicRMWInst::Or, Ptr, SetBits, Align(4),
                      AtomicOrdering::AcquireRelease, DeviceSS);
    return;
  }
  Value *Word = B.CreateZExt(Val, B.getInt32Ty());
  Value *Ptr = getStorageBufferWordPtr(B, RH, WordIdx);
  B.CreateAlignedStore(Word, Ptr, Align(4));
}

Error SPIRVLowerer::lowerCreateHandle(CallInst *CI) {
  auto *ClassC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  auto *IdC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
  if (!ClassC || !IdC)
    return makeError("SPIR-V backend: non-constant dx.op.createHandle class/id");

  auto RC = static_cast<ResourceClass>(ClassC->getZExtValue());
  uint32_t RecID = static_cast<uint32_t>(IdC->getZExtValue());
  const ResourceBinding *RB = findResourceByID(RC, RecID);
  if (!RB)
    return makeError("SPIR-V backend: unknown resource class={0} id={1}",
                     static_cast<unsigned>(RC), RecID);

  Type *ResTy = getSPIRVResourceType(*RB);
  if (!ResTy)
    return makeError("SPIR-V backend: unsupported resource kind {0} for '{1}'",
                     static_cast<unsigned>(RB->Kind), RB->Name);

  IRBuilder<> B(CI);
  Value *RegIdx = CI->getArgOperand(3);
  Value *ArrIdx = (RB->LowerBound == 0)
                      ? RegIdx
                      : B.CreateSub(RegIdx, B.getInt32(RB->LowerBound));
  if (auto *NUI = dyn_cast<ConstantInt>(CI->getArgOperand(4));
      NUI && !NUI->isZero()) {
    Function *NuiFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_resource_nonuniformindex);
    ArrIdx = B.CreateCall(NuiFn, {ArrIdx});
  }

  GlobalVariable *NameGV = getOrCreateResourceName(*RB);
  Function *HandleFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::spv_resource_handlefrombinding, {ResTy});
  Value *SpvHandle =
      B.CreateCall(HandleFn, {B.getInt32(RB->SPIRVSet),
                              B.getInt32(RB->SPIRVBinding),
                              B.getInt32(RB->RangeSize), ArrIdx, NameGV});
  HandleMap[CI] = ResolvedHandle{RB, SpvHandle, ResTy};
  return Error::success();
}

static bool extractConstantAggregateU32(Value *V, unsigned Idx, uint32_t &Out) {
  if (!V)
    return false;
  if (isa<ConstantAggregateZero>(V)) {
    Out = 0;
    return true;
  }
  if (auto *C = dyn_cast<Constant>(V)) {
    if (Constant *Elem = C->getAggregateElement(Idx)) {
      if (auto *CI = dyn_cast<ConstantInt>(Elem)) {
        Out = static_cast<uint32_t>(CI->getZExtValue());
        return true;
      }
      if (isa<UndefValue>(Elem)) {
        Out = 0;
        return true;
      }
    }
    return false;
  }
  if (auto *IV = dyn_cast<InsertValueInst>(V)) {
    if (IV->getNumIndices() == 1 && IV->getIndices()[0] == Idx) {
      if (auto *CI = dyn_cast<ConstantInt>(IV->getInsertedValueOperand())) {
        Out = static_cast<uint32_t>(CI->getZExtValue());
        return true;
      }
      return false;
    }
    return extractConstantAggregateU32(IV->getAggregateOperand(), Idx, Out);
  }
  return false;
}

Error SPIRVLowerer::lowerCreateHandleFromBinding(CallInst *CI) {
  Value *BindVal = CI->getArgOperand(1);
  uint32_t LB = 0, Space = 0, RawRC = 0;
  if (!extractConstantAggregateU32(BindVal, 0, LB) ||
      !extractConstantAggregateU32(BindVal, 2, Space) ||
      !extractConstantAggregateU32(BindVal, 3, RawRC))
    return makeError(
        "SPIR-V backend: non-constant ResBind in createHandleFromBinding");

  auto RC = static_cast<ResourceClass>(RawRC);

  const ResourceBinding *RB = findResourceByBinding(RC, Space, LB);
  if (!RB)
    return makeError(
        "SPIR-V backend: unknown resource binding class={0} space={1} reg={2}",
        static_cast<unsigned>(RC), Space, LB);

  Type *ResTy = getSPIRVResourceType(*RB);
  if (!ResTy)
    return makeError("SPIR-V backend: unsupported resource kind {0} for '{1}'",
                     static_cast<unsigned>(RB->Kind), RB->Name);

  IRBuilder<> B(CI);
  Value *RegIdx = CI->getArgOperand(2);
  Value *ArrIdx = (RB->LowerBound == 0)
                      ? RegIdx
                      : B.CreateSub(RegIdx, B.getInt32(RB->LowerBound));
  if (auto *NUI = dyn_cast<ConstantInt>(CI->getArgOperand(3));
      NUI && !NUI->isZero()) {
    Function *NuiFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_resource_nonuniformindex);
    ArrIdx = B.CreateCall(NuiFn, {ArrIdx});
  }

  GlobalVariable *NameGV = getOrCreateResourceName(*RB);
  Function *HandleFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::spv_resource_handlefrombinding, {ResTy});
  Value *SpvHandle =
      B.CreateCall(HandleFn, {B.getInt32(RB->SPIRVSet),
                              B.getInt32(RB->SPIRVBinding),
                              B.getInt32(RB->RangeSize), ArrIdx, NameGV});
  HandleMap[CI] = ResolvedHandle{RB, SpvHandle, ResTy};
  return Error::success();
}

Error SPIRVLowerer::lowerCreateHandleFromHeap(CallInst *CI) {
  PendingHeapHandle PH;
  PH.Index = CI->getArgOperand(1);
  if (auto *SC = dyn_cast<ConstantInt>(CI->getArgOperand(2)))
    PH.IsSamplerHeap = !SC->isZero();
  if (auto *NC = dyn_cast<ConstantInt>(CI->getArgOperand(3)))
    PH.NonUniform = !NC->isZero();

  auto nextFreeSet = [&]() -> uint32_t {
    uint32_t MaxSet = 0;
    for (const ResourceBinding &RB : Reflect.Resources)
      MaxSet = std::max(MaxSet, RB.SPIRVSet + 1u);
    if (Reflect.HasResourceHeap &&
        Reflect.ResourceHeapSPIRVBinding != UINT32_MAX)
      MaxSet = std::max(MaxSet, Reflect.ResourceHeapSPIRVSet + 1u);
    if (Reflect.HasSamplerHeap &&
        Reflect.SamplerHeapSPIRVBinding != UINT32_MAX)
      MaxSet = std::max(MaxSet, Reflect.SamplerHeapSPIRVSet + 1u);
    return MaxSet;
  };

  if (PH.IsSamplerHeap) {
    if (Reflect.SamplerHeapSPIRVBinding == UINT32_MAX) {
      Reflect.SamplerHeapSPIRVSet = nextFreeSet();
      Reflect.SamplerHeapSPIRVBinding = 0;
    }
    Reflect.HasSamplerHeap = true;
  } else {
    if (Reflect.ResourceHeapSPIRVBinding == UINT32_MAX) {
      Reflect.ResourceHeapSPIRVSet = nextFreeSet();
      Reflect.ResourceHeapSPIRVBinding = 0;
    }
    Reflect.HasResourceHeap = true;
  }

  PendingHeapHandles[CI] = PH;
  return Error::success();
}

Error SPIRVLowerer::lowerAnnotateHandle(CallInst *CI) {
  Value *Base = CI->getArgOperand(1);
  auto PendingIt = PendingHeapHandles.find(Base);
  if (PendingIt != PendingHeapHandles.end()) {
    const PendingHeapHandle &PH = PendingIt->second;
    auto *PropsC = dyn_cast<ConstantStruct>(CI->getArgOperand(2));
    if (!PropsC || PropsC->getNumOperands() < 2)
      return makeError(
          "SPIR-V backend: non-constant ResourceProperties in annotateHandle");

    uint32_t Prop0 = static_cast<uint32_t>(
        cast<ConstantInt>(PropsC->getOperand(0))->getZExtValue());
    uint32_t Prop1 = static_cast<uint32_t>(
        cast<ConstantInt>(PropsC->getOperand(1))->getZExtValue());

    auto RB = std::make_unique<ResourceBinding>();
    RB->Kind = static_cast<ResourceKind>(Prop0 & 0xFFu);
    bool IsUAV = ((Prop0 >> 12) & 1u) != 0;
    if (PH.IsSamplerHeap || RB->Kind == ResourceKind::Sampler) {
      RB->Class = ResourceClass::Sampler;
      RB->Kind = ResourceKind::Sampler;
      RB->SPIRVSet = Reflect.SamplerHeapSPIRVSet;
      RB->SPIRVBinding = (Reflect.SamplerHeapSPIRVBinding != UINT32_MAX)
                             ? Reflect.SamplerHeapSPIRVBinding
                             : 0u;
      RB->Name = "sampler_heap";
    } else if (RB->Kind == ResourceKind::CBuffer) {
      RB->Class = ResourceClass::CBV;
      RB->ElementStride = Prop1;
      RB->SPIRVSet = Reflect.ResourceHeapSPIRVSet;
      RB->SPIRVBinding = (Reflect.ResourceHeapSPIRVBinding != UINT32_MAX)
                             ? Reflect.ResourceHeapSPIRVBinding
                             : 0u;
      RB->Name = "resource_heap_cbv";
    } else {
      RB->Class = IsUAV ? ResourceClass::UAV : ResourceClass::SRV;
      if (RB->Kind == ResourceKind::StructuredBuffer)
        RB->ElementStride = Prop1;
      else
        RB->ElementType = static_cast<ComponentType>(Prop1 & 0xFFu);
      RB->SPIRVSet = Reflect.ResourceHeapSPIRVSet;
      RB->SPIRVBinding = (Reflect.ResourceHeapSPIRVBinding != UINT32_MAX)
                             ? Reflect.ResourceHeapSPIRVBinding
                             : 0u;
      RB->Name = IsUAV ? "resource_heap_uav" : "resource_heap_srv";
    }
    RB->RangeSize = 0xFFFFFFFFu;

    Type *ResTy = getSPIRVResourceType(*RB);
    if (!ResTy)
      return makeError(
          "SPIR-V backend: unsupported dynamic heap resource kind {0}",
          static_cast<unsigned>(RB->Kind));

    IRBuilder<> B(CI);
    Value *Idx = PH.Index;
    if (PH.NonUniform) {
      Function *NuiFn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::spv_resource_nonuniformindex);
      Idx = B.CreateCall(NuiFn, {Idx});
    }
    const ResourceBinding *RBPtr = RB.get();
    SyntheticBindings.push_back(std::move(RB));
    GlobalVariable *NameGV = getOrCreateResourceName(*RBPtr);
    Function *HandleFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_resource_handlefrombinding, {ResTy});
    Value *SpvHandle =
        B.CreateCall(HandleFn, {B.getInt32(RBPtr->SPIRVSet),
                                B.getInt32(RBPtr->SPIRVBinding),
                                B.getInt32(0xFFFFFFFFu), Idx, NameGV});
    HandleMap[CI] = ResolvedHandle{RBPtr, SpvHandle, ResTy};
    return Error::success();
  }

  auto RH = resolveHandle(Base);
  if (!RH)
    return RH.takeError();
  HandleMap[CI] = *RH;
  return Error::success();
}

Error SPIRVLowerer::lowerRawBufferLoad(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  auto *RetStructTy = dyn_cast<StructType>(CI->getType());
  if (!RetStructTy || RetStructTy->getNumElements() < 5)
    return makeError("SPIR-V backend: unexpected RawBufferLoad return type");

  Type *ElemTy = RetStructTy->getElementType(0);
  uint32_t ElemBytes =
      std::max<uint32_t>(1u, static_cast<uint32_t>(ElemTy->getPrimitiveSizeInBits() / 8u));
  uint32_t Mask = 0xF;
  if (auto *MC = dyn_cast<ConstantInt>(CI->getArgOperand(4)))
    Mask = static_cast<uint32_t>(MC->getZExtValue());

  IRBuilder<> B(CI);
  Value *BaseByte = computeBufferByteOffset(B, *RH.RB, CI->getArgOperand(2),
                                            CI->getArgOperand(3));
  Value *Res = PoisonValue::get(RetStructTy);
  for (unsigned C = 0; C < 4; ++C) {
    if (!(Mask & (1u << C)))
      continue;
    Value *Off = (C == 0) ? BaseByte
                          : B.CreateAdd(BaseByte, B.getInt32(C * ElemBytes));
    Value *Loaded = loadTypedValueFromStorageBuffer(B, RH, Off, ElemTy);
    Res = B.CreateInsertValue(Res, Loaded, {C});
  }
  Res = B.CreateInsertValue(Res, B.getInt32(0), {4});
  CI->replaceAllUsesWith(Res);
  return Error::success();
}

Error SPIRVLowerer::lowerRawBufferStore(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  Value *V0 = CI->getArgOperand(4);
  Type *ElemTy = V0->getType();
  uint32_t ElemBytes =
      std::max<uint32_t>(1u, static_cast<uint32_t>(ElemTy->getPrimitiveSizeInBits() / 8u));
  uint32_t Mask = 0xF;
  if (auto *MC = dyn_cast<ConstantInt>(CI->getArgOperand(8)))
    Mask = static_cast<uint32_t>(MC->getZExtValue());

  IRBuilder<> B(CI);
  Value *BaseByte = computeBufferByteOffset(B, *RH.RB, CI->getArgOperand(2),
                                            CI->getArgOperand(3));
  for (unsigned C = 0; C < 4; ++C) {
    if (!(Mask & (1u << C)))
      continue;
    Value *Off = (C == 0) ? BaseByte
                          : B.CreateAdd(BaseByte, B.getInt32(C * ElemBytes));
    storeTypedValueToStorageBuffer(B, RH, Off, CI->getArgOperand(4 + C));
  }
  return Error::success();
}

Error SPIRVLowerer::lowerBufferLoad(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  auto *RetStructTy = dyn_cast<StructType>(CI->getType());
  if (!RetStructTy || RetStructTy->getNumElements() < 5)
    return makeError("SPIR-V backend: unexpected BufferLoad return type");

  Type *ElemTy = RetStructTy->getElementType(0);
  IRBuilder<> B(CI);

  if (RH.RB->Kind == ResourceKind::TypedBuffer) {
    Type *SampledTy = getComponentLLVMType(RH.RB->ElementType);
    if (SampledTy->isHalfTy())
      SampledTy = Type::getFloatTy(Ctx);
    else if (SampledTy->isIntegerTy(16))
      SampledTy = Type::getInt32Ty(Ctx);
    Type *Vec4Ty = FixedVectorType::get(SampledTy, 4);
    Function *LoadFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_resource_load_typedbuffer, {Vec4Ty, RH.SpvHandleTy});
    Value *Vec = B.CreateCall(LoadFn, {RH.SpvHandle, CI->getArgOperand(2)});
    Value *Res = PoisonValue::get(RetStructTy);
    for (unsigned C = 0; C < 4; ++C) {
      Value *E = B.CreateExtractElement(Vec, B.getInt32(C));
      if (E->getType() != ElemTy) {
        if (E->getType()->isFloatTy() && ElemTy->isHalfTy())
          E = B.CreateFPTrunc(E, ElemTy);
        else if (E->getType()->isIntegerTy() && ElemTy->isIntegerTy())
          E = B.CreateTruncOrBitCast(E, ElemTy);
        else
          E = B.CreateBitCast(E, ElemTy);
      }
      Res = B.CreateInsertValue(Res, E, {C});
    }
    Res = B.CreateInsertValue(Res, B.getInt32(0), {4});
    CI->replaceAllUsesWith(Res);
    return Error::success();
  }

  uint32_t UsedMask = 0;
  for (User *U : CI->users()) {
    if (auto *EV = dyn_cast<ExtractValueInst>(U)) {
      if (EV->getNumIndices() == 1 && EV->getIndices()[0] < 4)
        UsedMask |= (1u << EV->getIndices()[0]);
    } else {
      UsedMask = 0xF;
      break;
    }
  }
  if (UsedMask == 0)
    UsedMask = 0x1;

  uint32_t ElemBytes =
      std::max<uint32_t>(1u, static_cast<uint32_t>(ElemTy->getPrimitiveSizeInBits() / 8u));
  Value *BaseByte = computeBufferByteOffset(B, *RH.RB, CI->getArgOperand(2),
                                            CI->getArgOperand(3));
  Value *Res = PoisonValue::get(RetStructTy);
  for (unsigned C = 0; C < 4; ++C) {
    if (!(UsedMask & (1u << C)))
      continue;
    Value *Off = (C == 0) ? BaseByte
                          : B.CreateAdd(BaseByte, B.getInt32(C * ElemBytes));
    Value *Loaded = loadTypedValueFromStorageBuffer(B, RH, Off, ElemTy);
    Res = B.CreateInsertValue(Res, Loaded, {C});
  }
  Res = B.CreateInsertValue(Res, B.getInt32(0), {4});
  CI->replaceAllUsesWith(Res);
  return Error::success();
}

Error SPIRVLowerer::lowerBufferStore(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  IRBuilder<> B(CI);
  if (RH.RB->Kind == ResourceKind::TypedBuffer) {
    Type *SampledTy = getComponentLLVMType(RH.RB->ElementType);
    if (SampledTy->isHalfTy())
      SampledTy = Type::getFloatTy(Ctx);
    else if (SampledTy->isIntegerTy(16))
      SampledTy = Type::getInt32Ty(Ctx);
    Type *Vec4Ty = FixedVectorType::get(SampledTy, 4);
    Value *Vec = PoisonValue::get(Vec4Ty);
    for (unsigned C = 0; C < 4; ++C) {
      Value *V = CI->getArgOperand(4 + C);
      if (isa<UndefValue>(V) || isa<PoisonValue>(V))
        V = Constant::getNullValue(SampledTy);
      else if (V->getType() != SampledTy) {
        if (V->getType()->isHalfTy() && SampledTy->isFloatTy())
          V = B.CreateFPExt(V, SampledTy);
        else if (V->getType()->isIntegerTy() && SampledTy->isIntegerTy())
          V = B.CreateZExtOrTrunc(V, SampledTy);
        else
          V = B.CreateBitCast(V, SampledTy);
      }
      Vec = B.CreateInsertElement(Vec, V, B.getInt32(C));
    }
    Function *StoreFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_resource_store_typedbuffer,
        {RH.SpvHandleTy, Vec4Ty});
    B.CreateCall(StoreFn, {RH.SpvHandle, CI->getArgOperand(2), Vec});
    return Error::success();
  }

  Value *V0 = CI->getArgOperand(4);
  Type *ElemTy = V0->getType();
  uint32_t ElemBytes =
      std::max<uint32_t>(1u, static_cast<uint32_t>(ElemTy->getPrimitiveSizeInBits() / 8u));
  uint32_t Mask = 0xF;
  if (auto *MC = dyn_cast<ConstantInt>(CI->getArgOperand(8)))
    Mask = static_cast<uint32_t>(MC->getZExtValue());

  Value *BaseByte = computeBufferByteOffset(B, *RH.RB, CI->getArgOperand(2),
                                            CI->getArgOperand(3));
  for (unsigned C = 0; C < 4; ++C) {
    if (!(Mask & (1u << C)))
      continue;
    Value *Off = (C == 0) ? BaseByte
                          : B.CreateAdd(BaseByte, B.getInt32(C * ElemBytes));
    storeTypedValueToStorageBuffer(B, RH, Off, CI->getArgOperand(4 + C));
  }
  return Error::success();
}

Error SPIRVLowerer::lowerCBufferLoadLegacy(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  auto *RetStructTy = dyn_cast<StructType>(CI->getType());
  if (!RetStructTy || RetStructTy->getNumElements() == 0)
    return makeError("SPIR-V backend: unexpected CBufferLoadLegacy return type");

  auto *ExtTy = cast<TargetExtType>(RH.SpvHandleTy);
  Type *CBStructTy = ExtTy->getTypeParameter(0);
  Type *Vec4I32 = FixedVectorType::get(Type::getInt32Ty(Ctx), 4);

  IRBuilder<> B(CI);
  Type *UniformPtrTy = PointerType::get(Ctx, 12); // Uniform address space
  Function *GetBaseFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::spv_resource_getbasepointer,
      {UniformPtrTy, RH.SpvHandleTy});
  Value *BasePtr = B.CreateCall(GetBaseFn, {RH.SpvHandle});
  Value *RowIdx = CI->getArgOperand(2);
  Value *RowPtr =
      B.CreateInBoundsGEP(CBStructTy, BasePtr, {B.getInt32(0), B.getInt32(0), RowIdx});
  Value *RawVec = B.CreateAlignedLoad(Vec4I32, RowPtr, Align(16));

  unsigned NumElems = RetStructTy->getNumElements();
  Type *ElemTy = RetStructTy->getElementType(0);
  Type *TargetVecTy = FixedVectorType::get(ElemTy, NumElems);
  Value *CastVec =
      (TargetVecTy == Vec4I32) ? RawVec : B.CreateBitCast(RawVec, TargetVecTy);

  Value *Res = PoisonValue::get(RetStructTy);
  for (unsigned I = 0; I < NumElems; ++I) {
    Value *E = B.CreateExtractElement(CastVec, B.getInt32(I));
    Res = B.CreateInsertValue(Res, E, {I});
  }
  CI->replaceAllUsesWith(Res);
  return Error::success();
}

Error SPIRVLowerer::lowerBufferUpdateCounter(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  IRBuilder<> B(CI);
  Type *CounterArrTy = ArrayType::get(Type::getInt32Ty(Ctx), 0);
  Type *CounterTy = TargetExtType::get(Ctx, "spirv.VulkanBuffer",
                                       {CounterArrTy}, {12u, 1u});
  std::string CntName = (RH.RB->Name.empty() ? "uav" : RH.RB->Name) + ".counter";
  Constant *NameInit = ConstantDataArray::getString(Ctx, CntName);
  auto *NameGV = new GlobalVariable(
      M, NameInit->getType(), /*isConstant=*/true, GlobalValue::PrivateLinkage,
      NameInit, ".spv.name." + CntName, /*InsertBefore=*/nullptr,
      GlobalValue::NotThreadLocal, /*AddressSpace=*/0);
  NameGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  Function *HandleFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::spv_resource_handlefrombinding, {CounterTy});
  Value *CounterH = B.CreateCall(
      HandleFn,
      {B.getInt32(RH.RB->SPIRVSet), B.getInt32(RH.RB->CounterSPIRVBinding),
       B.getInt32(RH.RB->RangeSize), B.getInt32(0), NameGV});
  PointerType *BufPtrTy = PointerType::get(Ctx, 11);
  Function *GetPtrFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::spv_resource_getpointer,
      {BufPtrTy, CounterTy, B.getInt32Ty()});
  Value *Ptr = B.CreateCall(GetPtrFn, {CounterH, B.getInt32(0)});
  Value *Inc32 = B.CreateSExt(CI->getArgOperand(2), B.getInt32Ty());
  SyncScope::ID DeviceSS = Ctx.getOrInsertSyncScopeID("device");
  Value *Old = B.CreateAtomicRMW(AtomicRMWInst::Add, Ptr, Inc32, Align(4),
                                 AtomicOrdering::AcquireRelease, DeviceSS);
  // HLSL IncrementCounter (+1) returns pre-increment (old);
  // DecrementCounter (-1) returns post-decrement (old - 1).
  Value *IsDec = B.CreateICmpSLT(Inc32, B.getInt32(0));
  Value *Result = B.CreateSelect(IsDec, B.CreateAdd(Old, Inc32), Old);
  CI->replaceAllUsesWith(Result);
  return Error::success();
}

static unsigned textureCoordDims(ResourceKind K) {
  switch (K) {
  case ResourceKind::Texture1D:
  case ResourceKind::TypedBuffer:
    return 1;
  case ResourceKind::Texture1DArray:
  case ResourceKind::Texture2D:
  case ResourceKind::Texture2DMS:
    return 2;
  case ResourceKind::Texture2DArray:
  case ResourceKind::Texture2DMSArray:
  case ResourceKind::Texture3D:
  case ResourceKind::TextureCube:
    return 3;
  case ResourceKind::TextureCubeArray:
    return 4;
  default:
    return 2;
  }
}

static unsigned textureOffsetDims(ResourceKind K) {
  switch (K) {
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
  default:
    return 0;
  }
}

static Value *packVector(IRBuilder<> &B, Type *ElemTy,
                         ArrayRef<Value *> Elems) {
  if (Elems.size() == 1)
    return Elems[0];
  Type *VecTy = FixedVectorType::get(ElemTy, Elems.size());
  Value *Vec = PoisonValue::get(VecTy);
  for (unsigned I = 0; I < Elems.size(); ++I) {
    Value *V = Elems[I];
    if (isa<UndefValue>(V) || isa<PoisonValue>(V))
      V = Constant::getNullValue(ElemTy);
    Vec = B.CreateInsertElement(Vec, V, B.getInt32(I));
  }
  return Vec;
}

static Value *packConstOffset(IRBuilder<> &B, unsigned Dims, CallInst *CI,
                              unsigned StartArg) {
  SmallVector<Constant *, 3> Vals;
  for (unsigned I = 0; I < Dims; ++I) {
    Value *Arg = CI->getArgOperand(StartArg + I);
    if (auto *C = dyn_cast<ConstantInt>(Arg))
      Vals.push_back(C);
    else
      Vals.push_back(B.getInt32(0));
  }
  if (Dims == 1)
    return Vals[0];
  return ConstantVector::get(Vals);
}

static Value *packResRet(IRBuilder<> &B, StructType *RetStructTy,
                         Value *Vec4) {
  Type *ElemTy = RetStructTy->getElementType(0);
  Value *Res = PoisonValue::get(RetStructTy);
  for (unsigned C = 0; C < 4; ++C) {
    Value *E = B.CreateExtractElement(Vec4, B.getInt32(C));
    if (E->getType() != ElemTy) {
      if (E->getType()->isFloatTy() && ElemTy->isHalfTy())
        E = B.CreateFPTrunc(E, ElemTy);
      else if (E->getType()->isIntegerTy() && ElemTy->isIntegerTy())
        E = B.CreateTruncOrBitCast(E, ElemTy);
      else
        E = B.CreateBitCast(E, ElemTy);
    }
    Res = B.CreateInsertValue(Res, E, {C});
  }
  Res = B.CreateInsertValue(Res, B.getInt32(0), {4});
  return Res;
}

Error SPIRVLowerer::lowerTextureLoad(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  auto *RetStructTy = dyn_cast<StructType>(CI->getType());
  if (!RetStructTy || RetStructTy->getNumElements() < 5)
    return makeError("SPIR-V backend: unexpected TextureLoad return type");

  IRBuilder<> B(CI);
  unsigned Coords = textureCoordDims(RH.RB->Kind);
  SmallVector<Value *, 4> CoordVals;
  for (unsigned I = 0; I < Coords; ++I)
    CoordVals.push_back(CI->getArgOperand(3 + I));
  Value *Coord = packVector(B, B.getInt32Ty(), CoordVals);

  Type *SampledTy = getComponentLLVMType(RH.RB->ElementType);
  if (SampledTy->isHalfTy())
    SampledTy = Type::getFloatTy(Ctx);
  else if (SampledTy->isIntegerTy(16))
    SampledTy = Type::getInt32Ty(Ctx);
  Type *Vec4Ty = FixedVectorType::get(SampledTy, 4);

  Value *Vec = nullptr;
  if (RH.RB->Class == ResourceClass::UAV) {
    Type *PtrTy = PointerType::get(Ctx, 11);
    Function *GetPtrFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_resource_getpointer,
        {PtrTy, RH.SpvHandleTy, Coord->getType()});
    Value *Ptr = B.CreateCall(GetPtrFn, {RH.SpvHandle, Coord});
    Vec = B.CreateAlignedLoad(Vec4Ty, Ptr, Align(16));
  } else {
    Value *Mip = CI->getArgOperand(2);
    if (isa<UndefValue>(Mip) || isa<PoisonValue>(Mip))
      Mip = B.getInt32(0);
    unsigned OffDims = textureOffsetDims(RH.RB->Kind);
    Value *Offset = packConstOffset(B, OffDims ? OffDims : Coords, CI, 6);
    Function *LoadFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_resource_load_level,
        {Vec4Ty, RH.SpvHandleTy, Coord->getType(), Mip->getType(),
         Offset->getType()});
    Vec = B.CreateCall(LoadFn, {RH.SpvHandle, Coord, Mip, Offset});
  }

  CI->replaceAllUsesWith(packResRet(B, RetStructTy, Vec));
  return Error::success();
}

Error SPIRVLowerer::lowerTextureStore(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  IRBuilder<> B(CI);
  unsigned Coords = textureCoordDims(RH.RB->Kind);
  SmallVector<Value *, 4> CoordVals;
  for (unsigned I = 0; I < Coords; ++I)
    CoordVals.push_back(CI->getArgOperand(2 + I));
  Value *Coord = packVector(B, B.getInt32Ty(), CoordVals);

  Type *SampledTy = getComponentLLVMType(RH.RB->ElementType);
  if (SampledTy->isHalfTy())
    SampledTy = Type::getFloatTy(Ctx);
  else if (SampledTy->isIntegerTy(16))
    SampledTy = Type::getInt32Ty(Ctx);
  Type *Vec4Ty = FixedVectorType::get(SampledTy, 4);

  Value *Vec = PoisonValue::get(Vec4Ty);
  for (unsigned C = 0; C < 4; ++C) {
    Value *V = CI->getArgOperand(5 + C);
    if (isa<UndefValue>(V) || isa<PoisonValue>(V))
      V = Constant::getNullValue(SampledTy);
    else if (V->getType() != SampledTy) {
      if (V->getType()->isHalfTy() && SampledTy->isFloatTy())
        V = B.CreateFPExt(V, SampledTy);
      else if (V->getType()->isIntegerTy() && SampledTy->isIntegerTy())
        V = B.CreateZExtOrTrunc(V, SampledTy);
      else
        V = B.CreateBitCast(V, SampledTy);
    }
    Vec = B.CreateInsertElement(Vec, V, B.getInt32(C));
  }

  Type *PtrTy = PointerType::get(Ctx, 11);
  Function *GetPtrFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::spv_resource_getpointer,
      {PtrTy, RH.SpvHandleTy, Coord->getType()});
  Value *Ptr = B.CreateCall(GetPtrFn, {RH.SpvHandle, Coord});
  B.CreateAlignedStore(Vec, Ptr, Align(16));
  return Error::success();
}

Error SPIRVLowerer::lowerTextureSample(CallInst *CI, DXILOp Op) {
  auto TexRH = resolveHandle(CI->getArgOperand(1));
  if (!TexRH)
    return TexRH.takeError();
  auto SampRH = resolveHandle(CI->getArgOperand(2));
  if (!SampRH)
    return SampRH.takeError();

  auto *RetStructTy = dyn_cast<StructType>(CI->getType());
  if (!RetStructTy || RetStructTy->getNumElements() < 5)
    return makeError("SPIR-V backend: unexpected Sample return type");

  IRBuilder<> B(CI);
  unsigned Coords = textureCoordDims(TexRH->RB->Kind);
  SmallVector<Value *, 4> CoordVals;
  for (unsigned I = 0; I < Coords; ++I)
    CoordVals.push_back(CI->getArgOperand(3 + I));
  Value *Coord = packVector(B, B.getFloatTy(), CoordVals);

  unsigned OffDims = textureOffsetDims(TexRH->RB->Kind);
  Value *Offset = OffDims ? packConstOffset(B, OffDims, CI, 7)
                          : ConstantVector::getSplat(ElementCount::getFixed(2),
                                                     B.getInt32(0));

  Type *SampledTy = getComponentLLVMType(TexRH->RB->ElementType);
  if (SampledTy->isHalfTy())
    SampledTy = Type::getFloatTy(Ctx);
  Type *Vec4Ty = FixedVectorType::get(SampledTy, 4);

  Value *ResVec = nullptr;
  switch (Op) {
  case DXILOp::Sample: {
    Value *Clamp = CI->getArgOperand(10);
    if (!isa<UndefValue>(Clamp) && !isa<PoisonValue>(Clamp)) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::spv_resource_sample_clamp,
          {Vec4Ty, TexRH->SpvHandleTy, SampRH->SpvHandleTy, Coord->getType(),
           Clamp->getType(), Offset->getType()});
      ResVec = B.CreateCall(
          Fn, {TexRH->SpvHandle, SampRH->SpvHandle, Coord, Clamp, Offset});
    } else {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::spv_resource_sample,
          {Vec4Ty, TexRH->SpvHandleTy, SampRH->SpvHandleTy, Coord->getType(),
           Offset->getType()});
      ResVec =
          B.CreateCall(Fn, {TexRH->SpvHandle, SampRH->SpvHandle, Coord, Offset});
    }
    break;
  }
  case DXILOp::SampleLevel: {
    Value *Lod = CI->getArgOperand(10);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_resource_samplelevel,
        {Vec4Ty, TexRH->SpvHandleTy, SampRH->SpvHandleTy, Coord->getType(),
         Offset->getType()});
    ResVec = B.CreateCall(
        Fn, {TexRH->SpvHandle, SampRH->SpvHandle, Coord, Lod, Offset});
    break;
  }
  case DXILOp::SampleBias: {
    Value *Bias = CI->getArgOperand(10);
    Value *Clamp = CI->getArgOperand(11);
    if (!isa<UndefValue>(Clamp) && !isa<PoisonValue>(Clamp)) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::spv_resource_samplebias_clamp,
          {Vec4Ty, TexRH->SpvHandleTy, SampRH->SpvHandleTy, Coord->getType(),
           Offset->getType()});
      ResVec = B.CreateCall(Fn, {TexRH->SpvHandle, SampRH->SpvHandle, Coord,
                                 Bias, Clamp, Offset});
    } else {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::spv_resource_samplebias,
          {Vec4Ty, TexRH->SpvHandleTy, SampRH->SpvHandleTy, Coord->getType(),
           Offset->getType()});
      ResVec = B.CreateCall(
          Fn, {TexRH->SpvHandle, SampRH->SpvHandle, Coord, Bias, Offset});
    }
    break;
  }
  case DXILOp::SampleGrad: {
    unsigned GradDims = (OffDims > 0) ? OffDims : 3u;
    SmallVector<Value *, 3> DdxVals, DdyVals;
    for (unsigned I = 0; I < GradDims; ++I) {
      DdxVals.push_back(CI->getArgOperand(10 + I));
      DdyVals.push_back(CI->getArgOperand(13 + I));
    }
    Value *Ddx = packVector(B, B.getFloatTy(), DdxVals);
    Value *Ddy = packVector(B, B.getFloatTy(), DdyVals);
    Value *Clamp = CI->getArgOperand(16);
    if (!isa<UndefValue>(Clamp) && !isa<PoisonValue>(Clamp)) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::spv_resource_samplegrad_clamp,
          {Vec4Ty, TexRH->SpvHandleTy, SampRH->SpvHandleTy, Coord->getType(),
           Ddx->getType(), Ddy->getType(), Clamp->getType(),
           Offset->getType()});
      ResVec = B.CreateCall(Fn, {TexRH->SpvHandle, SampRH->SpvHandle, Coord,
                                 Ddx, Ddy, Clamp, Offset});
    } else {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::spv_resource_samplegrad,
          {Vec4Ty, TexRH->SpvHandleTy, SampRH->SpvHandleTy, Coord->getType(),
           Ddx->getType(), Ddy->getType(), Offset->getType()});
      ResVec = B.CreateCall(
          Fn, {TexRH->SpvHandle, SampRH->SpvHandle, Coord, Ddx, Ddy, Offset});
    }
    break;
  }
  case DXILOp::SampleCmp:
  case DXILOp::SampleCmpLevelZero: {
    Value *CmpVal = CI->getArgOperand(10);
    Intrinsic::ID ID = (Op == DXILOp::SampleCmpLevelZero)
                           ? Intrinsic::spv_resource_samplecmplevelzero
                           : Intrinsic::spv_resource_samplecmp;
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, ID,
        {B.getFloatTy(), TexRH->SpvHandleTy, SampRH->SpvHandleTy,
         Coord->getType(), Offset->getType()});
    Value *Scalar = B.CreateCall(
        Fn, {TexRH->SpvHandle, SampRH->SpvHandle, Coord, CmpVal, Offset});
    Value *Res = PoisonValue::get(RetStructTy);
    Res = B.CreateInsertValue(Res, Scalar, {0});
    Res = B.CreateInsertValue(Res, B.getInt32(0), {4});
    CI->replaceAllUsesWith(Res);
    return Error::success();
  }
  default:
    return makeError("SPIR-V backend: unsupported sample op {0}",
                     static_cast<unsigned>(Op));
  }

  CI->replaceAllUsesWith(packResRet(B, RetStructTy, ResVec));
  return Error::success();
}

Error SPIRVLowerer::lowerTextureGather(CallInst *CI, DXILOp Op) {
  auto TexRH = resolveHandle(CI->getArgOperand(1));
  if (!TexRH)
    return TexRH.takeError();
  auto SampRH = resolveHandle(CI->getArgOperand(2));
  if (!SampRH)
    return SampRH.takeError();

  auto *RetStructTy = dyn_cast<StructType>(CI->getType());
  if (!RetStructTy || RetStructTy->getNumElements() < 5)
    return makeError("SPIR-V backend: unexpected TextureGather return type");

  IRBuilder<> B(CI);
  unsigned Coords = textureCoordDims(TexRH->RB->Kind);
  SmallVector<Value *, 4> CoordVals;
  for (unsigned I = 0; I < Coords; ++I)
    CoordVals.push_back(CI->getArgOperand(3 + I));
  Value *Coord = packVector(B, B.getFloatTy(), CoordVals);

  Value *Offset = packConstOffset(B, 2, CI, 7);
  Type *SampledTy = getComponentLLVMType(TexRH->RB->ElementType);
  if (SampledTy->isHalfTy())
    SampledTy = Type::getFloatTy(Ctx);
  Type *Vec4Ty = FixedVectorType::get(SampledTy, 4);

  Value *ResVec = nullptr;
  if (Op == DXILOp::TextureGatherCmp) {
    Value *CmpVal = CI->getArgOperand(10);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_resource_gather_cmp,
        {Vec4Ty, TexRH->SpvHandleTy, SampRH->SpvHandleTy, Coord->getType(),
         Offset->getType()});
    ResVec = B.CreateCall(
        Fn, {TexRH->SpvHandle, SampRH->SpvHandle, Coord, CmpVal, Offset});
  } else {
    Value *Channel = CI->getArgOperand(9);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_resource_gather,
        {Vec4Ty, TexRH->SpvHandleTy, SampRH->SpvHandleTy, Coord->getType(),
         Offset->getType()});
    ResVec = B.CreateCall(
        Fn, {TexRH->SpvHandle, SampRH->SpvHandle, Coord, Channel, Offset});
  }

  CI->replaceAllUsesWith(packResRet(B, RetStructTy, ResVec));
  return Error::success();
}

Error SPIRVLowerer::lowerCalculateLOD(CallInst *CI) {
  auto TexRH = resolveHandle(CI->getArgOperand(1));
  if (!TexRH)
    return TexRH.takeError();
  auto SampRH = resolveHandle(CI->getArgOperand(2));
  if (!SampRH)
    return SampRH.takeError();

  IRBuilder<> B(CI);
  unsigned Coords = textureCoordDims(TexRH->RB->Kind);
  SmallVector<Value *, 4> CoordVals;
  for (unsigned I = 0; I < Coords; ++I)
    CoordVals.push_back(CI->getArgOperand(3 + I));
  Value *Coord = packVector(B, B.getFloatTy(), CoordVals);

  bool Clamped = true;
  if (auto *C = dyn_cast<ConstantInt>(CI->getArgOperand(6)))
    Clamped = !C->isZero();

  Intrinsic::ID ID = Clamped ? Intrinsic::spv_resource_calculate_lod
                             : Intrinsic::spv_resource_calculate_lod_unclamped;
  Function *Fn = Intrinsic::getOrInsertDeclaration(
      &M, ID,
      {B.getFloatTy(), TexRH->SpvHandleTy, SampRH->SpvHandleTy,
       Coord->getType()});
  Value *Lod = B.CreateCall(Fn, {TexRH->SpvHandle, SampRH->SpvHandle, Coord});
  CI->replaceAllUsesWith(Lod);
  return Error::success();
}

Error SPIRVLowerer::lowerGetDimensions(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  auto *RetStructTy = dyn_cast<StructType>(CI->getType());
  if (!RetStructTy || RetStructTy->getNumElements() < 4)
    return makeError("SPIR-V backend: unexpected GetDimensions return type");

  IRBuilder<> B(CI);
  Value *Res = PoisonValue::get(RetStructTy);
  bool IsUAV = (RH.RB->Class == ResourceClass::UAV);
  bool IsMS = isMSResource(RH.RB->Kind);
  unsigned Coords = textureCoordDims(RH.RB->Kind);

  if (IsMS) {
    Intrinsic::ID ID = (Coords == 3) ? Intrinsic::spv_resource_getdimensions_ms_xyz
                                     : Intrinsic::spv_resource_getdimensions_ms_xy;
    unsigned RetElems = Coords + 1;
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, ID, {RH.SpvHandleTy});
    Value *Dims = B.CreateCall(Fn, {RH.SpvHandle});
    for (unsigned I = 0; I < RetElems; ++I)
      Res = B.CreateInsertValue(Res, B.CreateExtractElement(Dims, B.getInt32(I)),
                                {I});
    for (unsigned I = RetElems; I < 4; ++I)
      Res = B.CreateInsertValue(Res, B.getInt32(0), {I});
  } else if (IsUAV || RH.RB->Kind == ResourceKind::TypedBuffer) {
    if (Coords == 1) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::spv_resource_getdimensions_x, {RH.SpvHandleTy});
      Value *W = B.CreateCall(Fn, {RH.SpvHandle});
      Res = B.CreateInsertValue(Res, W, {0});
      for (unsigned I = 1; I < 4; ++I)
        Res = B.CreateInsertValue(Res, B.getInt32(0), {I});
    } else {
      Intrinsic::ID ID = (Coords >= 3)
                             ? Intrinsic::spv_resource_getdimensions_xyz
                             : Intrinsic::spv_resource_getdimensions_xy;
      unsigned RetElems = (Coords >= 3) ? 3u : 2u;
      Function *Fn = Intrinsic::getOrInsertDeclaration(&M, ID, {RH.SpvHandleTy});
      Value *Dims = B.CreateCall(Fn, {RH.SpvHandle});
      for (unsigned I = 0; I < RetElems; ++I)
        Res = B.CreateInsertValue(
            Res, B.CreateExtractElement(Dims, B.getInt32(I)), {I});
      for (unsigned I = RetElems; I < 4; ++I)
        Res = B.CreateInsertValue(Res, B.getInt32(0), {I});
    }
  } else {
    Value *Mip = CI->getArgOperand(2);
    if (isa<UndefValue>(Mip) || isa<PoisonValue>(Mip))
      Mip = B.getInt32(0);
    unsigned DimCount = (Coords >= 3) ? 3u : Coords;
    Intrinsic::ID ID = Intrinsic::spv_resource_getdimensions_levels_xy;
    if (DimCount == 1)
      ID = Intrinsic::spv_resource_getdimensions_levels_x;
    else if (DimCount == 3)
      ID = Intrinsic::spv_resource_getdimensions_levels_xyz;
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, ID, {RH.SpvHandleTy});
    Value *Dims = B.CreateCall(Fn, {RH.SpvHandle, Mip});
    for (unsigned I = 0; I < DimCount; ++I)
      Res = B.CreateInsertValue(
          Res, B.CreateExtractElement(Dims, B.getInt32(I)), {I});
    for (unsigned I = DimCount; I < 3; ++I)
      Res = B.CreateInsertValue(Res, B.getInt32(0), {I});
    Res = B.CreateInsertValue(
        Res, B.CreateExtractElement(Dims, B.getInt32(DimCount)), {3});
  }

  CI->replaceAllUsesWith(Res);
  return Error::success();
}

Error SPIRVLowerer::lowerAtomicBinOp(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  auto *OpC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
  if (!OpC)
    return makeError("SPIR-V backend: non-constant atomicBinOp opcode");

  AtomicRMWInst::BinOp BinOp = AtomicRMWInst::Add;
  switch (static_cast<DXILAtomicBinOp>(OpC->getZExtValue())) {
  case DXILAtomicBinOp::Add:
    BinOp = AtomicRMWInst::Add;
    break;
  case DXILAtomicBinOp::And:
    BinOp = AtomicRMWInst::And;
    break;
  case DXILAtomicBinOp::Or:
    BinOp = AtomicRMWInst::Or;
    break;
  case DXILAtomicBinOp::Xor:
    BinOp = AtomicRMWInst::Xor;
    break;
  case DXILAtomicBinOp::IMin:
    BinOp = AtomicRMWInst::Min;
    break;
  case DXILAtomicBinOp::IMax:
    BinOp = AtomicRMWInst::Max;
    break;
  case DXILAtomicBinOp::UMin:
    BinOp = AtomicRMWInst::UMin;
    break;
  case DXILAtomicBinOp::UMax:
    BinOp = AtomicRMWInst::UMax;
    break;
  case DXILAtomicBinOp::Exchange:
    BinOp = AtomicRMWInst::Xchg;
    break;
  default:
    return makeError("SPIR-V backend: unsupported atomicBinOp {0}",
                     OpC->getZExtValue());
  }

  IRBuilder<> B(CI);
  Value *Ptr = nullptr;
  if (RH.RB->Kind == ResourceKind::RawBuffer ||
      RH.RB->Kind == ResourceKind::StructuredBuffer) {
    Value *ByteOff = computeBufferByteOffset(B, *RH.RB, CI->getArgOperand(3),
                                             CI->getArgOperand(4));
    Value *WordIdx = B.CreateLShr(ByteOff, B.getInt32(2));
    Ptr = getStorageBufferWordPtr(B, RH, WordIdx);
  } else {
#if LLVM_VERSION_MAJOR >= 24
    unsigned Coords = textureCoordDims(RH.RB->Kind);
    SmallVector<Value *, 3> CoordVals;
    for (unsigned I = 0; I < Coords; ++I)
      CoordVals.push_back(CI->getArgOperand(3 + I));
    Value *Coord = packVector(B, B.getInt32Ty(), CoordVals);
    Type *ImgPtrTy = PointerType::get(Ctx, 12); // Image storage class (OpImageTexelPointer)
    Function *GetPtrFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_resource_getpointer,
        {ImgPtrTy, RH.SpvHandleTy, Coord->getType()});
    Ptr = B.CreateCall(GetPtrFn, {RH.SpvHandle, Coord});
#else
    return makeError("SPIR-V backend: atomicBinOp on non-storage buffer '{0}' "
                     "requires LLVM 24+",
                     RH.RB->Name);
#endif
  }
  SyncScope::ID DeviceScope = Ctx.getOrInsertSyncScopeID("device");
  Value *Val = CI->getArgOperand(6);
  auto *RMW = B.CreateAtomicRMW(BinOp, Ptr, Val, Align(4),
                                AtomicOrdering::AcquireRelease, DeviceScope);
  CI->replaceAllUsesWith(RMW);
  return Error::success();
}

Error SPIRVLowerer::lowerAtomicCompareExchange(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

#if LLVM_VERSION_MAJOR < 24
  return makeError("SPIR-V backend: atomicCompareExchange requires LLVM 24+ "
                   "(upstream LLVM 23 SPIRVLegalizePointerCast bug on cmpxchg)");
#else
  IRBuilder<> B(CI);
  Value *Ptr = nullptr;
  if (RH.RB->Kind == ResourceKind::RawBuffer ||
      RH.RB->Kind == ResourceKind::StructuredBuffer) {
    Value *ByteOff = computeBufferByteOffset(B, *RH.RB, CI->getArgOperand(2),
                                             CI->getArgOperand(3));
    Value *WordIdx = B.CreateLShr(ByteOff, B.getInt32(2));
    Ptr = getStorageBufferWordPtr(B, RH, WordIdx);
  } else {
    unsigned Coords = textureCoordDims(RH.RB->Kind);
    SmallVector<Value *, 3> CoordVals;
    for (unsigned I = 0; I < Coords; ++I)
      CoordVals.push_back(CI->getArgOperand(2 + I));
    Value *Coord = packVector(B, B.getInt32Ty(), CoordVals);
    Type *ImgPtrTy = PointerType::get(Ctx, 12); // Image storage class (OpImageTexelPointer)
    Function *GetPtrFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_resource_getpointer,
        {ImgPtrTy, RH.SpvHandleTy, Coord->getType()});
    Ptr = B.CreateCall(GetPtrFn, {RH.SpvHandle, Coord});
  }
  SyncScope::ID DeviceScope = Ctx.getOrInsertSyncScopeID("device");
  Value *Cmp = CI->getArgOperand(5);
  Value *Val = CI->getArgOperand(6);
  auto *CX = B.CreateAtomicCmpXchg(
      Ptr, Cmp, Val, Align(4), AtomicOrdering::AcquireRelease,
      AtomicOrdering::Acquire, DeviceScope);
  Value *Old = B.CreateExtractValue(CX, {0});
  CI->replaceAllUsesWith(Old);
  return Error::success();
#endif
}

Error SPIRVLowerer::lowerBarrier(CallInst *CI, DXILOp Op) {
  IRBuilder<> B(CI);
  Intrinsic::ID BarID = Intrinsic::spv_all_memory_barrier_with_group_sync;

  if (Op == DXILOp::Barrier) {
    uint32_t Mode = 0;
    if (auto *MC = dyn_cast<ConstantInt>(CI->getArgOperand(1)))
      Mode = static_cast<uint32_t>(MC->getZExtValue());
    bool Sync = (Mode & 1u) != 0;
    bool Device = (Mode & 2u) != 0 || (Mode & 4u) != 0;
    bool Group = (Mode & 8u) != 0;
    if (Device && Group) {
      BarID = Sync ? Intrinsic::spv_all_memory_barrier_with_group_sync
                   : Intrinsic::spv_all_memory_barrier;
    } else if (Device) {
      BarID = Sync ? Intrinsic::spv_device_memory_barrier_with_group_sync
                   : Intrinsic::spv_device_memory_barrier;
    } else {
      BarID = Sync ? Intrinsic::spv_group_memory_barrier_with_group_sync
                   : Intrinsic::spv_group_memory_barrier;
    }
  } else if (Op == DXILOp::BarrierByMemoryType) {
    uint32_t MemFlags = 0;
    uint32_t SemFlags = 0;
    if (auto *C = dyn_cast<ConstantInt>(CI->getArgOperand(1)))
      MemFlags = static_cast<uint32_t>(C->getZExtValue());
    if (auto *C = dyn_cast<ConstantInt>(CI->getArgOperand(2)))
      SemFlags = static_cast<uint32_t>(C->getZExtValue());
    bool Sync = (SemFlags & 1u) != 0;
    bool Group = (MemFlags & 1u) != 0;
    bool Device = (MemFlags & 2u) != 0;
    if (Device && Group)
      BarID = Sync ? Intrinsic::spv_all_memory_barrier_with_group_sync
                   : Intrinsic::spv_all_memory_barrier;
    else if (Device)
      BarID = Sync ? Intrinsic::spv_device_memory_barrier_with_group_sync
                   : Intrinsic::spv_device_memory_barrier;
    else
      BarID = Sync ? Intrinsic::spv_group_memory_barrier_with_group_sync
                   : Intrinsic::spv_group_memory_barrier;
  }

  Function *BarFn = Intrinsic::getOrInsertDeclaration(&M, BarID);
  B.CreateCall(BarFn);
  return Error::success();
}

Error SPIRVLowerer::lowerWaveOp(CallInst *CI, DXILOp Op) {
  IRBuilder<> B(CI);
  Value *Res = nullptr;

  switch (Op) {
  case DXILOp::WaveIsFirstLane: {
    Function *Fn =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::spv_wave_is_first_lane);
    Res = B.CreateCall(Fn);
    break;
  }
  case DXILOp::WaveGetLaneIndex: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_subgroup_local_invocation_id);
    Res = B.CreateCall(Fn);
    break;
  }
  case DXILOp::WaveGetLaneCount: {
    Function *Fn =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::spv_subgroup_size);
    Res = B.CreateCall(Fn);
    break;
  }
  case DXILOp::WaveAnyTrue: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::spv_wave_any);
    Res = B.CreateCall(Fn, {CI->getArgOperand(1)});
    break;
  }
  case DXILOp::WaveAllTrue: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::spv_wave_all);
    Res = B.CreateCall(Fn, {CI->getArgOperand(1)});
    break;
  }
  case DXILOp::WaveActiveAllEqual: {
    Value *Val = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_wave_all_equal, {Val->getType()});
    Res = B.CreateCall(Fn, {Val});
    break;
  }
  case DXILOp::WaveActiveBallot: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_subgroup_ballot);
    Value *Vec4 = B.CreateCall(Fn, {CI->getArgOperand(1)});
    auto *RetTy = cast<StructType>(CI->getType());
    Res = PoisonValue::get(RetTy);
    for (unsigned I = 0; I < 4; ++I)
      Res = B.CreateInsertValue(Res, B.CreateExtractElement(Vec4, B.getInt32(I)),
                                {I});
    break;
  }
  case DXILOp::WaveReadLaneAt: {
    Value *Val = CI->getArgOperand(1);
    Value *Lane = CI->getArgOperand(2);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_wave_readlane, {Val->getType()});
    Res = B.CreateCall(Fn, {Val, Lane});
    break;
  }
  case DXILOp::WaveReadLaneFirst: {
    Value *Val = CI->getArgOperand(1);
    Function *BallotFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_subgroup_ballot);
    Value *Ballot = B.CreateCall(BallotFn, {B.getTrue()});
    Function *FblFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_firstbitlow, {B.getInt32Ty()});
    Value *W0 = B.CreateExtractElement(Ballot, B.getInt32(0));
    Value *W1 = B.CreateExtractElement(Ballot, B.getInt32(1));
    Value *W2 = B.CreateExtractElement(Ballot, B.getInt32(2));
    Value *W3 = B.CreateExtractElement(Ballot, B.getInt32(3));
    Value *L3 = B.CreateAdd(B.CreateCall(FblFn, {W3}), B.getInt32(96));
    Value *L2 = B.CreateSelect(
        B.CreateICmpNE(W2, B.getInt32(0)),
        B.CreateAdd(B.CreateCall(FblFn, {W2}), B.getInt32(64)), L3);
    Value *L1 = B.CreateSelect(
        B.CreateICmpNE(W1, B.getInt32(0)),
        B.CreateAdd(B.CreateCall(FblFn, {W1}), B.getInt32(32)), L2);
    Value *FirstLane = B.CreateSelect(B.CreateICmpNE(W0, B.getInt32(0)),
                                      B.CreateCall(FblFn, {W0}), L1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_wave_readlane, {Val->getType()});
    Res = B.CreateCall(Fn, {Val, FirstLane});
    break;
  }
  case DXILOp::WaveActiveOp: {
    Value *Val = CI->getArgOperand(1);
    auto *KindC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
    auto *SopC = dyn_cast<ConstantInt>(CI->getArgOperand(3));
    if (!KindC || !SopC)
      return makeError("SPIR-V backend: non-constant waveActiveOp kind");
    bool IsUnsigned = (SopC->getZExtValue() == 1);
    Intrinsic::ID ID = Intrinsic::spv_wave_reduce_sum;
    switch (static_cast<DXILWaveOpKind>(KindC->getZExtValue())) {
    case DXILWaveOpKind::Sum:
      ID = Intrinsic::spv_wave_reduce_sum;
      break;
    case DXILWaveOpKind::Product:
      ID = Intrinsic::spv_wave_product;
      break;
    case DXILWaveOpKind::Min:
      ID = IsUnsigned ? Intrinsic::spv_wave_reduce_umin
                      : Intrinsic::spv_wave_reduce_min;
      break;
    case DXILWaveOpKind::Max:
      ID = IsUnsigned ? Intrinsic::spv_wave_reduce_umax
                      : Intrinsic::spv_wave_reduce_max;
      break;
    }
    Function *Fn =
        Intrinsic::getOrInsertDeclaration(&M, ID, {Val->getType()});
    Res = B.CreateCall(Fn, {Val});
    break;
  }
  case DXILOp::WaveActiveBit: {
    Value *Val = CI->getArgOperand(1);
    auto *KindC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
    if (!KindC)
      return makeError("SPIR-V backend: non-constant waveActiveBit kind");
    Intrinsic::ID ID = Intrinsic::spv_wave_reduce_and;
    switch (static_cast<DXILWaveBitOpKind>(KindC->getZExtValue())) {
    case DXILWaveBitOpKind::And:
      ID = Intrinsic::spv_wave_reduce_and;
      break;
    case DXILWaveBitOpKind::Or:
      ID = Intrinsic::spv_wave_reduce_or;
      break;
    case DXILWaveBitOpKind::Xor:
      ID = Intrinsic::spv_wave_reduce_xor;
      break;
    }
    Function *Fn =
        Intrinsic::getOrInsertDeclaration(&M, ID, {Val->getType()});
    Res = B.CreateCall(Fn, {Val});
    break;
  }
  case DXILOp::WavePrefixOp: {
    Value *Val = CI->getArgOperand(1);
    auto *KindC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
    if (!KindC)
      return makeError("SPIR-V backend: non-constant wavePrefixOp kind");
    Intrinsic::ID ID =
        (static_cast<DXILWaveOpKind>(KindC->getZExtValue()) ==
         DXILWaveOpKind::Product)
            ? Intrinsic::spv_wave_prefix_product
            : Intrinsic::spv_wave_prefix_sum;
    Function *Fn =
        Intrinsic::getOrInsertDeclaration(&M, ID, {Val->getType()});
    Res = B.CreateCall(Fn, {Val});
    break;
  }
  case DXILOp::WaveAllBitCount: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_wave_active_countbits);
    Res = B.CreateCall(Fn, {CI->getArgOperand(1)});
    break;
  }
  case DXILOp::WavePrefixBitCount: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_subgroup_prefix_bit_count);
    Res = B.CreateCall(Fn, {CI->getArgOperand(1)});
    break;
  }
  case DXILOp::WaveMultiPrefixBitCount:
  case DXILOp::WaveMultiPrefixOp: {
    // Compute active-lane-masked prefix words wm[0..3] (active bits < lane).
    Function *LaneFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_subgroup_local_invocation_id);
    Value *Lane = B.CreateCall(LaneFn);
    Function *BallotFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_subgroup_ballot);
    Value *ActiveBallot = B.CreateCall(BallotFn, {B.getTrue()});
    Value *WordMasks[4];
    for (unsigned W = 0; W < 4; ++W) {
      Value *ActiveW = B.CreateExtractElement(ActiveBallot, B.getInt32(W));
      Value *RawMask = B.CreateAnd(CI->getArgOperand(2 + W), ActiveW);
      Value *LoBound = B.getInt32(W * 32);
      Value *HiBound = B.getInt32((W + 1) * 32);
      Value *LocalBit = B.CreateSub(Lane, LoBound);
      Value *Shifted = B.CreateShl(B.getInt32(1), LocalBit);
      Value *PartialMask = B.CreateAnd(RawMask, B.CreateSub(Shifted, B.getInt32(1)));
      Value *Below = B.CreateICmpULE(Lane, LoBound);
      Value *Above = B.CreateICmpUGE(Lane, HiBound);
      WordMasks[W] = B.CreateSelect(
          Below, B.getInt32(0), B.CreateSelect(Above, RawMask, PartialMask));
    }

    DXILWaveMultiPrefixOpKind OpKind = DXILWaveMultiPrefixOpKind::Sum;
    Value *Val = nullptr;
    if (Op == DXILOp::WaveMultiPrefixBitCount) {
      Val = B.CreateZExt(CI->getArgOperand(1), B.getInt32Ty());
    } else {
      auto *KindC = dyn_cast<ConstantInt>(CI->getArgOperand(6));
      if (!KindC || KindC->getZExtValue() > 4)
        return makeError("SPIR-V backend: unsupported waveMultiPrefixOp kind");
      OpKind = static_cast<DXILWaveMultiPrefixOpKind>(KindC->getZExtValue());
      Val = CI->getArgOperand(1);
    }
    Type *ValTy = Val->getType();

    Value *Acc = nullptr;
    switch (OpKind) {
    case DXILWaveMultiPrefixOpKind::Sum:
    case DXILWaveMultiPrefixOpKind::Or:
    case DXILWaveMultiPrefixOpKind::Xor:
      Acc = Constant::getNullValue(ValTy);
      break;
    case DXILWaveMultiPrefixOpKind::And:
      Acc = ConstantInt::getAllOnesValue(ValTy);
      break;
    case DXILWaveMultiPrefixOpKind::Product:
      Acc = ValTy->isFloatingPointTy() ? ConstantFP::get(ValTy, 1.0)
                                       : ConstantInt::get(ValTy, 1);
      break;
    }

    Function *ReduceOrFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_wave_reduce_or, {B.getInt32Ty()});
    Function *FirstBitLowFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_firstbitlow, {B.getInt32Ty()});
    Function *ReadLaneFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_wave_readlane, {ValTy});

    for (unsigned W = 0; W < 4; ++W) {
      Value *UniformMask = B.CreateCall(ReduceOrFn, {WordMasks[W]});
      BasicBlock *CurBB = CI->getParent();
      BasicBlock *ContBB = CurBB->splitBasicBlock(CI, "wmp.cont");
      BasicBlock *LoopBB =
          BasicBlock::Create(Ctx, "wmp.loop", CurBB->getParent(), ContBB);
      CurBB->getTerminator()->eraseFromParent();
      B.SetInsertPoint(CurBB);
      B.CreateCondBr(B.CreateICmpNE(UniformMask, B.getInt32(0)), LoopBB, ContBB);

      B.SetInsertPoint(LoopBB);
      PHINode *RemPhi = B.CreatePHI(B.getInt32Ty(), 2, "wmp.rem");
      PHINode *AccPhi = B.CreatePHI(ValTy, 2, "wmp.acc");
      RemPhi->addIncoming(UniformMask, CurBB);
      AccPhi->addIncoming(Acc, CurBB);

      Value *BitIdx = B.CreateCall(FirstBitLowFn, {RemPhi});
      Value *SrcLane =
          (W == 0) ? BitIdx : B.CreateAdd(BitIdx, B.getInt32(W * 32));
      Value *Other = B.CreateCall(ReadLaneFn, {Val, SrcLane});
      Value *InGroup = B.CreateICmpNE(
          B.CreateAnd(B.CreateLShr(WordMasks[W], BitIdx), B.getInt32(1)),
          B.getInt32(0));

      Value *Combined = nullptr;
      switch (OpKind) {
      case DXILWaveMultiPrefixOpKind::Sum:
        Combined = ValTy->isFloatingPointTy() ? B.CreateFAdd(AccPhi, Other)
                                              : B.CreateAdd(AccPhi, Other);
        break;
      case DXILWaveMultiPrefixOpKind::Product:
        Combined = ValTy->isFloatingPointTy() ? B.CreateFMul(AccPhi, Other)
                                              : B.CreateMul(AccPhi, Other);
        break;
      case DXILWaveMultiPrefixOpKind::And:
        Combined = B.CreateAnd(AccPhi, Other);
        break;
      case DXILWaveMultiPrefixOpKind::Or:
        Combined = B.CreateOr(AccPhi, Other);
        break;
      case DXILWaveMultiPrefixOpKind::Xor:
        Combined = B.CreateXor(AccPhi, Other);
        break;
      }
      Value *NextAcc = B.CreateSelect(InGroup, Combined, AccPhi);
      Value *NextRem = B.CreateAnd(RemPhi, B.CreateSub(RemPhi, B.getInt32(1)));
      RemPhi->addIncoming(NextRem, LoopBB);
      AccPhi->addIncoming(NextAcc, LoopBB);
      B.CreateCondBr(B.CreateICmpNE(NextRem, B.getInt32(0)), LoopBB, ContBB);

      B.SetInsertPoint(CI);
      PHINode *OutAcc = B.CreatePHI(ValTy, 2, "wmp.out");
      OutAcc->addIncoming(Acc, CurBB);
      OutAcc->addIncoming(NextAcc, LoopBB);
      Acc = OutAcc;
    }
    Res = Acc;
    break;
  }
  case DXILOp::QuadOp: {
    Value *Val = CI->getArgOperand(1);
    auto *DirC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
    if (!DirC)
      return makeError("SPIR-V backend: non-constant quadOp direction");
    Intrinsic::ID ID = Intrinsic::spv_quad_read_across_x;
    if (DirC->getZExtValue() == 1)
      ID = Intrinsic::spv_quad_read_across_y;
    else if (DirC->getZExtValue() == 2)
      ID = Intrinsic::spv_quad_read_across_diagonal;
    Function *Fn =
        Intrinsic::getOrInsertDeclaration(&M, ID, {Val->getType()});
    Res = B.CreateCall(Fn, {Val});
    break;
  }
  default:
    return makeError("SPIR-V backend: unsupported wave op {0}",
                     static_cast<unsigned>(Op));
  }

  CI->replaceAllUsesWith(Res);
  return Error::success();
}

Error SPIRVLowerer::lowerMathOrBitOp(CallInst *CI, DXILOp Op) {
  IRBuilder<> B(CI);
  Value *Res = nullptr;

  auto CallUnary = [&](Intrinsic::ID ID) {
    Value *X = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, ID, {X->getType()});
    return B.CreateCall(Fn, {X});
  };
  auto CallBinary = [&](Intrinsic::ID ID) {
    Value *X = CI->getArgOperand(1);
    Value *Y = CI->getArgOperand(2);
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, ID, {X->getType()});
    return B.CreateCall(Fn, {X, Y});
  };

  switch (Op) {
  case DXILOp::FAbs:
    Res = CallUnary(Intrinsic::fabs);
    break;
  case DXILOp::Saturate:
    Res = CallUnary(Intrinsic::spv_saturate);
    break;
  case DXILOp::IsNaN: {
    Value *X = CI->getArgOperand(1);
    Res = B.CreateFCmpUNO(X, X);
    break;
  }
  case DXILOp::IsInf: {
    Value *X = CI->getArgOperand(1);
    Value *Abs = CallUnary(Intrinsic::fabs);
    Res = B.CreateFCmpOEQ(Abs, ConstantFP::getInfinity(X->getType()));
    break;
  }
  case DXILOp::IsFinite: {
    Value *X = CI->getArgOperand(1);
    Value *Abs = CallUnary(Intrinsic::fabs);
    Res = B.CreateFCmpOLT(Abs, ConstantFP::getInfinity(X->getType()));
    break;
  }
  case DXILOp::Cos:
    Res = CallUnary(Intrinsic::cos);
    break;
  case DXILOp::Sin:
    Res = CallUnary(Intrinsic::sin);
    break;
  case DXILOp::Tan:
    Res = CallUnary(Intrinsic::tan);
    break;
  case DXILOp::Acos:
    Res = CallUnary(Intrinsic::acos);
    break;
  case DXILOp::Asin:
    Res = CallUnary(Intrinsic::asin);
    break;
  case DXILOp::Atan:
    Res = CallUnary(Intrinsic::atan);
    break;
  case DXILOp::Hcos:
    Res = CallUnary(Intrinsic::cosh);
    break;
  case DXILOp::Hsin:
    Res = CallUnary(Intrinsic::sinh);
    break;
  case DXILOp::Htan:
    Res = CallUnary(Intrinsic::tanh);
    break;
  case DXILOp::Exp:
    Res = CallUnary(Intrinsic::exp2);
    break;
  case DXILOp::Frc:
    Res = CallUnary(Intrinsic::spv_frac);
    break;
  case DXILOp::Log:
    Res = CallUnary(Intrinsic::log2);
    break;
  case DXILOp::Sqrt:
    Res = CallUnary(Intrinsic::sqrt);
    break;
  case DXILOp::Rsqrt:
    Res = CallUnary(Intrinsic::spv_rsqrt);
    break;
  case DXILOp::RoundNE:
    Res = CallUnary(Intrinsic::roundeven);
    break;
  case DXILOp::RoundNI:
    Res = CallUnary(Intrinsic::floor);
    break;
  case DXILOp::RoundPI:
    Res = CallUnary(Intrinsic::ceil);
    break;
  case DXILOp::RoundZ:
    Res = CallUnary(Intrinsic::trunc);
    break;
  case DXILOp::Bfrev:
    Res = CallUnary(Intrinsic::bitreverse);
    break;
  case DXILOp::Countbits: {
    Value *X = CI->getArgOperand(1);
    unsigned Bits = X->getType()->getIntegerBitWidth();
    Function *Ctpop32 =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::ctpop, {B.getInt32Ty()});
    if (Bits == 32) {
      Res = B.CreateCall(Ctpop32, {X});
    } else if (Bits < 32) {
      Res = B.CreateCall(Ctpop32, {B.CreateZExt(X, B.getInt32Ty())});
    } else if (Bits == 64) {
      Value *Lo = B.CreateTrunc(X, B.getInt32Ty());
      Value *Hi = B.CreateTrunc(B.CreateLShr(X, B.getInt64(32)), B.getInt32Ty());
      Res = B.CreateAdd(B.CreateCall(Ctpop32, {Lo}), B.CreateCall(Ctpop32, {Hi}));
    } else {
      Res = CallUnary(Intrinsic::ctpop);
    }
    break;
  }
  case DXILOp::FirstbitLo: {
    Value *X = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_firstbitlow, {X->getType()});
    Res = B.CreateCall(Fn, {X});
    break;
  }
  case DXILOp::FirstbitHi:
  case DXILOp::FirstbitSHi: {
    Value *X = CI->getArgOperand(1);
    unsigned Bits = X->getType()->getIntegerBitWidth();
    bool IsSigned = (Op == DXILOp::FirstbitSHi);
    if (IsSigned) {
      Value *IsNeg = B.CreateICmpSLT(X, ConstantInt::get(X->getType(), 0));
      X = B.CreateSelect(IsNeg, B.CreateNot(X), X);
    }
    Function *UMsb32 = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_firstbituhigh, {B.getInt32Ty()});
    Value *Neg1 = B.getInt32(-1);
    if (Bits == 64) {
      Value *Lo = B.CreateTrunc(X, B.getInt32Ty());
      Value *Hi = B.CreateTrunc(B.CreateLShr(X, B.getInt64(32)), B.getInt32Ty());
      Value *HiMsb = B.CreateCall(UMsb32, {Hi});
      Value *LoMsb = B.CreateCall(UMsb32, {Lo});
      Value *LoRes = B.CreateSelect(B.CreateICmpEQ(LoMsb, Neg1), Neg1,
                                    B.CreateSub(B.getInt32(63), LoMsb));
      Res = B.CreateSelect(B.CreateICmpEQ(HiMsb, Neg1), LoRes,
                           B.CreateSub(B.getInt32(31), HiMsb));
    } else {
      Value *X32 = (Bits < 32) ? B.CreateZExt(X, B.getInt32Ty()) : X;
      Value *Msb = B.CreateCall(UMsb32, {X32});
      Res = B.CreateSelect(B.CreateICmpEQ(Msb, Neg1), Neg1,
                           B.CreateSub(B.getInt32(Bits - 1), Msb));
    }
    break;
  }
  case DXILOp::FMax:
    Res = CallBinary(Intrinsic::maxnum);
    break;
  case DXILOp::FMin:
    Res = CallBinary(Intrinsic::minnum);
    break;
  case DXILOp::IMax:
    Res = CallBinary(Intrinsic::smax);
    break;
  case DXILOp::IMin:
    Res = CallBinary(Intrinsic::smin);
    break;
  case DXILOp::UMax:
    Res = CallBinary(Intrinsic::umax);
    break;
  case DXILOp::UMin:
    Res = CallBinary(Intrinsic::umin);
    break;
  case DXILOp::FMad:
  case DXILOp::Fma: {
    Value *A = CI->getArgOperand(1);
    Value *Bv = CI->getArgOperand(2);
    Value *C = CI->getArgOperand(3);
    Function *Fn =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::fma, {A->getType()});
    Res = B.CreateCall(Fn, {A, Bv, C});
    break;
  }
  case DXILOp::IMad:
  case DXILOp::UMad: {
    Value *A = CI->getArgOperand(1);
    Value *Bv = CI->getArgOperand(2);
    Value *C = CI->getArgOperand(3);
    Res = B.CreateAdd(B.CreateMul(A, Bv), C);
    break;
  }
  case DXILOp::Ibfe:
  case DXILOp::Ubfe: {
    Value *Width = CI->getArgOperand(1);
    Value *Offset = CI->getArgOperand(2);
    Value *Val = CI->getArgOperand(3);
    Type *Ty = Val->getType();
    unsigned BitWidth = Ty->getIntegerBitWidth();
    Value *W = B.CreateAnd(Width, ConstantInt::get(Ty, BitWidth - 1));
    Value *O = B.CreateAnd(Offset, ConstantInt::get(Ty, BitWidth - 1));
    Value *IsZeroW = B.CreateICmpEQ(W, ConstantInt::get(Ty, 0));
    Value *ShlAmt = B.CreateSub(ConstantInt::get(Ty, BitWidth),
                                B.CreateAdd(W, O));
    Value *ShrAmt = B.CreateSub(ConstantInt::get(Ty, BitWidth), W);
    Value *Left = B.CreateShl(Val, ShlAmt);
    Value *Extracted = (Op == DXILOp::Ibfe) ? B.CreateAShr(Left, ShrAmt)
                                            : B.CreateLShr(Left, ShrAmt);
    Res = B.CreateSelect(IsZeroW, ConstantInt::get(Ty, 0), Extracted);
    break;
  }
  case DXILOp::Bfi: {
    Value *Width = CI->getArgOperand(1);
    Value *Offset = CI->getArgOperand(2);
    Value *NewBits = CI->getArgOperand(3);
    Value *OldBits = CI->getArgOperand(4);
    Type *Ty = OldBits->getType();
    unsigned BitWidth = Ty->getIntegerBitWidth();
    Value *W = B.CreateAnd(Width, ConstantInt::get(Ty, BitWidth - 1));
    Value *O = B.CreateAnd(Offset, ConstantInt::get(Ty, BitWidth - 1));
    Value *One = ConstantInt::get(Ty, 1);
    Value *MaskUnshifted = B.CreateSub(B.CreateShl(One, W), One);
    Value *Mask = B.CreateShl(MaskUnshifted, O);
    Value *Inserted = B.CreateAnd(B.CreateShl(NewBits, O), Mask);
    Value *Kept = B.CreateAnd(OldBits, B.CreateNot(Mask));
    Res = B.CreateOr(Inserted, Kept);
    break;
  }
  case DXILOp::Dot2:
  case DXILOp::Dot3:
  case DXILOp::Dot4: {
    unsigned N = (Op == DXILOp::Dot2) ? 2 : (Op == DXILOp::Dot3 ? 3 : 4);
    Type *ScalarTy = CI->getArgOperand(1)->getType();
    Type *VecTy = FixedVectorType::get(ScalarTy, N);
    Value *VA = PoisonValue::get(VecTy);
    Value *VB = PoisonValue::get(VecTy);
    for (unsigned I = 0; I < N; ++I) {
      VA = B.CreateInsertElement(VA, CI->getArgOperand(1 + I), B.getInt32(I));
      VB = B.CreateInsertElement(VB, CI->getArgOperand(1 + N + I),
                                 B.getInt32(I));
    }
    Function *Fn =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::spv_fdot, {VecTy});
    Res = B.CreateCall(Fn, {VA, VB});
    break;
  }
  case DXILOp::LegacyF32ToF16: {
    Value *H = B.CreateFPTrunc(CI->getArgOperand(1), Type::getHalfTy(Ctx));
    Value *I16 = B.CreateBitCast(H, B.getInt16Ty());
    Res = B.CreateZExt(I16, B.getInt32Ty());
    break;
  }
  case DXILOp::LegacyF16ToF32: {
    Value *I16 = B.CreateTrunc(CI->getArgOperand(1), B.getInt16Ty());
    Value *H = B.CreateBitCast(I16, Type::getHalfTy(Ctx));
    Res = B.CreateFPExt(H, Type::getFloatTy(Ctx));
    break;
  }
  case DXILOp::Dot4AddI8Packed: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_dot4add_i8packed);
    Res = B.CreateCall(Fn, {CI->getArgOperand(1), CI->getArgOperand(2),
                            CI->getArgOperand(3)});
    break;
  }
  case DXILOp::Dot4AddU8Packed: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_dot4add_u8packed);
    Res = B.CreateCall(Fn, {CI->getArgOperand(1), CI->getArgOperand(2),
                            CI->getArgOperand(3)});
    break;
  }
  case DXILOp::Pack4x8: {
    auto *ModeC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    uint64_t Mode = ModeC ? ModeC->getZExtValue() : 0;
    Value *Acc = B.getInt32(0);
    for (unsigned I = 0; I < 4; ++I) {
      Value *V = CI->getArgOperand(2 + I);
      Type *ElemTy = V->getType();
      if (Mode == 1) {
        // UClamp: signed clamp [0, 255]
        Value *Lo = ConstantInt::get(ElemTy, 0);
        Value *Hi = ConstantInt::get(ElemTy, 255);
        V = B.CreateSelect(B.CreateICmpSLT(V, Lo), Lo, V);
        V = B.CreateSelect(B.CreateICmpSGT(V, Hi), Hi, V);
      } else if (Mode == 2) {
        // SClamp: signed clamp [-128, 127]
        Value *Lo = ConstantInt::getSigned(ElemTy, -128);
        Value *Hi = ConstantInt::getSigned(ElemTy, 127);
        V = B.CreateSelect(B.CreateICmpSLT(V, Lo), Lo, V);
        V = B.CreateSelect(B.CreateICmpSGT(V, Hi), Hi, V);
      }
      Value *Byte = B.CreateZExt(B.CreateTrunc(V, B.getInt8Ty()), B.getInt32Ty());
      if (I > 0)
        Byte = B.CreateShl(Byte, B.getInt32(I * 8));
      Acc = B.CreateOr(Acc, Byte);
    }
    Res = Acc;
    break;
  }
  case DXILOp::Unpack4x8: {
    auto *ModeC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    uint64_t Mode = ModeC ? ModeC->getZExtValue() : 0;
    Value *Pk = CI->getArgOperand(2);
    auto *ST = cast<StructType>(CI->getType());
    Type *ElemTy = ST->getElementType(0);
    Value *Ret = PoisonValue::get(ST);
    for (unsigned I = 0; I < 4; ++I) {
      Value *Shifted = (I == 0) ? Pk : B.CreateLShr(Pk, B.getInt32(I * 8));
      Value *Byte = B.CreateTrunc(Shifted, B.getInt8Ty());
      Value *Ext = (Mode == 1) ? B.CreateSExt(Byte, ElemTy)
                               : B.CreateZExt(Byte, ElemTy);
      Ret = B.CreateInsertValue(Ret, Ext, {I});
    }
    Res = Ret;
    break;
  }
  case DXILOp::MakeDouble: {
    Value *Lo = B.CreateZExt(CI->getArgOperand(1), B.getInt64Ty());
    Value *Hi = B.CreateShl(B.CreateZExt(CI->getArgOperand(2), B.getInt64Ty()),
                            B.getInt64(32));
    Res = B.CreateBitCast(B.CreateOr(Lo, Hi), Type::getDoubleTy(Ctx));
    break;
  }
  case DXILOp::SplitDouble: {
    Value *Bits = B.CreateBitCast(CI->getArgOperand(1), B.getInt64Ty());
    Value *Lo = B.CreateTrunc(Bits, B.getInt32Ty());
    Value *Hi = B.CreateTrunc(B.CreateLShr(Bits, B.getInt64(32)), B.getInt32Ty());
    Value *Ret = PoisonValue::get(CI->getType());
    Ret = B.CreateInsertValue(Ret, Lo, {0});
    Ret = B.CreateInsertValue(Ret, Hi, {1});
    Res = Ret;
    break;
  }
  case DXILOp::UAddc: {
    Value *A = CI->getArgOperand(1);
    Value *BV = CI->getArgOperand(2);
    Value *Sum = B.CreateAdd(A, BV);
    Value *Carry = B.CreateICmpULT(Sum, A);
    auto *ST = cast<StructType>(CI->getType());
    Value *CarryVal = (ST->getElementType(1)->isIntegerTy(1))
                          ? Carry
                          : B.CreateZExt(Carry, ST->getElementType(1));
    Value *Ret = PoisonValue::get(ST);
    Ret = B.CreateInsertValue(Ret, Sum, {0});
    Ret = B.CreateInsertValue(Ret, CarryVal, {1});
    Res = Ret;
    break;
  }
  case DXILOp::Dot2AddHalf: {
    Type *F32 = Type::getFloatTy(Ctx);
    Value *Acc = CI->getArgOperand(1);
    Value *A0 = B.CreateFPExt(CI->getArgOperand(2), F32);
    Value *A1 = B.CreateFPExt(CI->getArgOperand(3), F32);
    Value *B0 = B.CreateFPExt(CI->getArgOperand(4), F32);
    Value *B1 = B.CreateFPExt(CI->getArgOperand(5), F32);
    Value *Prod = B.CreateFAdd(B.CreateFMul(A0, B0), B.CreateFMul(A1, B1));
    Res = B.CreateFAdd(Acc, Prod);
    break;
  }
  case DXILOp::CheckAccessFullyMapped: {
    // Non-sparse resources are always fully mapped; return status != 0.
    Res = B.CreateICmpNE(CI->getArgOperand(1),
                         ConstantInt::get(CI->getArgOperand(1)->getType(), 0));
    break;
  }
  default:
    return makeError("SPIR-V backend: unsupported math/bit op {0}",
                     static_cast<unsigned>(Op));
  }

  CI->replaceAllUsesWith(Res);
  return Error::success();
}

Error SPIRVLowerer::lowerPixelOp(CallInst *CI, DXILOp Op) {
  IRBuilder<> B(CI);
  switch (Op) {
  case DXILOp::Discard: {
    Value *Cond = CI->getArgOperand(1);
    if (auto *C = dyn_cast<ConstantInt>(Cond)) {
      if (!C->isZero()) {
        Function *Fn =
            Intrinsic::getOrInsertDeclaration(&M, Intrinsic::spv_discard);
        B.CreateCall(Fn);
      }
      return Error::success();
    }
    BasicBlock *CurBB = CI->getParent();
    BasicBlock *ContBB = CurBB->splitBasicBlock(CI->getNextNode(), "discard.cont");
    BasicBlock *KillBB =
        BasicBlock::Create(Ctx, "discard.kill", CurBB->getParent(), ContBB);
    CurBB->getTerminator()->eraseFromParent();
    B.SetInsertPoint(CurBB);
    B.CreateCondBr(Cond, KillBB, ContBB);
    B.SetInsertPoint(KillBB);
    Function *Fn =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::spv_discard);
    B.CreateCall(Fn);
    B.CreateBr(ContBB);
    return Error::success();
  }
  case DXILOp::DerivCoarseX:
  case DXILOp::DerivCoarseY:
  case DXILOp::DerivFineX:
  case DXILOp::DerivFineY: {
    Intrinsic::ID ID = Intrinsic::spv_ddx_coarse;
    if (Op == DXILOp::DerivCoarseY)
      ID = Intrinsic::spv_ddy_coarse;
    else if (Op == DXILOp::DerivFineX)
      ID = Intrinsic::spv_ddx_fine;
    else if (Op == DXILOp::DerivFineY)
      ID = Intrinsic::spv_ddy_fine;
    Value *X = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, ID, {X->getType()});
    CI->replaceAllUsesWith(B.CreateCall(Fn, {X}));
    return Error::success();
  }
  default:
    return makeError("SPIR-V backend: unsupported pixel op {0}",
                     static_cast<unsigned>(Op));
  }
}

Error SPIRVLowerer::lowerDXILCall(CallInst *CI) {
  auto *OpC = dyn_cast<ConstantInt>(CI->getArgOperand(0));
  if (!OpC)
    return makeError("SPIR-V backend: non-constant DXIL opcode in call to '{0}'",
                     CI->getCalledFunction()->getName());

  auto Op = static_cast<DXILOp>(OpC->getZExtValue());
  IRBuilder<> B(CI);

  switch (Op) {
  case DXILOp::CreateHandle:
    return lowerCreateHandle(CI);
  case DXILOp::CreateHandleFromBinding:
    return lowerCreateHandleFromBinding(CI);
  case DXILOp::CreateHandleFromHeap:
    return lowerCreateHandleFromHeap(CI);
  case DXILOp::AnnotateHandle:
    return lowerAnnotateHandle(CI);
  case DXILOp::RawBufferLoad:
    return lowerRawBufferLoad(CI);
  case DXILOp::RawBufferStore:
    return lowerRawBufferStore(CI);
  case DXILOp::BufferLoad:
    return lowerBufferLoad(CI);
  case DXILOp::BufferStore:
    return lowerBufferStore(CI);
  case DXILOp::CBufferLoadLegacy:
    return lowerCBufferLoadLegacy(CI);
  case DXILOp::BufferUpdateCounter:
    return lowerBufferUpdateCounter(CI);
  case DXILOp::TextureLoad:
    return lowerTextureLoad(CI);
  case DXILOp::TextureStore:
    return lowerTextureStore(CI);
  case DXILOp::Sample:
  case DXILOp::SampleBias:
  case DXILOp::SampleLevel:
  case DXILOp::SampleGrad:
  case DXILOp::SampleCmp:
  case DXILOp::SampleCmpLevelZero:
    return lowerTextureSample(CI, Op);
  case DXILOp::TextureGather:
  case DXILOp::TextureGatherCmp:
    return lowerTextureGather(CI, Op);
  case DXILOp::CalculateLOD:
    return lowerCalculateLOD(CI);
  case DXILOp::GetDimensions:
    return lowerGetDimensions(CI);
  case DXILOp::AtomicBinOp:
    return lowerAtomicBinOp(CI);
  case DXILOp::AtomicCompareExchange:
    return lowerAtomicCompareExchange(CI);
  case DXILOp::Barrier:
  case DXILOp::BarrierByMemoryType:
  case DXILOp::BarrierByMemoryHandle:
    return lowerBarrier(CI, Op);

  case DXILOp::ThreadId: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_thread_id, {B.getInt32Ty()});
    CI->replaceAllUsesWith(B.CreateCall(Fn, {CI->getArgOperand(1)}));
    return Error::success();
  }
  case DXILOp::GroupId: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_group_id, {B.getInt32Ty()});
    CI->replaceAllUsesWith(B.CreateCall(Fn, {CI->getArgOperand(1)}));
    return Error::success();
  }
  case DXILOp::ThreadIdInGroup: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_thread_id_in_group, {B.getInt32Ty()});
    CI->replaceAllUsesWith(B.CreateCall(Fn, {CI->getArgOperand(1)}));
    return Error::success();
  }
  case DXILOp::FlattenedThreadIdInGroup: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_flattened_thread_id_in_group);
    CI->replaceAllUsesWith(B.CreateCall(Fn));
    return Error::success();
  }

  case DXILOp::WaveIsFirstLane:
  case DXILOp::WaveGetLaneIndex:
  case DXILOp::WaveGetLaneCount:
  case DXILOp::WaveAnyTrue:
  case DXILOp::WaveAllTrue:
  case DXILOp::WaveActiveAllEqual:
  case DXILOp::WaveActiveBallot:
  case DXILOp::WaveReadLaneAt:
  case DXILOp::WaveReadLaneFirst:
  case DXILOp::WaveActiveOp:
  case DXILOp::WaveActiveBit:
  case DXILOp::WavePrefixOp:
  case DXILOp::WaveAllBitCount:
  case DXILOp::WavePrefixBitCount:
  case DXILOp::WaveMultiPrefixOp:
  case DXILOp::WaveMultiPrefixBitCount:
  case DXILOp::QuadOp:
    return lowerWaveOp(CI, Op);

  case DXILOp::FAbs:
  case DXILOp::Saturate:
  case DXILOp::IsNaN:
  case DXILOp::IsInf:
  case DXILOp::IsFinite:
  case DXILOp::Cos:
  case DXILOp::Sin:
  case DXILOp::Tan:
  case DXILOp::Acos:
  case DXILOp::Asin:
  case DXILOp::Atan:
  case DXILOp::Hcos:
  case DXILOp::Hsin:
  case DXILOp::Htan:
  case DXILOp::Exp:
  case DXILOp::Frc:
  case DXILOp::Log:
  case DXILOp::Sqrt:
  case DXILOp::Rsqrt:
  case DXILOp::RoundNE:
  case DXILOp::RoundNI:
  case DXILOp::RoundPI:
  case DXILOp::RoundZ:
  case DXILOp::Bfrev:
  case DXILOp::Countbits:
  case DXILOp::FirstbitLo:
  case DXILOp::FirstbitHi:
  case DXILOp::FirstbitSHi:
  case DXILOp::FMax:
  case DXILOp::FMin:
  case DXILOp::IMax:
  case DXILOp::IMin:
  case DXILOp::UMax:
  case DXILOp::UMin:
  case DXILOp::FMad:
  case DXILOp::Fma:
  case DXILOp::IMad:
  case DXILOp::UMad:
  case DXILOp::Ibfe:
  case DXILOp::Ubfe:
  case DXILOp::Bfi:
  case DXILOp::Dot2:
  case DXILOp::Dot3:
  case DXILOp::Dot4:
  case DXILOp::LegacyF32ToF16:
  case DXILOp::LegacyF16ToF32:
  case DXILOp::Dot4AddI8Packed:
  case DXILOp::Dot4AddU8Packed:
  case DXILOp::Pack4x8:
  case DXILOp::Unpack4x8:
  case DXILOp::MakeDouble:
  case DXILOp::SplitDouble:
  case DXILOp::UAddc:
  case DXILOp::Dot2AddHalf:
  case DXILOp::CheckAccessFullyMapped:
    return lowerMathOrBitOp(CI, Op);

  case DXILOp::Discard:
  case DXILOp::DerivCoarseX:
  case DXILOp::DerivCoarseY:
  case DXILOp::DerivFineX:
  case DXILOp::DerivFineY:
    return lowerPixelOp(CI, Op);

  case DXILOp::LoadInput:
    return lowerLoadInput(CI);
  case DXILOp::StoreOutput:
    return lowerStoreOutput(CI);

  case DXILOp::AllocateRayQuery:
  case DXILOp::RayQuery_TraceRayInline:
  case DXILOp::RayQuery_Proceed:
  case DXILOp::RayQuery_Abort:
  case DXILOp::RayQuery_CommitNonOpaqueTriangleHit:
  case DXILOp::RayQuery_CommitProceduralPrimitiveHit:
  case DXILOp::RayQuery_CommittedStatus:
  case DXILOp::RayQuery_CandidateType:
  case DXILOp::RayQuery_CommittedTriangleFrontFace:
  case DXILOp::RayQuery_CommittedTriangleBarycentrics:
  case DXILOp::RayQuery_RayFlags:
  case DXILOp::RayQuery_WorldRayOrigin:
  case DXILOp::RayQuery_WorldRayDirection:
  case DXILOp::RayQuery_RayTMin:
  case DXILOp::RayQuery_CommittedRayT:
  case DXILOp::RayQuery_CommittedInstanceID:
  case DXILOp::RayQuery_CommittedInstanceIndex:
  case DXILOp::RayQuery_CommittedPrimitiveIndex:
  case DXILOp::RayQuery_CommittedGeometryIndex:
  case DXILOp::RayQuery_CommittedInstanceContributionToHitGroupIndex:
  case DXILOp::DispatchRaysIndex:
  case DXILOp::DispatchRaysDimensions:
  case DXILOp::WorldRayOrigin:
  case DXILOp::WorldRayDirection:
  case DXILOp::RayTCurrent:
  case DXILOp::PrimitiveIndex:
  case DXILOp::TraceRay:
    return lowerRaytracingOp(CI, Op);

  default:
    return makeError("SPIR-V backend: unsupported DXIL op '{0}' (opcode {1})",
                     CI->getCalledFunction()->getName(),
                     static_cast<unsigned>(Op));
  }
}

void SPIRVLowerer::prepareStageInterface() {
  if (Reflect.Stage != ShaderStage::Vertex &&
      Reflect.Stage != ShaderStage::Pixel)
    return;

  auto buildStageVars = [&](const std::vector<StageParameter> &Params,
                            unsigned AddrSpace, bool IsOutput,
                            DenseMap<uint32_t, StageVar> &Table) {
    uint32_t NextLoc = 0;
    for (const StageParameter &P : Params) {
      StageVar SV;
      SV.Param = &P;

      std::optional<uint32_t> BuiltIn;
      bool IsBoolBI = false;
      switch (P.Kind) {
      case 1: // VertexID
        BuiltIn = 42; // VertexIndex
        break;
      case 2: // InstanceID
        BuiltIn = 43; // InstanceIndex
        break;
      case 3: // Position
        BuiltIn = (Reflect.Stage == ShaderStage::Pixel && !IsOutput) ? 15u : 0u;
        break;
      case 4: // RenderTargetArrayIndex
        BuiltIn = 9; // Layer
        break;
      case 5: // ViewPortArrayIndex
        BuiltIn = 10; // ViewportIndex
        break;
      case 6: // ClipDistance
        BuiltIn = 3;
        break;
      case 7: // CullDistance
        BuiltIn = 4;
        break;
      case 10: // PrimitiveID
        BuiltIn = 7;
        break;
      case 13: // IsFrontFace
        BuiltIn = 17; // FrontFacing
        IsBoolBI = true;
        break;
      case 14: // SampleIndex
        BuiltIn = 18; // SampleId
        break;
      case 17: // Depth
        BuiltIn = 22; // FragDepth
        break;
      case 18: // Coverage
        BuiltIn = 20; // SampleMask
        break;
      default:
        break;
      }

      SV.IsBoolBuiltIn = IsBoolBI;
      Type *ScalarTy =
          IsBoolBI ? static_cast<Type *>(Type::getInt1Ty(Ctx))
                   : getComponentLLVMType(P.Type);
      unsigned Cols = std::max(1u, P.Columns);
      unsigned Rows = std::max(1u, P.Rows);
      Type *RowTy =
          (Cols == 1) ? ScalarTy : FixedVectorType::get(ScalarTy, Cols);
      Type *VarTy = (Rows == 1) ? RowTy : ArrayType::get(RowTy, Rows);
      SV.VarTy = VarTy;

      std::string Name =
          (IsOutput ? "out_" : "in_") +
          (P.Semantic.empty() ? ("sig_" + std::to_string(P.ID)) : P.Semantic) +
          "_" + std::to_string(P.SemanticIndex);
      auto *GV = new GlobalVariable(
          M, VarTy, /*isConstant=*/false, GlobalValue::InternalLinkage,
          PoisonValue::get(VarTy), Name, /*InsertBefore=*/nullptr,
          GlobalValue::NotThreadLocal, AddrSpace);
      SV.GV = GV;

      if (BuiltIn) {
        SV.Decorations.push_back({11u /* BuiltIn */, *BuiltIn});
      } else if (P.Kind == 16 /* Target */) {
        SV.Decorations.push_back({30u /* Location */, P.SemanticIndex});
      } else {
        SV.Decorations.push_back({30u /* Location */, NextLoc});
        NextLoc += Rows;
      }

      if (Reflect.Stage == ShaderStage::Pixel && !IsOutput && !BuiltIn) {
        bool IsInt = ScalarTy->isIntegerTy();
        if (IsInt || P.Interpolation == 1 /* Constant */)
          SV.Decorations.push_back({14u /* Flat */});
        else if (P.Interpolation == 3 /* LinearCentroid */)
          SV.Decorations.push_back({16u /* Centroid */});
        else if (P.Interpolation == 4 /* LinearNoPerspective */)
          SV.Decorations.push_back({13u /* NoPerspective */});
        else if (P.Interpolation == 5 /* LinearNoPerspectiveCentroid */) {
          SV.Decorations.push_back({13u /* NoPerspective */});
          SV.Decorations.push_back({16u /* Centroid */});
        } else if (P.Interpolation == 6 /* LinearSample */)
          SV.Decorations.push_back({17u /* Sample */});
        else if (P.Interpolation == 7 /* LinearNoPerspectiveSample */) {
          SV.Decorations.push_back({13u /* NoPerspective */});
          SV.Decorations.push_back({17u /* Sample */});
        }
      }

      Table[P.ID] = std::move(SV);
    }
  };

  buildStageVars(Reflect.Inputs, 7 /* Input */, false, InputVars);
  buildStageVars(Reflect.Outputs, 8 /* Output */, true, OutputVars);
}

void SPIRVLowerer::emitStageDecorations() {
  if (InputVars.empty() && OutputVars.empty())
    return;

  IRBuilder<> B(&*Entry->getEntryBlock().getFirstInsertionPt());
  auto emitForTable = [&](DenseMap<uint32_t, StageVar> &Table) {
    for (auto &[ID, SV] : Table) {
      if (!SV.Referenced) {
        if (SV.GV && SV.GV->use_empty())
          SV.GV->eraseFromParent();
        continue;
      }
      SmallVector<Metadata *, 4> DecList;
      for (const auto &D : SV.Decorations) {
        SmallVector<Metadata *, 2> Ops;
        for (uint32_t V : D)
          Ops.push_back(ConstantAsMetadata::get(B.getInt32(V)));
        DecList.push_back(MDNode::get(Ctx, Ops));
      }
      MDNode *DecMD = MDNode::get(Ctx, DecList);
      Function *DecFn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::spv_assign_decoration, {SV.GV->getType()});
      B.CreateCall(DecFn, {SV.GV, MetadataAsValue::get(Ctx, DecMD)});
    }
  };
  emitForTable(InputVars);
  emitForTable(OutputVars);
}

Error SPIRVLowerer::lowerLoadInput(CallInst *CI) {
  auto *IdC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  if (!IdC)
    return makeError("SPIR-V backend: non-constant inputSigId in loadInput");

  auto It = InputVars.find(static_cast<uint32_t>(IdC->getZExtValue()));
  if (It == InputVars.end())
    return makeError("SPIR-V backend: unknown input signature ID {0}",
                     IdC->getZExtValue());
  StageVar &SV = It->second;
  SV.Referenced = true;

  IRBuilder<> B(CI);
  Value *Ptr = SV.GV;
  Type *RowTy = SV.VarTy;
  if (auto *AT = dyn_cast<ArrayType>(SV.VarTy)) {
    Value *RowIdx = CI->getArgOperand(2);
    Ptr = B.CreateInBoundsGEP(SV.VarTy, Ptr, {B.getInt32(0), RowIdx});
    RowTy = AT->getElementType();
  }
  Value *Loaded = B.CreateLoad(RowTy, Ptr);
  Value *Val = Loaded;
  if (isa<FixedVectorType>(RowTy)) {
    Value *ColIdx = CI->getArgOperand(3);
    Val = B.CreateExtractElement(Loaded, ColIdx);
  }
  if (Val->getType() != CI->getType()) {
    if (Val->getType()->isIntegerTy(1) && CI->getType()->isIntegerTy())
      Val = B.CreateZExt(Val, CI->getType());
    else if (Val->getType()->isIntegerTy() && CI->getType()->isIntegerTy())
      Val = B.CreateZExtOrTrunc(Val, CI->getType());
    else
      Val = B.CreateBitCast(Val, CI->getType());
  }
  CI->replaceAllUsesWith(Val);
  return Error::success();
}

Error SPIRVLowerer::lowerStoreOutput(CallInst *CI) {
  auto *IdC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  if (!IdC)
    return makeError("SPIR-V backend: non-constant outputSigId in storeOutput");

  auto It = OutputVars.find(static_cast<uint32_t>(IdC->getZExtValue()));
  if (It == OutputVars.end())
    return makeError("SPIR-V backend: unknown output signature ID {0}",
                     IdC->getZExtValue());
  StageVar &SV = It->second;
  SV.Referenced = true;

  IRBuilder<> B(CI);
  Value *Ptr = SV.GV;
  Type *ElemTy = SV.VarTy;
  SmallVector<Value *, 3> Indices;
  Indices.push_back(B.getInt32(0));
  if (auto *AT = dyn_cast<ArrayType>(ElemTy)) {
    Indices.push_back(CI->getArgOperand(2));
    ElemTy = AT->getElementType();
  }
  if (auto *VT = dyn_cast<FixedVectorType>(ElemTy)) {
    Indices.push_back(CI->getArgOperand(3));
    ElemTy = VT->getElementType();
  }
  if (Indices.size() > 1)
    Ptr = B.CreateInBoundsGEP(SV.VarTy, Ptr, Indices);

  Value *Val = CI->getArgOperand(4);
  if (Val->getType() != ElemTy) {
    if (Val->getType()->isIntegerTy() && ElemTy->isIntegerTy())
      Val = B.CreateZExtOrTrunc(Val, ElemTy);
    else
      Val = B.CreateBitCast(Val, ElemTy);
  }
  B.CreateStore(Val, Ptr);
  return Error::success();
}

Error SPIRVLowerer::lowerRaytracingOp(CallInst *CI, DXILOp Op) {
  IRBuilder<> B(CI);
  auto &EntryBB = Entry->getEntryBlock();

  auto getOrCreateRQ = [&](Value *Handle) -> RayQueryState & {
    auto It = RayQueries.find(Handle);
    if (It != RayQueries.end())
      return It->second;
    IRBuilder<> EB(&*EntryBB.getFirstInsertionPt());
    RayQueryState S;
    Type *I32 = EB.getInt32Ty();
    Type *F32 = EB.getFloatTy();
    Type *I1 = EB.getInt1Ty();
    S.Flags = EB.CreateAlloca(I32, nullptr, "rq.flags");
    for (int I = 0; I < 3; ++I) {
      S.Origin[I] = EB.CreateAlloca(F32, nullptr, "rq.orig");
      S.Dir[I] = EB.CreateAlloca(F32, nullptr, "rq.dir");
    }
    S.TMin = EB.CreateAlloca(F32, nullptr, "rq.tmin");
    S.CommittedT = EB.CreateAlloca(F32, nullptr, "rq.t");
    S.CommittedStatus = EB.CreateAlloca(I32, nullptr, "rq.status");
    S.CandidateType = EB.CreateAlloca(I32, nullptr, "rq.cand");
    S.FrontFace = EB.CreateAlloca(I1, nullptr, "rq.ff");
    S.Bary[0] = EB.CreateAlloca(F32, nullptr, "rq.bary0");
    S.Bary[1] = EB.CreateAlloca(F32, nullptr, "rq.bary1");
    S.InstanceID = EB.CreateAlloca(I32, nullptr, "rq.instid");
    S.InstanceIndex = EB.CreateAlloca(I32, nullptr, "rq.instidx");
    S.PrimitiveIndex = EB.CreateAlloca(I32, nullptr, "rq.primidx");
    S.GeometryIndex = EB.CreateAlloca(I32, nullptr, "rq.geomidx");
    S.InstanceContrib = EB.CreateAlloca(I32, nullptr, "rq.contrib");

    EB.CreateStore(EB.getInt32(0), S.Flags);
    EB.CreateStore(EB.getInt32(0), S.CommittedStatus);
    EB.CreateStore(EB.getInt32(0), S.CandidateType);
    EB.CreateStore(EB.getFalse(), S.FrontFace);
    EB.CreateStore(ConstantFP::get(F32, 0.0), S.TMin);
    EB.CreateStore(ConstantFP::get(F32, 0.0), S.CommittedT);
    EB.CreateStore(ConstantFP::get(F32, 0.0), S.Bary[0]);
    EB.CreateStore(ConstantFP::get(F32, 0.0), S.Bary[1]);
    EB.CreateStore(EB.getInt32(0), S.InstanceID);
    EB.CreateStore(EB.getInt32(0), S.InstanceIndex);
    EB.CreateStore(EB.getInt32(0), S.PrimitiveIndex);
    EB.CreateStore(EB.getInt32(0), S.GeometryIndex);
    EB.CreateStore(EB.getInt32(0), S.InstanceContrib);
    return RayQueries.try_emplace(Handle, S).first->second;
  };

  switch (Op) {
  case DXILOp::AllocateRayQuery: {
    RayQueryState &S = getOrCreateRQ(CI);
    B.CreateStore(CI->getArgOperand(1), S.Flags);
    CI->replaceAllUsesWith(B.getInt32(0));
    return Error::success();
  }
  case DXILOp::RayQuery_TraceRayInline: {
    // (op, rqHandle, accelHandle, rayFlags, mask, ox, oy, oz, tmin, dx, dy, dz, tmax)
    RayQueryState &S = getOrCreateRQ(CI->getArgOperand(1));
    auto AccelOrErr = resolveHandle(CI->getArgOperand(2));
    if (!AccelOrErr)
      return AccelOrErr.takeError();
    B.CreateStore(CI->getArgOperand(3), S.Flags);
    for (int I = 0; I < 3; ++I) {
      B.CreateStore(CI->getArgOperand(5 + I), S.Origin[I]);
      B.CreateStore(CI->getArgOperand(9 + I), S.Dir[I]);
    }
    B.CreateStore(CI->getArgOperand(8), S.TMin);
    B.CreateStore(CI->getArgOperand(12), S.CommittedT);
    // Read triangle count header from acceleration structure buffer so the
    // accel binding remains live in the SPIR-V interface.
    Value *Header = loadTypedValueFromStorageBuffer(B, *AccelOrErr,
                                                    B.getInt32(0), B.getInt32Ty());
    Value *HasHit = B.CreateICmpNE(Header, B.getInt32(0));
    B.CreateStore(B.CreateSelect(HasHit, B.getInt32(1), B.getInt32(0)),
                  S.CommittedStatus);
    return Error::success();
  }
  case DXILOp::RayQuery_Proceed: {
    CI->replaceAllUsesWith(B.getFalse());
    return Error::success();
  }
  case DXILOp::RayQuery_Abort:
    return Error::success();
  case DXILOp::RayQuery_CommitNonOpaqueTriangleHit: {
    RayQueryState &S = getOrCreateRQ(CI->getArgOperand(1));
    B.CreateStore(B.getInt32(1), S.CommittedStatus);
    return Error::success();
  }
  case DXILOp::RayQuery_CommitProceduralPrimitiveHit: {
    RayQueryState &S = getOrCreateRQ(CI->getArgOperand(1));
    B.CreateStore(B.getInt32(2), S.CommittedStatus);
    B.CreateStore(CI->getArgOperand(2), S.CommittedT);
    return Error::success();
  }
  case DXILOp::RayQuery_CommittedStatus:
    CI->replaceAllUsesWith(
        B.CreateLoad(B.getInt32Ty(), getOrCreateRQ(CI->getArgOperand(1)).CommittedStatus));
    return Error::success();
  case DXILOp::RayQuery_CandidateType:
    CI->replaceAllUsesWith(
        B.CreateLoad(B.getInt32Ty(), getOrCreateRQ(CI->getArgOperand(1)).CandidateType));
    return Error::success();
  case DXILOp::RayQuery_CommittedTriangleFrontFace:
    CI->replaceAllUsesWith(
        B.CreateLoad(B.getInt1Ty(), getOrCreateRQ(CI->getArgOperand(1)).FrontFace));
    return Error::success();
  case DXILOp::RayQuery_CommittedTriangleBarycentrics: {
    auto *CompC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
    unsigned Comp = (CompC && CompC->getZExtValue() == 1) ? 1u : 0u;
    CI->replaceAllUsesWith(
        B.CreateLoad(B.getFloatTy(), getOrCreateRQ(CI->getArgOperand(1)).Bary[Comp]));
    return Error::success();
  }
  case DXILOp::RayQuery_RayFlags:
    CI->replaceAllUsesWith(
        B.CreateLoad(B.getInt32Ty(), getOrCreateRQ(CI->getArgOperand(1)).Flags));
    return Error::success();
  case DXILOp::RayQuery_WorldRayOrigin: {
    auto *CompC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
    unsigned Comp = CompC ? std::min<unsigned>(CompC->getZExtValue(), 2u) : 0u;
    CI->replaceAllUsesWith(
        B.CreateLoad(B.getFloatTy(), getOrCreateRQ(CI->getArgOperand(1)).Origin[Comp]));
    return Error::success();
  }
  case DXILOp::RayQuery_WorldRayDirection: {
    auto *CompC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
    unsigned Comp = CompC ? std::min<unsigned>(CompC->getZExtValue(), 2u) : 0u;
    CI->replaceAllUsesWith(
        B.CreateLoad(B.getFloatTy(), getOrCreateRQ(CI->getArgOperand(1)).Dir[Comp]));
    return Error::success();
  }
  case DXILOp::RayQuery_RayTMin:
    CI->replaceAllUsesWith(
        B.CreateLoad(B.getFloatTy(), getOrCreateRQ(CI->getArgOperand(1)).TMin));
    return Error::success();
  case DXILOp::RayQuery_CommittedRayT:
    CI->replaceAllUsesWith(
        B.CreateLoad(B.getFloatTy(), getOrCreateRQ(CI->getArgOperand(1)).CommittedT));
    return Error::success();
  case DXILOp::RayQuery_CommittedInstanceID:
    CI->replaceAllUsesWith(
        B.CreateLoad(B.getInt32Ty(), getOrCreateRQ(CI->getArgOperand(1)).InstanceID));
    return Error::success();
  case DXILOp::RayQuery_CommittedInstanceIndex:
    CI->replaceAllUsesWith(
        B.CreateLoad(B.getInt32Ty(), getOrCreateRQ(CI->getArgOperand(1)).InstanceIndex));
    return Error::success();
  case DXILOp::RayQuery_CommittedPrimitiveIndex:
    CI->replaceAllUsesWith(
        B.CreateLoad(B.getInt32Ty(), getOrCreateRQ(CI->getArgOperand(1)).PrimitiveIndex));
    return Error::success();
  case DXILOp::RayQuery_CommittedGeometryIndex:
    CI->replaceAllUsesWith(
        B.CreateLoad(B.getInt32Ty(), getOrCreateRQ(CI->getArgOperand(1)).GeometryIndex));
    return Error::success();
  case DXILOp::RayQuery_CommittedInstanceContributionToHitGroupIndex:
    CI->replaceAllUsesWith(
        B.CreateLoad(B.getInt32Ty(), getOrCreateRQ(CI->getArgOperand(1)).InstanceContrib));
    return Error::success();
  case DXILOp::DispatchRaysIndex: {
    Function *TidFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_thread_id, {B.getInt32Ty()});
    CI->replaceAllUsesWith(B.CreateCall(TidFn, {CI->getArgOperand(1)}));
    return Error::success();
  }
  case DXILOp::DispatchRaysDimensions: {
    Function *NwgFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_num_workgroups, {B.getInt32Ty()});
    CI->replaceAllUsesWith(B.CreateCall(NwgFn, {CI->getArgOperand(1)}));
    return Error::success();
  }
  case DXILOp::WorldRayOrigin:
  case DXILOp::WorldRayDirection:
  case DXILOp::RayTCurrent:
    CI->replaceAllUsesWith(ConstantFP::get(CI->getType(), 0.0));
    return Error::success();
  case DXILOp::PrimitiveIndex:
    CI->replaceAllUsesWith(B.getInt32(0));
    return Error::success();
  case DXILOp::TraceRay: {
    auto AccelOrErr = resolveHandle(CI->getArgOperand(1));
    if (!AccelOrErr)
      return AccelOrErr.takeError();
    (void)loadTypedValueFromStorageBuffer(B, *AccelOrErr, B.getInt32(0),
                                          B.getInt32Ty());
    return Error::success();
  }
  default:
    return makeError("SPIR-V backend: unsupported raytracing op {0}",
                     static_cast<unsigned>(Op));
  }
}

void SPIRVLowerer::fixupExistingAtomics(Function &F) {
  SyncScope::ID WorkgroupScope = Ctx.getOrInsertSyncScopeID("workgroup");
  SyncScope::ID DeviceScope = Ctx.getOrInsertSyncScopeID("device");

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *RMW = dyn_cast<AtomicRMWInst>(&I)) {
        unsigned AS = RMW->getPointerAddressSpace();
        RMW->setSyncScopeID(AS == 3 ? WorkgroupScope : DeviceScope);
        if (RMW->getOrdering() == AtomicOrdering::Monotonic ||
            RMW->getOrdering() == AtomicOrdering::SequentiallyConsistent)
          RMW->setOrdering(AtomicOrdering::AcquireRelease);
      } else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I)) {
        unsigned AS = CX->getPointerAddressSpace();
        CX->setSyncScopeID(AS == 3 ? WorkgroupScope : DeviceScope);
        if (CX->getSuccessOrdering() == AtomicOrdering::Monotonic ||
            CX->getSuccessOrdering() == AtomicOrdering::SequentiallyConsistent)
          CX->setSuccessOrdering(AtomicOrdering::AcquireRelease);
        if (CX->getFailureOrdering() == AtomicOrdering::Monotonic ||
            CX->getFailureOrdering() == AtomicOrdering::SequentiallyConsistent)
          CX->setFailureOrdering(AtomicOrdering::Acquire);
      }
    }
  }
}

void SPIRVLowerer::normalizeArrayGlobalAccesses(Function &F) {
  auto FixPtr = [&](Instruction *I, unsigned PtrOpIdx, Type *AccessTy) {
    Value *Ptr = I->getOperand(PtrOpIdx);
    Type *BaseTy = nullptr;
    if (auto *GV = dyn_cast<GlobalVariable>(Ptr))
      BaseTy = GV->getValueType();
    else if (auto *AI = dyn_cast<AllocaInst>(Ptr))
      BaseTy = AI->getAllocatedType();
    if (!BaseTy || BaseTy == AccessTy)
      return;
    if (!isa<ArrayType>(BaseTy) && !isa<StructType>(BaseTy))
      return;
    IRBuilder<> B(I);
    SmallVector<Value *, 4> Indices;
    Indices.push_back(B.getInt32(0));
    Type *Cur = BaseTy;
    while (Cur != AccessTy) {
      if (auto *AT = dyn_cast<ArrayType>(Cur)) {
        Indices.push_back(B.getInt32(0));
        Cur = AT->getElementType();
      } else if (auto *ST = dyn_cast<StructType>(Cur);
                 ST && ST->getNumElements() > 0) {
        Indices.push_back(B.getInt32(0));
        Cur = ST->getElementType(0);
      } else {
        break;
      }
    }
    if (Indices.size() <= 1)
      return;
    auto *GEP = GetElementPtrInst::CreateInBounds(BaseTy, Ptr, Indices, "",
                                                  I->getIterator());
    I->setOperand(PtrOpIdx, GEP);
  };

  // Rewrite `gep ElemTy, ptr %arr, idx` (or `gep i8, ptr %arr, byteOff`) on an
  // `alloca [N x ElemTy]` or global `[N x ElemTy]` back into
  // `gep [N x ElemTy], ptr %arr, i32 0, elemIdx` so LLVM's SPIR-V backend does
  // not insert a `spv_ptrcast` that crashes SPIRVLegalizePointerCast.
  SmallVector<GetElementPtrInst *, 16> GEPsToRewrite;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
        GEPsToRewrite.push_back(GEP);
    }
  }
  const DataLayout &DL = M.getDataLayout();
  for (GetElementPtrInst *GEP : GEPsToRewrite) {
    Value *Ptr = GEP->getPointerOperand();
    Type *BaseTy = nullptr;
    if (auto *GV = dyn_cast<GlobalVariable>(Ptr))
      BaseTy = GV->getValueType();
    else if (auto *AI = dyn_cast<AllocaInst>(Ptr))
      BaseTy = AI->getAllocatedType();
    auto *AT = dyn_cast_or_null<ArrayType>(BaseTy);
    if (!AT || GEP->getSourceElementType() == BaseTy || GEP->getNumIndices() != 1)
      continue;
    Type *ElemTy = AT->getElementType();
    Value *Idx = GEP->getOperand(1);
    IRBuilder<> B(GEP);
    Value *ElemIdx = nullptr;
    if (GEP->getSourceElementType() == ElemTy) {
      ElemIdx = Idx;
    } else if (GEP->getSourceElementType()->isIntegerTy(8)) {
      uint64_t ElemSize = DL.getTypeAllocSize(ElemTy);
      if (ElemSize > 0 && isPowerOf2_64(ElemSize)) {
        unsigned Shift = llvm::Log2_64(ElemSize);
        ElemIdx = (Shift == 0)
                      ? Idx
                      : B.CreateAShr(Idx, ConstantInt::get(Idx->getType(), Shift));
      } else if (ElemSize > 0) {
        ElemIdx = B.CreateSDiv(Idx, ConstantInt::get(Idx->getType(), ElemSize));
      }
    }
    if (!ElemIdx)
      continue;
    Value *NewGEP =
        B.CreateInBoundsGEP(BaseTy, Ptr, {B.getInt32(0), ElemIdx}, GEP->getName());
    GEP->replaceAllUsesWith(NewGEP);
    GEP->eraseFromParent();
  }

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *LI = dyn_cast<LoadInst>(&I))
        FixPtr(LI, 0, LI->getType());
      else if (auto *SI = dyn_cast<StoreInst>(&I))
        FixPtr(SI, 1, SI->getValueOperand()->getType());
      else if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
        FixPtr(RMW, 0, RMW->getValOperand()->getType());
      else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I))
        FixPtr(CX, 0, CX->getNewValOperand()->getType());
    }
  }
}

void SPIRVLowerer::configureEntryAndLinkage(StringRef StageName) {
  // Unwrap single-element [1 x T] groupshared globals into scalar T globals to
  // avoid LLVM 23 SPIRVLegalizePointerCast crashes on zero-offset GEPs.
  SmallVector<GlobalVariable *, 4> SingleElemGVs;
  for (GlobalVariable &GV : M.globals()) {
    if (GV.getAddressSpace() == 3) {
      if (auto *AT = dyn_cast<ArrayType>(GV.getValueType());
          AT && AT->getNumElements() == 1)
        SingleElemGVs.push_back(&GV);
    }
  }
  for (GlobalVariable *OldGV : SingleElemGVs) {
    Type *ElemTy = cast<ArrayType>(OldGV->getValueType())->getElementType();
    auto *NewGV = new GlobalVariable(
        M, ElemTy, OldGV->isConstant(), GlobalValue::InternalLinkage,
        PoisonValue::get(ElemTy), "", OldGV, OldGV->getThreadLocalMode(),
        OldGV->getAddressSpace());
    NewGV->setAlignment(OldGV->getAlign());
    NewGV->takeName(OldGV);
    OldGV->replaceAllUsesWith(NewGV);
    OldGV->eraseFromParent();
  }

  for (GlobalVariable &GV : M.globals()) {
    if (GV.hasExternalLinkage()) {
      if (!GV.hasInitializer())
        GV.setInitializer(PoisonValue::get(GV.getValueType()));
      GV.setLinkage(GlobalValue::InternalLinkage);
    }
  }
  for (Function &F : M) {
    if (&F == Entry) {
      F.setLinkage(GlobalValue::ExternalLinkage);
      if (!Reflect.EntryName.empty())
        F.setName(Reflect.EntryName);
      F.addFnAttr("hlsl.shader", StageName);
      if (StageName == "compute") {
        std::string NT = std::to_string(Reflect.NumThreads[0]) + "," +
                         std::to_string(Reflect.NumThreads[1]) + "," +
                         std::to_string(Reflect.NumThreads[2]);
        F.addFnAttr("hlsl.numthreads", NT);
      }
    } else if (!F.isDeclaration()) {
      F.setLinkage(GlobalValue::InternalLinkage);
    }
  }
}

void SPIRVLowerer::cleanupDXILArtifacts() {
  SmallVector<NamedMDNode *, 8> ToEraseMD;
  for (NamedMDNode &NMD : M.named_metadata()) {
    if (NMD.getName().starts_with("dx."))
      ToEraseMD.push_back(&NMD);
  }
  for (NamedMDNode *NMD : ToEraseMD)
    M.eraseNamedMetadata(NMD);

  SmallVector<Function *, 16> DeadDecls;
  for (Function &F : M) {
    if (F.isDeclaration() && F.getName().starts_with("dx.op.") &&
        F.use_empty())
      DeadDecls.push_back(&F);
  }
  for (Function *F : DeadDecls)
    F->eraseFromParent();

  SmallVector<GlobalVariable *, 16> DeadGVs;
  for (GlobalVariable &GV : M.globals()) {
    unsigned AS = GV.getAddressSpace();
    if (GV.use_empty() && !GV.getName().starts_with(".spv.name.") &&
        AS != 7 && AS != 8)
      DeadGVs.push_back(&GV);
  }
  for (GlobalVariable *GV : DeadGVs)
    GV->eraseFromParent();
}

Error SPIRVLowerer::run() {
  auto StageOrErr = stageTripleSuffix(Reflect.Stage, Opts.StageAsCompute);
  if (!StageOrErr)
    return StageOrErr.takeError();

  prepareStageInterface();

  SmallVector<CallInst *, 64> DXILCalls;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    fixupExistingAtomics(F);
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (Function *Callee = CI->getCalledFunction()) {
            if (Callee->getName().starts_with("dx.op."))
              DXILCalls.push_back(CI);
          }
        }
      }
    }
  }

  for (CallInst *CI : DXILCalls) {
    if (Error E = lowerDXILCall(CI))
      return E;
  }

  for (auto It = DXILCalls.rbegin(), End = DXILCalls.rend(); It != End; ++It) {
    CallInst *CI = *It;
    if (!CI->use_empty())
      CI->replaceAllUsesWith(PoisonValue::get(CI->getType()));
    CI->eraseFromParent();
  }

  configureEntryAndLinkage(*StageOrErr);
  cleanupDXILArtifacts();

#if LLVM_VERSION_MAJOR >= 24
  PassBuilder PB(Ctx);
#else
  PassBuilder PB;
#endif
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM;
  FunctionPassManager FPM;
  FPM.addPass(SROAPass(SROAOptions::ModifyCFG));
  FPM.addPass(ADCEPass());
  FPM.addPass(SimplifyCFGPass());
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  MPM.run(M, MAM);

  for (Function &F : M) {
    if (!F.isDeclaration())
      normalizeArrayGlobalAccesses(F);
  }

  emitStageDecorations();

  return Error::success();
}

} // namespace

Expected<std::string> emitSPIRV(Module &M, Function *Entry, Reflection &Reflect,
                                const Options &Opts) {
  initializeSPIRVTargetOnce();

  auto StageOrErr = stageTripleSuffix(Reflect.Stage, Opts.StageAsCompute);
  if (!StageOrErr)
    return StageOrErr.takeError();

  std::string VulkanVer =
      Opts.VulkanVersion.empty() ? "vulkan1.3" : Opts.VulkanVersion;
  std::string TripleStr = "spirv1.6-unknown-" + VulkanVer + "-" + *StageOrErr;

  Triple TheTriple(TripleStr);
  std::string LookupErr;
  const Target *T = TargetRegistry::lookupTarget(TheTriple, LookupErr);
  if (!T)
    return makeError("SPIR-V target lookup failed for '{0}': {1}", TripleStr,
                     LookupErr);

  TargetOptions TOpts;
  std::unique_ptr<TargetMachine> TM(T->createTargetMachine(
      TheTriple, /*CPU=*/"", /*Features=*/"", TOpts, Reloc::Static,
      CodeModel::Small, CodeGenOptLevel::None));
  if (!TM)
    return makeError("failed to create SPIR-V TargetMachine for '{0}'",
                     TripleStr);

  M.setTargetTriple(TheTriple);
  M.setDataLayout(TM->createDataLayout());

  SPIRVLowerer Lowerer(M, Entry, Reflect, Opts);
  if (Error E = Lowerer.run())
    return std::move(E);

  std::string VerifyErr;
  raw_string_ostream VerifyOS(VerifyErr);
  if (verifyModule(M, &VerifyOS))
    return makeError("SPIR-V lowered LLVM IR failed verification:\n{0}",
                     VerifyErr);

  SmallString<0> OutBuf;
  raw_svector_ostream OutStream(OutBuf);
  legacy::PassManager CodeGenPM;
  CodeGenFileType FileType = Opts.EmitSPIRVAsm ? CodeGenFileType::AssemblyFile
                                               : CodeGenFileType::ObjectFile;
  if (TM->addPassesToEmitFile(CodeGenPM, OutStream, nullptr, FileType))
    return makeError("SPIR-V TargetMachine cannot emit requested file type");

  CodeGenPM.run(M);
  return std::string(OutBuf.data(), OutBuf.size());
}

} // namespace crucible

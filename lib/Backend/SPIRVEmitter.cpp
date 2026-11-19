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
  DenseMap<const ResourceBinding *, GlobalVariable *> NameGlobals;

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
  case ResourceKind::StructuredBuffer: {
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
    Value *OldWord = B.CreateAlignedLoad(B.getInt32Ty(), Ptr, Align(4));
    Value *Shift = B.CreateShl(B.CreateAnd(ByteOffset, B.getInt32(2)),
                               B.getInt32(3));
    Value *Mask = B.CreateNot(B.CreateShl(B.getInt32(0xFFFFu), Shift));
    Value *Inserted =
        B.CreateOr(B.CreateAnd(OldWord, Mask),
                   B.CreateShl(B.CreateZExt(Val16, B.getInt32Ty()), Shift));
    B.CreateAlignedStore(Inserted, Ptr, Align(4));
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

Error SPIRVLowerer::lowerCreateHandleFromBinding(CallInst *CI) {
  auto *BindC = dyn_cast<ConstantStruct>(CI->getArgOperand(1));
  if (!BindC || BindC->getNumOperands() < 4)
    return makeError(
        "SPIR-V backend: non-constant ResBind in createHandleFromBinding");

  uint32_t LB =
      static_cast<uint32_t>(cast<ConstantInt>(BindC->getOperand(0))->getZExtValue());
  uint32_t Space =
      static_cast<uint32_t>(cast<ConstantInt>(BindC->getOperand(2))->getZExtValue());
  auto RC = static_cast<ResourceClass>(
      cast<ConstantInt>(BindC->getOperand(3))->getZExtValue());

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

Error SPIRVLowerer::lowerAnnotateHandle(CallInst *CI) {
  auto RH = resolveHandle(CI->getArgOperand(1));
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
  GlobalVariable *NameGV = getOrCreateResourceName(*RH.RB);
  Function *CounterHandleFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::spv_resource_counterhandlefrombinding, {CounterTy});
  Value *CounterH = B.CreateCall(
      CounterHandleFn,
      {B.getInt32(RH.RB->SPIRVSet), B.getInt32(RH.RB->CounterSPIRVBinding),
       B.getInt32(RH.RB->RangeSize), B.getInt32(0), NameGV});
  Function *UpdateFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::spv_resource_updatecounter, {CounterTy});
  Value *Inc = CI->getArgOperand(2);
  Value *Old = B.CreateCall(UpdateFn, {CounterH, Inc});
  CI->replaceAllUsesWith(Old);
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
         Coord->getType(), CmpVal->getType(), Offset->getType()});
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
         CmpVal->getType(), Offset->getType()});
    ResVec = B.CreateCall(
        Fn, {TexRH->SpvHandle, SampRH->SpvHandle, Coord, CmpVal, Offset});
  } else {
    Value *Channel = CI->getArgOperand(9);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_resource_gather,
        {Vec4Ty, TexRH->SpvHandleTy, SampRH->SpvHandleTy, Coord->getType(),
         Channel->getType(), Offset->getType()});
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

  if (RH.RB->Kind != ResourceKind::RawBuffer &&
      RH.RB->Kind != ResourceKind::StructuredBuffer)
    return makeError("SPIR-V backend: atomicBinOp on non-storage buffer '{0}' "
                     "is not yet supported",
                     RH.RB->Name);

  IRBuilder<> B(CI);
  Value *ByteOff = computeBufferByteOffset(B, *RH.RB, CI->getArgOperand(3),
                                           CI->getArgOperand(4));
  Value *WordIdx = B.CreateLShr(ByteOff, B.getInt32(2));
  Value *Ptr = getStorageBufferWordPtr(B, RH, WordIdx);
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

  if (RH.RB->Kind != ResourceKind::RawBuffer &&
      RH.RB->Kind != ResourceKind::StructuredBuffer)
    return makeError("SPIR-V backend: atomicCompareExchange on non-storage "
                     "buffer '{0}' is not yet supported",
                     RH.RB->Name);

  return makeError("SPIR-V backend: atomicCompareExchange is not yet supported "
                   "on LLVM 23 (upstream SPIRVLegalizePointerCast bug on cmpxchg)");
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
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_wave_get_lane_count);
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
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_wave_readlane, {Val->getType()});
    Res = B.CreateCall(Fn, {Val, B.getInt32(0)});
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
  case DXILOp::Countbits:
    Res = CallUnary(Intrinsic::ctpop);
    break;
  case DXILOp::FirstbitLo: {
    Value *X = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_firstbitlow, {X->getType()});
    Res = B.CreateCall(Fn, {X});
    break;
  }
  case DXILOp::FirstbitHi: {
    Value *X = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_firstbituhigh, {X->getType()});
    Res = B.CreateCall(Fn, {X});
    break;
  }
  case DXILOp::FirstbitSHi: {
    Value *X = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_firstbitshigh, {X->getType()});
    Res = B.CreateCall(Fn, {X});
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
    return lowerMathOrBitOp(CI, Op);

  case DXILOp::Discard:
  case DXILOp::DerivCoarseX:
  case DXILOp::DerivCoarseY:
  case DXILOp::DerivFineX:
  case DXILOp::DerivFineY:
    return lowerPixelOp(CI, Op);

  default:
    return makeError("SPIR-V backend: unsupported DXIL op '{0}' (opcode {1})",
                     CI->getCalledFunction()->getName(),
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
    auto *GV = dyn_cast<GlobalVariable>(I->getOperand(PtrOpIdx));
    if (!GV)
      return;
    Type *GVTy = GV->getValueType();
    if (!isa<ArrayType>(GVTy) || GVTy == AccessTy)
      return;
    IRBuilder<> B(I);
    SmallVector<Value *, 4> Indices;
    Indices.push_back(B.getInt32(0));
    Type *Cur = GVTy;
    while (auto *AT = dyn_cast<ArrayType>(Cur)) {
      Indices.push_back(B.getInt32(0));
      Cur = AT->getElementType();
      if (Cur == AccessTy)
        break;
    }
    auto *GEP = GetElementPtrInst::CreateInBounds(GVTy, GV, Indices, "",
                                                  I->getIterator());
    I->setOperand(PtrOpIdx, GEP);
  };

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
      if (Reflect.Stage == ShaderStage::Compute ||
          Reflect.Stage == ShaderStage::Mesh ||
          Reflect.Stage == ShaderStage::Amplification) {
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
    if (GV.use_empty() && !GV.getName().starts_with(".spv.name."))
      DeadGVs.push_back(&GV);
  }
  for (GlobalVariable *GV : DeadGVs)
    GV->eraseFromParent();
}

Error SPIRVLowerer::run() {
  auto StageOrErr = stageTripleSuffix(Reflect.Stage);
  if (!StageOrErr)
    return StageOrErr.takeError();

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

  PassBuilder PB;
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

  return Error::success();
}

} // namespace

Expected<std::string> emitSPIRV(Module &M, Function *Entry, Reflection &Reflect,
                                const Options &Opts) {
  initializeSPIRVTargetOnce();

  auto StageOrErr = stageTripleSuffix(Reflect.Stage);
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
      CodeModel::Small, CodeGenOptLevel::Default));
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

//===- SPIRVEmitter.cpp - Crucible MLIR -> LLVM SPIR-V backend --*- C++ -*-===//
//
// Part of Crucible, an open-source multi-frontend, multi-backend shader compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Lowers unified Crucible MLIR (gpu.* + math.* + llvm.* + crucible.*, with zero
// DXILOp dependencies) into LLVM's Vulkan SPIR-V IR (target("spirv.*") types
// and llvm.spv.* / standard LLVM intrinsics), then invokes LLVM's SPIR-V
// TargetMachine to emit validated Vulkan SPIR-V.
//
//===----------------------------------------------------------------------===//

#include "SPIRVEmitter.h"
#include "crucible/Diagnostic.h"
#include "crucible/Dialect/CrucibleDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
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
#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>
#include <mutex>
#include <optional>

using namespace llvm;

namespace crucible {
namespace {

/// Private internal bridge opcodes between Crucible MLIR (gpu/math/crucible)
/// and LLVM SPIR-V lowering. Completely independent of DXIL.
enum class SpvOp : uint32_t {
  // Resources
  BindResource = 1,
  HeapResource,
  // Buffers & CBuffers
  BufferLoad,
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
  // Stage I/O & Pixel
  LoadInput,
  StoreOutput,
  EmitStream,
  CutStream,
  LoadControlPoint,
  LoadPatchConstant,
  StorePatchConstant,
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
  Tan,
  Asin,
  Acos,
  Atan,
  Sinh,
  Cosh,
  Tanh,
  FirstBit,
  Bfe,
  Bfi,
  Dot,
  DotPacked,
  Pack4x8,
  Unpack4x8,
  LegacyF32ToF16,
  LegacyF16ToF32,
  MakeDouble,
  SplitDouble,
  UAddc,
  // Raytracing & RayQuery
  SysValRT,
  TraceRay,
  RayQueryAlloc,
  RayQueryTraceInline,
  RayQueryProceed,
  RayQueryAbort,
  RayQueryCommit,
  RayQueryGet,
};

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
  case ShaderStage::Geometry:
    return "geometry";
  case ShaderStage::Hull:
    return "hull";
  case ShaderStage::Domain:
    return "domain";
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
  Value *ArrIdx = nullptr;
};

struct StageVar {
  const StageParameter *Param = nullptr;
  GlobalVariable *GV = nullptr;
  Type *VarTy = nullptr;
  SmallVector<SmallVector<uint32_t, 2>, 3> Decorations;
  bool IsBoolBuiltIn = false;
  bool IsFlatFloatArray = false;
  bool IsPerVertexArray = false;
  bool Referenced = false;
};

class SPIRVLowerer {
public:
  SPIRVLowerer(Module &M, Function *Entry, Reflection &Reflect,
               const Options &Opts)
      : M(M), Ctx(M.getContext()), Entry(Entry), Reflect(Reflect), Opts(Opts) {}

  Error run();
  bool usedArrayLength() const { return !ArrayLengthHelpers.empty(); }
  bool usedCmpXchg() const { return !CmpXchgHelpers.empty(); }
  bool usedImageAtomics() const {
    return !ImgAtomicHelpers.empty() || !ImgCmpXchgHelpers.empty();
  }
  bool usedGeometryStreamOps() const {
    return EmitVertexHelper != nullptr || EndPrimitiveHelper != nullptr;
  }
  uint32_t getDepthOutputMode() const { return DepthOutputMode; }
  bool usedClipDistance() const { return UsedClipDistance; }
  bool usedCullDistance() const { return UsedCullDistance; }
  bool usedCubeArray() const { return UsedCubeArray; }
  bool usedViewportIndex() const { return UsedViewportIndex; }
  bool usedShaderViewportIndex() const { return UsedShaderViewportIndex; }
  bool usedStencilExport() const { return UsedStencilExport; }
  bool usedRuntimeDescriptorArray() const { return UsedRuntimeDescriptorArray; }

private:
  bool UsedClipDistance = false;
  bool UsedCullDistance = false;
  bool UsedCubeArray = false;
  bool UsedViewportIndex = false;
  bool UsedShaderViewportIndex = false;
  bool UsedStencilExport = false;
  bool UsedRuntimeDescriptorArray = false;
  Module &M;
  LLVMContext &Ctx;
  Function *Entry;
  Reflection &Reflect;
  const Options &Opts;

  DenseMap<const Value *, ResolvedHandle> HandleMap;
  DenseSet<const ResourceBinding *> AtomicImageBindings;
  DenseMap<Type *, Function *> ArrayLengthHelpers;
  DenseMap<unsigned, Function *> CmpXchgHelpers;
  DenseMap<std::tuple<uint32_t, uint32_t, uint32_t, Type *>, Function *>
      ImgAtomicHelpers;
  DenseMap<std::tuple<uint32_t, uint32_t, Type *>, Function *>
      ImgCmpXchgHelpers;
  Function *EmitVertexHelper = nullptr;
  Function *EndPrimitiveHelper = nullptr;
  uint32_t DepthOutputMode = 0; // 1=Replacing, 2=Greater, 3=Less
  std::vector<std::unique_ptr<ResourceBinding>> SyntheticBindings;
  DenseMap<const ResourceBinding *, GlobalVariable *> NameGlobals;
  DenseMap<uint32_t, StageVar> InputVars;
  DenseMap<uint32_t, StageVar> OutputVars;
  DenseMap<uint32_t, StageVar> PatchConstVars;
  DenseMap<uint32_t, StageVar> ExtraBuiltInInputs;
  GlobalVariable *getOrCreateBuiltInInput(uint32_t BuiltInId, Type *Ty,
                                          StringRef Name) {
    auto It = ExtraBuiltInInputs.find(BuiltInId);
    if (It != ExtraBuiltInInputs.end())
      return It->second.GV;
    StageVar SV;
    SV.VarTy = Ty;
    SV.GV = new GlobalVariable(
        M, Ty, /*isConstant=*/false, GlobalValue::InternalLinkage,
        PoisonValue::get(Ty), Name, /*InsertBefore=*/nullptr,
        GlobalValue::NotThreadLocal, 7 /* Input */);
    SV.Decorations.push_back({11u /* BuiltIn */, BuiltInId});
    SV.Referenced = true;
    ExtraBuiltInInputs[BuiltInId] = std::move(SV);
    return ExtraBuiltInInputs[BuiltInId].GV;
  }
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

  Error lowerBridgeCall(CallInst *CI);
  Error lowerBindResource(CallInst *CI);
  Error lowerHeapResource(CallInst *CI);
  Error lowerBufferLoad(CallInst *CI);
  Error lowerBufferStore(CallInst *CI);
  Error lowerCBufferLoad(CallInst *CI);
  Error lowerBufferUpdateCounter(CallInst *CI);
  Error lowerTextureLoad(CallInst *CI);
  Error lowerTextureStore(CallInst *CI);
  Error lowerTextureSample(CallInst *CI);
  Error lowerTextureGather(CallInst *CI);
  Error lowerCalculateLOD(CallInst *CI);
  Error lowerGetDimensions(CallInst *CI);
  Error lowerAtomicBinOp(CallInst *CI);
  Error lowerAtomicCompareExchange(CallInst *CI);
  Error lowerBarrier(CallInst *CI);
  Error lowerWaveOp(CallInst *CI, SpvOp Op);
  Error lowerMathOrBitOp(CallInst *CI, SpvOp Op);
  Error lowerPixelOp(CallInst *CI, SpvOp Op);
  Error lowerLoadInput(CallInst *CI);
  Error lowerStoreOutput(CallInst *CI);
  Error lowerLoadControlPoint(CallInst *CI);
  Error lowerLoadPatchConstant(CallInst *CI);
  Error lowerStorePatchConstant(CallInst *CI);
  Error lowerRaytracingOp(CallInst *CI, SpvOp Op);

  void mergeHullPatchConstantFunc();
  void prepareStageInterface();
  void emitStageDecorations();
  void rewriteFloatAtomicGlobals();
  void fixupExistingAtomics(Function &F);
  void normalizeArrayGlobalAccesses(Function &F);
  void configureEntryAndLinkage(StringRef StageName);
  void cleanupBridgeArtifacts();
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
    return TargetExtType::get(Ctx, "spirv.VulkanBuffer", {ArrTy},
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
    unsigned Format = 0u; // Unknown
    if (IsWritable && AtomicImageBindings.contains(&RB)) {
      SampledTy = Type::getInt32Ty(Ctx);
      Format = 33u; // R32ui (required by OpImageTexelPointer)
    } else if (SampledTy->isHalfTy())
      SampledTy = Type::getFloatTy(Ctx);
    else if (SampledTy->isIntegerTy(16))
      SampledTy = Type::getInt32Ty(Ctx);
    else if (RB.Kind == ResourceKind::TypedBuffer &&
             (SampledTy->isIntegerTy(64) || SampledTy->isDoubleTy()))
      SampledTy = Type::getInt32Ty(Ctx);
    unsigned Dim = getSPIRVImageDim(RB.Kind);
    unsigned Arrayed = isArrayedResource(RB.Kind) ? 1u : 0u;
    if (RB.Kind == ResourceKind::TextureCubeArray) {
      // Upstream LLVM's SPIRVModuleAnalysis::initAvailableCapabilitiesForVulkan
      // omits SampledCubeArray, causing checkSatisfiable() to abort on
      // OpTypeImage Dim=Cube Arrayed=1. Emit Dim=3D Arrayed=1 (unique, no extra
      // capability requirement) and patch Dim back to Cube + emit
      // OpCapability SampledCubeArray (45) in the SPIR-V binary post-pass.
      Dim = 2u; // Dim 3D sentinel
      UsedCubeArray = true;
    }
    unsigned MS = isMSResource(RB.Kind) ? 1u : 0u;
    unsigned Sampled = IsWritable ? 2u : 1u;
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

Error SPIRVLowerer::lowerBindResource(CallInst *CI) {
  // (op, class, hasRecordId, recordId, space, binding, index, nonUniform)
  auto *ClassC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  auto *HasRecC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
  auto *RecC = dyn_cast<ConstantInt>(CI->getArgOperand(3));
  auto *SpaceC = dyn_cast<ConstantInt>(CI->getArgOperand(4));
  auto *BindC = dyn_cast<ConstantInt>(CI->getArgOperand(5));
  if (!ClassC || !HasRecC || !RecC || !SpaceC || !BindC)
    return makeError("SPIR-V backend: non-constant bind_resource metadata");

  auto RC = static_cast<ResourceClass>(ClassC->getZExtValue());
  const ResourceBinding *RB =
      !HasRecC->isZero()
          ? findResourceByID(RC, static_cast<uint32_t>(RecC->getZExtValue()))
          : findResourceByBinding(
                RC, static_cast<uint32_t>(SpaceC->getZExtValue()),
                static_cast<uint32_t>(BindC->getZExtValue()));
  if (!RB)
    return makeError("SPIR-V backend: unknown resource binding");

  Type *ResTy = getSPIRVResourceType(*RB);
  if (!ResTy)
    return makeError("SPIR-V backend: unsupported resource kind {0} for '{1}'",
                     static_cast<unsigned>(RB->Kind), RB->Name);

  IRBuilder<> B(CI);
  Value *RegIdx = CI->getArgOperand(6);
  Value *ArrIdx = (RB->LowerBound == 0)
                      ? RegIdx
                      : B.CreateSub(RegIdx, B.getInt32(RB->LowerBound));
  if (auto *NUI = dyn_cast<ConstantInt>(CI->getArgOperand(7));
      NUI && !NUI->isZero()) {
    Function *NuiFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_resource_nonuniformindex);
    ArrIdx = B.CreateCall(NuiFn, {ArrIdx});
  }

  GlobalVariable *NameGV = getOrCreateResourceName(*RB);
  Function *HandleFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::spv_resource_handlefrombinding, {ResTy});
  uint32_t SpvRangeSize = (RB->RangeSize == 0xFFFFFFFFu) ? 0u : RB->RangeSize;
  if (SpvRangeSize == 0u)
    UsedRuntimeDescriptorArray = true;
  Value *SpvHandle =
      B.CreateCall(HandleFn, {B.getInt32(RB->SPIRVSet),
                              B.getInt32(RB->SPIRVBinding),
                              B.getInt32(SpvRangeSize), ArrIdx, NameGV});
  HandleMap[CI] = ResolvedHandle{RB, SpvHandle, ResTy, ArrIdx};
  return Error::success();
}

Error SPIRVLowerer::lowerHeapResource(CallInst *CI) {
  // (op, index, isSamplerHeap, nonUniform, class, kind, strideOrComp, hasCounter)
  Value *Index = CI->getArgOperand(1);
  bool IsSamplerHeap = !cast<ConstantInt>(CI->getArgOperand(2))->isZero();
  bool NonUniform = !cast<ConstantInt>(CI->getArgOperand(3))->isZero();
  auto RC = static_cast<ResourceClass>(
      cast<ConstantInt>(CI->getArgOperand(4))->getZExtValue());
  uint32_t RawKind =
      static_cast<uint32_t>(cast<ConstantInt>(CI->getArgOperand(5))->getZExtValue());
  uint32_t Prop1 =
      static_cast<uint32_t>(cast<ConstantInt>(CI->getArgOperand(6))->getZExtValue());
  bool HasCounter = !cast<ConstantInt>(CI->getArgOperand(7))->isZero();

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

  if (IsSamplerHeap) {
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

  auto RB = std::make_unique<ResourceBinding>();
  RB->Kind = IsSamplerHeap ? ResourceKind::Sampler
                           : static_cast<ResourceKind>(RawKind);
  RB->HasCounter = HasCounter;
  if (IsSamplerHeap || RB->Kind == ResourceKind::Sampler) {
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
    RB->Class = RC;
    if (RB->Kind == ResourceKind::StructuredBuffer)
      RB->ElementStride = Prop1;
    else if (Prop1 != 0)
      RB->ElementType = static_cast<ComponentType>(Prop1 & 0xFFu);
    else
      RB->ElementType = ComponentType::F32;
    RB->SPIRVSet = Reflect.ResourceHeapSPIRVSet;
    RB->SPIRVBinding = (Reflect.ResourceHeapSPIRVBinding != UINT32_MAX)
                           ? Reflect.ResourceHeapSPIRVBinding
                           : 0u;
    RB->Name = RC == ResourceClass::UAV ? "resource_heap_uav"
                                        : "resource_heap_srv";
  }
  RB->RangeSize = 0xFFFFFFFFu;

  Type *ResTy = getSPIRVResourceType(*RB);
  if (!ResTy)
    return makeError(
        "SPIR-V backend: unsupported dynamic heap resource kind {0}",
        static_cast<unsigned>(RB->Kind));

  IRBuilder<> B(CI);
  Value *Idx = Index;
  if (NonUniform) {
    Function *NuiFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_resource_nonuniformindex);
    Idx = B.CreateCall(NuiFn, {Idx});
  }
  const ResourceBinding *RBPtr = RB.get();
  SyntheticBindings.push_back(std::move(RB));
  GlobalVariable *NameGV = getOrCreateResourceName(*RBPtr);
  Function *HandleFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::spv_resource_handlefrombinding, {ResTy});
  UsedRuntimeDescriptorArray = true;
  Value *SpvHandle =
      B.CreateCall(HandleFn, {B.getInt32(RBPtr->SPIRVSet),
                              B.getInt32(RBPtr->SPIRVBinding),
                              B.getInt32(0u), Idx, NameGV});
  HandleMap[CI] = ResolvedHandle{RBPtr, SpvHandle, ResTy};
  return Error::success();
}

Error SPIRVLowerer::lowerBufferLoad(CallInst *CI) {
  // (op, handle, index, elemOffset, mask) -> <4 x ElemTy>
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  auto *RetVecTy = dyn_cast<FixedVectorType>(CI->getType());
  if (!RetVecTy || RetVecTy->getNumElements() != 4)
    return makeError("SPIR-V backend: unexpected buffer_load return type");

  Type *ElemTy = RetVecTy->getElementType();
  IRBuilder<> B(CI);

  if (RH.RB->Kind == ResourceKind::TypedBuffer) {
    Type *SampledTy = getComponentLLVMType(RH.RB->ElementType);
    bool Is64BitTyped =
        SampledTy->isIntegerTy(64) || SampledTy->isDoubleTy() ||
        ElemTy->isIntegerTy(64) || ElemTy->isDoubleTy();
    if (AtomicImageBindings.contains(RH.RB))
      SampledTy = Type::getInt32Ty(Ctx);
    else if (SampledTy->isHalfTy())
      SampledTy = Type::getFloatTy(Ctx);
    else if (SampledTy->isIntegerTy(16) || Is64BitTyped)
      SampledTy = Type::getInt32Ty(Ctx);
    Type *Vec4Ty = FixedVectorType::get(SampledTy, 4);
    Function *LoadFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_resource_load_typedbuffer, {Vec4Ty, RH.SpvHandleTy});
    Value *Vec = B.CreateCall(LoadFn, {RH.SpvHandle, CI->getArgOperand(2)});
    Value *Res = PoisonValue::get(RetVecTy);
    if (Is64BitTyped && (ElemTy->isIntegerTy(64) || ElemTy->isDoubleTy())) {
      Type *I64Ty = B.getInt64Ty();
      for (unsigned C = 0; C < 2; ++C) {
        Value *Lo = B.CreateZExt(B.CreateExtractElement(Vec, B.getInt32(2 * C)), I64Ty);
        Value *Hi = B.CreateZExt(B.CreateExtractElement(Vec, B.getInt32(2 * C + 1)), I64Ty);
        Value *V64 = B.CreateOr(Lo, B.CreateShl(Hi, B.getInt64(32)));
        if (ElemTy->isDoubleTy())
          V64 = B.CreateBitCast(V64, ElemTy);
        Res = B.CreateInsertElement(Res, V64, B.getInt32(C));
      }
      for (unsigned C = 2; C < 4; ++C)
        Res = B.CreateInsertElement(Res, Constant::getNullValue(ElemTy), B.getInt32(C));
    } else {
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
        Res = B.CreateInsertElement(Res, E, B.getInt32(C));
      }
    }
    CI->replaceAllUsesWith(Res);
    return Error::success();
  }

  uint32_t Mask = 0;
  if (auto *MC = dyn_cast<ConstantInt>(CI->getArgOperand(4));
      MC && MC->getZExtValue() != 0) {
    Mask = static_cast<uint32_t>(MC->getZExtValue());
  } else {
    for (User *U : CI->users()) {
      if (auto *EE = dyn_cast<ExtractElementInst>(U)) {
        if (auto *IC = dyn_cast<ConstantInt>(EE->getIndexOperand());
            IC && IC->getZExtValue() < 4)
          Mask |= (1u << IC->getZExtValue());
        else {
          Mask = 0xF;
          break;
        }
      } else {
        Mask = 0xF;
        break;
      }
    }
    if (Mask == 0)
      Mask = 0x1;
  }

  uint32_t ElemBytes =
      std::max<uint32_t>(1u, static_cast<uint32_t>(ElemTy->getPrimitiveSizeInBits() / 8u));
  Value *BaseByte = computeBufferByteOffset(B, *RH.RB, CI->getArgOperand(2),
                                            CI->getArgOperand(3));
  Value *Res = PoisonValue::get(RetVecTy);
  for (unsigned C = 0; C < 4; ++C) {
    if (!(Mask & (1u << C)))
      continue;
    Value *Off = (C == 0) ? BaseByte
                          : B.CreateAdd(BaseByte, B.getInt32(C * ElemBytes));
    Value *Loaded = loadTypedValueFromStorageBuffer(B, RH, Off, ElemTy);
    Res = B.CreateInsertElement(Res, Loaded, B.getInt32(C));
  }
  CI->replaceAllUsesWith(Res);
  return Error::success();
}

Error SPIRVLowerer::lowerBufferStore(CallInst *CI) {
  // (op, handle, index, elemOffset, v0, v1, v2, v3, mask)
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  IRBuilder<> B(CI);
  if (RH.RB->Kind == ResourceKind::TypedBuffer) {
    Type *SampledTy = getComponentLLVMType(RH.RB->ElementType);
    Type *ValTy = CI->getArgOperand(4)->getType();
    bool Is64BitTyped =
        SampledTy->isIntegerTy(64) || SampledTy->isDoubleTy() ||
        ValTy->isIntegerTy(64) || ValTy->isDoubleTy();
    if (AtomicImageBindings.contains(RH.RB))
      SampledTy = Type::getInt32Ty(Ctx);
    else if (SampledTy->isHalfTy())
      SampledTy = Type::getFloatTy(Ctx);
    else if (SampledTy->isIntegerTy(16) || Is64BitTyped)
      SampledTy = Type::getInt32Ty(Ctx);
    Type *Vec4Ty = FixedVectorType::get(SampledTy, 4);
    Value *Vec = PoisonValue::get(Vec4Ty);
    if (Is64BitTyped && (ValTy->isIntegerTy(64) || ValTy->isDoubleTy())) {
      Type *I64Ty = B.getInt64Ty();
      for (unsigned C = 0; C < 2; ++C) {
        Value *V = CI->getArgOperand(4 + C);
        Value *Lo = B.getInt32(0);
        Value *Hi = B.getInt32(0);
        if (!isa<UndefValue>(V) && !isa<PoisonValue>(V)) {
          Value *V64 = V->getType()->isDoubleTy() ? B.CreateBitCast(V, I64Ty) : V;
          Lo = B.CreateTrunc(V64, B.getInt32Ty());
          Hi = B.CreateTrunc(B.CreateLShr(V64, B.getInt64(32)), B.getInt32Ty());
        }
        Vec = B.CreateInsertElement(Vec, Lo, B.getInt32(2 * C));
        Vec = B.CreateInsertElement(Vec, Hi, B.getInt32(2 * C + 1));
      }
    } else {
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

Error SPIRVLowerer::lowerCBufferLoad(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  auto *RetVecTy = dyn_cast<FixedVectorType>(CI->getType());
  if (!RetVecTy || RetVecTy->getNumElements() == 0)
    return makeError("SPIR-V backend: unexpected cbuffer_load return type");

  Type *Vec4I32 = FixedVectorType::get(Type::getInt32Ty(Ctx), 4);

  IRBuilder<> B(CI);
  Type *UniformPtrTy = PointerType::get(Ctx, 12); // Uniform address space
  Value *RowIdx = CI->getArgOperand(2);
  Function *GetPtrFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::spv_resource_getpointer,
      {UniformPtrTy, RH.SpvHandleTy, RowIdx->getType()});
  Value *RowPtr = B.CreateCall(GetPtrFn, {RH.SpvHandle, RowIdx});
  Value *RawVec = B.CreateAlignedLoad(Vec4I32, RowPtr, Align(16));

  Value *CastVec =
      (RetVecTy == Vec4I32) ? RawVec : B.CreateBitCast(RawVec, RetVecTy);
  CI->replaceAllUsesWith(CastVec);
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
  Value *CounterIdx = RH.ArrIdx ? RH.ArrIdx : B.getInt32(0);
  uint32_t SpvCounterRange =
      (RH.RB->RangeSize == 0xFFFFFFFFu) ? 0u : RH.RB->RangeSize;
  Value *CounterH = B.CreateCall(
      HandleFn,
      {B.getInt32(RH.RB->SPIRVSet), B.getInt32(RH.RB->CounterSPIRVBinding),
       B.getInt32(SpvCounterRange), CounterIdx, NameGV});
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

static Value *castResultVec4(IRBuilder<> &B, FixedVectorType *RetVecTy,
                             Value *Vec4) {
  if (Vec4->getType() == RetVecTy)
    return Vec4;
  Type *ElemTy = RetVecTy->getElementType();
  Value *Res = PoisonValue::get(RetVecTy);
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
    Res = B.CreateInsertElement(Res, E, B.getInt32(C));
  }
  return Res;
}

Error SPIRVLowerer::lowerTextureLoad(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  auto *RetVecTy = dyn_cast<FixedVectorType>(CI->getType());
  if (!RetVecTy || RetVecTy->getNumElements() != 4)
    return makeError("SPIR-V backend: unexpected tex_load return type");

  IRBuilder<> B(CI);
  unsigned Coords = textureCoordDims(RH.RB->Kind);
  SmallVector<Value *, 4> CoordVals;
  for (unsigned I = 0; I < Coords; ++I)
    CoordVals.push_back(CI->getArgOperand(3 + I));
  Value *Coord = packVector(B, B.getInt32Ty(), CoordVals);

  Type *SampledTy = getComponentLLVMType(RH.RB->ElementType);
  if (AtomicImageBindings.contains(RH.RB))
    SampledTy = Type::getInt32Ty(Ctx);
  else if (SampledTy->isHalfTy())
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

  CI->replaceAllUsesWith(castResultVec4(B, RetVecTy, Vec));
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
  if (AtomicImageBindings.contains(RH.RB))
    SampledTy = Type::getInt32Ty(Ctx);
  else if (SampledTy->isHalfTy())
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

Error SPIRVLowerer::lowerTextureSample(CallInst *CI) {
  // (op, mode, tex, samp, c0..c3, o0..o2, lodOrBias, ddx0..2, ddy0..2, cmpVal, clamp)
  auto Mode = static_cast<ir::SampleMode>(
      cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue());
  auto TexRH = resolveHandle(CI->getArgOperand(2));
  if (!TexRH)
    return TexRH.takeError();
  auto SampRH = resolveHandle(CI->getArgOperand(3));
  if (!SampRH)
    return SampRH.takeError();

  auto *RetVecTy = dyn_cast<FixedVectorType>(CI->getType());
  if (!RetVecTy || RetVecTy->getNumElements() != 4)
    return makeError("SPIR-V backend: unexpected tex_sample return type");

  IRBuilder<> B(CI);
  unsigned Coords = textureCoordDims(TexRH->RB->Kind);
  SmallVector<Value *, 4> CoordVals;
  for (unsigned I = 0; I < Coords; ++I)
    CoordVals.push_back(CI->getArgOperand(4 + I));
  Value *Coord = packVector(B, B.getFloatTy(), CoordVals);

  unsigned OffDims = textureOffsetDims(TexRH->RB->Kind);
  Value *Offset = OffDims ? packConstOffset(B, OffDims, CI, 8)
                          : ConstantVector::getSplat(ElementCount::getFixed(2),
                                                     B.getInt32(0));

  Type *SampledTy = getComponentLLVMType(TexRH->RB->ElementType);
  if (SampledTy->isHalfTy())
    SampledTy = Type::getFloatTy(Ctx);
  Type *Vec4Ty = FixedVectorType::get(SampledTy, 4);

  Value *LodOrBias = CI->getArgOperand(11);
  Value *CmpVal = CI->getArgOperand(18);
  Value *Clamp = CI->getArgOperand(19);
  bool HasClamp = !isa<UndefValue>(Clamp) && !isa<PoisonValue>(Clamp);

  Value *ResVec = nullptr;
  switch (Mode) {
  case ir::SampleMode::Implicit: {
    if (HasClamp) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::spv_resource_sample_clamp,
          {Vec4Ty, TexRH->SpvHandleTy, SampRH->SpvHandleTy, Coord->getType(),
           Offset->getType()});
      ResVec = B.CreateCall(
          Fn, {TexRH->SpvHandle, SampRH->SpvHandle, Coord, Offset, Clamp});
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
  case ir::SampleMode::Level: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_resource_samplelevel,
        {Vec4Ty, TexRH->SpvHandleTy, SampRH->SpvHandleTy, Coord->getType(),
         Offset->getType()});
    ResVec = B.CreateCall(
        Fn, {TexRH->SpvHandle, SampRH->SpvHandle, Coord, LodOrBias, Offset});
    break;
  }
  case ir::SampleMode::Bias: {
    if (HasClamp) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::spv_resource_samplebias_clamp,
          {Vec4Ty, TexRH->SpvHandleTy, SampRH->SpvHandleTy, Coord->getType(),
           Offset->getType()});
      ResVec = B.CreateCall(Fn, {TexRH->SpvHandle, SampRH->SpvHandle, Coord,
                                 LodOrBias, Offset, Clamp});
    } else {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::spv_resource_samplebias,
          {Vec4Ty, TexRH->SpvHandleTy, SampRH->SpvHandleTy, Coord->getType(),
           Offset->getType()});
      ResVec = B.CreateCall(
          Fn, {TexRH->SpvHandle, SampRH->SpvHandle, Coord, LodOrBias, Offset});
    }
    break;
  }
  case ir::SampleMode::Grad: {
    unsigned GradDims = (OffDims > 0) ? OffDims : 3u;
    SmallVector<Value *, 3> DdxVals, DdyVals;
    for (unsigned I = 0; I < GradDims; ++I) {
      DdxVals.push_back(CI->getArgOperand(12 + I));
      DdyVals.push_back(CI->getArgOperand(15 + I));
    }
    Value *Ddx = packVector(B, B.getFloatTy(), DdxVals);
    Value *Ddy = packVector(B, B.getFloatTy(), DdyVals);
    if (HasClamp) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::spv_resource_samplegrad_clamp,
          {Vec4Ty, TexRH->SpvHandleTy, SampRH->SpvHandleTy, Coord->getType(),
           Ddx->getType(), Ddy->getType(), Offset->getType()});
      ResVec = B.CreateCall(Fn, {TexRH->SpvHandle, SampRH->SpvHandle, Coord,
                                 Ddx, Ddy, Offset, Clamp});
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
  case ir::SampleMode::Cmp:
  case ir::SampleMode::CmpLevelZero: {
    Value *Scalar = nullptr;
    if (Mode == ir::SampleMode::Cmp && HasClamp) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::spv_resource_samplecmp_clamp,
          {B.getFloatTy(), TexRH->SpvHandleTy, SampRH->SpvHandleTy,
           Coord->getType(), Offset->getType()});
      Scalar = B.CreateCall(
          Fn, {TexRH->SpvHandle, SampRH->SpvHandle, Coord, CmpVal, Offset, Clamp});
    } else {
      Intrinsic::ID ID = (Mode == ir::SampleMode::CmpLevelZero)
                             ? Intrinsic::spv_resource_samplecmplevelzero
                             : Intrinsic::spv_resource_samplecmp;
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, ID,
          {B.getFloatTy(), TexRH->SpvHandleTy, SampRH->SpvHandleTy,
           Coord->getType(), Offset->getType()});
      Scalar = B.CreateCall(
          Fn, {TexRH->SpvHandle, SampRH->SpvHandle, Coord, CmpVal, Offset});
    }
    Value *Res = PoisonValue::get(RetVecTy);
    Res = B.CreateInsertElement(Res, Scalar, B.getInt32(0));
    CI->replaceAllUsesWith(Res);
    return Error::success();
  }
  }

  CI->replaceAllUsesWith(castResultVec4(B, RetVecTy, ResVec));
  return Error::success();
}

Error SPIRVLowerer::lowerTextureGather(CallInst *CI) {
  // (op, tex, samp, c0..c3, o0..o1, channel, cmpVal)
  auto TexRH = resolveHandle(CI->getArgOperand(1));
  if (!TexRH)
    return TexRH.takeError();
  auto SampRH = resolveHandle(CI->getArgOperand(2));
  if (!SampRH)
    return SampRH.takeError();

  auto *RetVecTy = dyn_cast<FixedVectorType>(CI->getType());
  if (!RetVecTy || RetVecTy->getNumElements() != 4)
    return makeError("SPIR-V backend: unexpected tex_gather return type");

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

  Value *CmpVal = CI->getArgOperand(10);
  bool HasCmp = !isa<UndefValue>(CmpVal) && !isa<PoisonValue>(CmpVal);
  Value *ResVec = nullptr;
  if (HasCmp) {
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

  CI->replaceAllUsesWith(castResultVec4(B, RetVecTy, ResVec));
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

  auto *RetVecTy = dyn_cast<FixedVectorType>(CI->getType());
  if (!RetVecTy || RetVecTy->getNumElements() != 4)
    return makeError("SPIR-V backend: unexpected tex_get_dimensions return type");

  IRBuilder<> B(CI);
  Value *Res = PoisonValue::get(RetVecTy);
  if (RH.RB->Kind == ResourceKind::RawBuffer ||
      RH.RB->Kind == ResourceKind::StructuredBuffer) {
    // StorageBuffer is modeled as target("spirv.VulkanBuffer", [0 x i32], 12, ...).
    // LLVM's SPIR-V backend does not have an OpArrayLength intrinsic yet, so
    // emit a sentinel helper call rewritten to OpArrayLength in the SPIR-V
    // binary post-pass.
    Function *&HelperFn = ArrayLengthHelpers[RH.SpvHandleTy];
    if (!HelperFn) {
      std::string FnName = "__crucible_spv_arraylength." +
                           std::to_string(ArrayLengthHelpers.size());
      FunctionType *FT =
          FunctionType::get(B.getInt32Ty(), {RH.SpvHandleTy}, false);
      HelperFn = Function::Create(FT, GlobalValue::InternalLinkage, FnName, M);
      HelperFn->addFnAttr(Attribute::NoInline);
      HelperFn->addFnAttr(Attribute::OptimizeNone);
      BasicBlock *HBB = BasicBlock::Create(Ctx, "entry", HelperFn);
      IRBuilder<> HB(HBB);
      HB.CreateRet(HB.getInt32(0));
    }
    Value *WordCount = B.CreateCall(HelperFn, {RH.SpvHandle});
    Value *ByteSize = B.CreateShl(WordCount, B.getInt32(2));
    if (RH.RB->Kind == ResourceKind::RawBuffer) {
      Res = B.CreateInsertElement(Res, ByteSize, B.getInt32(0));
      for (unsigned I = 1; I < 4; ++I)
        Res = B.CreateInsertElement(Res, B.getInt32(0), B.getInt32(I));
    } else {
      uint32_t Stride = RH.RB->ElementStride ? RH.RB->ElementStride : 4u;
      Value *NumStructs =
          (Stride == 4u)
              ? WordCount
              : ((Stride % 4u == 0u)
                     ? B.CreateUDiv(WordCount, B.getInt32(Stride / 4u))
                     : B.CreateUDiv(ByteSize, B.getInt32(Stride)));
      Res = B.CreateInsertElement(Res, NumStructs, B.getInt32(0));
      Res = B.CreateInsertElement(Res, B.getInt32(Stride), B.getInt32(1));
      Res = B.CreateInsertElement(Res, B.getInt32(0), B.getInt32(2));
      Res = B.CreateInsertElement(Res, B.getInt32(0), B.getInt32(3));
    }
    CI->replaceAllUsesWith(Res);
    return Error::success();
  }
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
      Res = B.CreateInsertElement(Res, B.CreateExtractElement(Dims, B.getInt32(I)),
                                  B.getInt32(I));
    for (unsigned I = RetElems; I < 4; ++I)
      Res = B.CreateInsertElement(Res, B.getInt32(0), B.getInt32(I));
  } else if (IsUAV || RH.RB->Kind == ResourceKind::TypedBuffer) {
    if (Coords == 1) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::spv_resource_getdimensions_x, {RH.SpvHandleTy});
      Value *W = B.CreateCall(Fn, {RH.SpvHandle});
      Res = B.CreateInsertElement(Res, W, B.getInt32(0));
      for (unsigned I = 1; I < 4; ++I)
        Res = B.CreateInsertElement(Res, B.getInt32(0), B.getInt32(I));
    } else {
      Intrinsic::ID ID = (Coords >= 3)
                             ? Intrinsic::spv_resource_getdimensions_xyz
                             : Intrinsic::spv_resource_getdimensions_xy;
      unsigned RetElems = (Coords >= 3) ? 3u : 2u;
      Function *Fn = Intrinsic::getOrInsertDeclaration(&M, ID, {RH.SpvHandleTy});
      Value *Dims = B.CreateCall(Fn, {RH.SpvHandle});
      for (unsigned I = 0; I < RetElems; ++I)
        Res = B.CreateInsertElement(
            Res, B.CreateExtractElement(Dims, B.getInt32(I)), B.getInt32(I));
      for (unsigned I = RetElems; I < 4; ++I)
        Res = B.CreateInsertElement(Res, B.getInt32(0), B.getInt32(I));
    }
  } else {
    Value *Mip = CI->getArgOperand(2);
    if (isa<UndefValue>(Mip) || isa<PoisonValue>(Mip))
      Mip = B.getInt32(0);
    unsigned DimCount = (RH.RB->Kind == ResourceKind::TextureCube)
                            ? 2u
                            : ((Coords >= 3) ? 3u : Coords);
    Intrinsic::ID ID = Intrinsic::spv_resource_getdimensions_levels_xy;
    if (DimCount == 1)
      ID = Intrinsic::spv_resource_getdimensions_levels_x;
    else if (DimCount == 3)
      ID = Intrinsic::spv_resource_getdimensions_levels_xyz;
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, ID, {RH.SpvHandleTy});
    Value *Dims = B.CreateCall(Fn, {RH.SpvHandle, Mip});
    for (unsigned I = 0; I < DimCount; ++I)
      Res = B.CreateInsertElement(
          Res, B.CreateExtractElement(Dims, B.getInt32(I)), B.getInt32(I));
    for (unsigned I = DimCount; I < 3; ++I)
      Res = B.CreateInsertElement(Res, B.getInt32(0), B.getInt32(I));
    Res = B.CreateInsertElement(
        Res, B.CreateExtractElement(Dims, B.getInt32(DimCount)), B.getInt32(3));
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
  switch (static_cast<ir::AtomicOp>(OpC->getZExtValue())) {
  case ir::AtomicOp::Add:
    BinOp = AtomicRMWInst::Add;
    break;
  case ir::AtomicOp::And:
    BinOp = AtomicRMWInst::And;
    break;
  case ir::AtomicOp::Or:
    BinOp = AtomicRMWInst::Or;
    break;
  case ir::AtomicOp::Xor:
    BinOp = AtomicRMWInst::Xor;
    break;
  case ir::AtomicOp::SMin:
    BinOp = AtomicRMWInst::Min;
    break;
  case ir::AtomicOp::SMax:
    BinOp = AtomicRMWInst::Max;
    break;
  case ir::AtomicOp::UMin:
    BinOp = AtomicRMWInst::UMin;
    break;
  case ir::AtomicOp::UMax:
    BinOp = AtomicRMWInst::UMax;
    break;
  case ir::AtomicOp::Xchg:
    BinOp = AtomicRMWInst::Xchg;
    break;
  }

  IRBuilder<> B(CI);
  Value *Val = CI->getArgOperand(6);
  if (RH.RB->Kind == ResourceKind::RawBuffer ||
      RH.RB->Kind == ResourceKind::StructuredBuffer) {
    Value *ByteOff = computeBufferByteOffset(B, *RH.RB, CI->getArgOperand(3),
                                             CI->getArgOperand(4));
    Value *WordIdx = B.CreateLShr(ByteOff, B.getInt32(2));
    Value *Ptr = getStorageBufferWordPtr(B, RH, WordIdx);
    SyncScope::ID DeviceScope = Ctx.getOrInsertSyncScopeID("device");
    auto *RMW = B.CreateAtomicRMW(BinOp, Ptr, Val, Align(4),
                                  AtomicOrdering::AcquireRelease, DeviceScope);
    CI->replaceAllUsesWith(RMW);
    return Error::success();
  }

  // Image / TypedBuffer atomic via sentinel rewritten to OpImageTexelPointer +
  // OpAtomic* in the SPIR-V binary post-pass.
  uint32_t SpvAtomicOp = 234u; // OpAtomicIAdd
  switch (static_cast<ir::AtomicOp>(OpC->getZExtValue())) {
  case ir::AtomicOp::Add: SpvAtomicOp = 234u; break;
  case ir::AtomicOp::And: SpvAtomicOp = 240u; break;
  case ir::AtomicOp::Or:  SpvAtomicOp = 241u; break;
  case ir::AtomicOp::Xor: SpvAtomicOp = 242u; break;
  case ir::AtomicOp::SMin: SpvAtomicOp = 236u; break;
  case ir::AtomicOp::SMax: SpvAtomicOp = 238u; break;
  case ir::AtomicOp::UMin: SpvAtomicOp = 237u; break;
  case ir::AtomicOp::UMax: SpvAtomicOp = 239u; break;
  case ir::AtomicOp::Xchg: SpvAtomicOp = 229u; break;
  }
  unsigned Coords = textureCoordDims(RH.RB->Kind);
  SmallVector<Value *, 3> CoordVals;
  for (unsigned I = 0; I < Coords; ++I)
    CoordVals.push_back(CI->getArgOperand(3 + I));
  Value *Coord = packVector(B, B.getInt32Ty(), CoordVals);
  if (Val->getType()->isFloatTy())
    Val = B.CreateBitCast(Val, B.getInt32Ty());
  // Force LLVM SPIR-V to materialize the OpVariable for this image handle in
  // the entry function, then pass a keep-alive scalar to a sentinel helper
  // encoding (spvOp, set, binding) in its name.
  Intrinsic::ID DimID =
      (Coords == 1)
          ? Intrinsic::spv_resource_getdimensions_x
          : ((Coords == 2) ? Intrinsic::spv_resource_getdimensions_xy
                           : Intrinsic::spv_resource_getdimensions_xyz);
  Function *DimFn =
      Intrinsic::getOrInsertDeclaration(&M, DimID, {RH.SpvHandleTy});
  Value *DimVal = B.CreateCall(DimFn, {RH.SpvHandle});
  Value *KeepAlive =
      (Coords == 1) ? DimVal : B.CreateExtractElement(DimVal, B.getInt32(0));

  auto Key = std::make_tuple(SpvAtomicOp, RH.RB->SPIRVSet, RH.RB->SPIRVBinding,
                             Coord->getType());
  Function *&HelperFn = ImgAtomicHelpers[Key];
  if (!HelperFn) {
    Type *I32Ty = B.getInt32Ty();
    FunctionType *FT = FunctionType::get(
        I32Ty, {Coord->getType(), I32Ty, I32Ty, I32Ty, I32Ty, I32Ty}, false);
    std::string FnName = "__crucible_spv_img_atomic." +
                         std::to_string(SpvAtomicOp) + "." +
                         std::to_string(RH.RB->SPIRVSet) + "." +
                         std::to_string(RH.RB->SPIRVBinding) + "." +
                         std::to_string(ImgAtomicHelpers.size());
    HelperFn = Function::Create(FT, GlobalValue::InternalLinkage, FnName, M);
    HelperFn->addFnAttr(Attribute::NoInline);
    HelperFn->addFnAttr(Attribute::OptimizeNone);
    BasicBlock *HBB = BasicBlock::Create(Ctx, "entry", HelperFn);
    IRBuilder<> HB(HBB);
    HB.CreateRet(HelperFn->getArg(5));
  }
  uint32_t Scope = 1u;   // Device
  uint32_t Sem = 0x808u; // AcquireRelease | ImageMemory
  Value *Old = B.CreateCall(HelperFn, {Coord, B.getInt32(0), B.getInt32(Scope),
                                       B.getInt32(Sem), Val, KeepAlive});
  if (CI->getType()->isFloatTy())
    Old = B.CreateBitCast(Old, CI->getType());
  CI->replaceAllUsesWith(Old);
  return Error::success();
}

Error SPIRVLowerer::lowerAtomicCompareExchange(CallInst *CI) {
  auto RHOrErr = resolveHandle(CI->getArgOperand(1));
  if (!RHOrErr)
    return RHOrErr.takeError();
  const ResolvedHandle &RH = *RHOrErr;

  IRBuilder<> B(CI);
  Value *Ptr = nullptr;
  unsigned AS = 11;
  if (RH.RB->Kind == ResourceKind::RawBuffer ||
      RH.RB->Kind == ResourceKind::StructuredBuffer) {
    Value *ByteOff = computeBufferByteOffset(B, *RH.RB, CI->getArgOperand(2),
                                             CI->getArgOperand(3));
    Value *WordIdx = B.CreateLShr(ByteOff, B.getInt32(2));
    Ptr = getStorageBufferWordPtr(B, RH, WordIdx);
    AS = 11;
  } else {
    unsigned Coords = textureCoordDims(RH.RB->Kind);
    SmallVector<Value *, 3> CoordVals;
    for (unsigned I = 0; I < Coords; ++I)
      CoordVals.push_back(CI->getArgOperand(2 + I));
    Value *Coord = packVector(B, B.getInt32Ty(), CoordVals);
    Value *Cmp = CI->getArgOperand(5);
    Value *Val = CI->getArgOperand(6);
    if (Cmp->getType()->isFloatTy())
      Cmp = B.CreateBitCast(Cmp, B.getInt32Ty());
    if (Val->getType()->isFloatTy())
      Val = B.CreateBitCast(Val, B.getInt32Ty());
    Intrinsic::ID DimID =
        (Coords == 1)
            ? Intrinsic::spv_resource_getdimensions_x
            : ((Coords == 2) ? Intrinsic::spv_resource_getdimensions_xy
                             : Intrinsic::spv_resource_getdimensions_xyz);
    Function *DimFn =
        Intrinsic::getOrInsertDeclaration(&M, DimID, {RH.SpvHandleTy});
    Value *DimVal = B.CreateCall(DimFn, {RH.SpvHandle});
    Value *KeepAlive =
        (Coords == 1) ? DimVal : B.CreateExtractElement(DimVal, B.getInt32(0));

    auto Key = std::make_tuple(RH.RB->SPIRVSet, RH.RB->SPIRVBinding,
                               Coord->getType());
    Function *&HelperFn = ImgCmpXchgHelpers[Key];
    if (!HelperFn) {
      Type *I32Ty = B.getInt32Ty();
      FunctionType *FT = FunctionType::get(
          I32Ty,
          {Coord->getType(), I32Ty, I32Ty, I32Ty, I32Ty, I32Ty, I32Ty, I32Ty},
          false);
      std::string FnName = "__crucible_spv_img_cmpxchg." +
                           std::to_string(RH.RB->SPIRVSet) + "." +
                           std::to_string(RH.RB->SPIRVBinding) + "." +
                           std::to_string(ImgCmpXchgHelpers.size());
      HelperFn = Function::Create(FT, GlobalValue::InternalLinkage, FnName, M);
      HelperFn->addFnAttr(Attribute::NoInline);
      HelperFn->addFnAttr(Attribute::OptimizeNone);
      BasicBlock *HBB = BasicBlock::Create(Ctx, "entry", HelperFn);
      IRBuilder<> HB(HBB);
      HB.CreateRet(HelperFn->getArg(7));
    }
    uint32_t Scope = 1u;       // Device
    uint32_t EqSem = 0x808u;   // AcquireRelease | ImageMemory
    uint32_t UneqSem = 0x802u; // Acquire | ImageMemory
    Value *Old = B.CreateCall(
        HelperFn, {Coord, B.getInt32(0), B.getInt32(Scope), B.getInt32(EqSem),
                   B.getInt32(UneqSem), Val, Cmp, KeepAlive});
    if (CI->getType()->isFloatTy())
      Old = B.CreateBitCast(Old, CI->getType());
    CI->replaceAllUsesWith(Old);
    return Error::success();
  }
  Value *Cmp = CI->getArgOperand(5);
  Value *Val = CI->getArgOperand(6);
  Function *&HelperFn = CmpXchgHelpers[AS];
  if (!HelperFn) {
    Type *I32Ty = Type::getInt32Ty(Ctx);
    Type *PtrTy = PointerType::get(Ctx, AS);
    FunctionType *FT = FunctionType::get(
        I32Ty, {PtrTy, I32Ty, I32Ty, I32Ty, I32Ty, I32Ty}, false);
    std::string FnName = "__crucible_spv_cmpxchg." + std::to_string(AS);
    HelperFn = Function::Create(FT, GlobalValue::InternalLinkage, FnName, M);
    HelperFn->addFnAttr(Attribute::NoInline);
    HelperFn->addFnAttr(Attribute::OptimizeNone);
    BasicBlock *HBB = BasicBlock::Create(Ctx, "entry", HelperFn);
    IRBuilder<> HB(HBB);
    Value *Dummy = HB.CreateAlignedLoad(I32Ty, HelperFn->getArg(0), Align(4));
    HB.CreateRet(Dummy);
  }
  uint32_t Scope = 1u;  // Device
  uint32_t MemSem = 0x40u; // UniformMemory
  uint32_t EqSem = 0x8u /* AcquireRelease */ | MemSem;
  uint32_t UneqSem = 0x2u /* Acquire */ | MemSem;
  Value *Old = B.CreateCall(
      HelperFn, {Ptr, B.getInt32(Scope), B.getInt32(EqSem),
                 B.getInt32(UneqSem), Val, Cmp});
  CI->replaceAllUsesWith(Old);
  return Error::success();
}

Error SPIRVLowerer::lowerBarrier(CallInst *CI) {
  // (op, execSync, tgsmFence, deviceFence)
  IRBuilder<> B(CI);
  bool Sync = !cast<ConstantInt>(CI->getArgOperand(1))->isZero();
  bool Group = !cast<ConstantInt>(CI->getArgOperand(2))->isZero();
  bool Device = !cast<ConstantInt>(CI->getArgOperand(3))->isZero();

  Intrinsic::ID BarID = Intrinsic::spv_all_memory_barrier_with_group_sync;
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

  Function *BarFn = Intrinsic::getOrInsertDeclaration(&M, BarID);
  B.CreateCall(BarFn);
  return Error::success();
}

Error SPIRVLowerer::lowerWaveOp(CallInst *CI, SpvOp Op) {
  IRBuilder<> B(CI);
  Value *Res = nullptr;

  switch (Op) {
  case SpvOp::WaveIsFirstLane: {
    Function *Fn =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::spv_wave_is_first_lane);
    Res = B.CreateCall(Fn);
    break;
  }
  case SpvOp::LaneId: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_subgroup_local_invocation_id);
    Res = B.CreateCall(Fn);
    break;
  }
  case SpvOp::SubgroupSize: {
    Function *Fn =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::spv_subgroup_size);
    Res = B.CreateCall(Fn);
    break;
  }
  case SpvOp::WaveVote: {
    auto Kind = static_cast<ir::WaveVoteKind>(
        cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue());
    Value *Val = CI->getArgOperand(1);
    if (Kind == ir::WaveVoteKind::Any) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::spv_wave_any);
      Res = B.CreateCall(Fn, {Val});
    } else if (Kind == ir::WaveVoteKind::All) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::spv_wave_all);
      Res = B.CreateCall(Fn, {Val});
    } else {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::spv_wave_all_equal, {Val->getType()});
      Res = B.CreateCall(Fn, {Val});
    }
    break;
  }
  case SpvOp::WaveBallot: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_subgroup_ballot);
    Res = B.CreateCall(Fn, {CI->getArgOperand(1)});
    break;
  }
  case SpvOp::WaveReadLaneAt: {
    Value *Val = CI->getArgOperand(1);
    Value *Lane = CI->getArgOperand(2);
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_wave_readlane, {Val->getType()});
    Res = B.CreateCall(Fn, {Val, Lane});
    break;
  }
  case SpvOp::WaveReadLaneFirst: {
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
  case SpvOp::WaveReduce: {
    // (op, val, allReduceOp, isExclusive)
    Value *Val = CI->getArgOperand(1);
    auto RedOp = static_cast<mlir::gpu::AllReduceOperation>(
        cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue());
    bool IsPrefix = !cast<ConstantInt>(CI->getArgOperand(3))->isZero();
    Intrinsic::ID ID = Intrinsic::spv_wave_reduce_sum;
    if (IsPrefix) {
      ID = (RedOp == mlir::gpu::AllReduceOperation::MUL)
               ? Intrinsic::spv_wave_prefix_product
               : Intrinsic::spv_wave_prefix_sum;
    } else {
      switch (RedOp) {
      case mlir::gpu::AllReduceOperation::ADD:
        ID = Intrinsic::spv_wave_reduce_sum;
        break;
      case mlir::gpu::AllReduceOperation::MUL:
        ID = Intrinsic::spv_wave_product;
        break;
      case mlir::gpu::AllReduceOperation::MINUI:
        ID = Intrinsic::spv_wave_reduce_umin;
        break;
      case mlir::gpu::AllReduceOperation::MINSI:
      case mlir::gpu::AllReduceOperation::MINNUMF:
      case mlir::gpu::AllReduceOperation::MINIMUMF:
        ID = Intrinsic::spv_wave_reduce_min;
        break;
      case mlir::gpu::AllReduceOperation::MAXUI:
        ID = Intrinsic::spv_wave_reduce_umax;
        break;
      case mlir::gpu::AllReduceOperation::MAXSI:
      case mlir::gpu::AllReduceOperation::MAXNUMF:
      case mlir::gpu::AllReduceOperation::MAXIMUMF:
        ID = Intrinsic::spv_wave_reduce_max;
        break;
      case mlir::gpu::AllReduceOperation::AND:
        ID = Intrinsic::spv_wave_reduce_and;
        break;
      case mlir::gpu::AllReduceOperation::OR:
        ID = Intrinsic::spv_wave_reduce_or;
        break;
      case mlir::gpu::AllReduceOperation::XOR:
        ID = Intrinsic::spv_wave_reduce_xor;
        break;
      }
    }
    Function *Fn =
        Intrinsic::getOrInsertDeclaration(&M, ID, {Val->getType()});
    Res = B.CreateCall(Fn, {Val});
    break;
  }
  case SpvOp::WaveBitCount:
  case SpvOp::WaveMultiPrefix: {
    if (Op == SpvOp::WaveBitCount) {
      auto Kind = static_cast<ir::WaveBitCountKind>(
          cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue());
      if (Kind == ir::WaveBitCountKind::All) {
        Function *Fn = Intrinsic::getOrInsertDeclaration(
            &M, Intrinsic::spv_wave_active_countbits);
        Res = B.CreateCall(Fn, {CI->getArgOperand(1)});
        break;
      }
      if (Kind == ir::WaveBitCountKind::Prefix) {
        Function *Fn = Intrinsic::getOrInsertDeclaration(
            &M, Intrinsic::spv_subgroup_prefix_bit_count);
        Res = B.CreateCall(Fn, {CI->getArgOperand(1)});
        break;
      }
    }
    // Multi-prefix (bit count or general op):
    // WaveBitCount(MultiPrefix): (op, cond, kind, m0, m1, m2, m3)
    // WaveMultiPrefix:           (op, val, opKind, m0, m1, m2, m3)
    Function *LaneFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_subgroup_local_invocation_id);
    Value *Lane = B.CreateCall(LaneFn);
    Function *BallotFn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_subgroup_ballot);
    Value *ActiveBallot = B.CreateCall(BallotFn, {B.getTrue()});
    Value *WordMasks[4];
    for (unsigned W = 0; W < 4; ++W) {
      Value *ActiveW = B.CreateExtractElement(ActiveBallot, B.getInt32(W));
      Value *RawMask = B.CreateAnd(CI->getArgOperand(3 + W), ActiveW);
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

    ir::WaveMultiPrefixOpKind OpKind = ir::WaveMultiPrefixOpKind::Sum;
    Value *Val = nullptr;
    if (Op == SpvOp::WaveBitCount) {
      Val = B.CreateZExt(CI->getArgOperand(1), B.getInt32Ty());
    } else {
      OpKind = static_cast<ir::WaveMultiPrefixOpKind>(
          cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue());
      Val = CI->getArgOperand(1);
    }
    Type *ValTy = Val->getType();

    Value *Acc = nullptr;
    switch (OpKind) {
    case ir::WaveMultiPrefixOpKind::Sum:
    case ir::WaveMultiPrefixOpKind::Or:
    case ir::WaveMultiPrefixOpKind::Xor:
      Acc = Constant::getNullValue(ValTy);
      break;
    case ir::WaveMultiPrefixOpKind::And:
      Acc = ConstantInt::getAllOnesValue(ValTy);
      break;
    case ir::WaveMultiPrefixOpKind::Product:
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
      case ir::WaveMultiPrefixOpKind::Sum:
        Combined = ValTy->isFloatingPointTy() ? B.CreateFAdd(AccPhi, Other)
                                              : B.CreateAdd(AccPhi, Other);
        break;
      case ir::WaveMultiPrefixOpKind::Product:
        Combined = ValTy->isFloatingPointTy() ? B.CreateFMul(AccPhi, Other)
                                              : B.CreateMul(AccPhi, Other);
        break;
      case ir::WaveMultiPrefixOpKind::And:
        Combined = B.CreateAnd(AccPhi, Other);
        break;
      case ir::WaveMultiPrefixOpKind::Or:
        Combined = B.CreateOr(AccPhi, Other);
        break;
      case ir::WaveMultiPrefixOpKind::Xor:
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
  case SpvOp::QuadOp: {
    Value *Val = CI->getArgOperand(1);
    auto Dir = static_cast<ir::QuadOpKind>(
        cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue());
    Intrinsic::ID ID = Intrinsic::spv_quad_read_across_x;
    if (Dir == ir::QuadOpKind::ReadAcrossY)
      ID = Intrinsic::spv_quad_read_across_y;
    else if (Dir == ir::QuadOpKind::ReadAcrossDiagonal)
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

Error SPIRVLowerer::lowerMathOrBitOp(CallInst *CI, SpvOp Op) {
  IRBuilder<> B(CI);
  Value *Res = nullptr;

  auto CallUnary = [&](Intrinsic::ID ID) {
    Value *X = CI->getArgOperand(1);
    Function *Fn = Intrinsic::getOrInsertDeclaration(&M, ID, {X->getType()});
    return B.CreateCall(Fn, {X});
  };

  switch (Op) {
  case SpvOp::Saturate:
    Res = CallUnary(Intrinsic::spv_saturate);
    break;
  case SpvOp::IsNaN: {
    Value *X = CI->getArgOperand(1);
    Res = B.CreateFCmpUNO(X, X);
    break;
  }
  case SpvOp::IsInf: {
    Value *X = CI->getArgOperand(1);
    Value *Abs = CallUnary(Intrinsic::fabs);
    Res = B.CreateFCmpOEQ(Abs, ConstantFP::getInfinity(X->getType()));
    break;
  }
  case SpvOp::IsFinite: {
    Value *X = CI->getArgOperand(1);
    Value *Abs = CallUnary(Intrinsic::fabs);
    Res = B.CreateFCmpOLT(Abs, ConstantFP::getInfinity(X->getType()));
    break;
  }
  case SpvOp::Tan:
    Res = CallUnary(Intrinsic::tan);
    break;
  case SpvOp::Acos:
    Res = CallUnary(Intrinsic::acos);
    break;
  case SpvOp::Asin:
    Res = CallUnary(Intrinsic::asin);
    break;
  case SpvOp::Atan:
    Res = CallUnary(Intrinsic::atan);
    break;
  case SpvOp::Cosh:
    Res = CallUnary(Intrinsic::cosh);
    break;
  case SpvOp::Sinh:
    Res = CallUnary(Intrinsic::sinh);
    break;
  case SpvOp::Tanh:
    Res = CallUnary(Intrinsic::tanh);
    break;
  case SpvOp::Frc:
    Res = CallUnary(Intrinsic::spv_frac);
    break;
  case SpvOp::Rsqrt:
    Res = CallUnary(Intrinsic::spv_rsqrt);
    break;
  case SpvOp::FirstBit: {
    Value *X = CI->getArgOperand(1);
    auto Kind = static_cast<ir::FirstBitKind>(
        cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue());
    if (Kind == ir::FirstBitKind::Lo) {
      Function *Fn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::spv_firstbitlow, {X->getType()});
      Res = B.CreateCall(Fn, {X});
    } else {
      unsigned Bits = X->getType()->getIntegerBitWidth();
      if (Kind == ir::FirstBitKind::SHi) {
        Value *IsNeg = B.CreateICmpSLT(X, ConstantInt::get(X->getType(), 0));
        X = B.CreateSelect(IsNeg, B.CreateNot(X), X);
      }
      Function *UMsb32 = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::spv_firstbituhigh, {B.getInt32Ty()});
      Value *Neg1 = B.getInt32(-1);
      if (Bits == 64) {
        Value *Lo = B.CreateTrunc(X, B.getInt32Ty());
        Value *Hi =
            B.CreateTrunc(B.CreateLShr(X, B.getInt64(32)), B.getInt32Ty());
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
    }
    break;
  }
  case SpvOp::Bfe: {
    Value *Width = CI->getArgOperand(1);
    Value *Offset = CI->getArgOperand(2);
    Value *Val = CI->getArgOperand(3);
    bool IsSigned = !cast<ConstantInt>(CI->getArgOperand(4))->isZero();
    Type *Ty = Val->getType();
    unsigned BitWidth = Ty->getIntegerBitWidth();
    Value *W = B.CreateAnd(Width, ConstantInt::get(Ty, BitWidth - 1));
    Value *O = B.CreateAnd(Offset, ConstantInt::get(Ty, BitWidth - 1));
    Value *IsZeroW = B.CreateICmpEQ(W, ConstantInt::get(Ty, 0));
    Value *ShlAmt = B.CreateSub(ConstantInt::get(Ty, BitWidth),
                                B.CreateAdd(W, O));
    Value *ShrAmt = B.CreateSub(ConstantInt::get(Ty, BitWidth), W);
    Value *Left = B.CreateShl(Val, ShlAmt);
    Value *Extracted = IsSigned ? B.CreateAShr(Left, ShrAmt)
                                : B.CreateLShr(Left, ShrAmt);
    Res = B.CreateSelect(IsZeroW, ConstantInt::get(Ty, 0), Extracted);
    break;
  }
  case SpvOp::Bfi: {
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
  case SpvOp::Dot: {
    unsigned N = (CI->arg_size() - 1) / 2;
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
  case SpvOp::LegacyF32ToF16: {
    Value *H = B.CreateFPTrunc(CI->getArgOperand(1), Type::getHalfTy(Ctx));
    Value *I16 = B.CreateBitCast(H, B.getInt16Ty());
    Res = B.CreateZExt(I16, B.getInt32Ty());
    break;
  }
  case SpvOp::LegacyF16ToF32: {
    Value *I16 = B.CreateTrunc(CI->getArgOperand(1), B.getInt16Ty());
    Value *H = B.CreateBitCast(I16, Type::getHalfTy(Ctx));
    Res = B.CreateFPExt(H, Type::getFloatTy(Ctx));
    break;
  }
  case SpvOp::DotPacked: {
    auto Kind = static_cast<ir::DotPackedKind>(
        cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue());
    if (Kind == ir::DotPackedKind::F16x2) {
      Type *F32 = Type::getFloatTy(Ctx);
      Value *Acc = CI->getArgOperand(2);
      Value *A0 = B.CreateFPExt(CI->getArgOperand(3), F32);
      Value *A1 = B.CreateFPExt(CI->getArgOperand(4), F32);
      Value *B0 = B.CreateFPExt(CI->getArgOperand(5), F32);
      Value *B1 = B.CreateFPExt(CI->getArgOperand(6), F32);
      Value *Prod = B.CreateFAdd(B.CreateFMul(A0, B0), B.CreateFMul(A1, B1));
      Res = B.CreateFAdd(Acc, Prod);
    } else {
      Intrinsic::ID ID = Kind == ir::DotPackedKind::I8x4
                             ? Intrinsic::spv_dot4add_i8packed
                             : Intrinsic::spv_dot4add_u8packed;
      Function *Fn = Intrinsic::getOrInsertDeclaration(&M, ID);
      Res = B.CreateCall(
          Fn, {CI->getArgOperand(2), CI->getArgOperand(3), CI->getArgOperand(4)});
    }
    break;
  }
  case SpvOp::Pack4x8: {
    auto *ModeC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    uint64_t Mode = ModeC ? ModeC->getZExtValue() : 0;
    Value *Acc = B.getInt32(0);
    for (unsigned I = 0; I < 4; ++I) {
      Value *V = CI->getArgOperand(2 + I);
      Type *ElemTy = V->getType();
      if (Mode == 1) {
        Value *Lo = ConstantInt::get(ElemTy, 0);
        Value *Hi = ConstantInt::get(ElemTy, 255);
        V = B.CreateSelect(B.CreateICmpSLT(V, Lo), Lo, V);
        V = B.CreateSelect(B.CreateICmpSGT(V, Hi), Hi, V);
      } else if (Mode == 2) {
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
  case SpvOp::Unpack4x8: {
    auto *ModeC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    uint64_t Mode = ModeC ? ModeC->getZExtValue() : 0;
    Value *Pk = CI->getArgOperand(2);
    auto *VT = cast<FixedVectorType>(CI->getType());
    Type *ElemTy = VT->getElementType();
    Value *Ret = PoisonValue::get(VT);
    for (unsigned I = 0; I < 4; ++I) {
      Value *Shifted = (I == 0) ? Pk : B.CreateLShr(Pk, B.getInt32(I * 8));
      Value *Byte = B.CreateTrunc(Shifted, B.getInt8Ty());
      Value *Ext = (Mode == 1) ? B.CreateSExt(Byte, ElemTy)
                               : B.CreateZExt(Byte, ElemTy);
      Ret = B.CreateInsertElement(Ret, Ext, B.getInt32(I));
    }
    Res = Ret;
    break;
  }
  case SpvOp::MakeDouble: {
    Value *Lo = B.CreateZExt(CI->getArgOperand(1), B.getInt64Ty());
    Value *Hi = B.CreateShl(B.CreateZExt(CI->getArgOperand(2), B.getInt64Ty()),
                            B.getInt64(32));
    Res = B.CreateBitCast(B.CreateOr(Lo, Hi), Type::getDoubleTy(Ctx));
    break;
  }
  case SpvOp::SplitDouble: {
    Value *Bits = B.CreateBitCast(CI->getArgOperand(1), B.getInt64Ty());
    Value *Lo = B.CreateTrunc(Bits, B.getInt32Ty());
    Value *Hi = B.CreateTrunc(B.CreateLShr(Bits, B.getInt64(32)), B.getInt32Ty());
    Value *Ret = PoisonValue::get(CI->getType());
    Ret = B.CreateInsertValue(Ret, Lo, {0});
    Ret = B.CreateInsertValue(Ret, Hi, {1});
    Res = Ret;
    break;
  }
  case SpvOp::UAddc: {
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
  default:
    return makeError("SPIR-V backend: unsupported math/bit op {0}",
                     static_cast<unsigned>(Op));
  }

  CI->replaceAllUsesWith(Res);
  return Error::success();
}

Error SPIRVLowerer::lowerPixelOp(CallInst *CI, SpvOp Op) {
  IRBuilder<> B(CI);
  switch (Op) {
  case SpvOp::Discard: {
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
  case SpvOp::Deriv: {
    auto Axis = static_cast<ir::DerivAxis>(
        cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue());
    auto Qual = static_cast<ir::DerivQuality>(
        cast<ConstantInt>(CI->getArgOperand(3))->getZExtValue());
    Intrinsic::ID ID =
        (Qual == ir::DerivQuality::Coarse)
            ? (Axis == ir::DerivAxis::X ? Intrinsic::spv_ddx_coarse
                                        : Intrinsic::spv_ddy_coarse)
            : (Axis == ir::DerivAxis::X ? Intrinsic::spv_ddx_fine
                                        : Intrinsic::spv_ddy_fine);
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

Error SPIRVLowerer::lowerBridgeCall(CallInst *CI) {
  auto *OpC = dyn_cast<ConstantInt>(CI->getArgOperand(0));
  if (!OpC)
    return makeError("SPIR-V backend: non-constant bridge opcode in call to '{0}'",
                     CI->getCalledFunction()->getName());

  auto Op = static_cast<SpvOp>(OpC->getZExtValue());
  IRBuilder<> B(CI);

  switch (Op) {
  case SpvOp::BindResource:
    return lowerBindResource(CI);
  case SpvOp::HeapResource:
    return lowerHeapResource(CI);
  case SpvOp::BufferLoad:
    return lowerBufferLoad(CI);
  case SpvOp::BufferStore:
    return lowerBufferStore(CI);
  case SpvOp::CBufferLoad:
    return lowerCBufferLoad(CI);
  case SpvOp::BufferUpdateCounter:
    return lowerBufferUpdateCounter(CI);
  case SpvOp::TexLoad:
    return lowerTextureLoad(CI);
  case SpvOp::TexStore:
    return lowerTextureStore(CI);
  case SpvOp::TexSample:
    return lowerTextureSample(CI);
  case SpvOp::TexGather:
    return lowerTextureGather(CI);
  case SpvOp::TexCalcLod:
    return lowerCalculateLOD(CI);
  case SpvOp::TexGetDimensions:
    return lowerGetDimensions(CI);
  case SpvOp::BufferAtomic:
    return lowerAtomicBinOp(CI);
  case SpvOp::BufferAtomicCmpXchg:
    return lowerAtomicCompareExchange(CI);
  case SpvOp::Barrier:
    return lowerBarrier(CI);

  case SpvOp::ThreadId: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_thread_id, {B.getInt32Ty()});
    CI->replaceAllUsesWith(B.CreateCall(Fn, {CI->getArgOperand(1)}));
    return Error::success();
  }
  case SpvOp::GroupId: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_group_id, {B.getInt32Ty()});
    CI->replaceAllUsesWith(B.CreateCall(Fn, {CI->getArgOperand(1)}));
    return Error::success();
  }
  case SpvOp::ThreadIdInGroup: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_thread_id_in_group, {B.getInt32Ty()});
    CI->replaceAllUsesWith(B.CreateCall(Fn, {CI->getArgOperand(1)}));
    return Error::success();
  }
  case SpvOp::FlattenedThreadIdInGroup: {
    Function *Fn = Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::spv_flattened_thread_id_in_group);
    CI->replaceAllUsesWith(B.CreateCall(Fn));
    return Error::success();
  }

  case SpvOp::WaveIsFirstLane:
  case SpvOp::LaneId:
  case SpvOp::SubgroupSize:
  case SpvOp::WaveVote:
  case SpvOp::WaveBallot:
  case SpvOp::WaveReadLaneAt:
  case SpvOp::WaveReadLaneFirst:
  case SpvOp::WaveReduce:
  case SpvOp::WaveBitCount:
  case SpvOp::WaveMultiPrefix:
  case SpvOp::QuadOp:
    return lowerWaveOp(CI, Op);

  case SpvOp::Saturate:
  case SpvOp::IsNaN:
  case SpvOp::IsInf:
  case SpvOp::IsFinite:
  case SpvOp::Tan:
  case SpvOp::Acos:
  case SpvOp::Asin:
  case SpvOp::Atan:
  case SpvOp::Cosh:
  case SpvOp::Sinh:
  case SpvOp::Tanh:
  case SpvOp::Frc:
  case SpvOp::Rsqrt:
  case SpvOp::FirstBit:
  case SpvOp::Bfe:
  case SpvOp::Bfi:
  case SpvOp::Dot:
  case SpvOp::LegacyF32ToF16:
  case SpvOp::LegacyF16ToF32:
  case SpvOp::DotPacked:
  case SpvOp::Pack4x8:
  case SpvOp::Unpack4x8:
  case SpvOp::MakeDouble:
  case SpvOp::SplitDouble:
  case SpvOp::UAddc:
    return lowerMathOrBitOp(CI, Op);

  case SpvOp::Discard:
  case SpvOp::Deriv:
    return lowerPixelOp(CI, Op);

  case SpvOp::LoadInput:
    return lowerLoadInput(CI);
  case SpvOp::StoreOutput:
    return lowerStoreOutput(CI);
  case SpvOp::EmitStream: {
    IRBuilder<> B(CI);
    if (!EmitVertexHelper) {
      FunctionType *FT = FunctionType::get(Type::getVoidTy(Ctx), {}, false);
      EmitVertexHelper = Function::Create(FT, GlobalValue::InternalLinkage,
                                          "__crucible_spv_emit_vertex", M);
      EmitVertexHelper->addFnAttr(Attribute::NoInline);
      EmitVertexHelper->addFnAttr(Attribute::OptimizeNone);
      BasicBlock *HBB = BasicBlock::Create(Ctx, "entry", EmitVertexHelper);
      IRBuilder<>(HBB).CreateRetVoid();
    }
    B.CreateCall(EmitVertexHelper);
    return Error::success();
  }
  case SpvOp::CutStream: {
    IRBuilder<> B(CI);
    if (!EndPrimitiveHelper) {
      FunctionType *FT = FunctionType::get(Type::getVoidTy(Ctx), {}, false);
      EndPrimitiveHelper = Function::Create(FT, GlobalValue::InternalLinkage,
                                            "__crucible_spv_end_primitive", M);
      EndPrimitiveHelper->addFnAttr(Attribute::NoInline);
      EndPrimitiveHelper->addFnAttr(Attribute::OptimizeNone);
      BasicBlock *HBB = BasicBlock::Create(Ctx, "entry", EndPrimitiveHelper);
      IRBuilder<>(HBB).CreateRetVoid();
    }
    B.CreateCall(EndPrimitiveHelper);
    return Error::success();
  }
  case SpvOp::LoadControlPoint:
    return lowerLoadControlPoint(CI);
  case SpvOp::LoadPatchConstant:
    return lowerLoadPatchConstant(CI);
  case SpvOp::StorePatchConstant:
    return lowerStorePatchConstant(CI);

  case SpvOp::RayQueryAlloc:
  case SpvOp::RayQueryTraceInline:
  case SpvOp::RayQueryProceed:
  case SpvOp::RayQueryAbort:
  case SpvOp::RayQueryCommit:
  case SpvOp::RayQueryGet:
  case SpvOp::SysValRT:
  case SpvOp::TraceRay:
    return lowerRaytracingOp(CI, Op);

  default:
    return makeError("SPIR-V backend: unsupported bridge op '{0}' (opcode {1})",
                     CI->getCalledFunction()->getName(),
                     static_cast<unsigned>(Op));
  }
}

void SPIRVLowerer::mergeHullPatchConstantFunc() {
  if (Reflect.Stage != ShaderStage::Hull || !Reflect.Hull)
    return;
  Function *PCF = nullptr;
  if (!Reflect.Hull->PatchConstantFuncName.empty())
    PCF = M.getFunction(Reflect.Hull->PatchConstantFuncName);
  if (!PCF) {
    for (Function &F : M) {
      if (!F.isDeclaration() && &F != Entry &&
          !F.getName().starts_with("__crucible_")) {
        PCF = &F;
        break;
      }
    }
  }
  if (!PCF || PCF->isDeclaration())
    return;

  SmallVector<ReturnInst *, 4> Rets;
  for (BasicBlock &BB : *Entry) {
    if (auto *RI = dyn_cast<ReturnInst>(BB.getTerminator()))
      Rets.push_back(RI);
  }
  if (Rets.empty())
    return;

  BasicBlock *BarBB = BasicBlock::Create(Ctx, "hs.barrier", Entry);
  BasicBlock *PcfBB = BasicBlock::Create(Ctx, "hs.patch_const", Entry);
  BasicBlock *ExitBB = BasicBlock::Create(Ctx, "hs.exit", Entry);

  for (ReturnInst *RI : Rets) {
    IRBuilder<>(RI).CreateBr(BarBB);
    RI->eraseFromParent();
  }

  IRBuilder<> BB(BarBB);
  Function *BarFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::spv_group_memory_barrier_with_group_sync);
  BB.CreateCall(BarFn);
  GlobalVariable *InvGV = getOrCreateBuiltInInput(8u /* InvocationId */,
                                                  BB.getInt32Ty(),
                                                  "in_SV_OutputControlPointID");
  Value *InvId = BB.CreateLoad(BB.getInt32Ty(), InvGV);
  Value *IsZero = BB.CreateICmpEQ(InvId, BB.getInt32(0));
  BB.CreateCondBr(IsZero, PcfBB, ExitBB);

  IRBuilder<> PB(PcfBB);
  CallInst *CallPCF = PB.CreateCall(PCF);
  PB.CreateBr(ExitBB);

  IRBuilder<>(ExitBB).CreateRetVoid();

  InlineFunctionInfo IFI;
  InlineFunction(*CallPCF, IFI);
  if (PCF->use_empty())
    PCF->eraseFromParent();
}

void SPIRVLowerer::prepareStageInterface() {
  if (Reflect.Stage != ShaderStage::Vertex &&
      Reflect.Stage != ShaderStage::Geometry &&
      Reflect.Stage != ShaderStage::Hull &&
      Reflect.Stage != ShaderStage::Domain &&
      Reflect.Stage != ShaderStage::Pixel)
    return;

  uint32_t InputVerts = 0;
  uint32_t OutputVerts = 0;
  if (Reflect.Stage == ShaderStage::Geometry) {
    uint32_t InPrim = Reflect.Geometry ? Reflect.Geometry->InputPrimitive : 3u;
    switch (InPrim) {
    case 1:
      InputVerts = 1;
      break; // Point
    case 2:
      InputVerts = 2;
      break; // Line
    case 6:
      InputVerts = 4;
      break; // LineAdj
    case 7:
      InputVerts = 6;
      break; // TriangleAdj
    default:
      InputVerts = 3;
      break; // Triangle
    }
  } else if (Reflect.Stage == ShaderStage::Hull && Reflect.Hull) {
    InputVerts = std::max(1u, Reflect.Hull->InputControlPointCount);
    OutputVerts = std::max(1u, Reflect.Hull->OutputControlPointCount);
  } else if (Reflect.Stage == ShaderStage::Domain && Reflect.Domain) {
    InputVerts = std::max(1u, Reflect.Domain->ControlPointCount);
  }

  uint32_t LastInputNextLoc = 0;
  uint32_t LastOutputNextLoc = 0;
  auto buildStageVars = [&](const std::vector<StageParameter> &Params,
                            unsigned AddrSpace, bool IsOutput,
                            DenseMap<uint32_t, StageVar> &Table,
                            uint32_t PerVertCount, bool IsPatchConst,
                            uint32_t StartLoc) -> uint32_t {
    uint32_t NextLoc = StartLoc;
    for (const StageParameter &P : Params) {
      StageVar SV;
      SV.Param = &P;

      std::optional<uint32_t> BuiltIn;
      bool IsBoolBI = false;
      bool IsTessOuter = false;
      bool IsTessInner = false;
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
        UsedViewportIndex = true;
        if (IsOutput && (Reflect.Stage == ShaderStage::Vertex ||
                         Reflect.Stage == ShaderStage::Domain))
          UsedShaderViewportIndex = true;
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
      case 15: // Coverage
        BuiltIn = 20; // SampleMask
        break;
      case 17: // Depth
        BuiltIn = 22; // FragDepth
        if (IsOutput)
          DepthOutputMode = 1u;
        break;
      case 18: // DepthLessEqual
        BuiltIn = 22; // FragDepth
        if (IsOutput)
          DepthOutputMode = 3u;
        break;
      case 19: // DepthGreaterEqual
        BuiltIn = 22; // FragDepth
        if (IsOutput)
          DepthOutputMode = 2u;
        break;
      case 20: // StencilRef
        BuiltIn = 5014; // FragStencilRefEXT
        if (IsOutput)
          UsedStencilExport = true;
        break;
      case 25: // TessFactor
        BuiltIn = 11; // TessLevelOuter
        IsTessOuter = true;
        break;
      case 26: // InsideTessFactor
        BuiltIn = 12; // TessLevelInner
        IsTessInner = true;
        break;
      default:
        break;
      }

      SV.IsBoolBuiltIn = IsBoolBI;
      SV.IsFlatFloatArray = (P.Kind == 6 || P.Kind == 7); // Clip/CullDistance
      if (P.Kind == 6)
        UsedClipDistance = true;
      else if (P.Kind == 7)
        UsedCullDistance = true;
      Type *ScalarTy =
          IsBoolBI ? static_cast<Type *>(Type::getInt1Ty(Ctx))
                   : getComponentLLVMType(P.Type);
      unsigned Cols = std::max(1u, P.Columns);
      unsigned Rows = std::max(1u, P.Rows);
      Type *VarTy = nullptr;
      if (IsTessOuter) {
        VarTy = ArrayType::get(Type::getFloatTy(Ctx), 4);
      } else if (IsTessInner) {
        VarTy = ArrayType::get(Type::getFloatTy(Ctx), 2);
      } else if (SV.IsFlatFloatArray) {
        VarTy = ArrayType::get(ScalarTy, Rows * Cols);
      } else {
        Type *RowTy =
            (Cols == 1) ? ScalarTy : FixedVectorType::get(ScalarTy, Cols);
        VarTy = (Rows == 1) ? RowTy : ArrayType::get(RowTy, Rows);
      }
      bool IsPerVertex =
          !IsPatchConst && PerVertCount > 0 && P.Kind != 10 /* PrimitiveID */ &&
          P.Kind != 11 /* GSInstanceID */;
      if (IsPerVertex) {
        VarTy = ArrayType::get(VarTy, PerVertCount);
        SV.IsPerVertexArray = true;
      }
      SV.VarTy = VarTy;

      std::string Name =
          (IsPatchConst ? (IsOutput ? "out_patch_" : "in_patch_")
                        : (IsOutput ? "out_" : "in_")) +
          (P.Semantic.empty() ? ("sig_" + std::to_string(P.ID)) : P.Semantic) +
          "_" + std::to_string(P.SemanticIndex);
      auto *GV = new GlobalVariable(
          M, VarTy, /*isConstant=*/false, GlobalValue::InternalLinkage,
          PoisonValue::get(VarTy), Name, /*InsertBefore=*/nullptr,
          GlobalValue::NotThreadLocal, AddrSpace);
      SV.GV = GV;

      if (BuiltIn) {
        SV.Decorations.push_back({11u /* BuiltIn */, *BuiltIn});
        if (Reflect.Stage == ShaderStage::Pixel && !IsOutput &&
            (*BuiltIn == 9u /* Layer */ || *BuiltIn == 10u /* ViewportIndex */))
          SV.Decorations.push_back({14u /* Flat */});
      } else if (P.Kind == 16 /* Target */) {
        SV.Decorations.push_back({30u /* Location */, P.SemanticIndex});
      } else {
        SV.Decorations.push_back({30u /* Location */, NextLoc});
        NextLoc += Rows;
      }
      if (IsPatchConst)
        SV.Decorations.push_back({15u /* Patch */});

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
    return NextLoc;
  };

  LastInputNextLoc = buildStageVars(Reflect.Inputs, 7 /* Input */, false,
                                    InputVars, InputVerts, false, 0);
  LastOutputNextLoc = buildStageVars(Reflect.Outputs, 8 /* Output */, true,
                                     OutputVars, OutputVerts, false, 0);
  if (Reflect.Stage == ShaderStage::Hull && Reflect.Hull) {
    buildStageVars(Reflect.Hull->PatchConstants, 8 /* Output */, true,
                   PatchConstVars, 0, true, LastOutputNextLoc);
  } else if (Reflect.Stage == ShaderStage::Domain && Reflect.Domain) {
    buildStageVars(Reflect.Domain->PatchConstants, 7 /* Input */, false,
                   PatchConstVars, 0, true, LastInputNextLoc);
  }
}

void SPIRVLowerer::emitStageDecorations() {
  if (InputVars.empty() && OutputVars.empty() && PatchConstVars.empty() &&
      ExtraBuiltInInputs.empty())
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
  emitForTable(PatchConstVars);
  emitForTable(ExtraBuiltInInputs);
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
  if (SV.IsPerVertexArray) {
    Value *VertIdx =
        (CI->arg_size() > 4) ? CI->getArgOperand(4) : B.getInt32(0);
    auto *OuterAT = cast<ArrayType>(RowTy);
    Ptr = B.CreateInBoundsGEP(OuterAT, Ptr, {B.getInt32(0), VertIdx});
    RowTy = OuterAT->getElementType();
  }
  if (SV.IsFlatFloatArray) {
    unsigned Cols = std::max(1u, SV.Param->Columns);
    unsigned Rows = std::max(1u, SV.Param->Rows);
    Value *RowIdx = CI->getArgOperand(2);
    Value *ColIdx = CI->getArgOperand(3);
    if (ColIdx->getType() != B.getInt32Ty())
      ColIdx = B.CreateZExtOrTrunc(ColIdx, B.getInt32Ty());
    Value *FlatIdx =
        (Rows == 1) ? ColIdx
                    : B.CreateAdd(B.CreateMul(RowIdx, B.getInt32(Cols)), ColIdx);
    Ptr = B.CreateInBoundsGEP(RowTy, Ptr, {B.getInt32(0), FlatIdx});
    RowTy = cast<ArrayType>(RowTy)->getElementType();
  } else if (auto *AT = dyn_cast<ArrayType>(RowTy)) {
    Value *RowIdx = CI->getArgOperand(2);
    Ptr = B.CreateInBoundsGEP(RowTy, Ptr, {B.getInt32(0), RowIdx});
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
  SmallVector<Value *, 4> Indices;
  Indices.push_back(B.getInt32(0));
  if (SV.IsPerVertexArray) {
    GlobalVariable *InvGV = getOrCreateBuiltInInput(
        8u /* InvocationId */, B.getInt32Ty(), "in_SV_OutputControlPointID");
    Indices.push_back(B.CreateLoad(B.getInt32Ty(), InvGV));
    ElemTy = cast<ArrayType>(ElemTy)->getElementType();
  }
  if (SV.IsFlatFloatArray) {
    unsigned Cols = std::max(1u, SV.Param->Columns);
    unsigned Rows = std::max(1u, SV.Param->Rows);
    Value *RowIdx = CI->getArgOperand(2);
    Value *ColIdx = CI->getArgOperand(3);
    if (ColIdx->getType() != B.getInt32Ty())
      ColIdx = B.CreateZExtOrTrunc(ColIdx, B.getInt32Ty());
    Value *FlatIdx =
        (Rows == 1) ? ColIdx
                    : B.CreateAdd(B.CreateMul(RowIdx, B.getInt32(Cols)), ColIdx);
    Indices.push_back(FlatIdx);
    ElemTy = cast<ArrayType>(ElemTy)->getElementType();
  } else {
    if (auto *AT = dyn_cast<ArrayType>(ElemTy)) {
      Indices.push_back(CI->getArgOperand(2));
      ElemTy = AT->getElementType();
    }
    if (auto *VT = dyn_cast<FixedVectorType>(ElemTy)) {
      Indices.push_back(CI->getArgOperand(3));
      ElemTy = VT->getElementType();
    }
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

Error SPIRVLowerer::lowerLoadControlPoint(CallInst *CI) {
  auto *IdC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  if (!IdC)
    return makeError("SPIR-V backend: non-constant inputSigId in loadControlPoint");
  auto It = OutputVars.find(static_cast<uint32_t>(IdC->getZExtValue()));
  if (It == OutputVars.end())
    return makeError("SPIR-V backend: unknown control point ID {0}",
                     IdC->getZExtValue());
  StageVar &SV = It->second;
  SV.Referenced = true;

  IRBuilder<> B(CI);
  Value *Ptr = SV.GV;
  Type *RowTy = SV.VarTy;
  if (SV.IsPerVertexArray) {
    Value *CpIdx = CI->getArgOperand(4);
    auto *OuterAT = cast<ArrayType>(RowTy);
    Ptr = B.CreateInBoundsGEP(OuterAT, Ptr, {B.getInt32(0), CpIdx});
    RowTy = OuterAT->getElementType();
  }
  if (auto *AT = dyn_cast<ArrayType>(RowTy)) {
    Ptr = B.CreateInBoundsGEP(RowTy, Ptr, {B.getInt32(0), CI->getArgOperand(2)});
    RowTy = AT->getElementType();
  }
  Value *Loaded = B.CreateLoad(RowTy, Ptr);
  Value *Val = Loaded;
  if (isa<FixedVectorType>(RowTy))
    Val = B.CreateExtractElement(Loaded, CI->getArgOperand(3));
  if (Val->getType() != CI->getType()) {
    if (Val->getType()->isIntegerTy() && CI->getType()->isIntegerTy())
      Val = B.CreateZExtOrTrunc(Val, CI->getType());
    else
      Val = B.CreateBitCast(Val, CI->getType());
  }
  CI->replaceAllUsesWith(Val);
  return Error::success();
}

Error SPIRVLowerer::lowerLoadPatchConstant(CallInst *CI) {
  auto *IdC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  if (!IdC)
    return makeError("SPIR-V backend: non-constant id in loadPatchConstant");
  auto It = PatchConstVars.find(static_cast<uint32_t>(IdC->getZExtValue()));
  if (It == PatchConstVars.end())
    return makeError("SPIR-V backend: unknown patch constant ID {0}",
                     IdC->getZExtValue());
  StageVar &SV = It->second;
  SV.Referenced = true;

  IRBuilder<> B(CI);
  Value *Ptr = SV.GV;
  Type *RowTy = SV.VarTy;
  if (auto *AT = dyn_cast<ArrayType>(RowTy)) {
    Ptr = B.CreateInBoundsGEP(RowTy, Ptr, {B.getInt32(0), CI->getArgOperand(2)});
    RowTy = AT->getElementType();
  }
  Value *Loaded = B.CreateLoad(RowTy, Ptr);
  Value *Val = Loaded;
  if (isa<FixedVectorType>(RowTy))
    Val = B.CreateExtractElement(Loaded, CI->getArgOperand(3));
  if (Val->getType() != CI->getType()) {
    if (Val->getType()->isIntegerTy() && CI->getType()->isIntegerTy())
      Val = B.CreateZExtOrTrunc(Val, CI->getType());
    else
      Val = B.CreateBitCast(Val, CI->getType());
  }
  CI->replaceAllUsesWith(Val);
  return Error::success();
}

Error SPIRVLowerer::lowerStorePatchConstant(CallInst *CI) {
  auto *IdC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  if (!IdC)
    return makeError("SPIR-V backend: non-constant id in storePatchConstant");
  auto It = PatchConstVars.find(static_cast<uint32_t>(IdC->getZExtValue()));
  if (It == PatchConstVars.end())
    return makeError("SPIR-V backend: unknown patch constant ID {0}",
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

Error SPIRVLowerer::lowerRaytracingOp(CallInst *CI, SpvOp Op) {
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
  case SpvOp::RayQueryAlloc: {
    RayQueryState &S = getOrCreateRQ(CI);
    B.CreateStore(CI->getArgOperand(1), S.Flags);
    RayQueries[S.Flags] = S;
    CI->replaceAllUsesWith(S.Flags);
    return Error::success();
  }
  case SpvOp::RayQueryTraceInline: {
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
  case SpvOp::RayQueryProceed: {
    CI->replaceAllUsesWith(B.getFalse());
    return Error::success();
  }
  case SpvOp::RayQueryAbort:
    return Error::success();
  case SpvOp::RayQueryCommit: {
    RayQueryState &S = getOrCreateRQ(CI->getArgOperand(1));
    auto Kind = static_cast<ir::RayQueryCommitKind>(
        cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue());
    if (Kind == ir::RayQueryCommitKind::Triangle) {
      B.CreateStore(B.getInt32(1), S.CommittedStatus);
    } else {
      B.CreateStore(B.getInt32(2), S.CommittedStatus);
      B.CreateStore(CI->getArgOperand(3), S.CommittedT);
    }
    return Error::success();
  }
  case SpvOp::RayQueryGet: {
    RayQueryState &S = getOrCreateRQ(CI->getArgOperand(1));
    auto Field = static_cast<ir::RayQueryField>(
        cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue());
    auto *CompC = dyn_cast<ConstantInt>(CI->getArgOperand(3));
    switch (Field) {
    case ir::RayQueryField::CommittedStatus:
      CI->replaceAllUsesWith(B.CreateLoad(B.getInt32Ty(), S.CommittedStatus));
      return Error::success();
    case ir::RayQueryField::CandidateType:
      CI->replaceAllUsesWith(B.CreateLoad(B.getInt32Ty(), S.CandidateType));
      return Error::success();
    case ir::RayQueryField::CommittedTriangleFrontFace:
      CI->replaceAllUsesWith(B.CreateLoad(B.getInt1Ty(), S.FrontFace));
      return Error::success();
    case ir::RayQueryField::CommittedTriangleBarycentrics: {
      unsigned Comp = (CompC && CompC->getZExtValue() == 1) ? 1u : 0u;
      CI->replaceAllUsesWith(B.CreateLoad(B.getFloatTy(), S.Bary[Comp]));
      return Error::success();
    }
    case ir::RayQueryField::RayFlags:
      CI->replaceAllUsesWith(B.CreateLoad(B.getInt32Ty(), S.Flags));
      return Error::success();
    case ir::RayQueryField::WorldRayOrigin: {
      unsigned Comp = CompC ? std::min<unsigned>(CompC->getZExtValue(), 2u) : 0u;
      CI->replaceAllUsesWith(B.CreateLoad(B.getFloatTy(), S.Origin[Comp]));
      return Error::success();
    }
    case ir::RayQueryField::WorldRayDirection: {
      unsigned Comp = CompC ? std::min<unsigned>(CompC->getZExtValue(), 2u) : 0u;
      CI->replaceAllUsesWith(B.CreateLoad(B.getFloatTy(), S.Dir[Comp]));
      return Error::success();
    }
    case ir::RayQueryField::RayTMin:
      CI->replaceAllUsesWith(B.CreateLoad(B.getFloatTy(), S.TMin));
      return Error::success();
    case ir::RayQueryField::CommittedRayT:
      CI->replaceAllUsesWith(B.CreateLoad(B.getFloatTy(), S.CommittedT));
      return Error::success();
    case ir::RayQueryField::CommittedInstanceId:
      CI->replaceAllUsesWith(B.CreateLoad(B.getInt32Ty(), S.InstanceID));
      return Error::success();
    case ir::RayQueryField::CommittedInstanceIndex:
      CI->replaceAllUsesWith(B.CreateLoad(B.getInt32Ty(), S.InstanceIndex));
      return Error::success();
    case ir::RayQueryField::CommittedPrimitiveIndex:
      CI->replaceAllUsesWith(B.CreateLoad(B.getInt32Ty(), S.PrimitiveIndex));
      return Error::success();
    case ir::RayQueryField::CommittedGeometryIndex:
      CI->replaceAllUsesWith(B.CreateLoad(B.getInt32Ty(), S.GeometryIndex));
      return Error::success();
    case ir::RayQueryField::CommittedInstanceContrib:
      CI->replaceAllUsesWith(B.CreateLoad(B.getInt32Ty(), S.InstanceContrib));
      return Error::success();
    }
    return makeError("SPIR-V backend: unsupported RayQueryField");
  }
  case SpvOp::SysValRT: {
    auto Kind = static_cast<ir::SysValKind>(
        cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue());
    switch (Kind) {
    case ir::SysValKind::DispatchRaysIndex: {
      Function *TidFn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::spv_thread_id, {B.getInt32Ty()});
      CI->replaceAllUsesWith(B.CreateCall(TidFn, {CI->getArgOperand(2)}));
      return Error::success();
    }
    case ir::SysValKind::DispatchRaysDimensions: {
      Function *NwgFn = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::spv_num_workgroups, {B.getInt32Ty()});
      CI->replaceAllUsesWith(B.CreateCall(NwgFn, {CI->getArgOperand(2)}));
      return Error::success();
    }
    case ir::SysValKind::WorldRayOrigin:
    case ir::SysValKind::WorldRayDirection:
    case ir::SysValKind::RayTCurrent:
      CI->replaceAllUsesWith(ConstantFP::get(CI->getType(), 0.0));
      return Error::success();
    case ir::SysValKind::PrimitiveId: {
      GlobalVariable *GV = getOrCreateBuiltInInput(7u /* PrimitiveId */,
                                                   B.getInt32Ty(),
                                                   "in_SV_PrimitiveID");
      Value *V = B.CreateLoad(B.getInt32Ty(), GV);
      if (V->getType() != CI->getType())
        V = B.CreateZExtOrTrunc(V, CI->getType());
      CI->replaceAllUsesWith(V);
      return Error::success();
    }
    case ir::SysValKind::OutputControlPointId: {
      GlobalVariable *GV = getOrCreateBuiltInInput(
          8u /* InvocationId */, B.getInt32Ty(), "in_SV_OutputControlPointID");
      Value *V = B.CreateLoad(B.getInt32Ty(), GV);
      if (V->getType() != CI->getType())
        V = B.CreateZExtOrTrunc(V, CI->getType());
      CI->replaceAllUsesWith(V);
      return Error::success();
    }
    case ir::SysValKind::DomainLocation: {
      Type *Vec3F = FixedVectorType::get(B.getFloatTy(), 3);
      GlobalVariable *GV = getOrCreateBuiltInInput(13u /* TessCoord */, Vec3F,
                                                   "in_SV_DomainLocation");
      Value *Coord = B.CreateLoad(Vec3F, GV);
      Value *Comp = B.CreateExtractElement(Coord, CI->getArgOperand(2));
      CI->replaceAllUsesWith(Comp);
      return Error::success();
    }
    case ir::SysValKind::PrimitiveIndex:
      CI->replaceAllUsesWith(B.getInt32(0));
      return Error::success();
    default:
      return makeError("SPIR-V backend: unsupported SysValKind {0}",
                       static_cast<unsigned>(Kind));
    }
  }
  case SpvOp::TraceRay: {
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

static Type *replaceFloatWithI32Type(Type *Ty, LLVMContext &Ctx) {
  if (Ty->isFloatTy())
    return Type::getInt32Ty(Ctx);
  if (auto *AT = dyn_cast<ArrayType>(Ty)) {
    Type *NewElem = replaceFloatWithI32Type(AT->getElementType(), Ctx);
    if (NewElem != AT->getElementType())
      return ArrayType::get(NewElem, AT->getNumElements());
  }
  return Ty;
}

static bool hasAtomicUse(Value *V) {
  SmallVector<Value *, 16> Worklist = {V};
  SmallPtrSet<Value *, 16> Visited;
  while (!Worklist.empty()) {
    Value *Cur = Worklist.pop_back_val();
    if (!Visited.insert(Cur).second)
      continue;
    for (User *U : Cur->users()) {
      if (isa<AtomicRMWInst>(U) || isa<AtomicCmpXchgInst>(U))
        return true;
      if (isa<GetElementPtrInst>(U) || isa<BitCastInst>(U) ||
          isa<AddrSpaceCastInst>(U) || isa<GEPOperator>(U))
        Worklist.push_back(U);
    }
  }
  return false;
}

void SPIRVLowerer::rewriteFloatAtomicGlobals() {
  SmallVector<GlobalVariable *, 8> FloatAtomics;
  for (GlobalVariable &GV : M.globals()) {
    if (GV.getAddressSpace() != 3)
      continue;
    Type *NewTy = replaceFloatWithI32Type(GV.getValueType(), Ctx);
    if (NewTy != GV.getValueType() && hasAtomicUse(&GV))
      FloatAtomics.push_back(&GV);
  }
  for (GlobalVariable *OldGV : FloatAtomics) {
    Type *NewTy = replaceFloatWithI32Type(OldGV->getValueType(), Ctx);
    auto *NewGV = new GlobalVariable(
        M, NewTy, OldGV->isConstant(), OldGV->getLinkage(),
        OldGV->hasInitializer() ? UndefValue::get(NewTy) : nullptr, "", OldGV,
        OldGV->getThreadLocalMode(), OldGV->getAddressSpace());
    NewGV->takeName(OldGV);
    NewGV->setAlignment(OldGV->getAlign());

    // Update GEPs and float loads/stores reachable from OldGV.
    SmallVector<Value *, 16> Worklist = {OldGV};
    SmallPtrSet<Value *, 16> Visited;
    SmallVector<GetElementPtrInst *, 16> GEPs;
    SmallVector<LoadInst *, 16> Loads;
    SmallVector<StoreInst *, 16> Stores;
    SmallVector<AtomicRMWInst *, 16> FloatRMWs;
    while (!Worklist.empty()) {
      Value *Cur = Worklist.pop_back_val();
      if (!Visited.insert(Cur).second)
        continue;
      for (User *U : Cur->users()) {
        if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
          GEPs.push_back(GEP);
          Worklist.push_back(GEP);
        } else if (auto *LI = dyn_cast<LoadInst>(U)) {
          if (LI->getType()->isFloatTy())
            Loads.push_back(LI);
        } else if (auto *SI = dyn_cast<StoreInst>(U)) {
          if (SI->getPointerOperand() == Cur &&
              SI->getValueOperand()->getType()->isFloatTy())
            Stores.push_back(SI);
        } else if (auto *RMW = dyn_cast<AtomicRMWInst>(U)) {
          if (RMW->getValOperand()->getType()->isFloatTy())
            FloatRMWs.push_back(RMW);
        }
      }
    }
    OldGV->replaceAllUsesWith(NewGV);
    for (GetElementPtrInst *GEP : GEPs) {
      Type *NewSrc = replaceFloatWithI32Type(GEP->getSourceElementType(), Ctx);
      if (NewSrc != GEP->getSourceElementType()) {
        IRBuilder<> B(GEP);
        SmallVector<Value *, 4> Idxs(GEP->indices());
        Value *NewGEP = GEP->isInBounds()
                            ? B.CreateInBoundsGEP(NewSrc, GEP->getPointerOperand(), Idxs)
                            : B.CreateGEP(NewSrc, GEP->getPointerOperand(), Idxs);
        GEP->replaceAllUsesWith(NewGEP);
        GEP->eraseFromParent();
      }
    }
    for (LoadInst *LI : Loads) {
      IRBuilder<> B(LI);
      Value *IL = B.CreateAlignedLoad(B.getInt32Ty(), LI->getPointerOperand(),
                                      LI->getAlign(), LI->isVolatile());
      Value *FL = B.CreateBitCast(IL, B.getFloatTy());
      LI->replaceAllUsesWith(FL);
      LI->eraseFromParent();
    }
    for (StoreInst *SI : Stores) {
      IRBuilder<> B(SI);
      Value *IV = B.CreateBitCast(SI->getValueOperand(), B.getInt32Ty());
      B.CreateAlignedStore(IV, SI->getPointerOperand(), SI->getAlign(),
                           SI->isVolatile());
      SI->eraseFromParent();
    }
    for (AtomicRMWInst *RMW : FloatRMWs) {
      IRBuilder<> B(RMW);
      Value *IV = B.CreateBitCast(RMW->getValOperand(), B.getInt32Ty());
      auto *NewRMW = B.CreateAtomicRMW(
          RMW->getOperation(), RMW->getPointerOperand(), IV, RMW->getAlign(),
          RMW->getOrdering(), RMW->getSyncScopeID());
      Value *FV = B.CreateBitCast(NewRMW, B.getFloatTy());
      RMW->replaceAllUsesWith(FV);
      RMW->eraseFromParent();
    }
    OldGV->eraseFromParent();
  }
}

void SPIRVLowerer::fixupExistingAtomics(Function &F) {
  SyncScope::ID WorkgroupScope = Ctx.getOrInsertSyncScopeID("workgroup");
  SyncScope::ID DeviceScope = Ctx.getOrInsertSyncScopeID("device");

  SmallVector<AtomicCmpXchgInst *, 16> CmpXchgs;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *RMW = dyn_cast<AtomicRMWInst>(&I)) {
        unsigned AS = RMW->getPointerAddressSpace();
        RMW->setSyncScopeID(AS == 3 ? WorkgroupScope : DeviceScope);
        if (RMW->getOrdering() == AtomicOrdering::Monotonic ||
            RMW->getOrdering() == AtomicOrdering::SequentiallyConsistent)
          RMW->setOrdering(AtomicOrdering::AcquireRelease);
      } else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I)) {
        if (CX->getCompareOperand()->getType()->isIntegerTy(32))
          CmpXchgs.push_back(CX);
      }
    }
  }

  for (AtomicCmpXchgInst *CX : CmpXchgs) {
    unsigned AS = CX->getPointerAddressSpace();
    Function *&HelperFn = CmpXchgHelpers[AS];
    if (!HelperFn) {
      Type *I32Ty = Type::getInt32Ty(Ctx);
      Type *PtrTy = PointerType::get(Ctx, AS);
      FunctionType *FT = FunctionType::get(
          I32Ty, {PtrTy, I32Ty, I32Ty, I32Ty, I32Ty, I32Ty}, false);
      std::string FnName = "__crucible_spv_cmpxchg." + std::to_string(AS);
      HelperFn = Function::Create(FT, GlobalValue::InternalLinkage, FnName, M);
      HelperFn->addFnAttr(Attribute::NoInline);
      HelperFn->addFnAttr(Attribute::OptimizeNone);
      BasicBlock *HBB = BasicBlock::Create(Ctx, "entry", HelperFn);
      IRBuilder<> HB(HBB);
      Value *Dummy = HB.CreateAlignedLoad(I32Ty, HelperFn->getArg(0), Align(4));
      HB.CreateRet(Dummy);
    }
    IRBuilder<> B(CX);
    uint32_t Scope = (AS == 3) ? 2u /* Workgroup */ : 1u /* Device */;
    uint32_t MemSem =
        (AS == 3) ? 0x100u /* WorkgroupMemory */ : 0x40u /* UniformMemory */;
    uint32_t EqSem = 0x8u /* AcquireRelease */ | MemSem;
    uint32_t UneqSem = 0x2u /* Acquire */ | MemSem;
    Value *Old = B.CreateCall(
        HelperFn, {CX->getPointerOperand(), B.getInt32(Scope),
                   B.getInt32(EqSem), B.getInt32(UneqSem),
                   CX->getNewValOperand(), CX->getCompareOperand()});
    Value *Ok = B.CreateICmpEQ(Old, CX->getCompareOperand());
    Value *Res = PoisonValue::get(CX->getType());
    Res = B.CreateInsertValue(Res, Old, {0});
    Res = B.CreateInsertValue(Res, Ok, {1});
    CX->replaceAllUsesWith(Res);
    CX->eraseFromParent();
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
      } else if (Reflect.Stage == ShaderStage::Geometry ||
                 Reflect.Stage == ShaderStage::Hull ||
                 Reflect.Stage == ShaderStage::Domain) {
        NamedMDNode *EMNode = M.getOrInsertNamedMetadata("spirv.ExecutionMode");
        auto *I32Ty = Type::getInt32Ty(Ctx);
        auto addEM = [&](uint32_t Mode,
                         std::optional<uint32_t> Arg = std::nullopt) {
          SmallVector<Metadata *, 3> Ops;
          Ops.push_back(ValueAsMetadata::get(&F));
          Ops.push_back(ConstantAsMetadata::get(ConstantInt::get(I32Ty, Mode)));
          if (Arg)
            Ops.push_back(
                ConstantAsMetadata::get(ConstantInt::get(I32Ty, *Arg)));
          EMNode->addOperand(MDNode::get(Ctx, Ops));
        };
        if (Reflect.Stage == ShaderStage::Geometry) {
          uint32_t InPrim =
              Reflect.Geometry ? Reflect.Geometry->InputPrimitive : 3u;
          uint32_t OutTopo =
              Reflect.Geometry ? Reflect.Geometry->OutputTopology : 5u;
          uint32_t MaxVerts =
              Reflect.Geometry ? Reflect.Geometry->MaxVertexCount : 3u;
          uint32_t InstCount =
              Reflect.Geometry ? std::max(1u, Reflect.Geometry->InstanceCount)
                               : 1u;
          uint32_t InMode = 22u; // Triangles
          if (InPrim == 1u)
            InMode = 19u; // InputPoints
          else if (InPrim == 2u)
            InMode = 20u; // InputLines
          else if (InPrim == 6u)
            InMode = 21u; // InputLinesAdjacency
          else if (InPrim == 7u)
            InMode = 23u; // InputTrianglesAdjacency
          uint32_t OutMode = 29u; // OutputTriangleStrip
          if (OutTopo == 1u)
            OutMode = 27u; // OutputPoints
          else if (OutTopo == 3u)
            OutMode = 28u; // OutputLineStrip
          addEM(InMode);
          addEM(0u /* Invocations */, InstCount);
          addEM(OutMode);
          addEM(26u /* OutputVertices */, MaxVerts);
        } else if (Reflect.Stage == ShaderStage::Hull) {
          uint32_t OutCPs =
              Reflect.Hull ? std::max(1u, Reflect.Hull->OutputControlPointCount)
                           : 3u;
          addEM(26u /* OutputVertices */, OutCPs);
        } else {
          TessellationDomain Dom = Reflect.Domain
                                       ? Reflect.Domain->Domain
                                       : TessellationDomain::Triangle;
          uint32_t DomMode = 22u; // Triangles
          if (Dom == TessellationDomain::Quad) DomMode = 24u; // Quads
          else if (Dom == TessellationDomain::Isoline) DomMode = 25u; // Isolines
          addEM(DomMode);

          uint32_t SpacingMode = 1u; // SpacingEqual
          if (Opts.TessPartitioning ==
              static_cast<uint8_t>(TessellationPartitioning::FractionalEven))
            SpacingMode = 2u;
          else if (Opts.TessPartitioning ==
                   static_cast<uint8_t>(
                       TessellationPartitioning::FractionalOdd))
            SpacingMode = 3u;
          addEM(SpacingMode);

          uint32_t OrderMode = 4u; // VertexOrderCw
          if (Opts.TessOutputPrimitive ==
              static_cast<uint8_t>(TessellationOutputPrimitive::TriangleCCW))
            OrderMode = 5u; // VertexOrderCcw
          addEM(OrderMode);
          if (Opts.TessOutputPrimitive ==
              static_cast<uint8_t>(TessellationOutputPrimitive::Point))
            addEM(10u /* PointMode */);
        }
      }
    } else if (!F.isDeclaration()) {
      F.setLinkage(GlobalValue::InternalLinkage);
    }
  }
}

void SPIRVLowerer::cleanupBridgeArtifacts() {
  SmallVector<NamedMDNode *, 8> ToEraseMD;
  for (NamedMDNode &NMD : M.named_metadata()) {
    if (NMD.getName().starts_with("dx."))
      ToEraseMD.push_back(&NMD);
  }
  for (NamedMDNode *NMD : ToEraseMD)
    M.eraseNamedMetadata(NMD);

  SmallVector<Function *, 16> DeadDecls;
  for (Function &F : M) {
    if (F.isDeclaration() && F.getName().starts_with("__crucible_spv.") &&
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

static bool isScalarOrVecFloat32(Type *Ty) {
  if (Ty->isFloatTy())
    return true;
  if (auto *VT = dyn_cast<FixedVectorType>(Ty))
    return VT->getElementType()->isFloatTy();
  return false;
}

static Value *flushF32Denorm(IRBuilder<> &B, Value *V) {
  if (isa<Constant>(V) || !isScalarOrVecFloat32(V->getType()))
    return V;
  Type *Ty = V->getType();
  Type *IntTy = Ty->isFloatTy()
                    ? static_cast<Type *>(B.getInt32Ty())
                    : static_cast<Type *>(FixedVectorType::get(
                          B.getInt32Ty(),
                          cast<FixedVectorType>(Ty)->getNumElements()));
  Value *Bits = B.CreateBitCast(V, IntTy);
  Value *AbsBits = B.CreateAnd(Bits, ConstantInt::get(IntTy, 0x7FFFFFFFu));
  Value *IsSubnorm =
      B.CreateICmpULT(AbsBits, ConstantInt::get(IntTy, 0x00800000u));
  Value *SignOnly = B.CreateAnd(Bits, ConstantInt::get(IntTy, 0x80000000u));
  Value *Flushed = B.CreateSelect(IsSubnorm, SignOnly, Bits);
  return B.CreateBitCast(Flushed, Ty);
}

// DXIL defaults to flush-to-zero for 32-bit float denormals, whereas CPU
// Vulkan drivers like Lavapipe preserve subnormals by default. Flush f32
// denormal operands on math/FMul operations to signed zero for portable
// HLSL compliance without altering raw bit-preserving loads/stores/casts.
static void flushF32DenormalsForMath(Function &F) {
  SmallVector<Instruction *, 32> Work;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (I.getOpcode() == Instruction::FMul &&
          isScalarOrVecFloat32(I.getType())) {
        Work.push_back(&I);
      } else if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
        switch (II->getIntrinsicID()) {
        case Intrinsic::log2:
        case Intrinsic::log:
        case Intrinsic::log10:
        case Intrinsic::sqrt:
        case Intrinsic::sinh:
        case Intrinsic::fma:
        case Intrinsic::fmuladd:
        case Intrinsic::spv_rsqrt:
        case Intrinsic::spv_degrees:
        case Intrinsic::spv_radians:
          if (isScalarOrVecFloat32(II->getType()))
            Work.push_back(II);
          break;
        default:
          break;
        }
      }
    }
  }
  for (Instruction *I : Work) {
    IRBuilder<> B(I);
    if (auto *II = dyn_cast<IntrinsicInst>(I)) {
      for (unsigned Arg = 0, E = II->arg_size(); Arg < E; ++Arg) {
        Value *V = II->getArgOperand(Arg);
        Value *FV = flushF32Denorm(B, V);
        if (FV != V)
          II->setArgOperand(Arg, FV);
      }
    } else {
      for (unsigned OpIdx = 0, E = I->getNumOperands(); OpIdx < E; ++OpIdx) {
        Value *V = I->getOperand(OpIdx);
        Value *FV = flushF32Denorm(B, V);
        if (FV != V)
          I->setOperand(OpIdx, FV);
      }
    }
  }
}

Error SPIRVLowerer::run() {
  auto StageOrErr = stageTripleSuffix(Reflect.Stage, Opts.StageAsCompute);
  if (!StageOrErr)
    return StageOrErr.takeError();

  prepareStageInterface();
  mergeHullPatchConstantFunc();
  rewriteFloatAtomicGlobals();

  SmallVector<CallInst *, 64> BridgeCalls;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    fixupExistingAtomics(F);
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (Function *Callee = CI->getCalledFunction()) {
            if (Callee->getName().starts_with("__crucible_spv."))
              BridgeCalls.push_back(CI);
          }
        }
      }
    }
  }

  // Pre-scan for image/typed-buffer resources used by atomic ops so
  // getSPIRVResourceType emits explicit R32ui format (required by
  // OpImageTexelPointer).
  for (CallInst *CI : BridgeCalls) {
    auto *OpC = dyn_cast<ConstantInt>(CI->getArgOperand(0));
    if (!OpC)
      continue;
    auto Op = static_cast<SpvOp>(OpC->getZExtValue());
    if (Op != SpvOp::BufferAtomic && Op != SpvOp::BufferAtomicCmpXchg)
      continue;
    Value *H = CI->getArgOperand(1);
    if (auto *BindCI = dyn_cast<CallInst>(H)) {
      if (auto *BOpC = dyn_cast<ConstantInt>(BindCI->getArgOperand(0));
          BOpC && static_cast<SpvOp>(BOpC->getZExtValue()) == SpvOp::BindResource) {
        auto *ClassC = dyn_cast<ConstantInt>(BindCI->getArgOperand(1));
        auto *HasRecC = dyn_cast<ConstantInt>(BindCI->getArgOperand(2));
        auto *RecC = dyn_cast<ConstantInt>(BindCI->getArgOperand(3));
        auto *SpaceC = dyn_cast<ConstantInt>(BindCI->getArgOperand(4));
        auto *BindC = dyn_cast<ConstantInt>(BindCI->getArgOperand(5));
        if (ClassC && HasRecC && RecC && SpaceC && BindC) {
          auto RC = static_cast<ResourceClass>(ClassC->getZExtValue());
          const ResourceBinding *RB =
              !HasRecC->isZero()
                  ? findResourceByID(RC, static_cast<uint32_t>(RecC->getZExtValue()))
                  : findResourceByBinding(
                        RC, static_cast<uint32_t>(SpaceC->getZExtValue()),
                        static_cast<uint32_t>(BindC->getZExtValue()));
          if (RB && RB->Kind != ResourceKind::RawBuffer &&
              RB->Kind != ResourceKind::StructuredBuffer) {
            AtomicImageBindings.insert(RB);
            const_cast<ResourceBinding *>(RB)->RequiresTextureAtomics = true;
          }
        }
      }
    }
  }

  for (CallInst *CI : BridgeCalls) {
    if (Error E = lowerBridgeCall(CI))
      return E;
  }

  for (auto It = BridgeCalls.rbegin(), End = BridgeCalls.rend(); It != End;
       ++It) {
    CallInst *CI = *It;
    if (!CI->use_empty())
      CI->replaceAllUsesWith(PoisonValue::get(CI->getType()));
    CI->eraseFromParent();
  }

  configureEntryAndLinkage(*StageOrErr);
  cleanupBridgeArtifacts();

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
    if (!F.isDeclaration()) {
      normalizeArrayGlobalAccesses(F);
      flushF32DenormalsForMath(F);
    }
  }

  emitStageDecorations();

  return Error::success();
}

/// Lowers `crucible.*`, `gpu.*`, `math.*`, and `arith.*` ops in `Mod` to pure
/// `llvm.*` dialect ops (using `@__crucible_spv.<N>` bridge declarations for
/// SPIR-V backend operations) and exports the result to an `llvm::Module`.
Expected<std::unique_ptr<Module>>
lowerMLIRToSPIRVLLVM(mlir::ModuleOp Mod, LLVMContext &LLVMCtx) {
  mlir::MLIRContext *MCtx = Mod.getContext();
  auto PtrTy = mlir::LLVM::LLVMPointerType::get(MCtx);
  auto I1Ty = mlir::IntegerType::get(MCtx, 1);
  auto I8Ty = mlir::IntegerType::get(MCtx, 8);
  auto I32Ty = mlir::IntegerType::get(MCtx, 32);
  auto F32Ty = mlir::Float32Type::get(MCtx);
  auto VoidTy = mlir::LLVM::LLVMVoidType::get(MCtx);

  auto convertType = [&](mlir::Type T) -> mlir::Type {
    if (mlir::isa<ir::ResourceType, ir::RayQueryType>(T))
      return PtrTy;
    if (mlir::isa<mlir::IndexType>(T))
      return I32Ty;
    return T;
  };

  unsigned BridgeCounter = 0;
  auto makeBridgeCall = [&](mlir::OpBuilder &B, mlir::Location Loc, SpvOp Op,
                            mlir::Type RetTy,
                            ArrayRef<mlir::Value> Args) -> mlir::Value {
    SmallVector<mlir::Value, 16> CallArgs;
    CallArgs.push_back(mlir::LLVM::ConstantOp::create(
        B, Loc, I32Ty, B.getI32IntegerAttr(static_cast<int32_t>(Op))));
    CallArgs.append(Args.begin(), Args.end());

    SmallVector<mlir::Type, 16> ArgTypes;
    for (mlir::Value V : CallArgs)
      ArgTypes.push_back(V.getType());

    mlir::Type FuncRet = RetTy ? RetTy : mlir::Type(VoidTy);
    auto FuncTy = mlir::LLVM::LLVMFunctionType::get(FuncRet, ArgTypes);
    std::string FnName = "__crucible_spv." + std::to_string(BridgeCounter++);
    {
      mlir::OpBuilder::InsertionGuard G(B);
      B.setInsertionPointToEnd(Mod.getBody());
      auto Decl = mlir::LLVM::LLVMFuncOp::create(B, Loc, FnName, FuncTy);
      Decl.setLinkage(mlir::LLVM::Linkage::External);
    }
    SmallVector<mlir::Type, 1> ResTypes;
    if (RetTy)
      ResTypes.push_back(RetTy);
    auto Call = mlir::LLVM::CallOp::create(
        B, Loc, ResTypes, mlir::SymbolRefAttr::get(MCtx, FnName), CallArgs);
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
  auto gpuDimToI32 = [](mlir::gpu::Dimension D) -> int32_t {
    switch (D) {
    case mlir::gpu::Dimension::x:
      return 0;
    case mlir::gpu::Dimension::y:
      return 1;
    case mlir::gpu::Dimension::z:
      return 2;
    }
    return 0;
  };

  // Pass 1: convert any block arguments with custom/index types to LLVM types.
  Mod.walk([&](mlir::LLVM::LLVMFuncOp F) {
    if (F.isExternal())
      return;
    for (mlir::Block &BB : F.getBody()) {
      for (mlir::BlockArgument Arg : BB.getArguments()) {
        mlir::Type NewT = convertType(Arg.getType());
        if (NewT != Arg.getType())
          Arg.setType(NewT);
      }
    }
  });

  // Pass 2: lower all non-LLVM ops in each function body.
  SmallVector<mlir::Operation *, 128> OpsToLower;
  Mod.walk([&](mlir::LLVM::LLVMFuncOp F) {
    if (F.isExternal())
      return;
    for (mlir::Block &BB : F.getBody()) {
      for (mlir::Operation &Op : BB) {
        if (!mlir::isa<mlir::LLVM::LLVMDialect>(Op.getDialect()))
          OpsToLower.push_back(&Op);
        else {
          for (mlir::Value Res : Op.getResults()) {
            mlir::Type NewT = convertType(Res.getType());
            if (NewT != Res.getType())
              Res.setType(NewT);
          }
        }
      }
    }
  });

  for (mlir::Operation *Op : OpsToLower) {
    mlir::OpBuilder B(Op);
    mlir::Location Loc = Op->getLoc();

    // --- arith dialect ---
    if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(Op)) {
      mlir::Type NewT = convertType(C.getType());
      mlir::TypedAttr ValAttr = C.getValue();
      if (mlir::isa<mlir::IndexType>(C.getType())) {
        auto IA = mlir::cast<mlir::IntegerAttr>(ValAttr);
        ValAttr = B.getI32IntegerAttr(static_cast<int32_t>(IA.getInt()));
      }
      mlir::Value Rep = mlir::LLVM::ConstantOp::create(B, Loc, NewT, ValAttr);
      C.getResult().replaceAllUsesWith(Rep);
      C.erase();
      continue;
    }
    if (auto Cast = mlir::dyn_cast<mlir::arith::IndexCastUIOp>(Op)) {
      mlir::Value In = Cast.getIn();
      mlir::Type OutT = convertType(Cast.getType());
      mlir::Value Rep = In;
      if (In.getType() != OutT) {
        unsigned InW = In.getType().getIntOrFloatBitWidth();
        unsigned OutW = OutT.getIntOrFloatBitWidth();
        if (InW < OutW)
          Rep = mlir::LLVM::ZExtOp::create(B, Loc, OutT, In);
        else if (InW > OutW)
          Rep = mlir::LLVM::TruncOp::create(B, Loc, OutT, In);
      }
      Cast.getResult().replaceAllUsesWith(Rep);
      Cast.erase();
      continue;
    }

    // --- gpu dialect ---
    if (auto Tid = mlir::dyn_cast<mlir::gpu::ThreadIdOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::ThreadIdInGroup, I32Ty,
          {i32C(B, Loc, gpuDimToI32(Tid.getDimension()))});
      Tid.getResult().replaceAllUsesWith(Rep);
      Tid.erase();
      continue;
    }
    if (auto Bid = mlir::dyn_cast<mlir::gpu::BlockIdOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::GroupId, I32Ty,
          {i32C(B, Loc, gpuDimToI32(Bid.getDimension()))});
      Bid.getResult().replaceAllUsesWith(Rep);
      Bid.erase();
      continue;
    }
    if (auto Gid = mlir::dyn_cast<mlir::gpu::GlobalIdOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::ThreadId, I32Ty,
          {i32C(B, Loc, gpuDimToI32(Gid.getDimension()))});
      Gid.getResult().replaceAllUsesWith(Rep);
      Gid.erase();
      continue;
    }
    if (auto Lid = mlir::dyn_cast<mlir::gpu::LaneIdOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(B, Loc, SpvOp::LaneId, I32Ty, {});
      Lid.getResult().replaceAllUsesWith(Rep);
      Lid.erase();
      continue;
    }
    if (auto Sgs = mlir::dyn_cast<mlir::gpu::SubgroupSizeOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(B, Loc, SpvOp::SubgroupSize, I32Ty, {});
      Sgs.getResult().replaceAllUsesWith(Rep);
      Sgs.erase();
      continue;
    }
    if (auto Bar = mlir::dyn_cast<mlir::gpu::BarrierOp>(Op)) {
      makeBridgeCall(B, Loc, SpvOp::Barrier, mlir::Type(),
                     {i32C(B, Loc, 1), i32C(B, Loc, 1), i32C(B, Loc, 0)});
      Bar.erase();
      continue;
    }
    if (auto BC = mlir::dyn_cast<mlir::gpu::SubgroupBroadcastOp>(Op)) {
      mlir::Value Rep;
      if (BC.getBroadcastType() ==
          mlir::gpu::BroadcastType::first_active_lane) {
        Rep = makeBridgeCall(B, Loc, SpvOp::WaveReadLaneFirst, BC.getType(),
                             {BC.getSrc()});
      } else {
        Rep = makeBridgeCall(B, Loc, SpvOp::WaveReadLaneAt, BC.getType(),
                             {BC.getSrc(), BC.getOperand(1)});
      }
      BC.getResult().replaceAllUsesWith(Rep);
      BC.erase();
      continue;
    }
    if (auto Shuf = mlir::dyn_cast<mlir::gpu::ShuffleOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::WaveReadLaneAt, Shuf.getShuffleResult().getType(),
          {Shuf.getValue(), Shuf.getOffset()});
      Shuf.getShuffleResult().replaceAllUsesWith(Rep);
      if (!Shuf.getValid().use_empty()) {
        mlir::Value TrueV = mlir::LLVM::ConstantOp::create(
            B, Loc, I1Ty, B.getBoolAttr(true));
        Shuf.getValid().replaceAllUsesWith(TrueV);
      }
      Shuf.erase();
      continue;
    }
    if (auto Red = mlir::dyn_cast<mlir::gpu::SubgroupReduceOp>(Op)) {
      int32_t IsExclusive = 0;
      if (auto SA = Red->getAttrOfType<mlir::StringAttr>("crucible.scan_kind"))
        if (SA.getValue() == "exclusive")
          IsExclusive = 1;
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::WaveReduce, Red.getType(),
          {Red.getValue(), i32C(B, Loc, static_cast<int32_t>(Red.getOp())),
           i32C(B, Loc, IsExclusive)});
      Red.getResult().replaceAllUsesWith(Rep);
      Red.erase();
      continue;
    }

    // --- math dialect ---
    if (auto MOp = mlir::dyn_cast<mlir::math::AbsFOp>(Op)) {
      auto Rep = mlir::LLVM::FAbsOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep);
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::CosOp>(Op)) {
      auto Rep = mlir::LLVM::CosOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep);
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::SinOp>(Op)) {
      auto Rep = mlir::LLVM::SinOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep);
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::Exp2Op>(Op)) {
      auto Rep = mlir::LLVM::Exp2Op::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep);
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::Log2Op>(Op)) {
      auto Rep = mlir::LLVM::Log2Op::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep);
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::SqrtOp>(Op)) {
      auto Rep = mlir::LLVM::SqrtOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep);
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::FloorOp>(Op)) {
      auto Rep = mlir::LLVM::FFloorOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep);
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::CeilOp>(Op)) {
      auto Rep = mlir::LLVM::FCeilOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep);
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::TruncOp>(Op)) {
      auto Rep = mlir::LLVM::FTruncOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep);
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::RoundEvenOp>(Op)) {
      auto Rep = mlir::LLVM::RoundEvenOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep);
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::FmaOp>(Op)) {
      auto Rep = mlir::LLVM::FMAOp::create(B, Loc, MOp.getA(), MOp.getB(),
                                           MOp.getC());
      MOp.getResult().replaceAllUsesWith(Rep);
      MOp.erase();
      continue;
    }
    if (auto MOp = mlir::dyn_cast<mlir::math::CtPopOp>(Op)) {
      auto Rep = mlir::LLVM::CtPopOp::create(B, Loc, MOp.getOperand());
      MOp.getResult().replaceAllUsesWith(Rep);
      MOp.erase();
      continue;
    }
    auto lowerUnaryBridge = [&](SpvOp SOp) -> bool {
      mlir::Value Rep =
          makeBridgeCall(B, Loc, SOp, Op->getResult(0).getType(),
                         {Op->getOperand(0)});
      Op->getResult(0).replaceAllUsesWith(Rep);
      Op->erase();
      return true;
    };
    if (mlir::isa<mlir::math::TanOp>(Op) && lowerUnaryBridge(SpvOp::Tan))
      continue;
    if (mlir::isa<mlir::math::AcosOp>(Op) && lowerUnaryBridge(SpvOp::Acos))
      continue;
    if (mlir::isa<mlir::math::AsinOp>(Op) && lowerUnaryBridge(SpvOp::Asin))
      continue;
    if (mlir::isa<mlir::math::AtanOp>(Op) && lowerUnaryBridge(SpvOp::Atan))
      continue;
    if (mlir::isa<mlir::math::CoshOp>(Op) && lowerUnaryBridge(SpvOp::Cosh))
      continue;
    if (mlir::isa<mlir::math::SinhOp>(Op) && lowerUnaryBridge(SpvOp::Sinh))
      continue;
    if (mlir::isa<mlir::math::TanhOp>(Op) && lowerUnaryBridge(SpvOp::Tanh))
      continue;
    if (mlir::isa<mlir::math::RsqrtOp>(Op) && lowerUnaryBridge(SpvOp::Rsqrt))
      continue;
    if (mlir::isa<mlir::math::IsNaNOp>(Op) && lowerUnaryBridge(SpvOp::IsNaN))
      continue;
    if (mlir::isa<mlir::math::IsInfOp>(Op) && lowerUnaryBridge(SpvOp::IsInf))
      continue;
    if (mlir::isa<mlir::math::IsFiniteOp>(Op) &&
        lowerUnaryBridge(SpvOp::IsFinite))
      continue;

    // --- crucible dialect ---
    if (auto COp = mlir::dyn_cast<ir::BindResourceOp>(Op)) {
      int32_t HasRec = COp.getRecordId().has_value() ? 1 : 0;
      int32_t RecId = COp.getRecordId().value_or(0);
      int32_t Space = COp.getSpace().value_or(0);
      int32_t Binding = COp.getBinding().value_or(0);
      mlir::Value Idx = COp.getIndex() ? COp.getIndex() : i32C(B, Loc, 0);
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::BindResource, PtrTy,
          {i32C(B, Loc, static_cast<int32_t>(COp.getResourceClass())),
           i32C(B, Loc, HasRec), i32C(B, Loc, RecId), i32C(B, Loc, Space),
           i32C(B, Loc, Binding), Idx,
           i32C(B, Loc, COp.getNonUniform() ? 1 : 0)});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::HeapResourceOp>(Op)) {
      int32_t ResCls = COp.getResourceClass().has_value()
                           ? static_cast<int32_t>(*COp.getResourceClass())
                           : (COp.getIsSamplerHeap() ? 3 : 0);
      int32_t Kind = COp.getResourceKind().value_or(0);
      int32_t Prop1 = COp.getElementStride().value_or(0);
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::HeapResource, PtrTy,
          {COp.getIndex(), i32C(B, Loc, COp.getIsSamplerHeap() ? 1 : 0),
           i32C(B, Loc, COp.getNonUniform() ? 1 : 0), i32C(B, Loc, ResCls),
           i32C(B, Loc, Kind), i32C(B, Loc, Prop1),
           i32C(B, Loc, COp.getHasCounter() ? 1 : 0)});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::CBufferLoadOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::CBufferLoad, COp.getResult().getType(),
          {COp.getHandle(), COp.getRegIndex()});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::BufferLoadOp>(Op)) {
      mlir::Value ElemOff = COp.getElementOffset()
                                ? COp.getElementOffset()
                                : undefV(B, Loc, I32Ty);
      int32_t Mask = COp.getMask().value_or(0xF);
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::BufferLoad, COp.getResult().getType(),
          {COp.getHandle(), COp.getIndex(), ElemOff, i32C(B, Loc, Mask)});
      COp.getResult().replaceAllUsesWith(Rep);
      if (COp.getStatus())
        COp.getStatus().replaceAllUsesWith(i32C(B, Loc, 1));
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::BufferStoreOp>(Op)) {
      mlir::Value ElemOff = COp.getElementOffset()
                                ? COp.getElementOffset()
                                : undefV(B, Loc, I32Ty);
      SmallVector<mlir::Value, 8> Args = {COp.getHandle(), COp.getIndex(),
                                          ElemOff};
      mlir::Type ValTy =
          COp.getValues().empty() ? mlir::Type(I32Ty) : COp.getValues()[0].getType();
      for (unsigned I = 0; I < 4; ++I)
        Args.push_back(I < COp.getValues().size() ? COp.getValues()[I]
                                                  : undefV(B, Loc, ValTy));
      Args.push_back(i32C(B, Loc, COp.getMask()));
      makeBridgeCall(B, Loc, SpvOp::BufferStore, mlir::Type(), Args);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::BufferAtomicOp>(Op)) {
      SmallVector<mlir::Value, 7> Args = {
          COp.getHandle(),
          i32C(B, Loc, static_cast<int32_t>(COp.getOp()))};
      for (unsigned I = 0; I < 3; ++I)
        Args.push_back(I < COp.getCoords().size() ? COp.getCoords()[I]
                                                  : undefV(B, Loc, I32Ty));
      Args.push_back(COp.getValue());
      mlir::Value Rep = makeBridgeCall(B, Loc, SpvOp::BufferAtomic,
                                       COp.getResult().getType(), Args);
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::BufferAtomicCmpXchgOp>(Op)) {
      SmallVector<mlir::Value, 7> Args = {COp.getHandle()};
      for (unsigned I = 0; I < 3; ++I)
        Args.push_back(I < COp.getCoords().size() ? COp.getCoords()[I]
                                                  : undefV(B, Loc, I32Ty));
      Args.push_back(COp.getCompare());
      Args.push_back(COp.getValue());
      mlir::Value Rep = makeBridgeCall(B, Loc, SpvOp::BufferAtomicCmpXchg,
                                       COp.getResult().getType(), Args);
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::BufferUpdateCounterOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::BufferUpdateCounter, I32Ty,
          {COp.getHandle(), i8C(B, Loc, COp.getInc())});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::TexLoadOp>(Op)) {
      mlir::Value Mip = COp.getMipOrSample() ? COp.getMipOrSample()
                                             : undefV(B, Loc, I32Ty);
      SmallVector<mlir::Value, 8> Args = {COp.getHandle(), Mip};
      for (unsigned I = 0; I < 3; ++I)
        Args.push_back(I < COp.getCoords().size() ? COp.getCoords()[I]
                                                  : undefV(B, Loc, I32Ty));
      for (unsigned I = 0; I < 3; ++I)
        Args.push_back(I < COp.getOffsets().size() ? COp.getOffsets()[I]
                                                   : undefV(B, Loc, I32Ty));
      mlir::Value Rep = makeBridgeCall(B, Loc, SpvOp::TexLoad,
                                       COp.getResult().getType(), Args);
      COp.getResult().replaceAllUsesWith(Rep);
      if (COp.getStatus())
        COp.getStatus().replaceAllUsesWith(i32C(B, Loc, 1));
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::TexStoreOp>(Op)) {
      SmallVector<mlir::Value, 9> Args = {COp.getHandle()};
      for (unsigned I = 0; I < 3; ++I)
        Args.push_back(I < COp.getCoords().size() ? COp.getCoords()[I]
                                                  : undefV(B, Loc, I32Ty));
      mlir::Type ValTy =
          COp.getValues().empty() ? mlir::Type(F32Ty) : COp.getValues()[0].getType();
      for (unsigned I = 0; I < 4; ++I)
        Args.push_back(I < COp.getValues().size() ? COp.getValues()[I]
                                                  : undefV(B, Loc, ValTy));
      Args.push_back(i32C(B, Loc, COp.getMask()));
      makeBridgeCall(B, Loc, SpvOp::TexStore, mlir::Type(), Args);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::TexSampleOp>(Op)) {
      SmallVector<mlir::Value, 20> Args = {
          i32C(B, Loc, static_cast<int32_t>(COp.getMode())),
          COp.getTex(), COp.getSampler()};
      for (unsigned I = 0; I < 4; ++I)
        Args.push_back(I < COp.getCoords().size() ? COp.getCoords()[I]
                                                  : undefV(B, Loc, F32Ty));
      for (unsigned I = 0; I < 3; ++I)
        Args.push_back(I < COp.getOffsets().size() ? COp.getOffsets()[I]
                                                   : undefV(B, Loc, I32Ty));
      Args.push_back(COp.getLodOrBias() ? COp.getLodOrBias()
                                        : undefV(B, Loc, F32Ty));
      for (unsigned I = 0; I < 3; ++I)
        Args.push_back(I < COp.getDdx().size() ? COp.getDdx()[I]
                                               : undefV(B, Loc, F32Ty));
      for (unsigned I = 0; I < 3; ++I)
        Args.push_back(I < COp.getDdy().size() ? COp.getDdy()[I]
                                               : undefV(B, Loc, F32Ty));
      Args.push_back(COp.getCmpVal() ? COp.getCmpVal() : undefV(B, Loc, F32Ty));
      Args.push_back(COp.getClamp() ? COp.getClamp() : undefV(B, Loc, F32Ty));
      mlir::Value Rep = makeBridgeCall(B, Loc, SpvOp::TexSample,
                                       COp.getResult().getType(), Args);
      COp.getResult().replaceAllUsesWith(Rep);
      if (COp.getStatus())
        COp.getStatus().replaceAllUsesWith(i32C(B, Loc, 1));
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::TexGatherOp>(Op)) {
      SmallVector<mlir::Value, 12> Args = {COp.getTex(), COp.getSampler()};
      for (unsigned I = 0; I < 4; ++I)
        Args.push_back(I < COp.getCoords().size() ? COp.getCoords()[I]
                                                  : undefV(B, Loc, F32Ty));
      for (unsigned I = 0; I < 2; ++I)
        Args.push_back(I < COp.getOffsets().size() ? COp.getOffsets()[I]
                                                   : undefV(B, Loc, I32Ty));
      Args.push_back(i32C(B, Loc, COp.getChannel()));
      Args.push_back(COp.getCmpVal() ? COp.getCmpVal() : undefV(B, Loc, F32Ty));
      mlir::Value Rep = makeBridgeCall(B, Loc, SpvOp::TexGather,
                                       COp.getResult().getType(), Args);
      COp.getResult().replaceAllUsesWith(Rep);
      if (COp.getStatus())
        COp.getStatus().replaceAllUsesWith(i32C(B, Loc, 1));
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::TexCalcLodOp>(Op)) {
      SmallVector<mlir::Value, 6> Args = {COp.getTex(), COp.getSampler()};
      for (unsigned I = 0; I < 3; ++I)
        Args.push_back(I < COp.getCoords().size() ? COp.getCoords()[I]
                                                  : undefV(B, Loc, F32Ty));
      Args.push_back(i32C(B, Loc, COp.getClamped() ? 1 : 0));
      mlir::Value Rep =
          makeBridgeCall(B, Loc, SpvOp::TexCalcLod, F32Ty, Args);
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::TexGetDimensionsOp>(Op)) {
      mlir::Value Mip = COp.getMipLevel() ? COp.getMipLevel()
                                          : undefV(B, Loc, I32Ty);
      mlir::Value Rep = makeBridgeCall(B, Loc, SpvOp::TexGetDimensions,
                                       COp.getResult().getType(),
                                       {COp.getHandle(), Mip});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::LoadInputOp>(Op)) {
      mlir::Value VertId =
          COp.getVertexId() ? COp.getVertexId() : i32C(B, Loc, 0);
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::LoadInput, COp.getResult().getType(),
          {i32C(B, Loc, COp.getInputId()), COp.getRow(),
           i8C(B, Loc, COp.getCol()), VertId});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::StoreOutputOp>(Op)) {
      makeBridgeCall(B, Loc, SpvOp::StoreOutput, mlir::Type(),
                     {i32C(B, Loc, COp.getOutputId()), COp.getRow(),
                      i8C(B, Loc, COp.getCol()), COp.getValue()});
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::EmitStreamOp>(Op)) {
      makeBridgeCall(B, Loc, SpvOp::EmitStream, mlir::Type(),
                     {i8C(B, Loc, COp.getStreamId())});
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::CutStreamOp>(Op)) {
      makeBridgeCall(B, Loc, SpvOp::CutStream, mlir::Type(),
                     {i8C(B, Loc, COp.getStreamId())});
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::LoadControlPointOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::LoadControlPoint, COp.getResult().getType(),
          {i32C(B, Loc, COp.getInputId()), COp.getRow(),
           i8C(B, Loc, COp.getCol()), COp.getCpId()});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::LoadPatchConstantOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::LoadPatchConstant, COp.getResult().getType(),
          {i32C(B, Loc, COp.getInputId()), COp.getRow(),
           i8C(B, Loc, COp.getCol())});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::StorePatchConstantOp>(Op)) {
      makeBridgeCall(B, Loc, SpvOp::StorePatchConstant, mlir::Type(),
                     {i32C(B, Loc, COp.getOutputId()), COp.getRow(),
                      i8C(B, Loc, COp.getCol()), COp.getValue()});
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::DiscardOp>(Op)) {
      makeBridgeCall(B, Loc, SpvOp::Discard, mlir::Type(), {COp.getCond()});
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::DerivOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::Deriv, COp.getResult().getType(),
          {COp.getValue(),
           i32C(B, Loc, static_cast<int32_t>(COp.getAxis())),
           i32C(B, Loc, static_cast<int32_t>(COp.getQuality()))});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::SysValOp>(Op)) {
      mlir::Value Rep;
      if (COp.getKind() == ir::SysValKind::FlattenedThreadIdInGroup) {
        Rep = makeBridgeCall(B, Loc, SpvOp::FlattenedThreadIdInGroup,
                             COp.getResult().getType(), {});
      } else {
        Rep = makeBridgeCall(
            B, Loc, SpvOp::SysValRT, COp.getResult().getType(),
            {i32C(B, Loc, static_cast<int32_t>(COp.getKind())),
             i32C(B, Loc, COp.getComponent().value_or(0))});
      }
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::BarrierOp>(Op)) {
      makeBridgeCall(B, Loc, SpvOp::Barrier, mlir::Type(),
                     {i32C(B, Loc, COp.getExecSync() ? 1 : 0),
                      i32C(B, Loc, COp.getTgsmFence() ? 1 : 0),
                      i32C(B, Loc, COp.getDeviceFence() ? 1 : 0)});
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::WaveIsFirstLaneOp>(Op)) {
      mlir::Value Rep =
          makeBridgeCall(B, Loc, SpvOp::WaveIsFirstLane, I1Ty, {});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::WaveVoteOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::WaveVote, I1Ty,
          {COp.getValue(), i32C(B, Loc, static_cast<int32_t>(COp.getKind()))});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::WaveBallotOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(B, Loc, SpvOp::WaveBallot,
                                       COp.getResult().getType(),
                                       {COp.getCond()});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::WaveBitCountOp>(Op)) {
      SmallVector<mlir::Value, 6> Args = {
          COp.getCond(), i32C(B, Loc, static_cast<int32_t>(COp.getKind()))};
      for (mlir::Value Mv : COp.getMask())
        Args.push_back(Mv);
      mlir::Value Rep =
          makeBridgeCall(B, Loc, SpvOp::WaveBitCount, I32Ty, Args);
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::WaveMultiPrefixOp>(Op)) {
      SmallVector<mlir::Value, 6> Args = {
          COp.getValue(), i32C(B, Loc, static_cast<int32_t>(COp.getOp()))};
      for (mlir::Value Mv : COp.getMask())
        Args.push_back(Mv);
      mlir::Value Rep = makeBridgeCall(B, Loc, SpvOp::WaveMultiPrefix,
                                       COp.getResult().getType(), Args);
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::QuadReadLaneAtOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::QuadReadLaneAt, COp.getResult().getType(),
          {COp.getValue(), COp.getQuadLane()});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::QuadOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::QuadOp, COp.getResult().getType(),
          {COp.getValue(), i32C(B, Loc, static_cast<int32_t>(COp.getKind()))});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (mlir::isa<ir::SaturateOp>(Op) && lowerUnaryBridge(SpvOp::Saturate))
      continue;
    if (mlir::isa<ir::FrcOp>(Op) && lowerUnaryBridge(SpvOp::Frc))
      continue;
    if (auto COp = mlir::dyn_cast<ir::FirstBitOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::FirstBit, I32Ty,
          {COp.getValue(), i32C(B, Loc, static_cast<int32_t>(COp.getKind()))});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::BfeOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::Bfe, COp.getResult().getType(),
          {COp.getWidth(), COp.getOffset(), COp.getValue(),
           i32C(B, Loc, COp.getIsSigned() ? 1 : 0)});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::BfiOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::Bfi, COp.getResult().getType(),
          {COp.getWidth(), COp.getOffset(), COp.getInsert(), COp.getBase()});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::DotOp>(Op)) {
      SmallVector<mlir::Value, 8> Args;
      Args.append(COp.getLhs().begin(), COp.getLhs().end());
      Args.append(COp.getRhs().begin(), COp.getRhs().end());
      mlir::Value Rep = makeBridgeCall(B, Loc, SpvOp::Dot,
                                       COp.getResult().getType(), Args);
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::DotPackedOp>(Op)) {
      SmallVector<mlir::Value, 8> Args = {
          i32C(B, Loc, static_cast<int32_t>(COp.getKind())), COp.getAcc()};
      Args.append(COp.getInputs().begin(), COp.getInputs().end());
      mlir::Value Rep = makeBridgeCall(B, Loc, SpvOp::DotPacked,
                                       COp.getResult().getType(), Args);
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::Pack4x8Op>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::Pack4x8, I32Ty,
          {i32C(B, Loc, COp.getMode()), COp.getV0(), COp.getV1(), COp.getV2(),
           COp.getV3()});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::Unpack4x8Op>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::Unpack4x8, COp.getResult().getType(),
          {i32C(B, Loc, COp.getMode()), COp.getPacked()});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (mlir::isa<ir::LegacyF32ToF16Op>(Op) &&
        lowerUnaryBridge(SpvOp::LegacyF32ToF16))
      continue;
    if (mlir::isa<ir::LegacyF16ToF32Op>(Op) &&
        lowerUnaryBridge(SpvOp::LegacyF16ToF32))
      continue;
    if (auto COp = mlir::dyn_cast<ir::MakeDoubleOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::MakeDouble, COp.getResult().getType(),
          {COp.getLo(), COp.getHi()});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::SplitDoubleOp>(Op)) {
      auto RetTy = mlir::LLVM::LLVMStructType::getLiteral(MCtx, {I32Ty, I32Ty});
      mlir::Value Pair = makeBridgeCall(B, Loc, SpvOp::SplitDouble, RetTy,
                                        {COp.getValue()});
      mlir::Value Lo = mlir::LLVM::ExtractValueOp::create(B, Loc, Pair, 0);
      mlir::Value Hi = mlir::LLVM::ExtractValueOp::create(B, Loc, Pair, 1);
      COp.getLo().replaceAllUsesWith(Lo);
      COp.getHi().replaceAllUsesWith(Hi);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::UAddcOp>(Op)) {
      mlir::Type SumTy = COp.getSum().getType();
      auto RetTy = mlir::LLVM::LLVMStructType::getLiteral(MCtx, {SumTy, I1Ty});
      mlir::Value Pair = makeBridgeCall(B, Loc, SpvOp::UAddc, RetTy,
                                        {COp.getLhs(), COp.getRhs()});
      mlir::Value Sum = mlir::LLVM::ExtractValueOp::create(B, Loc, Pair, 0);
      mlir::Value Carry = mlir::LLVM::ExtractValueOp::create(B, Loc, Pair, 1);
      COp.getSum().replaceAllUsesWith(Sum);
      COp.getCarry().replaceAllUsesWith(Carry);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::TraceRayOp>(Op)) {
      makeBridgeCall(B, Loc, SpvOp::TraceRay, mlir::Type(), {COp.getAccel()});
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::RayQueryAllocOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(B, Loc, SpvOp::RayQueryAlloc, PtrTy,
                                       {COp.getConstRayFlags()});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::RayQueryTraceInlineOp>(Op)) {
      makeBridgeCall(
          B, Loc, SpvOp::RayQueryTraceInline, mlir::Type(),
          {COp.getQuery(), COp.getAccel(), COp.getRayFlags(),
           COp.getInstanceMask(), COp.getOriginX(), COp.getOriginY(),
           COp.getOriginZ(), COp.getTMin(), COp.getDirX(), COp.getDirY(),
           COp.getDirZ(), COp.getTMax()});
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::RayQueryProceedOp>(Op)) {
      mlir::Value Rep = makeBridgeCall(B, Loc, SpvOp::RayQueryProceed, I1Ty,
                                       {COp.getQuery()});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::RayQueryAbortOp>(Op)) {
      makeBridgeCall(B, Loc, SpvOp::RayQueryAbort, mlir::Type(),
                     {COp.getQuery()});
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::RayQueryCommitOp>(Op)) {
      mlir::Value THit = COp.getTHit() ? COp.getTHit() : undefV(B, Loc, F32Ty);
      makeBridgeCall(B, Loc, SpvOp::RayQueryCommit, mlir::Type(),
                     {COp.getQuery(),
                      i32C(B, Loc, static_cast<int32_t>(COp.getKind())), THit});
      COp.erase();
      continue;
    }
    if (auto COp = mlir::dyn_cast<ir::RayQueryGetOp>(Op)) {
      mlir::Value Comp =
          COp.getComponent() ? COp.getComponent() : i32C(B, Loc, 0);
      mlir::Value Rep = makeBridgeCall(
          B, Loc, SpvOp::RayQueryGet, COp.getResult().getType(),
          {COp.getQuery(),
           i32C(B, Loc, static_cast<int32_t>(COp.getField())), Comp});
      COp.getResult().replaceAllUsesWith(Rep);
      COp.erase();
      continue;
    }

    return makeError("SPIR-V backend: unsupported MLIR op '{0}'",
                     Op->getName().getStringRef());
  }

  // Erase any unused external function declarations left over from frontend
  // import (e.g. unreferenced runtime stubs) so they do not pollute LLVM IR.
  SmallVector<mlir::LLVM::LLVMFuncOp, 16> UnusedDecls;
  for (auto F : Mod.getOps<mlir::LLVM::LLVMFuncOp>()) {
    if (F.isExternal() && !F.getName().starts_with("__crucible_spv.") &&
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
    return makeError("SPIR-V backend: failed to export MLIR module to LLVM IR");
  return LLVMMod;
}

} // namespace

Expected<std::string> emitSPIRV(mlir::ModuleOp Mod, Reflection &Reflect,
                                const Options &Opts) {
  initializeSPIRVTargetOnce();

  auto StageOrErr = stageTripleSuffix(Reflect.Stage, Opts.StageAsCompute);
  if (!StageOrErr)
    return StageOrErr.takeError();

  LLVMContext LLVMCtx;
  auto MOrErr = lowerMLIRToSPIRVLLVM(Mod, LLVMCtx);
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
    return makeError("SPIR-V backend: no entry function found in module");

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

  // Nest multi-way shared merge blocks (e.g., if/else-if/else chains whose
  // arms all branch directly to a common outer merge block). LLVM's
  // SPIRVStructurizer can otherwise produce invalid selection exits
  // (%_new_exit* escaping an enclosing OpSelectionMerge).
  auto nestSharedMergeBlocks = [](Function &F) {
    bool Changed = true;
    unsigned MergeCounter = 0;
    while (Changed) {
      Changed = false;
      DominatorTree DT(F);
      for (BasicBlock &M : F) {
        if (pred_size(&M) <= 2)
          continue;
        DomTreeNode *MN = DT.getNode(&M);
        if (!MN || !MN->getIDom())
          continue;
        BasicBlock *DomM = MN->getIDom()->getBlock();

        // Collect predecessors that unconditionally branch to M and are
        // dominated by DomM (excluding backedges from M's own subtree).
        SmallVector<BasicBlock *, 8> UncondPreds;
        for (BasicBlock *P : predecessors(&M)) {
          auto *BI = dyn_cast<UncondBrInst>(P->getTerminator());
          if (!BI || BI->getSuccessor(0) != &M)
            continue;
          if (!DT.isReachableFromEntry(P) || DT.dominates(&M, P))
            continue;
          UncondPreds.push_back(P);
        }
        if (UncondPreds.size() < 2)
          continue;

        // Pick a pair (P1, P2) whose nearest common dominator NCD is strictly
        // dominated by DomM, preferring the deepest NCD in the domtree.
        BasicBlock *BestP1 = nullptr, *BestP2 = nullptr;
        unsigned BestDepth = 0;
        for (size_t I = 0; I < UncondPreds.size(); ++I) {
          for (size_t J = I + 1; J < UncondPreds.size(); ++J) {
            BasicBlock *P1 = UncondPreds[I];
            BasicBlock *P2 = UncondPreds[J];
            if (P1 == P2)
              continue;
            BasicBlock *NCD = DT.findNearestCommonDominator(P1, P2);
            if (!NCD || NCD == DomM || !DT.dominates(DomM, NCD))
              continue;
            unsigned Depth = DT.getNode(NCD)->getLevel();
            if (!BestP1 || Depth > BestDepth) {
              BestP1 = P1;
              BestP2 = P2;
              BestDepth = Depth;
            }
          }
        }
        if (!BestP1)
          continue;

        BasicBlock *Mid = BasicBlock::Create(
            F.getContext(), "spv.merge." + std::to_string(MergeCounter++), &F,
            &M);
        IRBuilder<> MB(Mid);
        for (PHINode &PN : M.phis()) {
          int I1 = PN.getBasicBlockIndex(BestP1);
          int I2 = PN.getBasicBlockIndex(BestP2);
          if (I1 < 0 || I2 < 0)
            continue;
          Value *V1 = PN.getIncomingValue(I1);
          Value *V2 = PN.getIncomingValue(I2);
          PHINode *MidPN = MB.CreatePHI(PN.getType(), 2, PN.getName() + ".m");
          MidPN->addIncoming(V1, BestP1);
          MidPN->addIncoming(V2, BestP2);
          PN.removeIncomingValue(BestP1, false);
          PN.removeIncomingValue(BestP2, false);
          PN.addIncoming(MidPN, Mid);
        }
        MB.CreateBr(&M);
        cast<UncondBrInst>(BestP1->getTerminator())->setSuccessor(0, Mid);
        cast<UncondBrInst>(BestP2->getTerminator())->setSuccessor(0, Mid);
        Changed = true;
        break;
      }
    }
  };
  nestSharedMergeBlocks(*Entry);

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
  std::string Result(OutBuf.data(), OutBuf.size());

  // Emit missing SPIR-V capabilities and extensions when referenced by BuiltIn
  // variables (ClipDistance, CullDistance, ViewportIndex, FragStencilRefEXT),
  // as the LLVM SPIR-V backend does not always emit them automatically.
  if (!Opts.EmitSPIRVAsm &&
      (Lowerer.usedClipDistance() || Lowerer.usedCullDistance() ||
       Lowerer.usedViewportIndex() || Lowerer.usedShaderViewportIndex() ||
       Lowerer.usedStencilExport() || Lowerer.usedRuntimeDescriptorArray()) &&
      Result.size() >= 20 && (Result.size() % 4) == 0) {
    const uint32_t *Words = reinterpret_cast<const uint32_t *>(Result.data());
    size_t NumWords = Result.size() / 4;
    if (Words[0] == 0x07230203u) {
      constexpr uint16_t OpExtension = 10;
      constexpr uint16_t OpCapability = 17;
      constexpr uint32_t CapClipDistance = 32;
      constexpr uint32_t CapCullDistance = 33;
      constexpr uint32_t CapMultiViewport = 57;
      constexpr uint32_t CapStencilExportEXT = 5013;
      constexpr uint32_t CapShaderViewportIndex = 70;
      constexpr uint32_t CapRuntimeDescriptorArray = 5302;
      bool HasClip = false, HasCull = false, HasMultiVP = false,
           HasShaderVP = false, HasStencilExp = false, HasRuntimeDescArr = false;
      size_t CapInsertPos = 5;
      for (size_t I = 5; I < NumWords;) {
        uint32_t WC = Words[I] >> 16;
        uint16_t Op = Words[I] & 0xFFFFu;
        if (WC == 0 || I + WC > NumWords)
          break;
        if (Op == OpCapability && WC == 2) {
          uint32_t Cap = Words[I + 1];
          if (Cap == CapClipDistance) HasClip = true;
          if (Cap == CapCullDistance) HasCull = true;
          if (Cap == CapMultiViewport) HasMultiVP = true;
          if (Cap == CapShaderViewportIndex) HasShaderVP = true;
          if (Cap == CapStencilExportEXT) HasStencilExp = true;
          if (Cap == CapRuntimeDescriptorArray) HasRuntimeDescArr = true;
          CapInsertPos = I + WC;
        } else {
          break;
        }
        I += WC;
      }
      SmallVector<uint32_t, 16> ExtraCaps;
      auto pushCap = [&](uint32_t Cap) {
        ExtraCaps.push_back((2u << 16) | OpCapability);
        ExtraCaps.push_back(Cap);
      };
      if (Lowerer.usedClipDistance() && !HasClip) pushCap(CapClipDistance);
      if (Lowerer.usedCullDistance() && !HasCull) pushCap(CapCullDistance);
      if (Lowerer.usedViewportIndex() && !HasMultiVP) pushCap(CapMultiViewport);
      if (Lowerer.usedShaderViewportIndex() && !HasShaderVP)
        pushCap(CapShaderViewportIndex);
      if (Lowerer.usedStencilExport() && !HasStencilExp)
        pushCap(CapStencilExportEXT);
      if (Lowerer.usedRuntimeDescriptorArray() && !HasRuntimeDescArr)
        pushCap(CapRuntimeDescriptorArray);

      SmallVector<uint32_t, 16> ExtraExts;
      auto pushExt = [&](StringRef Name) {
        size_t StrBytes = Name.size() + 1;
        size_t StrWords = (StrBytes + 3) / 4;
        ExtraExts.push_back(
            (static_cast<uint32_t>(1 + StrWords) << 16) | OpExtension);
        for (size_t W = 0; W < StrWords; ++W) {
          uint32_t Word = 0;
          for (size_t B = 0; B < 4; ++B) {
            size_t Idx = W * 4 + B;
            uint8_t Ch = (Idx < Name.size()) ? static_cast<uint8_t>(Name[Idx]) : 0;
            Word |= static_cast<uint32_t>(Ch) << (B * 8);
          }
          ExtraExts.push_back(Word);
        }
      };
      if (Lowerer.usedStencilExport() && !HasStencilExp)
        pushExt("SPV_EXT_shader_stencil_export");

      if (!ExtraCaps.empty() || !ExtraExts.empty()) {
        std::vector<uint32_t> Patched;
        Patched.reserve(NumWords + ExtraCaps.size() + ExtraExts.size());
        Patched.insert(Patched.end(), Words, Words + CapInsertPos);
        Patched.insert(Patched.end(), ExtraCaps.begin(), ExtraCaps.end());
        Patched.insert(Patched.end(), ExtraExts.begin(), ExtraExts.end());
        Patched.insert(Patched.end(), Words + CapInsertPos, Words + NumWords);
        Result.assign(reinterpret_cast<const char *>(Patched.data()),
                      Patched.size() * 4);
      }
    }
  }

  // Patch TextureCubeArray sentinel OpTypeImage (Dim=3D, Arrayed=1) back to
  // Dim=Cube, Arrayed=1 and emit OpCapability SampledCubeArray (45).
  if (!Opts.EmitSPIRVAsm && Lowerer.usedCubeArray() &&
      Result.size() >= 20 && (Result.size() % 4) == 0) {
    uint32_t *Words = reinterpret_cast<uint32_t *>(Result.data());
    size_t NumWords = Result.size() / 4;
    if (Words[0] == 0x07230203u) {
      constexpr uint16_t OpCapability = 17;
      constexpr uint16_t OpTypeImage = 25;
      constexpr uint32_t CapSampledCubeArray = 45;
      bool HasCap = false;
      bool PatchedAny = false;
      size_t CapInsertPos = 5;
      bool InCaps = true;
      for (size_t I = 5; I < NumWords;) {
        uint32_t WC = Words[I] >> 16;
        uint16_t Op = Words[I] & 0xFFFFu;
        if (WC == 0 || I + WC > NumWords)
          break;
        if (InCaps && Op == OpCapability && WC == 2) {
          if (Words[I + 1] == CapSampledCubeArray)
            HasCap = true;
          CapInsertPos = I + WC;
        } else {
          InCaps = false;
        }
        if (Op == OpTypeImage && WC >= 9 && Words[I + 3] == 2u /* 3D */ &&
            Words[I + 5] == 1u /* Arrayed */) {
          Words[I + 3] = 3u; // Dim::Cube
          PatchedAny = true;
        }
        I += WC;
      }
      if (PatchedAny && !HasCap) {
        std::vector<uint32_t> Patched;
        Patched.reserve(NumWords + 2);
        Patched.insert(Patched.end(), Words, Words + CapInsertPos);
        Patched.push_back((2u << 16) | OpCapability);
        Patched.push_back(CapSampledCubeArray);
        Patched.insert(Patched.end(), Words + CapInsertPos, Words + NumWords);
        Result.assign(reinterpret_cast<const char *>(Patched.data()),
                      Patched.size() * 4);
      }
    }
  }

  // Rewrite sentinel __crucible_spv_arraylength.*, __crucible_spv_cmpxchg.*,
  // and __crucible_spv_img_{atomic,cmpxchg}.* calls into native SPIR-V
  // OpArrayLength (68) / OpAtomicCompareExchange (230) / OpImageTexelPointer
  // (98) + OpAtomic* and strip the stub functions.
  if (!Opts.EmitSPIRVAsm &&
      (Lowerer.usedArrayLength() || Lowerer.usedCmpXchg() ||
       Lowerer.usedImageAtomics() || Lowerer.usedGeometryStreamOps()) &&
      Result.size() >= 20 && (Result.size() % 4) == 0) {
    const uint32_t *Words = reinterpret_cast<const uint32_t *>(Result.data());
    size_t NumWords = Result.size() / 4;
    if (Words[0] == 0x07230203u) {
      constexpr uint16_t OpName = 5;
      constexpr uint16_t OpTypeInt = 21;
      constexpr uint16_t OpTypeRuntimeArray = 29;
      constexpr uint16_t OpTypeStruct = 30;
      constexpr uint16_t OpTypePointer = 32;
      constexpr uint16_t OpFunction = 54;
      constexpr uint16_t OpFunctionParameter = 55;
      constexpr uint16_t OpFunctionEnd = 56;
      constexpr uint16_t OpFunctionCall = 57;
      constexpr uint16_t OpLoad = 61;
      constexpr uint16_t OpArrayLength = 68;
      constexpr uint16_t OpDecorate = 71;
      constexpr uint16_t OpMemberDecorate = 72;
      constexpr uint16_t OpCopyObject = 83;
      constexpr uint16_t OpImageTexelPointer = 60;
      constexpr uint16_t OpBitcast = 124;
      constexpr uint16_t OpEmitVertex = 218;
      constexpr uint16_t OpEndPrimitive = 219;
      constexpr uint16_t OpAtomicCompareExchange = 230;
      constexpr uint16_t OpLabel = 248;

      struct ImgHelperInfo {
        uint16_t SpvOp = 0;
        uint32_t Set = 0;
        uint32_t Binding = 0;
      };
      DenseSet<uint32_t> ArrayLenHelperIds;
      DenseSet<uint32_t> CmpXchgHelperIds;
      DenseSet<uint32_t> EmitVertexHelperIds;
      DenseSet<uint32_t> EndPrimitiveHelperIds;
      DenseMap<uint32_t, ImgHelperInfo> ImgAtomicHelpersMap;
      DenseMap<uint32_t, ImgHelperInfo> ImgCmpXchgHelpersMap;
      DenseSet<uint32_t> HelperIds;
      DenseMap<uint32_t, uint32_t> CopyOf;
      DenseMap<uint32_t, uint32_t> VarSet;
      DenseMap<uint32_t, uint32_t> VarBinding;
      DenseSet<uint32_t> RuntimeArrays;
      DenseSet<uint32_t> BlockedStructs;
      DenseSet<uint32_t> DeadBitcasts;
      uint32_t UintTyId = 0;
      for (size_t I = 5; I < NumWords;) {
        uint32_t WC = Words[I] >> 16;
        uint16_t Op = Words[I] & 0xFFFFu;
        if (WC == 0 || I + WC > NumWords)
          break;
        if (Op == OpName && WC >= 3) {
          const char *Str = reinterpret_cast<const char *>(&Words[I + 2]);
          size_t MaxLen = (WC - 2) * 4;
          StringRef Name(Str, strnlen(Str, MaxLen));
          if (Name.starts_with("__crucible_spv_arraylength.")) {
            ArrayLenHelperIds.insert(Words[I + 1]);
            HelperIds.insert(Words[I + 1]);
          } else if (Name.starts_with("__crucible_spv_cmpxchg.")) {
            CmpXchgHelperIds.insert(Words[I + 1]);
            HelperIds.insert(Words[I + 1]);
          } else if (Name.starts_with("__crucible_spv_img_atomic.")) {
            SmallVector<StringRef, 6> Parts;
            Name.drop_front(strlen("__crucible_spv_img_atomic."))
                .split(Parts, '.');
            uint32_t SpvOpNum = 0, SetNum = 0, BindNum = 0;
            if (Parts.size() >= 3 && !Parts[0].getAsInteger(10, SpvOpNum) &&
                !Parts[1].getAsInteger(10, SetNum) &&
                !Parts[2].getAsInteger(10, BindNum)) {
              ImgAtomicHelpersMap[Words[I + 1]] = {
                  static_cast<uint16_t>(SpvOpNum), SetNum, BindNum};
              HelperIds.insert(Words[I + 1]);
            }
          } else if (Name == "__crucible_spv_emit_vertex") {
            EmitVertexHelperIds.insert(Words[I + 1]);
            HelperIds.insert(Words[I + 1]);
          } else if (Name == "__crucible_spv_end_primitive") {
            EndPrimitiveHelperIds.insert(Words[I + 1]);
            HelperIds.insert(Words[I + 1]);
          } else if (Name.starts_with("__crucible_spv_img_cmpxchg.")) {
            SmallVector<StringRef, 5> Parts;
            Name.drop_front(strlen("__crucible_spv_img_cmpxchg."))
                .split(Parts, '.');
            uint32_t SetNum = 0, BindNum = 0;
            if (Parts.size() >= 2 && !Parts[0].getAsInteger(10, SetNum) &&
                !Parts[1].getAsInteger(10, BindNum)) {
              ImgCmpXchgHelpersMap[Words[I + 1]] = {OpAtomicCompareExchange,
                                                    SetNum, BindNum};
              HelperIds.insert(Words[I + 1]);
            }
          }
        } else if ((Op == OpCopyObject || Op == OpBitcast) && WC == 4) {
          CopyOf[Words[I + 2]] = Words[I + 3];
        } else if (Op == OpTypeInt && WC == 4 && Words[I + 2] == 32u &&
                   Words[I + 3] == 0u && UintTyId == 0) {
          UintTyId = Words[I + 1];
        } else if (Op == OpTypeRuntimeArray && WC >= 3) {
          RuntimeArrays.insert(Words[I + 1]);
        } else if (Op == OpDecorate && WC >= 3) {
          if (Words[I + 2] == 2u /*Block*/)
            BlockedStructs.insert(Words[I + 1]);
          else if (Words[I + 2] == 33u /*Binding*/ && WC >= 4)
            VarBinding[Words[I + 1]] = Words[I + 3];
          else if (Words[I + 2] == 34u /*DescriptorSet*/ && WC >= 4)
            VarSet[Words[I + 1]] = Words[I + 3];
        }
        I += WC;
      }

      DenseMap<std::pair<uint32_t, uint32_t>, uint32_t> SetBindToVar;
      for (const auto &[VarId, Bind] : VarBinding) {
        auto It = VarSet.find(VarId);
        uint32_t Set = (It != VarSet.end()) ? It->second : 0u;
        SetBindToVar[{Set, Bind}] = VarId;
      }

      if (!HelperIds.empty()) {
        // Collect all IDs defined inside the removed helper functions, plus
        // any non-Block runtime-array helper structs and dead bitcasts.
        DenseSet<uint32_t> RemovedIds = HelperIds;
        bool InHelper = false;
        for (size_t I = 5; I < NumWords;) {
          uint32_t WC = Words[I] >> 16;
          uint16_t Op = Words[I] & 0xFFFFu;
          if (WC == 0 || I + WC > NumWords)
            break;
          if (InHelper) {
            if (Op == OpFunctionEnd) {
              InHelper = false;
            } else if (Op == OpLabel && WC >= 2) {
              RemovedIds.insert(Words[I + 1]);
            } else if (Op == OpFunctionParameter && WC >= 3) {
              RemovedIds.insert(Words[I + 2]);
            }
          } else if (Op == OpFunction && WC >= 3 &&
                     HelperIds.contains(Words[I + 2])) {
            InHelper = true;
          } else if (Op == OpTypeStruct && WC >= 3 &&
                     !BlockedStructs.contains(Words[I + 1])) {
            for (uint32_t K = 2; K < WC; ++K) {
              if (RuntimeArrays.contains(Words[I + K])) {
                RemovedIds.insert(Words[I + 1]);
                break;
              }
            }
          } else if (Op == OpTypePointer && WC >= 4 &&
                     RemovedIds.contains(Words[I + 3])) {
            RemovedIds.insert(Words[I + 1]);
          } else if (Op == OpFunctionCall && WC == 5 &&
                     ArrayLenHelperIds.contains(Words[I + 3])) {
            uint32_t Arg = Words[I + 4];
            if (CopyOf.contains(Arg))
              DeadBitcasts.insert(Arg);
          }
          I += WC;
        }

        uint32_t Bound = Words[3];
        uint32_t PtrImageUintTyId = 0;
        bool NeedPtrImageUint =
            (!ImgAtomicHelpersMap.empty() || !ImgCmpXchgHelpersMap.empty()) &&
            UintTyId != 0;
        if (NeedPtrImageUint)
          PtrImageUintTyId = Bound++;

        std::vector<uint32_t> Patched;
        Patched.reserve(NumWords + 64);
        for (size_t I = 0; I < 5; ++I)
          Patched.push_back(Words[I]);

        bool SkippingFn = false;
        bool EmittedPtrImageUint = false;
        for (size_t I = 5; I < NumWords;) {
          uint32_t WC = Words[I] >> 16;
          uint16_t Op = Words[I] & 0xFFFFu;
          if (WC == 0 || I + WC > NumWords)
            break;
          if (SkippingFn) {
            if (Op == OpFunctionEnd)
              SkippingFn = false;
            I += WC;
            continue;
          }
          if (Op == OpFunction && WC >= 3 && HelperIds.contains(Words[I + 2])) {
            SkippingFn = true;
            I += WC;
            continue;
          }
          if ((Op == OpName || Op == OpDecorate || Op == OpMemberDecorate) &&
              WC >= 2 && RemovedIds.contains(Words[I + 1])) {
            I += WC;
            continue;
          }
          if ((Op == OpTypeStruct || Op == OpTypePointer) && WC >= 2 &&
              RemovedIds.contains(Words[I + 1])) {
            I += WC;
            continue;
          }
          if (Op == OpBitcast && WC == 4 && DeadBitcasts.contains(Words[I + 2])) {
            I += WC;
            continue;
          }
          if (Op == OpFunctionCall && WC == 5 &&
              ArrayLenHelperIds.contains(Words[I + 3])) {
            uint32_t StructPtr = Words[I + 4];
            for (auto It = CopyOf.find(StructPtr); It != CopyOf.end();
                 It = CopyOf.find(StructPtr))
              StructPtr = It->second;
            Patched.push_back((5u << 16) | OpArrayLength);
            Patched.push_back(Words[I + 1]); // ResultType (uint)
            Patched.push_back(Words[I + 2]); // ResultId
            Patched.push_back(StructPtr);    // StructurePtr (OpVariable)
            Patched.push_back(0u);           // Member 0 (runtime array)
            I += WC;
            continue;
          }
          if (Op == OpFunctionCall && WC == 10 &&
              CmpXchgHelperIds.contains(Words[I + 3])) {
            Patched.push_back((9u << 16) | OpAtomicCompareExchange);
            Patched.push_back(Words[I + 1]); // ResultType (uint)
            Patched.push_back(Words[I + 2]); // ResultId
            Patched.push_back(Words[I + 4]); // Pointer
            Patched.push_back(Words[I + 5]); // MemoryScope
            Patched.push_back(Words[I + 6]); // EqualSemantics
            Patched.push_back(Words[I + 7]); // UnequalSemantics
            Patched.push_back(Words[I + 8]); // Value
            Patched.push_back(Words[I + 9]); // Comparator
            I += WC;
            continue;
          }
          if (Op == OpFunctionCall && WC == 10 &&
              ImgAtomicHelpersMap.contains(Words[I + 3]) && NeedPtrImageUint) {
            const ImgHelperInfo &Info = ImgAtomicHelpersMap[Words[I + 3]];
            uint32_t ImgVarPtr = SetBindToVar.lookup({Info.Set, Info.Binding});
            uint32_t TexelPtrId = Bound++;
            Patched.push_back((6u << 16) | OpImageTexelPointer);
            Patched.push_back(PtrImageUintTyId);
            Patched.push_back(TexelPtrId);
            Patched.push_back(ImgVarPtr);
            Patched.push_back(Words[I + 4]); // Coordinate
            Patched.push_back(Words[I + 5]); // Sample (0)
            Patched.push_back((7u << 16) | Info.SpvOp);
            Patched.push_back(Words[I + 1]); // ResultType (uint)
            Patched.push_back(Words[I + 2]); // ResultId
            Patched.push_back(TexelPtrId);   // Pointer (Image)
            Patched.push_back(Words[I + 6]); // Scope
            Patched.push_back(Words[I + 7]); // Semantics
            Patched.push_back(Words[I + 8]); // Value
            I += WC;
            continue;
          }
          if (Op == OpFunctionCall && WC == 12 &&
              ImgCmpXchgHelpersMap.contains(Words[I + 3]) && NeedPtrImageUint) {
            const ImgHelperInfo &Info = ImgCmpXchgHelpersMap[Words[I + 3]];
            uint32_t ImgVarPtr = SetBindToVar.lookup({Info.Set, Info.Binding});
            uint32_t TexelPtrId = Bound++;
            Patched.push_back((6u << 16) | OpImageTexelPointer);
            Patched.push_back(PtrImageUintTyId);
            Patched.push_back(TexelPtrId);
            Patched.push_back(ImgVarPtr);
            Patched.push_back(Words[I + 4]);  // Coordinate
            Patched.push_back(Words[I + 5]);  // Sample (0)
            Patched.push_back((9u << 16) | OpAtomicCompareExchange);
            Patched.push_back(Words[I + 1]);  // ResultType (uint)
            Patched.push_back(Words[I + 2]);  // ResultId
            Patched.push_back(TexelPtrId);    // Pointer (Image)
            Patched.push_back(Words[I + 6]);  // Scope
            Patched.push_back(Words[I + 7]);  // EqualSemantics
            Patched.push_back(Words[I + 8]);  // UnequalSemantics
            Patched.push_back(Words[I + 9]);  // Value
            Patched.push_back(Words[I + 10]); // Comparator
            I += WC;
            continue;
          }
          if (Op == OpFunctionCall && WC == 4 &&
              EmitVertexHelperIds.contains(Words[I + 3])) {
            Patched.push_back((1u << 16) | OpEmitVertex);
            I += WC;
            continue;
          }
          if (Op == OpFunctionCall && WC == 4 &&
              EndPrimitiveHelperIds.contains(Words[I + 3])) {
            Patched.push_back((1u << 16) | OpEndPrimitive);
            I += WC;
            continue;
          }
          for (uint32_t K = 0; K < WC; ++K)
            Patched.push_back(Words[I + K]);
          if (NeedPtrImageUint && !EmittedPtrImageUint && Op == OpTypeInt &&
              WC == 4 && Words[I + 1] == UintTyId) {
            // Emit OpTypePointer %PtrImageUintTyId Image(11) %UintTyId
            Patched.push_back((4u << 16) | OpTypePointer);
            Patched.push_back(PtrImageUintTyId);
            Patched.push_back(11u); // StorageClass::Image
            Patched.push_back(UintTyId);
            EmittedPtrImageUint = true;
          }
          I += WC;
        }
        Patched[3] = Bound;
        Result.assign(reinterpret_cast<const char *>(Patched.data()),
                      Patched.size() * 4);
      }
    }
  }
  // Emit OpExecutionMode DepthReplacing / DepthGreater / DepthLess /
  // StencilRefReplacingEXT for PS shaders writing SV_Depth* / SV_StencilRef.
  if (!Opts.EmitSPIRVAsm &&
      (Lowerer.getDepthOutputMode() != 0 || Lowerer.usedStencilExport()) &&
      Result.size() >= 20 && (Result.size() % 4) == 0) {
    const uint32_t *Words = reinterpret_cast<const uint32_t *>(Result.data());
    size_t NumWords = Result.size() / 4;
    if (Words[0] == 0x07230203u) {
      constexpr uint16_t OpEntryPoint = 15;
      constexpr uint16_t OpExecutionMode = 16;
      uint32_t EntryId = 0;
      for (size_t I = 5; I < NumWords;) {
        uint32_t WC = Words[I] >> 16;
        uint16_t Op = Words[I] & 0xFFFFu;
        if (WC == 0 || I + WC > NumWords)
          break;
        if (Op == OpEntryPoint && WC >= 3) {
          EntryId = Words[I + 2];
          break;
        }
        I += WC;
      }
      if (EntryId != 0) {
        std::vector<uint32_t> Patched;
        Patched.reserve(NumWords + 9);
        for (size_t I = 0; I < 5; ++I)
          Patched.push_back(Words[I]);
        bool Inserted = false;
        for (size_t I = 5; I < NumWords;) {
          uint32_t WC = Words[I] >> 16;
          uint16_t Op = Words[I] & 0xFFFFu;
          if (WC == 0 || I + WC > NumWords)
            break;
          for (uint32_t K = 0; K < WC; ++K)
            Patched.push_back(Words[I + K]);
          if (!Inserted && Op == OpExecutionMode && WC >= 2 &&
              Words[I + 1] == EntryId) {
            if (Lowerer.getDepthOutputMode() != 0) {
              // OpExecutionMode %entry DepthReplacing (12)
              Patched.push_back((3u << 16) | OpExecutionMode);
              Patched.push_back(EntryId);
              Patched.push_back(12u);
              if (Lowerer.getDepthOutputMode() == 2u) {
                Patched.push_back((3u << 16) | OpExecutionMode);
                Patched.push_back(EntryId);
                Patched.push_back(14u); // DepthGreater
              } else if (Lowerer.getDepthOutputMode() == 3u) {
                Patched.push_back((3u << 16) | OpExecutionMode);
                Patched.push_back(EntryId);
                Patched.push_back(15u); // DepthLess
              }
            }
            if (Lowerer.usedStencilExport()) {
              // OpExecutionMode %entry StencilRefReplacingEXT (5027)
              Patched.push_back((3u << 16) | OpExecutionMode);
              Patched.push_back(EntryId);
              Patched.push_back(5027u);
            }
            Inserted = true;
          }
          I += WC;
        }
        Result.assign(reinterpret_cast<const char *>(Patched.data()),
                      Patched.size() * 4);
      }
    }
  }
  // Patch OpControlBarrier in Hull (TessellationControl) shaders to use
  // Memory=Invocation (4) and Semantics=None (0) per
  // VUID-StandaloneSpirv-ExecutionModel-07320 (GLSL450 forbids Workgroup
  // memory scope in TessellationControl).
  if (!Opts.EmitSPIRVAsm && Reflect.Stage == ShaderStage::Hull &&
      Result.size() >= 20 && (Result.size() % 4) == 0) {
    const uint32_t *Words = reinterpret_cast<const uint32_t *>(Result.data());
    size_t NumWords = Result.size() / 4;
    if (Words[0] == 0x07230203u) {
      constexpr uint16_t OpTypeInt = 21;
      constexpr uint16_t OpConstant = 43;
      constexpr uint16_t OpControlBarrier = 224;
      bool HasBarrier = false;
      uint32_t UintTyId = 0;
      for (size_t I = 5; I < NumWords;) {
        uint32_t WC = Words[I] >> 16;
        uint16_t Op = Words[I] & 0xFFFFu;
        if (WC == 0 || I + WC > NumWords)
          break;
        if (Op == OpTypeInt && WC == 4 && Words[I + 2] == 32u &&
            Words[I + 3] == 0u && UintTyId == 0)
          UintTyId = Words[I + 1];
        else if (Op == OpControlBarrier && WC == 4)
          HasBarrier = true;
        I += WC;
      }
      if (HasBarrier && UintTyId != 0) {
        uint32_t Bound = Words[3];
        uint32_t InvocScopeId = Bound++;
        uint32_t ZeroSemId = Bound++;
        bool EmittedBarrierConsts = false;
        std::vector<uint32_t> Patched;
        Patched.reserve(NumWords + 8);
        for (size_t I = 0; I < 5; ++I)
          Patched.push_back(Words[I]);
        Patched[3] = Bound;
        for (size_t I = 5; I < NumWords;) {
          uint32_t WC = Words[I] >> 16;
          uint16_t Op = Words[I] & 0xFFFFu;
          if (WC == 0 || I + WC > NumWords)
            break;
          if (Op == OpControlBarrier && WC == 4) {
            Patched.push_back(Words[I]);
            Patched.push_back(Words[I + 1]); // Execution = Workgroup
            Patched.push_back(InvocScopeId); // Memory = Invocation (4)
            Patched.push_back(ZeroSemId);    // Semantics = None (0)
            I += WC;
            continue;
          }
          for (uint32_t K = 0; K < WC; ++K)
            Patched.push_back(Words[I + K]);
          if (!EmittedBarrierConsts && Op == OpTypeInt && WC == 4 &&
              Words[I + 1] == UintTyId) {
            Patched.push_back((4u << 16) | OpConstant);
            Patched.push_back(UintTyId);
            Patched.push_back(InvocScopeId);
            Patched.push_back(4u); // Scope::Invocation
            Patched.push_back((4u << 16) | OpConstant);
            Patched.push_back(UintTyId);
            Patched.push_back(ZeroSemId);
            Patched.push_back(0u); // MemorySemantics::None
            EmittedBarrierConsts = true;
          }
          I += WC;
        }
        Result.assign(reinterpret_cast<const char *>(Patched.data()),
                      Patched.size() * 4);
      }
    }
  }
  return Result;
}

} // namespace crucible

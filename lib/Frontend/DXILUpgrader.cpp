//===- DXILUpgrader.cpp - Bitcode load & DXIL intrinsic lifting --*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Parses frozen LLVM 3.7 DXIL bitcode with the upstream BitcodeReader (which
// auto-upgrades typed pointers to opaque pointers), fixes the legacy DXIL
// data layout via ParserCallbacks, runs the upgrade/cleanup pass pipeline,
// and extracts reflection metadata from !dx.shaderModel, !dx.resources and
// !dx.entryPoints.
//
//===----------------------------------------------------------------------===//

#include "Frontend.h"
#include "crucible/Diagnostic.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Scalar/Scalarizer.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/DXILUpgrade.h"
#include "llvm/Transforms/Utils/LowerSwitch.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

using namespace llvm;

namespace crucible {

//===----------------------------------------------------------------------===//
// Data layout upgrade
//===----------------------------------------------------------------------===//

// DXC emits "e-m:e-p:32:32-i1:32-i8:32-i16:32-i32:32-i64:64-f16:32-f32:32-
// f64:64-n8:16:32:64". Modern LLVM requires i8 to be 8-bit aligned (and uses
// natural i16/f16 alignment for the DirectX target), so rewrite the handful
// of stale specifications before module verification runs.
static std::string upgradeDataLayout(StringRef DL) {
  std::string S = DL.str();
  auto Fix = [&S](StringRef From, StringRef To) {
    size_t Pos = S.find(From.str());
    if (Pos != std::string::npos)
      S.replace(Pos, From.size(), To.str());
  };
  Fix("i8:32", "i8:8");
  Fix("i16:32", "i16:16");
  Fix("f16:32", "f16:16");
  return S;
}

//===----------------------------------------------------------------------===//
// Metadata helpers
//===----------------------------------------------------------------------===//

static uint64_t mdUInt(const MDOperand &Op, uint64_t Default = 0) {
  if (auto *CI = mdconst::dyn_extract_or_null<ConstantInt>(Op))
    return CI->getZExtValue();
  return Default;
}

static StringRef mdString(const MDOperand &Op) {
  if (auto *S = dyn_cast_or_null<MDString>(Op.get()))
    return S->getString();
  return {};
}

// DXIL extended property tags (hlsl::DXIL::PropertyTag).
enum : uint32_t {
  kDxilElementTypeTag = 0,
  kDxilStructuredBufferElementSizeTag = 1,
};

static void parseExtendedProps(const MDNode *Props, ResourceBinding &RB) {
  if (!Props)
    return;
  for (unsigned I = 0, E = Props->getNumOperands(); I + 1 < E; I += 2) {
    uint64_t Tag = mdUInt(Props->getOperand(I));
    uint64_t Val = mdUInt(Props->getOperand(I + 1));
    if (Tag == kDxilElementTypeTag)
      RB.ElementType = static_cast<ComponentType>(Val);
    else if (Tag == kDxilStructuredBufferElementSizeTag)
      RB.ElementStride = static_cast<uint32_t>(Val);
  }
}

using ResourceWidths = DenseMap<const Metadata *, unsigned>;

static void parseResourceList(const MDNode *List, ResourceClass Class,
                              const ResourceWidths &Widths,
                              std::vector<ResourceBinding> &Out) {
  if (!List)
    return;
  for (const MDOperand &Op : List->operands()) {
    const auto *Rec = dyn_cast_or_null<MDNode>(Op.get());
    if (!Rec || Rec->getNumOperands() < 7)
      continue;
    ResourceBinding RB;
    RB.Class = Class;
    RB.RecordID = static_cast<uint32_t>(mdUInt(Rec->getOperand(0)));
    RB.Name = mdString(Rec->getOperand(2)).str();
    RB.Space = static_cast<uint32_t>(mdUInt(Rec->getOperand(3)));
    RB.LowerBound = static_cast<uint32_t>(mdUInt(Rec->getOperand(4)));
    RB.RangeSize = static_cast<uint32_t>(mdUInt(Rec->getOperand(5)));

    switch (Class) {
    case ResourceClass::SRV:
      // {id, sym, name, space, lb, size, kind, sampleCount, props}
      RB.Kind = static_cast<ResourceKind>(mdUInt(Rec->getOperand(6)));
      if (Rec->getNumOperands() > 8)
        parseExtendedProps(dyn_cast_or_null<MDNode>(Rec->getOperand(8).get()),
                           RB);
      break;
    case ResourceClass::UAV:
      // {id, sym, name, space, lb, size, kind, gc, counter, rov, props}
      RB.Kind = static_cast<ResourceKind>(mdUInt(Rec->getOperand(6)));
      if (Rec->getNumOperands() > 7)
        RB.GloballyCoherent = mdUInt(Rec->getOperand(7)) != 0;
      if (Rec->getNumOperands() > 8)
        RB.HasCounter = mdUInt(Rec->getOperand(8)) != 0;
      if (Rec->getNumOperands() > 10)
        parseExtendedProps(dyn_cast_or_null<MDNode>(Rec->getOperand(10).get()),
                           RB);
      break;
    case ResourceClass::CBV:
      // {id, sym, name, space, lb, size, byteSize, props}
      RB.Kind = ResourceKind::CBuffer;
      RB.ElementStride = static_cast<uint32_t>(mdUInt(Rec->getOperand(6)));
      break;
    case ResourceClass::Sampler:
      RB.Kind = ResourceKind::Sampler;
      break;
    }
    if (RB.Kind == ResourceKind::TypedBuffer)
      RB.TypedComponents = Widths.lookup(Rec->getOperand(1).get());
    Out.push_back(std::move(RB));
  }
}

static void parseResources(const Module &M, const ResourceWidths &Widths,
                           Reflection &R) {
  const NamedMDNode *Res = M.getNamedMetadata("dx.resources");
  if (!Res || Res->getNumOperands() == 0)
    return;
  const MDNode *Root = Res->getOperand(0);
  if (Root->getNumOperands() < 4)
    return;
  // Root operands: {SRVs, UAVs, CBVs, Samplers}
  parseResourceList(dyn_cast_or_null<MDNode>(Root->getOperand(0).get()),
                    ResourceClass::SRV, Widths, R.Resources);
  parseResourceList(dyn_cast_or_null<MDNode>(Root->getOperand(1).get()),
                    ResourceClass::UAV, Widths, R.Resources);
  parseResourceList(dyn_cast_or_null<MDNode>(Root->getOperand(2).get()),
                    ResourceClass::CBV, Widths, R.Resources);
  parseResourceList(dyn_cast_or_null<MDNode>(Root->getOperand(3).get()),
                    ResourceClass::Sampler, Widths, R.Resources);
}

static ShaderStage parseStage(StringRef Name) {
  return llvm::StringSwitch<ShaderStage>(Name)
      .Case("ps", ShaderStage::Pixel)
      .Case("vs", ShaderStage::Vertex)
      .Case("gs", ShaderStage::Geometry)
      .Case("hs", ShaderStage::Hull)
      .Case("ds", ShaderStage::Domain)
      .Case("cs", ShaderStage::Compute)
      .Case("lib", ShaderStage::Library)
      .Case("ms", ShaderStage::Mesh)
      .Case("as", ShaderStage::Amplification)
      .Default(ShaderStage::Unknown);
}

// DXIL entry-point property tags (hlsl::DXIL::EntryPropsTag).
enum : uint32_t {
  kDxilShaderFlagsTag = 0,
  kDxilNumThreadsTag = 4,
};

static void parseEntryProps(const MDNode *Props, Reflection &R) {
  if (!Props)
    return;
  for (unsigned I = 0, E = Props->getNumOperands(); I + 1 < E; I += 2) {
    uint64_t Tag = mdUInt(Props->getOperand(I));
    if (Tag == kDxilNumThreadsTag) {
      if (const auto *NT =
              dyn_cast_or_null<MDNode>(Props->getOperand(I + 1).get())) {
        for (unsigned D = 0; D < 3 && D < NT->getNumOperands(); ++D)
          R.NumThreads[D] = static_cast<uint32_t>(mdUInt(NT->getOperand(D), 1));
      }
    }
  }
}

static Error parseStageParameters(const MDNode *List,
                                  std::vector<StageParameter> &Out) {
  if (!List)
    return Error::success();
  for (const MDOperand &Operand : List->operands()) {
    auto *N = dyn_cast_or_null<MDNode>(Operand.get());
    if (!N || N->getNumOperands() < 8)
      return makeError("invalid DXIL stage signature element");
    auto *Indices = dyn_cast_or_null<MDNode>(N->getOperand(4).get());
    if (!Indices || Indices->getNumOperands() == 0)
      return makeError("missing DXIL signature semantic index");
    Out.push_back({mdString(N->getOperand(1)).str(),
                   static_cast<uint32_t>(mdUInt(Indices->getOperand(0))),
                   static_cast<uint32_t>(mdUInt(N->getOperand(0))),
                   static_cast<ComponentType>(mdUInt(N->getOperand(2))),
                   static_cast<uint32_t>(mdUInt(N->getOperand(3))),
                   static_cast<uint32_t>(mdUInt(N->getOperand(5))),
                   static_cast<uint32_t>(mdUInt(N->getOperand(6))),
                   static_cast<uint32_t>(mdUInt(N->getOperand(7)))});
  }
  return Error::success();
}

static Expected<Function *> parseEntryPoints(Module &M, Reflection &R,
                                             StringRef EntryOverride) {
  const NamedMDNode *EPs = M.getNamedMetadata("dx.entryPoints");
  if (!EPs || EPs->getNumOperands() == 0)
    return makeError("module has no dx.entryPoints metadata");

  for (const MDNode *EP : EPs->operands()) {
    if (EP->getNumOperands() < 5)
      continue;
    StringRef Name = mdString(EP->getOperand(1));
    if (!EntryOverride.empty() && Name != EntryOverride)
      continue;
    auto *FMD = dyn_cast_or_null<ValueAsMetadata>(EP->getOperand(0).get());
    Function *F =
        FMD ? dyn_cast<Function>(FMD->getValue()) : M.getFunction(Name);
    if (!F)
      continue;
    R.EntryName = Name.str();
    if (R.Stage == ShaderStage::Vertex || R.Stage == ShaderStage::Pixel) {
      if (auto *Sig = dyn_cast_or_null<MDNode>(EP->getOperand(2).get())) {
        if (Sig->getNumOperands() < 2)
          return makeError("invalid DXIL entry signature");
        if (Error E = parseStageParameters(
                dyn_cast_or_null<MDNode>(Sig->getOperand(0).get()), R.Inputs))
          return E;
        if (Error E = parseStageParameters(
                dyn_cast_or_null<MDNode>(Sig->getOperand(1).get()), R.Outputs))
          return E;
      }
    }
    parseEntryProps(dyn_cast_or_null<MDNode>(EP->getOperand(4).get()), R);
    return F;
  }
  return makeError("entry point '{0}' not found in dx.entryPoints",
                   EntryOverride.empty() ? "<default>" : EntryOverride);
}

//===----------------------------------------------------------------------===//
// loadDXIL
//===----------------------------------------------------------------------===//

namespace {
class LowerLongVectorsPass : public PassInfoMixin<LowerLongVectorsPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    auto IsLongVector = [](Type *T) {
      auto *VT = dyn_cast<FixedVectorType>(T);
      return VT && VT->getNumElements() > 4;
    };
    bool HasLongVector = false;
    SmallVector<CallInst *, 4> DXCalls;
    for (Instruction &I : instructions(F)) {
      if (!IsLongVector(I.getType()) &&
          none_of(I.operands(), [&](const Use &U) {
            return IsLongVector(U->getType());
          }))
        continue;
      HasLongVector = true;
      auto *Call = dyn_cast<CallInst>(&I);
      if (Call && Call->getCalledFunction() &&
          Call->getCalledFunction()->getName().starts_with("dx.op."))
        DXCalls.push_back(Call);
    }
    if (!HasLongVector)
      return PreservedAnalyses::all();

    // Expose the DXIL operations as ordinary IR before LLVM scalarizes
    // the surrounding vector arithmetic, extracts, and memory accesses.
    for (CallInst *Call : DXCalls) {
      if (Call->arg_size() != 3)
        continue;
      auto *Opcode = dyn_cast<ConstantInt>(Call->getArgOperand(0));
      Value *A = Call->getArgOperand(1), *B = Call->getArgOperand(2);
      auto *VT = dyn_cast<FixedVectorType>(A->getType());
      if (!Opcode || !VT || !IsLongVector(VT) || B->getType() != VT)
        continue;
      IRBuilder<> Builder(Call);
      Value *Result;
      switch (Opcode->getZExtValue()) {
      case 37: // IMax
      case 38: // IMin
      case 39: // UMax
      case 40: { // UMin
        if (Call->getType() != VT || !VT->getElementType()->isIntegerTy())
          continue;
        auto Pred = Opcode->equalsInt(37) ? CmpInst::ICMP_SGT
                    : Opcode->equalsInt(38) ? CmpInst::ICMP_SLT
                    : Opcode->equalsInt(39) ? CmpInst::ICMP_UGT
                                           : CmpInst::ICMP_ULT;
        Result = Builder.CreateSelect(Builder.CreateICmp(Pred, A, B), A, B);
        break;
      }
      case 311: { // FDot
        Type *Elem = VT->getElementType();
        if (Call->getType() != Elem || !(Elem->isHalfTy() || Elem->isFloatTy()))
          continue;
        Builder.setFastMathFlags(Call->getFastMathFlags());
        Result = nullptr;
        for (unsigned Lane = 0; Lane != VT->getNumElements(); ++Lane) {
          Value *Product =
              Builder.CreateFMul(Builder.CreateExtractElement(A, Lane),
                                 Builder.CreateExtractElement(B, Lane));
          Result = Result ? Builder.CreateFAdd(Result, Product) : Product;
        }
        break;
      }
      default:
        continue;
      }
      Call->replaceAllUsesWith(Result);
      Call->eraseFromParent();
    }
    ScalarizerPassOptions Options;
    Options.ScalarizeLoadStore = true;
    ScalarizerPass(Options).run(F, AM);
    return PreservedAnalyses::none();
  }
};
} // namespace

Expected<LoadedShader> loadDXIL(MemoryBufferRef Buf, LLVMContext &Ctx,
                                StringRef EntryOverride) {
  auto Bitcode = extractDXILBitcode(Buf);
  if (!Bitcode)
    return Bitcode.takeError();

  ParserCallbacks Callbacks(
      [](StringRef, StringRef DL) -> std::optional<std::string> {
        return upgradeDataLayout(DL);
      });
  ResourceWidths Widths;
  Callbacks.MDType = [&](Metadata **MD, unsigned ID, GetTypeByIDTy GetType,
                         GetContainedTypeIDTy GetContained) {
    if (!isa<PointerType>(GetType(ID)))
      return;
    auto *Ty = dyn_cast_or_null<StructType>(GetType(GetContained(ID, 0)));
    if (!Ty || !Ty->hasName() || Ty->isOpaque() || Ty->getNumElements() != 1 ||
        !(Ty->getName().starts_with("class.Buffer<") ||
          Ty->getName().starts_with("class.RWBuffer<")))
      return;
    Type *Elem = Ty->getElementType(0);
    auto *Vec = dyn_cast<FixedVectorType>(Elem);
    unsigned Components = Vec ? Vec->getNumElements() : 1;
    Type *Scalar = Elem->getScalarType();
    if (Components > 4 || !(Scalar->isIntegerTy() || Scalar->isFloatingPointTy()))
      return;
    // Typed undef symbols with different pointees become the same opaque
    // pointer value. Keep the metadata records distinct until reflection.
    *MD = MDNode::getDistinct(Ctx, {*MD});
    Widths.try_emplace(*MD, Components);
  };

  auto ModOrErr = parseBitcodeFile(
      MemoryBufferRef(*Bitcode, Buf.getBufferIdentifier()), Ctx, Callbacks);
  if (!ModOrErr)
    return ModOrErr.takeError();

  LoadedShader Shader;
  Shader.M = std::move(*ModOrErr);
  Module &M = *Shader.M;

  // Reflection metadata must be captured before passes strip it.
  if (const NamedMDNode *SM = M.getNamedMetadata("dx.shaderModel")) {
    if (SM->getNumOperands() > 0) {
      const MDNode *N = SM->getOperand(0);
      if (N->getNumOperands() >= 3) {
        Shader.Reflect.Stage = parseStage(mdString(N->getOperand(0)));
        Shader.Reflect.ShaderModelMajor =
            static_cast<uint32_t>(mdUInt(N->getOperand(1)));
        Shader.Reflect.ShaderModelMinor =
            static_cast<uint32_t>(mdUInt(N->getOperand(2)));
      }
    }
  }
  parseResources(M, Widths, Shader.Reflect);

  auto EntryOrErr = parseEntryPoints(M, Shader.Reflect, EntryOverride);
  if (!EntryOrErr)
    return EntryOrErr.takeError();
  Shader.Entry = *EntryOrErr;

  // Cleanup pipeline: upgrade dx.* constructs, promote allocas, scalarize
  // aggregates, remove switches, and canonicalize the CFG so the emitter sees
  // tidy SSA.
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
  MPM.addPass(DXILUpgradePass());
  FunctionPassManager FPM;
  FPM.addPass(PromotePass());
  FPM.addPass(LowerLongVectorsPass());
  FPM.addPass(SROAPass(SROAOptions::ModifyCFG));
  FPM.addPass(SimplifyCFGPass());
  FPM.addPass(LowerSwitchPass());
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  MPM.run(M, MAM);

  return std::move(Shader);
}

} // namespace crucible

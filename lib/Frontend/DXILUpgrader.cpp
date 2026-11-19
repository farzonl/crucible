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
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
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

  // Infer finite bounds for unbounded arrays (rangeSize == -1) when every
  // createHandle referencing them uses a constant register index.
  for (ResourceBinding &RB : R.Resources) {
    if (RB.RangeSize != 0xFFFFFFFFu)
      continue;
    bool AllConst = true;
    uint32_t MaxReg = RB.LowerBound;
    bool SawAny = false;
    for (const Function &F : M) {
      for (const BasicBlock &BB : F) {
        for (const Instruction &I : BB) {
          const auto *CI = dyn_cast<CallInst>(&I);
          if (!CI || !CI->getCalledFunction())
            continue;
          if (!CI->getCalledFunction()->getName().starts_with(
                  "dx.op.createHandle"))
            continue;
          if (CI->arg_size() < 4)
            continue;
          auto *OpC = dyn_cast<ConstantInt>(CI->getArgOperand(0));
          if (!OpC || OpC->getZExtValue() != 57) // CreateHandle
            continue;
          auto *ClsC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
          auto *IdC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
          if (!ClsC || !IdC ||
              static_cast<ResourceClass>(ClsC->getZExtValue()) != RB.Class ||
              IdC->getZExtValue() != RB.RecordID)
            continue;
          SawAny = true;
          if (auto *RegC = dyn_cast<ConstantInt>(CI->getArgOperand(3))) {
            uint32_t Reg = static_cast<uint32_t>(RegC->getZExtValue());
            if (Reg > MaxReg)
              MaxReg = Reg;
          } else {
            AllConst = false;
            break;
          }
        }
        if (!AllConst)
          break;
      }
      if (!AllConst)
        break;
    }
    if (AllConst && SawAny && MaxReg >= RB.LowerBound)
      RB.RangeSize = MaxReg - RB.LowerBound + 1;
  }
  for (const Function &F : M) {
    if (F.getName().starts_with("dx.op.rayQuery_StateScalar")) {
      for (const User *U : F.users()) {
        if (const auto *CI = dyn_cast<CallInst>(U)) {
          if (CI->arg_size() >= 1) {
            if (const auto *C = dyn_cast<ConstantInt>(CI->getArgOperand(0))) {
              if (C->getZExtValue() == 215) {
                R.HasInstanceContribution = true;
                break;
              }
            }
          }
        }
      }
    }
    if (F.getName().starts_with("dx.op.getDimensions")) {
      for (const ResourceBinding &RB : R.Resources) {
        if (RB.Kind == ResourceKind::StructuredBuffer ||
            RB.Kind == ResourceKind::RawBuffer ||
            RB.Kind == ResourceKind::TypedBuffer) {
          R.HasBufferDimensions = true;
          break;
        }
      }
      break;
    }
  }
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
  kDxilGSStateTag = 1,
  kDxilDSStateTag = 2,
  kDxilHSStateTag = 3,
  kDxilNumThreadsTag = 4,
  kDxilShaderKindTag = 8,
  kDxilMSStateTag = 9,
  kDxilASStateTag = 10,
};

static StringRef unmangleDXILName(StringRef S) {
  if (S.starts_with("\01?")) {
    size_t End = S.find("@@", 2);
    if (End != StringRef::npos)
      return S.slice(2, End);
  }
  return S;
}

static uint32_t getEntryShaderKind(const MDNode *Props) {
  if (!Props)
    return 0;
  for (unsigned I = 0, E = Props->getNumOperands(); I + 1 < E; I += 2) {
    if (mdUInt(Props->getOperand(I)) == kDxilShaderKindTag)
      return static_cast<uint32_t>(mdUInt(Props->getOperand(I + 1)));
  }
  return 0;
}

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
                   static_cast<uint32_t>(mdUInt(N->getOperand(7))), {}});
    for (const MDOperand &Index : Indices->operands())
      Out.back().RowSemanticIndices.push_back(mdUInt(Index));
  }
  return Error::success();
}

static Error parseDomainInterface(const MDNode *Props, const MDNode *Sig,
                                  Reflection &R) {
  if (!Props || !Sig || Sig->getNumOperands() < 3)
    return makeError("missing DXIL domain shader metadata");
  for (unsigned I = 0; I + 1 < Props->getNumOperands(); I += 2) {
    if (mdUInt(Props->getOperand(I)) != kDxilDSStateTag)
      continue;
    auto *State = dyn_cast_or_null<MDNode>(Props->getOperand(I + 1).get());
    if (!State || State->getNumOperands() != 2)
      return makeError("invalid DXIL domain shader state");
    auto Kind = mdUInt(State->getOperand(0));
    auto Count = mdUInt(State->getOperand(1));
    if (Kind != static_cast<unsigned>(TessellationDomain::Isoline) &&
        Kind != static_cast<unsigned>(TessellationDomain::Triangle) &&
        Kind != static_cast<unsigned>(TessellationDomain::Quad))
      return makeError("unsupported DXIL tessellation domain");
    if (Count < 1 || Count > 32)
      return makeError("invalid DXIL domain control-point count");
    R.Domain = DomainInterface{static_cast<TessellationDomain>(Kind),
                               static_cast<uint32_t>(Count), {}};
    if (Error E = parseStageParameters(
            dyn_cast_or_null<MDNode>(Sig->getOperand(2).get()),
            R.Domain->PatchConstants))
      return E;
    for (const auto *Params : {&R.Inputs, &R.Domain->PatchConstants}) {
      unsigned Words = 0;
      for (const auto &P : *Params) {
        if (P.Rows < 1 || P.Rows > 32 || P.Columns < 1 || P.Columns > 4)
          return makeError("invalid DXIL domain signature shape for '{0}'",
                           P.Semantic);
        Words += P.Rows * P.Columns;
        if (Words > 128)
          return makeError("DXIL domain signature exceeds 128 components");
      }
    }
    return Error::success();
  }
  return makeError("missing DXIL domain shader state");
}

static Error parseHullInterface(const MDNode *Props, const MDNode *Sig,
                              Reflection &R) {
  if (!Props || !Sig || Sig->getNumOperands() < 3)
    return makeError("missing DXIL hull shader metadata");
  for (unsigned I = 0; I + 1 < Props->getNumOperands(); I += 2) {
    if (mdUInt(Props->getOperand(I)) != kDxilHSStateTag)
      continue;
    auto *State = dyn_cast_or_null<MDNode>(Props->getOperand(I + 1).get());
    if (!State || State->getNumOperands() < 7)
      return makeError("invalid DXIL hull shader state");
    auto *PCFMD = dyn_cast_or_null<ValueAsMetadata>(State->getOperand(0).get());
    auto *PCF = PCFMD ? dyn_cast<Function>(PCFMD->getValue()) : nullptr;
    if (!PCF)
      return makeError("missing DXIL hull patch-constant function");
    auto InputCP = mdUInt(State->getOperand(1));
    auto OutputCP = mdUInt(State->getOperand(2));
    auto Domain = mdUInt(State->getOperand(3));
    auto Partitioning = mdUInt(State->getOperand(4));
    auto OutputPrim = mdUInt(State->getOperand(5));
    float MaxTess = 64.0f;
    if (auto *C = dyn_cast_or_null<ConstantAsMetadata>(State->getOperand(6).get()))
      if (auto *CFP = dyn_cast<ConstantFP>(C->getValue()))
        MaxTess = CFP->getValueAPF().convertToFloat();
    if (Domain != static_cast<unsigned>(TessellationDomain::Isoline) &&
        Domain != static_cast<unsigned>(TessellationDomain::Triangle) &&
        Domain != static_cast<unsigned>(TessellationDomain::Quad))
      return makeError("unsupported DXIL hull tessellation domain");
    if (InputCP < 1 || InputCP > 32 || OutputCP < 1 || OutputCP > 32)
      return makeError("invalid DXIL hull control-point count");
    HullInterface HI;
    HI.Domain = static_cast<TessellationDomain>(Domain);
    HI.Partitioning = static_cast<TessellationPartitioning>(Partitioning);
    HI.OutputPrimitive = static_cast<TessellationOutputPrimitive>(OutputPrim);
    HI.InputControlPointCount = static_cast<uint32_t>(InputCP);
    HI.OutputControlPointCount = static_cast<uint32_t>(OutputCP);
    HI.MaxTessFactor = MaxTess;
    HI.PatchConstantFuncName = PCF->getName().str();
    if (Error E = parseStageParameters(
            dyn_cast_or_null<MDNode>(Sig->getOperand(2).get()),
            HI.PatchConstants))
      return E;
    R.Hull = std::move(HI);
    return Error::success();
  }
  return makeError("missing DXIL hull shader state");
}

static Error parseGeometryInterface(const MDNode *Props, Reflection &R) {
  if (!Props)
    return makeError("missing DXIL geometry shader metadata");
  for (unsigned I = 0; I + 1 < Props->getNumOperands(); I += 2) {
    if (mdUInt(Props->getOperand(I)) != kDxilGSStateTag)
      continue;
    auto *State = dyn_cast_or_null<MDNode>(Props->getOperand(I + 1).get());
    if (!State || State->getNumOperands() < 5)
      return makeError("invalid DXIL geometry shader state");
    GeometryInterface GI;
    GI.InputPrimitive = static_cast<uint32_t>(mdUInt(State->getOperand(0)));
    GI.MaxVertexCount = static_cast<uint32_t>(mdUInt(State->getOperand(1)));
    GI.InstanceCount = static_cast<uint32_t>(mdUInt(State->getOperand(2)));
    GI.OutputTopology = static_cast<uint32_t>(mdUInt(State->getOperand(3)));
    GI.StreamMask = static_cast<uint32_t>(mdUInt(State->getOperand(4)));
    R.Geometry = std::move(GI);
    return Error::success();
  }
  return makeError("missing DXIL geometry shader state");
}

static Error parseMeshInterface(const MDNode *Props, Reflection &R) {
  if (!Props)
    return makeError("missing DXIL mesh shader metadata");
  for (unsigned I = 0, E = Props->getNumOperands(); I + 1 < E; I += 2) {
    if (mdUInt(Props->getOperand(I)) != kDxilMSStateTag)
      continue;
    auto *State = dyn_cast_or_null<MDNode>(Props->getOperand(I + 1).get());
    if (!State || State->getNumOperands() < 5)
      return makeError("invalid DXIL mesh shader state");
    if (const auto *NT = dyn_cast_or_null<MDNode>(State->getOperand(0).get())) {
      for (unsigned D = 0; D < 3 && D < NT->getNumOperands(); ++D)
        R.NumThreads[D] = static_cast<uint32_t>(mdUInt(NT->getOperand(D), 1));
    }
    MeshInterface MI;
    MI.MaxVertexCount = static_cast<uint32_t>(mdUInt(State->getOperand(1)));
    MI.MaxPrimitiveCount = static_cast<uint32_t>(mdUInt(State->getOperand(2)));
    uint64_t Topo = mdUInt(State->getOperand(3));
    if (Topo == 1)
      MI.Topology = MeshOutputTopology::Line;
    else if (Topo == 2)
      MI.Topology = MeshOutputTopology::Triangle;
    else
      return makeError("unsupported mesh shader output topology {0}", Topo);
    MI.PayloadSize = static_cast<uint32_t>(mdUInt(State->getOperand(4)));
    R.Mesh = std::move(MI);
    return Error::success();
  }
  return makeError("missing DXIL mesh shader state");
}

static Error parseAmplificationInterface(const MDNode *Props, Reflection &R) {
  if (!Props)
    return makeError("missing DXIL amplification shader metadata");
  for (unsigned I = 0, E = Props->getNumOperands(); I + 1 < E; I += 2) {
    if (mdUInt(Props->getOperand(I)) != kDxilASStateTag)
      continue;
    auto *State = dyn_cast_or_null<MDNode>(Props->getOperand(I + 1).get());
    if (!State || State->getNumOperands() < 2)
      return makeError("invalid DXIL amplification shader state");
    if (const auto *NT = dyn_cast_or_null<MDNode>(State->getOperand(0).get())) {
      for (unsigned D = 0; D < 3 && D < NT->getNumOperands(); ++D)
        R.NumThreads[D] = static_cast<uint32_t>(mdUInt(NT->getOperand(D), 1));
    }
    R.Amplification =
        AmplificationInterface{static_cast<uint32_t>(mdUInt(State->getOperand(1)))};
    return Error::success();
  }
  return makeError("missing DXIL amplification shader state");
}

static Expected<Function *> parseEntryPoints(Module &M, Reflection &R,
                                             StringRef EntryOverride) {
  const NamedMDNode *EPs = M.getNamedMetadata("dx.entryPoints");
  if (!EPs || EPs->getNumOperands() == 0)
    return makeError("module has no dx.entryPoints metadata");

  if (R.Stage == ShaderStage::Library) {
    Function *Primary = nullptr;
    for (const MDNode *EP : EPs->operands()) {
      if (EP->getNumOperands() < 5)
        continue;
      auto *FMD = dyn_cast_or_null<ValueAsMetadata>(EP->getOperand(0).get());
      Function *F = FMD ? dyn_cast<Function>(FMD->getValue()) : nullptr;
      if (!F)
        continue;
      StringRef RawName = mdString(EP->getOperand(1));
      StringRef Name = unmangleDXILName(RawName);
      auto *Props = dyn_cast_or_null<MDNode>(EP->getOperand(4).get());
      uint32_t Kind = getEntryShaderKind(Props);
      if (Kind == 7) { // RayGeneration
        if (!EntryOverride.empty() && Name != EntryOverride &&
            RawName != EntryOverride)
          continue;
        if (!Primary) {
          Primary = F;
          R.Stage = ShaderStage::RayGeneration;
          R.EntryName = F->getName().str();
          parseEntryProps(Props, R);
        }
      } else if (Kind == 10 || Kind == 11) { // ClosestHit or Miss
        R.RTShaders.push_back({Name.str(), F->getName().str(), Kind});
      }
    }
    if (Primary)
      return Primary;
    return makeError("no raygeneration entry point found in library");
  }

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
    auto *Sig = dyn_cast_or_null<MDNode>(EP->getOperand(2).get());
    auto *Props = dyn_cast_or_null<MDNode>(EP->getOperand(4).get());
    if (R.Stage == ShaderStage::Vertex || R.Stage == ShaderStage::Pixel ||
        R.Stage == ShaderStage::Domain || R.Stage == ShaderStage::Hull ||
        R.Stage == ShaderStage::Geometry || R.Stage == ShaderStage::Mesh) {
      if (Sig) {
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
    if (R.Stage == ShaderStage::Domain)
      if (Error E = parseDomainInterface(Props, Sig, R))
        return E;
    if (R.Stage == ShaderStage::Hull)
      if (Error E = parseHullInterface(Props, Sig, R))
        return E;
    if (R.Stage == ShaderStage::Geometry)
      if (Error E = parseGeometryInterface(Props, R))
        return E;
    if (R.Stage == ShaderStage::Mesh)
      if (Error E = parseMeshInterface(Props, R))
        return E;
    if (R.Stage == ShaderStage::Amplification)
      if (Error E = parseAmplificationInterface(Props, R))
        return E;
    parseEntryProps(Props, R);
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
    auto HasLongVectorType = [&](Type *T) {
      if (IsLongVector(T))
        return true;
      if (auto *ST = dyn_cast<StructType>(T))
        return any_of(ST->elements(), IsLongVector);
      return false;
    };
    bool HasLongVector = false;
    SmallVector<CallInst *, 8> DXCalls;
    for (Instruction &I : instructions(F)) {
      if (!HasLongVectorType(I.getType()) &&
          none_of(I.operands(), [&](const Use &U) {
            return HasLongVectorType(U->getType());
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

    Module *M = F.getParent();
    auto AdjustBufferCoords = [](IRBuilder<> &Builder, Value *Coord0,
                                 Value *Coord1, uint32_t ByteDelta)
        -> std::pair<Value *, Value *> {
      if (ByteDelta == 0)
        return {Coord0, Coord1};
      if (isa<UndefValue>(Coord1))
        return {Builder.CreateAdd(Coord0, Builder.getInt32(ByteDelta)), Coord1};
      return {Coord0, Builder.CreateAdd(Coord1, Builder.getInt32(ByteDelta))};
    };

    // Expose the DXIL operations as ordinary IR before LLVM scalarizes
    // the surrounding vector arithmetic, extracts, and memory accesses.
    for (CallInst *Call : DXCalls) {
      auto *Opcode =
          Call->arg_size() >= 1 ? dyn_cast<ConstantInt>(Call->getArgOperand(0))
                                : nullptr;
      if (!Opcode)
        continue;
      IRBuilder<> Builder(Call);

      if (Opcode->equalsInt(303) && Call->arg_size() == 5) {
        // RawBufferVectorLoad -> chunked RawBufferLoad (up to 4 lanes each).
        auto *RetST = dyn_cast<StructType>(Call->getType());
        auto *VT = RetST && RetST->getNumElements() == 2
                       ? dyn_cast<FixedVectorType>(RetST->getElementType(0))
                       : nullptr;
        if (!VT)
          continue;
        unsigned N = VT->getNumElements();
        Type *ElemTy = VT->getElementType();
        unsigned ElemBytes = ElemTy->getPrimitiveSizeInBits() / 8;
        if (ElemBytes == 0)
          continue;
        Type *I32Ty = Builder.getInt32Ty();
        Type *I8Ty = Builder.getInt8Ty();
        Type *HandleTy = Call->getArgOperand(1)->getType();
        StructType *ScalarRetTy =
            StructType::get(F.getContext(),
                            {ElemTy, ElemTy, ElemTy, ElemTy, I32Ty});
        FunctionType *LoadFTy = FunctionType::get(
            ScalarRetTy,
            {I32Ty, HandleTy, I32Ty, I32Ty, I8Ty, I32Ty}, false);
        std::string FnName =
            (Call->getCalledFunction()->getName() + ".scalar").str();
        FunctionCallee LoadFn = M->getOrInsertFunction(FnName, LoadFTy);

        Value *Handle = Call->getArgOperand(1);
        Value *Coord0 = Call->getArgOperand(2);
        Value *Coord1 = Call->getArgOperand(3);
        Value *Align = Call->getArgOperand(4);
        Value *Vec = PoisonValue::get(VT);
        Value *Status = Builder.getInt32(0);
        for (unsigned Base = 0; Base < N; Base += 4) {
          unsigned Count = std::min(4u, N - Base);
          uint8_t Mask = static_cast<uint8_t>((1u << Count) - 1u);
          auto [C0, C1] =
              AdjustBufferCoords(Builder, Coord0, Coord1, Base * ElemBytes);
          CallInst *Chunk = Builder.CreateCall(
              LoadFn, {Builder.getInt32(139), Handle, C0, C1,
                       Builder.getInt8(Mask), Align});
          Chunk->setCallingConv(Call->getCallingConv());
          for (unsigned Lane = 0; Lane < Count; ++Lane) {
            Value *Elem = Builder.CreateExtractValue(Chunk, Lane);
            Vec = Builder.CreateInsertElement(Vec, Elem, Base + Lane);
          }
          Status = Builder.CreateExtractValue(Chunk, 4);
        }
        for (User *U : make_early_inc_range(Call->users())) {
          if (auto *EV = dyn_cast<ExtractValueInst>(U)) {
            if (EV->getNumIndices() == 1 && EV->getIndices()[0] == 0) {
              EV->replaceAllUsesWith(Vec);
              EV->eraseFromParent();
              continue;
            }
            if (EV->getNumIndices() == 1 && EV->getIndices()[0] == 1) {
              EV->replaceAllUsesWith(Status);
              EV->eraseFromParent();
              continue;
            }
          }
        }
        if (!Call->use_empty()) {
          Value *Res = PoisonValue::get(RetST);
          Res = Builder.CreateInsertValue(Res, Vec, 0);
          Res = Builder.CreateInsertValue(Res, Status, 1);
          Call->replaceAllUsesWith(Res);
        }
        Call->eraseFromParent();
        continue;
      }

      if (Opcode->equalsInt(304) && Call->arg_size() == 6) {
        // RawBufferVectorStore -> chunked RawBufferStore (up to 4 lanes each).
        Value *Handle = Call->getArgOperand(1);
        Value *Coord0 = Call->getArgOperand(2);
        Value *Coord1 = Call->getArgOperand(3);
        Value *VecVal = Call->getArgOperand(4);
        Value *Align = Call->getArgOperand(5);
        auto *VT = dyn_cast<FixedVectorType>(VecVal->getType());
        if (!VT)
          continue;
        unsigned N = VT->getNumElements();
        Type *ElemTy = VT->getElementType();
        unsigned ElemBytes = ElemTy->getPrimitiveSizeInBits() / 8;
        if (ElemBytes == 0)
          continue;
        Type *I32Ty = Builder.getInt32Ty();
        Type *I8Ty = Builder.getInt8Ty();
        FunctionType *StoreFTy = FunctionType::get(
            Builder.getVoidTy(),
            {I32Ty, Handle->getType(), I32Ty, I32Ty, ElemTy, ElemTy, ElemTy,
             ElemTy, I8Ty, I32Ty},
            false);
        std::string FnName =
            (Call->getCalledFunction()->getName() + ".scalar").str();
        FunctionCallee StoreFn = M->getOrInsertFunction(FnName, StoreFTy);

        for (unsigned Base = 0; Base < N; Base += 4) {
          unsigned Count = std::min(4u, N - Base);
          uint8_t Mask = static_cast<uint8_t>((1u << Count) - 1u);
          auto [C0, C1] =
              AdjustBufferCoords(Builder, Coord0, Coord1, Base * ElemBytes);
          Value *V[4];
          for (unsigned Lane = 0; Lane < 4; ++Lane) {
            V[Lane] = Lane < Count
                          ? Builder.CreateExtractElement(VecVal, Base + Lane)
                          : UndefValue::get(ElemTy);
          }
          CallInst *Chunk = Builder.CreateCall(
              StoreFn, {Builder.getInt32(140), Handle, C0, C1, V[0], V[1],
                        V[2], V[3], Builder.getInt8(Mask), Align});
          Chunk->setCallingConv(Call->getCallingConv());
        }
        Call->eraseFromParent();
        continue;
      }

      if (Call->arg_size() == 3) {
        Value *A = Call->getArgOperand(1), *B = Call->getArgOperand(2);
        auto *VT = dyn_cast<FixedVectorType>(A->getType());
        if (VT && IsLongVector(VT) && B->getType() == VT) {
          Value *Result = nullptr;
          switch (Opcode->getZExtValue()) {
          case 37: // IMax
          case 38: // IMin
          case 39: // UMax
          case 40: { // UMin
            if (Call->getType() == VT && VT->getElementType()->isIntegerTy()) {
              auto Pred = Opcode->equalsInt(37) ? CmpInst::ICMP_SGT
                          : Opcode->equalsInt(38) ? CmpInst::ICMP_SLT
                          : Opcode->equalsInt(39) ? CmpInst::ICMP_UGT
                                                 : CmpInst::ICMP_ULT;
              Result =
                  Builder.CreateSelect(Builder.CreateICmp(Pred, A, B), A, B);
            }
            break;
          }
          case 311: { // FDot
            Type *Elem = VT->getElementType();
            if (Call->getType() == Elem &&
                (Elem->isHalfTy() || Elem->isFloatTy())) {
              Builder.setFastMathFlags(Call->getFastMathFlags());
              for (unsigned Lane = 0; Lane != VT->getNumElements(); ++Lane) {
                Value *Product =
                    Builder.CreateFMul(Builder.CreateExtractElement(A, Lane),
                                       Builder.CreateExtractElement(B, Lane));
                Result = Result ? Builder.CreateFAdd(Result, Product) : Product;
              }
            }
            break;
          }
          default:
            break;
          }
          if (Result) {
            Call->replaceAllUsesWith(Result);
            Call->eraseFromParent();
            continue;
          }
        }
      }

      // Elementwise scalarization for remaining long-vector dx.op.* calls
      // (unary, unaryBits, binary, tertiary, isSpecialFloat, etc.).
      auto *RetVT = dyn_cast<FixedVectorType>(Call->getType());
      if (!RetVT || !IsLongVector(RetVT))
        continue;
      unsigned N = RetVT->getNumElements();
      bool AllArgsCompatible = all_of(Call->args(), [&](const Use &U) {
        auto *ArgVT = dyn_cast<FixedVectorType>(U->getType());
        return !ArgVT || ArgVT->getNumElements() == N;
      });
      if (!AllArgsCompatible)
        continue;

      SmallVector<Type *, 6> ScalarParamTys;
      for (const Use &U : Call->args()) {
        if (auto *ArgVT = dyn_cast<FixedVectorType>(U->getType()))
          ScalarParamTys.push_back(ArgVT->getElementType());
        else
          ScalarParamTys.push_back(U->getType());
      }
      FunctionType *ScalarFTy = FunctionType::get(
          RetVT->getElementType(), ScalarParamTys, false);
      std::string FnName =
          (Call->getCalledFunction()->getName() + ".scalar").str();
      FunctionCallee ScalarFn = M->getOrInsertFunction(FnName, ScalarFTy);

      Value *Vec = PoisonValue::get(RetVT);
      for (unsigned Lane = 0; Lane < N; ++Lane) {
        SmallVector<Value *, 6> ScalarArgs;
        for (const Use &U : Call->args()) {
          if (isa<FixedVectorType>(U->getType()))
            ScalarArgs.push_back(Builder.CreateExtractElement(U.get(), Lane));
          else
            ScalarArgs.push_back(U.get());
        }
        CallInst *ScalarCall = Builder.CreateCall(ScalarFn, ScalarArgs);
        ScalarCall->setCallingConv(Call->getCallingConv());
        if (isa<FPMathOperator>(Call) && isa<FPMathOperator>(ScalarCall))
          ScalarCall->setFastMathFlags(Call->getFastMathFlags());
        Vec = Builder.CreateInsertElement(Vec, ScalarCall, Lane);
      }
      Call->replaceAllUsesWith(Vec);
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
    Type *Pointee = GetType(GetContained(ID, 0));
    while (auto *Arr = dyn_cast_or_null<ArrayType>(Pointee))
      Pointee = Arr->getElementType();
    auto *Ty = dyn_cast_or_null<StructType>(Pointee);
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
    // DXIL typed buffer operations access 64-bit integers as pairs of i32s.
    if (Scalar->isIntegerTy(64))
      Components *= 2;
    if (Components > 4)
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

  // Rewrite dx.op.createHandleForLib -> dx.op.createHandle for lib_6_x shaders.
  if (const NamedMDNode *Res = M.getNamedMetadata("dx.resources");
      Res && Res->getNumOperands() > 0) {
    DenseMap<const GlobalVariable *, const ResourceBinding *> GVMap;
    const MDNode *Root = Res->getOperand(0);
    auto MapClass = [&](unsigned OpIdx, ResourceClass RC) {
      if (OpIdx >= Root->getNumOperands())
        return;
      auto *List = dyn_cast_or_null<MDNode>(Root->getOperand(OpIdx).get());
      if (!List)
        return;
      for (const MDOperand &Op : List->operands()) {
        auto *Rec = dyn_cast_or_null<MDNode>(Op.get());
        if (!Rec || Rec->getNumOperands() < 2)
          continue;
        uint32_t RecID = static_cast<uint32_t>(mdUInt(Rec->getOperand(0)));
        auto *VMD = dyn_cast_or_null<ValueAsMetadata>(Rec->getOperand(1).get());
        auto *GV = VMD ? dyn_cast<GlobalVariable>(VMD->getValue()) : nullptr;
        if (!GV)
          continue;
        for (const ResourceBinding &RB : Shader.Reflect.Resources) {
          if (RB.Class == RC && RB.RecordID == RecID) {
            GVMap[GV] = &RB;
            break;
          }
        }
      }
    };
    MapClass(0, ResourceClass::SRV);
    MapClass(1, ResourceClass::UAV);
    MapClass(2, ResourceClass::CBV);
    MapClass(3, ResourceClass::Sampler);

    SmallVector<CallInst *, 8> LibCalls;
    Type *HandleTy = nullptr;
    for (Function &F : M) {
      if (!F.getName().starts_with("dx.op.createHandleForLib"))
        continue;
      HandleTy = F.getReturnType();
      for (User *U : F.users())
        if (auto *CI = dyn_cast<CallInst>(U))
          LibCalls.push_back(CI);
    }
    if (HandleTy && !LibCalls.empty()) {
      FunctionType *CHTy = FunctionType::get(
          HandleTy,
          {Type::getInt32Ty(Ctx), Type::getInt8Ty(Ctx), Type::getInt32Ty(Ctx),
           Type::getInt32Ty(Ctx), Type::getInt1Ty(Ctx)},
          false);
      FunctionCallee CHFn = M.getOrInsertFunction("dx.op.createHandle", CHTy);
      for (CallInst *CI : LibCalls) {
        auto *LI = dyn_cast<LoadInst>(CI->getArgOperand(1));
        if (!LI)
          continue;
        Value *Ptr = LI->getPointerOperand();
        Value *DynIdx = nullptr;
        if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
          if (GEP->getNumIndices() >= 2)
            DynIdx = GEP->getOperand(2);
          Ptr = GEP->getPointerOperand();
        }
        auto *GV = dyn_cast<GlobalVariable>(Ptr);
        const ResourceBinding *RB = GV ? GVMap.lookup(GV) : nullptr;
        if (!RB)
          continue;
        IRBuilder<> B(CI);
        Value *IdxVal = B.getInt32(RB->LowerBound);
        if (DynIdx) {
          if (DynIdx->getType() != B.getInt32Ty())
            DynIdx = B.CreateZExtOrTrunc(DynIdx, B.getInt32Ty());
          IdxVal = B.CreateAdd(IdxVal, DynIdx);
        }
        CallInst *NewCI = B.CreateCall(
            CHFn, {B.getInt32(57), B.getInt8(static_cast<uint8_t>(RB->Class)),
                   B.getInt32(RB->RecordID), IdxVal, B.getFalse()});
        CI->replaceAllUsesWith(NewCI);
        CI->eraseFromParent();
        if (LI->use_empty())
          LI->eraseFromParent();
      }
      for (auto &[GV, RB] : GVMap) {
        if (GV->use_empty())
          const_cast<GlobalVariable *>(GV)->eraseFromParent();
      }
    }
  }

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
  FPM.addPass(createFunctionToLoopPassAdaptor(LoopFullUnrollPass(3)));
  FPM.addPass(SROAPass(SROAOptions::ModifyCFG));
  FPM.addPass(SimplifyCFGPass());
  FPM.addPass(LowerSwitchPass());
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  MPM.run(M, MAM);

  return std::move(Shader);
}

} // namespace crucible

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

struct BoundCBufferInfo {
  size_t ReflectIndex = 0;
  mlir::spirv::StructType StructTy;
};

struct WorkgroupGlobalInfo {
  std::string Name;
  mlir::Type LLVMType;
};

struct BoundImageInfo {
  size_t ReflectIndex = 0;
  ResourceKind Kind = ResourceKind::Invalid;
  mlir::Type ElemType;
  bool IsUAV = false;
};

struct StageIOInfo {
  uint32_t ID = 0;
  bool IsInput = true;
  mlir::Type ValueType;
  uint32_t Columns = 1;
};

// Split a DXC SPIR-V stage variable name like "in.var.COLOR2" or
// "out.var.SV_Target0" into (semantic, index).
std::pair<std::string, uint32_t> parseStageVarSemantic(StringRef SymName,
                                                       uint32_t DefaultIdx) {
  StringRef S = SymName;
  if (S.starts_with("in.var."))
    S = S.drop_front(7);
  else if (S.starts_with("out.var."))
    S = S.drop_front(8);
  size_t End = S.size();
  while (End > 0 && std::isdigit(static_cast<unsigned char>(S[End - 1])))
    --End;
  if (End < S.size() && End > 0) {
    uint32_t Idx = 0;
    if (!S.substr(End).getAsInteger(10, Idx))
      return {S.substr(0, End).str(), Idx};
  }
  return {S.str(), DefaultIdx};
}

ResourceKind mapImageKind(mlir::spirv::ImageType ImgTy) {
  bool Arrayed = ImgTy.getArrayedInfo() == mlir::spirv::ImageArrayedInfo::Arrayed;
  switch (ImgTy.getDim()) {
  case mlir::spirv::Dim::Buffer:
    return ResourceKind::TypedBuffer;
  case mlir::spirv::Dim::Dim1D:
    return Arrayed ? ResourceKind::Texture1DArray : ResourceKind::Texture1D;
  case mlir::spirv::Dim::Dim2D:
    return Arrayed ? ResourceKind::Texture2DArray : ResourceKind::Texture2D;
  case mlir::spirv::Dim::Dim3D:
    return ResourceKind::Texture3D;
  case mlir::spirv::Dim::Cube:
    return Arrayed ? ResourceKind::TextureCubeArray : ResourceKind::TextureCube;
  default:
    return ResourceKind::Invalid;
  }
}

uint32_t imageFormatComponents(mlir::spirv::ImageFormat Fmt) {
  switch (Fmt) {
  case mlir::spirv::ImageFormat::R32f:
  case mlir::spirv::ImageFormat::R16f:
  case mlir::spirv::ImageFormat::R32i:
  case mlir::spirv::ImageFormat::R32ui:
  case mlir::spirv::ImageFormat::R16i:
  case mlir::spirv::ImageFormat::R16ui:
  case mlir::spirv::ImageFormat::R8i:
  case mlir::spirv::ImageFormat::R8ui:
  case mlir::spirv::ImageFormat::R8:
  case mlir::spirv::ImageFormat::R16:
  case mlir::spirv::ImageFormat::R8Snorm:
  case mlir::spirv::ImageFormat::R16Snorm:
    return 1;
  case mlir::spirv::ImageFormat::Rg32f:
  case mlir::spirv::ImageFormat::Rg16f:
  case mlir::spirv::ImageFormat::Rg32i:
  case mlir::spirv::ImageFormat::Rg32ui:
  case mlir::spirv::ImageFormat::Rg16i:
  case mlir::spirv::ImageFormat::Rg16ui:
  case mlir::spirv::ImageFormat::Rg8i:
  case mlir::spirv::ImageFormat::Rg8ui:
  case mlir::spirv::ImageFormat::Rg8:
  case mlir::spirv::ImageFormat::Rg16:
  case mlir::spirv::ImageFormat::Rg8Snorm:
  case mlir::spirv::ImageFormat::Rg16Snorm:
    return 2;
  default:
    return 4;
  }
}

uint32_t unshiftVulkanBinding(ResourceClass RC, uint32_t Binding) {
  if (RC == ResourceClass::SRV && Binding >= 10000u && Binding < 20000u)
    return Binding - 10000u;
  if (RC == ResourceClass::UAV && Binding >= 20000u && Binding < 30000u)
    return Binding - 20000u;
  if (RC == ResourceClass::Sampler && Binding >= 30000u)
    return Binding - 30000u;
  return Binding;
}

mlir::Type convertSpirvDataType(mlir::Type T, mlir::OpBuilder &B) {
  if (auto IT = mlir::dyn_cast<mlir::IntegerType>(T))
    return B.getIntegerType(IT.getWidth());
  if (auto Vec = mlir::dyn_cast<mlir::VectorType>(T))
    return mlir::VectorType::get(Vec.getShape(),
                                 convertSpirvDataType(Vec.getElementType(), B));
  if (auto Mat = mlir::dyn_cast<mlir::spirv::MatrixType>(T))
    return mlir::LLVM::LLVMArrayType::get(
        convertSpirvDataType(Mat.getColumnType(), B), Mat.getNumColumns());
  if (auto Arr = mlir::dyn_cast<mlir::spirv::ArrayType>(T))
    return mlir::LLVM::LLVMArrayType::get(
        convertSpirvDataType(Arr.getElementType(), B), Arr.getNumElements());
  if (auto ST = mlir::dyn_cast<mlir::spirv::StructType>(T)) {
    SmallVector<mlir::Type, 4> Elems;
    for (unsigned I = 0; I < ST.getNumElements(); ++I)
      Elems.push_back(convertSpirvDataType(ST.getElementType(I), B));
    return mlir::LLVM::LLVMStructType::getLiteral(B.getContext(), Elems);
  }
  return T;
}

uint32_t scalarTypeBytes(mlir::Type T) {
  if (T.isF16() || T.isInteger(16))
    return 2;
  if (T.isF64() || T.isInteger(64))
    return 8;
  return 4;
}

uint32_t spirvTypeBytes(mlir::Type T) {
  if (auto Vec = mlir::dyn_cast<mlir::VectorType>(T))
    return static_cast<uint32_t>(Vec.getNumElements()) *
           scalarTypeBytes(Vec.getElementType());
  if (auto Mat = mlir::dyn_cast<mlir::spirv::MatrixType>(T))
    return Mat.getNumColumns() * 16u;
  if (auto Arr = mlir::dyn_cast<mlir::spirv::ArrayType>(T)) {
    uint32_t Stride = Arr.getArrayStride();
    if (Stride == 0)
      Stride = spirvTypeBytes(Arr.getElementType());
    return Arr.getNumElements() * Stride;
  }
  if (auto ST = mlir::dyn_cast<mlir::spirv::StructType>(T)) {
    unsigned N = ST.getNumElements();
    if (N == 0)
      return 0;
    uint32_t LastOff = ST.getMemberOffset(N - 1);
    return LastOff + spirvTypeBytes(ST.getElementType(N - 1));
  }
  return scalarTypeBytes(T);
}

std::optional<int64_t> getConstSpirvInt(mlir::Value V) {
  if (auto C = V.getDefiningOp<mlir::spirv::ConstantOp>()) {
    if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue()))
      return IA.getInt();
  }
  return std::nullopt;
}

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
  if (auto Vec = mlir::dyn_cast<mlir::VectorType>(T)) {
    mlir::Type E = makeSignless(Vec.getElementType(), B);
    if (E != Vec.getElementType())
      return mlir::VectorType::get(Vec.getShape(), E);
  }
  return T;
}

mlir::Value castToSignless(mlir::OpBuilder &B, mlir::Location Loc,
                           mlir::Value V) {
  mlir::Type DstTy = makeSignless(V.getType(), B);
  if (DstTy == V.getType())
    return V;
  return mlir::spirv::BitcastOp::create(B, Loc, DstTy, V);
}

mlir::Value castFromSignless(mlir::OpBuilder &B, mlir::Location Loc,
                             mlir::Value V, mlir::Type OrigTy) {
  if (OrigTy == V.getType())
    return V;
  return mlir::spirv::BitcastOp::create(B, Loc, OrigTy, V);
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

mlir::LLVM::LLVMStructType getOrInitCBufRetType(mlir::MLIRContext *Ctx,
                                                mlir::Type ElemTy) {
  std::string Name = "dx.types.CBufRet." + typeSuffix(ElemTy);
  auto ST = mlir::LLVM::LLVMStructType::getIdentified(Ctx, Name);
  if (!ST.isInitialized()) {
    if (ElemTy.isF16() || ElemTy.isInteger(16))
      (void)ST.setBody(
          {ElemTy, ElemTy, ElemTy, ElemTy, ElemTy, ElemTy, ElemTy, ElemTy},
          /*isPacked=*/false);
    else if (ElemTy.isF64() || ElemTy.isInteger(64))
      (void)ST.setBody({ElemTy, ElemTy}, /*isPacked=*/false);
    else
      (void)ST.setBody({ElemTy, ElemTy, ElemTy, ElemTy}, /*isPacked=*/false);
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

  // 2. Inspect spirv.GlobalVariables for built-ins, workgroup memory, stage
  // I/O, textures/samplers, and descriptor bindings.
  StringMap<mlir::spirv::BuiltIn> BuiltInGlobals;
  StringMap<BoundBufferInfo> BufferGlobals;
  StringMap<BoundCBufferInfo> CBufferGlobals;
  StringMap<WorkgroupGlobalInfo> WorkgroupGlobals;
  StringMap<BoundImageInfo> ImageGlobals;
  StringMap<size_t> SamplerGlobals;
  StringMap<StageIOInfo> StageIOGlobals;
  SmallVector<mlir::spirv::GlobalVariableOp, 8> GlobalsToErase;
  uint32_t NextSRVId = 0, NextUAVId = 0, NextCBVId = 0, NextSamplerId = 0;
  uint32_t NextInputID = 0, NextOutputID = 0;
  bool IsGraphicsStage = Reflect.Stage == ShaderStage::Vertex ||
                         Reflect.Stage == ShaderStage::Pixel;

  for (auto GV : SpvModule.getOps<mlir::spirv::GlobalVariableOp>()) {
    auto PtrTy = mlir::dyn_cast<mlir::spirv::PointerType>(GV.getType());
    if (!PtrTy)
      continue;
    auto SC = PtrTy.getStorageClass();

    if (auto BIName = GV.getBuiltIn()) {
      auto BI = mlir::spirv::symbolizeBuiltIn(*BIName);
      if (!BI)
        return makeError("unknown SPIR-V built_in '{0}'", *BIName);

      // Graphics stage built-ins map to StageParameters (Input / Output).
      if (IsGraphicsStage &&
          (SC == mlir::spirv::StorageClass::Input ||
           SC == mlir::spirv::StorageClass::Output)) {
        bool IsInput = SC == mlir::spirv::StorageClass::Input;
        mlir::Type ValTy = PtrTy.getPointeeType();
        mlir::Type ElemTy = ValTy;
        uint32_t Cols = 1;
        if (auto Vec = mlir::dyn_cast<mlir::VectorType>(ValTy)) {
          ElemTy = Vec.getElementType();
          Cols = static_cast<uint32_t>(Vec.getNumElements());
        }
        StageParameter SP{};
        SP.ID = IsInput ? NextInputID++ : NextOutputID++;
        SP.Type = toComponentType(ElemTy);
        SP.Rows = 1;
        SP.Columns = Cols;
        SP.Interpolation = 2; // Linear
        switch (*BI) {
        case mlir::spirv::BuiltIn::Position:
        case mlir::spirv::BuiltIn::FragCoord:
          SP.Semantic = "SV_Position";
          SP.Kind = 3; // Position
          break;
        case mlir::spirv::BuiltIn::VertexIndex:
          SP.Semantic = "SV_VertexID";
          SP.Kind = 1;
          break;
        case mlir::spirv::BuiltIn::InstanceIndex:
          SP.Semantic = "SV_InstanceID";
          SP.Kind = 2;
          break;
        case mlir::spirv::BuiltIn::PrimitiveId:
          SP.Semantic = "SV_PrimitiveID";
          SP.Kind = 7;
          break;
        case mlir::spirv::BuiltIn::FrontFacing:
          SP.Semantic = "SV_IsFrontFace";
          SP.Kind = 13;
          break;
        case mlir::spirv::BuiltIn::FragDepth:
          SP.Semantic = "SV_Depth";
          SP.Kind = 17;
          break;
        default:
          BuiltInGlobals[GV.getSymName()] = *BI;
          GlobalsToErase.push_back(GV);
          continue;
        }
        StageIOInfo IO{SP.ID, IsInput, ValTy, Cols};
        StageIOGlobals[GV.getSymName()] = IO;
        if (IsInput)
          Reflect.Inputs.push_back(std::move(SP));
        else
          Reflect.Outputs.push_back(std::move(SP));
        GlobalsToErase.push_back(GV);
        continue;
      }

      BuiltInGlobals[GV.getSymName()] = *BI;
      GlobalsToErase.push_back(GV);
      continue;
    }

    // Graphics stage user varyings (Input / Output with location attribute).
    if (IsGraphicsStage &&
        (SC == mlir::spirv::StorageClass::Input ||
         SC == mlir::spirv::StorageClass::Output)) {
      bool IsInput = SC == mlir::spirv::StorageClass::Input;
      uint32_t LocAttr = 0;
      if (auto LA = GV->getAttrOfType<mlir::IntegerAttr>("location"))
        LocAttr = static_cast<uint32_t>(LA.getInt());
      mlir::Type ValTy = PtrTy.getPointeeType();
      mlir::Type ElemTy = ValTy;
      uint32_t Cols = 1;
      if (auto Vec = mlir::dyn_cast<mlir::VectorType>(ValTy)) {
        ElemTy = Vec.getElementType();
        Cols = static_cast<uint32_t>(Vec.getNumElements());
      }
      auto [Sem, SemIdx] = parseStageVarSemantic(GV.getSymName(), LocAttr);
      StageParameter SP{};
      SP.ID = IsInput ? NextInputID++ : NextOutputID++;
      SP.Semantic = Sem;
      SP.SemanticIndex = SemIdx;
      SP.Type = toComponentType(ElemTy);
      SP.Rows = 1;
      SP.Columns = Cols;
      SP.Interpolation = 2; // Linear
      SP.RowSemanticIndices = {SemIdx};
      if (!IsInput && Reflect.Stage == ShaderStage::Pixel) {
        SP.Semantic = "SV_Target";
        SP.SemanticIndex = LocAttr;
        SP.Kind = 16; // Target
      } else {
        SP.Kind = 0; // Arbitrary
      }
      StageIOInfo IO{SP.ID, IsInput, ValTy, Cols};
      StageIOGlobals[GV.getSymName()] = IO;
      if (IsInput)
        Reflect.Inputs.push_back(std::move(SP));
      else
        Reflect.Outputs.push_back(std::move(SP));
      GlobalsToErase.push_back(GV);
      continue;
    }

    if (SC == mlir::spirv::StorageClass::Workgroup) {
      mlir::OpBuilder B(&Ctx);
      WorkgroupGlobalInfo Info;
      Info.Name = GV.getSymName().str();
      Info.LLVMType = convertSpirvDataType(PtrTy.getPointeeType(), B);
      WorkgroupGlobals[GV.getSymName()] = std::move(Info);
      GlobalsToErase.push_back(GV);
      continue;
    }

    auto DS = GV.getDescriptorSet();
    auto Binding = GV.getBinding();
    if (!DS || !Binding)
      continue;

    // UniformConstant images (typed buffers & textures) and samplers.
    if (SC == mlir::spirv::StorageClass::UniformConstant) {
      if (auto ImgTy =
              mlir::dyn_cast<mlir::spirv::ImageType>(PtrTy.getPointeeType())) {
        ResourceKind Kind = mapImageKind(ImgTy);
        if (Kind == ResourceKind::Invalid)
          continue;
        bool IsUAV = ImgTy.getSamplerUseInfo() ==
                     mlir::spirv::ImageSamplerUseInfo::NoSampler;
        ResourceBinding RB;
        RB.Class = IsUAV ? ResourceClass::UAV : ResourceClass::SRV;
        RB.Kind = Kind;
        RB.ElementType = toComponentType(ImgTy.getElementType());
        RB.RecordID = IsUAV ? NextUAVId++ : NextSRVId++;
        RB.Name = GV.getSymName().str();
        RB.Space = *DS;
        RB.LowerBound = unshiftVulkanBinding(RB.Class, *Binding);
        RB.RangeSize = 1;
        if (Kind == ResourceKind::TypedBuffer)
          RB.TypedComponents = imageFormatComponents(ImgTy.getImageFormat());

        BoundImageInfo Info;
        Info.ReflectIndex = Reflect.Resources.size();
        Info.Kind = Kind;
        Info.ElemType = ImgTy.getElementType();
        Info.IsUAV = IsUAV;
        ImageGlobals[GV.getSymName()] = Info;
        Reflect.Resources.push_back(std::move(RB));
        GlobalsToErase.push_back(GV);
        continue;
      }
      if (mlir::isa<mlir::spirv::SamplerType>(PtrTy.getPointeeType())) {
        ResourceBinding RB;
        RB.Class = ResourceClass::Sampler;
        RB.Kind = ResourceKind::Sampler;
        RB.RecordID = NextSamplerId++;
        RB.Name = GV.getSymName().str();
        RB.Space = *DS;
        RB.LowerBound = unshiftVulkanBinding(RB.Class, *Binding);
        RB.RangeSize = 1;
        SamplerGlobals[GV.getSymName()] = Reflect.Resources.size();
        Reflect.Resources.push_back(std::move(RB));
        GlobalsToErase.push_back(GV);
        continue;
      }
      continue;
    }

    if (SC != mlir::spirv::StorageClass::Uniform &&
        SC != mlir::spirv::StorageClass::StorageBuffer)
      continue;

    auto StructTy =
        mlir::dyn_cast<mlir::spirv::StructType>(PtrTy.getPointeeType());
    if (!StructTy || StructTy.getNumElements() == 0)
      continue;

    if (auto RTArr = mlir::dyn_cast<mlir::spirv::RuntimeArrayType>(
            StructTy.getElementType(0))) {
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
      RB.LowerBound = unshiftVulkanBinding(RB.Class, *Binding);
      RB.RangeSize = 1;
      RB.ElementStride = Stride;

      BoundBufferInfo Info;
      Info.ReflectIndex = Reflect.Resources.size();
      Info.ElemType = ElemTy;
      Info.ElemStride = Stride;
      BufferGlobals[GV.getSymName()] = Info;
      Reflect.Resources.push_back(std::move(RB));
      GlobalsToErase.push_back(GV);
      continue;
    }

    // Uniform Block struct without RuntimeArray -> Constant Buffer (CBV).
    if (SC == mlir::spirv::StorageClass::Uniform) {
      uint32_t RawBytes = spirvTypeBytes(StructTy);
      uint32_t PaddedBytes = ((RawBytes + 15u) / 16u) * 16u;
      if (PaddedBytes == 0)
        PaddedBytes = 16;

      ResourceBinding RB;
      RB.Class = ResourceClass::CBV;
      RB.Kind = ResourceKind::CBuffer;
      RB.RecordID = NextCBVId++;
      RB.Name = GV.getSymName().str();
      RB.Space = *DS;
      RB.LowerBound = unshiftVulkanBinding(RB.Class, *Binding);
      RB.RangeSize = 1;
      RB.ElementStride = PaddedBytes;

      BoundCBufferInfo Info;
      Info.ReflectIndex = Reflect.Resources.size();
      Info.StructTy = StructTy;
      CBufferGlobals[GV.getSymName()] = Info;
      Reflect.Resources.push_back(std::move(RB));
      GlobalsToErase.push_back(GV);
      continue;
    }
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
  auto getHandleForIndex = [&](size_t ReflectIndex) -> mlir::Value {
    auto It = HandleCache.find(ReflectIndex);
    if (It != HandleCache.end())
      return It->second;

    EntryBuilder.setInsertionPointToStart(&EntryFn.getBody().front());
    const ResourceBinding &RB = Reflect.Resources[ReflectIndex];
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
    HandleCache[ReflectIndex] = Call.getResult();
    return Call.getResult();
  };

  SmallVector<mlir::spirv::AddressOfOp, 8> AddrOps;
  EntryFn.walk([&](mlir::spirv::AddressOfOp AO) { AddrOps.push_back(AO); });
  DenseSet<mlir::Operation *> ErasedAddrOps;

  for (auto AO : AddrOps) {
    if (ErasedAddrOps.contains(AO))
      continue;
    StringRef Sym = AO.getVariable();

    // 3a. Built-in globals (vector IDs -> gpu.*, LocalInvocationIndex -> gidx).
    auto BIIt = BuiltInGlobals.find(Sym);
    if (BIIt != BuiltInGlobals.end()) {
      mlir::spirv::BuiltIn BI = BIIt->second;
      SmallVector<mlir::Operation *, 4> Loads(AO->getUsers().begin(),
                                              AO->getUsers().end());
      for (mlir::Operation *U : Loads) {
        if (auto Chain = mlir::dyn_cast<mlir::spirv::AccessChainOp>(U)) {
          if (Chain.getIndices().size() != 1)
            return makeError("unsupported access chain on built-in '{0}'", Sym);
          auto ConstComp = getConstSpirvInt(Chain.getIndices()[0]);
          if (!ConstComp)
            return makeError("dynamic index on built-in '{0}'", Sym);
          uint32_t Comp = static_cast<uint32_t>(*ConstComp);
          SmallVector<mlir::Operation *, 4> CLoads(Chain->getUsers().begin(),
                                                   Chain->getUsers().end());
          for (mlir::Operation *CLU : CLoads) {
            auto CLoad = mlir::dyn_cast<mlir::spirv::LoadOp>(CLU);
            if (!CLoad)
              return makeError("unsupported user of built-in chain '{0}'", Sym);
            mlir::OpBuilder B(CLoad);
            mlir::Value Rep =
                emitGPUIdComponent(B, CLoad.getLoc(), BI, Comp, CLoad.getType());
            if (!Rep)
              return makeError("unsupported SPIR-V built-in '{0}'", Sym);
            Rep = castFromSignless(B, CLoad.getLoc(), Rep, CLoad.getType());
            CLoad.replaceAllUsesWith(Rep);
            CLoad.erase();
          }
          Chain.erase();
          continue;
        }
        auto Load = mlir::dyn_cast<mlir::spirv::LoadOp>(U);
        if (!Load)
          return makeError("unsupported user of built-in global '{0}'", Sym);
        if (BI == mlir::spirv::BuiltIn::SubgroupLocalInvocationId ||
            BI == mlir::spirv::BuiltIn::SubgroupSize) {
          mlir::OpBuilder B(Load);
          mlir::Location Loc = Load.getLoc();
          mlir::Value IdxVal =
              BI == mlir::spirv::BuiltIn::SubgroupLocalInvocationId
                  ? mlir::Value(mlir::gpu::LaneIdOp::create(B, Loc, /*upper_bound=*/nullptr))
                  : mlir::Value(mlir::gpu::SubgroupSizeOp::create(B, Loc, /*upper_bound=*/nullptr));
          mlir::Type DstTy = makeSignless(Load.getType(), B);
          auto Cast =
              mlir::arith::IndexCastUIOp::create(B, Loc, DstTy, IdxVal);
          Load.replaceAllUsesWith(Cast.getResult());
          Load.erase();
          continue;
        }
        if (BI == mlir::spirv::BuiltIn::LocalInvocationIndex) {
          mlir::OpBuilder B(Load);
          mlir::Location Loc = Load.getLoc();
          auto FnTy = mlir::LLVM::LLVMFunctionType::get(I32Ty, {I32Ty});
          getOrCreateDecl(SpvModule, "dx.op.flattenedThreadIdInGroup.i32", FnTy);
          auto OpC = mlir::LLVM::ConstantOp::create(
              B, Loc, I32Ty,
              static_cast<int32_t>(DXILOp::FlattenedThreadIdInGroup));
          auto Call = mlir::LLVM::CallOp::create(
              B, Loc, mlir::TypeRange{I32Ty},
              "dx.op.flattenedThreadIdInGroup.i32", mlir::ValueRange{OpC});
          Load.replaceAllUsesWith(Call.getResult());
          Load.erase();
          continue;
        }
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
      mlir::Value Handle = getHandleForIndex(Info.ReflectIndex);

      SmallVector<mlir::Operation *, 8> ChainUsers(AO->getUsers().begin(),
                                                   AO->getUsers().end());
      for (mlir::Operation *CU : ChainUsers) {
        auto Chain = mlir::dyn_cast<mlir::spirv::AccessChainOp>(CU);
        if (!Chain || Chain.getIndices().size() < 2)
          return makeError("unsupported access pattern on buffer '{0}'", Sym);

        mlir::Value ElemIdx = Chain.getIndices()[1];
        // Compute sub-element byte offset (constant or dynamic) for struct/
        // array/vector fields.
        auto computeByteOff = [&](mlir::OpBuilder &B,
                                  mlir::Location Loc) -> Expected<mlir::Value> {
          uint32_t ConstOff = 0;
          mlir::Value DynOff;
          mlir::Type CurTy = Info.ElemType;
          for (size_t K = 2; K < Chain.getIndices().size(); ++K) {
            mlir::Value IdxV = Chain.getIndices()[K];
            auto CIdx = getConstSpirvInt(IdxV);
            if (auto ST = mlir::dyn_cast<mlir::spirv::StructType>(CurTy)) {
              if (!CIdx)
                return makeError("dynamic struct index on buffer '{0}'", Sym);
              uint32_t I = static_cast<uint32_t>(*CIdx);
              ConstOff += ST.getMemberOffset(I);
              CurTy = ST.getElementType(I);
            } else if (auto Arr =
                           mlir::dyn_cast<mlir::spirv::ArrayType>(CurTy)) {
              uint32_t S = Arr.getArrayStride();
              if (S == 0)
                S = spirvTypeBytes(Arr.getElementType());
              if (CIdx) {
                ConstOff += static_cast<uint32_t>(*CIdx) * S;
              } else {
                mlir::Value Idx32 = castToSignless(B, Loc, IdxV);
                auto StrideC = mlir::LLVM::ConstantOp::create(
                    B, Loc, I32Ty, static_cast<int32_t>(S));
                mlir::Value Term =
                    mlir::LLVM::MulOp::create(B, Loc, Idx32, StrideC);
                DynOff = DynOff
                             ? mlir::Value(
                                   mlir::LLVM::AddOp::create(B, Loc, DynOff, Term))
                             : Term;
              }
              CurTy = Arr.getElementType();
            } else if (auto Vec = mlir::dyn_cast<mlir::VectorType>(CurTy)) {
              uint32_t S = scalarTypeBytes(Vec.getElementType());
              if (CIdx) {
                ConstOff += static_cast<uint32_t>(*CIdx) * S;
              } else {
                mlir::Value Idx32 = castToSignless(B, Loc, IdxV);
                auto StrideC = mlir::LLVM::ConstantOp::create(
                    B, Loc, I32Ty, static_cast<int32_t>(S));
                mlir::Value Term =
                    mlir::LLVM::MulOp::create(B, Loc, Idx32, StrideC);
                DynOff = DynOff
                             ? mlir::Value(
                                   mlir::LLVM::AddOp::create(B, Loc, DynOff, Term))
                             : Term;
              }
              CurTy = Vec.getElementType();
            }
          }
          mlir::Value Base = mlir::LLVM::ConstantOp::create(
              B, Loc, I32Ty, static_cast<int32_t>(ConstOff));
          if (DynOff) {
            if (ConstOff == 0)
              return DynOff;
            return mlir::Value(mlir::LLVM::AddOp::create(B, Loc, Base, DynOff));
          }
          return Base;
        };
        SmallVector<mlir::Operation *, 8> MemUsers(Chain->getUsers().begin(),
                                                   Chain->getUsers().end());
        for (mlir::Operation *MU : MemUsers) {
          if (auto Load = mlir::dyn_cast<mlir::spirv::LoadOp>(MU)) {
            mlir::OpBuilder B(Load);
            mlir::Location Loc = Load.getLoc();
            auto OffValOrErr = computeByteOff(B, Loc);
            if (!OffValOrErr)
              return OffValOrErr.takeError();
            mlir::Value OffC = *OffValOrErr;
            mlir::Type OrigTy = Load.getType();
            auto VecTy = mlir::dyn_cast<mlir::VectorType>(OrigTy);
            mlir::Type ScalarTy = makeSignless(
                VecTy ? VecTy.getElementType() : OrigTy, B);
            unsigned NumComps =
                VecTy ? static_cast<unsigned>(VecTy.getNumElements()) : 1u;
            uint8_t Mask = static_cast<uint8_t>((1u << std::min(NumComps, 4u)) - 1u);
            auto ResRetTy = getOrInitResRetType(&Ctx, ScalarTy);
            std::string Callee = "dx.op.rawBufferLoad." + typeSuffix(ScalarTy);
            auto FnTy = mlir::LLVM::LLVMFunctionType::get(
                ResRetTy, {I32Ty, HandleTy, I32Ty, I32Ty, I8Ty, I32Ty});
            getOrCreateDecl(SpvModule, Callee, FnTy);

            mlir::Value IdxVal = castToSignless(B, Loc, ElemIdx);
            auto OpC = mlir::LLVM::ConstantOp::create(
                B, Loc, I32Ty, static_cast<int32_t>(DXILOp::RawBufferLoad));
            auto MaskC = mlir::LLVM::ConstantOp::create(
                B, Loc, I8Ty, static_cast<int8_t>(Mask));
            auto AlignC = mlir::LLVM::ConstantOp::create(
                B, Loc, I32Ty, static_cast<int32_t>(scalarTypeBytes(ScalarTy)));

            auto Call = mlir::LLVM::CallOp::create(
                B, Loc, mlir::TypeRange{ResRetTy}, Callee,
                mlir::ValueRange{OpC, Handle, IdxVal, OffC, MaskC, AlignC});
            mlir::Value ResultVal;
            if (!VecTy) {
              ResultVal = mlir::LLVM::ExtractValueOp::create(
                  B, Loc, Call.getResult(), ArrayRef<int64_t>{0});
            } else {
              mlir::Type SignlessVecTy = makeSignless(VecTy, B);
              mlir::Value Vec = mlir::LLVM::UndefOp::create(B, Loc, SignlessVecTy);
              for (unsigned C = 0; C < NumComps && C < 4; ++C) {
                auto VC = mlir::LLVM::ExtractValueOp::create(
                    B, Loc, Call.getResult(), ArrayRef<int64_t>{C});
                auto IdxC = mlir::LLVM::ConstantOp::create(
                    B, Loc, I32Ty, static_cast<int32_t>(C));
                Vec = mlir::LLVM::InsertElementOp::create(B, Loc, Vec, VC, IdxC);
              }
              ResultVal = Vec;
            }
            ResultVal = castFromSignless(B, Loc, ResultVal, OrigTy);
            Load.replaceAllUsesWith(ResultVal);
            Load.erase();
          } else if (auto Store = mlir::dyn_cast<mlir::spirv::StoreOp>(MU)) {
            mlir::OpBuilder B(Store);
            mlir::Location Loc = Store.getLoc();
            auto OffValOrErr = computeByteOff(B, Loc);
            if (!OffValOrErr)
              return OffValOrErr.takeError();
            mlir::Value OffC = *OffValOrErr;
            mlir::Value StoreVal = castToSignless(B, Loc, Store.getValue());
            auto VecTy = mlir::dyn_cast<mlir::VectorType>(StoreVal.getType());
            mlir::Type ScalarTy =
                VecTy ? VecTy.getElementType() : StoreVal.getType();
            unsigned NumComps =
                VecTy ? static_cast<unsigned>(VecTy.getNumElements()) : 1u;
            uint8_t Mask = static_cast<uint8_t>((1u << std::min(NumComps, 4u)) - 1u);
            std::string Callee = "dx.op.rawBufferStore." + typeSuffix(ScalarTy);
            auto FnTy = mlir::LLVM::LLVMFunctionType::get(
                VoidTy, {I32Ty, HandleTy, I32Ty, I32Ty, ScalarTy, ScalarTy,
                         ScalarTy, ScalarTy, I8Ty, I32Ty});
            getOrCreateDecl(SpvModule, Callee, FnTy);

            mlir::Value IdxVal = castToSignless(B, Loc, ElemIdx);
            auto OpC = mlir::LLVM::ConstantOp::create(
                B, Loc, I32Ty, static_cast<int32_t>(DXILOp::RawBufferStore));
            auto Undef = mlir::LLVM::UndefOp::create(B, Loc, ScalarTy);
            SmallVector<mlir::Value, 4> Vals(4, Undef.getResult());
            if (!VecTy) {
              Vals[0] = StoreVal;
            } else {
              for (unsigned C = 0; C < NumComps && C < 4; ++C) {
                auto IdxC = mlir::LLVM::ConstantOp::create(
                    B, Loc, I32Ty, static_cast<int32_t>(C));
                Vals[C] = mlir::LLVM::ExtractElementOp::create(B, Loc, StoreVal,
                                                               IdxC);
              }
            }
            auto MaskC = mlir::LLVM::ConstantOp::create(
                B, Loc, I8Ty, static_cast<int8_t>(Mask));
            auto AlignC = mlir::LLVM::ConstantOp::create(
                B, Loc, I32Ty, static_cast<int32_t>(scalarTypeBytes(ScalarTy)));

            mlir::LLVM::CallOp::create(
                B, Loc, mlir::TypeRange{}, Callee,
                mlir::ValueRange{OpC, Handle, IdxVal, OffC, Vals[0], Vals[1],
                                 Vals[2], Vals[3], MaskC, AlignC});
            Store.erase();
          } else {
            // Structured buffer atomics -> dx.op.atomicBinOp.
            std::optional<DXILAtomicBinOp> ABinOp;
            mlir::Value AVal;
            if (auto A = mlir::dyn_cast<mlir::spirv::AtomicIAddOp>(MU)) {
              ABinOp = DXILAtomicBinOp::Add;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicAndOp>(MU)) {
              ABinOp = DXILAtomicBinOp::And;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicOrOp>(MU)) {
              ABinOp = DXILAtomicBinOp::Or;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicXorOp>(MU)) {
              ABinOp = DXILAtomicBinOp::Xor;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicSMinOp>(MU)) {
              ABinOp = DXILAtomicBinOp::IMin;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicSMaxOp>(MU)) {
              ABinOp = DXILAtomicBinOp::IMax;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicUMinOp>(MU)) {
              ABinOp = DXILAtomicBinOp::UMin;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicUMaxOp>(MU)) {
              ABinOp = DXILAtomicBinOp::UMax;
              AVal = A.getValue();
            } else if (auto A =
                           mlir::dyn_cast<mlir::spirv::AtomicExchangeOp>(MU)) {
              ABinOp = DXILAtomicBinOp::Exchange;
              AVal = A.getValue();
            }
            if (!ABinOp)
              return makeError("unsupported memory op '{0}' on buffer '{1}'",
                               MU->getName().getStringRef(), Sym);

            mlir::OpBuilder B(MU);
            mlir::Location Loc = MU->getLoc();
            mlir::Type OrigTy = AVal.getType();
            mlir::Value SignlessVal = castToSignless(B, Loc, AVal);
            mlir::Value IdxVal = castToSignless(B, Loc, ElemIdx);
            mlir::Type ValTy = SignlessVal.getType();
            std::string Callee = "dx.op.atomicBinOp." + typeSuffix(ValTy);
            auto FnTy = mlir::LLVM::LLVMFunctionType::get(
                ValTy,
                {I32Ty, HandleTy, I32Ty, I32Ty, I32Ty, I32Ty, ValTy});
            getOrCreateDecl(SpvModule, Callee, FnTy);

            auto OpC = mlir::LLVM::ConstantOp::create(
                B, Loc, I32Ty, static_cast<int32_t>(DXILOp::AtomicBinOp));
            auto BinC = mlir::LLVM::ConstantOp::create(
                B, Loc, I32Ty, static_cast<int32_t>(*ABinOp));
            auto ZeroC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 0);
            auto UndefI32 = mlir::LLVM::UndefOp::create(B, Loc, I32Ty);

            auto Call = mlir::LLVM::CallOp::create(
                B, Loc, mlir::TypeRange{ValTy}, Callee,
                mlir::ValueRange{OpC, Handle, BinC, IdxVal, ZeroC, UndefI32,
                                 SignlessVal});
            MU->getResult(0).replaceAllUsesWith(
                castFromSignless(B, Loc, Call.getResult(), OrigTy));
            MU->erase();
          }
        }
        Chain.erase();
      }
      AO.erase();
      continue;
    }

    // 3c. Constant buffers (Uniform Block) -> dx.op.cbufferLoadLegacy.
    auto CBIt = CBufferGlobals.find(Sym);
    if (CBIt != CBufferGlobals.end()) {
      const BoundCBufferInfo &Info = CBIt->second;
      mlir::Value Handle = getHandleForIndex(Info.ReflectIndex);

      SmallVector<mlir::Operation *, 8> ChainUsers(AO->getUsers().begin(),
                                                   AO->getUsers().end());
      for (mlir::Operation *CU : ChainUsers) {
        auto Chain = mlir::dyn_cast<mlir::spirv::AccessChainOp>(CU);
        if (!Chain || Chain.getIndices().empty())
          return makeError("unsupported access pattern on cbuffer '{0}'", Sym);

        // Compute static byte offset through nested structs/arrays.
        mlir::Type CurTy = Info.StructTy;
        uint32_t ByteOffset = 0;
        for (mlir::Value IdxV : Chain.getIndices()) {
          auto ConstIdx = getConstSpirvInt(IdxV);
          if (!ConstIdx)
            return makeError("dynamic indexing into cbuffer '{0}' not yet supported",
                             Sym);
          uint32_t I = static_cast<uint32_t>(*ConstIdx);
          if (auto ST = mlir::dyn_cast<mlir::spirv::StructType>(CurTy)) {
            ByteOffset += ST.getMemberOffset(I);
            CurTy = ST.getElementType(I);
          } else if (auto Arr = mlir::dyn_cast<mlir::spirv::ArrayType>(CurTy)) {
            uint32_t Stride = Arr.getArrayStride();
            if (Stride == 0)
              Stride = spirvTypeBytes(Arr.getElementType());
            ByteOffset += I * Stride;
            CurTy = Arr.getElementType();
          } else if (auto Mat = mlir::dyn_cast<mlir::spirv::MatrixType>(CurTy)) {
            ByteOffset += I * 16u;
            CurTy = Mat.getColumnType();
          } else if (auto Vec = mlir::dyn_cast<mlir::VectorType>(CurTy)) {
            ByteOffset += I * scalarTypeBytes(Vec.getElementType());
            CurTy = Vec.getElementType();
          } else {
            return makeError("unsupported cbuffer index step on '{0}'", Sym);
          }
        }

        uint32_t Row = ByteOffset / 16u;

        SmallVector<mlir::Operation *, 4> MemUsers(Chain->getUsers().begin(),
                                                   Chain->getUsers().end());
        for (mlir::Operation *MU : MemUsers) {
          auto Load = mlir::dyn_cast<mlir::spirv::LoadOp>(MU);
          if (!Load)
            return makeError("unsupported non-load user of cbuffer '{0}'", Sym);
          mlir::OpBuilder B(Load);
          mlir::Location Loc = Load.getLoc();
          mlir::Type ValTy = makeSignless(Load.getType(), B);
          uint32_t ElemBytes = scalarTypeBytes(ValTy);
          uint32_t Comp = (ByteOffset % 16u) / ElemBytes;
          auto RetTy = getOrInitCBufRetType(&Ctx, ValTy);
          std::string Callee = "dx.op.cbufferLoadLegacy." + typeSuffix(ValTy);
          auto FnTy = mlir::LLVM::LLVMFunctionType::get(
              RetTy, {I32Ty, HandleTy, I32Ty});
          getOrCreateDecl(SpvModule, Callee, FnTy);

          auto OpC = mlir::LLVM::ConstantOp::create(
              B, Loc, I32Ty, static_cast<int32_t>(DXILOp::CBufferLoadLegacy));
          auto RowC = mlir::LLVM::ConstantOp::create(
              B, Loc, I32Ty, static_cast<int32_t>(Row));
          auto Call = mlir::LLVM::CallOp::create(
              B, Loc, mlir::TypeRange{RetTy}, Callee,
              mlir::ValueRange{OpC, Handle, RowC});
          auto Val = mlir::LLVM::ExtractValueOp::create(
              B, Loc, Call.getResult(), ArrayRef<int64_t>{Comp});
          Load.replaceAllUsesWith(Val.getResult());
          Load.erase();
        }
        Chain.erase();
      }
      AO.erase();
      continue;
    }

    // 3c2. Workgroup (groupshared) globals -> addrspace(3) LLVM ops.
    auto WGIt = WorkgroupGlobals.find(Sym);
    if (WGIt != WorkgroupGlobals.end()) {
      const WorkgroupGlobalInfo &Info = WGIt->second;
      mlir::OpBuilder AOB(AO);
      auto Ptr3Ty = mlir::LLVM::LLVMPointerType::get(&Ctx, 3);
      auto LLVMAddr = mlir::LLVM::AddressOfOp::create(
          AOB, AO.getLoc(), Ptr3Ty,
          mlir::FlatSymbolRefAttr::get(&Ctx, Info.Name));

      auto rewritePtrUsers = [&](mlir::Value OldPtr, mlir::Value NewPtr) -> Error {
        SmallVector<mlir::Operation *, 8> Users(OldPtr.getUsers().begin(),
                                                OldPtr.getUsers().end());
        for (mlir::Operation *MU : Users) {
          if (auto Load = mlir::dyn_cast<mlir::spirv::LoadOp>(MU)) {
            mlir::OpBuilder B(Load);
            mlir::Location Loc = Load.getLoc();
            mlir::Type OrigTy = Load.getType();
            mlir::Type ValTy = makeSignless(OrigTy, B);
            auto LLVMLoad = mlir::LLVM::LoadOp::create(B, Loc, ValTy, NewPtr);
            Load.replaceAllUsesWith(
                castFromSignless(B, Loc, LLVMLoad.getResult(), OrigTy));
            Load.erase();
          } else if (auto Store = mlir::dyn_cast<mlir::spirv::StoreOp>(MU)) {
            mlir::OpBuilder B(Store);
            mlir::Location Loc = Store.getLoc();
            mlir::Value Val = castToSignless(B, Loc, Store.getValue());
            mlir::LLVM::StoreOp::create(B, Loc, Val, NewPtr);
            Store.erase();
          } else {
            std::optional<mlir::LLVM::AtomicBinOp> BinOp;
            mlir::Value AVal;
            if (auto A = mlir::dyn_cast<mlir::spirv::AtomicIAddOp>(MU)) {
              BinOp = mlir::LLVM::AtomicBinOp::add;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicISubOp>(MU)) {
              BinOp = mlir::LLVM::AtomicBinOp::sub;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicAndOp>(MU)) {
              BinOp = mlir::LLVM::AtomicBinOp::_and;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicOrOp>(MU)) {
              BinOp = mlir::LLVM::AtomicBinOp::_or;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicXorOp>(MU)) {
              BinOp = mlir::LLVM::AtomicBinOp::_xor;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicSMinOp>(MU)) {
              BinOp = mlir::LLVM::AtomicBinOp::min;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicSMaxOp>(MU)) {
              BinOp = mlir::LLVM::AtomicBinOp::max;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicUMinOp>(MU)) {
              BinOp = mlir::LLVM::AtomicBinOp::umin;
              AVal = A.getValue();
            } else if (auto A = mlir::dyn_cast<mlir::spirv::AtomicUMaxOp>(MU)) {
              BinOp = mlir::LLVM::AtomicBinOp::umax;
              AVal = A.getValue();
            } else if (auto A =
                           mlir::dyn_cast<mlir::spirv::AtomicExchangeOp>(MU)) {
              BinOp = mlir::LLVM::AtomicBinOp::xchg;
              AVal = A.getValue();
            }
            if (!BinOp)
              return makeError("unsupported op '{0}' on workgroup global '{1}'",
                               MU->getName().getStringRef(), Sym);
            mlir::OpBuilder B(MU);
            mlir::Location Loc = MU->getLoc();
            mlir::Type OrigTy = AVal.getType();
            mlir::Value SignlessVal = castToSignless(B, Loc, AVal);
            auto RMW = mlir::LLVM::AtomicRMWOp::create(
                B, Loc, *BinOp, NewPtr, SignlessVal,
                mlir::LLVM::AtomicOrdering::monotonic);
            MU->getResult(0).replaceAllUsesWith(
                castFromSignless(B, Loc, RMW.getResult(), OrigTy));
            MU->erase();
          }
        }
        return Error::success();
      };

      SmallVector<mlir::Operation *, 8> AOUsers(AO->getUsers().begin(),
                                                AO->getUsers().end());
      for (mlir::Operation *U : AOUsers) {
        if (auto Chain = mlir::dyn_cast<mlir::spirv::AccessChainOp>(U)) {
          mlir::OpBuilder B(Chain);
          SmallVector<mlir::LLVM::GEPArg, 4> GEPArgs;
          GEPArgs.push_back(int32_t(0));
          for (mlir::Value IdxV : Chain.getIndices()) {
            if (auto C = getConstSpirvInt(IdxV))
              GEPArgs.push_back(static_cast<int32_t>(*C));
            else
              GEPArgs.push_back(IdxV);
          }
          auto GEP = mlir::LLVM::GEPOp::create(
              B, Chain.getLoc(), Ptr3Ty, Info.LLVMType, LLVMAddr.getResult(),
              GEPArgs);
          if (Error E = rewritePtrUsers(Chain.getResult(), GEP.getResult()))
            return std::move(E);
          Chain.erase();
        } else {
          if (Error E = rewritePtrUsers(AO.getResult(), LLVMAddr.getResult()))
            return std::move(E);
          break;
        }
      }
      AO.erase();
      continue;
    }

    // 3c3. Graphics stage I/O globals -> dx.op.loadInput / dx.op.storeOutput.
    auto IOIt = StageIOGlobals.find(Sym);
    if (IOIt != StageIOGlobals.end()) {
      const StageIOInfo &Info = IOIt->second;
      SmallVector<mlir::Operation *, 8> Users(AO->getUsers().begin(),
                                              AO->getUsers().end());
      for (mlir::Operation *U : Users) {
        if (auto Load = mlir::dyn_cast<mlir::spirv::LoadOp>(U)) {
          mlir::OpBuilder B(Load);
          mlir::Location Loc = Load.getLoc();
          mlir::Type ValTy = Load.getType();
          auto VecTy = mlir::dyn_cast<mlir::VectorType>(ValTy);
          mlir::Type ElemTy =
              makeSignless(VecTy ? VecTy.getElementType() : ValTy, B);
          std::string Callee = "dx.op.loadInput." + typeSuffix(ElemTy);
          auto FnTy = mlir::LLVM::LLVMFunctionType::get(
              ElemTy, {I32Ty, I32Ty, I32Ty, I8Ty, I32Ty});
          getOrCreateDecl(SpvModule, Callee, FnTy);

          auto OpC = mlir::LLVM::ConstantOp::create(
              B, Loc, I32Ty, static_cast<int32_t>(DXILOp::LoadInput));
          auto IdC = mlir::LLVM::ConstantOp::create(
              B, Loc, I32Ty, static_cast<int32_t>(Info.ID));
          auto RowC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 0);
          auto UndefI32 = mlir::LLVM::UndefOp::create(B, Loc, I32Ty);

          if (!VecTy) {
            auto ColC = mlir::LLVM::ConstantOp::create(
                B, Loc, I8Ty, static_cast<int8_t>(0));
            auto Call = mlir::LLVM::CallOp::create(
                B, Loc, mlir::TypeRange{ElemTy}, Callee,
                mlir::ValueRange{OpC, IdC, RowC, ColC, UndefI32});
            Load.replaceAllUsesWith(Call.getResult());
          } else {
            SmallVector<mlir::Value, 4> Comps;
            for (unsigned C = 0; C < Info.Columns; ++C) {
              auto ColC = mlir::LLVM::ConstantOp::create(
                  B, Loc, I8Ty, static_cast<int8_t>(C));
              auto Call = mlir::LLVM::CallOp::create(
                  B, Loc, mlir::TypeRange{ElemTy}, Callee,
                  mlir::ValueRange{OpC, IdC, RowC, ColC, UndefI32});
              Comps.push_back(Call.getResult());
            }
            auto Vec = mlir::spirv::CompositeConstructOp::create(
                B, Loc, ValTy, Comps);
            Load.replaceAllUsesWith(Vec.getResult());
          }
          Load.erase();
        } else if (auto Store = mlir::dyn_cast<mlir::spirv::StoreOp>(U)) {
          mlir::OpBuilder B(Store);
          mlir::Location Loc = Store.getLoc();
          mlir::Value Val = Store.getValue();
          auto VecTy = mlir::dyn_cast<mlir::VectorType>(Val.getType());
          mlir::Type ElemTy =
              makeSignless(VecTy ? VecTy.getElementType() : Val.getType(), B);
          std::string Callee = "dx.op.storeOutput." + typeSuffix(ElemTy);
          auto FnTy = mlir::LLVM::LLVMFunctionType::get(
              VoidTy, {I32Ty, I32Ty, I32Ty, I8Ty, ElemTy});
          getOrCreateDecl(SpvModule, Callee, FnTy);

          auto OpC = mlir::LLVM::ConstantOp::create(
              B, Loc, I32Ty, static_cast<int32_t>(DXILOp::StoreOutput));
          auto IdC = mlir::LLVM::ConstantOp::create(
              B, Loc, I32Ty, static_cast<int32_t>(Info.ID));
          auto RowC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 0);

          if (!VecTy) {
            auto ColC = mlir::LLVM::ConstantOp::create(
                B, Loc, I8Ty, static_cast<int8_t>(0));
            mlir::LLVM::CallOp::create(
                B, Loc, mlir::TypeRange{}, Callee,
                mlir::ValueRange{OpC, IdC, RowC, ColC, Val});
          } else {
            for (unsigned C = 0; C < Info.Columns; ++C) {
              auto ColC = mlir::LLVM::ConstantOp::create(
                  B, Loc, I8Ty, static_cast<int8_t>(C));
              auto Lane = mlir::spirv::CompositeExtractOp::create(
                  B, Loc, ElemTy, Val, B.getI32ArrayAttr({static_cast<int32_t>(C)}));
              mlir::LLVM::CallOp::create(
                  B, Loc, mlir::TypeRange{}, Callee,
                  mlir::ValueRange{OpC, IdC, RowC, ColC, Lane.getResult()});
            }
          }
          Store.erase();
        } else {
          return makeError("unsupported user '{0}' of stage I/O global '{1}'",
                           U->getName().getStringRef(), Sym);
        }
      }
      AO.erase();
      continue;
    }

    // 3c4. UniformConstant images (typed buffers & textures).
    auto ImgIt = ImageGlobals.find(Sym);
    if (ImgIt != ImageGlobals.end()) {
      const BoundImageInfo &Info = ImgIt->second;
      mlir::Value Handle = getHandleForIndex(Info.ReflectIndex);
      SmallVector<mlir::Operation *, 8> Loads(AO->getUsers().begin(),
                                              AO->getUsers().end());
      for (mlir::Operation *LU : Loads) {
        auto ImgLoad = mlir::dyn_cast<mlir::spirv::LoadOp>(LU);
        if (!ImgLoad)
          return makeError("unsupported user of image global '{0}'", Sym);
        SmallVector<mlir::Operation *, 8> ImgUsers(ImgLoad->getUsers().begin(),
                                                   ImgLoad->getUsers().end());
        for (mlir::Operation *IU : ImgUsers) {
          if (auto IW = mlir::dyn_cast<mlir::spirv::ImageWriteOp>(IU)) {
            mlir::OpBuilder B(IW);
            mlir::Location Loc = IW.getLoc();
            mlir::Value Coord = castToSignless(B, Loc, IW.getCoordinate());
            mlir::Value Texel = castToSignless(B, Loc, IW.getTexel());
            auto TexVec = mlir::dyn_cast<mlir::VectorType>(Texel.getType());
            mlir::Type ElemTy =
                TexVec ? TexVec.getElementType() : Texel.getType();
            auto UndefVal = mlir::LLVM::UndefOp::create(B, Loc, ElemTy);
            auto UndefI32 = mlir::LLVM::UndefOp::create(B, Loc, I32Ty);

            SmallVector<mlir::Value, 4> Vals(4, UndefVal.getResult());
            uint8_t Mask = 1;
            if (!TexVec) {
              Vals[0] = Texel;
              Mask = 1;
            } else {
              unsigned N = static_cast<unsigned>(TexVec.getNumElements());
              Mask = static_cast<uint8_t>((1u << std::min(N, 4u)) - 1u);
              for (unsigned C = 0; C < N && C < 4; ++C) {
                auto IdxC = mlir::LLVM::ConstantOp::create(
                    B, Loc, I32Ty, static_cast<int32_t>(C));
                Vals[C] =
                    mlir::LLVM::ExtractElementOp::create(B, Loc, Texel, IdxC);
              }
            }

            if (Info.Kind == ResourceKind::TypedBuffer) {
              std::string Callee = "dx.op.bufferStore." + typeSuffix(ElemTy);
              auto FnTy = mlir::LLVM::LLVMFunctionType::get(
                  VoidTy, {I32Ty, HandleTy, I32Ty, I32Ty, ElemTy, ElemTy,
                           ElemTy, ElemTy, I8Ty});
              getOrCreateDecl(SpvModule, Callee, FnTy);
              auto OpC = mlir::LLVM::ConstantOp::create(
                  B, Loc, I32Ty, static_cast<int32_t>(DXILOp::BufferStore));
              auto MaskC = mlir::LLVM::ConstantOp::create(
                  B, Loc, I8Ty, static_cast<int8_t>(Mask));
              mlir::LLVM::CallOp::create(
                  B, Loc, mlir::TypeRange{}, Callee,
                  mlir::ValueRange{OpC, Handle, Coord, UndefI32, Vals[0],
                                   Vals[1], Vals[2], Vals[3], MaskC});
            } else {
              std::string Callee = "dx.op.textureStore." + typeSuffix(ElemTy);
              auto FnTy = mlir::LLVM::LLVMFunctionType::get(
                  VoidTy, {I32Ty, HandleTy, I32Ty, I32Ty, I32Ty, ElemTy, ElemTy,
                           ElemTy, ElemTy, I8Ty});
              getOrCreateDecl(SpvModule, Callee, FnTy);
              auto OpC = mlir::LLVM::ConstantOp::create(
                  B, Loc, I32Ty, static_cast<int32_t>(DXILOp::TextureStore));
              auto MaskC = mlir::LLVM::ConstantOp::create(
                  B, Loc, I8Ty, static_cast<int8_t>(Mask));
              SmallVector<mlir::Value, 3> Coords(3, UndefI32.getResult());
              if (auto CVec =
                      mlir::dyn_cast<mlir::VectorType>(Coord.getType())) {
                for (unsigned C = 0; C < CVec.getNumElements() && C < 3; ++C) {
                  Coords[C] = mlir::spirv::CompositeExtractOp::create(
                      B, Loc, I32Ty, Coord,
                      B.getI32ArrayAttr({static_cast<int32_t>(C)}));
                }
              } else {
                Coords[0] = Coord;
              }
              mlir::LLVM::CallOp::create(
                  B, Loc, mlir::TypeRange{}, Callee,
                  mlir::ValueRange{OpC, Handle, Coords[0], Coords[1], Coords[2],
                                   Vals[0], Vals[1], Vals[2], Vals[3], MaskC});
            }
            IW.erase();
          } else if (mlir::isa<mlir::spirv::ImageFetchOp,
                               mlir::spirv::ImageReadOp>(IU)) {
            mlir::OpBuilder B(IU);
            mlir::Location Loc = IU->getLoc();
            mlir::Value Coord =
                mlir::isa<mlir::spirv::ImageFetchOp>(IU)
                    ? mlir::cast<mlir::spirv::ImageFetchOp>(IU).getCoordinate()
                    : mlir::cast<mlir::spirv::ImageReadOp>(IU).getCoordinate();
            Coord = castToSignless(B, Loc, Coord);
            mlir::Type OrigResTy = IU->getResult(0).getType();
            auto ResVec = mlir::dyn_cast<mlir::VectorType>(OrigResTy);
            mlir::Type ElemTy = makeSignless(
                ResVec ? ResVec.getElementType() : OrigResTy, B);
            auto ResRetTy = getOrInitResRetType(&Ctx, ElemTy);
            auto UndefI32 = mlir::LLVM::UndefOp::create(B, Loc, I32Ty);
            auto ZeroI32 = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 0);

            mlir::LLVM::CallOp Call;
            if (Info.Kind == ResourceKind::TypedBuffer) {
              std::string Callee = "dx.op.bufferLoad." + typeSuffix(ElemTy);
              auto FnTy = mlir::LLVM::LLVMFunctionType::get(
                  ResRetTy, {I32Ty, HandleTy, I32Ty, I32Ty});
              getOrCreateDecl(SpvModule, Callee, FnTy);
              auto OpC = mlir::LLVM::ConstantOp::create(
                  B, Loc, I32Ty, static_cast<int32_t>(DXILOp::BufferLoad));
              Call = mlir::LLVM::CallOp::create(
                  B, Loc, mlir::TypeRange{ResRetTy}, Callee,
                  mlir::ValueRange{OpC, Handle, Coord, UndefI32});
            } else {
              std::string Callee = "dx.op.textureLoad." + typeSuffix(ElemTy);
              auto FnTy = mlir::LLVM::LLVMFunctionType::get(
                  ResRetTy, {I32Ty, HandleTy, I32Ty, I32Ty, I32Ty, I32Ty, I32Ty,
                             I32Ty, I32Ty});
              getOrCreateDecl(SpvModule, Callee, FnTy);
              auto OpC = mlir::LLVM::ConstantOp::create(
                  B, Loc, I32Ty, static_cast<int32_t>(DXILOp::TextureLoad));
              SmallVector<mlir::Value, 3> Coords(3, UndefI32.getResult());
              if (auto CVec =
                      mlir::dyn_cast<mlir::VectorType>(Coord.getType())) {
                for (unsigned C = 0; C < CVec.getNumElements() && C < 3; ++C) {
                  auto IdxC = mlir::LLVM::ConstantOp::create(
                      B, Loc, I32Ty, static_cast<int32_t>(C));
                  Coords[C] = mlir::LLVM::ExtractElementOp::create(
                      B, Loc, Coord, IdxC);
                }
              } else {
                Coords[0] = Coord;
              }
              mlir::Value MipVal = ZeroI32;
              if (auto FetchOp = mlir::dyn_cast<mlir::spirv::ImageFetchOp>(IU)) {
                if (auto IO = FetchOp.getImageOperands();
                    IO && mlir::spirv::bitEnumContainsAny(
                              *IO, mlir::spirv::ImageOperands::Lod) &&
                    !FetchOp.getOperandArguments().empty()) {
                  MipVal = castToSignless(B, Loc,
                                          FetchOp.getOperandArguments()[0]);
                }
              }
              Call = mlir::LLVM::CallOp::create(
                  B, Loc, mlir::TypeRange{ResRetTy}, Callee,
                  mlir::ValueRange{OpC, Handle, MipVal, Coords[0], Coords[1],
                                   Coords[2], UndefI32, UndefI32, UndefI32});
            }
            mlir::Value ResultVal;
            if (!ResVec) {
              ResultVal = mlir::LLVM::ExtractValueOp::create(
                  B, Loc, Call.getResult(), ArrayRef<int64_t>{0});
            } else {
              mlir::Type SignlessVecTy = makeSignless(ResVec, B);
              unsigned N = static_cast<unsigned>(ResVec.getNumElements());
              mlir::Value Vec =
                  mlir::LLVM::UndefOp::create(B, Loc, SignlessVecTy);
              for (unsigned C = 0; C < N && C < 4; ++C) {
                auto VC = mlir::LLVM::ExtractValueOp::create(
                    B, Loc, Call.getResult(), ArrayRef<int64_t>{C});
                auto IdxC = mlir::LLVM::ConstantOp::create(
                    B, Loc, I32Ty, static_cast<int32_t>(C));
                Vec = mlir::LLVM::InsertElementOp::create(B, Loc, Vec, VC, IdxC);
              }
              ResultVal = Vec;
            }
            ResultVal = castFromSignless(B, Loc, ResultVal, OrigResTy);
            IU->getResult(0).replaceAllUsesWith(ResultVal);
            IU->erase();
          } else if (auto SI = mlir::dyn_cast<mlir::spirv::SampledImageOp>(IU)) {
            // Lower SampledImage + ImageSample{Implicit,Explicit}Lod.
            auto SampLoad =
                SI.getSampler().getDefiningOp<mlir::spirv::LoadOp>();
            auto SampAO =
                SampLoad
                    ? SampLoad.getPtr().getDefiningOp<mlir::spirv::AddressOfOp>()
                    : nullptr;
            if (!SampAO || !SamplerGlobals.contains(SampAO.getVariable()))
              return makeError("unresolved sampler in spirv.SampledImage");
            mlir::Value SampHandle =
                getHandleForIndex(SamplerGlobals[SampAO.getVariable()]);

            auto lowerSample = [&](mlir::Operation *SampleOp, mlir::Value Coord,
                                   mlir::Value LodVal,
                                   bool IsExplicit) -> Error {
              mlir::OpBuilder B(SampleOp);
              mlir::Location Loc = SampleOp->getLoc();
              mlir::Type ResTy = SampleOp->getResult(0).getType();
              auto ResVec = mlir::dyn_cast<mlir::VectorType>(ResTy);
              mlir::Type ElemTy = makeSignless(
                  ResVec ? ResVec.getElementType() : ResTy, B);
              auto ResRetTy = getOrInitResRetType(&Ctx, ElemTy);
              auto F32Ty = B.getF32Type();
              auto UndefF32 = mlir::LLVM::UndefOp::create(B, Loc, F32Ty);
              auto UndefI32 = mlir::LLVM::UndefOp::create(B, Loc, I32Ty);

              SmallVector<mlir::Value, 4> Coords(4, UndefF32.getResult());
              if (auto CVec =
                      mlir::dyn_cast<mlir::VectorType>(Coord.getType())) {
                for (unsigned C = 0; C < CVec.getNumElements() && C < 4; ++C) {
                  Coords[C] = mlir::spirv::CompositeExtractOp::create(
                      B, Loc, F32Ty, Coord,
                      B.getI32ArrayAttr({static_cast<int32_t>(C)}));
                }
              } else {
                Coords[0] = Coord;
              }

              mlir::LLVM::CallOp Call;
              if (IsExplicit) {
                std::string Callee = "dx.op.sampleLevel." + typeSuffix(ElemTy);
                auto FnTy = mlir::LLVM::LLVMFunctionType::get(
                    ResRetTy, {I32Ty, HandleTy, HandleTy, F32Ty, F32Ty, F32Ty,
                               F32Ty, I32Ty, I32Ty, I32Ty, F32Ty});
                getOrCreateDecl(SpvModule, Callee, FnTy);
                auto OpC = mlir::LLVM::ConstantOp::create(
                    B, Loc, I32Ty, static_cast<int32_t>(DXILOp::SampleLevel));
                Call = mlir::LLVM::CallOp::create(
                    B, Loc, mlir::TypeRange{ResRetTy}, Callee,
                    mlir::ValueRange{OpC, Handle, SampHandle, Coords[0],
                                     Coords[1], Coords[2], Coords[3], UndefI32,
                                     UndefI32, UndefI32, LodVal});
              } else {
                std::string Callee = "dx.op.sample." + typeSuffix(ElemTy);
                auto FnTy = mlir::LLVM::LLVMFunctionType::get(
                    ResRetTy, {I32Ty, HandleTy, HandleTy, F32Ty, F32Ty, F32Ty,
                               F32Ty, I32Ty, I32Ty, I32Ty, F32Ty});
                getOrCreateDecl(SpvModule, Callee, FnTy);
                auto OpC = mlir::LLVM::ConstantOp::create(
                    B, Loc, I32Ty, static_cast<int32_t>(DXILOp::Sample));
                Call = mlir::LLVM::CallOp::create(
                    B, Loc, mlir::TypeRange{ResRetTy}, Callee,
                    mlir::ValueRange{OpC, Handle, SampHandle, Coords[0],
                                     Coords[1], Coords[2], Coords[3], UndefI32,
                                     UndefI32, UndefI32, UndefF32});
              }

              if (!ResVec) {
                auto V0 = mlir::LLVM::ExtractValueOp::create(
                    B, Loc, Call.getResult(), ArrayRef<int64_t>{0});
                SampleOp->getResult(0).replaceAllUsesWith(V0.getResult());
              } else {
                unsigned N = static_cast<unsigned>(ResVec.getNumElements());
                mlir::Value Vec = mlir::LLVM::UndefOp::create(B, Loc, ResTy);
                for (unsigned C = 0; C < N; ++C) {
                  auto VC = mlir::LLVM::ExtractValueOp::create(
                      B, Loc, Call.getResult(), ArrayRef<int64_t>{C});
                  auto IdxC = mlir::LLVM::ConstantOp::create(
                      B, Loc, I32Ty, static_cast<int32_t>(C));
                  Vec = mlir::LLVM::InsertElementOp::create(B, Loc, Vec, VC,
                                                            IdxC);
                }
                SampleOp->getResult(0).replaceAllUsesWith(Vec);
              }
              SampleOp->erase();
              return Error::success();
            };

            SmallVector<mlir::Operation *, 4> SIUsers(SI->getUsers().begin(),
                                                      SI->getUsers().end());
            for (mlir::Operation *SU : SIUsers) {
              if (auto Imp =
                      mlir::dyn_cast<mlir::spirv::ImageSampleImplicitLodOp>(
                          SU)) {
                if (Error E = lowerSample(Imp, Imp.getCoordinate(), nullptr,
                                          /*IsExplicit=*/false))
                  return std::move(E);
              } else if (auto Exp = mlir::dyn_cast<
                             mlir::spirv::ImageSampleExplicitLodOp>(SU)) {
                mlir::Value LodVal = Exp.getOperandArguments().empty()
                                         ? nullptr
                                         : Exp.getOperandArguments()[0];
                if (!LodVal)
                  return makeError(
                      "spirv.ImageSampleExplicitLod missing LOD operand");
                if (Error E = lowerSample(Exp, Exp.getCoordinate(), LodVal,
                                          /*IsExplicit=*/true))
                  return std::move(E);
              } else {
                return makeError("unsupported sampled-image op '{0}'",
                                 SU->getName().getStringRef());
              }
            }
            SI.erase();
            if (SampLoad && SampLoad->use_empty())
              SampLoad.erase();
            if (SampAO && SampAO->use_empty()) {
              ErasedAddrOps.insert(SampAO);
              SampAO.erase();
            }
          } else {
            return makeError("unsupported image op '{0}' on '{1}'",
                             IU->getName().getStringRef(), Sym);
          }
        }
        if (ImgLoad->use_empty())
          ImgLoad.erase();
      }
      if (AO->use_empty())
        AO.erase();
      continue;
    }
  }

  // 3d. Lower spirv.ControlBarrier & spirv.MemoryBarrier -> Layer 1 gpu.barrier.
  SmallVector<mlir::spirv::ControlBarrierOp, 4> Barriers;
  EntryFn.walk(
      [&](mlir::spirv::ControlBarrierOp CB) { Barriers.push_back(CB); });
  for (auto CB : Barriers) {
    mlir::OpBuilder B(CB);
    auto GPU = mlir::gpu::BarrierOp::create(B, CB.getLoc());
    auto Sem = CB.getMemorySemantics();
    if (mlir::spirv::bitEnumContainsAny(
            Sem, mlir::spirv::MemorySemantics::UniformMemory |
                     mlir::spirv::MemorySemantics::ImageMemory)) {
      GPU->setAttr("crucible.device_fence", B.getUnitAttr());
    }
    CB.erase();
  }
  SmallVector<mlir::spirv::MemoryBarrierOp, 4> MemBarriers;
  EntryFn.walk(
      [&](mlir::spirv::MemoryBarrierOp MB) { MemBarriers.push_back(MB); });
  for (auto MB : MemBarriers) {
    mlir::OpBuilder B(MB);
    auto GPU = mlir::gpu::BarrierOp::create(B, MB.getLoc());
    auto Sem = MB.getMemorySemantics();
    if (mlir::spirv::bitEnumContainsAny(
            Sem, mlir::spirv::MemorySemantics::UniformMemory |
                     mlir::spirv::MemorySemantics::ImageMemory)) {
      GPU->setAttr("crucible.device_fence", B.getUnitAttr());
    }
    if (!mlir::spirv::bitEnumContainsAny(
            Sem, mlir::spirv::MemorySemantics::WorkgroupMemory)) {
      GPU->setAttr("crucible.no_tgsm_fence", B.getUnitAttr());
    }
    MB.erase();
  }

  // 3e. Lower spirv.GroupNonUniform* subgroup ops -> Layer 1 gpu.* ops.
  auto lowerGroupOp = [&](mlir::Operation *Op) {
    mlir::OpBuilder B(Op);
    mlir::Location Loc = Op->getLoc();

    auto emitReduce = [&](mlir::Value RawVal,
                          mlir::gpu::AllReduceOperation ReduceOp,
                          mlir::spirv::GroupOperation GroupOp) {
      mlir::Type OrigTy = RawVal.getType();
      mlir::Value Val = castToSignless(B, Loc, RawVal);
      auto Red = mlir::gpu::SubgroupReduceOp::create(B, Loc, Val, ReduceOp,
                                                     /*uniform=*/false);
      if (GroupOp == mlir::spirv::GroupOperation::ExclusiveScan)
        Red->setAttr("crucible.scan_kind", B.getStringAttr("exclusive"));
      else if (GroupOp == mlir::spirv::GroupOperation::InclusiveScan)
        Red->setAttr("crucible.scan_kind", B.getStringAttr("inclusive"));
      mlir::Value Res = castFromSignless(B, Loc, Red.getResult(), OrigTy);
      Op->getResult(0).replaceAllUsesWith(Res);
      Op->erase();
    };

    if (auto G = mlir::dyn_cast<mlir::spirv::GroupNonUniformIAddOp>(Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::ADD,
                 G.getGroupOperation());
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformFAddOp>(Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::ADD,
                 G.getGroupOperation());
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformIMulOp>(Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::MUL,
                 G.getGroupOperation());
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformFMulOp>(Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::MUL,
                 G.getGroupOperation());
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformSMinOp>(Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::MINSI,
                 G.getGroupOperation());
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformUMinOp>(Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::MINUI,
                 G.getGroupOperation());
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformFMinOp>(Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::MINNUMF,
                 G.getGroupOperation());
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformSMaxOp>(Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::MAXSI,
                 G.getGroupOperation());
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformUMaxOp>(Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::MAXUI,
                 G.getGroupOperation());
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformFMaxOp>(Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::MAXNUMF,
                 G.getGroupOperation());
    } else if (auto G = mlir::dyn_cast<mlir::spirv::GroupNonUniformBitwiseAndOp>(
                   Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::AND,
                 G.getGroupOperation());
    } else if (auto G = mlir::dyn_cast<mlir::spirv::GroupNonUniformBitwiseOrOp>(
                   Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::OR,
                 G.getGroupOperation());
    } else if (auto G = mlir::dyn_cast<mlir::spirv::GroupNonUniformBitwiseXorOp>(
                   Op)) {
      emitReduce(G.getValue(), mlir::gpu::AllReduceOperation::XOR,
                 G.getGroupOperation());
    } else if (auto G = mlir::dyn_cast<
                   mlir::spirv::GroupNonUniformBroadcastFirstOp>(Op)) {
      mlir::Type OrigTy = G.getType();
      mlir::Value Val = castToSignless(B, Loc, G.getValue());
      auto BC = mlir::gpu::SubgroupBroadcastOp::create(
          B, Loc, Val, /*subgroupId=*/nullptr,
          mlir::gpu::BroadcastType::first_active_lane);
      G.replaceAllUsesWith(castFromSignless(B, Loc, BC.getResult(), OrigTy));
      G.erase();
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformBroadcastOp>(Op)) {
      mlir::Type OrigTy = G.getType();
      mlir::Value Val = castToSignless(B, Loc, G.getValue());
      mlir::Value Id = castToSignless(B, Loc, G.getId());
      auto BC = mlir::gpu::SubgroupBroadcastOp::create(
          B, Loc, Val, Id, mlir::gpu::BroadcastType::specific_lane);
      G.replaceAllUsesWith(castFromSignless(B, Loc, BC.getResult(), OrigTy));
      G.erase();
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformShuffleOp>(Op)) {
      mlir::Type OrigTy = G.getType();
      mlir::Value Val = castToSignless(B, Loc, G.getValue());
      mlir::Value Id = castToSignless(B, Loc, G.getId());
      auto Width = mlir::arith::ConstantIntOp::create(B, Loc, 32, 32);
      auto Shuf = mlir::gpu::ShuffleOp::create(
          B, Loc, Val, Id, Width, mlir::gpu::ShuffleMode::IDX);
      G.replaceAllUsesWith(
          castFromSignless(B, Loc, Shuf.getShuffleResult(), OrigTy));
      G.erase();
    } else if (auto G = mlir::dyn_cast<mlir::spirv::GroupNonUniformShuffleXorOp>(
                   Op)) {
      mlir::Type OrigTy = G.getType();
      mlir::Value Val = castToSignless(B, Loc, G.getValue());
      mlir::Value Mask = castToSignless(B, Loc, G.getMask());
      auto Width = mlir::arith::ConstantIntOp::create(B, Loc, 32, 32);
      auto Shuf = mlir::gpu::ShuffleOp::create(
          B, Loc, Val, Mask, Width, mlir::gpu::ShuffleMode::XOR);
      G.replaceAllUsesWith(
          castFromSignless(B, Loc, Shuf.getShuffleResult(), OrigTy));
      G.erase();
    } else if (auto G = mlir::dyn_cast<mlir::spirv::GroupNonUniformShuffleUpOp>(
                   Op)) {
      mlir::Type OrigTy = G.getType();
      mlir::Value Val = castToSignless(B, Loc, G.getValue());
      mlir::Value Delta = castToSignless(B, Loc, G.getDelta());
      auto Width = mlir::arith::ConstantIntOp::create(B, Loc, 32, 32);
      auto Shuf = mlir::gpu::ShuffleOp::create(
          B, Loc, Val, Delta, Width, mlir::gpu::ShuffleMode::UP);
      G.replaceAllUsesWith(
          castFromSignless(B, Loc, Shuf.getShuffleResult(), OrigTy));
      G.erase();
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformShuffleDownOp>(
                       Op)) {
      mlir::Type OrigTy = G.getType();
      mlir::Value Val = castToSignless(B, Loc, G.getValue());
      mlir::Value Delta = castToSignless(B, Loc, G.getDelta());
      auto Width = mlir::arith::ConstantIntOp::create(B, Loc, 32, 32);
      auto Shuf = mlir::gpu::ShuffleOp::create(
          B, Loc, Val, Delta, Width, mlir::gpu::ShuffleMode::DOWN);
      G.replaceAllUsesWith(
          castFromSignless(B, Loc, Shuf.getShuffleResult(), OrigTy));
      G.erase();
    } else if (auto G = mlir::dyn_cast<mlir::spirv::GroupNonUniformAllEqualOp>(
                   Op)) {
      mlir::Value Val = castToSignless(B, Loc, G.getValue());
      mlir::Type ValTy = Val.getType();
      std::string Callee = "dx.op.waveActiveAllEqual." + typeSuffix(ValTy);
      auto FnTy = mlir::LLVM::LLVMFunctionType::get(I1Ty, {I32Ty, ValTy});
      getOrCreateDecl(SpvModule, Callee, FnTy);
      auto OpC = mlir::LLVM::ConstantOp::create(
          B, Loc, I32Ty, static_cast<int32_t>(DXILOp::WaveActiveAllEqual));
      auto Call = mlir::LLVM::CallOp::create(
          B, Loc, mlir::TypeRange{I1Ty}, Callee, mlir::ValueRange{OpC, Val});
      G.replaceAllUsesWith(Call.getResult());
      G.erase();
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformElectOp>(Op)) {
      auto One = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 1);
      auto Zero = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, 0);
      auto Scan = mlir::gpu::SubgroupReduceOp::create(
          B, Loc, One.getResult(), mlir::gpu::AllReduceOperation::ADD, false);
      Scan->setAttr("crucible.scan_kind", B.getStringAttr("exclusive"));
      auto Cmp = mlir::LLVM::ICmpOp::create(
          B, Loc, mlir::LLVM::ICmpPredicate::eq, Scan.getResult(),
          Zero.getResult());
      G.replaceAllUsesWith(Cmp.getResult());
      G.erase();
    } else if (auto G =
                   mlir::dyn_cast<mlir::spirv::GroupNonUniformBallotOp>(Op)) {
      auto Bal = mlir::gpu::BallotOp::create(B, Loc, I32Ty, G.getPredicate());
      SmallVector<mlir::Operation *, 4> Users(G->getUsers().begin(),
                                              G->getUsers().end());
      for (mlir::Operation *U : Users) {
        if (auto Ext = mlir::dyn_cast<mlir::spirv::CompositeExtractOp>(U)) {
          if (Ext.getIndices().size() == 1) {
            int64_t Idx =
                mlir::cast<mlir::IntegerAttr>(Ext.getIndices()[0]).getInt();
            mlir::OpBuilder EB(Ext);
            mlir::Value Rep =
                Idx == 0
                    ? Bal.getResult()
                    : mlir::LLVM::ConstantOp::create(EB, Ext.getLoc(), I32Ty, 0)
                          .getResult();
            Ext.replaceAllUsesWith(Rep);
            Ext.erase();
          }
        } else if (auto BC = mlir::dyn_cast<
                       mlir::spirv::GroupNonUniformBallotBitCountOp>(U)) {
          mlir::OpBuilder BCB(BC);
          mlir::Location BLoc = BC.getLoc();
          mlir::Value Masked = Bal.getResult();
          if (BC.getGroupOperation() ==
              mlir::spirv::GroupOperation::ExclusiveScan) {
            auto LaneIdx =
                mlir::gpu::LaneIdOp::create(BCB, BLoc, /*upper_bound=*/nullptr);
            auto Lane =
                mlir::arith::IndexCastUIOp::create(BCB, BLoc, I32Ty, LaneIdx);
            auto One = mlir::LLVM::ConstantOp::create(BCB, BLoc, I32Ty, 1);
            auto Shl = mlir::LLVM::ShlOp::create(BCB, BLoc, One, Lane);
            auto Mask = mlir::LLVM::SubOp::create(BCB, BLoc, Shl, One);
            Masked = mlir::LLVM::AndOp::create(BCB, BLoc, Masked, Mask);
          }
          auto Pop = mlir::math::CtPopOp::create(BCB, BLoc, Masked);
          BC.replaceAllUsesWith(Pop.getResult());
          BC.erase();
        }
      }
      if (G->use_empty())
        G.erase();
    }
  };

  SmallVector<mlir::Operation *, 16> SubgroupOps;
  EntryFn.walk([&](mlir::Operation *Op) {
    StringRef Name = Op->getName().getStringRef();
    if (Name.starts_with("spirv.GroupNonUniform") &&
        Name != "spirv.GroupNonUniformBallotBitCount")
      SubgroupOps.push_back(Op);
  });
  for (mlir::Operation *Op : SubgroupOps)
    lowerGroupOp(Op);

  // 3f. Lower spirv.VectorTimesScalar and spirv.CompositeConstruct on vectors
  // to LLVM dialect ops (ConvertSPIRVToLLVM does not cover these directly).
  SmallVector<mlir::spirv::VectorTimesScalarOp, 8> VTSOps;
  EntryFn.walk([&](mlir::spirv::VectorTimesScalarOp VTS) { VTSOps.push_back(VTS); });
  for (auto VTS : VTSOps) {
    mlir::OpBuilder B(VTS);
    mlir::Location Loc = VTS.getLoc();
    auto VecTy = mlir::cast<mlir::VectorType>(VTS.getType());
    mlir::Value Splat = mlir::LLVM::UndefOp::create(B, Loc, VecTy);
    for (unsigned C = 0; C < VecTy.getNumElements(); ++C) {
      auto IdxC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty,
                                                 static_cast<int32_t>(C));
      Splat = mlir::LLVM::InsertElementOp::create(B, Loc, Splat,
                                                  VTS.getScalar(), IdxC);
    }
    auto Mul = mlir::LLVM::FMulOp::create(B, Loc, VTS.getVector(), Splat);
    VTS.replaceAllUsesWith(Mul.getResult());
    VTS.erase();
  }

  SmallVector<mlir::spirv::CompositeConstructOp, 8> CCOps;
  EntryFn.walk(
      [&](mlir::spirv::CompositeConstructOp CC) { CCOps.push_back(CC); });
  for (auto CC : CCOps) {
    auto OrigVecTy = mlir::dyn_cast<mlir::VectorType>(CC.getType());
    if (!OrigVecTy)
      continue;
    mlir::OpBuilder B(CC);
    mlir::Location Loc = CC.getLoc();
    mlir::Type SignlessVecTy = makeSignless(OrigVecTy, B);
    mlir::Value Vec = mlir::LLVM::UndefOp::create(B, Loc, SignlessVecTy);
    unsigned Lane = 0;
    for (mlir::Value RawOpnd : CC.getConstituents()) {
      mlir::Value Opnd = castToSignless(B, Loc, RawOpnd);
      if (auto SubVec = mlir::dyn_cast<mlir::VectorType>(Opnd.getType())) {
        for (unsigned K = 0; K < SubVec.getNumElements(); ++K) {
          auto KIdx = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty,
                                                     static_cast<int32_t>(K));
          auto Elem = mlir::LLVM::ExtractElementOp::create(B, Loc, Opnd, KIdx);
          auto DstIdx = mlir::LLVM::ConstantOp::create(
              B, Loc, I32Ty, static_cast<int32_t>(Lane++));
          Vec = mlir::LLVM::InsertElementOp::create(B, Loc, Vec, Elem, DstIdx);
        }
      } else {
        auto DstIdx = mlir::LLVM::ConstantOp::create(
            B, Loc, I32Ty, static_cast<int32_t>(Lane++));
        Vec = mlir::LLVM::InsertElementOp::create(B, Loc, Vec, Opnd, DstIdx);
      }
    }
    Vec = castFromSignless(B, Loc, Vec, OrigVecTy);
    CC.replaceAllUsesWith(Vec);
    CC.erase();
  }

  // 3g. Lower spirv.Kill -> dx.op.discard(true).
  SmallVector<mlir::spirv::KillOp, 4> KillOps;
  EntryFn.walk([&](mlir::spirv::KillOp K) { KillOps.push_back(K); });
  for (auto K : KillOps) {
    mlir::OpBuilder B(K);
    mlir::Location Loc = K.getLoc();
    auto FnTy = mlir::LLVM::LLVMFunctionType::get(VoidTy, {I32Ty, I1Ty});
    getOrCreateDecl(SpvModule, "dx.op.discard", FnTy);
    auto OpC = mlir::LLVM::ConstantOp::create(
        B, Loc, I32Ty, static_cast<int32_t>(DXILOp::Discard));
    auto TrueC = mlir::LLVM::ConstantOp::create(B, Loc, I1Ty, true);
    mlir::LLVM::CallOp::create(B, Loc, mlir::TypeRange{}, "dx.op.discard",
                               mlir::ValueRange{OpC, TrueC});
    mlir::spirv::ReturnOp::create(B, Loc);
    K.erase();
  }

  // 3g2. Lower spirv.All, spirv.Any, spirv.IsNan, spirv.IsInf to LLVM ops.
  SmallVector<mlir::Operation *, 8> LogicOps;
  EntryFn.walk([&](mlir::Operation *Op) {
    if (mlir::isa<mlir::spirv::AllOp, mlir::spirv::AnyOp, mlir::spirv::IsNanOp,
                  mlir::spirv::IsInfOp>(Op))
      LogicOps.push_back(Op);
  });
  for (mlir::Operation *Op : LogicOps) {
    mlir::OpBuilder B(Op);
    mlir::Location Loc = Op->getLoc();
    if (auto All = mlir::dyn_cast<mlir::spirv::AllOp>(Op)) {
      mlir::Value Vec = All.getVector();
      auto VTy = mlir::cast<mlir::VectorType>(Vec.getType());
      mlir::Value Acc;
      for (unsigned C = 0; C < VTy.getNumElements(); ++C) {
        auto IdxC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty,
                                                   static_cast<int32_t>(C));
        mlir::Value Lane =
            mlir::LLVM::ExtractElementOp::create(B, Loc, Vec, IdxC);
        Acc = Acc ? mlir::Value(mlir::LLVM::AndOp::create(B, Loc, Acc, Lane))
                  : Lane;
      }
      All.replaceAllUsesWith(Acc);
      All.erase();
    } else if (auto Any = mlir::dyn_cast<mlir::spirv::AnyOp>(Op)) {
      mlir::Value Vec = Any.getVector();
      auto VTy = mlir::cast<mlir::VectorType>(Vec.getType());
      mlir::Value Acc;
      for (unsigned C = 0; C < VTy.getNumElements(); ++C) {
        auto IdxC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty,
                                                   static_cast<int32_t>(C));
        mlir::Value Lane =
            mlir::LLVM::ExtractElementOp::create(B, Loc, Vec, IdxC);
        Acc = Acc ? mlir::Value(mlir::LLVM::OrOp::create(B, Loc, Acc, Lane))
                  : Lane;
      }
      Any.replaceAllUsesWith(Acc);
      Any.erase();
    } else if (auto IsN = mlir::dyn_cast<mlir::spirv::IsNanOp>(Op)) {
      mlir::Value X = IsN.getOperand();
      auto Cmp = mlir::LLVM::FCmpOp::create(
          B, Loc, mlir::LLVM::FCmpPredicate::uno, X, X);
      IsN.replaceAllUsesWith(Cmp.getResult());
      IsN.erase();
    } else if (auto IsI = mlir::dyn_cast<mlir::spirv::IsInfOp>(Op)) {
      mlir::Value X = IsI.getOperand();
      mlir::Type XTy = X.getType();
      mlir::Type ElemTy = XTy;
      if (auto VTy = mlir::dyn_cast<mlir::VectorType>(XTy))
        ElemTy = VTy.getElementType();
      auto FTy = mlir::cast<mlir::FloatType>(ElemTy);
      auto InfVal = APFloat::getInf(FTy.getFloatSemantics());
      mlir::Attribute InfAttr =
          mlir::isa<mlir::VectorType>(XTy)
              ? mlir::Attribute(mlir::DenseElementsAttr::get(
                    mlir::cast<mlir::VectorType>(XTy), InfVal))
              : mlir::Attribute(B.getFloatAttr(FTy, InfVal));
      auto Abs = mlir::LLVM::FAbsOp::create(B, Loc, X);
      auto InfC = mlir::LLVM::ConstantOp::create(B, Loc, XTy, InfAttr);
      auto Cmp = mlir::LLVM::FCmpOp::create(
          B, Loc, mlir::LLVM::FCmpPredicate::oeq, Abs, InfC);
      IsI.replaceAllUsesWith(Cmp.getResult());
      IsI.erase();
    }
  }

  // 3g3. Lower spirv.Switch -> chain of spirv.IEqual + spirv.BranchConditional
  // so ConvertSPIRVToLLVM + LiftControlFlowToSCF handle switches cleanly.
  SmallVector<mlir::spirv::SwitchOp, 4> SwitchOps;
  EntryFn.walk([&](mlir::spirv::SwitchOp SW) { SwitchOps.push_back(SW); });
  for (auto SW : SwitchOps) {
    mlir::Location Loc = SW.getLoc();
    mlir::Value Sel = SW.getSelector();
    mlir::Block *DefaultBB = SW.getDefaultTarget();
    mlir::ValueRange DefaultArgs = SW.getDefaultOperands();
    auto Literals = SW.getLiterals();
    auto Targets = SW.getTargets();

    if (!Literals || Targets.empty()) {
      mlir::OpBuilder B(SW);
      mlir::spirv::BranchOp::create(B, Loc, DefaultBB, DefaultArgs);
      SW.erase();
      continue;
    }

    auto LitVals = Literals->getValues<llvm::APInt>();
    unsigned NumCases = static_cast<unsigned>(Targets.size());
    mlir::Block *CurBB = SW->getBlock();
    mlir::Region *Reg = CurBB->getParent();

    for (unsigned I = 0; I < NumCases; ++I) {
      mlir::OpBuilder B = mlir::OpBuilder::atBlockEnd(CurBB);
      if (I == 0)
        B.setInsertionPoint(SW);
      mlir::Block *NextBB = DefaultBB;
      mlir::ValueRange NextArgs = DefaultArgs;
      if (I + 1 < NumCases) {
        NextBB = new mlir::Block();
        Reg->push_back(NextBB);
        NextArgs = {};
      }
      auto CaseAttr = B.getIntegerAttr(Sel.getType(), LitVals[I]);
      auto CaseC = mlir::spirv::ConstantOp::create(B, Loc, Sel.getType(),
                                                   CaseAttr);
      auto Cmp = mlir::spirv::IEqualOp::create(B, Loc, Sel, CaseC);
      mlir::spirv::BranchConditionalOp::create(
          B, Loc, Cmp, Targets[I], SW.getTargetOperands(I), NextBB, NextArgs);
      CurBB = NextBB;
    }
    SW.erase();
  }

  // 3g4. Lower spirv.GL.* and spirv.*Dot ops -> math.* / llvm.*.
  SmallVector<mlir::Operation *, 16> GLMathOps;
  EntryFn.walk([&](mlir::Operation *Op) {
    if (mlir::isa<mlir::spirv::GLAtan2Op, mlir::spirv::GLStepOp,
                  mlir::spirv::GLNormalizeOp, mlir::spirv::DotOp,
                  mlir::spirv::SDotOp, mlir::spirv::UDotOp,
                  mlir::spirv::SUDotOp, mlir::spirv::GLFSignOp,
                  mlir::spirv::GLSSignOp, mlir::spirv::GLLengthOp,
                  mlir::spirv::GLDistanceOp, mlir::spirv::GLCrossOp,
                  mlir::spirv::GLReflectOp, mlir::spirv::GLSmoothStepOp,
                  mlir::spirv::GLPackHalf2x16Op,
                  mlir::spirv::GLUnpackHalf2x16Op, mlir::spirv::GLFindILsbOp,
                  mlir::spirv::GLFindUMsbOp, mlir::spirv::GLFindSMsbOp>(Op))
      GLMathOps.push_back(Op);
  });
  for (mlir::Operation *Op : GLMathOps) {
    mlir::OpBuilder B(Op);
    mlir::Location Loc = Op->getLoc();
    auto mkFloatConst = [&](mlir::Type Ty, double V) -> mlir::Value {
      mlir::Type ElemTy = Ty;
      if (auto VTy = mlir::dyn_cast<mlir::VectorType>(Ty))
        ElemTy = VTy.getElementType();
      auto FTy = mlir::cast<mlir::FloatType>(ElemTy);
      mlir::Attribute A =
          mlir::isa<mlir::VectorType>(Ty)
              ? mlir::Attribute(mlir::DenseElementsAttr::get(
                    mlir::cast<mlir::VectorType>(Ty), B.getFloatAttr(FTy, V)))
              : mlir::Attribute(B.getFloatAttr(FTy, V));
      return mlir::LLVM::ConstantOp::create(B, Loc, Ty, A);
    };
    auto mkIntConst = [&](mlir::Type Ty, int64_t V) -> mlir::Value {
      mlir::Type ElemTy = Ty;
      if (auto VTy = mlir::dyn_cast<mlir::VectorType>(Ty))
        ElemTy = VTy.getElementType();
      auto ITy = mlir::cast<mlir::IntegerType>(ElemTy);
      mlir::Attribute A =
          mlir::isa<mlir::VectorType>(Ty)
              ? mlir::Attribute(mlir::DenseElementsAttr::get(
                    mlir::cast<mlir::VectorType>(Ty),
                    B.getIntegerAttr(ITy, V)))
              : mlir::Attribute(B.getIntegerAttr(ITy, V));
      return mlir::LLVM::ConstantOp::create(B, Loc, Ty, A);
    };
    auto emitFloatDot = [&](mlir::Value A, mlir::Value BVal) -> mlir::Value {
      mlir::Value Prod = mlir::LLVM::FMulOp::create(B, Loc, A, BVal);
      if (auto VTy = mlir::dyn_cast<mlir::VectorType>(A.getType())) {
        mlir::Value Sum;
        for (unsigned C = 0; C < VTy.getNumElements(); ++C) {
          auto IdxC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty,
                                                     static_cast<int32_t>(C));
          mlir::Value Lane =
              mlir::LLVM::ExtractElementOp::create(B, Loc, Prod, IdxC);
          Sum = Sum ? mlir::Value(mlir::LLVM::FAddOp::create(B, Loc, Sum, Lane))
                    : Lane;
        }
        return Sum;
      }
      return Prod;
    };
    auto splatScalar = [&](mlir::Value Scalar,
                           mlir::VectorType VTy) -> mlir::Value {
      mlir::Value Splat = mlir::LLVM::UndefOp::create(B, Loc, VTy);
      for (unsigned C = 0; C < VTy.getNumElements(); ++C) {
        auto IdxC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty,
                                                   static_cast<int32_t>(C));
        Splat = mlir::LLVM::InsertElementOp::create(B, Loc, Splat, Scalar, IdxC);
      }
      return Splat;
    };

    if (auto At2 = mlir::dyn_cast<mlir::spirv::GLAtan2Op>(Op)) {
      auto Rep = mlir::math::Atan2Op::create(B, Loc, At2->getOperand(0),
                                             At2->getOperand(1));
      At2.replaceAllUsesWith(Rep.getResult());
      At2.erase();
    } else if (auto St = mlir::dyn_cast<mlir::spirv::GLStepOp>(Op)) {
      mlir::Value Edge = St->getOperand(0);
      mlir::Value X = St->getOperand(1);
      mlir::Type Ty = X.getType();
      auto Cmp = mlir::LLVM::FCmpOp::create(
          B, Loc, mlir::LLVM::FCmpPredicate::olt, X, Edge);
      auto Sel = mlir::LLVM::SelectOp::create(
          B, Loc, Cmp, mkFloatConst(Ty, 0.0), mkFloatConst(Ty, 1.0));
      St.replaceAllUsesWith(Sel.getResult());
      St.erase();
    } else if (auto Norm = mlir::dyn_cast<mlir::spirv::GLNormalizeOp>(Op)) {
      mlir::Value X = Norm.getOperand();
      mlir::Type Ty = X.getType();
      if (auto VTy = mlir::dyn_cast<mlir::VectorType>(Ty)) {
        mlir::Value Sum = emitFloatDot(X, X);
        mlir::Value InvLen = mlir::math::RsqrtOp::create(B, Loc, Sum);
        mlir::Value Splat = splatScalar(InvLen, VTy);
        auto Res = mlir::LLVM::FMulOp::create(B, Loc, X, Splat);
        Norm.replaceAllUsesWith(Res.getResult());
        Norm.erase();
      } else {
        auto Abs = mlir::math::AbsFOp::create(B, Loc, X);
        auto Res = mlir::LLVM::FDivOp::create(B, Loc, X, Abs);
        Norm.replaceAllUsesWith(Res.getResult());
        Norm.erase();
      }
    } else if (auto Dot = mlir::dyn_cast<mlir::spirv::DotOp>(Op)) {
      mlir::Value Sum = emitFloatDot(Dot.getVector1(), Dot.getVector2());
      Dot.replaceAllUsesWith(Sum);
      Dot.erase();
    } else if (mlir::isa<mlir::spirv::SDotOp, mlir::spirv::UDotOp,
                         mlir::spirv::SUDotOp>(Op)) {
      bool Signed1 =
          mlir::isa<mlir::spirv::SDotOp, mlir::spirv::SUDotOp>(Op);
      bool Signed2 = mlir::isa<mlir::spirv::SDotOp>(Op);
      mlir::Value V1 = castToSignless(B, Loc, Op->getOperand(0));
      mlir::Value V2 = castToSignless(B, Loc, Op->getOperand(1));
      mlir::Type OrigResTy = Op->getResult(0).getType();
      mlir::Type SignlessResTy =
          B.getIntegerType(OrigResTy.getIntOrFloatBitWidth());
      auto I8V4Ty = mlir::VectorType::get({4}, B.getI8Type());
      if (V1.getType().isInteger(32))
        V1 = mlir::LLVM::BitcastOp::create(B, Loc, I8V4Ty, V1);
      if (V2.getType().isInteger(32))
        V2 = mlir::LLVM::BitcastOp::create(B, Loc, I8V4Ty, V2);
      auto VTy = mlir::cast<mlir::VectorType>(V1.getType());
      mlir::Value Sum;
      for (unsigned C = 0; C < VTy.getNumElements(); ++C) {
        auto IdxC = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty,
                                                   static_cast<int32_t>(C));
        mlir::Value E1 = mlir::LLVM::ExtractElementOp::create(B, Loc, V1, IdxC);
        mlir::Value E2 = mlir::LLVM::ExtractElementOp::create(B, Loc, V2, IdxC);
        if (E1.getType() != SignlessResTy)
          E1 = Signed1
                   ? mlir::Value(mlir::LLVM::SExtOp::create(B, Loc,
                                                            SignlessResTy, E1))
                   : mlir::Value(mlir::LLVM::ZExtOp::create(B, Loc,
                                                            SignlessResTy, E1));
        if (E2.getType() != SignlessResTy)
          E2 = Signed2
                   ? mlir::Value(mlir::LLVM::SExtOp::create(B, Loc,
                                                            SignlessResTy, E2))
                   : mlir::Value(mlir::LLVM::ZExtOp::create(B, Loc,
                                                            SignlessResTy, E2));
        mlir::Value P = mlir::LLVM::MulOp::create(B, Loc, E1, E2);
        Sum = Sum ? mlir::Value(mlir::LLVM::AddOp::create(B, Loc, Sum, P)) : P;
      }
      Op->getResult(0).replaceAllUsesWith(
          castFromSignless(B, Loc, Sum, OrigResTy));
      Op->erase();
    } else if (auto FS = mlir::dyn_cast<mlir::spirv::GLFSignOp>(Op)) {
      mlir::Value X = FS.getOperand();
      mlir::Type Ty = X.getType();
      mlir::Value Zero = mkFloatConst(Ty, 0.0);
      mlir::Value Gt = mlir::LLVM::FCmpOp::create(
          B, Loc, mlir::LLVM::FCmpPredicate::ogt, X, Zero);
      mlir::Value Lt = mlir::LLVM::FCmpOp::create(
          B, Loc, mlir::LLVM::FCmpPredicate::olt, X, Zero);
      mlir::Value Pos = mlir::LLVM::UIToFPOp::create(B, Loc, Ty, Gt);
      mlir::Value Neg = mlir::LLVM::UIToFPOp::create(B, Loc, Ty, Lt);
      mlir::Value Res = mlir::LLVM::FSubOp::create(B, Loc, Pos, Neg);
      FS.replaceAllUsesWith(Res);
      FS.erase();
    } else if (auto SS = mlir::dyn_cast<mlir::spirv::GLSSignOp>(Op)) {
      mlir::Type OrigTy = SS.getType();
      mlir::Value X = castToSignless(B, Loc, SS.getOperand());
      mlir::Type Ty = X.getType();
      mlir::Value Zero = mkIntConst(Ty, 0);
      mlir::Value Gt = mlir::LLVM::ICmpOp::create(
          B, Loc, mlir::LLVM::ICmpPredicate::sgt, X, Zero);
      mlir::Value Lt = mlir::LLVM::ICmpOp::create(
          B, Loc, mlir::LLVM::ICmpPredicate::slt, X, Zero);
      mlir::Value Pos = mlir::LLVM::ZExtOp::create(B, Loc, Ty, Gt);
      mlir::Value Neg = mlir::LLVM::ZExtOp::create(B, Loc, Ty, Lt);
      mlir::Value Res = mlir::LLVM::SubOp::create(B, Loc, Pos, Neg);
      SS.replaceAllUsesWith(castFromSignless(B, Loc, Res, OrigTy));
      SS.erase();
    } else if (auto Len = mlir::dyn_cast<mlir::spirv::GLLengthOp>(Op)) {
      mlir::Value X = Len.getOperand();
      mlir::Value Res;
      if (mlir::isa<mlir::VectorType>(X.getType())) {
        mlir::Value Sum = emitFloatDot(X, X);
        Res = mlir::math::SqrtOp::create(B, Loc, Sum);
      } else {
        Res = mlir::math::AbsFOp::create(B, Loc, X);
      }
      Len.replaceAllUsesWith(Res);
      Len.erase();
    } else if (auto Dist = mlir::dyn_cast<mlir::spirv::GLDistanceOp>(Op)) {
      mlir::Value Diff =
          mlir::LLVM::FSubOp::create(B, Loc, Dist->getOperand(0),
                                     Dist->getOperand(1));
      mlir::Value Res;
      if (mlir::isa<mlir::VectorType>(Diff.getType())) {
        mlir::Value Sum = emitFloatDot(Diff, Diff);
        Res = mlir::math::SqrtOp::create(B, Loc, Sum);
      } else {
        Res = mlir::math::AbsFOp::create(B, Loc, Diff);
      }
      Dist.replaceAllUsesWith(Res);
      Dist.erase();
    } else if (auto Cross = mlir::dyn_cast<mlir::spirv::GLCrossOp>(Op)) {
      mlir::Value A = Cross->getOperand(0);
      mlir::Value BVal = Cross->getOperand(1);
      auto VTy = mlir::cast<mlir::VectorType>(A.getType());
      auto c0 = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, int32_t(0));
      auto c1 = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, int32_t(1));
      auto c2 = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty, int32_t(2));
      mlir::Value ax = mlir::LLVM::ExtractElementOp::create(B, Loc, A, c0);
      mlir::Value ay = mlir::LLVM::ExtractElementOp::create(B, Loc, A, c1);
      mlir::Value az = mlir::LLVM::ExtractElementOp::create(B, Loc, A, c2);
      mlir::Value bx = mlir::LLVM::ExtractElementOp::create(B, Loc, BVal, c0);
      mlir::Value by = mlir::LLVM::ExtractElementOp::create(B, Loc, BVal, c1);
      mlir::Value bz = mlir::LLVM::ExtractElementOp::create(B, Loc, BVal, c2);
      mlir::Value rx = mlir::LLVM::FSubOp::create(
          B, Loc, mlir::LLVM::FMulOp::create(B, Loc, ay, bz),
          mlir::LLVM::FMulOp::create(B, Loc, az, by));
      mlir::Value ry = mlir::LLVM::FSubOp::create(
          B, Loc, mlir::LLVM::FMulOp::create(B, Loc, az, bx),
          mlir::LLVM::FMulOp::create(B, Loc, ax, bz));
      mlir::Value rz = mlir::LLVM::FSubOp::create(
          B, Loc, mlir::LLVM::FMulOp::create(B, Loc, ax, by),
          mlir::LLVM::FMulOp::create(B, Loc, ay, bx));
      mlir::Value Res = mlir::LLVM::UndefOp::create(B, Loc, VTy);
      Res = mlir::LLVM::InsertElementOp::create(B, Loc, Res, rx, c0);
      Res = mlir::LLVM::InsertElementOp::create(B, Loc, Res, ry, c1);
      Res = mlir::LLVM::InsertElementOp::create(B, Loc, Res, rz, c2);
      Cross.replaceAllUsesWith(Res);
      Cross.erase();
    } else if (auto Refl = mlir::dyn_cast<mlir::spirv::GLReflectOp>(Op)) {
      mlir::Value I = Refl->getOperand(0);
      mlir::Value N = Refl->getOperand(1);
      mlir::Value D = emitFloatDot(N, I);
      mlir::Value Two = mkFloatConst(D.getType(), 2.0);
      mlir::Value TwoD = mlir::LLVM::FMulOp::create(B, Loc, Two, D);
      if (auto VTy = mlir::dyn_cast<mlir::VectorType>(N.getType()))
        TwoD = splatScalar(TwoD, VTy);
      mlir::Value Scaled = mlir::LLVM::FMulOp::create(B, Loc, TwoD, N);
      mlir::Value Res = mlir::LLVM::FSubOp::create(B, Loc, I, Scaled);
      Refl.replaceAllUsesWith(Res);
      Refl.erase();
    } else if (auto Sm = mlir::dyn_cast<mlir::spirv::GLSmoothStepOp>(Op)) {
      mlir::Value Edge0 = Sm->getOperand(0);
      mlir::Value Edge1 = Sm->getOperand(1);
      mlir::Value X = Sm->getOperand(2);
      mlir::Type Ty = X.getType();
      mlir::Value Num = mlir::LLVM::FSubOp::create(B, Loc, X, Edge0);
      mlir::Value Den = mlir::LLVM::FSubOp::create(B, Loc, Edge1, Edge0);
      mlir::Value Ratio = mlir::LLVM::FDivOp::create(B, Loc, Num, Den);
      mlir::Value ClampedLo = mlir::LLVM::MaxNumOp::create(
          B, Loc, Ratio, mkFloatConst(Ty, 0.0));
      mlir::Value T = mlir::LLVM::MinNumOp::create(B, Loc, ClampedLo,
                                                   mkFloatConst(Ty, 1.0));
      mlir::Value TT = mlir::LLVM::FMulOp::create(B, Loc, T, T);
      mlir::Value TwoT =
          mlir::LLVM::FMulOp::create(B, Loc, mkFloatConst(Ty, 2.0), T);
      mlir::Value ThreeMinus2T =
          mlir::LLVM::FSubOp::create(B, Loc, mkFloatConst(Ty, 3.0), TwoT);
      mlir::Value Res = mlir::LLVM::FMulOp::create(B, Loc, TT, ThreeMinus2T);
      Sm.replaceAllUsesWith(Res);
      Sm.erase();
    } else if (auto Pack = mlir::dyn_cast<mlir::spirv::GLPackHalf2x16Op>(Op)) {
      auto F16V2Ty = mlir::VectorType::get({2}, B.getF16Type());
      mlir::Value H2 =
          mlir::LLVM::FPTruncOp::create(B, Loc, F16V2Ty, Pack.getOperand());
      mlir::Value Bits = mlir::LLVM::BitcastOp::create(B, Loc, I32Ty, H2);
      Pack.replaceAllUsesWith(
          castFromSignless(B, Loc, Bits, Pack.getType()));
      Pack.erase();
    } else if (auto Unpack =
                   mlir::dyn_cast<mlir::spirv::GLUnpackHalf2x16Op>(Op)) {
      auto F16V2Ty = mlir::VectorType::get({2}, B.getF16Type());
      mlir::Value S = castToSignless(B, Loc, Unpack.getOperand());
      mlir::Value H2 = mlir::LLVM::BitcastOp::create(B, Loc, F16V2Ty, S);
      mlir::Value Res =
          mlir::LLVM::FPExtOp::create(B, Loc, Unpack.getType(), H2);
      Unpack.replaceAllUsesWith(Res);
      Unpack.erase();
    } else if (auto FLsb = mlir::dyn_cast<mlir::spirv::GLFindILsbOp>(Op)) {
      mlir::Type OrigTy = FLsb.getType();
      mlir::Value X = castToSignless(B, Loc, FLsb.getOperand());
      mlir::Type Ty = X.getType();
      mlir::Value Ctz = mlir::LLVM::CountTrailingZerosOp::create(
          B, Loc, X, /*is_zero_poison=*/false);
      mlir::Value IsZero = mlir::LLVM::ICmpOp::create(
          B, Loc, mlir::LLVM::ICmpPredicate::eq, X, mkIntConst(Ty, 0));
      mlir::Value Mask = mlir::LLVM::SExtOp::create(B, Loc, Ty, IsZero);
      mlir::Value Res = mlir::LLVM::OrOp::create(B, Loc, Ctz, Mask);
      FLsb.replaceAllUsesWith(castFromSignless(B, Loc, Res, OrigTy));
      FLsb.erase();
    } else if (auto FUMsb = mlir::dyn_cast<mlir::spirv::GLFindUMsbOp>(Op)) {
      mlir::Type OrigTy = FUMsb.getType();
      mlir::Value X = castToSignless(B, Loc, FUMsb.getOperand());
      mlir::Type Ty = X.getType();
      mlir::Type ElemTy = Ty;
      if (auto VTy = mlir::dyn_cast<mlir::VectorType>(Ty))
        ElemTy = VTy.getElementType();
      int64_t BitsMinus1 =
          static_cast<int64_t>(ElemTy.getIntOrFloatBitWidth()) - 1;
      mlir::Value Clz = mlir::LLVM::CountLeadingZerosOp::create(
          B, Loc, X, /*is_zero_poison=*/false);
      mlir::Value Res = mlir::LLVM::SubOp::create(
          B, Loc, mkIntConst(Ty, BitsMinus1), Clz);
      FUMsb.replaceAllUsesWith(castFromSignless(B, Loc, Res, OrigTy));
      FUMsb.erase();
    } else if (auto FSMsb = mlir::dyn_cast<mlir::spirv::GLFindSMsbOp>(Op)) {
      mlir::Type OrigTy = FSMsb.getType();
      mlir::Value X = castToSignless(B, Loc, FSMsb.getOperand());
      mlir::Type Ty = X.getType();
      mlir::Type ElemTy = Ty;
      if (auto VTy = mlir::dyn_cast<mlir::VectorType>(Ty))
        ElemTy = VTy.getElementType();
      int64_t BitsMinus1 =
          static_cast<int64_t>(ElemTy.getIntOrFloatBitWidth()) - 1;
      mlir::Value SignMask = mlir::LLVM::AShrOp::create(
          B, Loc, X, mkIntConst(Ty, BitsMinus1));
      mlir::Value Norm = mlir::LLVM::XOrOp::create(B, Loc, X, SignMask);
      mlir::Value Clz = mlir::LLVM::CountLeadingZerosOp::create(
          B, Loc, Norm, /*is_zero_poison=*/false);
      mlir::Value Res = mlir::LLVM::SubOp::create(
          B, Loc, mkIntConst(Ty, BitsMinus1), Clz);
      FSMsb.replaceAllUsesWith(castFromSignless(B, Loc, Res, OrigTy));
      FSMsb.erase();
    }
  }

  // 3g5. Lower spirv integer comparison ops with non-signless integer operands
  // (si32/ui32/vectors) directly to llvm.icmp with signless casts, working
  // around ConvertSPIRVToLLVM leaving signed operands on llvm.icmp.
  SmallVector<mlir::Operation *, 16> ICmpOps;
  EntryFn.walk([&](mlir::Operation *Op) {
    if (mlir::isa<mlir::spirv::IEqualOp, mlir::spirv::INotEqualOp,
                  mlir::spirv::SLessThanOp, mlir::spirv::SLessThanEqualOp,
                  mlir::spirv::SGreaterThanOp, mlir::spirv::SGreaterThanEqualOp,
                  mlir::spirv::ULessThanOp, mlir::spirv::ULessThanEqualOp,
                  mlir::spirv::UGreaterThanOp,
                  mlir::spirv::UGreaterThanEqualOp>(Op)) {
      mlir::Type OpTy = Op->getOperand(0).getType();
      mlir::Type ElemTy = OpTy;
      if (auto VTy = mlir::dyn_cast<mlir::VectorType>(OpTy))
        ElemTy = VTy.getElementType();
      if (auto IT = mlir::dyn_cast<mlir::IntegerType>(ElemTy);
          IT && !IT.isSignless())
        ICmpOps.push_back(Op);
    }
  });
  for (mlir::Operation *Op : ICmpOps) {
    mlir::OpBuilder B(Op);
    mlir::Location Loc = Op->getLoc();
    mlir::LLVM::ICmpPredicate Pred;
    if (mlir::isa<mlir::spirv::IEqualOp>(Op))
      Pred = mlir::LLVM::ICmpPredicate::eq;
    else if (mlir::isa<mlir::spirv::INotEqualOp>(Op))
      Pred = mlir::LLVM::ICmpPredicate::ne;
    else if (mlir::isa<mlir::spirv::SLessThanOp>(Op))
      Pred = mlir::LLVM::ICmpPredicate::slt;
    else if (mlir::isa<mlir::spirv::SLessThanEqualOp>(Op))
      Pred = mlir::LLVM::ICmpPredicate::sle;
    else if (mlir::isa<mlir::spirv::SGreaterThanOp>(Op))
      Pred = mlir::LLVM::ICmpPredicate::sgt;
    else if (mlir::isa<mlir::spirv::SGreaterThanEqualOp>(Op))
      Pred = mlir::LLVM::ICmpPredicate::sge;
    else if (mlir::isa<mlir::spirv::ULessThanOp>(Op))
      Pred = mlir::LLVM::ICmpPredicate::ult;
    else if (mlir::isa<mlir::spirv::ULessThanEqualOp>(Op))
      Pred = mlir::LLVM::ICmpPredicate::ule;
    else if (mlir::isa<mlir::spirv::UGreaterThanOp>(Op))
      Pred = mlir::LLVM::ICmpPredicate::ugt;
    else
      Pred = mlir::LLVM::ICmpPredicate::uge;

    mlir::Value LHS = castToSignless(B, Loc, Op->getOperand(0));
    mlir::Value RHS = castToSignless(B, Loc, Op->getOperand(1));
    auto Cmp = mlir::LLVM::ICmpOp::create(B, Loc, Pred, LHS, RHS);
    Op->getResult(0).replaceAllUsesWith(Cmp.getResult());
    Op->erase();
  }

  // 3g6. Fold spirv.CompositeExtract(spirv.CompositeConstruct(...), idx) for
  // array/struct/matrix composites (e.g. matrix construction by sub-matrices).
  bool FoldedComposite = true;
  while (FoldedComposite) {
    FoldedComposite = false;
    SmallVector<mlir::spirv::CompositeExtractOp, 16> Extracts;
    EntryFn.walk(
        [&](mlir::spirv::CompositeExtractOp E) { Extracts.push_back(E); });
    for (auto E : Extracts) {
      auto CC = E.getComposite().getDefiningOp<mlir::spirv::CompositeConstructOp>();
      if (!CC || E.getIndices().empty())
        continue;
      if (mlir::isa<mlir::VectorType>(CC.getType()))
        continue;
      int64_t Idx = mlir::cast<mlir::IntegerAttr>(E.getIndices()[0]).getInt();
      if (Idx < 0 || static_cast<size_t>(Idx) >= CC.getConstituents().size())
        continue;
      mlir::Value Elem = CC.getConstituents()[Idx];
      if (E.getIndices().size() == 1) {
        E.replaceAllUsesWith(Elem);
        E.erase();
        FoldedComposite = true;
      } else {
        mlir::OpBuilder B(E);
        SmallVector<mlir::Attribute, 4> Rest(E.getIndices().begin() + 1,
                                             E.getIndices().end());
        auto NewE = mlir::spirv::CompositeExtractOp::create(
            B, E.getLoc(), E.getType(), Elem, B.getArrayAttr(Rest));
        E.replaceAllUsesWith(NewE.getResult());
        E.erase();
        FoldedComposite = true;
      }
    }
    SmallVector<mlir::spirv::CompositeConstructOp, 16> DeadCCs;
    EntryFn.walk([&](mlir::spirv::CompositeConstructOp CC) {
      if (CC->use_empty())
        DeadCCs.push_back(CC);
    });
    for (auto CC : DeadCCs) {
      CC.erase();
      FoldedComposite = true;
    }
  }

  // 3h. Lower Function-scope array/matrix spirv.Variable (including constant-
  // array & CompositeConstruct initializers) -> llvm.alloca + element stores.
  SmallVector<mlir::spirv::VariableOp, 4> FuncArrayVars;
  EntryFn.walk([&](mlir::spirv::VariableOp V) {
    auto PtrTy = mlir::dyn_cast<mlir::spirv::PointerType>(V.getType());
    if (PtrTy && PtrTy.getStorageClass() == mlir::spirv::StorageClass::Function &&
        mlir::isa<mlir::spirv::ArrayType, mlir::spirv::MatrixType>(
            PtrTy.getPointeeType()))
      FuncArrayVars.push_back(V);
  });
  for (auto V : FuncArrayVars) {
    auto PtrTy = mlir::cast<mlir::spirv::PointerType>(V.getType());
    mlir::Type PointeeTy = PtrTy.getPointeeType();
    mlir::OpBuilder VB(V);
    mlir::Location Loc = V.getLoc();
    mlir::Type LLVMArrTy = convertSpirvDataType(PointeeTy, VB);
    auto Ptr0Ty = mlir::LLVM::LLVMPointerType::get(&Ctx, 0);
    auto OneC = mlir::LLVM::ConstantOp::create(VB, Loc, I32Ty, 1);
    auto Alloca = mlir::LLVM::AllocaOp::create(VB, Loc, Ptr0Ty, LLVMArrTy, OneC);

    SmallVector<mlir::Operation *, 8> Users(V->getUsers().begin(),
                                            V->getUsers().end());
    for (mlir::Operation *U : Users) {
      if (auto Store = mlir::dyn_cast<mlir::spirv::StoreOp>(U)) {
        if (auto ConstArr =
                Store.getValue().getDefiningOp<mlir::spirv::ConstantOp>()) {
          if (auto AA = mlir::dyn_cast<mlir::ArrayAttr>(ConstArr.getValue())) {
            mlir::OpBuilder SB(Store);
            for (unsigned I = 0; I < AA.size(); ++I) {
              SmallVector<mlir::LLVM::GEPArg, 2> Idx{int32_t(0),
                                                     static_cast<int32_t>(I)};
              auto ElemPtr = mlir::LLVM::GEPOp::create(
                  SB, Store.getLoc(), Ptr0Ty, LLVMArrTy, Alloca, Idx);
              auto ArrTy = mlir::cast<mlir::spirv::ArrayType>(PointeeTy);
              auto ElemConst = mlir::LLVM::ConstantOp::create(
                  SB, Store.getLoc(),
                  convertSpirvDataType(ArrTy.getElementType(), SB), AA[I]);
              mlir::LLVM::StoreOp::create(SB, Store.getLoc(), ElemConst,
                                          ElemPtr);
            }
            Store.erase();
            if (ConstArr->use_empty())
              ConstArr.erase();
            continue;
          }
        }
        if (auto CC = Store.getValue()
                          .getDefiningOp<mlir::spirv::CompositeConstructOp>()) {
          mlir::OpBuilder SB(Store);
          auto Elems = CC.getConstituents();
          for (unsigned I = 0; I < Elems.size(); ++I) {
            SmallVector<mlir::LLVM::GEPArg, 2> Idx{int32_t(0),
                                                   static_cast<int32_t>(I)};
            auto ElemPtr = mlir::LLVM::GEPOp::create(
                SB, Store.getLoc(), Ptr0Ty, LLVMArrTy, Alloca, Idx);
            mlir::Value ElemVal = castToSignless(SB, Store.getLoc(), Elems[I]);
            mlir::LLVM::StoreOp::create(SB, Store.getLoc(), ElemVal, ElemPtr);
          }
          Store.erase();
          if (CC->use_empty())
            CC.erase();
          continue;
        }
      } else if (auto Chain = mlir::dyn_cast<mlir::spirv::AccessChainOp>(U)) {
        mlir::OpBuilder CB(Chain);
        SmallVector<mlir::LLVM::GEPArg, 4> GEPArgs;
        GEPArgs.push_back(int32_t(0));
        for (mlir::Value IdxV : Chain.getIndices()) {
          if (auto C = getConstSpirvInt(IdxV))
            GEPArgs.push_back(static_cast<int32_t>(*C));
          else
            GEPArgs.push_back(castToSignless(CB, Chain.getLoc(), IdxV));
        }
        auto GEP = mlir::LLVM::GEPOp::create(CB, Chain.getLoc(), Ptr0Ty,
                                             LLVMArrTy, Alloca, GEPArgs);
        SmallVector<mlir::Operation *, 4> CUsers(Chain->getUsers().begin(),
                                                 Chain->getUsers().end());
        for (mlir::Operation *CU : CUsers) {
          if (auto Load = mlir::dyn_cast<mlir::spirv::LoadOp>(CU)) {
            mlir::OpBuilder LB(Load);
            mlir::Type OrigTy = Load.getType();
            auto LLVMLoad = mlir::LLVM::LoadOp::create(
                LB, Load.getLoc(), makeSignless(OrigTy, LB), GEP);
            Load.replaceAllUsesWith(
                castFromSignless(LB, Load.getLoc(), LLVMLoad.getResult(),
                                 OrigTy));
            Load.erase();
          } else if (auto Store = mlir::dyn_cast<mlir::spirv::StoreOp>(CU)) {
            mlir::OpBuilder SB(Store);
            mlir::Value Val = castToSignless(SB, Store.getLoc(), Store.getValue());
            mlir::LLVM::StoreOp::create(SB, Store.getLoc(), Val, GEP);
            Store.erase();
          }
        }
        Chain.erase();
      }
    }
    if (V->use_empty())
      V.erase();
  }

  // Erase rewritten shader-ABI globals and entry/execution-mode ops before
  // running ConvertSPIRVToLLVM.
  for (auto GV : GlobalsToErase)
    GV.erase();

  // Emit addrspace(3) LLVM globals for Workgroup variables now that the
  // original spirv.GlobalVariable symbols have been erased.
  {
    mlir::OpBuilder GB =
        mlir::OpBuilder::atBlockBegin(&SpvModule.getBodyRegion().front());
    for (const auto &KV : WorkgroupGlobals) {
      const WorkgroupGlobalInfo &Info = KV.second;
      mlir::LLVM::GlobalOp::create(
          GB, SpvModule.getLoc(), Info.LLVMType, /*isConstant=*/false,
          mlir::LLVM::Linkage::Internal, Info.Name, /*value=*/mlir::Attribute(),
          /*alignment=*/0, /*addrSpace=*/3);
    }
  }
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

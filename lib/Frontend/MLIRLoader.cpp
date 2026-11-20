//===- MLIRLoader.cpp - Pure Layer 0+1 MLIR frontend ------------*- C++ -*-===//
//
// Part of Crucible, an open-source shader-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Parses standalone Layer 0 + Layer 1 MLIR text modules (func + scf + llvm/
// arith + gpu + math, with zero DXIL or SPIR-V artifacts) to validate the
// frontend-agnostic backend contract for future compute IRs (e.g. PTX).
//
//===----------------------------------------------------------------------===//

#include "Frontend.h"
#include "crucible/Diagnostic.h"
#include "crucible/Dialect/CrucibleDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Parser/Parser.h"

using namespace llvm;

namespace crucible {

bool isMLIRText(MemoryBufferRef Buf) {
  StringRef S = Buf.getBuffer().ltrim();
  while (S.starts_with("//")) {
    size_t NL = S.find('\n');
    if (NL == StringRef::npos)
      return false;
    S = S.drop_front(NL + 1).ltrim();
  }
  return S.starts_with("module") || S.starts_with("func.func");
}

Expected<LoadedMLIRShader> loadMLIR(MemoryBufferRef Buf,
                                    mlir::MLIRContext &Ctx,
                                    StringRef EntryOverride) {
  mlir::DialectRegistry Registry;
  Registry.insert<mlir::LLVM::LLVMDialect, mlir::DLTIDialect,
                  mlir::func::FuncDialect, mlir::cf::ControlFlowDialect,
                  mlir::scf::SCFDialect, mlir::arith::ArithDialect,
                  mlir::gpu::GPUDialect, mlir::math::MathDialect,
                  mlir::ub::UBDialect, crucible::ir::CrucibleDialect>();
  Ctx.appendDialectRegistry(Registry);
  Ctx.loadAllAvailableDialects();

  auto Mod = mlir::parseSourceString<mlir::ModuleOp>(Buf.getBuffer(), &Ctx);
  if (!Mod)
    return makeError("failed to parse MLIR module");

  mlir::func::FuncOp EntryFn;
  Mod->walk([&](mlir::func::FuncOp F) {
    if (F.isDeclaration())
      return;
    if (EntryOverride.empty() || F.getName() == EntryOverride) {
      if (!EntryFn)
        EntryFn = F;
    }
  });
  if (!EntryFn)
    return makeError("no matching func.func entry point found in MLIR module");

  Reflection Reflect;
  Reflect.Stage = ShaderStage::Compute;
  Reflect.EntryName = EntryFn.getName().str();
  Reflect.NumThreads = {64, 1, 1};
  if (auto NT = EntryFn->getAttrOfType<mlir::DenseI32ArrayAttr>(
          "crucible.num_threads")) {
    if (NT.size() == 3) {
      Reflect.NumThreads = {static_cast<uint32_t>(NT[0]),
                            static_cast<uint32_t>(NT[1]),
                            static_cast<uint32_t>(NT[2])};
    }
  }

  LoadedMLIRShader Out;
  Out.Module = std::move(Mod);
  Out.Reflect = std::move(Reflect);
  return std::move(Out);
}

} // namespace crucible

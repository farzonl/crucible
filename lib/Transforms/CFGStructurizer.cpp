//===- CFGStructurizer.cpp - Lift DXIL CFG to structured SCF -----*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// MSL forbids unstructured control flow, but DXIL is plain (unstructured)
// LLVM IR. This file imports the upgraded LLVM module into MLIR and runs
// LiftControlFlowToSCF -- MLIR's implementation of the Bahmann et al.
// "perfect reconstructability" algorithm -- turning arbitrary shader CFGs
// into nested scf.if / scf.while regions with explicit value yields (which
// double as PHI demotion). Ported from Nacelle's mlir_pipeline.cpp.
//
// The frontend pipeline has already run LowerSwitchPass, so only br/cond_br/
// ret/unreachable terminators reach this point.
//
//===----------------------------------------------------------------------===//

#include "Transforms.h"
#include "crucible/Diagnostic.h"

#include "llvm/IR/Module.h"

#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Transforms/Passes.h"

using namespace llvm;

namespace crucible {

// Rewrite the entry llvm.func into a func.func with cf.* terminators (the
// shape LiftControlFlowToSCF requires), unifying multiple exits into one
// return block.
static Error convertEntryToCF(mlir::ModuleOp Mod, StringRef EntryName) {
  mlir::MLIRContext &Ctx = *Mod.getContext();
  mlir::LLVM::LLVMFuncOp LLVMFunc;
  Mod.walk([&](mlir::LLVM::LLVMFuncOp F) {
    if (F.getName() == EntryName)
      LLVMFunc = F;
  });
  if (!LLVMFunc)
    return makeError("entry '{0}' not found after MLIR import", EntryName);

  mlir::OpBuilder B(&Ctx);
  B.setInsertionPoint(LLVMFunc);
  auto LLVMFnTy = LLVMFunc.getFunctionType();
  SmallVector<mlir::Type, 1> ResTys;
  if (!isa<mlir::LLVM::LLVMVoidType>(LLVMFnTy.getReturnType()))
    ResTys.push_back(LLVMFnTy.getReturnType());
  auto FnTy = B.getFunctionType(LLVMFnTy.getParams(), ResTys);
  auto FuncOp =
      mlir::func::FuncOp::create(B, LLVMFunc.getLoc(), LLVMFunc.getName(), FnTy);
  FuncOp.getBody().takeBody(LLVMFunc.getBody());
  LLVMFunc.erase();

  SmallVector<mlir::Block *, 16> Blocks;
  for (mlir::Block &BB : FuncOp.getBody())
    Blocks.push_back(&BB);
  for (mlir::Block *BB : Blocks) {
    if (BB->empty())
      continue;
    mlir::Operation *Term = BB->getTerminator();
    mlir::OpBuilder TB(Term);
    mlir::Location Loc = Term->getLoc();
    if (auto Br = dyn_cast<mlir::LLVM::BrOp>(Term)) {
      mlir::cf::BranchOp::create(TB, Loc, Br.getDest(), Br.getDestOperands());
      Br.erase();
    } else if (auto CBr = dyn_cast<mlir::LLVM::CondBrOp>(Term)) {
      mlir::cf::CondBranchOp::create(
          TB, Loc, CBr.getCondition(), CBr.getTrueDest(),
          CBr.getTrueDestOperands(), CBr.getFalseDest(),
          CBr.getFalseDestOperands());
      CBr.erase();
    } else if (auto Ret = dyn_cast<mlir::LLVM::ReturnOp>(Term)) {
      mlir::func::ReturnOp::create(TB, Loc, Ret.getOperands());
      Ret.erase();
    } else if (auto Unreach = dyn_cast<mlir::LLVM::UnreachableOp>(Term)) {
      SmallVector<mlir::Value, 1> Poison;
      for (mlir::Type RT : ResTys)
        Poison.push_back(mlir::LLVM::PoisonOp::create(TB, Loc, RT));
      mlir::func::ReturnOp::create(TB, Loc, Poison);
      Unreach.erase();
    } else if (isa<mlir::LLVM::SwitchOp>(Term)) {
      return makeError("unexpected llvm.switch after LowerSwitchPass");
    }
  }

  // LiftControlFlowToSCF requires at most one exit block.
  SmallVector<mlir::Block *, 4> ReturnBlocks;
  for (mlir::Block &BB : FuncOp.getBody())
    if (!BB.empty() && isa<mlir::func::ReturnOp>(BB.getTerminator()))
      ReturnBlocks.push_back(&BB);
  if (ReturnBlocks.size() > 1) {
    mlir::Block *Exit = new mlir::Block();
    FuncOp.getBody().push_back(Exit);
    SmallVector<mlir::Value, 1> ExitArgs;
    for (mlir::Type RT : ResTys)
      ExitArgs.push_back(Exit->addArgument(RT, FuncOp.getLoc()));
    for (mlir::Block *RB : ReturnBlocks) {
      auto RetOp = cast<mlir::func::ReturnOp>(RB->getTerminator());
      mlir::OpBuilder EB(RetOp);
      mlir::cf::BranchOp::create(EB, RetOp.getLoc(), Exit, RetOp.getOperands());
      RetOp.erase();
    }
    mlir::OpBuilder ExitB = mlir::OpBuilder::atBlockEnd(Exit);
    mlir::func::ReturnOp::create(ExitB, FuncOp.getLoc(), ExitArgs);
  }
  return Error::success();
}

Expected<StructurizedShader>
structurizeMLIRModule(mlir::OwningOpRef<mlir::ModuleOp> Mod,
                      const Reflection &Reflect) {
  mlir::MLIRContext &Ctx = *Mod->getContext();
  std::string EntryName = Reflect.EntryName;

  if (!Mod->lookupSymbol<mlir::func::FuncOp>(EntryName)) {
    if (Error E = convertEntryToCF(*Mod, EntryName))
      return std::move(E);
  }
  if (Reflect.Hull) {
    if (Error E = convertEntryToCF(*Mod, Reflect.Hull->PatchConstantFuncName))
      return std::move(E);
  }
  for (const RTShaderInfo &RT : Reflect.RTShaders) {
    if (Error E = convertEntryToCF(*Mod, RT.SymbolName))
      return std::move(E);
  }

  mlir::PassManager PM(&Ctx);
  PM.addPass(mlir::createLiftControlFlowToSCFPass());
  PM.addPass(mlir::createCanonicalizerPass());
  PM.addPass(mlir::createLoopInvariantCodeMotionPass());
  PM.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(PM.run(*Mod)))
    return makeError("CFG structurization (LiftControlFlowToSCF) failed");
  if (mlir::failed(mlir::verify(*Mod)))
    return makeError("MLIR verification failed after structurization");

  StructurizedShader Out;
  Out.Module = std::move(Mod);
  Out.Module->walk([&](mlir::func::FuncOp F) {
    if (F.getName() == EntryName)
      Out.Entry = F;
  });
  if (!Out.Entry)
    return makeError("entry '{0}' lost during structurization", EntryName);
  return std::move(Out);
}

Expected<StructurizedShader> structurizeCFG(LoadedShader &Shader,
                                            mlir::MLIRContext &Ctx) {
  // The MLIR importer rejects DXIL's named metadata; reflection was already
  // captured by the frontend, so drop it.
  llvm::Module &M = *Shader.M;
  SmallVector<llvm::NamedMDNode *, 8> Named;
  for (llvm::NamedMDNode &N : M.named_metadata())
    Named.push_back(&N);
  for (auto *N : Named)
    N->eraseFromParent();

  mlir::DialectRegistry Registry;
  Registry.insert<mlir::LLVM::LLVMDialect, mlir::DLTIDialect,
                  mlir::func::FuncDialect, mlir::cf::ControlFlowDialect,
                  mlir::scf::SCFDialect, mlir::arith::ArithDialect,
                  mlir::gpu::GPUDialect, mlir::math::MathDialect>();
  mlir::registerLLVMDialectImport(Registry);
  Ctx.appendDialectRegistry(Registry);
  Ctx.loadDialect<mlir::gpu::GPUDialect, mlir::math::MathDialect>();

  auto Mod = mlir::translateLLVMIRToModule(std::move(Shader.M), &Ctx);
  if (!Mod)
    return makeError("MLIR import of the DXIL module failed");

  lowerDXILToLayer1(*Mod);
  return structurizeMLIRModule(std::move(Mod), Shader.Reflect);
}

} // namespace crucible

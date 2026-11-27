//===- Compiler.cpp - Public DXIL -> MSL compile driver ----------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "crucible/Compiler.h"

#include "Backend/DXILEmitter.h"
#include "Backend/MSLEmitter.h"
#include "Backend/SPIRVEmitter.h"
#include "Frontend/Frontend.h"
#include "Transforms/MiddleEnd.h"
#include "Transforms/Transforms.h"

#include "llvm/IR/LLVMContext.h"
#include "mlir/IR/MLIRContext.h"

using namespace llvm;

namespace crucible {

Expected<CompileResult> compileToMSL(MemoryBufferRef Input,
                                       const Options &Opts) {
  if (isSPIRVBinary(Input) || isMLIRText(Input)) {
    mlir::MLIRContext MCtx(mlir::MLIRContext::Threading::DISABLED);
    auto Loaded = isSPIRVBinary(Input)
                      ? loadSPIRV(Input, MCtx, Opts.EntryPoint,
                                  Opts.SpecConstants)
                      : loadMLIR(Input, MCtx, Opts.EntryPoint);
    if (!Loaded)
      return Loaded.takeError();

    if (Error E = runMiddleEnd(*Loaded->Module, Loaded->Reflect,
                               TargetCaps::metal()))
      return std::move(E);

    if (Error E = assignMetalBindings(Loaded->Reflect, Opts))
      return std::move(E);

    auto Structured =
        structurizeMLIRModule(std::move(Loaded->Module), Loaded->Reflect);
    if (!Structured)
      return Structured.takeError();

    if (Opts.DumpMLIR) {
      Structured->Entry.print(llvm::errs());
      llvm::errs() << "\n";
    }

    auto MSL = emitMSL(Structured->Entry, Loaded->Reflect, Opts);
    if (!MSL)
      return MSL.takeError();

    CompileResult Result;
    Result.MSL = std::move(*MSL);
    Result.Reflect = std::move(Loaded->Reflect);
    return Result;
  }

  LLVMContext Ctx;
  auto Shader = loadDXIL(Input, Ctx, Opts.EntryPoint);
  if (!Shader)
    return Shader.takeError();

  mlir::MLIRContext MCtx(mlir::MLIRContext::Threading::DISABLED);
  auto Mod = importDXILToMLIR(*Shader, MCtx);
  if (!Mod)
    return Mod.takeError();

  if (Error E = runMiddleEnd(**Mod, Shader->Reflect, TargetCaps::metal()))
    return std::move(E);

  if (Error E = assignMetalBindings(Shader->Reflect, Opts))
    return std::move(E);

  auto Structured = structurizeMLIRModule(std::move(*Mod), Shader->Reflect);
  if (!Structured)
    return Structured.takeError();

  if (Opts.DumpMLIR) {
    Structured->Entry.print(llvm::errs());
    llvm::errs() << "\n";
  }

  auto MSL = emitMSL(Structured->Entry, Shader->Reflect, Opts);
  if (!MSL)
    return MSL.takeError();

  CompileResult Result;
  Result.MSL = std::move(*MSL);
  Result.Reflect = std::move(Shader->Reflect);
  return Result;
}

Expected<SPIRVCompileResult> compileDXILToSPIRV(MemoryBufferRef DXIL,
                                                const Options &Opts) {
  LLVMContext Ctx;
  auto Shader = loadDXIL(DXIL, Ctx, Opts.EntryPoint);
  if (!Shader)
    return Shader.takeError();

  mlir::MLIRContext MCtx(mlir::MLIRContext::Threading::DISABLED);
  auto Mod = importDXILToMLIR(*Shader, MCtx);
  if (!Mod)
    return Mod.takeError();

  if (Error E = runMiddleEnd(**Mod, Shader->Reflect, TargetCaps::spirv()))
    return std::move(E);

  if (Error E = assignSPIRVBindings(Shader->Reflect, Opts))
    return std::move(E);

  if (Opts.DumpMLIR) {
    (*Mod)->print(llvm::errs());
    llvm::errs() << "\n";
  }

  auto Out = emitSPIRV(**Mod, Shader->Reflect, Opts);
  if (!Out)
    return Out.takeError();

  SPIRVCompileResult Result;
  Result.Output = std::move(*Out);
  Result.IsAssembly = Opts.EmitSPIRVAsm;
  Result.Reflect = std::move(Shader->Reflect);
  return Result;
}

Expected<CompileResult> compileDXILToMSL(MemoryBufferRef DXIL,
                                         const Options &Opts) {
  return compileToMSL(DXIL, Opts);
}

Expected<DXILCompileResult> compileToDXIL(MemoryBufferRef Input,
                                          const Options &Opts) {
  if (isSPIRVBinary(Input) || isMLIRText(Input)) {
    mlir::MLIRContext MCtx(mlir::MLIRContext::Threading::DISABLED);
    auto Loaded = isSPIRVBinary(Input)
                      ? loadSPIRV(Input, MCtx, Opts.EntryPoint,
                                  Opts.SpecConstants)
                      : loadMLIR(Input, MCtx, Opts.EntryPoint);
    if (!Loaded)
      return Loaded.takeError();

    // Lower any frontend dx.op.* calls into Crucible's unified MLIR contract
    // (gpu.* + math.* + llvm.* + crucible.*).
    lowerDXILToLayer1(*Loaded->Module);

    if (Error E = runMiddleEnd(*Loaded->Module, Loaded->Reflect,
                               TargetCaps::dxil()))
      return std::move(E);

    if (Error E = assignDXILBindings(Loaded->Reflect, Opts))
      return std::move(E);

    if (Opts.DumpMLIR) {
      (*Loaded->Module).print(llvm::errs());
      llvm::errs() << "\n";
    }

    auto Out = emitDXIL(*Loaded->Module, Loaded->Reflect, Opts);
    if (!Out)
      return Out.takeError();

    DXILCompileResult Result;
    Result.Output = std::move(*Out);
    Result.IsAssembly = Opts.EmitDXILAsm;
    Result.Reflect = std::move(Loaded->Reflect);
    return Result;
  }

  LLVMContext Ctx;
  auto Shader = loadDXIL(Input, Ctx, Opts.EntryPoint);
  if (!Shader)
    return Shader.takeError();

  mlir::MLIRContext MCtx(mlir::MLIRContext::Threading::DISABLED);
  auto Mod = importDXILToMLIR(*Shader, MCtx);
  if (!Mod)
    return Mod.takeError();

  if (Error E = runMiddleEnd(**Mod, Shader->Reflect, TargetCaps::dxil()))
    return std::move(E);

  if (Error E = assignDXILBindings(Shader->Reflect, Opts))
    return std::move(E);

  if (Opts.DumpMLIR) {
    (*Mod)->print(llvm::errs());
    llvm::errs() << "\n";
  }

  auto Out = emitDXIL(**Mod, Shader->Reflect, Opts);
  if (!Out)
    return Out.takeError();

  DXILCompileResult Result;
  Result.Output = std::move(*Out);
  Result.IsAssembly = Opts.EmitDXILAsm;
  Result.Reflect = std::move(Shader->Reflect);
  return Result;
}
} // namespace crucible

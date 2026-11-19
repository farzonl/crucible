//===- Compiler.cpp - Public DXIL -> MSL compile driver ----------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "crucible/Compiler.h"

#include "Backend/MSLEmitter.h"
#include "Backend/SPIRVEmitter.h"
#include "Frontend/Frontend.h"
#include "Transforms/Transforms.h"

#include "llvm/IR/LLVMContext.h"
#include "mlir/IR/MLIRContext.h"

using namespace llvm;

namespace crucible {

Expected<CompileResult> compileToMSL(MemoryBufferRef Input,
                                       const Options &Opts) {
  if (isSPIRVBinary(Input) || isMLIRText(Input)) {
    mlir::MLIRContext MCtx;
    auto Loaded = isSPIRVBinary(Input)
                      ? loadSPIRV(Input, MCtx, Opts.EntryPoint)
                      : loadMLIR(Input, MCtx, Opts.EntryPoint);
    if (!Loaded)
      return Loaded.takeError();

    if (Error E = assignMetalBindings(Loaded->Reflect, Opts))
      return std::move(E);

    auto Structured =
        structurizeMLIRModule(std::move(Loaded->Module), Loaded->Reflect);
    if (!Structured)
      return Structured.takeError();

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

  if (Error E = assignMetalBindings(Shader->Reflect, Opts))
    return std::move(E);

  mlir::MLIRContext MCtx;
  auto Structured = structurizeCFG(*Shader, MCtx);
  if (!Structured)
    return Structured.takeError();

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

  if (Error E = assignSPIRVBindings(Shader->Reflect, Opts))
    return std::move(E);

  auto Out = emitSPIRV(*Shader->M, Shader->Entry, Shader->Reflect, Opts);
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
} // namespace crucible

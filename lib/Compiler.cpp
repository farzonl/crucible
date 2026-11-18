//===- Compiler.cpp - Public DXIL -> MSL compile driver ----------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "crucible/Compiler.h"

#include "Backend/MSLEmitter.h"
#include "Frontend/Frontend.h"
#include "Transforms/Transforms.h"

#include "llvm/IR/LLVMContext.h"

using namespace llvm;

namespace crucible {

Expected<CompileResult> compileDXILToMSL(MemoryBufferRef DXIL,
                                         const Options &Opts) {
  LLVMContext Ctx;
  auto Shader = loadDXIL(DXIL, Ctx, Opts.EntryPoint);
  if (!Shader)
    return Shader.takeError();

  if (Error E = assignMetalBindings(Shader->Reflect, Opts))
    return std::move(E);

  auto MSL = emitMSL(*Shader, Opts);
  if (!MSL)
    return MSL.takeError();

  CompileResult Result;
  Result.MSL = std::move(*MSL);
  Result.Reflect = std::move(Shader->Reflect);
  return Result;
}

} // namespace crucible

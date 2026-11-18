//===- Compiler.h - Public DXIL -> MSL compile API --------------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_COMPILER_H
#define CRUCIBLE_COMPILER_H

#include "crucible/Options.h"
#include "crucible/Reflection.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <string>

namespace crucible {

struct CompileResult {
  /// Human-readable Metal Shading Language source.
  std::string MSL;
  /// Binding reflection for the host runtime.
  Reflection Reflect;
};

/// Translate a DXContainer (.dxil) binary -- or bare DXIL bitcode -- into
/// Metal Shading Language source plus binding reflection.
llvm::Expected<CompileResult> compileDXILToMSL(llvm::MemoryBufferRef DXIL,
                                               const Options &Opts = {});

} // namespace crucible

#endif // CRUCIBLE_COMPILER_H

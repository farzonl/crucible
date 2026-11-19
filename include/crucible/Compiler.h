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

/// Translate a shader binary (DXContainer, bare DXIL bitcode, or SPIR-V) into
/// Metal Shading Language source plus binding reflection.
llvm::Expected<CompileResult> compileToMSL(llvm::MemoryBufferRef Input,
                                           const Options &Opts = {});

/// Alias for compileToMSL retained for backwards compatibility.
llvm::Expected<CompileResult> compileDXILToMSL(llvm::MemoryBufferRef DXIL,
                                               const Options &Opts = {});

struct SPIRVCompileResult {
  /// Compiled SPIR-V binary (when Opts.EmitSPIRVAsm == false) or
  /// human-readable SPIR-V assembly (when Opts.EmitSPIRVAsm == true).
  std::string Output;
  /// True when Output contains human-readable SPIR-V assembly text.
  bool IsAssembly = false;
  /// Binding reflection for the host runtime.
  Reflection Reflect;
};

/// Translate a DXContainer (.dxil) binary -- or bare DXIL bitcode -- into
/// Vulkan SPIR-V (binary or assembly) using LLVM's SPIR-V backend, plus
/// binding reflection.
llvm::Expected<SPIRVCompileResult>
compileDXILToSPIRV(llvm::MemoryBufferRef DXIL, const Options &Opts = {});

} // namespace crucible

#endif // CRUCIBLE_COMPILER_H

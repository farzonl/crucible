//===- Frontend.h - Internal frontend interfaces ----------------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_LIB_FRONTEND_FRONTEND_H
#define CRUCIBLE_LIB_FRONTEND_FRONTEND_H

#include "crucible/Reflection.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <memory>

namespace crucible {

/// Result of loading and upgrading a DXIL shader into modern LLVM IR.
struct LoadedShader {
  std::unique_ptr<llvm::Module> M;
  llvm::Function *Entry = nullptr;
  Reflection Reflect;
};

/// Result of loading a shader directly into MLIR (e.g. from SPIR-V) ready for
/// CFG structurization and MSL emission.
struct LoadedMLIRShader {
  mlir::OwningOpRef<mlir::ModuleOp> Module;
  Reflection Reflect;
};

/// Extract the raw DXIL bitcode slice from a DXContainer binary. Buffers that
/// are already bare LLVM bitcode ('BC\xC0\xDE') are returned unchanged.
llvm::Expected<llvm::StringRef> extractDXILBitcode(llvm::MemoryBufferRef Buf);

/// Parse DXIL bitcode (frozen LLVM 3.7) into a modern LLVM module, run the
/// upgrade/cleanup pipeline, and extract reflection metadata.
llvm::Expected<LoadedShader> loadDXIL(llvm::MemoryBufferRef Buf,
                                      llvm::LLVMContext &Ctx,
                                      llvm::StringRef EntryOverride = {});

/// Returns true when Buf begins with the SPIR-V binary magic (0x07230203).
bool isSPIRVBinary(llvm::MemoryBufferRef Buf);

/// Deserialize a SPIR-V binary into MLIR, extract reflection metadata, and
/// lower shader-ABI ops into Crucible's layered MLIR representation.
llvm::Expected<LoadedMLIRShader> loadSPIRV(llvm::MemoryBufferRef Buf,
                                           mlir::MLIRContext &Ctx,
                                           llvm::StringRef EntryOverride = {});

/// Returns true when Buf contains standalone MLIR text (Layer 0 + Layer 1).
bool isMLIRText(llvm::MemoryBufferRef Buf);

/// Parse a standalone Layer 0 + Layer 1 MLIR text module directly for MSL
/// emission (validates the PTX/compute-IR backend contract).
llvm::Expected<LoadedMLIRShader> loadMLIR(llvm::MemoryBufferRef Buf,
                                          mlir::MLIRContext &Ctx,
                                          llvm::StringRef EntryOverride = {});

} // namespace crucible

#endif // CRUCIBLE_LIB_FRONTEND_FRONTEND_H

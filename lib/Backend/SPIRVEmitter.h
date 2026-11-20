//===- SPIRVEmitter.h - DXIL -> LLVM SPIR-V backend -------------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_LIB_BACKEND_SPIRVEMITTER_H
#define CRUCIBLE_LIB_BACKEND_SPIRVEMITTER_H

#include "crucible/Options.h"
#include "crucible/Reflection.h"
#include "llvm/Support/Error.h"
#include "mlir/IR/BuiltinOps.h"

#include <string>

namespace crucible {

/// Lower a unified Crucible MLIR module (gpu.* + math.* + llvm.* + crucible.*,
/// with zero DXILOp dependencies) to LLVM SPIR-V IR and emit Vulkan SPIR-V
/// (binary words when Opts.EmitSPIRVAsm is false, or human-readable SPIR-V
/// assembly when true) via LLVM's SPIR-V TargetMachine.
llvm::Expected<std::string> emitSPIRV(mlir::ModuleOp Mod, Reflection &Reflect,
                                      const Options &Opts);

} // namespace crucible

#endif // CRUCIBLE_LIB_BACKEND_SPIRVEMITTER_H

//===- DXILEmitter.h - Crucible MLIR -> LLVM DirectX backend ----*- C++ -*-===//
//
// Part of Crucible, an open-source multi-frontend, multi-backend shader compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_LIB_BACKEND_DXILEMITTER_H
#define CRUCIBLE_LIB_BACKEND_DXILEMITTER_H

#include "crucible/Options.h"
#include "crucible/Reflection.h"
#include "llvm/Support/Error.h"
#include "mlir/IR/BuiltinOps.h"

#include <string>

namespace crucible {

/// Lower a unified Crucible MLIR module (gpu.* + math.* + llvm.* + crucible.*,
/// with zero DXILOp dependencies) to LLVM DirectX target IR and emit either a
/// DXContainer binary (when Opts.EmitDXILAsm is false) or human-readable DXIL
/// LLVM IR assembly (when true) via LLVM 24's DirectX TargetMachine.
llvm::Expected<std::string> emitDXIL(mlir::ModuleOp Mod, Reflection &Reflect,
                                     const Options &Opts);

} // namespace crucible

#endif // CRUCIBLE_LIB_BACKEND_DXILEMITTER_H

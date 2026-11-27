//===- MiddleEnd.h - Crucible shared MLIR middle-end pipeline ----*- C++ -*-===//
//
// Part of Crucible, an open-source multi-frontend, multi-backend shader compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shared middle-end between all frontends (DXIL, SPIR-V, MLIR) and all backends
// (MSL, SPIR-V, DXIL). Normalizes and verifies the Layer 0/1/2 MLIR + Reflection
// contract once, then runs capability-driven target legalizations.
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_LIB_TRANSFORMS_MIDDLE_END_H
#define CRUCIBLE_LIB_TRANSFORMS_MIDDLE_END_H

#include "crucible/Reflection.h"
#include "llvm/Support/Error.h"
#include "mlir/IR/BuiltinOps.h"

namespace crucible {

/// Backend capability flags controlling opt-in middle-end legalizations.
struct TargetCaps {
  /// Target requires structured SCF control flow (e.g., MSL).
  bool RequiresStructuredCFG = false;
  /// Target has a native atan2(y, x) operation.
  bool HasNativeAtan2 = true;
  /// Target natively lowers math.ctlz / math.cttz on i32.
  bool HasNativeCtlzCttz = true;
  /// Target natively lowers math.fma on f16.
  bool HasNativeF16Fma = true;
  /// Target natively lowers math.fma on f32.
  bool HasNativeF32Fma = true;

  static TargetCaps metal() {
    TargetCaps C;
    C.RequiresStructuredCFG = true;
    return C;
  }

  static TargetCaps spirv() {
    TargetCaps C;
    return C;
  }

  static TargetCaps dxil() {
    TargetCaps C;
    C.HasNativeAtan2 = false;
    C.HasNativeCtlzCttz = false;
    C.HasNativeF16Fma = false;
    C.HasNativeF32Fma = false;
    return C;
  }
};

/// Normalize signed/unsigned integer attributes on llvm.mlir.constant ops to
/// match their signless LLVM integer result types.
void normalizeSignlessLLVMConstants(mlir::ModuleOp Mod);

/// Run the shared Crucible middle-end pipeline:
///   1. Normalize signless LLVM constant attributes.
///   2. Normalize and validate Reflection invariants (e.g., CBuffer byte size
///      against constant cbuffer_load register indices, non-zero RangeSize).
///   3. Verify the Layer 0/1/2 MLIR contract (no leaked spirv.* ops, valid IR).
///   4. Apply capability-driven legalizations (atan2, ctlz/cttz, f16/f32 fma).
llvm::Error runMiddleEnd(mlir::ModuleOp Mod, Reflection &Reflect,
                         const TargetCaps &Caps);

} // namespace crucible

#endif // CRUCIBLE_LIB_TRANSFORMS_MIDDLE_END_H

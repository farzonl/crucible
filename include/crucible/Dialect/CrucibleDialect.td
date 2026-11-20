//===- CrucibleDialect.td - Crucible MLIR dialect definition -*- tablegen -*-===//
//
// Part of Crucible, an open-source multi-frontend, multi-backend shader compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_DIALECT_TD
#define CRUCIBLE_DIALECT_TD

include "mlir/IR/OpBase.td"

def Crucible_Dialect : Dialect {
  let name = "crucible";
  let summary = "Frontend-agnostic GPU shader dialect for Crucible";
  let description = [{
    The `crucible` dialect models GPU shader concepts not covered by upstream
    MLIR dialects (`scf`, `arith`, `math`, `gpu`, `llvm`), including shader
    resource handles, buffer/texture operations, stage I/O, pixel derivatives,
    wave/quad extensions, mesh/amplification shaders, and raytracing/RayQuery.
  }];
  let cppNamespace = "::crucible::ir";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // CRUCIBLE_DIALECT_TD

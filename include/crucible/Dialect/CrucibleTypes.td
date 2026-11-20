//===- CrucibleTypes.td - Crucible MLIR dialect types -------*- tablegen -*-===//
//
// Part of Crucible, an open-source multi-frontend, multi-backend shader compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_TYPES_TD
#define CRUCIBLE_TYPES_TD

include "crucible/Dialect/CrucibleDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Crucible_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Crucible_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Crucible_ResourceType : Crucible_Type<"Resource", "resource"> {
  let summary = "Opaque shader resource handle (SRV, UAV, CBV, or Sampler)";
}

def Crucible_RayQueryType : Crucible_Type<"RayQuery", "ray_query"> {
  let summary = "Opaque inline RayQuery object handle";
}

#endif // CRUCIBLE_TYPES_TD

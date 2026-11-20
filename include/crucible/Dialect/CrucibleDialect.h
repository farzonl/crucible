//===- CrucibleDialect.h - Crucible MLIR dialect header ---------*- C++ -*-===//
//
// Part of Crucible, an open-source multi-frontend, multi-backend shader compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_DIALECT_CRUCIBLEDIALECT_H
#define CRUCIBLE_DIALECT_CRUCIBLEDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "crucible/Dialect/CrucibleDialect.h.inc"

#include "crucible/Dialect/CrucibleEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "crucible/Dialect/CrucibleEnumAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "crucible/Dialect/CrucibleTypes.h.inc"

#define GET_OP_CLASSES
#include "crucible/Dialect/CrucibleOps.h.inc"

#endif // CRUCIBLE_DIALECT_CRUCIBLEDIALECT_H

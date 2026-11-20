//===- CrucibleDialect.cpp - Crucible MLIR dialect implementation *- C++ -*-===//
//
// Part of Crucible, an open-source multi-frontend, multi-backend shader compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "crucible/Dialect/CrucibleDialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace crucible::ir;

#include "crucible/Dialect/CrucibleDialect.cpp.inc"

#include "crucible/Dialect/CrucibleEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "crucible/Dialect/CrucibleEnumAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "crucible/Dialect/CrucibleTypes.cpp.inc"

#define GET_OP_CLASSES
#include "crucible/Dialect/CrucibleOps.cpp.inc"

void CrucibleDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "crucible/Dialect/CrucibleTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "crucible/Dialect/CrucibleEnumAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "crucible/Dialect/CrucibleOps.cpp.inc"
      >();
}

//===- Diagnostic.h - Structured error reporting ----------------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_DIAGNOSTIC_H
#define CRUCIBLE_DIAGNOSTIC_H

#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

namespace crucible {

/// Create a structured compilation error. All crucible APIs report failures
/// through llvm::Expected / llvm::Error carrying one of these.
template <typename... Ts>
llvm::Error makeError(const char *Fmt, Ts &&...Vals) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 llvm::formatv(Fmt, std::forward<Ts>(Vals)...)
                                     .str()
                                     .c_str());
}

inline llvm::Error makeError(const llvm::Twine &Msg) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), Msg.str());
}

} // namespace crucible

#endif // CRUCIBLE_DIAGNOSTIC_H

//===- Options.h - Crucible compiler options -------------------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_OPTIONS_H
#define CRUCIBLE_OPTIONS_H

#include <cstdint>
#include <string>

namespace crucible {

/// How resources are bound in the emitted MSL.
enum class BindingMode : uint8_t {
  /// Zero-overhead direct slots: each DXIL resource maps deterministically to
  /// a sequential Metal [[buffer(n)]] / [[texture(n)]] index.
  Direct,
  /// Compact custom argument buffer (not yet implemented).
  ArgumentBuffer,
};

struct Options {
  BindingMode Binding = BindingMode::Direct;

  /// MSL language version passed to the Apple toolchain (e.g. "metal3.1").
  std::string MSLStd = "metal3.1";

  /// Entry point to compile when the container exports multiple (libraries).
  /// Empty selects the sole entry point.
  std::string EntryPoint;
};

} // namespace crucible

#endif // CRUCIBLE_OPTIONS_H

//===- Options.h - Crucible compiler options -------------------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_OPTIONS_H
#define CRUCIBLE_OPTIONS_H

#include "crucible/Reflection.h"

#include <cstdint>
#include <string>
#include <vector>

namespace crucible {

/// How resources are bound in the emitted MSL.
enum class BindingMode : uint8_t {
  /// Zero-overhead direct slots: each DXIL resource maps deterministically to
  /// a sequential Metal [[buffer(n)]] / [[texture(n)]] index.
  Direct,
  /// Compact custom argument buffer (not yet implemented).
  ArgumentBuffer,
};

struct ArraySizeOverride {
  ResourceClass Class = ResourceClass::SRV;
  uint32_t Space = 0;
  uint32_t Register = 0;
  uint32_t Size = 1;
};

struct Options {
  BindingMode Binding = BindingMode::Direct;

  /// MSL language version passed to the Apple toolchain (e.g. "metal3.1").
  std::string MSLStd = "metal3.1";

  /// Entry point to compile when the container exports multiple (libraries).
  /// Empty selects the sole entry point.
  std::string EntryPoint;

  /// Flush f32 denormals to signed zero around math builtins (D3D12 FTZ).
  bool FlushDenormsF32 = false;

  /// Explicit array sizes for unbounded DXIL resource arrays (rangeSize == -1).
  std::vector<ArraySizeOverride> ArraySizeOverrides;
};

} // namespace crucible

#endif // CRUCIBLE_OPTIONS_H

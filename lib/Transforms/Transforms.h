//===- Transforms.h - Internal transform interfaces -------------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_LIB_TRANSFORMS_TRANSFORMS_H
#define CRUCIBLE_LIB_TRANSFORMS_TRANSFORMS_H

#include "crucible/Options.h"
#include "crucible/Reflection.h"
#include "llvm/Support/Error.h"

namespace crucible {

/// Assign deterministic Metal binding slots to every DXIL resource range.
/// In Direct mode buffers receive sequential [[buffer(n)]] indices ordered by
/// (class: SRV, UAV, CBV; then space; then register), and samplers receive
/// sequential [[sampler(n)]] indices. The mapping is recorded in-place so the
/// reflection JSON and the MSL emitter always agree.
llvm::Error assignMetalBindings(Reflection &R, const Options &Opts);

} // namespace crucible

#endif // CRUCIBLE_LIB_TRANSFORMS_TRANSFORMS_H

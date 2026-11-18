//===- MSLEmitter.h - Internal MSL emission interface -----------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_LIB_BACKEND_MSLEMITTER_H
#define CRUCIBLE_LIB_BACKEND_MSLEMITTER_H

#include "../Frontend/Frontend.h"
#include "crucible/Options.h"
#include "llvm/Support/Error.h"

#include <string>

namespace crucible {

/// Emit Metal Shading Language source for a loaded shader. Metal binding
/// slots must already be assigned (see assignMetalBindings). Updates
/// Shader.Reflect.MSLEntryName with the (possibly sanitized) kernel name.
llvm::Expected<std::string> emitMSL(LoadedShader &Shader, const Options &Opts);

} // namespace crucible

#endif // CRUCIBLE_LIB_BACKEND_MSLEMITTER_H

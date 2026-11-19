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
#include "../Frontend/Frontend.h"
#include "llvm/Support/Error.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace crucible {

/// Assign deterministic Metal binding slots to every DXIL resource range.
/// In Direct mode buffers receive sequential [[buffer(n)]] indices ordered by
/// (class: SRV, UAV, CBV; then space; then register), and samplers receive
/// sequential [[sampler(n)]] indices. The mapping is recorded in-place so the
/// reflection JSON and the MSL emitter always agree.
llvm::Error assignMetalBindings(Reflection &R, const Options &Opts);

/// Assign deterministic Vulkan (DescriptorSet, Binding) slots to every DXIL
/// resource range. DescriptorSet matches DXIL Space; bindings within each set
/// are assigned sequentially ordered by (class: SRV, UAV, CBV, Sampler; then
/// register).
llvm::Error assignSPIRVBindings(Reflection &R, const Options &Opts);

/// A shader whose CFG has been lifted into structured MLIR scf.if / scf.while
/// regions ready for source emission.
struct StructurizedShader {
  mlir::OwningOpRef<mlir::ModuleOp> Module;
  mlir::func::FuncOp Entry;
};

/// Import the upgraded LLVM module into MLIR and lift its unstructured CFG to
/// structured SCF control flow (MLIR's LiftControlFlowToSCF implements the
/// "perfect reconstructability" algorithm, so arbitrary CFGs become goto-free
/// if/else and loop regions). Consumes Shader.M; Ctx must outlive the result.
llvm::Expected<StructurizedShader> structurizeCFG(LoadedShader &Shader,
                                                  mlir::MLIRContext &Ctx);

} // namespace crucible

#endif // CRUCIBLE_LIB_TRANSFORMS_TRANSFORMS_H

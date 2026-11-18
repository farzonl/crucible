//===- ResourceLowering.cpp - DXIL resource -> Metal slot mapping -*- C++ -*-=//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Direct binding mode: map each (class, space, register) range to a
// deterministic, zero-overhead Metal slot. Buffer-like resources (typed/raw/
// structured buffers and constant buffers) share the [[buffer(n)]] namespace;
// textures and samplers get their own namespaces.
//
//===----------------------------------------------------------------------===//

#include "Transforms.h"
#include "crucible/Diagnostic.h"

#include <algorithm>

namespace crucible {

static bool isBufferLike(const ResourceBinding &RB) {
  switch (RB.Kind) {
  case ResourceKind::TypedBuffer:
  case ResourceKind::RawBuffer:
  case ResourceKind::StructuredBuffer:
  case ResourceKind::CBuffer:
  case ResourceKind::TBuffer:
    return true;
  default:
    return false;
  }
}

static bool isTexture(const ResourceBinding &RB) {
  switch (RB.Kind) {
  case ResourceKind::Texture1D:
  case ResourceKind::Texture2D:
  case ResourceKind::Texture2DMS:
  case ResourceKind::Texture3D:
  case ResourceKind::TextureCube:
  case ResourceKind::Texture1DArray:
  case ResourceKind::Texture2DArray:
  case ResourceKind::Texture2DMSArray:
  case ResourceKind::TextureCubeArray:
    return true;
  default:
    return false;
  }
}

llvm::Error assignMetalBindings(Reflection &R, const Options &Opts) {
  if (Opts.Binding != BindingMode::Direct)
    return makeError("argument-buffer binding mode is not implemented yet");

  // Deterministic order: SRVs, then UAVs, then CBVs (matching the natural
  // read-mostly -> read-write -> constants layout), each sorted by
  // (space, register).
  std::vector<ResourceBinding *> Order;
  Order.reserve(R.Resources.size());
  for (ResourceBinding &RB : R.Resources)
    Order.push_back(&RB);
  std::stable_sort(Order.begin(), Order.end(),
                   [](const ResourceBinding *A, const ResourceBinding *B) {
                     auto Key = [](const ResourceBinding *RB) {
                       return std::tuple(static_cast<unsigned>(RB->Class),
                                         RB->Space, RB->LowerBound);
                     };
                     return Key(A) < Key(B);
                   });

  uint32_t NextBuffer = 0, NextTexture = 0, NextSampler = 0;
  for (ResourceBinding *RB : Order) {
    if (RB->RangeSize != 1)
      return makeError("resource '{0}': descriptor arrays (range size {1}) "
                       "are not supported yet",
                       RB->Name, RB->RangeSize);
    if (RB->Class == ResourceClass::Sampler)
      RB->MetalIndex = NextSampler++;
    else if (isTexture(*RB))
      RB->MetalIndex = NextTexture++;
    else if (isBufferLike(*RB))
      RB->MetalIndex = NextBuffer++;
    else
      return makeError("resource '{0}': unsupported resource kind {1}",
                       RB->Name, static_cast<unsigned>(RB->Kind));
  }
  return llvm::Error::success();
}

} // namespace crucible

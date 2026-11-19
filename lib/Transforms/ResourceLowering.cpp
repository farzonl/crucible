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
#include <map>
#include <optional>
#include <set>

namespace crucible {

static bool isBufferLike(const ResourceBinding &RB) {
  switch (RB.Kind) {
  case ResourceKind::TypedBuffer:
  case ResourceKind::RawBuffer:
  case ResourceKind::StructuredBuffer:
  case ResourceKind::CBuffer:
  case ResourceKind::TBuffer:
  case ResourceKind::RTAccelerationStructure:
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
    if (RB->RangeSize == 0xFFFFFFFFu) {
      for (const ArraySizeOverride &O : Opts.ArraySizeOverrides) {
        if (O.Class == RB->Class && O.Space == RB->Space &&
            O.Register == RB->LowerBound && O.Size > 0) {
          RB->RangeSize = O.Size;
          break;
        }
      }
    }
    if (RB->RangeSize == 0 || RB->RangeSize == 0xFFFFFFFFu)
      return makeError("resource '{0}': unbounded descriptor arrays with "
                       "dynamic indexing are not supported yet",
                       RB->Name);
    if (RB->Class == ResourceClass::Sampler) {
      RB->MetalIndex = NextSampler;
      NextSampler += RB->RangeSize;
    } else if (isTexture(*RB)) {
      RB->MetalIndex = NextTexture;
      NextTexture += RB->RangeSize;
    } else if (isBufferLike(*RB)) {
      RB->MetalIndex = NextBuffer;
      NextBuffer += RB->RangeSize;
      if (RB->HasCounter) {
        RB->CounterMetalIndex = NextBuffer;
        NextBuffer += RB->RangeSize;
      }
    } else {
      return makeError("resource '{0}': unsupported resource kind {1}",
                       RB->Name, static_cast<unsigned>(RB->Kind));
    }
  }
  bool NeedsBufferSizes = R.HasBufferDimensions;
  if (!NeedsBufferSizes && Opts.BoundsCheckBuffers) {
    for (const ResourceBinding &RB : R.Resources) {
      if (RB.Kind == ResourceKind::StructuredBuffer ||
          RB.Kind == ResourceKind::RawBuffer ||
          RB.Kind == ResourceKind::TypedBuffer) {
        NeedsBufferSizes = true;
        break;
      }
    }
  }
  if (NeedsBufferSizes)
    R.BufferSizesMetalIndex = NextBuffer++;
  if (R.HasInstanceContribution)
    R.InstanceContributionMetalIndex = NextBuffer++;
  return llvm::Error::success();
}

llvm::Error assignSPIRVBindings(Reflection &R, const Options &Opts) {
  std::vector<ResourceBinding *> Order;
  Order.reserve(R.Resources.size());
  for (ResourceBinding &RB : R.Resources)
    Order.push_back(&RB);
  std::stable_sort(Order.begin(), Order.end(),
                   [](const ResourceBinding *A, const ResourceBinding *B) {
                     auto Key = [](const ResourceBinding *RB) {
                       return std::tuple(RB->Space,
                                         static_cast<unsigned>(RB->Class),
                                         RB->LowerBound);
                     };
                     return Key(A) < Key(B);
                   });

  auto findShift = [&](ResourceClass RC, uint32_t Space) -> std::optional<uint32_t> {
    std::optional<uint32_t> Wildcard;
    for (const VulkanRegisterShift &S : Opts.VulkanRegisterShifts) {
      if (S.Class != RC)
        continue;
      if (!S.AllSpaces && S.Space == Space)
        return S.Shift;
      if (S.AllSpaces)
        Wildcard = S.Shift;
    }
    return Wildcard;
  };

  if (Opts.VulkanRegisterShifts.empty()) {
    uint32_t CurrentSpace = UINT32_MAX;
    uint32_t NextBinding = 0;
    for (ResourceBinding *RB : Order) {
      if (RB->RangeSize == 0xFFFFFFFFu) {
        for (const ArraySizeOverride &O : Opts.ArraySizeOverrides) {
          if (O.Class == RB->Class && O.Space == RB->Space &&
              O.Register == RB->LowerBound && O.Size > 0) {
            RB->RangeSize = O.Size;
            break;
          }
        }
      }
      if (RB->RangeSize == 0 || RB->RangeSize == 0xFFFFFFFFu)
        return makeError("resource '{0}': unbounded descriptor arrays with "
                         "dynamic indexing are not supported yet",
                         RB->Name);
      if (RB->Space != CurrentSpace) {
        CurrentSpace = RB->Space;
        NextBinding = 0;
      }
      RB->SPIRVSet = RB->Space;
      RB->SPIRVBinding = NextBinding;
      NextBinding += RB->RangeSize;
      if (RB->HasCounter) {
        RB->CounterSPIRVBinding = NextBinding;
        NextBinding += RB->RangeSize;
      }
    }
    return llvm::Error::success();
  }

  // Shift-override mode: apply explicit shifts first, then place unshifted
  // resources and counters in the lowest non-colliding binding per set.
  std::map<uint32_t, std::set<uint32_t>> UsedBindings;
  auto reserveRange = [&](uint32_t Set, uint32_t Start, uint32_t Count,
                          const std::string &Name) -> llvm::Error {
    auto &Used = UsedBindings[Set];
    for (uint32_t I = 0; I < Count; ++I) {
      if (!Used.insert(Start + I).second)
        return makeError("SPIR-V binding collision at set {0}, binding {1} "
                         "for resource '{2}'",
                         Set, Start + I, Name);
    }
    return llvm::Error::success();
  };

  std::vector<ResourceBinding *> Unshifted;
  for (ResourceBinding *RB : Order) {
    if (RB->RangeSize == 0xFFFFFFFFu) {
      for (const ArraySizeOverride &O : Opts.ArraySizeOverrides) {
        if (O.Class == RB->Class && O.Space == RB->Space &&
            O.Register == RB->LowerBound && O.Size > 0) {
          RB->RangeSize = O.Size;
          break;
        }
      }
    }
    if (RB->RangeSize == 0 || RB->RangeSize == 0xFFFFFFFFu)
      return makeError("resource '{0}': unbounded descriptor arrays with "
                       "dynamic indexing are not supported yet",
                       RB->Name);
    RB->SPIRVSet = RB->Space;
    if (auto Shift = findShift(RB->Class, RB->Space)) {
      RB->SPIRVBinding = RB->LowerBound + *Shift;
      if (llvm::Error E = reserveRange(RB->SPIRVSet, RB->SPIRVBinding,
                                       RB->RangeSize, RB->Name))
        return E;
    } else {
      Unshifted.push_back(RB);
    }
  }

  auto allocFreeRange = [&](uint32_t Set, uint32_t Count) -> uint32_t {
    auto &Used = UsedBindings[Set];
    uint32_t Cand = 0;
    while (true) {
      bool Free = true;
      for (uint32_t I = 0; I < Count; ++I) {
        if (Used.count(Cand + I)) {
          Free = false;
          Cand = Cand + I + 1;
          break;
        }
      }
      if (Free) {
        for (uint32_t I = 0; I < Count; ++I)
          Used.insert(Cand + I);
        return Cand;
      }
    }
  };

  for (ResourceBinding *RB : Unshifted)
    RB->SPIRVBinding = allocFreeRange(RB->SPIRVSet, RB->RangeSize);

  for (ResourceBinding *RB : Order) {
    if (RB->HasCounter)
      RB->CounterSPIRVBinding = allocFreeRange(RB->SPIRVSet, RB->RangeSize);
  }
  return llvm::Error::success();
}

} // namespace crucible

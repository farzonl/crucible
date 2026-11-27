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
  using Overrides = std::map<std::pair<uint32_t, uint32_t>, uint32_t>;
  Overrides BufferOverrides, TextureOverrides, SamplerOverrides;
  auto Collect = [&](const std::vector<MetalResourceBinding> &Bindings,
                     Overrides &Map, const char *Kind, uint32_t Limit) -> llvm::Error {
    for (const auto &B : Bindings) {
      if (B.Index >= Limit)
        return makeError("Metal {0} ABI index {1} exceeds the {0} slot limit",
                         Kind, B.Index);
      if (!Map.emplace(std::make_pair(B.Space, B.Register), B.Index).second)
        return makeError("duplicate Metal {0} ABI binding ({1}, {2})",
                         Kind, B.Space, B.Register);
    }
    return llvm::Error::success();
  };
  if (auto E = Collect(Opts.MetalBufferBindings, BufferOverrides, "buffer", 31))
    return E;
  if (auto E = Collect(Opts.MetalTextureBindings, TextureOverrides, "texture", 128))
    return E;
  if (auto E = Collect(Opts.MetalSamplerBindings, SamplerOverrides, "sampler", 16))
    return E;
  bool ExplicitABI = Opts.RequireMetalBufferBindings || !BufferOverrides.empty() ||
                     !TextureOverrides.empty() || !SamplerOverrides.empty();
  std::set<uint32_t> BufferSlots, TextureSlots, SamplerSlots;
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
    bool Sampler = RB->Class == ResourceClass::Sampler;
    bool Texture = isTexture(*RB);
    auto &Map = Sampler ? SamplerOverrides : Texture ? TextureOverrides : BufferOverrides;
    auto &Slots = Sampler ? SamplerSlots : Texture ? TextureSlots : BufferSlots;
    auto &Next = Sampler ? NextSampler : Texture ? NextTexture : NextBuffer;
    const char *Kind = Sampler ? "sampler" : Texture ? "texture" : "buffer";
    auto Override = Map.find({RB->Space, RB->LowerBound});
    if (Override != Map.end()) {
      if ((!Sampler && !Texture && !isBufferLike(*RB)) ||
          RB->RangeSize != 1 || RB->HasCounter)
        return makeError("resource '{0}': explicit Metal ABI supports only "
                         "single resources without counters", RB->Name);
      if (!Slots.insert(Override->second).second)
        return makeError("Metal {0} ABI slot {1} is used by multiple resources",
                         Kind, Override->second);
      RB->MetalIndex = Override->second;
      Next = std::max(Next, RB->MetalIndex + 1);
      continue;
    }
    if (ExplicitABI)
      return makeError("resource '{0}': missing explicit Metal {1} ABI "
                       "binding ({2}, {3})",
                       RB->Name, Kind, RB->Space, RB->LowerBound);
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
  if ((NeedsBufferSizes || R.HasInstanceContribution) &&
      ExplicitABI)
    return makeError("explicit Metal buffer ABI does not support Crucible's "
                     "auxiliary buffers");
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

  // Remap huge/sparse HLSL register spaces (e.g. space4294967279) into compact
  // dense Vulkan DescriptorSet indices so LLVM SPIR-V's binding legalizer and
  // Vulkan vkCreatePipelineLayout don't OOM allocating 2^32 set slots.
  // Keep small spaces (< 64) as identity to preserve explicit space0..N layouts.
  bool HasLargeSpace = false;
  for (const ResourceBinding *RB : Order) {
    if (RB->Space >= 64u) {
      HasLargeSpace = true;
      break;
    }
  }
  std::map<uint32_t, uint32_t> SpaceToSet;
  if (HasLargeSpace) {
    uint32_t NextSet = 0;
    for (const ResourceBinding *RB : Order) {
      if (SpaceToSet.find(RB->Space) == SpaceToSet.end())
        SpaceToSet[RB->Space] = NextSet++;
    }
  }
  auto mapSpaceToSet = [&](uint32_t Space) -> uint32_t {
    return HasLargeSpace ? SpaceToSet[Space] : Space;
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
      if (RB->RangeSize == 0)
        return makeError("resource '{0}': zero-sized descriptor array is invalid",
                         RB->Name);
      if (RB->Space != CurrentSpace) {
        CurrentSpace = RB->Space;
        NextBinding = 0;
      }
      RB->SPIRVSet = mapSpaceToSet(RB->Space);
      RB->SPIRVBinding = NextBinding;
      uint32_t SlotStep = (RB->RangeSize == 0xFFFFFFFFu) ? 1u : RB->RangeSize;
      NextBinding += SlotStep;
      if (RB->HasCounter) {
        RB->CounterSPIRVBinding = NextBinding;
        NextBinding += SlotStep;
      }
    }
    uint32_t MaxSet = 0;
    for (const ResourceBinding &RB : R.Resources)
      MaxSet = std::max(MaxSet, RB.SPIRVSet + 1u);
    if (R.HasResourceHeap) {
      R.ResourceHeapSPIRVSet = MaxSet++;
      R.ResourceHeapSPIRVBinding = 0;
    }
    if (R.HasSamplerHeap) {
      R.SamplerHeapSPIRVSet = MaxSet++;
      R.SamplerHeapSPIRVBinding = 0;
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
    if (RB->RangeSize == 0)
      return makeError("resource '{0}': zero-sized descriptor array is invalid",
                       RB->Name);
    RB->SPIRVSet = mapSpaceToSet(RB->Space);
    uint32_t SlotStep = (RB->RangeSize == 0xFFFFFFFFu) ? 1u : RB->RangeSize;
    if (auto Shift = findShift(RB->Class, RB->Space)) {
      RB->SPIRVBinding = RB->LowerBound + *Shift;
      if (llvm::Error E =
              reserveRange(RB->SPIRVSet, RB->SPIRVBinding, SlotStep, RB->Name))
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

  for (ResourceBinding *RB : Unshifted) {
    uint32_t SlotStep = (RB->RangeSize == 0xFFFFFFFFu) ? 1u : RB->RangeSize;
    RB->SPIRVBinding = allocFreeRange(RB->SPIRVSet, SlotStep);
  }

  for (ResourceBinding *RB : Order) {
    if (RB->HasCounter) {
      uint32_t SlotStep = (RB->RangeSize == 0xFFFFFFFFu) ? 1u : RB->RangeSize;
      RB->CounterSPIRVBinding = allocFreeRange(RB->SPIRVSet, SlotStep);
    }
  }

  uint32_t MaxSet = 0;
  for (const ResourceBinding &RB : R.Resources)
    MaxSet = std::max(MaxSet, RB.SPIRVSet + 1u);
  if (R.HasResourceHeap) {
    R.ResourceHeapSPIRVSet = MaxSet++;
    R.ResourceHeapSPIRVBinding = 0;
  }
  if (R.HasSamplerHeap) {
    R.SamplerHeapSPIRVSet = MaxSet++;
    R.SamplerHeapSPIRVBinding = 0;
  }
  return llvm::Error::success();
}

llvm::Error assignDXILBindings(Reflection &R, const Options &Opts) {
  // Remap Vulkan PushConstant sentinel space (0xFFFFFFFEu, which is reserved
  // in D3D12 root signatures) to a valid unused register space.
  uint32_t MaxNormalSpace = 0;
  for (const ResourceBinding &RB : R.Resources) {
    if (!RB.IsPushConstant && RB.Space < 0xFFFFFFF0u)
      MaxNormalSpace = std::max(MaxNormalSpace, RB.Space + 1u);
  }
  for (ResourceBinding &RB : R.Resources) {
    if (RB.IsPushConstant || RB.Space == 0xFFFFFFFEu) {
      RB.IsPushConstant = true;
      RB.Space = MaxNormalSpace;
      RB.LowerBound = 0;
    }
  }

  // Preserve explicit (Class, Space, LowerBound) register assignments when
  // they do not collide (e.g. SPIR-V DescriptorSet -> space, Binding ->
  // register within each class). Resolve unbounded array overrides if given.
  std::map<std::pair<ResourceClass, uint32_t>, std::set<uint32_t>> UsedRegs;
  std::map<std::pair<ResourceClass, uint32_t>, uint32_t> UnboundedStart;
  for (ResourceBinding &RB : R.Resources) {
    if (RB.RangeSize == 0xFFFFFFFFu) {
      for (const ArraySizeOverride &O : Opts.ArraySizeOverrides) {
        if (O.Class == RB.Class && O.Space == RB.Space &&
            O.Register == RB.LowerBound && O.Size > 0) {
          RB.RangeSize = O.Size;
          break;
        }
      }
    }
    if (RB.RangeSize == 0)
      return makeError("resource '{0}': zero-sized descriptor array is invalid",
                       RB.Name);
    auto Key = std::make_pair(RB.Class, RB.Space);
    bool IsUnbounded = (RB.RangeSize == 0xFFFFFFFFu);

    // Unbounded ranges [LowerBound, inf) cannot share a (Class, Space) with
    // another unbounded range or any register >= LowerBound; move colliding
    // unbounded resources to a fresh register space.
    auto hasUnboundedOverlap = [&](const std::pair<ResourceClass, uint32_t> &K,
                                   uint32_t Start, bool Unb) {
      auto It = UnboundedStart.find(K);
      if (It != UnboundedStart.end()) {
        if (Unb || Start >= It->second)
          return true;
      }
      if (Unb) {
        auto UIt = UsedRegs.find(K);
        if (UIt != UsedRegs.end() && !UIt->second.empty() &&
            *UIt->second.rbegin() >= Start)
          return true;
      }
      return false;
    };

    if (hasUnboundedOverlap(Key, RB.LowerBound, IsUnbounded)) {
      uint32_t NewSpace = RB.Space + 1u;
      while (UsedRegs.count({RB.Class, NewSpace}) ||
             UnboundedStart.count({RB.Class, NewSpace}))
        ++NewSpace;
      RB.Space = NewSpace;
      RB.LowerBound = 0;
      Key = std::make_pair(RB.Class, RB.Space);
    }

    auto &Used = UsedRegs[Key];
    if (IsUnbounded) {
      UnboundedStart[Key] = RB.LowerBound;
      Used.insert(RB.LowerBound);
      continue;
    }
    uint32_t Count = RB.RangeSize;
    bool Collides = false;
    for (uint32_t I = 0; I < Count; ++I) {
      if (Used.count(RB.LowerBound + I) ||
          (UnboundedStart.count(Key) && RB.LowerBound + I >= UnboundedStart[Key])) {
        Collides = true;
        break;
      }
    }
    if (Collides) {
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
          RB.LowerBound = Cand;
          break;
        }
      }
    }
    for (uint32_t I = 0; I < Count; ++I)
      Used.insert(RB.LowerBound + I);
  }
  return llvm::Error::success();
}

} // namespace crucible

//===- Reflection.h - Resource binding reflection ---------------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_REFLECTION_H
#define CRUCIBLE_REFLECTION_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace crucible {

enum class ShaderStage : uint8_t {
  Pixel = 0,
  Vertex = 1,
  Geometry = 2,
  Hull = 3,
  Domain = 4,
  Compute = 5,
  Library = 6,
  Mesh = 13,
  Amplification = 14,
  Unknown = 255,
};

enum class ResourceClass : uint8_t { SRV = 0, UAV = 1, CBV = 2, Sampler = 3 };

/// DXIL resource kinds (matches DXC's hlsl::DXIL::ResourceKind).
enum class ResourceKind : uint8_t {
  Invalid = 0,
  Texture1D = 1,
  Texture2D = 2,
  Texture2DMS = 3,
  Texture3D = 4,
  TextureCube = 5,
  Texture1DArray = 6,
  Texture2DArray = 7,
  Texture2DMSArray = 8,
  TextureCubeArray = 9,
  TypedBuffer = 10,
  RawBuffer = 11,
  StructuredBuffer = 12,
  CBuffer = 13,
  Sampler = 14,
  TBuffer = 15,
};

/// One DXIL register range mapped to a Metal binding slot.
struct ResourceBinding {
  ResourceClass Class = ResourceClass::SRV;
  ResourceKind Kind = ResourceKind::Invalid;
  uint32_t RecordID = 0; ///< DXIL range ID (unique per class).
  std::string Name;      ///< HLSL-visible name, may be empty.
  uint32_t Space = 0;
  uint32_t LowerBound = 0;
  uint32_t RangeSize = 1;
  uint32_t ElementStride = 0; ///< Structured buffer stride; CBV size in bytes.
  bool GloballyCoherent = false;
  bool HasCounter = false;

  /// Assigned Metal slot ([[buffer(n)]] / [[texture(n)]] / [[sampler(n)]]).
  uint32_t MetalIndex = 0;
};

/// Full reflection for one compiled entry point.
struct Reflection {
  ShaderStage Stage = ShaderStage::Unknown;
  uint32_t ShaderModelMajor = 0;
  uint32_t ShaderModelMinor = 0;
  std::string EntryName;
  /// Name of the emitted MSL kernel. Differs from EntryName when the HLSL
  /// name is reserved in MSL (e.g. "main").
  std::string MSLEntryName;
  std::array<uint32_t, 3> NumThreads = {1, 1, 1};
  std::vector<ResourceBinding> Resources;

  /// Serialize the (space, register, kind) -> metal_index mapping as JSON.
  std::string toJSON() const;
};

} // namespace crucible

#endif // CRUCIBLE_REFLECTION_H

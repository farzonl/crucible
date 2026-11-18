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

/// DXIL element component type (matches hlsl::DXIL::ComponentType).
enum class ComponentType : uint8_t {
  Invalid = 0,
  I1 = 1,
  I16 = 2,
  U16 = 3,
  I32 = 4,
  U32 = 5,
  I64 = 6,
  U64 = 7,
  F16 = 8,
  F32 = 9,
  F64 = 10,
  SNormF16 = 11,
  UNormF16 = 12,
  SNormF32 = 13,
  UNormF32 = 14,
  SNormF64 = 15,
  UNormF64 = 16,
};

/// One DXIL register range mapped to a Metal binding slot.
struct ResourceBinding {
  ResourceClass Class = ResourceClass::SRV;
  ResourceKind Kind = ResourceKind::Invalid;
  ComponentType ElementType = ComponentType::Invalid;
  uint32_t RecordID = 0; ///< DXIL range ID (unique per class).
  std::string Name;      ///< HLSL-visible name, may be empty.
  uint32_t Space = 0;
  uint32_t LowerBound = 0;
  uint32_t RangeSize = 1;
  uint32_t ElementStride = 0; ///< Structured buffer stride; CBV size in bytes.
  uint32_t TypedComponents = 0; ///< Number of scalar components in a typed buffer.
  bool GloballyCoherent = false;
  bool HasCounter = false;
  bool RequiresTextureAtomics = false; ///< Requires MTLTextureUsageShaderAtomic.

  /// Assigned Metal slot ([[buffer(n)]] / [[texture(n)]] / [[sampler(n)]]).
  uint32_t MetalIndex = 0;
};

struct StageParameter {
  std::string Semantic;
  uint32_t SemanticIndex;
  uint32_t ID;
  ComponentType Type;
  uint32_t Kind; ///< DXIL semantic kind.
  uint32_t Interpolation; ///< DXIL interpolation mode.
  uint32_t Rows;
  uint32_t Columns;
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
  std::vector<StageParameter> Inputs, Outputs;

  /// Serialize the (space, register, kind) -> metal_index mapping as JSON.
  std::string toJSON() const;
};

} // namespace crucible

#endif // CRUCIBLE_REFLECTION_H

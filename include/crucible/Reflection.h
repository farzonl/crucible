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
#include <optional>
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
  RayGeneration = 7,
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
  RTAccelerationStructure = 16,
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
  uint32_t TypedComponents = 0; ///< Number of DXIL storage components per element.
  bool GloballyCoherent = false;
  bool HasCounter = false;
  bool RequiresTextureAtomics = false; ///< Requires MTLTextureUsageShaderAtomic.

  /// Assigned Metal slot ([[buffer(n)]] / [[texture(n)]] / [[sampler(n)]]).
  uint32_t MetalIndex = 0;
  /// Assigned Metal [[buffer(n)]] slot for UAV atomic counter (when HasCounter).
  uint32_t CounterMetalIndex = 0;
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
  std::vector<uint32_t> RowSemanticIndices;
};

enum class TessellationDomain : uint8_t { Isoline = 1, Triangle = 2, Quad = 3 };

enum class MeshOutputTopology : uint8_t { Line = 1, Triangle = 2 };

struct MeshInterface {
  MeshOutputTopology Topology = MeshOutputTopology::Triangle;
  uint32_t MaxVertexCount = 0;
  uint32_t MaxPrimitiveCount = 0;
  uint32_t PayloadSize = 0;
};

struct AmplificationInterface {
  uint32_t PayloadSize = 0;
};

enum class TessellationPartitioning : uint8_t {
  Integer = 1,
  Pow2 = 2,
  FractionalOdd = 3,
  FractionalEven = 4,
};

enum class TessellationOutputPrimitive : uint8_t {
  Point = 1,
  Line = 2,
  TriangleCW = 3,
  TriangleCCW = 4,
};

struct HullInterface {
  TessellationDomain Domain = TessellationDomain::Triangle;
  TessellationPartitioning Partitioning = TessellationPartitioning::Integer;
  TessellationOutputPrimitive OutputPrimitive =
      TessellationOutputPrimitive::TriangleCW;
  uint32_t InputControlPointCount = 3;
  uint32_t OutputControlPointCount = 3;
  float MaxTessFactor = 64.0f;
  std::string PatchConstantFuncName;
  std::vector<StageParameter> PatchConstants;
  static constexpr uint32_t InputDataMetalIndex = 28;
  static constexpr uint32_t TessFactorMetalIndex = 29;
  static constexpr uint32_t PatchDataMetalIndex = 30;
};

struct GeometryInterface {
  uint32_t InputPrimitive = 3;   // 1=point, 2=line, 3=triangle
  uint32_t MaxVertexCount = 3;
  uint32_t InstanceCount = 1;
  uint32_t OutputTopology = 5;   // 1=point, 3=linestrip, 5=trianglestrip
  uint32_t StreamMask = 1;
  static constexpr uint32_t InputDataMetalIndex = 28;
  static constexpr uint32_t CounterMetalIndex = 29;
  static constexpr uint32_t OutputDataMetalIndex = 30;
};

struct DomainInterface {
  TessellationDomain Domain;
  uint32_t ControlPointCount;
  std::vector<StageParameter> PatchConstants;
  // Each patch contains its control points (Inputs order), then its patch
  // constants. Each signature element is packed row-major as 32-bit words.
  static constexpr uint32_t DataMetalIndex = 30;
};

struct RTShaderInfo {
  std::string Name;       ///< Unmangled HLSL name.
  std::string SymbolName; ///< DXIL function symbol name.
  uint32_t Kind = 0;      ///< DXIL ShaderKind (10=ClosestHit, 11=Miss).
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
  std::optional<DomainInterface> Domain;
  std::optional<HullInterface> Hull;
  std::optional<GeometryInterface> Geometry;
  std::optional<MeshInterface> Mesh;
  std::optional<AmplificationInterface> Amplification;
  bool HasBufferDimensions = false;
  uint32_t BufferSizesMetalIndex = UINT32_MAX;
  bool HasInstanceContribution = false;
  uint32_t InstanceContributionMetalIndex = UINT32_MAX;
  bool HasResourceHeap = false;
  uint32_t ResourceHeapMetalIndex = UINT32_MAX;
  bool HasCounterHeap = false;
  uint32_t CounterHeapMetalIndex = UINT32_MAX;
  bool HasSamplerHeap = false;
  uint32_t SamplerHeapMetalIndex = UINT32_MAX;
  std::vector<RTShaderInfo> RTShaders;

  /// Serialize the (space, register, kind) -> metal_index mapping as JSON.
  std::string toJSON() const;
};

} // namespace crucible

#endif // CRUCIBLE_REFLECTION_H

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
#include <unordered_map>
#include <vector>

namespace crucible {

/// Output backend target.
enum class TargetBackend : uint8_t {
  /// Metal Shading Language (MSL) source via MLIR SCF structurization.
  MSL,
  /// Vulkan SPIR-V binary or assembly via LLVM's SPIR-V backend.
  SPIRV,
};

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

struct MetalResourceBinding {
  uint32_t Space = 0;
  uint32_t Register = 0;
  uint32_t Index = 0;
};
using MetalBufferBinding = MetalResourceBinding;

/// DXC-style Vulkan register shift override (-fvk-b/t/u/s-shift).
struct VulkanRegisterShift {
  ResourceClass Class = ResourceClass::CBV;
  uint32_t Shift = 0;
  uint32_t Space = 0;
  /// When true, applies to all descriptor spaces unless overridden by an
  /// exact-space entry.
  bool AllSpaces = true;
};

struct Options {
  TargetBackend Target = TargetBackend::MSL;
  BindingMode Binding = BindingMode::Direct;

  /// MSL language version passed to the Apple toolchain (e.g. "metal3.1").
  std::string MSLStd = "metal3.1";

  /// When targeting SPIR-V, emit human-readable SPIR-V assembly text instead
  /// of binary SPIR-V words.
  bool EmitSPIRVAsm = false;

  /// Vulkan target environment for the LLVM SPIR-V backend (e.g. "vulkan1.3").
  std::string VulkanVersion = "vulkan1.3";

  /// Optional DXC-style Vulkan register shifts (--vk-b/t/u/s-shift).
  std::vector<VulkanRegisterShift> VulkanRegisterShifts;

  /// Entry point to compile when the container exports multiple (libraries).
  /// Empty selects the sole entry point.
  std::string EntryPoint;

  /// Explicit Metal ABI used by external Vulkan runtimes.
  std::string MSLEntryPoint;
  std::vector<MetalBufferBinding> MetalBufferBindings;
  std::vector<MetalResourceBinding> MetalTextureBindings;
  std::vector<MetalResourceBinding> MetalSamplerBindings;
  bool RequireMetalBufferBindings = false;

  /// Flush f32 denormals to signed zero around math builtins (D3D12 FTZ).
  bool FlushDenormsF32 = false;

  /// Bounds-check StructuredBuffer/RawBuffer/TypedBuffer loads and stores
  /// using the crucible_buf_dims buffer (D3D12 out-of-bounds contract).
  bool BoundsCheckBuffers = false;

  /// Emulate 64-bit floating-point (double) operations using software
  /// IEEE-754 binary64 arithmetic on ulong.
  bool EmulateF64 = false;

  /// Print the structurized MLIR entry function to stderr before MSL emission.
  bool DumpMLIR = false;

  /// Explicit array sizes for unbounded DXIL resource arrays (rangeSize == -1).
  std::vector<ArraySizeOverride> ArraySizeOverrides;

  /// Ordered SBT Miss shader names and HitGroup ClosestHit shader names
  /// for lib_6_x raytracing pipelines.
  std::vector<std::string> MissShaders;
  std::vector<std::string> HitGroupClosestHits;

  /// SPIR-V specialization constant overrides (ConstantID -> raw bit pattern).
  std::unordered_map<uint32_t, uint64_t> SpecConstants;

  /// Compile a Vertex or Domain shader as a compute kernel that writes its
  /// packed 32-bit signature outputs to [[buffer(30)]] for downstream HS/GS.
  bool StageAsCompute = false;

  /// Vertex attribute layout used when compiling a Vertex shader with
  /// StageAsCompute = true (reads raw vertex buffer from [[buffer(28)]]).
  struct VertexInputAttribute {
    std::string Name;
    uint32_t Offset = 0;
    uint32_t Channels = 0;
    enum class ElementType : uint8_t { Float32, Int32, UInt32 } Type = ElementType::Float32;
  };
  uint32_t VertexInputStride = 0;
  std::vector<VertexInputAttribute> VertexInputLayout;

  /// Optional tessellation partitioning (1=Integer, 2=Pow2, 3=FractionalOdd,
  /// 4=FractionalEven) and output primitive (1=Point, 2=Line, 3=TriangleCW,
  /// 4=TriangleCCW) forwarded from Hull to Domain shader compilation for SPIR-V.
  uint8_t TessPartitioning = 0;
  uint8_t TessOutputPrimitive = 0;
};

} // namespace crucible

#endif // CRUCIBLE_OPTIONS_H

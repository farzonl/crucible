//===- CrucibleOps.td - Crucible MLIR dialect operations ----*- tablegen -*-===//
//
// Part of Crucible, an open-source multi-frontend, multi-backend shader compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_OPS_TD
#define CRUCIBLE_OPS_TD

include "crucible/Dialect/CrucibleDialect.td"
include "crucible/Dialect/CrucibleEnums.td"
include "crucible/Dialect/CrucibleTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Crucible_Op<string mnemonic, list<Trait> traits = []>
    : Op<Crucible_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Resource Binding Operations
//===----------------------------------------------------------------------===//

def Crucible_BindResourceOp : Crucible_Op<"bind_resource", [Pure]> {
  let summary = "Bind a shader resource by reflection record ID or space/binding";
  let arguments = (ins
    Crucible_ResourceClassAttr:$resource_class,
    OptionalAttr<I32Attr>:$record_id,
    OptionalAttr<I32Attr>:$space,
    OptionalAttr<I32Attr>:$binding,
    DefaultValuedAttr<BoolAttr, "false">:$non_uniform,
    Optional<I32>:$index
  );
  let results = (outs Crucible_ResourceType:$result);
}

def Crucible_HeapResourceOp : Crucible_Op<"heap_resource", [Pure]> {
  let summary = "Fetch a bindless resource handle from the descriptor or sampler heap";
  let arguments = (ins
    I32:$index,
    BoolAttr:$is_sampler_heap,
    DefaultValuedAttr<BoolAttr, "false">:$non_uniform,
    OptionalAttr<Crucible_ResourceClassAttr>:$resource_class,
    OptionalAttr<I32Attr>:$resource_kind,
    OptionalAttr<I32Attr>:$element_stride,
    DefaultValuedAttr<BoolAttr, "false">:$has_counter
  );
  let results = (outs Crucible_ResourceType:$result);
}

//===----------------------------------------------------------------------===//
// Buffer & Constant Buffer Operations
//===----------------------------------------------------------------------===//

def Crucible_CBufferLoadOp : Crucible_Op<"cbuffer_load", [Pure]> {
  let summary = "Load a 16-byte register from a constant buffer";
  let arguments = (ins
    Crucible_ResourceType:$handle,
    I32:$reg_index
  );
  let results = (outs AnyType:$result);
}

def Crucible_BufferLoadOp : Crucible_Op<"buffer_load"> {
  let summary = "Load from a typed, raw, or structured shader buffer";
  let arguments = (ins
    Crucible_ResourceType:$handle,
    Crucible_BufferKindAttr:$kind,
    I32:$index,
    Optional<I32>:$element_offset,
    OptionalAttr<I32Attr>:$mask,
    OptionalAttr<I32Attr>:$alignment
  );
  let results = (outs
    AnyType:$result,
    Optional<I32>:$status
  );
}

def Crucible_BufferStoreOp : Crucible_Op<"buffer_store", [
    AttrSizedOperandSegments
]> {
  let summary = "Store to a typed, raw, or structured shader buffer";
  let arguments = (ins
    Crucible_ResourceType:$handle,
    Crucible_BufferKindAttr:$kind,
    I32:$index,
    Optional<I32>:$element_offset,
    Variadic<AnyType>:$values,
    I32Attr:$mask,
    OptionalAttr<I32Attr>:$alignment
  );
}

def Crucible_BufferAtomicOp : Crucible_Op<"buffer_atomic"> {
  let summary = "Atomic binary read-modify-write on a UAV buffer or texture";
  let arguments = (ins
    Crucible_ResourceType:$handle,
    Crucible_AtomicOpAttr:$op,
    Variadic<I32>:$coords,
    AnyType:$value
  );
  let results = (outs AnyType:$result);
}

def Crucible_BufferAtomicCmpXchgOp : Crucible_Op<"buffer_atomic_cmpxchg"> {
  let summary = "Atomic compare-exchange on a UAV buffer or texture";
  let arguments = (ins
    Crucible_ResourceType:$handle,
    Variadic<I32>:$coords,
    AnyType:$compare,
    AnyType:$value
  );
  let results = (outs AnyType:$result);
}

def Crucible_BufferUpdateCounterOp : Crucible_Op<"buffer_update_counter"> {
  let summary = "Increment or decrement a UAV structured buffer counter";
  let arguments = (ins
    Crucible_ResourceType:$handle,
    I8Attr:$inc
  );
  let results = (outs I32:$result);
}

//===----------------------------------------------------------------------===//
// Texture & Sampling Operations
//===----------------------------------------------------------------------===//

def Crucible_TexLoadOp : Crucible_Op<"tex_load", [
    AttrSizedOperandSegments
]> {
  let summary = "Load a texel from a texture resource";
  let arguments = (ins
    Crucible_ResourceType:$handle,
    Optional<I32>:$mip_or_sample,
    Variadic<I32>:$coords,
    Variadic<I32>:$offsets
  );
  let results = (outs
    AnyType:$result,
    Optional<I32>:$status
  );
}

def Crucible_TexStoreOp : Crucible_Op<"tex_store", [
    AttrSizedOperandSegments
]> {
  let summary = "Store a texel to a UAV texture resource";
  let arguments = (ins
    Crucible_ResourceType:$handle,
    Variadic<I32>:$coords,
    Variadic<AnyType>:$values,
    I32Attr:$mask
  );
}

def Crucible_TexSampleOp : Crucible_Op<"tex_sample", [
    AttrSizedOperandSegments
]> {
  let summary = "Sample a texture with a sampler";
  let arguments = (ins
    Crucible_ResourceType:$tex,
    Crucible_ResourceType:$sampler,
    Crucible_SampleModeAttr:$mode,
    Variadic<F32>:$coords,
    Variadic<I32>:$offsets,
    Optional<F32>:$lod_or_bias,
    Variadic<F32>:$ddx,
    Variadic<F32>:$ddy,
    Optional<F32>:$cmp_val,
    Optional<F32>:$clamp
  );
  let results = (outs
    AnyType:$result,
    Optional<I32>:$status
  );
}

def Crucible_TexGatherOp : Crucible_Op<"tex_gather", [
    AttrSizedOperandSegments
]> {
  let summary = "Gather four texels from a texture";
  let arguments = (ins
    Crucible_ResourceType:$tex,
    Crucible_ResourceType:$sampler,
    Variadic<F32>:$coords,
    Variadic<I32>:$offsets,
    Optional<F32>:$cmp_val,
    I32Attr:$channel
  );
  let results = (outs
    AnyType:$result,
    Optional<I32>:$status
  );
}

def Crucible_TexGetDimensionsOp : Crucible_Op<"tex_get_dimensions", [Pure]> {
  let summary = "Query dimensions of a texture or buffer resource";
  let arguments = (ins
    Crucible_ResourceType:$handle,
    Optional<I32>:$mip_level
  );
  let results = (outs AnyType:$result);
}

def Crucible_TexCalcLodOp : Crucible_Op<"tex_calc_lod", [Pure]> {
  let summary = "Calculate texture sampling level of detail";
  let arguments = (ins
    Crucible_ResourceType:$tex,
    Crucible_ResourceType:$sampler,
    Variadic<F32>:$coords,
    BoolAttr:$clamped
  );
  let results = (outs F32:$result);
}

//===----------------------------------------------------------------------===//
// Stage I/O, Pixel, Tessellation & Geometry Operations
//===----------------------------------------------------------------------===//

def Crucible_LoadInputOp : Crucible_Op<"load_input", [Pure]> {
  let summary = "Load a shader stage input element";
  let arguments = (ins
    I32Attr:$input_id,
    I32:$row,
    I8Attr:$col,
    Optional<I32>:$vertex_id
  );
  let results = (outs AnyType:$result);
}

def Crucible_StoreOutputOp : Crucible_Op<"store_output"> {
  let summary = "Store a shader stage output element";
  let arguments = (ins
    I32Attr:$output_id,
    I32:$row,
    I8Attr:$col,
    AnyType:$value
  );
}

def Crucible_DiscardOp : Crucible_Op<"discard"> {
  let summary = "Conditionally discard the current fragment";
  let arguments = (ins I1:$cond);
}

def Crucible_DerivOp : Crucible_Op<"deriv", [Pure]> {
  let summary = "Screen-space partial derivative (ddx/ddy coarse/fine)";
  let arguments = (ins
    AnyType:$value,
    Crucible_DerivAxisAttr:$axis,
    Crucible_DerivQualityAttr:$quality
  );
  let results = (outs AnyType:$result);
}

def Crucible_SysValOp : Crucible_Op<"sysval", [Pure]> {
  let summary = "Read a shader built-in system value";
  let arguments = (ins
    Crucible_SysValKindAttr:$kind,
    OptionalAttr<I32Attr>:$component
  );
  let results = (outs AnyType:$result);
}

def Crucible_LoadControlPointOp : Crucible_Op<"load_control_point", [Pure]> {
  let summary = "Load a hull/domain shader control point element";
  let arguments = (ins
    I32Attr:$input_id,
    I32:$row,
    I8Attr:$col,
    I32:$cp_id
  );
  let results = (outs AnyType:$result);
}

def Crucible_LoadPatchConstantOp : Crucible_Op<"load_patch_constant", [Pure]> {
  let summary = "Load a tessellation patch constant element";
  let arguments = (ins
    I32Attr:$input_id,
    I32:$row,
    I8Attr:$col
  );
  let results = (outs AnyType:$result);
}

def Crucible_StorePatchConstantOp : Crucible_Op<"store_patch_constant"> {
  let summary = "Store a tessellation patch constant element";
  let arguments = (ins
    I32Attr:$output_id,
    I32:$row,
    I8Attr:$col,
    AnyType:$value
  );
}

def Crucible_EmitStreamOp : Crucible_Op<"emit_stream"> {
  let summary = "Emit a vertex to a geometry shader stream";
  let arguments = (ins I8Attr:$stream_id);
}

def Crucible_CutStreamOp : Crucible_Op<"cut_stream"> {
  let summary = "End the current primitive on a geometry shader stream";
  let arguments = (ins I8Attr:$stream_id);
}

//===----------------------------------------------------------------------===//
// Barrier, Wave & Quad Operations
//===----------------------------------------------------------------------===//

def Crucible_BarrierOp : Crucible_Op<"barrier"> {
  let summary = "Shader memory and/or execution barrier";
  let arguments = (ins
    BoolAttr:$exec_sync,
    BoolAttr:$tgsm_fence,
    BoolAttr:$device_fence,
    Optional<Crucible_ResourceType>:$handle
  );
}

def Crucible_WaveIsFirstLaneOp : Crucible_Op<"wave_is_first_lane", [Pure]> {
  let summary = "True for the lowest active lane in the subgroup";
  let results = (outs I1:$result);
}

def Crucible_WaveVoteOp : Crucible_Op<"wave_vote", [Pure]> {
  let summary = "Subgroup vote (any, all, or all_equal)";
  let arguments = (ins
    AnyType:$value,
    Crucible_WaveVoteKindAttr:$kind
  );
  let results = (outs I1:$result);
}

def Crucible_WaveBallotOp : Crucible_Op<"wave_ballot", [Pure]> {
  let summary = "Subgroup ballot returning a 4xi32 bitmask";
  let arguments = (ins I1:$cond);
  let results = (outs AnyType:$result);
}

def Crucible_WaveBitCountOp : Crucible_Op<"wave_bit_count", [Pure]> {
  let summary = "Subgroup active/prefix/multi-prefix bit count";
  let arguments = (ins
    I1:$cond,
    Crucible_WaveBitCountKindAttr:$kind,
    Variadic<I32>:$mask
  );
  let results = (outs I32:$result);
}

def Crucible_WaveMultiPrefixOp : Crucible_Op<"wave_multi_prefix", [Pure]> {
  let summary = "Subgroup multi-prefix reduction with partition mask";
  let arguments = (ins
    AnyType:$value,
    Variadic<I32>:$mask,
    Crucible_WaveMultiPrefixOpKindAttr:$op,
    BoolAttr:$is_unsigned
  );
  let results = (outs AnyType:$result);
}

def Crucible_QuadReadLaneAtOp : Crucible_Op<"quad_read_lane_at", [Pure]> {
  let summary = "Read a value from a specific lane (0..3) in the current quad";
  let arguments = (ins
    AnyType:$value,
    I32:$quad_lane
  );
  let results = (outs AnyType:$result);
}

def Crucible_QuadOp : Crucible_Op<"quad_op", [Pure]> {
  let summary = "Quad-group cross-lane read (X, Y, or diagonal)";
  let arguments = (ins
    AnyType:$value,
    Crucible_QuadOpKindAttr:$kind
  );
  let results = (outs AnyType:$result);
}

//===----------------------------------------------------------------------===//
// Mesh & Amplification Shader Operations
//===----------------------------------------------------------------------===//

def Crucible_GetMeshPayloadOp : Crucible_Op<"get_mesh_payload", [Pure]> {
  let summary = "Obtain pointer to the incoming mesh shader payload";
  let results = (outs AnyType:$result);
}

def Crucible_SetMeshOutputCountsOp : Crucible_Op<"set_mesh_output_counts"> {
  let summary = "Set vertex and primitive output counts for a mesh shader";
  let arguments = (ins
    I32:$num_vertices,
    I32:$num_primitives
  );
}

def Crucible_EmitIndicesOp : Crucible_Op<"emit_indices"> {
  let summary = "Emit primitive vertex indices in a mesh shader";
  let arguments = (ins
    I32:$primitive_index,
    Variadic<I32>:$vertex_indices
  );
}

def Crucible_StoreVertexOutputOp : Crucible_Op<"store_vertex_output"> {
  let summary = "Store a mesh shader per-vertex output element";
  let arguments = (ins
    I32Attr:$output_id,
    I32:$row,
    I8Attr:$col,
    AnyType:$value,
    I32:$vertex_index
  );
}

def Crucible_StorePrimitiveOutputOp : Crucible_Op<"store_primitive_output"> {
  let summary = "Store a mesh shader per-primitive output element";
  let arguments = (ins
    I32Attr:$output_id,
    I32:$row,
    I8Attr:$col,
    AnyType:$value,
    I32:$primitive_index
  );
}

def Crucible_DispatchMeshOp : Crucible_Op<"dispatch_mesh"> {
  let summary = "Dispatch mesh threadgroups from an amplification shader";
  let arguments = (ins
    I32:$tg_x,
    I32:$tg_y,
    I32:$tg_z,
    AnyType:$payload
  );
}

//===----------------------------------------------------------------------===//
// Raytracing & RayQuery Operations
//===----------------------------------------------------------------------===//

def Crucible_TraceRayOp : Crucible_Op<"trace_ray"> {
  let summary = "Trace a ray against an acceleration structure";
  let arguments = (ins
    Crucible_ResourceType:$accel,
    I32:$ray_flags,
    I32:$instance_mask,
    I32:$hit_group_offset,
    I32:$hit_group_stride,
    I32:$miss_index,
    F32:$origin_x, F32:$origin_y, F32:$origin_z,
    F32:$t_min,
    F32:$dir_x, F32:$dir_y, F32:$dir_z,
    F32:$t_max,
    AnyType:$payload
  );
}

def Crucible_RayQueryAllocOp : Crucible_Op<"ray_query_alloc"> {
  let summary = "Allocate an inline RayQuery object";
  let arguments = (ins I32:$const_ray_flags);
  let results = (outs Crucible_RayQueryType:$result);
}

def Crucible_RayQueryTraceInlineOp : Crucible_Op<"ray_query_trace_inline"> {
  let summary = "Initialize an inline RayQuery traversal";
  let arguments = (ins
    Crucible_RayQueryType:$query,
    Crucible_ResourceType:$accel,
    I32:$ray_flags,
    I32:$instance_mask,
    F32:$origin_x, F32:$origin_y, F32:$origin_z,
    F32:$t_min,
    F32:$dir_x, F32:$dir_y, F32:$dir_z,
    F32:$t_max
  );
}

def Crucible_RayQueryProceedOp : Crucible_Op<"ray_query_proceed"> {
  let summary = "Advance an inline RayQuery traversal";
  let arguments = (ins Crucible_RayQueryType:$query);
  let results = (outs I1:$result);
}

def Crucible_RayQueryAbortOp : Crucible_Op<"ray_query_abort"> {
  let summary = "Abort an inline RayQuery traversal";
  let arguments = (ins Crucible_RayQueryType:$query);
}

def Crucible_RayQueryCommitOp : Crucible_Op<"ray_query_commit"> {
  let summary = "Commit a candidate triangle or procedural hit on a RayQuery";
  let arguments = (ins
    Crucible_RayQueryType:$query,
    Crucible_RayQueryCommitKindAttr:$kind,
    Optional<F32>:$t_hit
  );
}

def Crucible_RayQueryGetOp : Crucible_Op<"ray_query_get", [Pure]> {
  let summary = "Query a state property of an inline RayQuery";
  let arguments = (ins
    Crucible_RayQueryType:$query,
    Crucible_RayQueryFieldAttr:$field,
    Optional<I32>:$component
  );
  let results = (outs AnyType:$result);
}

//===----------------------------------------------------------------------===//
// Shader Math & Bit-Manipulation Gaps (not in upstream math/arith/llvm)
//===----------------------------------------------------------------------===//

def Crucible_SaturateOp : Crucible_Op<"saturate", [Pure]> {
  let summary = "Clamp floating-point value to [0.0, 1.0]";
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
}

def Crucible_FrcOp : Crucible_Op<"frc", [Pure]> {
  let summary = "Fractional part: x - floor(x)";
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
}

def Crucible_FirstBitOp : Crucible_Op<"first_bit", [Pure]> {
  let summary = "Find first set bit (low, unsigned high, or signed high)";
  let arguments = (ins
    AnyType:$value,
    Crucible_FirstBitKindAttr:$kind
  );
  let results = (outs I32:$result);
}

def Crucible_BfeOp : Crucible_Op<"bfe", [Pure]> {
  let summary = "Signed or unsigned bitfield extract";
  let arguments = (ins
    I32:$width,
    I32:$offset,
    AnyType:$value,
    BoolAttr:$is_signed
  );
  let results = (outs AnyType:$result);
}

def Crucible_BfiOp : Crucible_Op<"bfi", [Pure]> {
  let summary = "Bitfield insert";
  let arguments = (ins
    I32:$width,
    I32:$offset,
    AnyType:$insert,
    AnyType:$base
  );
  let results = (outs AnyType:$result);
}

def Crucible_DotOp : Crucible_Op<"dot", [Pure, AttrSizedOperandSegments]> {
  let summary = "Scalarized float/half 2/3/4-component vector dot product";
  let arguments = (ins
    Variadic<AnyType>:$lhs,
    Variadic<AnyType>:$rhs
  );
  let results = (outs AnyType:$result);
}

def Crucible_DotPackedOp : Crucible_Op<"dot_packed", [Pure]> {
  let summary = "Packed dot-product-accumulate (2xf16, 4xi8, or 4xu8)";
  let arguments = (ins
    AnyType:$acc,
    Variadic<AnyType>:$inputs,
    Crucible_DotPackedKindAttr:$kind
  );
  let results = (outs AnyType:$result);
}

def Crucible_Pack4x8Op : Crucible_Op<"pack_4x8", [Pure]> {
  let summary = "Pack 4 16/32-bit integers into a 32-bit 4x8-bit value";
  let arguments = (ins
    I8Attr:$mode,
    AnyType:$v0, AnyType:$v1, AnyType:$v2, AnyType:$v3
  );
  let results = (outs I32:$result);
}

def Crucible_Unpack4x8Op : Crucible_Op<"unpack_4x8", [Pure]> {
  let summary = "Unpack a 32-bit 4x8-bit value into 4 32-bit integers";
  let arguments = (ins
    I8Attr:$mode,
    I32:$packed
  );
  let results = (outs AnyType:$result);
}

def Crucible_LegacyF32ToF16Op : Crucible_Op<"legacy_f32_to_f16", [Pure]> {
  let summary = "Convert f32 to IEEE binary16 bits in an i32";
  let arguments = (ins F32:$value);
  let results = (outs I32:$result);
}

def Crucible_LegacyF16ToF32Op : Crucible_Op<"legacy_f16_to_f32", [Pure]> {
  let summary = "Convert IEEE binary16 bits in an i32 to f32";
  let arguments = (ins I32:$value);
  let results = (outs F32:$result);
}

def Crucible_MakeDoubleOp : Crucible_Op<"make_double", [Pure]> {
  let summary = "Construct an f64 from low and high i32 bit words";
  let arguments = (ins I32:$lo, I32:$hi);
  let results = (outs F64:$result);
}

def Crucible_SplitDoubleOp : Crucible_Op<"split_double", [Pure]> {
  let summary = "Split an f64 into low and high i32 bit words";
  let arguments = (ins F64:$value);
  let results = (outs I32:$lo, I32:$hi);
}

def Crucible_UAddcOp : Crucible_Op<"uaddc", [Pure]> {
  let summary = "Unsigned integer addition returning (sum, carry)";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$sum, I1:$carry);
}

#endif // CRUCIBLE_OPS_TD

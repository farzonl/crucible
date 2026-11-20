//===- CrucibleEnums.td - Crucible MLIR dialect enums -------*- tablegen -*-===//
//
// Part of Crucible, an open-source multi-frontend, multi-backend shader compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRUCIBLE_ENUMS_TD
#define CRUCIBLE_ENUMS_TD

include "crucible/Dialect/CrucibleDialect.td"
include "mlir/IR/EnumAttr.td"

def Crucible_ResourceClass : I32EnumAttr<"ResourceClass",
    "Shader resource binding class", [
  I32EnumAttrCase<"SRV", 0, "srv">,
  I32EnumAttrCase<"UAV", 1, "uav">,
  I32EnumAttrCase<"CBV", 2, "cbv">,
  I32EnumAttrCase<"Sampler", 3, "sampler">,
]> {
  let cppNamespace = "::crucible::ir";
  let genSpecializedAttr = 0;
}
def Crucible_ResourceClassAttr
    : EnumAttr<Crucible_Dialect, Crucible_ResourceClass, "resource_class">;

def Crucible_BufferKind : I32EnumAttr<"BufferKind",
    "Shader buffer access kind", [
  I32EnumAttrCase<"Typed", 0, "typed">,
  I32EnumAttrCase<"Raw", 1, "raw">,
  I32EnumAttrCase<"Structured", 2, "structured">,
]> {
  let cppNamespace = "::crucible::ir";
  let genSpecializedAttr = 0;
}
def Crucible_BufferKindAttr
    : EnumAttr<Crucible_Dialect, Crucible_BufferKind, "buffer_kind">;

def Crucible_AtomicOp : I32EnumAttr<"AtomicOp",
    "Shader buffer/texture atomic binary operation", [
  I32EnumAttrCase<"Add", 0, "add">,
  I32EnumAttrCase<"And", 1, "and">,
  I32EnumAttrCase<"Or", 2, "or">,
  I32EnumAttrCase<"Xor", 3, "xor">,
  I32EnumAttrCase<"SMin", 4, "smin">,
  I32EnumAttrCase<"SMax", 5, "smax">,
  I32EnumAttrCase<"UMin", 6, "umin">,
  I32EnumAttrCase<"UMax", 7, "umax">,
  I32EnumAttrCase<"Xchg", 8, "xchg">,
]> {
  let cppNamespace = "::crucible::ir";
  let genSpecializedAttr = 0;
}
def Crucible_AtomicOpAttr
    : EnumAttr<Crucible_Dialect, Crucible_AtomicOp, "atomic_op">;

def Crucible_SampleMode : I32EnumAttr<"SampleMode",
    "Texture sampling mode", [
  I32EnumAttrCase<"Implicit", 0, "implicit">,
  I32EnumAttrCase<"Bias", 1, "bias">,
  I32EnumAttrCase<"Level", 2, "level">,
  I32EnumAttrCase<"Grad", 3, "grad">,
  I32EnumAttrCase<"Cmp", 4, "cmp">,
  I32EnumAttrCase<"CmpLevelZero", 5, "cmp_level_zero">,
]> {
  let cppNamespace = "::crucible::ir";
  let genSpecializedAttr = 0;
}
def Crucible_SampleModeAttr
    : EnumAttr<Crucible_Dialect, Crucible_SampleMode, "sample_mode">;

def Crucible_DerivAxis : I32EnumAttr<"DerivAxis",
    "Screen-space derivative axis", [
  I32EnumAttrCase<"X", 0, "x">,
  I32EnumAttrCase<"Y", 1, "y">,
]> {
  let cppNamespace = "::crucible::ir";
  let genSpecializedAttr = 0;
}
def Crucible_DerivAxisAttr
    : EnumAttr<Crucible_Dialect, Crucible_DerivAxis, "deriv_axis">;

def Crucible_DerivQuality : I32EnumAttr<"DerivQuality",
    "Screen-space derivative precision", [
  I32EnumAttrCase<"Coarse", 0, "coarse">,
  I32EnumAttrCase<"Fine", 1, "fine">,
]> {
  let cppNamespace = "::crucible::ir";
  let genSpecializedAttr = 0;
}
def Crucible_DerivQualityAttr
    : EnumAttr<Crucible_Dialect, Crucible_DerivQuality, "deriv_quality">;

def Crucible_SysValKind : I32EnumAttr<"SysValKind",
    "Shader built-in system value kind", [
  I32EnumAttrCase<"FlattenedThreadIdInGroup", 0, "flattened_thread_id_in_group">,
  I32EnumAttrCase<"PrimitiveId", 1, "primitive_id">,
  I32EnumAttrCase<"DomainLocation", 2, "domain_location">,
  I32EnumAttrCase<"OutputControlPointId", 3, "output_control_point_id">,
  I32EnumAttrCase<"DispatchRaysIndex", 4, "dispatch_rays_index">,
  I32EnumAttrCase<"DispatchRaysDimensions", 5, "dispatch_rays_dimensions">,
  I32EnumAttrCase<"WorldRayOrigin", 6, "world_ray_origin">,
  I32EnumAttrCase<"WorldRayDirection", 7, "world_ray_direction">,
  I32EnumAttrCase<"RayTCurrent", 8, "ray_t_current">,
  I32EnumAttrCase<"PrimitiveIndex", 9, "primitive_index">,
]> {
  let cppNamespace = "::crucible::ir";
  let genSpecializedAttr = 0;
}
def Crucible_SysValKindAttr
    : EnumAttr<Crucible_Dialect, Crucible_SysValKind, "sysval_kind">;

def Crucible_WaveVoteKind : I32EnumAttr<"WaveVoteKind",
    "Subgroup vote operation", [
  I32EnumAttrCase<"Any", 0, "any">,
  I32EnumAttrCase<"All", 1, "all">,
  I32EnumAttrCase<"AllEqual", 2, "all_equal">,
]> {
  let cppNamespace = "::crucible::ir";
  let genSpecializedAttr = 0;
}
def Crucible_WaveVoteKindAttr
    : EnumAttr<Crucible_Dialect, Crucible_WaveVoteKind, "wave_vote_kind">;

def Crucible_WaveBitCountKind : I32EnumAttr<"WaveBitCountKind",
    "Subgroup bit-count operation", [
  I32EnumAttrCase<"All", 0, "all">,
  I32EnumAttrCase<"Prefix", 1, "prefix">,
  I32EnumAttrCase<"MultiPrefix", 2, "multi_prefix">,
]> {
  let cppNamespace = "::crucible::ir";
  let genSpecializedAttr = 0;
}
def Crucible_WaveBitCountKindAttr
    : EnumAttr<Crucible_Dialect, Crucible_WaveBitCountKind, "wave_bit_count_kind">;

def Crucible_WaveMultiPrefixOpKind : I32EnumAttr<"WaveMultiPrefixOpKind",
    "Subgroup multi-prefix reduction operation", [
  I32EnumAttrCase<"Sum", 0, "sum">,
  I32EnumAttrCase<"And", 1, "and">,
  I32EnumAttrCase<"Or", 2, "or">,
  I32EnumAttrCase<"Xor", 3, "xor">,
  I32EnumAttrCase<"Product", 4, "product">,
]> {
  let cppNamespace = "::crucible::ir";
  let genSpecializedAttr = 0;
}
def Crucible_WaveMultiPrefixOpKindAttr
    : EnumAttr<Crucible_Dialect, Crucible_WaveMultiPrefixOpKind, "wave_multi_prefix_op">;

def Crucible_QuadOpKind : I32EnumAttr<"QuadOpKind",
    "Quad-group shuffle operation", [
  I32EnumAttrCase<"ReadAcrossX", 0, "read_across_x">,
  I32EnumAttrCase<"ReadAcrossY", 1, "read_across_y">,
  I32EnumAttrCase<"ReadAcrossDiagonal", 2, "read_across_diagonal">,
]> {
  let cppNamespace = "::crucible::ir";
  let genSpecializedAttr = 0;
}
def Crucible_QuadOpKindAttr
    : EnumAttr<Crucible_Dialect, Crucible_QuadOpKind, "quad_op_kind">;

def Crucible_FirstBitKind : I32EnumAttr<"FirstBitKind",
    "First-set-bit scan variant", [
  I32EnumAttrCase<"Lo", 0, "lo">,
  I32EnumAttrCase<"Hi", 1, "hi">,
  I32EnumAttrCase<"SHi", 2, "shi">,
]> {
  let cppNamespace = "::crucible::ir";
  let genSpecializedAttr = 0;
}
def Crucible_FirstBitKindAttr
    : EnumAttr<Crucible_Dialect, Crucible_FirstBitKind, "first_bit_kind">;

def Crucible_DotPackedKind : I32EnumAttr<"DotPackedKind",
    "Packed dot-product-accumulate variant", [
  I32EnumAttrCase<"F16x2", 0, "f16x2">,
  I32EnumAttrCase<"I8x4", 1, "i8x4">,
  I32EnumAttrCase<"U8x4", 2, "u8x4">,
]> {
  let cppNamespace = "::crucible::ir";
  let genSpecializedAttr = 0;
}
def Crucible_DotPackedKindAttr
    : EnumAttr<Crucible_Dialect, Crucible_DotPackedKind, "dot_packed_kind">;

def Crucible_RayQueryCommitKind : I32EnumAttr<"RayQueryCommitKind",
    "RayQuery candidate hit commit kind", [
  I32EnumAttrCase<"Triangle", 0, "triangle">,
  I32EnumAttrCase<"Procedural", 1, "procedural">,
]> {
  let cppNamespace = "::crucible::ir";
  let genSpecializedAttr = 0;
}
def Crucible_RayQueryCommitKindAttr
    : EnumAttr<Crucible_Dialect, Crucible_RayQueryCommitKind, "ray_query_commit_kind">;

def Crucible_RayQueryField : I32EnumAttr<"RayQueryField",
    "RayQuery state property to query", [
  I32EnumAttrCase<"CommittedStatus", 0, "committed_status">,
  I32EnumAttrCase<"CandidateType", 1, "candidate_type">,
  I32EnumAttrCase<"CommittedTriangleFrontFace", 2, "committed_triangle_front_face">,
  I32EnumAttrCase<"CommittedTriangleBarycentrics", 3, "committed_triangle_barycentrics">,
  I32EnumAttrCase<"RayFlags", 4, "ray_flags">,
  I32EnumAttrCase<"WorldRayOrigin", 5, "world_ray_origin">,
  I32EnumAttrCase<"WorldRayDirection", 6, "world_ray_direction">,
  I32EnumAttrCase<"RayTMin", 7, "ray_t_min">,
  I32EnumAttrCase<"CommittedRayT", 8, "committed_ray_t">,
  I32EnumAttrCase<"CommittedInstanceId", 9, "committed_instance_id">,
  I32EnumAttrCase<"CommittedInstanceIndex", 10, "committed_instance_index">,
  I32EnumAttrCase<"CommittedPrimitiveIndex", 11, "committed_primitive_index">,
  I32EnumAttrCase<"CommittedGeometryIndex", 12, "committed_geometry_index">,
  I32EnumAttrCase<"CommittedInstanceContrib", 13, "committed_instance_contrib">,
]> {
  let cppNamespace = "::crucible::ir";
  let genSpecializedAttr = 0;
}
def Crucible_RayQueryFieldAttr
    : EnumAttr<Crucible_Dialect, Crucible_RayQueryField, "ray_query_field">;

#endif // CRUCIBLE_ENUMS_TD

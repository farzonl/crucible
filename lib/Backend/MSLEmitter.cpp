//===- MSLEmitter.cpp - Structured MLIR -> Metal Shading Language -*- C++ -*-=//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Prints a structurized shader (nested scf.if / scf.while regions over
// LLVM/GPU/Math/Crucible dialect ops) as clean,
// idiomatic MSL. Pure single-use values are inlined into their consumer
// expression so the output reads like hand-written Metal; multi-use values
// become typed local temporaries; scf region results become mutable locals
// (PHI demotion falls out of SCF value yields).
//
//===----------------------------------------------------------------------===//

#include "MSLEmitter.h"
#include "crucible/Diagnostic.h"
#include "crucible/Dialect/CrucibleDialect.h"

#include <deque>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"

#include <optional>
#include <string>

using namespace llvm;

namespace crucible {
namespace {

//===----------------------------------------------------------------------===//
// Small helpers
//===----------------------------------------------------------------------===//

// C operator precedence levels used to parenthesize only when required.
enum Precedence : int {
  PrecUnknown = 0,
  PrecTernary = 4,
  PrecLOr = 5,
  PrecLAnd = 6,
  PrecOr = 7,
  PrecXor = 8,
  PrecAnd = 9,
  PrecEq = 10,
  PrecRel = 11,
  PrecShift = 12,
  PrecAdd = 13,
  PrecMul = 14,
  PrecUnary = 15,
  PrecPrimary = 16,
};

struct Expr {
  std::string Text;
  int Prec = PrecPrimary;
};

std::string signedView(mlir::Type T, StringRef Value) {
  if (T.isInteger(1))
    return "(-int(" + Value.str() + "))";
  if (auto VT = mlir::dyn_cast<mlir::VectorType>(T)) {
    mlir::Type E = VT.getElementType();
    if (E.isInteger(1))
      return "(-int" + std::to_string(VT.getNumElements()) + "(" + Value.str() + "))";
    StringRef Base = E.isInteger(8) ? "char" : E.isInteger(16) ? "short" : E.isInteger(64) ? "long" : "int";
    return ("as_type<" + Base + std::to_string(VT.getNumElements()) + ">(" + Value + ")").str();
  }
  StringRef Ty = T.isInteger(8) ? "char" : T.isInteger(16) ? "short" : T.isInteger(64) ? "long" : "int";
  if (T.isInteger(32) && Value.ends_with("u")) {
    uint64_t U = 0;
    if (!Value.drop_back(1).getAsInteger(10, U) && U <= UINT32_MAX) {
      int32_t S = static_cast<int32_t>(static_cast<uint32_t>(U));
      if (S >= 0)
        return std::to_string(S);
      if (S == INT32_MIN)
        return "(-2147483647 - 1)";
      return "(" + std::to_string(S) + ")";
    }
  }
  if (T.isInteger(64) && Value.ends_with("ul")) {
    uint64_t U = 0;
    if (!Value.drop_back(2).getAsInteger(10, U) && U <= uint64_t(INT64_MAX))
      return std::to_string(U) + "l";
  }
  return ("as_type<" + Ty + ">(" + Value + ")").str();
}

std::string unsignedView(mlir::Type T, StringRef Value) {
  if (auto VT = mlir::dyn_cast<mlir::VectorType>(T)) {
    mlir::Type E = VT.getElementType();
    StringRef Base = E.isInteger(8) ? "uchar" : E.isInteger(16) ? "ushort" : E.isInteger(64) ? "ulong" : "uint";
    return ("as_type<" + Base + std::to_string(VT.getNumElements()) + ">(" + Value + ")").str();
  }
  StringRef Ty = T.isInteger(8) ? "uchar" : T.isInteger(16) ? "ushort" : T.isInteger(64) ? "ulong" : "uint";
  return (Ty + "(" + Value + ")").str();
}

// Source for on-demand MSL helper functions implementing exact DXIL
// semantics where metal_stdlib differs (bit scans return -1 on zero input,
// bitfield ops mask their operands, packed dot products, ballot shape,
// compare-exchange result semantics).
StringRef helperSource(StringRef Name) {
  if (Name == "crucible_firstbitlo")
    return "template <typename T> static inline uint crucible_firstbitlo(T v) {\n"
           "  return v == T(0) ? 0xffffffffu : uint(ctz(v));\n"
           "}\n";
  if (Name == "crucible_firstbithi")
    return "template <typename T> static inline uint crucible_firstbithi(T v) {\n"
           "  return v == T(0) ? 0xffffffffu : uint(clz(v));\n"
           "}\n";
  if (Name == "crucible_firstbitshi")
    return "template <typename T> static inline uint crucible_firstbitshi(T v) {\n"
           "  if ((v >> (sizeof(T) * 8 - 1)) != 0) v = T(~v);\n"
           "  return v == T(0) ? 0xffffffffu : uint(clz(v));\n"
           "}\n";
  if (Name == "crucible_ubfe")
    return "static inline uint crucible_ubfe(uint width, uint offset, uint v) "
           "{\n"
           "  width &= 31u; offset &= 31u;\n"
           "  if (width == 0u) return 0u;\n"
           "  if (width + offset < 32u) return (v << (32u - width - offset)) "
           ">> (32u - width);\n"
           "  return v >> offset;\n"
           "}\n";
  if (Name == "crucible_ibfe")
    return "static inline uint crucible_ibfe(uint width, uint offset, uint v) "
           "{\n"
           "  width &= 31u; offset &= 31u;\n"
           "  if (width == 0u) return 0u;\n"
           "  int sv = as_type<int>(v);\n"
           "  if (width + offset < 32u)\n"
           "    return as_type<uint>((sv << (32u - width - offset)) >> (32u - "
           "width));\n"
           "  return as_type<uint>(sv >> offset);\n"
           "}\n";
  if (Name == "crucible_bfi")
    return "static inline uint crucible_bfi(uint width, uint offset, uint "
           "src, uint dst) {\n"
           "  width &= 31u; offset &= 31u;\n"
           "  uint mask = (width == 0u ? 0u : (0xffffffffu >> (32u - width))) "
           "<< offset;\n"
           "  return (src << offset & mask) | (dst & ~mask);\n"
           "}\n";
  if (Name == "crucible_dot4add_u8packed")
    return "static inline uint crucible_dot4add_u8packed(uint acc, uint a, "
           "uint b) {\n"
           "  uchar4 ua = as_type<uchar4>(a), ub = as_type<uchar4>(b);\n"
           "  return acc + uint(ua.x) * uint(ub.x) + uint(ua.y) * uint(ub.y) "
           "+\n"
           "         uint(ua.z) * uint(ub.z) + uint(ua.w) * uint(ub.w);\n"
           "}\n";
  if (Name == "crucible_dot4add_i8packed")
    return "static inline uint crucible_dot4add_i8packed(uint acc, uint a, "
           "uint b) {\n"
           "  char4 sa = as_type<char4>(a), sb = as_type<char4>(b);\n"
           "  int r = as_type<int>(acc) + int(sa.x) * int(sb.x) + int(sa.y) * "
           "int(sb.y) +\n"
           "          int(sa.z) * int(sb.z) + int(sa.w) * int(sb.w);\n"
           "  return as_type<uint>(r);\n"
           "}\n";
  if (Name == "crucible_pack4x8")
    return "template <uint Mode>\n"
           "static inline uint crucible_pack4x8(int4 v) {\n"
           "  if (Mode == 1u) v = clamp(v, int4(0), int4(255));\n"
           "  if (Mode == 2u) v = clamp(v, int4(-128), int4(127));\n"
           "  uint4 u = as_type<uint4>(v) & 0xFFu;\n"
           "  return u.x | (u.y << 8u) | (u.z << 16u) | (u.w << 24u);\n"
           "}\n";
  if (Name == "crucible_ldexp")
    return "template <typename T, typename I>\n"
           "static inline T crucible_ldexp(T x, I e) {\n"
           "#pragma METAL fp math_mode(safe)\n"
           "  return ldexp(x, e);\n"
           "}\n";
  if (Name == "crucible_ftz")
    return "static inline float crucible_ftz(float x) {\n"
           "  return (as_type<uint>(x) & 0x7f800000u) == 0u\n"
           "             ? as_type<float>(as_type<uint>(x) & 0x80000000u)\n"
           "             : x;\n"
           "}\n"
           "static inline float2 crucible_ftz(float2 x) {\n"
           "  return select(x, as_type<float2>(as_type<uint2>(x) & 0x80000000u),\n"
           "                (as_type<uint2>(x) & 0x7f800000u) == 0u);\n"
           "}\n"
           "static inline float3 crucible_ftz(float3 x) {\n"
           "  return select(x, as_type<float3>(as_type<uint3>(x) & 0x80000000u),\n"
           "                (as_type<uint3>(x) & 0x7f800000u) == 0u);\n"
           "}\n"
           "static inline float4 crucible_ftz(float4 x) {\n"
           "  return select(x, as_type<float4>(as_type<uint4>(x) & 0x80000000u),\n"
           "                (as_type<uint4>(x) & 0x7f800000u) == 0u);\n"
           "}\n";
  if (Name == "crucible_powi")
    return "template <int N, typename T> static inline T crucible_powi(T x) {\n"
           "  T r = T(1), b = x;\n"
           "  int e = N;\n"
           "  while (e > 0) {\n"
           "    if (e & 1) r *= b;\n"
           "    e >>= 1;\n"
           "    if (e > 0) b *= b;\n"
           "  }\n"
           "  return r;\n"
           "}\n";
  if (Name == "crucible_pow2_5")
    return "template <typename T> static inline T crucible_pow2_5(T x) {\n"
           "  return x * x * sqrt(x);\n"
           "}\n";
  if (Name == "crucible_ray_params")
    return "static inline intersection_params crucible_ray_params(uint flags) {\n"
           "  intersection_params p;\n"
           "  if (flags & 0x01u) p.force_opacity(forced_opacity::opaque);\n"
           "  if (flags & 0x02u) p.force_opacity(forced_opacity::non_opaque);\n"
           "  if (flags & 0x04u) p.accept_any_intersection(true);\n"
           "  if (flags & 0x10u) p.set_triangle_cull_mode(triangle_cull_mode::back);\n"
           "  if (flags & 0x20u) p.set_triangle_cull_mode(triangle_cull_mode::front);\n"
           "  if (flags & 0x40u) p.set_opacity_cull_mode(opacity_cull_mode::opaque);\n"
           "  if (flags & 0x80u) p.set_opacity_cull_mode(opacity_cull_mode::non_opaque);\n"
           "  if (flags & 0x100u) p.set_geometry_cull_mode(geometry_cull_mode::triangle);\n"
           "  if (flags & 0x200u) p.set_geometry_cull_mode(geometry_cull_mode::bounding_box);\n"
           "  return p;\n"
           "}\n";
  if (Name == "crucible_wave_ballot")
    return "static inline uint4 crucible_wave_ballot(bool b) {\n"
           "  simd_vote v = simd_ballot(b);\n"
           "  uint64_t bits = (simd_vote::vote_t)v;\n"
           "  return uint4(uint(bits), uint(bits >> 32), 0u, 0u);\n"
           "}\n";
  if (Name == "crucible_wave_wide")
    return R"MSL(static inline ulong crucible_wave_shuffle(ulong v, ushort lane) {
  return as_type<ulong>(simd_shuffle(as_type<uint2>(v), lane));
}
template <typename T, uint Op, bool Prefix>
static inline T crucible_wave_wide(T v, uint lane) {
  ulong bits = ulong((simd_vote::vote_t)simd_ballot(true));
  T acc = Op == 1u ? T(1) : T(0);
  bool first = true;
  // Keep the shuffle converged on the original active mask, even for prefixes.
  while (bits != 0ul) {
    uint lo = uint(bits);
    uint src = lo != 0u ? ctz(lo) : 32u + ctz(uint(bits >> 32));
    T x = as_type<T>(crucible_wave_shuffle(as_type<ulong>(v), ushort(src)));
    if (!Prefix || src < lane) {
      if (Op == 0u) acc += x;
      if (Op == 1u) acc *= x;
      if (Op == 2u) acc = first ? x : min(acc, x);
      if (Op == 3u) acc = first ? x : max(acc, x);
      first = false;
    }
    bits &= bits - 1ul;
  }
  return acc;
}
)MSL";
  if (Name == "crucible_wave_wide_f64")
    return R"MSL(template <uint Op, bool Prefix>
static inline ulong crucible_wave_wide_f64(ulong v, uint lane) {
  ulong bits = ulong((simd_vote::vote_t)simd_ballot(true));
  ulong acc = Op == 1u ? 0x3ff0000000000000ul : 0ul;
  bool first = true;
  while (bits != 0ul) {
    uint lo = uint(bits);
    uint src = lo != 0u ? ctz(lo) : 32u + ctz(uint(bits >> 32));
    ulong x = as_type<ulong>(simd_shuffle(as_type<uint2>(v), ushort(src)));
    if (!Prefix || src < lane) {
      if (Op == 0u) acc = first ? x : crucible_f64_add(acc, x);
      if (Op == 1u) acc = first ? x : crucible_f64_mul(acc, x);
      if (Op == 2u) acc = first ? x : crucible_f64_min(acc, x);
      if (Op == 3u) acc = first ? x : crucible_f64_max(acc, x);
      first = false;
    }
    bits &= bits - 1ul;
  }
  return acc;
}
)MSL";
  if (Name == "crucible_wave_multi_prefix_f64")
    return R"MSL(template <uint Op>
static inline ulong crucible_wave_multi_prefix_f64(ulong v, uint mask0,
                                                   uint mask1, uint lane) {
  ulong active = ulong((simd_vote::vote_t)simd_ballot(true));
  ulong acc = Op == 4u ? 0x3ff0000000000000ul : 0ul;
  while (active != 0ul) {
    uint lo = uint(active);
    uint src = lo != 0u ? ctz(lo) : 32u + ctz(uint(active >> 32));
    ulong x = as_type<ulong>(simd_shuffle(as_type<uint2>(v), ushort(src)));
    uint m = src < 32u ? mask0 : mask1;
    if (src < lane && ((m >> (src & 31u)) & 1u) != 0u) {
      if (Op == 0u) acc = crucible_f64_add(acc, x);
      if (Op == 4u) acc = crucible_f64_mul(acc, x);
    }
    active &= active - 1ul;
  }
  return acc;
}
)MSL";
  if (Name == "crucible_wave_prefix_sum")
    return "template <typename T> static inline T crucible_wave_prefix_sum(T v) {\n"
           "  T sum = simd_prefix_exclusive_sum(v);\n"
           "  return simd_is_first() ? T(0) : sum;\n"
           "}\n";
  if (Name == "crucible_wave_multi_prefix")
    return R"MSL(template <typename T>
static inline T crucible_mp_shuffle(T v, ushort lane) {
  return simd_shuffle(v, lane);
}
__attribute__((unused)) static inline ulong crucible_mp_shuffle(
    ulong v, ushort lane) {
  return as_type<ulong>(simd_shuffle(as_type<uint2>(v), lane));
}
template <typename T, uint Op>
static inline T crucible_wave_multi_prefix_arith(T v, uint mask0, uint mask1,
                                                 uint lane) {
  ulong active = ulong((simd_vote::vote_t)simd_ballot(true));
  T acc = Op == 4u ? T(1) : T(0);
  // Keep simd_shuffle converged on all active lanes.
  while (active != 0ul) {
    uint lo = uint(active);
    uint src = lo != 0u ? ctz(lo) : 32u + ctz(uint(active >> 32));
    T x = crucible_mp_shuffle(v, ushort(src));
    uint m = src < 32u ? mask0 : mask1;
    if (src < lane && ((m >> (src & 31u)) & 1u) != 0u) {
      if (Op == 0u) acc += x;
      if (Op == 4u) acc *= x;
    }
    active &= active - 1ul;
  }
  return acc;
}
template <typename T, uint Op>
static inline T crucible_wave_multi_prefix_bit(T v, uint mask0, uint mask1,
                                               uint lane) {
  ulong active = ulong((simd_vote::vote_t)simd_ballot(true));
  T acc = Op == 1u ? ~T(0) : T(0);
  while (active != 0ul) {
    uint lo = uint(active);
    uint src = lo != 0u ? ctz(lo) : 32u + ctz(uint(active >> 32));
    T x = crucible_mp_shuffle(v, ushort(src));
    uint m = src < 32u ? mask0 : mask1;
    if (src < lane && ((m >> (src & 31u)) & 1u) != 0u) {
      if (Op == 1u) acc &= x;
      if (Op == 2u) acc |= x;
      if (Op == 3u) acc ^= x;
    }
    active &= active - 1ul;
  }
  return acc;
}
)MSL";
  if (Name == "crucible_wave_multi_prefix_bitcount")
    return "";
  if (Name == "crucible_atomic_cmpxchg_texture2d")
    return R"MSL(template <typename T>
static inline T crucible_atomic_cmpxchg_texture2d(
    texture2d<T, access::read_write> tex, uint2 coord, T compare, T desired) {
  vec<T, 4> expected(compare);
  while (!tex.atomic_compare_exchange_weak(coord, &expected, vec<T, 4>(desired))) {
    if (expected.x != compare) break;
  }
  return expected.x;
}
)MSL";
  if (Name == "crucible_atomic_cmpxchg_device")
    return "static inline uint crucible_atomic_cmpxchg(device atomic_uint "
           "*p, uint cmp, uint val) {\n"
           "  uint expected = cmp;\n"
           "  while (!atomic_compare_exchange_weak_explicit(\n"
           "      p, &expected, val, memory_order_relaxed, "
           "memory_order_relaxed)) {\n"
           "    if (expected != cmp) break;\n"
           "    expected = cmp;\n"
           "  }\n"
           "  return expected;\n"
           "}\n";
  if (Name == "crucible_atomic_cmpxchg_threadgroup")
    return "static inline uint crucible_atomic_cmpxchg(threadgroup "
           "atomic_uint *p, uint cmp, uint val) {\n"
           "  uint expected = cmp;\n"
           "  while (!atomic_compare_exchange_weak_explicit(\n"
           "      p, &expected, val, memory_order_relaxed, "
           "memory_order_relaxed)) {\n"
           "    if (expected != cmp) break;\n"
           "    expected = cmp;\n"
           "  }\n"
           "  return expected;\n"
           "}\n";
  if (Name == "crucible_atomic_cmpxchg_device_i64")
    return "static inline ulong crucible_atomic_cmpxchg(device atomic_ulong "
           "*p, ulong cmp, ulong val) {\n"
           "  ulong expected = cmp;\n"
           "  while (!atomic_compare_exchange_weak_explicit(\n"
           "      p, &expected, val, memory_order_relaxed, "
           "memory_order_relaxed)) {\n"
           "    if (expected != cmp) break;\n"
           "    expected = cmp;\n"
           "  }\n"
           "  return expected;\n"
           "}\n";
  if (Name == "crucible_atomic_cmpxchg_threadgroup_i64")
    return "static inline ulong crucible_atomic_cmpxchg(threadgroup "
           "atomic_ulong *p, ulong cmp, ulong val) {\n"
           "  ulong expected = cmp;\n"
           "  while (!atomic_compare_exchange_weak_explicit(\n"
           "      p, &expected, val, memory_order_relaxed, "
           "memory_order_relaxed)) {\n"
           "    if (expected != cmp) break;\n"
           "    expected = cmp;\n"
           "  }\n"
           "  return expected;\n"
           "}\n";
  if (Name == "crucible_atomic_imin_device_i64")
    return "static inline ulong crucible_atomic_imin(device atomic_ulong *p, "
           "ulong val) {\n"
           "  ulong expected = atomic_load_explicit(p, memory_order_relaxed);\n"
           "  while (long(val) < long(expected)) {\n"
           "    if (atomic_compare_exchange_weak_explicit(\n"
           "            p, &expected, val, memory_order_relaxed, "
           "memory_order_relaxed))\n"
           "      break;\n"
           "  }\n"
           "  return expected;\n"
           "}\n";
  if (Name == "crucible_atomic_imin_threadgroup_i64")
    return "static inline ulong crucible_atomic_imin(threadgroup "
           "atomic_ulong *p, ulong val) {\n"
           "  ulong expected = atomic_load_explicit(p, memory_order_relaxed);\n"
           "  while (long(val) < long(expected)) {\n"
           "    if (atomic_compare_exchange_weak_explicit(\n"
           "            p, &expected, val, memory_order_relaxed, "
           "memory_order_relaxed))\n"
           "      break;\n"
           "  }\n"
           "  return expected;\n"
           "}\n";
  if (Name == "crucible_atomic_imax_device_i64")
    return "static inline ulong crucible_atomic_imax(device atomic_ulong *p, "
           "ulong val) {\n"
           "  ulong expected = atomic_load_explicit(p, memory_order_relaxed);\n"
           "  while (long(val) > long(expected)) {\n"
           "    if (atomic_compare_exchange_weak_explicit(\n"
           "            p, &expected, val, memory_order_relaxed, "
           "memory_order_relaxed))\n"
           "      break;\n"
           "  }\n"
           "  return expected;\n"
           "}\n";
  if (Name == "crucible_atomic_imax_threadgroup_i64")
    return "static inline ulong crucible_atomic_imax(threadgroup "
           "atomic_ulong *p, ulong val) {\n"
           "  ulong expected = atomic_load_explicit(p, memory_order_relaxed);\n"
           "  while (long(val) > long(expected)) {\n"
           "    if (atomic_compare_exchange_weak_explicit(\n"
           "            p, &expected, val, memory_order_relaxed, "
           "memory_order_relaxed))\n"
           "      break;\n"
           "  }\n"
           "  return expected;\n"
           "}\n";
  if (Name == "crucible_f64")
    return R"MSL(__attribute__((unused)) static inline bool crucible_f64_isnan(ulong a) {
  return (a & 0x7ffffffffffffffful) > 0x7ff0000000000000ul;
}
__attribute__((unused)) static inline bool crucible_f64_isinf(ulong a) {
  return (a & 0x7ffffffffffffffful) == 0x7ff0000000000000ul;
}
__attribute__((unused)) static inline bool crucible_f64_isfinite(ulong a) {
  return (a & 0x7ff0000000000000ul) != 0x7ff0000000000000ul;
}
__attribute__((unused)) static inline ulong crucible_f64_norm_round_pack(bool sign, int exp, ulong sig) {
  if (sig == 0ul) return sign ? 0x8000000000000000ul : 0ul;
  int lz = int(clz(sig));
  if (lz == 0) {
    sig = (sig >> 1) | (sig & 1ul);
    exp += 1;
  } else if (lz > 1) {
    int shift = lz - 1;
    sig <<= shift;
    exp -= shift;
  }
  if (exp <= 0) {
    int shift = 1 - exp;
    if (shift >= 64) {
      sig = 1ul;
    } else {
      ulong sticky = (sig & ((1ul << shift) - 1ul)) != 0ul ? 1ul : 0ul;
      sig = (sig >> shift) | sticky;
    }
    exp = 0;
  }
  ulong round_bits = sig & 0x3fful;
  ulong mant = sig >> 10;
  if (round_bits > 0x200ul || (round_bits == 0x200ul && (mant & 1ul) != 0ul))
    mant += 1ul;
  ulong sbit = sign ? 0x8000000000000000ul : 0ul;
  if (exp > 0) {
    if (mant >= 0x0020000000000000ul) {
      mant >>= 1;
      exp += 1;
    }
    if (exp >= 0x7ff) return sbit | 0x7ff0000000000000ul;
    return sbit | (ulong(exp) << 52) | (mant & 0x000ffffffffffffful);
  }
  return sbit | mant;
}
__attribute__((unused)) static inline ulong crucible_f64_add_mags(ulong a, ulong b, bool sign) {
  int ea = int((a >> 52) & 0x7fful), eb = int((b >> 52) & 0x7fful);
  ulong fa = a & 0x000ffffffffffffful, fb = b & 0x000ffffffffffffful;
  if (ea == 0x7ff) {
    if (fa != 0ul) return a | 0x0008000000000000ul;
    if (eb == 0x7ff && fb != 0ul) return b | 0x0008000000000000ul;
    return (sign ? 0x8000000000000000ul : 0ul) | 0x7ff0000000000000ul;
  }
  if (eb == 0x7ff) {
    if (fb != 0ul) return b | 0x0008000000000000ul;
    return (sign ? 0x8000000000000000ul : 0ul) | 0x7ff0000000000000ul;
  }
  if (ea != 0) fa |= 0x0010000000000000ul; else ea = 1;
  if (eb != 0) fb |= 0x0010000000000000ul; else eb = 1;
  fa <<= 10;
  fb <<= 10;
  if (ea < eb) {
    int te = ea; ea = eb; eb = te;
    ulong tf = fa; fa = fb; fb = tf;
  }
  int diff = ea - eb;
  if (diff > 0) {
    if (diff >= 64) {
      fb = fb != 0ul ? 1ul : 0ul;
    } else {
      ulong sticky = (fb & ((1ul << diff) - 1ul)) != 0ul ? 1ul : 0ul;
      fb = (fb >> diff) | sticky;
    }
  }
  if (fa == 0ul && fb == 0ul) return sign ? 0x8000000000000000ul : 0ul;
  return crucible_f64_norm_round_pack(sign, ea, fa + fb);
}
__attribute__((unused)) static inline ulong crucible_f64_sub_mags(ulong a, ulong b, bool sign) {
  int ea = int((a >> 52) & 0x7fful), eb = int((b >> 52) & 0x7fful);
  ulong fa = a & 0x000ffffffffffffful, fb = b & 0x000ffffffffffffful;
  if (ea == 0x7ff) {
    if (fa != 0ul) return a | 0x0008000000000000ul;
    if (eb == 0x7ff) {
      if (fb != 0ul) return b | 0x0008000000000000ul;
      return 0x7ff8000000000000ul;
    }
    return (sign ? 0x8000000000000000ul : 0ul) | 0x7ff0000000000000ul;
  }
  if (eb == 0x7ff) {
    if (fb != 0ul) return b | 0x0008000000000000ul;
    return (!sign ? 0x8000000000000000ul : 0ul) | 0x7ff0000000000000ul;
  }
  if (ea != 0) fa |= 0x0010000000000000ul; else ea = 1;
  if (eb != 0) fb |= 0x0010000000000000ul; else eb = 1;
  fa <<= 10;
  fb <<= 10;
  if (ea < eb || (ea == eb && fa < fb)) {
    int te = ea; ea = eb; eb = te;
    ulong tf = fa; fa = fb; fb = tf;
    sign = !sign;
  }
  int diff = ea - eb;
  if (diff > 0) {
    if (diff >= 64) {
      fb = fb != 0ul ? 1ul : 0ul;
    } else {
      ulong sticky = (fb & ((1ul << diff) - 1ul)) != 0ul ? 1ul : 0ul;
      fb = (fb >> diff) | sticky;
    }
  }
  if (fa == fb) return 0ul;
  return crucible_f64_norm_round_pack(sign, ea, fa - fb);
}
__attribute__((unused)) static inline ulong crucible_f64_add(ulong a, ulong b) {
  bool sa = (a >> 63) != 0ul, sb = (b >> 63) != 0ul;
  return sa == sb ? crucible_f64_add_mags(a, b, sa)
                  : crucible_f64_sub_mags(a, b, sa);
}
__attribute__((unused)) static inline ulong crucible_f64_sub(ulong a, ulong b) {
  return crucible_f64_add(a, b ^ 0x8000000000000000ul);
}
__attribute__((unused)) static inline ulong crucible_f64_mul(ulong a, ulong b) {
  bool sign = ((a ^ b) >> 63) != 0ul;
  int ea = int((a >> 52) & 0x7fful), eb = int((b >> 52) & 0x7fful);
  ulong fa = a & 0x000ffffffffffffful, fb = b & 0x000ffffffffffffful;
  if (ea == 0x7ff) {
    if (fa != 0ul) return a | 0x0008000000000000ul;
    if (eb == 0x7ff && fb != 0ul) return b | 0x0008000000000000ul;
    if ((b & 0x7ffffffffffffffful) == 0ul) return 0x7ff8000000000000ul;
    return (sign ? 0x8000000000000000ul : 0ul) | 0x7ff0000000000000ul;
  }
  if (eb == 0x7ff) {
    if (fb != 0ul) return b | 0x0008000000000000ul;
    if ((a & 0x7ffffffffffffffful) == 0ul) return 0x7ff8000000000000ul;
    return (sign ? 0x8000000000000000ul : 0ul) | 0x7ff0000000000000ul;
  }
  if ((a & 0x7ffffffffffffffful) == 0ul || (b & 0x7ffffffffffffffful) == 0ul)
    return sign ? 0x8000000000000000ul : 0ul;
  if (ea == 0) {
    int lz = int(clz(fa)) - 11;
    fa <<= lz;
    ea = 1 - lz;
  } else {
    fa |= 0x0010000000000000ul;
  }
  if (eb == 0) {
    int lz = int(clz(fb)) - 11;
    fb <<= lz;
    eb = 1 - lz;
  } else {
    fb |= 0x0010000000000000ul;
  }
  fa <<= 10;
  fb <<= 11;
  ulong hi = mulhi(fa, fb);
  ulong lo = fa * fb;
  return crucible_f64_norm_round_pack(sign, ea + eb - 1022,
                                      hi | (lo != 0ul ? 1ul : 0ul));
}
__attribute__((unused)) static inline ulong crucible_f64_fma(ulong a, ulong b, ulong c) {
  bool sp = ((a ^ b) >> 63) != 0ul, sc = (c >> 63) != 0ul;
  int ea = int((a >> 52) & 0x7fful), eb = int((b >> 52) & 0x7fful),
      ec = int((c >> 52) & 0x7fful);
  ulong fa = a & 0x000ffffffffffffful, fb = b & 0x000ffffffffffffful,
        fc = c & 0x000ffffffffffffful;
  if (ea == 0x7ff && fa != 0ul) return a | 0x0008000000000000ul;
  if (eb == 0x7ff && fb != 0ul) return b | 0x0008000000000000ul;
  if (ec == 0x7ff && fc != 0ul) return c | 0x0008000000000000ul;
  if (ea == 0x7ff || eb == 0x7ff) {
    if ((a & 0x7ffffffffffffffful) == 0ul || (b & 0x7ffffffffffffffful) == 0ul)
      return 0x7ff8000000000000ul;
    if (ec == 0x7ff && sp != sc) return 0x7ff8000000000000ul;
    return (sp ? 0x8000000000000000ul : 0ul) | 0x7ff0000000000000ul;
  }
  if (ec == 0x7ff) return c;
  if ((a & 0x7ffffffffffffffful) == 0ul || (b & 0x7ffffffffffffffful) == 0ul) {
    if ((c & 0x7ffffffffffffffful) == 0ul)
      return (sp && sc) ? 0x8000000000000000ul : 0ul;
    return c;
  }
  if (ea == 0) {
    int lz = int(clz(fa)) - 11; fa <<= lz; ea = 1 - lz;
  } else { fa |= 0x0010000000000000ul; }
  if (eb == 0) {
    int lz = int(clz(fb)) - 11; fb <<= lz; eb = 1 - lz;
  } else { fb |= 0x0010000000000000ul; }
  fa <<= 10;
  fb <<= 11;
  ulong p_hi = mulhi(fa, fb);
  ulong p_lo = fa * fb;
  int p_exp = ea + eb - 1022;
  if ((p_hi & 0x4000000000000000ul) == 0ul) {
    p_hi = (p_hi << 1) | (p_lo >> 63);
    p_lo <<= 1;
    p_exp -= 1;
  }
  if ((c & 0x7ffffffffffffffful) == 0ul)
    return crucible_f64_norm_round_pack(sp, p_exp,
                                        p_hi | (p_lo != 0ul ? 1ul : 0ul));
  if (ec == 0) {
    int lz = int(clz(fc)) - 11; fc <<= lz; ec = 1 - lz;
  } else { fc |= 0x0010000000000000ul; }
  ulong c_hi = fc << 10;
  ulong c_lo = 0ul;
  int c_exp = ec;
  bool sign = sp;
  if (sp != sc && p_exp == c_exp && p_hi == c_hi && p_lo == c_lo)
    return 0ul;
  if (p_exp < c_exp ||
      (p_exp == c_exp && (p_hi < c_hi || (p_hi == c_hi && p_lo < c_lo)))) {
    int te = p_exp; p_exp = c_exp; c_exp = te;
    ulong th = p_hi; p_hi = c_hi; c_hi = th;
    ulong tl = p_lo; p_lo = c_lo; c_lo = tl;
    sign = sc;
  }
  int diff = p_exp - c_exp;
  if (diff > 0) {
    if (diff >= 128) {
      c_lo = (c_hi != 0ul || c_lo != 0ul) ? 1ul : 0ul;
      c_hi = 0ul;
    } else if (diff >= 64) {
      int s = diff - 64;
      ulong sticky = (c_lo != 0ul || (s > 0 && (c_hi & ((1ul << s) - 1ul)) != 0ul)) ? 1ul : 0ul;
      c_lo = (c_hi >> s) | sticky;
      c_hi = 0ul;
    } else {
      ulong sticky = (c_lo & ((1ul << diff) - 1ul)) != 0ul ? 1ul : 0ul;
      c_lo = (c_hi << (64 - diff)) | (c_lo >> diff) | sticky;
      c_hi >>= diff;
    }
  }
  ulong r_hi, r_lo;
  if (sp == sc) {
    r_lo = p_lo + c_lo;
    r_hi = p_hi + c_hi + (r_lo < p_lo ? 1ul : 0ul);
  } else {
    r_lo = p_lo - c_lo;
    r_hi = p_hi - c_hi - (p_lo < c_lo ? 1ul : 0ul);
  }
  if (r_hi == 0ul && r_lo == 0ul) return 0ul;
  if (r_hi == 0ul) {
    int lz = int(clz(r_lo));
    r_hi = r_lo << lz;
    r_lo = 0ul;
    p_exp -= 64 + lz - 1;
    // Place leading 1 at bit 62
    r_hi = (r_hi >> 1);
  }
  return crucible_f64_norm_round_pack(sign, p_exp,
                                      r_hi | (r_lo != 0ul ? 1ul : 0ul));
}
__attribute__((unused)) static inline ulong crucible_f64_div(ulong a, ulong b) {
  bool sign = ((a ^ b) >> 63) != 0ul;
  int ea = int((a >> 52) & 0x7fful), eb = int((b >> 52) & 0x7fful);
  ulong fa = a & 0x000ffffffffffffful, fb = b & 0x000ffffffffffffful;
  if (ea == 0x7ff && fa != 0ul) return a | 0x0008000000000000ul;
  if (eb == 0x7ff && fb != 0ul) return b | 0x0008000000000000ul;
  if (ea == 0x7ff) {
    if (eb == 0x7ff) return 0x7ff8000000000000ul;
    return (sign ? 0x8000000000000000ul : 0ul) | 0x7ff0000000000000ul;
  }
  if (eb == 0x7ff) return sign ? 0x8000000000000000ul : 0ul;
  if ((b & 0x7ffffffffffffffful) == 0ul) {
    if ((a & 0x7ffffffffffffffful) == 0ul) return 0x7ff8000000000000ul;
    return (sign ? 0x8000000000000000ul : 0ul) | 0x7ff0000000000000ul;
  }
  if ((a & 0x7ffffffffffffffful) == 0ul)
    return sign ? 0x8000000000000000ul : 0ul;
  if (ea == 0) {
    int lz = int(clz(fa)) - 11; fa <<= lz; ea = 1 - lz;
  } else { fa |= 0x0010000000000000ul; }
  if (eb == 0) {
    int lz = int(clz(fb)) - 11; fb <<= lz; eb = 1 - lz;
  } else { fb |= 0x0010000000000000ul; }
  int exp = ea - eb + 1023;
  if (fa < fb) {
    fa <<= 1;
    exp -= 1;
  }
  ulong q = 0ul;
  for (int i = 0; i < 55; ++i) {
    q <<= 1;
    if (fa >= fb) {
      fa -= fb;
      q |= 1ul;
    }
    fa <<= 1;
  }
  return crucible_f64_norm_round_pack(sign, exp, (q << 8) | (fa != 0ul ? 1ul : 0ul));
}
__attribute__((unused)) static inline ulong crucible_f64_sqrt(ulong a) {
  int ea = int((a >> 52) & 0x7fful);
  ulong fa = a & 0x000ffffffffffffful;
  if (ea == 0x7ff && fa != 0ul) return a | 0x0008000000000000ul;
  if ((a & 0x7ffffffffffffffful) == 0ul) return a;
  if ((a >> 63) != 0ul) return 0x7ff8000000000000ul;
  if (ea == 0x7ff) return a;
  if (ea == 0) {
    int lz = int(clz(fa)) - 11;
    fa <<= lz;
    ea = 1 - lz;
  } else {
    fa |= 0x0010000000000000ul;
  }
  int eu = ea - 1023;
  int er = eu & 1;
  int eh = (eu - er) / 2;
  if (er != 0) fa <<= 1;
  ulong r_hi = fa << 10;
  ulong rem_hi = 0ul, rem_lo = 0ul, q = 0ul;
  for (int i = 0; i < 55; ++i) {
    rem_hi = (rem_hi << 2) | (rem_lo >> 62);
    rem_lo = (rem_lo << 2) | (r_hi >> 62);
    r_hi <<= 2;
    ulong trial = (q << 2) | 1ul;
    if (rem_hi > 0ul || rem_lo >= trial) {
      if (rem_lo < trial) rem_hi -= 1ul;
      rem_lo -= trial;
      q = (q << 1) | 1ul;
    } else {
      q <<= 1;
    }
  }
  return crucible_f64_norm_round_pack(false, eh + 1023,
                                      (q << 8) | ((rem_hi != 0ul || rem_lo != 0ul) ? 1ul : 0ul));
}
__attribute__((unused)) static inline bool crucible_f64_cmp_eq(ulong a, ulong b) {
  if (crucible_f64_isnan(a) || crucible_f64_isnan(b)) return false;
  if (((a | b) & 0x7ffffffffffffffful) == 0ul) return true;
  return a == b;
}
__attribute__((unused)) static inline bool crucible_f64_cmp_lt(ulong a, ulong b) {
  if (crucible_f64_isnan(a) || crucible_f64_isnan(b)) return false;
  if (((a | b) & 0x7ffffffffffffffful) == 0ul) return false;
  bool sa = (a >> 63) != 0ul, sb = (b >> 63) != 0ul;
  if (sa != sb) return sa;
  return sa ? (a > b) : (a < b);
}
__attribute__((unused)) static inline bool crucible_f64_cmp_le(ulong a, ulong b) {
  if (crucible_f64_isnan(a) || crucible_f64_isnan(b)) return false;
  return crucible_f64_cmp_lt(a, b) || crucible_f64_cmp_eq(a, b);
}
__attribute__((unused)) static inline ulong crucible_f64_min(ulong a, ulong b) {
  if (crucible_f64_isnan(a)) return b;
  if (crucible_f64_isnan(b)) return a;
  if (((a | b) & 0x7ffffffffffffffful) == 0ul) return a | b;
  return crucible_f64_cmp_lt(a, b) ? a : b;
}
__attribute__((unused)) static inline ulong crucible_f64_max(ulong a, ulong b) {
  if (crucible_f64_isnan(a)) return b;
  if (crucible_f64_isnan(b)) return a;
  if (((a | b) & 0x7ffffffffffffffful) == 0ul) return a & b;
  return crucible_f64_cmp_lt(a, b) ? b : a;
}
__attribute__((unused)) static inline ulong crucible_f64_round(ulong a, int mode) {
  int ea = int((a >> 52) & 0x7fful);
  if (ea >= 1075) return a;
  ulong sbit = a & 0x8000000000000000ul;
  bool sign = sbit != 0ul;
  if (ea < 1023) {
    if ((a & 0x7ffffffffffffffful) == 0ul) return a;
    if (mode == 0) return sbit;
    if (mode == 1) return sign ? 0xbff0000000000000ul : 0ul;
    if (mode == 2) return sign ? 0x8000000000000000ul : 0x3ff0000000000000ul;
    ulong mag = a & 0x7ffffffffffffffful;
    return mag > 0x3fe0000000000000ul ? (sbit | 0x3ff0000000000000ul) : sbit;
  }
  int fbits = 52 - (ea - 1023);
  ulong mask = (1ul << fbits) - 1ul;
  ulong frac = a & mask;
  if (frac == 0ul) return a;
  ulong res = a & ~mask;
  ulong inc = 1ul << fbits;
  if ((mode == 1 && sign) || (mode == 2 && !sign)) {
    res += inc;
  } else if (mode == 3) {
    ulong half_bit = 1ul << (fbits - 1);
    if (frac > half_bit || (frac == half_bit && (res & inc) != 0ul)) res += inc;
  }
  return res;
}
__attribute__((unused)) static inline ulong crucible_f32_to_f64(float f) {
  uint u = as_type<uint>(f);
  ulong sbit = ulong(u & 0x80000000u) << 32;
  uint e = (u >> 23) & 0xffu;
  uint frac = u & 0x7fffffu;
  if (e == 0xffu) {
    if (frac != 0u) return sbit | 0x7ff8000000000000ul | (ulong(frac) << 29);
    return sbit | 0x7ff0000000000000ul;
  }
  if (e == 0u) {
    if (frac == 0u) return sbit;
    int lz = int(clz(frac)) - 8;
    frac <<= lz;
    return sbit | (ulong(1023 - 126 - lz) << 52) | (ulong(frac & 0x7fffffu) << 29);
  }
  return sbit | (ulong(e - 127u + 1023u) << 52) | (ulong(frac) << 29);
}
__attribute__((unused)) static inline float crucible_f64_to_f32(ulong a) {
  uint sbit = uint((a >> 32) & 0x80000000ul);
  int ea = int((a >> 52) & 0x7fful);
  ulong fa = a & 0x000ffffffffffffful;
  if (ea == 0x7ff) {
    if (fa != 0ul) return as_type<float>(sbit | 0x7fc00000u | uint(fa >> 29));
    return as_type<float>(sbit | 0x7f800000u);
  }
  if ((a & 0x7ffffffffffffffful) == 0ul) return as_type<float>(sbit);
  if (ea != 0) fa |= 0x0010000000000000ul; else ea = 1;
  int e32 = ea - 1023 + 127;
  if (e32 >= 255) return as_type<float>(sbit | 0x7f800000u);
  // Align bit 52 to bit 30 (7 rounding bits at 0..6, bit 23+7=30)
  ulong sig = fa >> 22;
  if ((fa & 0x3ffffful) != 0ul) sig |= 1ul;
  if (e32 <= 0) {
    int shift = 1 - e32;
    if (shift >= 32) return as_type<float>(sbit);
    ulong sticky = (sig & ((1ul << shift) - 1ul)) != 0ul ? 1ul : 0ul;
    sig = (sig >> shift) | sticky;
    e32 = 0;
  }
  uint rb = uint(sig & 0x7ful);
  uint mant = uint(sig >> 7);
  if (rb > 0x40u || (rb == 0x40u && (mant & 1u) != 0u)) mant += 1u;
  if (e32 > 0) {
    if (mant >= 0x01000000u) { mant >>= 1; e32 += 1; }
    if (e32 >= 255) return as_type<float>(sbit | 0x7f800000u);
    return as_type<float>(sbit | (uint(e32) << 23) | (mant & 0x7fffffu));
  }
  return as_type<float>(sbit | mant);
}
__attribute__((unused)) static inline ulong crucible_u64_to_f64(ulong u) {
  if (u == 0ul) return 0ul;
  int lz = int(clz(u));
  int exp = 1023 + 63 - lz;
  ulong sig = (lz == 0) ? ((u >> 1) | (u & 1ul)) : (u << (lz - 1));
  if (lz == 0) exp = 1023 + 63; // bit 62 corresponds to 2^63 when shifted right 1
  return crucible_f64_norm_round_pack(false, exp, sig);
}
__attribute__((unused)) static inline ulong crucible_i64_to_f64(long s) {
  if (s == 0l) return 0ul;
  bool sign = s < 0l;
  ulong u = sign ? ulong(-s) : ulong(s);
  int lz = int(clz(u));
  int exp = 1023 + 63 - lz;
  ulong sig = (lz == 0) ? ((u >> 1) | (u & 1ul)) : (u << (lz - 1));
  return crucible_f64_norm_round_pack(sign, exp, sig);
}
__attribute__((unused)) static inline ulong crucible_f64_to_u64(ulong a) {
  if ((a >> 63) != 0ul) return 0ul;
  int ea = int((a >> 52) & 0x7fful);
  if (ea < 1023) return 0ul;
  if (ea >= 1023 + 64) return 0xfffffffffffffffful;
  ulong fa = (a & 0x000ffffffffffffful) | 0x0010000000000000ul;
  int shift = ea - 1023 - 52;
  return shift >= 0 ? (fa << shift) : (fa >> (-shift));
}
__attribute__((unused)) static inline long crucible_f64_to_i64(ulong a) {
  bool sign = (a >> 63) != 0ul;
  int ea = int((a >> 52) & 0x7fful);
  if (ea < 1023) return 0l;
  if (ea >= 1023 + 63)
    return sign ? long(0x8000000000000000ul) : long(0x7ffffffffffffffful);
  ulong fa = (a & 0x000ffffffffffffful) | 0x0010000000000000ul;
  int shift = ea - 1023 - 52;
  ulong mag = shift >= 0 ? (fa << shift) : (fa >> (-shift));
  return sign ? -long(mag) : long(mag);
}
)MSL";
  return {};
}

std::string sanitizeIdentifier(StringRef Name, StringRef Fallback) {
  std::string Out;
  for (char C : Name)
    Out += (isalnum(static_cast<unsigned char>(C)) || C == '_') ? C : '_';
  if (Out.empty() || isdigit(static_cast<unsigned char>(Out[0])))
    return Fallback.str();
  // Reserved in MSL.
  static const char *Reserved[] = {"main",     "kernel", "vertex",
                                   "fragment", "constant", "device",
                                   "thread",   "threadgroup"};
  for (const char *R : Reserved)
    if (Out == R)
      return (Out + "0");
  return Out;
}

std::string formatFloatLiteral(const APFloat &APF, bool IsHalf) {
  if (APF.isInfinity())
    return std::string(APF.isNegative() ? "-" : "") +
           (IsHalf ? "half(HUGE_VALF)" : "HUGE_VALF");
  if (APF.isNaN())
    return IsHalf ? "half(NAN)" : "NAN";
  SmallString<32> Str;
  APFloat V = APF;
  bool LosesInfo = false;
  V.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &LosesInfo);
  double D = V.convertToDouble();
  raw_svector_ostream OS(Str);
  OS << format("%.9g", D);
  std::string S(Str.str());
  if (S.find('.') == std::string::npos && S.find('e') == std::string::npos)
    S += ".0";
  S += IsHalf ? "h" : "f";
  return S;
}

std::optional<uint64_t> constInt(mlir::Value V) {
  mlir::Operation *Def = V.getDefiningOp();
  if (!Def)
    return std::nullopt;
  if (auto C = mlir::dyn_cast<mlir::LLVM::ConstantOp>(Def)) {
    if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue()))
      return IA.getValue().getZExtValue();
  } else if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(Def)) {
    if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue()))
      return IA.getValue().getZExtValue();
  }
  return std::nullopt;
}

bool isUndefLike(mlir::Value V) {
  mlir::Operation *Def = V.getDefiningOp();
  return Def && mlir::isa<mlir::LLVM::UndefOp, mlir::LLVM::PoisonOp>(Def);
}

bool isConstantLike(mlir::Operation *Op) {
  return mlir::isa<mlir::LLVM::ConstantOp, mlir::arith::ConstantOp,
                   mlir::LLVM::UndefOp, mlir::LLVM::PoisonOp,
                   mlir::LLVM::ZeroOp, mlir::ub::PoisonOp>(Op);
}

//===----------------------------------------------------------------------===//
// Emitter
//===----------------------------------------------------------------------===//

class Emitter {
public:
  Emitter(mlir::func::FuncOp Entry, Reflection &R, const Options &Opts)
      : Entry(Entry), R(R), Opts(Opts) {}

  Expected<std::string> run();

private:
  mlir::func::FuncOp Entry;
  Reflection &R;
  const Options &Opts;

  struct ResState {
    ResourceBinding *RB = nullptr;
    std::string ParamName;
    std::string ElemType; // scalar MSL type for buffers; row type for CBVs
    bool IsSampled = false;
    bool IsDepthSampled = false;
  };

  struct HandleRef {
    ResState *RS = nullptr;
    mlir::Value RegIdx;
    bool IsHeap = false;
    bool IsSamplerHeap = false;
  };

  std::vector<ResState> ResStates; // parallel to R.Resources
  struct KernelPtrArg {
    std::string Name;
    std::string ElemType;
    uint32_t BufferIndex = 0;
  };
  std::vector<KernelPtrArg> KernelPtrArgs; // Layer 0 addrspace(1) ptr args
  std::deque<ResourceBinding> HeapBindings;
  std::deque<ResState> HeapStates;
  DenseMap<mlir::Operation *, HandleRef> HandleMap; // handle def -> resource
  DenseMap<mlir::Value, std::string> ValueNames;    // declared locals
  unsigned NextTemp = 0;
  bool UsesDTid = false, UsesGTid = false, UsesGid = false, UsesGidx = false;
  bool UsesLane = false, UsesLaneCount = false;
  bool UsesRaytracing = false;
  std::string MeshPayloadMSLType;
  bool UsesGridSize = false;
  // MSL helper functions required by the emitted code, in emission order.
  std::vector<std::string> HelperOrder;
  llvm::StringSet<> HelperSet;
  std::string GroupsharedDecls; // threadgroup declarations (kernel prologue)
  std::string ConstantGlobalDecls;
  std::string StructDecls;
  std::string RTFuncsDecl;
  DenseMap<mlir::Type, std::string> StructNames;
  struct EmittedRTFunc {
    std::string MSLName;
    mlir::Type PayloadTy;
    mlir::Type AttrTy;
    uint32_t Kind = 0;
  };
  llvm::StringMap<EmittedRTFunc> RTFuncMap;

  void requireHelper(StringRef Name) {
    if (HelperSet.insert(Name).second)
      HelperOrder.push_back(Name.str());
  }

  std::string Body;

  void line(unsigned Ind, const Twine &T) {
    Body.append(2 * Ind, ' ');
    Body += T.str();
    Body += '\n';
  }
  std::string freshName() { return "v" + std::to_string(NextTemp++); }

  // --- types & constants ---
  Expected<std::string> ensureStruct(mlir::LLVM::LLVMStructType ST);
  Expected<std::string> aggregateMemberPath(mlir::Type Ty,
                                             ArrayRef<int64_t> Indices);
  Expected<std::string> mslType(mlir::Type T);
  Expected<Expr> constant(mlir::Operation *Def);

  // --- resource discovery ---
  Error resolveHandles();
  ResState *stateFor(mlir::Value Handle) {
    auto It = HandleMap.find(Handle.getDefiningOp());
    return It == HandleMap.end() ? nullptr : It->second.RS;
  }
  Expected<std::string> resourceExpr(mlir::Value Handle,
                                     StringRef Suffix = "");
  Error inferElementTypes();

  // --- expressions ---
  Expected<Expr> exprFor(mlir::Value V, int ParentPrec);
  Expected<Expr> opExpr(mlir::Operation *Op);
  Expected<Expr> bufferElementExpr(mlir::LLVM::ExtractValueOp EV);
  Expected<Expr> crucibleExtractExpr(mlir::Operation *Producer, unsigned Comp,
                                     mlir::Type ElemTy);
  Expected<std::string> bufferIndexExpr(ResState &RS, mlir::Value Coord0,
                                        mlir::Value Coord1, unsigned Comp);
  Expected<std::string> bufferDimExpr(ResState &RS, mlir::Value Handle);
  Expected<std::string> bufferBoundExpr(ResState &RS, mlir::Value Handle);
  /// Byte size of one scalar element of an MSL type name.
  static unsigned elemByteSize(StringRef MSLTy) {
    if (MSLTy == "uchar")
      return 1;
    if (MSLTy == "half" || MSLTy == "ushort")
      return 2;
    if (MSLTy == "ulong")
      return 8;
    return 4;
  }
  bool shouldInline(mlir::Operation *Op);
  bool canVectorizeBufferAccess(ResState &RS, mlir::Value Coord0,
                                mlir::Value Coord1, unsigned Width);
  DenseMap<mlir::Operation *, mlir::Operation *> SinCosPartner;

  // --- pointers (groupshared / local arrays) ---
  Error collectGroupshared();
  Error collectConstantGlobals();
  Expected<Expr> scalarConstant(mlir::Attribute ValAttr);
  /// Render a pointer-typed SSA value as an MSL lvalue (e.g. "gs[idx]").
  /// When the pointer is a bare array base and DecayArray is set, the first
  /// element is addressed (C array-to-pointer decay in reverse).
  Expected<Expr> lvalueFor(mlir::Value Ptr, bool DecayArray = true);
  /// MSL address space of a pointer value ("threadgroup", "device", "thread").
  Expected<std::string> addrSpaceOf(mlir::Value Ptr);
  DenseMap<mlir::Operation *, std::string> GlobalNames; // llvm.mlir.global
  DenseMap<mlir::Operation *, std::string> GlobalTypes; // element MSL type

  // --- statements ---
  Error emitBlock(mlir::Block &B, unsigned Ind,
                  mlir::Operation *StopBefore = nullptr);
  Error emitSwitch(mlir::scf::IndexSwitchOp Switch, unsigned Ind);
  Error emitStatement(mlir::Operation *Op, unsigned Ind);
  Error emitIf(mlir::scf::IfOp If, unsigned Ind);
  Error emitWhile(mlir::scf::WhileOp While, unsigned Ind);
  Error emitBufferStore(ir::BufferStoreOp Store, unsigned Ind);
  Error emitTextureLoad(ir::TexLoadOp Load, unsigned Ind);
  Error emitTextureStore(ir::TexStoreOp Store, unsigned Ind);
  Error emitTextureSample(ir::TexSampleOp Sample, unsigned Ind);
  Error emitTextureGather(ir::TexGatherOp Gather, unsigned Ind);
  Error emitAtomic(mlir::Operation *Op, unsigned Ind);
  Error emitTemp(mlir::Operation *Op, unsigned Ind);
  Expected<std::string> localDeclaration(mlir::Type Ty, StringRef Name);
  void emitAggregateCopy(mlir::Type Ty, const std::string &Dst,
                         const std::string &Src, unsigned Ind);

  /// Guarantee V is referable by a stable name (or literal) and return it.
  Expected<std::string> ensureNamed(mlir::Value V, unsigned Ind,
                                    bool CopyBlockArgs = false);

  Expected<std::string> kernelSignature();
  Expected<std::string> stageStruct(bool Input);
  Expected<std::string> stageElement(uint32_t ID, mlir::Value RowVal,
                                     uint32_t Col, bool Input);
  Expected<Expr> domainLoad(uint32_t ID, mlir::Value RowVal, uint32_t Col,
                            mlir::Value CPVal, mlir::Type ResTy,
                            bool IsConstant);
  Expected<Expr> hullLoad(uint32_t ID, mlir::Value RowVal, uint32_t Col,
                          mlir::Value CPVal, mlir::Type ResTy, bool IsOutputCP);
  Error hullStore(uint32_t ID, mlir::Value RowVal, uint32_t Col,
                  mlir::Value Val, bool IsPatchConstant, unsigned Ind);
  Expected<Expr> geometryLoad(uint32_t ID, mlir::Value RowVal, uint32_t Col,
                              mlir::Value VtxVal, mlir::Type ResTy);
  Error geometryEmit(unsigned Ind, bool Emit, bool Cut);
};

namespace {
enum SemanticKind : unsigned {
  UserSemantic = 0, VertexID = 1, InstanceID = 2, Position = 3,
  ViewportArrayIndex = 5, ClipDistance = 6, CullDistance = 7, PrimitiveID = 10,
  FrontFace = 13, ColorTarget = 16, Depth = 17, DepthLess = 18,
  DepthGreater = 19, StencilRef = 20, TessFactor = 25, InsideTessFactor = 26,
  Barycentrics = 28,
};

unsigned clipDistanceOffset(ArrayRef<StageParameter> Params) {
  unsigned Offset = 0;
  for (const auto &P : Params)
    if (P.Kind == ClipDistance || P.Kind == CullDistance)
      Offset += P.Rows * P.Columns;
  return Offset;
}

std::string varyingName(const StageParameter &P, unsigned Row) {
  return "crucible_" + toHex(StringRef(P.Semantic).lower(), true) + "_" +
         std::to_string(P.RowSemanticIndices[Row]);
}

std::string signatureField(const StageParameter &P, unsigned Row) {
  std::string Name = "sig" + std::to_string(P.ID);
  if (P.Rows > 1)
    Name += "_" + std::to_string(Row);
  return Name;
}
} // namespace

Expected<std::string> Emitter::stageStruct(bool Input) {
  bool Vertex = R.Stage == ShaderStage::Vertex ||
                R.Stage == ShaderStage::Mesh || R.Domain.has_value() ||
                R.Geometry.has_value();
  std::string Text;
  raw_string_ostream OS(Text);
  OS << "struct " << (Input ? "CrucibleStageIn" : "CrucibleStageOut")
     << " {\n";
  unsigned Count = 0;
  const auto &Params = Input ? R.Inputs : R.Outputs;
  for (const StageParameter &P : Params) {
    if (Input && ((Vertex && (P.Kind == VertexID || P.Kind == InstanceID)) ||
                  (!Vertex && (P.Kind == FrontFace || P.Kind == PrimitiveID ||
                               P.Kind == Barycentrics))))
      continue;
    if (Vertex && !Input &&
        (P.Kind == ClipDistance || P.Kind == CullDistance)) {
      if (P.Type != ComponentType::F32 || !P.Rows || P.Rows > 8 ||
          !P.Columns || P.Columns > 4)
        return makeError("invalid DXIL clip-distance signature");
      continue;
    }
    bool UserVarying = Vertex != Input && P.Kind == UserSemantic;
    if (!P.Rows || (P.Rows != 1 && !UserVarying) ||
        P.Columns < 1 || P.Columns > 4 ||
        (UserVarying && P.RowSemanticIndices.size() != P.Rows))
      return makeError("unsupported stage signature shape for '{0}'", P.Semantic);
    StringRef Type;
    switch (P.Type) {
    case ComponentType::F32: Type = "float"; break;
    case ComponentType::F16: Type = "half"; break;
    case ComponentType::I32: Type = "int"; break;
    case ComponentType::U32: Type = "uint"; break;
    default: return makeError("unsupported stage signature type for '{0}'", P.Semantic);
    }
    std::string Attr;
    if (Vertex && Input && P.Kind == UserSemantic)
      Attr = "attribute(" + std::to_string(P.ID) + ")";
    else if (Vertex != Input && P.Kind == Position &&
             Type == "float" && P.Columns == 4)
      Attr = "position";
    else if (Vertex != Input && P.Kind == ViewportArrayIndex &&
             Type == "uint" && P.Columns == 1)
      Attr = "viewport_array_index";
    else if (!Vertex && !Input && P.Kind == ColorTarget)
      Attr = "color(" + std::to_string(P.SemanticIndex) + ")";
    else if (!Vertex && !Input && Type == "float" && P.Columns == 1 &&
             (P.Kind == Depth || P.Kind == DepthLess || P.Kind == DepthGreater))
      Attr = P.Kind == Depth       ? "depth(any)"
             : P.Kind == DepthLess ? "depth(less)"
                                   : "depth(greater)";
    else if (!Vertex && !Input && Type == "uint" && P.Columns == 1 &&
             P.Kind == StencilRef)
      Attr = "stencil";
    else if (!UserVarying)
      return makeError("unsupported stage semantic '{0}'", P.Semantic);
    for (unsigned Row = 0; Row < P.Rows; ++Row) {
      std::string RowAttr =
          UserVarying ? "user(" + varyingName(P, Row) + ")" : Attr;
      if (UserVarying && Input) {
        static constexpr const char *Modes[] = {
            "", "flat", "center_perspective", "centroid_perspective",
            "center_no_perspective", "centroid_no_perspective",
            "sample_perspective", "sample_no_perspective"};
        if (P.Interpolation >= std::size(Modes))
          return makeError("unsupported interpolation for '{0}'", P.Semantic);
        if (Type == "int" || Type == "uint" || Type == "short" ||
            Type == "ushort")
          RowAttr += ", flat";
        else if (P.Interpolation != 0)
          RowAttr += std::string(", ") + Modes[P.Interpolation];
      }
      OS << "  " << Type;
      if (P.Columns > 1)
        OS << P.Columns;
      OS << " " << signatureField(P, Row) << " [[" << RowAttr << "]];\n";
      ++Count;
    }
  }
  if (Vertex && !Input) {
    unsigned ClipCount = clipDistanceOffset(Params);
    if (ClipCount > 8)
      return makeError("too many DXIL clip-distance components");
    if (ClipCount) {
      OS << "  float clipDistances [[clip_distance]] [" << ClipCount << "];\n";
      ++Count;
    }
  }
  OS << "};\n\n";
  return Count ? Text : std::string();
}

Expected<std::string> Emitter::stageElement(uint32_t ID, mlir::Value RowVal,
                                            uint32_t Col, bool Input) {
  if (R.Stage != ShaderStage::Vertex && R.Stage != ShaderStage::Pixel &&
      !(R.Domain && !Input) && !(R.Geometry && !Input))
    return makeError("graphics signature access outside a graphics stage");
  auto Row = constInt(RowVal);
  const auto &Params = Input ? R.Inputs : R.Outputs;
  auto P = llvm::find_if(Params, [&](const StageParameter &S) {
    return S.ID == ID;
  });
  if (P == Params.end() || !Row || *Row >= P->Rows || Col >= P->Columns)
    return makeError("unsupported signature component access");
  if (Input && R.Stage == ShaderStage::Vertex && P->Kind == VertexID)
    return std::string("crucible_vertex_id");
  if (Input && R.Stage == ShaderStage::Vertex && P->Kind == InstanceID)
    return std::string("crucible_instance_id");
  if (Input && R.Stage == ShaderStage::Pixel && P->Kind == FrontFace)
    return std::string("crucible_front_face");
  if (Input && R.Stage == ShaderStage::Pixel && P->Kind == PrimitiveID)
    return std::string("crucible_primitive_id");
  if (Input && R.Stage == ShaderStage::Pixel && P->Kind == Barycentrics)
    return "crucible_barycentrics[" + std::to_string(Col) + "]";
  if (!Input && (R.Stage == ShaderStage::Vertex || R.Domain) &&
      (P->Kind == ClipDistance || P->Kind == CullDistance)) {
    unsigned Offset = clipDistanceOffset(ArrayRef(Params).take_front(
        std::distance(Params.begin(), P)));
    return "stageOut.clipDistances[" +
           std::to_string(Offset + *Row * P->Columns + Col) + "]";
  }
  std::string Access = std::string(Input ? "stageIn." : "stageOut.") +
                       signatureField(*P, *Row);
  if (P->Columns > 1)
    Access += "[" + std::to_string(Col) + "]";
  return Access;
}

static unsigned signatureWordCount(ArrayRef<StageParameter> Params) {
  unsigned Words = 0;
  for (const auto &P : Params)
    Words += P.Rows * P.Columns;
  return Words;
}

Expected<Expr> Emitter::domainLoad(uint32_t ID, mlir::Value RowVal,
                                   uint32_t Col, mlir::Value CPVal,
                                   mlir::Type ResTy, bool IsConstant) {
  auto Row = constInt(RowVal);
  const auto &Params = IsConstant ? R.Domain->PatchConstants : R.Inputs;
  auto P = llvm::find_if(Params, [&](const StageParameter &S) {
    return S.ID == ID;
  });
  if (P == Params.end() || !Row || *Row >= P->Rows || Col >= P->Columns)
    return makeError("unsupported domain signature component access");
  unsigned CPWords = signatureWordCount(R.Inputs);
  unsigned ConstantBase = CPWords * R.Domain->ControlPointCount;
  unsigned PatchWords = ConstantBase + signatureWordCount(R.Domain->PatchConstants);
  unsigned Offset = signatureWordCount(ArrayRef(Params).take_front(
                        std::distance(Params.begin(), P))) +
                    *Row * P->Columns + Col;
  std::string Index = "crucible_patch_id * " + std::to_string(PatchWords) + "u";
  if (IsConstant) {
    Offset += ConstantBase;
  } else {
    if (!CPVal)
      return makeError("missing domain control-point index");
    if (auto CP = constInt(CPVal); CP && *CP >= R.Domain->ControlPointCount)
      return makeError("domain control-point index out of range");
    auto CP = exprFor(CPVal, PrecMul);
    if (!CP)
      return CP.takeError();
    Index += " + " + CP->Text + " * " + std::to_string(CPWords) + "u";
  }
  if (Offset)
    Index += " + " + std::to_string(Offset) + "u";
  std::string Word = "crucible_patch_data[" + Index + "]";
  if (ResTy.isF32())
    return Expr{"as_type<float>(" + Word + ")", PrecPrimary};
  if (ResTy.isInteger(32))
    return Expr{std::move(Word), PrecPrimary};
  return makeError("domain signature loads require 32-bit components");
}

static unsigned hullTessFactorOffset(ArrayRef<StageParameter> PatchConstants,
                                     const StageParameter &Target,
                                     unsigned Row) {
  // Metal tessellation factor buffer layout:
  //   Triangle: edge[0..2] (3), inside[0] (1) => stride 4
  //   Quad:     edge[0..3] (4), inside[0..1] (2) => stride 6
  //   Isoline:  edge[0..1] (2) => stride 2
  unsigned EdgeCount = 0;
  for (const auto &P : PatchConstants)
    if (P.Kind == TessFactor)
      EdgeCount += P.Rows;
  if (Target.Kind == TessFactor) {
    unsigned Off = 0;
    for (const auto &P : PatchConstants) {
      if (&P == &Target)
        return Off + Row;
      if (P.Kind == TessFactor)
        Off += P.Rows;
    }
  }
  unsigned Off = EdgeCount;
  for (const auto &P : PatchConstants) {
    if (&P == &Target)
      return Off + Row;
    if (P.Kind == InsideTessFactor)
      Off += P.Rows;
  }
  return Off + Row;
}

Expected<Expr> Emitter::hullLoad(uint32_t ID, mlir::Value RowVal, uint32_t Col,
                                 mlir::Value CPVal, mlir::Type ResTy,
                                 bool IsOutputCP) {
  auto Row = constInt(RowVal);
  const auto &Params = IsOutputCP ? R.Outputs : R.Inputs;
  auto P = llvm::find_if(Params, [&](const StageParameter &S) {
    return S.ID == ID;
  });
  if (P == Params.end() || !Row || *Row >= P->Rows || Col >= P->Columns || !CPVal)
    return makeError("unsupported hull signature component access");
  unsigned Offset = signatureWordCount(ArrayRef(Params).take_front(
                        std::distance(Params.begin(), P))) +
                    *Row * P->Columns + Col;
  auto CP = exprFor(CPVal, PrecMul);
  if (!CP)
    return CP.takeError();
  std::string Word;
  if (IsOutputCP) {
    unsigned CPWords = signatureWordCount(R.Outputs);
    unsigned PatchWords =
        CPWords * R.Hull->OutputControlPointCount +
        signatureWordCount(R.Hull->PatchConstants);
    Word = "crucible_patch_data[crucible_patch_id * " +
           std::to_string(PatchWords) + "u + (" + CP->Text + ") * " +
           std::to_string(CPWords) + "u + " + std::to_string(Offset) + "u]";
  } else {
    unsigned InWords = signatureWordCount(R.Inputs);
    unsigned PatchInWords = InWords * R.Hull->InputControlPointCount;
    Word = "crucible_hs_in[crucible_patch_id * " +
           std::to_string(PatchInWords) + "u + (" + CP->Text + ") * " +
           std::to_string(InWords) + "u + " + std::to_string(Offset) + "u]";
  }
  if (ResTy.isF32())
    return Expr{"as_type<float>(" + Word + ")", PrecPrimary};
  if (ResTy.isInteger(32))
    return Expr{std::move(Word), PrecPrimary};
  return makeError("hull signature loads require 32-bit components");
}

Error Emitter::hullStore(uint32_t ID, mlir::Value RowVal, uint32_t Col,
                         mlir::Value ValOp, bool IsPatchConstant,
                         unsigned Ind) {
  auto Row = constInt(RowVal);
  const auto &Params = IsPatchConstant ? R.Hull->PatchConstants : R.Outputs;
  auto P = llvm::find_if(Params, [&](const StageParameter &S) {
    return S.ID == ID;
  });
  if (P == Params.end() || !Row || *Row >= P->Rows || Col >= P->Columns)
    return makeError("unsupported hull signature store");
  auto Val = exprFor(ValOp, PrecUnknown);
  if (!Val)
    return Val.takeError();
  unsigned CPWords = signatureWordCount(R.Outputs);
  unsigned ConstantBase = CPWords * R.Hull->OutputControlPointCount;
  unsigned PatchWords = ConstantBase + signatureWordCount(R.Hull->PatchConstants);
  unsigned Offset = signatureWordCount(ArrayRef(Params).take_front(
                        std::distance(Params.begin(), P))) +
                    *Row * P->Columns + Col;
  std::string Index = "crucible_patch_id * " + std::to_string(PatchWords) + "u";
  if (IsPatchConstant)
    Index += " + " + std::to_string(ConstantBase + Offset) + "u";
  else
    Index += " + crucible_cp_id * " + std::to_string(CPWords) + "u + " +
             std::to_string(Offset) + "u";
  std::string Stored = ValOp.getType().isF32()
                           ? "as_type<uint>(float(" + Val->Text + "))"
                           : "uint(" + Val->Text + ")";
  line(Ind, "crucible_patch_data[" + Index + "] = " + Stored + ";");
  if (IsPatchConstant &&
      (P->Kind == TessFactor || P->Kind == InsideTessFactor)) {
    bool UseFloat = R.Hull->Domain == TessellationDomain::Isoline ||
                    R.Hull->OutputPrimitive == TessellationOutputPrimitive::Point;
    unsigned TFStride =
        R.Hull->Domain == TessellationDomain::Triangle ? 4u
        : R.Hull->Domain == TessellationDomain::Quad   ? 6u
                                                       : 2u;
    unsigned TFOff = hullTessFactorOffset(R.Hull->PatchConstants, *P, *Row);
    line(Ind, "crucible_tess_factors[crucible_patch_id * " +
                  std::to_string(TFStride) + "u + " + std::to_string(TFOff) +
                  "u] = " + (UseFloat ? "float(" : "half(") + Val->Text + ");");
  }
  return Error::success();
}

Expected<Expr> Emitter::geometryLoad(uint32_t ID, mlir::Value RowVal,
                                     uint32_t Col, mlir::Value VtxVal,
                                     mlir::Type ResTy) {
  auto Row = constInt(RowVal);
  auto P = llvm::find_if(R.Inputs, [&](const StageParameter &S) {
    return S.ID == ID;
  });
  if (P == R.Inputs.end() || !Row || *Row >= P->Rows || Col >= P->Columns || !VtxVal)
    return makeError("unsupported geometry signature component access");
  if (P->Kind == PrimitiveID)
    return Expr{"crucible_prim_id", PrecPrimary};
  unsigned InWords = 0;
  unsigned Offset = 0;
  bool Found = false;
  for (const StageParameter &S : R.Inputs) {
    if (S.Kind == PrimitiveID)
      continue;
    if (&S == &*P)
      Found = true;
    unsigned W = S.Rows * S.Columns;
    if (!Found)
      Offset += W;
    InWords += W;
  }
  Offset += *Row * P->Columns + Col;
  unsigned InputVerts =
      R.Geometry->InputPrimitive == 1 ? 1u
      : R.Geometry->InputPrimitive == 2 ? 2u
      : R.Geometry->InputPrimitive == 3 ? 3u
      : R.Geometry->InputPrimitive == 6 ? 4u : 6u;
  auto Vtx = exprFor(VtxVal, PrecMul);
  if (!Vtx)
    return Vtx.takeError();
  std::string Word = "crucible_gs_in[(crucible_prim_id * " +
                     std::to_string(InputVerts) + "u + (" + Vtx->Text +
                     ")) * " + std::to_string(InWords) + "u + " +
                     std::to_string(Offset) + "u]";
  if (ResTy.isF32())
    return Expr{"as_type<float>(" + Word + ")", PrecPrimary};
  if (ResTy.isInteger(32))
    return Expr{std::move(Word), PrecPrimary};
  return makeError("geometry signature loads require 32-bit components");
}

Error Emitter::geometryEmit(unsigned Ind, bool Emit, bool Cut) {
  if (Emit) {
    unsigned OutWords = signatureWordCount(R.Outputs);
    auto EmitVertexTo = [&](const std::string &SrcVar, const std::string &SlotExpr) {
      unsigned WordOff = 0;
      for (const StageParameter &P : R.Outputs) {
        for (uint32_t RIdx = 0; RIdx < P.Rows; ++RIdx) {
          std::string Field = SrcVar + "." + signatureField(P, RIdx);
          for (uint32_t CIdx = 0; CIdx < P.Columns; ++CIdx) {
            std::string Src = P.Columns > 1
                                  ? Field + "[" + std::to_string(CIdx) + "]"
                                  : Field;
            line(Ind, "crucible_gs_out[(" + SlotExpr + ") * " +
                          std::to_string(OutWords) + "u + " +
                          std::to_string(WordOff) + "u] = as_type<uint>(" +
                          Src + ");");
            ++WordOff;
          }
        }
      }
    };
    if (R.Geometry->OutputTopology == 1) {
      std::string Base = freshName();
      line(Ind, "uint " + Base + " = atomic_fetch_add_explicit(crucible_gs_counter, 1u, memory_order_relaxed);");
      EmitVertexTo("stageOut", Base);
    } else if (R.Geometry->OutputTopology == 3) {
      line(Ind, "if (crucible_strip_len >= 1u) {");
      std::string Base = freshName();
      line(Ind + 1, "uint " + Base + " = atomic_fetch_add_explicit(crucible_gs_counter, 2u, memory_order_relaxed);");
      EmitVertexTo("crucible_prev[1]", Base);
      EmitVertexTo("stageOut", Base + " + 1u");
      line(Ind, "}");
      line(Ind, "crucible_prev[1] = stageOut;");
      line(Ind, "++crucible_strip_len;");
    } else {
      line(Ind, "if (crucible_strip_len >= 2u) {");
      std::string Base = freshName();
      line(Ind + 1, "uint " + Base + " = atomic_fetch_add_explicit(crucible_gs_counter, 3u, memory_order_relaxed);");
      line(Ind + 1, "if ((crucible_strip_len & 1u) == 0u) {");
      EmitVertexTo("crucible_prev[0]", Base);
      EmitVertexTo("crucible_prev[1]", Base + " + 1u");
      EmitVertexTo("stageOut", Base + " + 2u");
      line(Ind + 1, "} else {");
      EmitVertexTo("crucible_prev[1]", Base);
      EmitVertexTo("crucible_prev[0]", Base + " + 1u");
      EmitVertexTo("stageOut", Base + " + 2u");
      line(Ind + 1, "}");
      line(Ind, "}");
      line(Ind, "crucible_prev[0] = crucible_prev[1];");
      line(Ind, "crucible_prev[1] = stageOut;");
      line(Ind, "++crucible_strip_len;");
    }
  }
  if (Cut)
    line(Ind, "crucible_strip_len = 0u;");
  return Error::success();
}

//===----------------------------------------------------------------------===//
// Types and constants
//===----------------------------------------------------------------------===//

Expected<std::string> Emitter::ensureStruct(mlir::LLVM::LLVMStructType ST) {
  auto It = StructNames.find(ST);
  if (It != StructNames.end())
    return It->second;
  std::string SName = "CrucibleStruct" + std::to_string(StructNames.size());
  StructNames[ST] = SName;
  std::string Decl = "struct " + SName + " {\n";
  auto BodyTys = ST.getBody();
  for (size_t I = 0; I < BodyTys.size(); ++I) {
    mlir::Type FTy = BodyTys[I];
    std::string ArraySuffix;
    if (auto Arr = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(FTy)) {
      ArraySuffix = "[" + std::to_string(Arr.getNumElements()) + "]";
      FTy = Arr.getElementType();
    }
    auto MTy = mslType(FTy);
    if (!MTy)
      return MTy.takeError();
    Decl += "  " + *MTy + " f" + std::to_string(I) + ArraySuffix + ";\n";
  }
  Decl += "};\n";
  StructDecls += Decl;
  return SName;
}

Expected<std::string> Emitter::aggregateMemberPath(
    mlir::Type Ty, ArrayRef<int64_t> Indices) {
  std::string Path;
  for (int64_t I : Indices) {
    if (auto ST = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(Ty)) {
      if (I < 0 || static_cast<size_t>(I) >= ST.getBody().size())
        return makeError("invalid struct aggregate member index");
      Path += ".f" + std::to_string(I);
      Ty = ST.getBody()[I];
    } else if (auto Arr = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(Ty)) {
      if (I < 0 || static_cast<uint64_t>(I) >= Arr.getNumElements())
        return makeError("invalid array aggregate member index");
      Path += "[" + std::to_string(I) + "]";
      Ty = Arr.getElementType();
    } else {
      return makeError("unsupported aggregate member type");
    }
  }
  return Path;
}

Expected<std::string> Emitter::localDeclaration(mlir::Type Ty, StringRef Name) {
  std::string Suffix;
  while (auto Arr = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(Ty)) {
    Suffix += "[" + std::to_string(Arr.getNumElements()) + "]";
    Ty = Arr.getElementType();
  }
  auto Type = mslType(Ty);
  if (!Type)
    return Type.takeError();
  return *Type + " " + Name.str() + Suffix;
}

void Emitter::emitAggregateCopy(mlir::Type Ty, const std::string &Dst,
                                const std::string &Src, unsigned Ind) {
  if (auto Arr = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(Ty)) {
    auto Index = freshName();
    line(Ind, "for (uint " + Index + " = 0u; " + Index + " < " +
                  std::to_string(Arr.getNumElements()) + "u; ++" + Index +
                  ") {");
    auto Suffix = "[" + Index + "]";
    emitAggregateCopy(Arr.getElementType(), Dst + Suffix, Src + Suffix,
                      Ind + 1);
    line(Ind, "}");
  } else {
    line(Ind, Dst + " = " + Src + ";");
  }
}

Expected<std::string> Emitter::mslType(mlir::Type T) {
  if (auto ST = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(T))
    return ensureStruct(ST);
  if (auto VT = mlir::dyn_cast<mlir::VectorType>(T)) {
    if (VT.getRank() == 1 && VT.getNumElements() >= 2 &&
        VT.getNumElements() <= 4) {
      auto Elem = mslType(VT.getElementType());
      if (!Elem)
        return Elem.takeError();
      return *Elem + std::to_string(VT.getNumElements());
    }
  }
  if (T.isIndex())
    return std::string("long");
  if (T.isF32())
    return std::string("float");
  if (T.isF16())
    return std::string("half");
  if (auto IT = mlir::dyn_cast<mlir::IntegerType>(T)) {
    switch (IT.getWidth()) {
    case 1:
      return std::string("bool");
    case 8:
      return std::string("uchar");
    case 16:
      return std::string("ushort");
    case 32:
      return std::string("uint");
    case 64:
      return std::string("ulong");
    default:
      break;
    }
  }
  if (T.isF64()) {
    if (Opts.EmulateF64)
      return std::string("ulong");
    return makeError("MSL has no double type; use float in the shader");
  }
  std::string S;
  raw_string_ostream OS(S);
  T.print(OS);
  return makeError("unsupported type '{0}'", S);
}

Expected<Expr> Emitter::constant(mlir::Operation *Def) {
  mlir::Attribute ValAttr;
  if (auto C = mlir::dyn_cast<mlir::LLVM::ConstantOp>(Def))
    ValAttr = C.getValue();
  else if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(Def))
    ValAttr = C.getValue();
  else if (mlir::isa<mlir::LLVM::UndefOp, mlir::LLVM::PoisonOp,
                     mlir::LLVM::ZeroOp, mlir::ub::PoisonOp>(Def))
    return Expr{"0", PrecPrimary};
  else
    return makeError("unsupported constant-like op '{0}'",
                     Def->getName().getStringRef());

  return scalarConstant(ValAttr);
}

Expected<Expr> Emitter::scalarConstant(mlir::Attribute ValAttr) {
  if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(ValAttr)) {
    if (IA.getType().isInteger(1))
      return Expr{IA.getValue().isZero() ? "false" : "true", PrecPrimary};
    std::string S = std::to_string(IA.getValue().getZExtValue());
    S += IA.getType().isInteger(64) ? "ul" : "u";
    if (IA.getType().isInteger(8))
      S = "uchar(" + S + ")";
    else if (IA.getType().isInteger(16))
      S = "ushort(" + S + ")";
    return Expr{S, PrecPrimary};
  }
  if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(ValAttr)) {
    if (FA.getType().isF64()) {
      if (!Opts.EmulateF64)
        return makeError("MSL has no double type; use float in the shader");
      uint64_t Bits = FA.getValue().bitcastToAPInt().getZExtValue();
      SmallString<24> Hex;
      raw_svector_ostream OS(Hex);
      OS << "0x";
      OS.write_hex(Bits);
      OS << "ul";
      return Expr{std::string(Hex.str()), PrecPrimary};
    }
    return Expr{formatFloatLiteral(FA.getValue(), FA.getType().isF16()),
                PrecPrimary};
  }
  if (auto DA = mlir::dyn_cast<mlir::DenseElementsAttr>(ValAttr)) {
    if (auto VT = mlir::dyn_cast<mlir::VectorType>(DA.getType());
        VT && VT.getRank() == 1 && VT.getNumElements() >= 2 &&
        VT.getNumElements() <= 4) {
      auto Ty = mslType(VT);
      if (!Ty)
        return Ty.takeError();
      std::string S = *Ty + "(";
      bool First = true;
      for (mlir::Attribute Elem : DA.getValues<mlir::Attribute>()) {
        auto E = scalarConstant(Elem);
        if (!E)
          return E.takeError();
        if (!First)
          S += ", ";
        First = false;
        S += E->Text;
      }
      S += ")";
      return Expr{S, PrecPrimary};
    }
  }
  return makeError("unsupported constant attribute");
}

//===----------------------------------------------------------------------===//
// Resource discovery
//===----------------------------------------------------------------------===//

static bool isTextureKind(ResourceKind K);

Error Emitter::resolveHandles() {
  auto &Rs = R.Resources;
  ResStates.resize(Rs.size());
  for (size_t I = 0; I < Rs.size(); ++I) {
    ResourceBinding &RB = Rs[I];
    const char *Prefix =
        RB.Class == ResourceClass::SRV   ? "srv"
        : RB.Class == ResourceClass::UAV ? "uav"
        : RB.Class == ResourceClass::CBV ? "cb"
                                         : "samp";
    std::string Fallback = (Twine(Prefix) + Twine(RB.RecordID)).str();
    ResStates[I].RB = &RB;
    ResStates[I].ParamName = sanitizeIdentifier(RB.Name, Fallback);
  }

  auto FindBinding = [&](ResourceClass Class, uint32_t RecordID) -> ResState * {
    for (ResState &RS : ResStates)
      if (RS.RB->Class == Class && RS.RB->RecordID == RecordID)
        return &RS;
    return nullptr;
  };
  auto FindByRegister = [&](ResourceClass Class, uint32_t Space,
                            uint32_t Reg) -> ResState * {
    for (ResState &RS : ResStates)
      if (RS.RB->Class == Class && RS.RB->Space == Space &&
          Reg >= RS.RB->LowerBound &&
          Reg < RS.RB->LowerBound + RS.RB->RangeSize)
        return &RS;
    return nullptr;
  };

  Error Result = Error::success();
  std::string FirstError;

  Entry.walk([&](mlir::Operation *Op) {
    if (!FirstError.empty())
      return;
    if (auto BR = mlir::dyn_cast<ir::BindResourceOp>(Op)) {
      ResourceClass RC = static_cast<ResourceClass>(BR.getResourceClass());
      ResState *RS = nullptr;
      if (auto RecID = BR.getRecordId()) {
        RS = FindBinding(RC, static_cast<uint32_t>(*RecID));
      } else if (BR.getSpace() && BR.getBinding()) {
        uint32_t LookupReg = static_cast<uint32_t>(*BR.getBinding());
        if (BR.getIndex())
          if (auto Idx = constInt(BR.getIndex()))
            LookupReg = static_cast<uint32_t>(*Idx);
        RS = FindByRegister(RC, static_cast<uint32_t>(*BR.getSpace()),
                            LookupReg);
      }
      if (!RS) {
        FirstError = "crucible.bind_resource references unknown resource binding";
        return;
      }
      HandleMap[BR] = HandleRef{RS, BR.getIndex()};
      return;
    }
    if (auto HR = mlir::dyn_cast<ir::HeapResourceOp>(Op)) {
      bool IsSamp = HR.getIsSamplerHeap();
      ResourceBinding RB;
      uint32_t Kind = static_cast<uint32_t>(HR.getResourceKind().value_or(0));
      RB.HasCounter = HR.getHasCounter();
      RB.Class = HR.getResourceClass()
                     ? static_cast<ResourceClass>(*HR.getResourceClass())
                     : (IsSamp ? ResourceClass::Sampler : ResourceClass::SRV);
      switch (Kind) {
      case 1: RB.Kind = ResourceKind::Texture1D; break;
      case 2: RB.Kind = ResourceKind::Texture2D; break;
      case 3: RB.Kind = ResourceKind::Texture2DMS; break;
      case 4: RB.Kind = ResourceKind::Texture3D; break;
      case 5: RB.Kind = ResourceKind::TextureCube; break;
      case 6: RB.Kind = ResourceKind::Texture1DArray; break;
      case 7: RB.Kind = ResourceKind::Texture2DArray; break;
      case 9: RB.Kind = ResourceKind::TextureCubeArray; break;
      case 10: RB.Kind = ResourceKind::TypedBuffer; break;
      case 11: RB.Kind = ResourceKind::RawBuffer; break;
      case 12:
        RB.Kind = ResourceKind::StructuredBuffer;
        RB.ElementStride =
            static_cast<uint32_t>(HR.getElementStride().value_or(0));
        break;
      case 13: RB.Kind = ResourceKind::CBuffer; break;
      case 14: RB.Kind = ResourceKind::Sampler; break;
      default:
        if (IsSamp) {
          RB.Kind = ResourceKind::Sampler;
          break;
        }
        FirstError = "unsupported dynamic heap resource kind";
        return;
      }
      HeapBindings.push_back(std::move(RB));
      ResState HS;
      HS.RB = &HeapBindings.back();
      HS.ParamName = "heap";
      if (HS.RB->Class == ResourceClass::CBV)
        HS.ElemType = "uint4";
      HeapStates.push_back(std::move(HS));
      HandleMap[HR] = HandleRef{&HeapStates.back(), HR.getIndex(), true, IsSamp};
      if (IsSamp)
        R.HasSamplerHeap = true;
      else
        R.HasResourceHeap = true;
      if (HeapBindings.back().HasCounter)
        R.HasCounterHeap = true;
      return;
    }
  });
  if (FirstError.empty()) {
    uint32_t NextBuffer = 0;
    for (const ResourceBinding &RB : R.Resources) {
      if (!isTextureKind(RB.Kind) && RB.Class != ResourceClass::Sampler)
        NextBuffer = std::max(NextBuffer, RB.MetalIndex + RB.RangeSize);
      if (RB.HasCounter && RB.CounterMetalIndex != UINT32_MAX)
        NextBuffer = std::max(NextBuffer, RB.CounterMetalIndex + RB.RangeSize);
    }
    if (R.BufferSizesMetalIndex != UINT32_MAX)
      NextBuffer = std::max(NextBuffer, R.BufferSizesMetalIndex + 1);
    if (R.InstanceContributionMetalIndex != UINT32_MAX)
      NextBuffer = std::max(NextBuffer, R.InstanceContributionMetalIndex + 1);
    if (R.HasResourceHeap)
      R.ResourceHeapMetalIndex = NextBuffer++;
    if (R.HasCounterHeap)
      R.CounterHeapMetalIndex = NextBuffer++;
    if (R.HasSamplerHeap)
      R.SamplerHeapMetalIndex = NextBuffer++;
  }
  if (!FirstError.empty())
    return makeError("{0}", FirstError);
  return Error::success();
}

Expected<std::string> Emitter::resourceExpr(mlir::Value Handle,
                                           StringRef Suffix) {
  auto It = HandleMap.find(Handle.getDefiningOp());
  if (It == HandleMap.end())
    return makeError("unresolved resource handle");
  const HandleRef &HR = It->second;
  if (HR.IsHeap) {
    auto E = exprFor(HR.RegIdx, PrecUnknown);
    if (!E)
      return E.takeError();
    if (HR.IsSamplerHeap)
      return "(*((const device sampler*)(crucible_sampler_heap + " + E->Text + ")))";
    if (Suffix == "_counter")
      return "((device atomic_uint*)(crucible_counter_heap[" + E->Text + "]))";
    const ResourceBinding &RB = *HR.RS->RB;
    if (isTextureKind(RB.Kind)) {
      const char *Access = RB.Class == ResourceClass::UAV ? "access::read_write"
                                                          : "access::sample";
      std::string Elem = HR.RS->ElemType.empty() ? "float" : HR.RS->ElemType;
      std::string TexTy =
          RB.Kind == ResourceKind::Texture3D      ? "texture3d"
          : RB.Kind == ResourceKind::TextureCube  ? "texturecube"
          : RB.Kind == ResourceKind::Texture2DArray ? "texture2d_array"
                                                  : "texture2d";
      return "(*((const device " + TexTy + "<" + Elem + ", " + Access +
             ">*)(crucible_res_heap + " + E->Text + ")))";
    }
    const char *PtrPrefix =
        RB.Class == ResourceClass::UAV ? "device " : "device const ";
    std::string Elem = HR.RS->ElemType.empty() ? "uint" : HR.RS->ElemType;
    return "((" + std::string(PtrPrefix) + Elem + "*)(crucible_res_heap[" +
           E->Text + "]))";
  }
  std::string Base = HR.RS->ParamName + Suffix.str();
  if (HR.RS->RB->RangeSize == 1)
    return Base;
  if (auto C = constInt(HR.RegIdx)) {
    uint32_t Offset = static_cast<uint32_t>(*C) - HR.RS->RB->LowerBound;
    return Base + "[" + std::to_string(Offset) + "u]";
  }
  auto E = exprFor(HR.RegIdx, PrecAdd);
  if (!E)
    return E.takeError();
  if (HR.RS->RB->LowerBound == 0)
    return Base + "[" + E->Text + "]";
  return Base + "[" + E->Text + " - " +
         std::to_string(HR.RS->RB->LowerBound) + "u]";
}

static bool isTextureKind(ResourceKind K) {
  switch (K) {
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

static std::string textureElementType(ComponentType CT) {
  switch (CT) {
  case ComponentType::F16:
  case ComponentType::SNormF16:
  case ComponentType::UNormF16:
    return "half";
  case ComponentType::I16:
    return "short";
  case ComponentType::U16:
    return "ushort";
  case ComponentType::I32:
    return "int";
  case ComponentType::U32:
    return "uint";
  case ComponentType::F32:
  case ComponentType::SNormF32:
  case ComponentType::UNormF32:
  default:
    return "float";
  }
}

Error Emitter::inferElementTypes() {
  for (ResState &RS : ResStates) {
    if (isTextureKind(RS.RB->Kind) &&
        RS.RB->ElementType != ComponentType::Invalid)
      RS.ElemType = textureElementType(RS.RB->ElementType);
  }

  std::string FirstError;
  auto Record = [&](mlir::Value Handle, mlir::Type Elem) {
    ResState *RS = stateFor(Handle);
    if (!RS) {
      FirstError = "resource access through unresolved handle";
      return;
    }
    auto TyOrErr = mslType(Elem);
    if (!TyOrErr) {
      FirstError = toString(TyOrErr.takeError());
      return;
    }
    if (RS->ElemType.empty())
      RS->ElemType = *TyOrErr;
    else if (elemByteSize(RS->ElemType) != elemByteSize(*TyOrErr))
      FirstError = "resource '" + RS->ParamName +
                   "' is accessed with mixed element sizes (" + RS->ElemType +
                   " vs " + *TyOrErr + "); not supported yet";
    // Same-size mixed types (float vs uint) keep the first-seen canonical
    // type; load/store sites reinterpret with as_type.
  };

  Entry.walk([&](mlir::Operation *Op) {
    if (!FirstError.empty())
      return;
    if (auto BL = mlir::dyn_cast<ir::BufferLoadOp>(Op)) {
      auto VecTy = mlir::cast<mlir::VectorType>(BL.getResult().getType());
      Record(BL.getHandle(), VecTy.getElementType());
      return;
    }
    if (auto BS = mlir::dyn_cast<ir::BufferStoreOp>(Op)) {
      if (!BS.getValues().empty())
        Record(BS.getHandle(), BS.getValues()[0].getType());
      return;
    }
    if (auto TL = mlir::dyn_cast<ir::TexLoadOp>(Op)) {
      ResState *RS = stateFor(TL.getHandle());
      if (RS && RS->ElemType.empty()) {
        auto VecTy = mlir::cast<mlir::VectorType>(TL.getResult().getType());
        Record(TL.getHandle(), VecTy.getElementType());
      }
      return;
    }
    if (auto TS = mlir::dyn_cast<ir::TexStoreOp>(Op)) {
      ResState *RS = stateFor(TS.getHandle());
      if (RS && RS->ElemType.empty() && !TS.getValues().empty())
        Record(TS.getHandle(), TS.getValues()[0].getType());
      return;
    }
    if (auto Samp = mlir::dyn_cast<ir::TexSampleOp>(Op)) {
      ResState *RS = stateFor(Samp.getTex());
      if (RS) {
        RS->IsSampled = true;
        if (Samp.getMode() == ir::SampleMode::Cmp ||
            Samp.getMode() == ir::SampleMode::CmpLevelZero)
          RS->IsDepthSampled = true;
        if (RS->ElemType.empty()) {
          auto VecTy = mlir::cast<mlir::VectorType>(Samp.getResult().getType());
          Record(Samp.getTex(), VecTy.getElementType());
        }
      }
      return;
    }
    if (auto Gath = mlir::dyn_cast<ir::TexGatherOp>(Op)) {
      ResState *RS = stateFor(Gath.getTex());
      if (RS) {
        RS->IsSampled = true;
        if (Gath.getCmpVal())
          RS->IsDepthSampled = true;
        if (RS->ElemType.empty()) {
          auto VecTy = mlir::cast<mlir::VectorType>(Gath.getResult().getType());
          Record(Gath.getTex(), VecTy.getElementType());
        }
      }
      return;
    }
    if (auto Lod = mlir::dyn_cast<ir::TexCalcLodOp>(Op)) {
      ResState *RS = stateFor(Lod.getTex());
      if (RS) {
        RS->IsSampled = true;
        if (RS->ElemType.empty())
          Record(Lod.getTex(), Lod.getResult().getType());
      }
      return;
    }
    if (auto BA = mlir::dyn_cast<ir::BufferAtomicOp>(Op)) {
      ResState *RS = stateFor(BA.getHandle());
      if (RS && isTextureKind(RS->RB->Kind)) {
        RS->RB->RequiresTextureAtomics = true;
        if (RS->ElemType == "float")
          RS->ElemType = "uint";
      }
      Record(BA.getHandle(), BA.getValue().getType());
      return;
    }
    if (auto BCX = mlir::dyn_cast<ir::BufferAtomicCmpXchgOp>(Op)) {
      ResState *RS = stateFor(BCX.getHandle());
      if (RS && isTextureKind(RS->RB->Kind)) {
        RS->RB->RequiresTextureAtomics = true;
        if (RS->ElemType == "float")
          RS->ElemType = "uint";
      }
      Record(BCX.getHandle(), BCX.getValue().getType());
      return;
    }
    if (auto CB = mlir::dyn_cast<ir::CBufferLoadOp>(Op)) {
      auto VecTy = mlir::cast<mlir::VectorType>(CB.getResult().getType());
      mlir::Type Elem = VecTy.getElementType();
      ResState *RS = stateFor(CB.getHandle());
      if (!RS) {
        FirstError = "cbuffer access through unresolved handle";
        return;
      }
      std::string Row;
      if (Elem.isF32())
        Row = "float4";
      else if (Elem.isInteger(16) || Elem.isInteger(32) ||
               Elem.isInteger(64) || Elem.isF16() ||
               (Opts.EmulateF64 && Elem.isF64()))
        Row = "uint4";
      else {
        FirstError = "cbuffer '" + RS->ParamName +
                     "' uses an unsupported element type";
        return;
      }
      if (RS->ElemType.empty())
        RS->ElemType = Row;
      else if (RS->ElemType != Row)
        RS->ElemType = "float4";
      return;
    }
  });
  if (!FirstError.empty())
    return makeError("{0}", FirstError);

  auto DefaultElemType = [](ResState &RS) {
    if (RS.ElemType.empty())
      RS.ElemType = RS.RB->Class == ResourceClass::CBV ? "float4"
                    : isTextureKind(RS.RB->Kind)       ? "float"
                                                       : "uint";
  };
  for (ResState &RS : ResStates)
    DefaultElemType(RS);
  for (ResState &RS : HeapStates)
    DefaultElemType(RS);
  return Error::success();
}

//===----------------------------------------------------------------------===//
// Globals, local arrays, and pointer lvalues
//===----------------------------------------------------------------------===//

Error Emitter::collectConstantGlobals() {
  auto Mod = Entry->getParentOfType<mlir::ModuleOp>();
  Error Err = Error::success();
  Entry.walk([&](mlir::LLVM::AddressOfOp AO) {
    auto G = Mod.lookupSymbol<mlir::LLVM::GlobalOp>(AO.getGlobalNameAttr());
    if (!G || G.getAddrSpace() != 0 || !G.getConstant() ||
        GlobalNames.contains(G))
      return mlir::WalkResult::advance();

    auto Arr = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(G.getGlobalType());
    auto Init = mlir::dyn_cast_if_present<mlir::DenseElementsAttr>(
        G.getValueAttr());
    if (!Arr || !Init || Init.getType().getRank() != 1 ||
        Init.getElementType() != Arr.getElementType() ||
        static_cast<uint64_t>(Init.getNumElements()) != Arr.getNumElements()) {
      Err = makeError("unsupported initializer for constant global '{0}'",
                      G.getSymName());
      return mlir::WalkResult::interrupt();
    }

    auto Elem = mslType(Arr.getElementType());
    if (!Elem) {
      Err = Elem.takeError();
      return mlir::WalkResult::interrupt();
    }
    std::string Name = "crucible_const_" + std::to_string(GlobalNames.size());
    while (Name == R.MSLEntryName ||
           llvm::any_of(GlobalNames, [&](const auto &KV) {
             return KV.second == Name;
           }))
      Name += "_";

    raw_string_ostream OS(ConstantGlobalDecls);
    OS << "constant " << *Elem << " " << Name << "[" << Arr.getNumElements()
       << "] = {";
    bool First = true;
    for (auto Attr : Init.getValues<mlir::Attribute>()) {
      auto Value = scalarConstant(Attr);
      if (!Value) {
        Err = Value.takeError();
        return mlir::WalkResult::interrupt();
      }
      if (!First)
        OS << ", ";
      First = false;
      OS << Value->Text;
    }
    OS << "};\n";
    GlobalNames[G] = std::move(Name);
    return mlir::WalkResult::advance();
  });
  return Err;
}

Error Emitter::collectGroupshared() {
  auto Mod = Entry->getParentOfType<mlir::ModuleOp>();
  unsigned Idx = 0;
  for (auto G : Mod.getOps<mlir::LLVM::GlobalOp>()) {
    if (G.getAddrSpace() != 3)
      continue;
    mlir::Type T = G.getGlobalType();
    uint64_t N = 1;
    uint64_t VecN = 0;
    mlir::Type Elem = T;
    bool IsArray = false;
    if (auto Arr = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(T)) {
      N = Arr.getNumElements();
      Elem = Arr.getElementType();
      IsArray = true;
      if (auto Vec = mlir::dyn_cast<mlir::VectorType>(Elem);
          Vec && !Vec.isScalable() && Vec.getRank() == 1) {
        VecN = Vec.getNumElements();
        Elem = Vec.getElementType();
      }
    } else if (auto Vec = mlir::dyn_cast<mlir::VectorType>(T)) {
      if (!Vec.isScalable() && Vec.getRank() == 1) {
        N = Vec.getNumElements();
        Elem = Vec.getElementType();
        IsArray = true;
      }
    }
    auto ETy = mslType(Elem);
    if (!ETy)
      return ETy.takeError();
    std::string Fallback = "gs" + std::to_string(Idx++);
    std::string Name = sanitizeIdentifier(G.getSymName(), Fallback);
    for (auto &[Op, Existing] : GlobalNames)
      if (Existing == Name)
        Name += "_" + std::to_string(Idx);
    GlobalNames[G] = Name;
    GlobalTypes[G] = *ETy;
    GroupsharedDecls += "  threadgroup " + *ETy + " " + Name +
                        (IsArray ? "[" + std::to_string(N) + "]" : "") +
                        (VecN ? "[" + std::to_string(VecN) + "]" : "") + ";\n";
  }
  return Error::success();
}

Expected<std::string> Emitter::addrSpaceOf(mlir::Value Ptr) {
  auto PT = mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(Ptr.getType());
  if (!PT)
    return makeError("expected a pointer type");
  switch (PT.getAddressSpace()) {
  case 0:
    return std::string("thread");
  case 3:
    return std::string("threadgroup");
  default:
    return makeError("unsupported pointer address space {0}",
                     PT.getAddressSpace());
  }
}

Expected<Expr> Emitter::lvalueFor(mlir::Value Ptr, bool DecayArray) {
  mlir::Operation *Def = Ptr.getDefiningOp();
  if (auto Cast = mlir::dyn_cast_or_null<mlir::LLVM::IntToPtrOp>(Def)) {
    auto Ty = Cast->getAttrOfType<mlir::TypeAttr>("crucible.physical_type");
    if (!Ty)
      return makeError("integer-to-pointer cast lacks physical access type");
    auto T = mslType(Ty.getValue());
    auto Address = exprFor(Cast.getArg(), PrecUnknown);
    if (!T)
      return T.takeError();
    if (!Address)
      return Address.takeError();
    return Expr{"*((device " + *T + " *)(" + Address->Text + "))", PrecUnary};
  }
  if (!Def) {
    auto It = ValueNames.find(Ptr);
    if (It != ValueNames.end())
      return Expr{It->second, PrecPrimary};
    return makeError("pointer block arguments are not supported");
  }

  if (auto AO = mlir::dyn_cast<mlir::LLVM::AddressOfOp>(Def)) {
    auto Mod = Entry->getParentOfType<mlir::ModuleOp>();
    auto G = Mod.lookupSymbol<mlir::LLVM::GlobalOp>(AO.getGlobalNameAttr());
    auto It = G ? GlobalNames.find(G) : GlobalNames.end();
    if (It == GlobalNames.end())
      return makeError("reference to unsupported global '{0}'",
                       AO.getGlobalName());
    bool IsArray = mlir::isa<mlir::LLVM::LLVMArrayType>(G.getGlobalType()) ||
                   (G.getAddrSpace() == 3 &&
                    mlir::isa<mlir::VectorType>(G.getGlobalType()));
    return Expr{It->second + (IsArray && DecayArray ? "[0]" : ""),
                PrecPrimary};
  }

  if (auto AL = mlir::dyn_cast<mlir::LLVM::AllocaOp>(Def)) {
    auto It = ValueNames.find(AL.getResult());
    if (It == ValueNames.end())
      return makeError("use of alloca before declaration");
    bool IsArray =
        mlir::isa<mlir::LLVM::LLVMArrayType, mlir::VectorType>(AL.getElemType());
    return Expr{It->second + (IsArray && DecayArray ? "[0]" : ""),
                PrecPrimary};
  }

  if (mlir::isa<ir::GetMeshPayloadOp>(Def))
    return Expr{"crucible_payload", PrecPrimary};

  if (auto GEP = mlir::dyn_cast<mlir::LLVM::GEPOp>(Def)) {
    auto Base = lvalueFor(GEP.getBase(), /*DecayArray=*/false);
    if (!Base)
      return Base.takeError();
    std::string Text = Base->Text;
    auto Indices = GEP.getIndices();
    auto AL = GEP.getBase().getDefiningOp<mlir::LLVM::AllocaOp>();
    auto Vec = AL ? mlir::dyn_cast<mlir::VectorType>(AL.getElemType())
                  : mlir::VectorType{};
    if (Vec && GEP.getElemType() == Vec.getElementType() &&
        Indices.size() == 1) {
      // LLVM's scalarizer addresses each lane via a scalar GEP.
      if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(Indices[0]))
        return Expr{Text + "[" + std::to_string(IA.getInt()) + "]",
                    PrecPrimary};
    }
    // Flat 1D pointer indexing on Layer 0 addrspace(1) kernel pointer args.
    if (auto BArg = mlir::dyn_cast<mlir::BlockArgument>(GEP.getBase());
        BArg && Indices.size() == 1) {
      if (auto PT = mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(BArg.getType());
          PT && PT.getAddressSpace() == 1) {
        if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(Indices[0]))
          return Expr{Text + "[" +
                          std::to_string(IA.getValue().getZExtValue()) + "u]",
                      PrecPrimary};
        auto V = mlir::cast<mlir::Value>(Indices[0]);
        auto E = exprFor(V, PrecUnknown);
        if (!E)
          return E.takeError();
        return Expr{Text + "[" + E->Text + "]", PrecPrimary};
      }
    }
    mlir::Type CurTy = GEP.getElemType();
    for (unsigned I = 0; I < Indices.size(); ++I) {
      std::string IdxText;
      std::optional<uint64_t> ConstIdx;
      if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(Indices[I])) {
        ConstIdx = IA.getValue().getZExtValue();
        if (I == 0) {
          if (*ConstIdx != 0)
            return makeError("pointer-arithmetic GEPs are not supported");
          continue; // leading zero: address of the aggregate itself
        }
        IdxText = std::to_string(*ConstIdx) + "u";
      } else {
        auto V = mlir::cast<mlir::Value>(Indices[I]);
        auto E = exprFor(V, PrecUnknown);
        if (!E)
          return E.takeError();
        if (I == 0)
          return makeError("pointer-arithmetic GEPs are not supported");
        IdxText = E->Text;
      }
      if (auto ST = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(CurTy)) {
        if (!ConstIdx || *ConstIdx >= ST.getBody().size())
          return makeError("non-constant or out-of-bounds struct GEP");
        Text += ".f" + std::to_string(*ConstIdx);
        CurTy = ST.getBody()[*ConstIdx];
      } else {
        Text += "[" + IdxText + "]";
        if (auto Arr = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(CurTy))
          CurTy = Arr.getElementType();
        else if (auto Vec = mlir::dyn_cast<mlir::VectorType>(CurTy))
          CurTy = Vec.getElementType();
      }
    }
    return Expr{Text, PrecPrimary};
  }

  return makeError("unsupported pointer producer '{0}'",
                   Def->getName().getStringRef());
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

static bool mayWriteMemory(mlir::Operation *Op) {
  if (Op->hasAttr("crucible.volatile"))
    return true;
  if (auto Load = mlir::dyn_cast<mlir::LLVM::LoadOp>(Op);
      Load && Load.getVolatile_())
    return true;
  if (Op->getNumRegions() > 0)
    return true; // conservative: nested control flow may contain writes
  if (mlir::isa<mlir::LLVM::StoreOp, mlir::LLVM::AtomicRMWOp,
                mlir::LLVM::AtomicCmpXchgOp, mlir::gpu::BarrierOp,
                ir::BarrierOp, ir::StoreOutputOp, ir::StorePatchConstantOp,
                ir::DiscardOp, ir::EmitStreamOp, ir::CutStreamOp,
                ir::BufferStoreOp, ir::TexStoreOp, ir::BufferAtomicOp,
                ir::BufferAtomicCmpXchgOp, ir::BufferUpdateCounterOp,
                ir::SetMeshOutputCountsOp, ir::EmitIndicesOp,
                ir::StoreVertexOutputOp, ir::StorePrimitiveOutputOp,
                ir::DispatchMeshOp, ir::TraceRayOp, ir::RayQueryTraceInlineOp,
                ir::RayQueryProceedOp, ir::RayQueryAbortOp,
                ir::RayQueryCommitOp>(Op))
    return true;
  if (mlir::isa<mlir::LLVM::CallOp>(Op))
    return true; // unknown call
  return false;  // the pure ops we support
}

bool Emitter::shouldInline(mlir::Operation *Op) {
  if (SinCosPartner.count(Op))
    return false;
  if (Op->hasAttr("crucible.volatile") || Op->hasAttr("crucible.address_table"))
    return false;
  if (auto Load = mlir::dyn_cast<mlir::LLVM::LoadOp>(Op);
      Load && Load.getVolatile_())
    return false;
  if (isConstantLike(Op))
    return true;

  // Pointer producers are always folded into the lvalue they feed.
  if (mlir::isa<mlir::LLVM::GEPOp, mlir::LLVM::AddressOfOp,
                mlir::LLVM::IntToPtrOp, ir::GetMeshPayloadOp>(Op))
    return true;

  // Trivial GPU builtin reads & resource handles: never materialized directly.
  if (mlir::isa<mlir::gpu::GlobalIdOp, mlir::gpu::BlockIdOp,
                mlir::gpu::ThreadIdOp, mlir::gpu::LaneIdOp,
                mlir::gpu::SubgroupSizeOp, ir::SysValOp,
                ir::BindResourceOp, ir::HeapResourceOp,
                ir::BufferLoadOp, ir::CBufferLoadOp,
                ir::TexGetDimensionsOp>(Op))
    return true;
  if (mlir::isa<mlir::arith::IndexCastOp, mlir::arith::IndexCastUIOp>(Op)) {
    if (auto *SrcOp = Op->getOperand(0).getDefiningOp()) {
      if (mlir::isa<mlir::gpu::GlobalIdOp, mlir::gpu::BlockIdOp,
                    mlir::gpu::ThreadIdOp, mlir::gpu::LaneIdOp,
                    mlir::gpu::SubgroupSizeOp>(SrcOp))
        return true;
    }
  }
  // Convergent subgroup/quad ops, texture ops, atomics, ray query, and
  // multi-result ops must be materialized as statements.
  if (mlir::isa<mlir::gpu::SubgroupReduceOp, mlir::gpu::SubgroupBroadcastOp,
                mlir::gpu::BallotOp, mlir::gpu::ShuffleOp,
                ir::WaveIsFirstLaneOp, ir::WaveVoteOp, ir::WaveBallotOp,
                ir::WaveBitCountOp, ir::WaveMultiPrefixOp,
                ir::QuadReadLaneAtOp, ir::QuadOp, ir::Unpack4x8Op,
                ir::SplitDoubleOp, ir::UAddcOp, ir::BufferAtomicOp,
                ir::BufferAtomicCmpXchgOp, ir::BufferUpdateCounterOp,
                ir::TexLoadOp, ir::TexSampleOp, ir::TexGatherOp,
                ir::TexCalcLodOp, ir::RayQueryAllocOp,
                ir::RayQueryProceedOp>(Op))
    return false;

  // Side-effecting or aggregate-producing ops are materialized as statements.
  if (mlir::isa<mlir::LLVM::StoreOp, mlir::LLVM::AtomicRMWOp,
                mlir::LLVM::AtomicCmpXchgOp, mlir::LLVM::AllocaOp,
                mlir::LLVM::InsertValueOp>(Op))
    return false;
  if (auto Ld = mlir::dyn_cast<mlir::LLVM::LoadOp>(Op);
      Ld && mlir::isa<mlir::LLVM::LLVMArrayType>(Ld.getType()))
    return false;

  if (Op->getNumResults() != 1 || !Op->getResult(0).hasOneUse())
    return false;

  mlir::Operation *User = *Op->getResult(0).getUsers().begin();

  // Never sink an expression across region/block boundaries (e.g., back into a
  // loop body after LoopInvariantCodeMotion, or into an scf.if branch).
  if (User->getBlock() != Op->getBlock())
    return false;

  // Ops whose MSL emission expands their operand more than once (e.g.
  // `crucible.frc` -> `x - floor(x)`) would duplicate the entire inlined
  // subexpression (and blow up exponentially when nested). Materialize the
  // operand into a temporary instead.
  if (mlir::isa<ir::FrcOp>(User))
    return false;

  // Avoid sinking an expression across a long span of statements in the same
  // block, which stretches operand live ranges (driving up GPU register
  // pressure / spilling) and creates huge deeply-nested expressions.
  constexpr unsigned kMaxSinkDistance = 16;
  unsigned Dist = 0;
  for (mlir::Operation *P = Op->getNextNode(); P && P != User;
       P = P->getNextNode()) {
    if (!isConstantLike(P) && ++Dist > kMaxSinkDistance)
      return false;
  }

  // Cap the transitive inlined AST size so chained single-use reductions
  // (e.g., dozens of nested min/max/sqrt SDF terms) materialize into local
  // temporaries instead of multi-kilobyte single-line expressions.
  constexpr unsigned kMaxInlineNodes = 24;
  auto CountNodes = [&](auto &Self, mlir::Operation *Cur,
                        unsigned Budget) -> unsigned {
    unsigned Total = 1;
    for (mlir::Value Operand : Cur->getOperands()) {
      if (ValueNames.count(Operand))
        continue;
      mlir::Operation *Def = Operand.getDefiningOp();
      if (!Def || isConstantLike(Def) || Def->getBlock() != Cur->getBlock())
        continue;
      if (Def->getNumResults() == 1 && Def->getResult(0).hasOneUse()) {
        Total += Self(Self, Def, Budget > Total ? Budget - Total : 0);
        if (Total > Budget)
          return Total;
      }
    }
    return Total;
  };
  if (CountNodes(CountNodes, Op, kMaxInlineNodes) > kMaxInlineNodes)
    return false;

  // A memory read must not be folded forward across a write or barrier,
  // since the inlined load would observe the mutation. Walk through any
  // chain of inlined single-use consumers to find the actual emission sink.
  auto PinAgainstWrites = [&]() {
    mlir::Operation *Sink = User;
    while (Sink && Sink->getBlock() == Op->getBlock() && shouldInline(Sink) &&
           Sink->getNumResults() == 1 && Sink->getResult(0).hasOneUse()) {
      Sink = *Sink->getResult(0).getUsers().begin();
    }
    if (!Sink || Sink->getBlock() != Op->getBlock())
      return false; // conservative across region boundaries
    for (mlir::Operation *P = Op->getNextNode(); P && P != Sink;
         P = P->getNextNode())
      if (mayWriteMemory(P))
        return false;
    return true;
  };

  if (mlir::isa<mlir::LLVM::LoadOp>(Op))
    return PinAgainstWrites();

  if (auto EE = mlir::dyn_cast<mlir::LLVM::ExtractElementOp>(Op)) {
    if (auto BL = EE.getVector().getDefiningOp<ir::BufferLoadOp>()) {
      ResState *RS = stateFor(BL.getHandle());
      if (RS && RS->RB->Class == ResourceClass::UAV)
        return PinAgainstWrites();
    }
    return true;
  }

  return !mayWriteMemory(Op);
}

Expected<Expr> Emitter::exprFor(mlir::Value V, int ParentPrec) {
  Expr E;
  auto It = ValueNames.find(V);
  if (It != ValueNames.end()) {
    E = Expr{It->second, PrecPrimary};
  } else if (mlir::Operation *Def = V.getDefiningOp()) {
    if (auto BL = mlir::dyn_cast<ir::BufferLoadOp>(Def);
        BL && V == BL.getStatus()) {
      ResState *RS = stateFor(BL.getHandle());
      if (!RS)
        return makeError("buffer status through unresolved handle");
      if (!Opts.BoundsCheckBuffers || R.BufferSizesMetalIndex == UINT32_MAX) {
        E = Expr{"1u", PrecPrimary};
      } else {
        auto Index = bufferIndexExpr(*RS, BL.getIndex(), BL.getElementOffset(), 0);
        if (!Index)
          return Index.takeError();
        auto Bound = bufferBoundExpr(*RS, BL.getHandle());
        if (!Bound)
          return Bound.takeError();
        E = Expr{"uint(uint(" + *Index + ") < " + *Bound + ")", PrecPrimary};
      }
    } else if (isConstantLike(Def)) {
      auto CE = constant(Def);
      if (!CE)
        return CE.takeError();
      E = *CE;
    } else {
      auto OE = opExpr(Def);
      if (!OE)
        return OE.takeError();
      E = *OE;
    }
  } else {
    return makeError("use of unnamed block argument");
  }
  if (E.Prec < ParentPrec)
    return Expr{"(" + E.Text + ")", PrecPrimary};
  return E;
}

namespace {
struct BinOpInfo {
  const char *Spelling;
  int Prec;
  bool Signed; // operate on reinterpreted signed values
};
} // namespace

static std::optional<BinOpInfo> binOpInfo(mlir::Operation *Op) {
  using namespace mlir::LLVM;
  if (mlir::isa<FAddOp, AddOp>(Op))
    return BinOpInfo{"+", PrecAdd, false};
  if (mlir::isa<FSubOp, SubOp>(Op))
    return BinOpInfo{"-", PrecAdd, false};
  if (mlir::isa<FMulOp, MulOp>(Op))
    return BinOpInfo{"*", PrecMul, false};
  if (mlir::isa<FDivOp, UDivOp>(Op))
    return BinOpInfo{"/", PrecMul, false};
  if (mlir::isa<SDivOp>(Op))
    return BinOpInfo{"/", PrecMul, true};
  if (mlir::isa<FRemOp, URemOp>(Op))
    return BinOpInfo{"%", PrecMul, false};
  if (mlir::isa<SRemOp>(Op))
    return BinOpInfo{"%", PrecMul, true};
  if (mlir::isa<ShlOp>(Op))
    return BinOpInfo{"<<", PrecShift, false};
  if (mlir::isa<LShrOp>(Op))
    return BinOpInfo{">>", PrecShift, false};
  if (mlir::isa<AShrOp>(Op))
    return BinOpInfo{">>", PrecShift, true};
  if (mlir::isa<AndOp>(Op))
    return BinOpInfo{"&", PrecAnd, false};
  if (mlir::isa<OrOp>(Op))
    return BinOpInfo{"|", PrecOr, false};
  if (mlir::isa<XOrOp>(Op))
    return BinOpInfo{"^", PrecXor, false};
  return std::nullopt;
}

Expected<Expr> Emitter::opExpr(mlir::Operation *Op) {
  using namespace mlir::LLVM;

  if (mlir::isa<SAddSat, UAddSat>(Op)) {
    auto L = exprFor(Op->getOperand(0), PrecUnknown);
    if (!L)
      return L.takeError();
    auto R2 = exprFor(Op->getOperand(1), PrecUnknown);
    if (!R2)
      return R2.takeError();
    mlir::Type T = Op->getResult(0).getType();
    if (mlir::isa<SAddSat>(Op))
      return Expr{unsignedView(T, "addsat(" + signedView(T, L->Text) + ", " +
                                     signedView(T, R2->Text) + ")"),
                  PrecPrimary};
    return Expr{"addsat(" + L->Text + ", " + R2->Text + ")", PrecPrimary};
  }

  if (auto Info = binOpInfo(Op)) {
    mlir::Type T = Op->getResult(0).getType();
    int LPrec = Info->Prec == PrecShift ? PrecMul : Info->Prec;
    // Metal requires parentheses around comparisons used in bitwise operands.
    if (Info->Prec >= PrecOr && Info->Prec <= PrecAnd)
      LPrec = PrecShift;
    auto L = exprFor(Op->getOperand(0), Info->Signed ? PrecUnknown : LPrec);
    if (!L)
      return L.takeError();
    auto R2 = exprFor(Op->getOperand(1),
                      Info->Signed ? PrecUnknown : LPrec + 1);
    if (!R2)
      return R2.takeError();
    if (Info->Signed) {
      return Expr{unsignedView(T, signedView(T, L->Text) + " " +
                                     Info->Spelling + " " +
                                     (mlir::isa<AShrOp>(Op)
                                          ? R2->Text
                                          : signedView(T, R2->Text))),
                  PrecPrimary};
    }
    if (Opts.EmulateF64 && T.isF64()) {
      requireHelper("crucible_f64");
      const char *Fn = nullptr;
      if (mlir::isa<FAddOp>(Op))
        Fn = "crucible_f64_add";
      else if (mlir::isa<FSubOp>(Op))
        Fn = "crucible_f64_sub";
      else if (mlir::isa<FMulOp>(Op))
        Fn = "crucible_f64_mul";
      else if (mlir::isa<FDivOp>(Op))
        Fn = "crucible_f64_div";
      else
        return makeError("unsupported f64 binary op");
      auto L64 = exprFor(Op->getOperand(0), PrecUnknown);
      if (!L64)
        return L64.takeError();
      auto R64 = exprFor(Op->getOperand(1), PrecUnknown);
      if (!R64)
        return R64.takeError();
      return Expr{std::string(Fn) + "(" + L64->Text + ", " + R64->Text + ")",
                  PrecPrimary};
    }
    if (T.isInteger(16))
      return Expr{"ushort(uint(" + L->Text + ") " + Info->Spelling +
                      " uint(" + R2->Text + "))",
                  PrecPrimary};
    if (mlir::isa<FRemOp>(Op))
      return Expr{"fmod(" + L->Text + ", " + R2->Text + ")", PrecPrimary};
    bool IsBool = Op->getResult(0).getType().isInteger(1);
    if (IsBool) {
      // i1 bitwise ops read better as logical operators (operands are pure).
      StringRef Sp(Info->Spelling);
      if (Sp == "&")
        return Expr{L->Text + " && " + R2->Text, PrecLAnd};
      if (Sp == "|")
        return Expr{L->Text + " || " + R2->Text, PrecLOr};
      if (Sp == "^")
        return Expr{L->Text + " != " + R2->Text, PrecEq};
    }
    return Expr{L->Text + " " + Info->Spelling + " " + R2->Text, Info->Prec};
  }

  if (auto FNeg = mlir::dyn_cast<FNegOp>(Op)) {
    if (Opts.EmulateF64 && FNeg.getType().isF64()) {
      auto A = exprFor(FNeg.getOperand(), PrecXor);
      if (!A)
        return A.takeError();
      return Expr{A->Text + " ^ 0x8000000000000000ul", PrecXor};
    }
    auto A = exprFor(FNeg.getOperand(), PrecUnary);
    if (!A)
      return A.takeError();
    return Expr{"-" + A->Text, PrecUnary};
  }

  if (auto Cmp = mlir::dyn_cast<ICmpOp>(Op)) {
    const char *Sp = nullptr;
    bool Signed = false;
    switch (Cmp.getPredicate()) {
    case ICmpPredicate::eq:
      Sp = "==";
      break;
    case ICmpPredicate::ne:
      Sp = "!=";
      break;
    case ICmpPredicate::ult:
      Sp = "<";
      break;
    case ICmpPredicate::ule:
      Sp = "<=";
      break;
    case ICmpPredicate::ugt:
      Sp = ">";
      break;
    case ICmpPredicate::uge:
      Sp = ">=";
      break;
    case ICmpPredicate::slt:
      Sp = "<";
      Signed = true;
      break;
    case ICmpPredicate::sle:
      Sp = "<=";
      Signed = true;
      break;
    case ICmpPredicate::sgt:
      Sp = ">";
      Signed = true;
      break;
    case ICmpPredicate::sge:
      Sp = ">=";
      Signed = true;
      break;
    }
    int Prec = (StringRef(Sp) == "==" || StringRef(Sp) == "!=") ? PrecEq
                                                                : PrecRel;
    auto L = exprFor(Cmp.getLhs(), Signed ? PrecUnknown : Prec);
    if (!L)
      return L.takeError();
    auto R2 = exprFor(Cmp.getRhs(), Signed ? PrecUnknown : Prec + 1);
    if (!R2)
      return R2.takeError();
    if (Signed)
      return Expr{signedView(Cmp.getLhs().getType(), L->Text) + " " + Sp + " " +
                      signedView(Cmp.getRhs().getType(), R2->Text),
                  Prec};
    return Expr{L->Text + " " + Sp + " " + R2->Text, Prec};
  }

  if (auto Cmp = mlir::dyn_cast<FCmpOp>(Op)) {
    if (Opts.EmulateF64 && Cmp.getLhs().getType().isF64()) {
      requireHelper("crucible_f64");
      auto L = exprFor(Cmp.getLhs(), PrecUnknown);
      if (!L)
        return L.takeError();
      auto R2 = exprFor(Cmp.getRhs(), PrecUnknown);
      if (!R2)
        return R2.takeError();
      std::string A = L->Text, B = R2->Text;
      std::string NanCheck =
          "(crucible_f64_isnan(" + A + ") || crucible_f64_isnan(" + B + "))";
      switch (Cmp.getPredicate()) {
      case FCmpPredicate::oeq:
        return Expr{"crucible_f64_cmp_eq(" + A + ", " + B + ")", PrecPrimary};
      case FCmpPredicate::ueq:
        return Expr{"(" + NanCheck + " || crucible_f64_cmp_eq(" + A + ", " + B +
                        "))",
                    PrecPrimary};
      case FCmpPredicate::one:
        return Expr{"(!" + NanCheck + " && !crucible_f64_cmp_eq(" + A + ", " +
                        B + "))",
                    PrecPrimary};
      case FCmpPredicate::une:
        return Expr{"!crucible_f64_cmp_eq(" + A + ", " + B + ")", PrecUnary};
      case FCmpPredicate::olt:
        return Expr{"crucible_f64_cmp_lt(" + A + ", " + B + ")", PrecPrimary};
      case FCmpPredicate::ult:
        return Expr{"(" + NanCheck + " || crucible_f64_cmp_lt(" + A + ", " + B +
                        "))",
                    PrecPrimary};
      case FCmpPredicate::ole:
        return Expr{"crucible_f64_cmp_le(" + A + ", " + B + ")", PrecPrimary};
      case FCmpPredicate::ule:
        return Expr{"(" + NanCheck + " || crucible_f64_cmp_le(" + A + ", " + B +
                        "))",
                    PrecPrimary};
      case FCmpPredicate::ogt:
        return Expr{"crucible_f64_cmp_lt(" + B + ", " + A + ")", PrecPrimary};
      case FCmpPredicate::ugt:
        return Expr{"(" + NanCheck + " || crucible_f64_cmp_lt(" + B + ", " + A +
                        "))",
                    PrecPrimary};
      case FCmpPredicate::oge:
        return Expr{"crucible_f64_cmp_le(" + B + ", " + A + ")", PrecPrimary};
      case FCmpPredicate::uge:
        return Expr{"(" + NanCheck + " || crucible_f64_cmp_le(" + B + ", " + A +
                        "))",
                    PrecPrimary};
      case FCmpPredicate::ord:
        return Expr{"!" + NanCheck, PrecUnary};
      case FCmpPredicate::uno:
        return Expr{NanCheck, PrecPrimary};
      default:
        return makeError("unsupported f64 fcmp predicate");
      }
    }
    if (Cmp.getPredicate() == FCmpPredicate::uno ||
        Cmp.getPredicate() == FCmpPredicate::ord) {
      auto L = exprFor(Cmp.getLhs(), PrecUnknown);
      auto Rhs = exprFor(Cmp.getRhs(), PrecUnknown);
      if (!L || !Rhs)
        return makeError("invalid fcmp operand");
      bool IsUno = Cmp.getPredicate() == FCmpPredicate::uno;
      if (Cmp.getLhs() == Cmp.getRhs())
        return Expr{(IsUno ? "isnan(" : "!isnan(") + L->Text + ")",
                    IsUno ? PrecPrimary : PrecUnary};
      return Expr{IsUno ? "(isnan(" + L->Text + ") || isnan(" + Rhs->Text + "))"
                        : "(!isnan(" + L->Text + ") && !isnan(" + Rhs->Text +
                              "))",
                  PrecPrimary};
    }
    const char *Sp = nullptr;
    switch (Cmp.getPredicate()) {
    case FCmpPredicate::oeq:
    case FCmpPredicate::ueq:
      Sp = "==";
      break;
    case FCmpPredicate::one:
    case FCmpPredicate::une:
      Sp = "!=";
      break;
    case FCmpPredicate::olt:
    case FCmpPredicate::ult:
      Sp = "<";
      break;
    case FCmpPredicate::ole:
    case FCmpPredicate::ule:
      Sp = "<=";
      break;
    case FCmpPredicate::ogt:
    case FCmpPredicate::ugt:
      Sp = ">";
      break;
    case FCmpPredicate::oge:
    case FCmpPredicate::uge:
      Sp = ">=";
      break;
    default:
      return makeError("unsupported fcmp predicate");
    }
    int Prec = (StringRef(Sp) == "==" || StringRef(Sp) == "!=") ? PrecEq
                                                                : PrecRel;
    auto L = exprFor(Cmp.getLhs(), Prec);
    if (!L)
      return L.takeError();
    auto R2 = exprFor(Cmp.getRhs(), Prec + 1);
    if (!R2)
      return R2.takeError();
    return Expr{L->Text + " " + Sp + " " + R2->Text, Prec};
  }

  if (auto Cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(Op)) {
    const char *Sp = nullptr;
    bool Signed = false;
    using P = mlir::arith::CmpIPredicate;
    switch (Cmp.getPredicate()) {
    case P::eq:
      Sp = "==";
      break;
    case P::ne:
      Sp = "!=";
      break;
    case P::ult:
      Sp = "<";
      break;
    case P::ule:
      Sp = "<=";
      break;
    case P::ugt:
      Sp = ">";
      break;
    case P::uge:
      Sp = ">=";
      break;
    case P::slt:
      Sp = "<";
      Signed = true;
      break;
    case P::sle:
      Sp = "<=";
      Signed = true;
      break;
    case P::sgt:
      Sp = ">";
      Signed = true;
      break;
    case P::sge:
      Sp = ">=";
      Signed = true;
      break;
    }
    int Prec = (StringRef(Sp) == "==" || StringRef(Sp) == "!=") ? PrecEq
                                                                : PrecRel;
    auto L = exprFor(Cmp.getLhs(), Signed ? PrecUnknown : Prec);
    if (!L)
      return L.takeError();
    auto R2 = exprFor(Cmp.getRhs(), Signed ? PrecUnknown : Prec + 1);
    if (!R2)
      return R2.takeError();
    if (Signed)
      return Expr{signedView(Cmp.getLhs().getType(), L->Text) + " " + Sp + " " +
                      signedView(Cmp.getRhs().getType(), R2->Text),
                  Prec};
    return Expr{L->Text + " " + Sp + " " + R2->Text, Prec};
  }

  // arith.xori introduced by the structurizer for condition negation.
  if (auto Xor = mlir::dyn_cast<mlir::arith::XOrIOp>(Op)) {
    if (Op->getResult(0).getType().isInteger(1)) {
      auto RC = constInt(Xor.getRhs());
      if (RC && *RC == 1) {
        auto A = exprFor(Xor.getLhs(), PrecUnary);
        if (!A)
          return A.takeError();
        return Expr{"!" + A->Text, PrecUnary};
      }
      auto L = exprFor(Xor.getLhs(), PrecEq);
      if (!L)
        return L.takeError();
      auto R2 = exprFor(Xor.getRhs(), PrecEq + 1);
      if (!R2)
        return R2.takeError();
      return Expr{L->Text + " != " + R2->Text, PrecEq};
    }
  }
  if (auto And = mlir::dyn_cast<mlir::arith::AndIOp>(Op)) {
    auto L = exprFor(And.getLhs(), PrecLAnd);
    if (!L)
      return L.takeError();
    auto R2 = exprFor(And.getRhs(), PrecLAnd + 1);
    if (!R2)
      return R2.takeError();
    return Expr{L->Text + " && " + R2->Text, PrecLAnd};
  }
  if (auto Or = mlir::dyn_cast<mlir::arith::OrIOp>(Op)) {
    auto L = exprFor(Or.getLhs(), PrecLOr);
    if (!L)
      return L.takeError();
    auto R2 = exprFor(Or.getRhs(), PrecLOr + 1);
    if (!R2)
      return R2.takeError();
    return Expr{L->Text + " || " + R2->Text, PrecLOr};
  }

  if (auto Sel = mlir::dyn_cast<SelectOp>(Op)) {
    bool VecCond = mlir::isa<mlir::VectorType>(Sel.getCondition().getType());
    auto C = exprFor(Sel.getCondition(), VecCond ? PrecUnknown : PrecLOr);
    if (!C)
      return C.takeError();
    auto T = exprFor(Sel.getTrueValue(), VecCond ? PrecUnknown : PrecTernary);
    if (!T)
      return T.takeError();
    auto F = exprFor(Sel.getFalseValue(), VecCond ? PrecUnknown : PrecTernary);
    if (!F)
      return F.takeError();
    if (VecCond)
      return Expr{"select(" + F->Text + ", " + T->Text + ", " + C->Text + ")",
                  PrecPrimary};
    return Expr{C->Text + " ? " + T->Text + " : " + F->Text, PrecTernary};
  }
  if (auto Sel = mlir::dyn_cast<mlir::arith::SelectOp>(Op)) {
    bool VecCond = mlir::isa<mlir::VectorType>(Sel.getCondition().getType());
    auto C = exprFor(Sel.getCondition(), VecCond ? PrecUnknown : PrecLOr);
    if (!C)
      return C.takeError();
    auto T = exprFor(Sel.getTrueValue(), VecCond ? PrecUnknown : PrecTernary);
    if (!T)
      return T.takeError();
    auto F = exprFor(Sel.getFalseValue(), VecCond ? PrecUnknown : PrecTernary);
    if (!F)
      return F.takeError();
    if (VecCond)
      return Expr{"select(" + F->Text + ", " + T->Text + ", " + C->Text + ")",
                  PrecPrimary};
    return Expr{C->Text + " ? " + T->Text + " : " + F->Text, PrecTernary};
  }

  // Casts.
  if (mlir::isa<mlir::arith::IndexCastOp, mlir::arith::IndexCastUIOp>(Op)) {
    auto Src = exprFor(Op->getOperand(0), PrecUnknown);
    if (!Src)
      return Src.takeError();
    auto DstTy = mslType(Op->getResult(0).getType());
    if (!DstTy)
      return DstTy.takeError();
    mlir::Type T = Op->getOperand(0).getType();
    if (T.isIndex() && *DstTy == "uint")
      return Src;
    if (!T.isIndex() && mlir::isa<mlir::arith::IndexCastOp>(Op))
      Src->Text = signedView(T, Src->Text);
    return Expr{*DstTy + "(" + Src->Text + ")", PrecPrimary};
  }
  if (mlir::isa<TruncOp, ZExtOp, FPTruncOp, FPExtOp, FPToUIOp, UIToFPOp,
                mlir::arith::ExtUIOp, mlir::arith::TruncIOp>(Op)) {
    auto Src = exprFor(Op->getOperand(0), PrecUnknown);
    if (!Src)
      return Src.takeError();
    mlir::Type SrcTy = Op->getOperand(0).getType();
    mlir::Type DstMlirTy = Op->getResult(0).getType();
    if (Opts.EmulateF64 && (SrcTy.isF64() || DstMlirTy.isF64())) {
      requireHelper("crucible_f64");
      if (mlir::isa<FPExtOp>(Op) && DstMlirTy.isF64()) {
        std::string F32 = SrcTy.isF32() ? Src->Text : "float(" + Src->Text + ")";
        return Expr{"crucible_f32_to_f64(" + F32 + ")", PrecPrimary};
      }
      if (mlir::isa<FPTruncOp>(Op) && SrcTy.isF64()) {
        std::string F32 = "crucible_f64_to_f32(" + Src->Text + ")";
        return Expr{DstMlirTy.isF16() ? "half(" + F32 + ")" : F32, PrecPrimary};
      }
      if (mlir::isa<UIToFPOp>(Op) && DstMlirTy.isF64())
        return Expr{"crucible_u64_to_f64(ulong(" + Src->Text + "))",
                    PrecPrimary};
      if (mlir::isa<FPToUIOp>(Op) && SrcTy.isF64()) {
        auto DstTy = mslType(DstMlirTy);
        if (!DstTy)
          return DstTy.takeError();
        std::string U64 = "crucible_f64_to_u64(" + Src->Text + ")";
        return Expr{DstMlirTy.isInteger(64) ? U64 : *DstTy + "(" + U64 + ")",
                    PrecPrimary};
      }
    }
    auto DstTy = mslType(DstMlirTy);
    if (!DstTy)
      return DstTy.takeError();
    if (mlir::isa<TruncOp, mlir::arith::TruncIOp>(Op)) {
      if (DstMlirTy.isInteger(1))
        return Expr{"bool((" + Src->Text + ") & 1u)", PrecPrimary};
      if (auto VT = mlir::dyn_cast<mlir::VectorType>(DstMlirTy);
          VT && VT.getElementType().isInteger(1))
        return Expr{*DstTy + "((" + Src->Text + ") & 1u)", PrecPrimary};
    }
    return Expr{*DstTy + "(" + Src->Text + ")", PrecPrimary};
  }
  if (mlir::isa<SExtOp, SIToFPOp, mlir::arith::ExtSIOp>(Op)) {
    // Match DXC's lowered HLSL sign(x) idiom:
    //   sitofp(sub(zext(fcmp ogt x, 0.0), zext(fcmp olt x, 0.0))) -> sign(x)
    // MSL sign(x) returns +1 for x>0, -1 for x<0, and 0 for ±0 and NaN,
    // matching the expanded comparison semantics exactly.
    if (auto S2F = mlir::dyn_cast<SIToFPOp>(Op);
        S2F && (S2F.getType().isF32() || S2F.getType().isF16())) {
      auto MatchZeroCmp = [](mlir::Value V, FCmpPredicate Pred,
                             mlir::Value &OutX) -> bool {
        mlir::Value CmpVal = V;
        if (auto Z = V.getDefiningOp<ZExtOp>())
          CmpVal = Z.getArg();
        else if (auto Z = V.getDefiningOp<mlir::arith::ExtUIOp>())
          CmpVal = Z.getIn();
        else
          return false;
        auto FCmp = CmpVal.getDefiningOp<FCmpOp>();
        if (!FCmp || FCmp.getPredicate() != Pred)
          return false;
        auto Cst = FCmp.getRhs().getDefiningOp<ConstantOp>();
        if (!Cst)
          return false;
        auto FA = mlir::dyn_cast<mlir::FloatAttr>(Cst.getValue());
        if (!FA || !FA.getValue().isZero())
          return false;
        OutX = FCmp.getLhs();
        return true;
      };
      mlir::Value Lhs, Rhs;
      if (auto Sub = S2F.getArg().getDefiningOp<SubOp>()) {
        Lhs = Sub.getLhs();
        Rhs = Sub.getRhs();
      } else if (auto Sub = S2F.getArg().getDefiningOp<mlir::arith::SubIOp>()) {
        Lhs = Sub.getLhs();
        Rhs = Sub.getRhs();
      }
      mlir::Value X1, X2;
      if (Lhs && Rhs && MatchZeroCmp(Lhs, FCmpPredicate::ogt, X1) &&
          MatchZeroCmp(Rhs, FCmpPredicate::olt, X2) && X1 == X2 &&
          X1.getType() == S2F.getType()) {
        auto XE = exprFor(X1, PrecUnknown);
        if (!XE)
          return XE.takeError();
        return Expr{"sign(" + XE->Text + ")", PrecPrimary};
      }
    }
    auto Src = exprFor(Op->getOperand(0), PrecUnknown);
    if (!Src)
      return Src.takeError();
    if (Opts.EmulateF64 && Op->getResult(0).getType().isF64()) {
      requireHelper("crucible_f64");
      return Expr{"crucible_i64_to_f64(long(" +
                      signedView(Op->getOperand(0).getType(), Src->Text) + "))",
                  PrecPrimary};
    }
    auto DstTy = mslType(Op->getResult(0).getType());
    if (!DstTy)
      return DstTy.takeError();
    return Expr{*DstTy + "(" +
                    signedView(Op->getOperand(0).getType(), Src->Text) + ")",
                PrecPrimary};
  }
  if (mlir::isa<FPToSIOp>(Op)) {
    auto Src = exprFor(Op->getOperand(0), PrecUnknown);
    if (!Src)
      return Src.takeError();
    mlir::Type T = Op->getResult(0).getType();
    if (Opts.EmulateF64 && Op->getOperand(0).getType().isF64()) {
      requireHelper("crucible_f64");
      std::string I64 = "crucible_f64_to_i64(" + Src->Text + ")";
      if (T.isInteger(64))
        return Expr{unsignedView(T, I64), PrecPrimary};
      StringRef Ty = T.isInteger(16) ? "short" : "int";
      return Expr{unsignedView(T, (Ty + "(" + I64 + ")").str()), PrecPrimary};
    }
    std::string Ty;
    if (auto VT = mlir::dyn_cast<mlir::VectorType>(T)) {
      mlir::Type E = VT.getElementType();
      StringRef Base = E.isInteger(8) ? "char" : E.isInteger(16) ? "short" : E.isInteger(64) ? "long" : "int";
      Ty = (Base + std::to_string(VT.getNumElements())).str();
    } else {
      Ty = T.isInteger(8) ? "char" : T.isInteger(16) ? "short" : T.isInteger(64) ? "long" : "int";
    }
    return Expr{unsignedView(T, Ty + "(" + Src->Text + ")"), PrecPrimary};
  }
  if (mlir::isa<BitcastOp>(Op)) {
    auto Src = exprFor(Op->getOperand(0), PrecUnknown);
    if (!Src)
      return Src.takeError();
    auto SrcTy = mslType(Op->getOperand(0).getType());
    if (!SrcTy)
      return SrcTy.takeError();
    auto DstTy = mslType(Op->getResult(0).getType());
    if (!DstTy)
      return DstTy.takeError();
    if (*SrcTy == *DstTy)
      return Src;
    return Expr{"as_type<" + *DstTy + ">(" + Src->Text + ")", PrecPrimary};
  }
  if (auto Freeze = mlir::dyn_cast<FreezeOp>(Op))
    return exprFor(Freeze.getOperand(), PrecUnknown);

  if (auto Load = mlir::dyn_cast<LoadOp>(Op)) {
    if (Load.getVolatile_()) {
      if (mlir::isa<mlir::VectorType>(Load.getType()))
        return makeError("volatile LLVM vector loads are unsupported");
      auto LV = lvalueFor(Load.getAddr());
      auto AS = addrSpaceOf(Load.getAddr());
      auto Ty = mslType(Load.getType());
      if (!LV)
        return LV.takeError();
      if (!AS)
        return AS.takeError();
      if (!Ty)
        return Ty.takeError();
      return Expr{"*((volatile " + *AS + " " + *Ty + " *)&" + LV->Text +
                      ")", PrecUnary};
    }
    if (auto VecTy = mlir::dyn_cast<mlir::VectorType>(Load.getType())) {
      std::string BaseName;
      if (auto AO = Load.getAddr().getDefiningOp<mlir::LLVM::AddressOfOp>()) {
        auto Mod = Entry->getParentOfType<mlir::ModuleOp>();
        if (auto G =
                Mod.lookupSymbol<mlir::LLVM::GlobalOp>(AO.getGlobalNameAttr());
            G && G.getAddrSpace() == 3 &&
            mlir::isa<mlir::VectorType>(G.getGlobalType())) {
          BaseName = GlobalNames.lookup(G);
        }
      } else if (auto AL = Load.getAddr().getDefiningOp<mlir::LLVM::AllocaOp>();
                 AL && mlir::isa<mlir::VectorType>(AL.getElemType())) {
        BaseName = ValueNames.lookup(AL.getResult());
      } else if (auto GEP = Load.getAddr().getDefiningOp<mlir::LLVM::GEPOp>()) {
        if (auto AO = GEP.getBase().getDefiningOp<mlir::LLVM::AddressOfOp>()) {
          auto Mod = Entry->getParentOfType<mlir::ModuleOp>();
          if (auto G = Mod.lookupSymbol<mlir::LLVM::GlobalOp>(
                  AO.getGlobalNameAttr());
              G && G.getAddrSpace() == 3) {
            if (auto Arr =
                    mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(G.getGlobalType());
                Arr && mlir::isa<mlir::VectorType>(Arr.getElementType())) {
              auto LV = lvalueFor(Load.getAddr());
              if (!LV)
                return LV.takeError();
              BaseName = LV->Text;
            }
          }
        }
      }
      if (!BaseName.empty()) {
        auto VTy = mslType(VecTy);
        if (!VTy)
          return VTy.takeError();
        std::string S = *VTy + "(";
        for (int64_t I = 0; I < VecTy.getNumElements(); ++I) {
          if (I)
            S += ", ";
          S += BaseName + "[" + std::to_string(I) + "]";
        }
        S += ")";
        return Expr{std::move(S), PrecPrimary};
      }
    }
    return lvalueFor(Load.getAddr());
  }

  if (auto EE = mlir::dyn_cast<ExtractElementOp>(Op)) {
    if (auto *Prod = EE.getVector().getDefiningOp()) {
      if (auto Pos = constInt(EE.getPosition())) {
        if (Prod->hasAttr("crucible.volatile")) {
          auto It = ValueNames.find(EE.getVector());
          if (It == ValueNames.end())
            return makeError("volatile load used before materialization");
          return Expr{It->second + "." + std::string(1, "xyzw"[*Pos & 3]),
                      PrecPrimary};
        }
        if (mlir::isa<ir::BufferLoadOp, ir::CBufferLoadOp>(Prod)) {
          if (auto It = ValueNames.find(EE.getVector());
              It != ValueNames.end()) {
            return Expr{It->second + "." + std::string(1, "xyzw"[*Pos & 3]),
                        PrecPrimary};
          }
        }
        if (mlir::isa<ir::BufferLoadOp, ir::CBufferLoadOp,
                      ir::TexGetDimensionsOp>(Prod))
          return crucibleExtractExpr(Prod, static_cast<unsigned>(*Pos),
                                     EE.getType());
        if (mlir::isa<ir::TexLoadOp, ir::TexSampleOp, ir::TexGatherOp,
                      ir::WaveBallotOp, ir::Unpack4x8Op>(Prod)) {
          auto It = ValueNames.find(EE.getVector());
          if (It == ValueNames.end())
            return makeError("use of materialized vector before materialization");
          return Expr{It->second + "." + std::string(1, "xyzw"[*Pos & 3]),
                      PrecPrimary};
        }
      }
      if (auto CB = mlir::dyn_cast<ir::CBufferLoadOp>(Prod)) {
        auto RName = resourceExpr(CB.getHandle());
        auto Row = exprFor(CB.getRegIndex(), PrecUnknown);
        auto Idx = exprFor(EE.getPosition(), PrecUnknown);
        auto Ty = mslType(EE.getType());
        if (!RName || !Row || !Idx || !Ty)
          return makeError("invalid dynamic cbuffer extractelement");
        return Expr{"((constant " + *Ty + " *)(&(" + *RName + ")[" + Row->Text +
                        "]))[" + Idx->Text + "]",
                    PrecPrimary};
      }
    }
    auto Vec = exprFor(EE.getVector(), PrecPrimary);
    auto Idx = exprFor(EE.getPosition(), PrecUnknown);
    if (!Vec || !Idx)
      return makeError("invalid extractelement operand");
    return Expr{Vec->Text + "[" + Idx->Text + "]", PrecPrimary};
  }

  if (auto IE = mlir::dyn_cast<InsertElementOp>(Op)) {
    auto VecTy = mlir::dyn_cast<mlir::VectorType>(IE.getType());
    if (!VecTy || VecTy.isScalable() || VecTy.getRank() != 1)
      return makeError("unsupported insertelement vector type");
    unsigned N = static_cast<unsigned>(VecTy.getNumElements());
    SmallVector<mlir::Value, 4> Lanes(N);
    mlir::Value Cur = IE.getResult();
    while (auto CurIE = Cur.getDefiningOp<InsertElementOp>()) {
      auto Pos = constInt(CurIE.getPosition());
      if (!Pos || *Pos >= N)
        return makeError("dynamic or out-of-range insertelement index");
      if (!Lanes[*Pos])
        Lanes[*Pos] = CurIE.getValue();
      Cur = CurIE.getVector();
    }
    auto VecName = mslType(VecTy);
    auto ElemName = mslType(VecTy.getElementType());
    if (!VecName || !ElemName)
      return makeError("unsupported insertelement element type");
    bool BaseIsUndef = isUndefLike(Cur) ||
                       (Cur.getDefiningOp() &&
                        mlir::isa<mlir::LLVM::ZeroOp>(Cur.getDefiningOp()));
    std::string BaseText;
    if (!BaseIsUndef) {
      auto BE = exprFor(Cur, PrecPrimary);
      if (!BE)
        return BE.takeError();
      BaseText = BE->Text;
    }
    std::string ArgsText;
    for (unsigned I = 0; I < N; ++I) {
      if (I)
        ArgsText += ", ";
      if (Lanes[I]) {
        auto LE = exprFor(Lanes[I], PrecUnknown);
        if (!LE)
          return LE.takeError();
        ArgsText += LE->Text;
      } else if (!BaseIsUndef) {
        ArgsText += BaseText + "[" + std::to_string(I) + "]";
      } else {
        ArgsText += *ElemName + "(0)";
      }
    }
    return Expr{*VecName + "(" + ArgsText + ")", PrecPrimary};
  }

  if (auto SV = mlir::dyn_cast<mlir::LLVM::ShuffleVectorOp>(Op)) {
    auto ResVec = mlir::dyn_cast<mlir::VectorType>(SV.getType());
    auto V1Ty = mlir::dyn_cast<mlir::VectorType>(SV.getV1().getType());
    if (!ResVec || !V1Ty)
      return makeError("unsupported shufflevector types");
    auto VecName = mslType(ResVec);
    auto ElemName = mslType(ResVec.getElementType());
    if (!VecName || !ElemName)
      return makeError("unsupported shufflevector element type");
    auto E1 = exprFor(SV.getV1(), PrecPrimary);
    auto E2 = exprFor(SV.getV2(), PrecPrimary);
    if (!E1 || !E2)
      return makeError("invalid shufflevector operand");
    int64_t N1 = V1Ty.getNumElements();
    std::string ArgsText;
    auto Mask = SV.getMask();
    for (size_t I = 0; I < Mask.size(); ++I) {
      if (I)
        ArgsText += ", ";
      int32_t M = Mask[I];
      if (M < 0)
        ArgsText += *ElemName + "(0)";
      else if (M < N1)
        ArgsText += E1->Text + "[" + std::to_string(M) + "]";
      else
        ArgsText += E2->Text + "[" + std::to_string(M - N1) + "]";
    }
    return Expr{*VecName + "(" + ArgsText + ")", PrecPrimary};
  }

  if (auto EV = mlir::dyn_cast<ExtractValueOp>(Op))
    return bufferElementExpr(EV);

  // Layer 1 GPU compute IDs.
  auto gpuDimSwizzle = [](mlir::gpu::Dimension D) -> char {
    switch (D) {
    case mlir::gpu::Dimension::x:
      return 'x';
    case mlir::gpu::Dimension::y:
      return 'y';
    case mlir::gpu::Dimension::z:
      return 'z';
    }
    return 'x';
  };
  if (auto G = mlir::dyn_cast<mlir::gpu::GlobalIdOp>(Op)) {
    UsesDTid = true;
    return Expr{std::string("dtid.") + gpuDimSwizzle(G.getDimension()),
                PrecPrimary};
  }
  if (auto B = mlir::dyn_cast<mlir::gpu::BlockIdOp>(Op)) {
    UsesGid = true;
    return Expr{std::string("gid.") + gpuDimSwizzle(B.getDimension()),
                PrecPrimary};
  }
  if (auto T = mlir::dyn_cast<mlir::gpu::ThreadIdOp>(Op)) {
    UsesGTid = true;
    return Expr{std::string("gtid.") + gpuDimSwizzle(T.getDimension()),
                PrecPrimary};
  }
  if (mlir::isa<mlir::gpu::LaneIdOp>(Op)) {
    UsesLane = true;
    return Expr{"lane", PrecPrimary};
  }
  if (mlir::isa<mlir::gpu::SubgroupSizeOp>(Op)) {
    UsesLaneCount = true;
    return Expr{"lanecount", PrecPrimary};
  }

  // Layer 1 GPU subgroup / wave operations.
  if (auto Red = mlir::dyn_cast<mlir::gpu::SubgroupReduceOp>(Op)) {
    auto V = exprFor(Red.getValue(), PrecUnknown);
    if (!V)
      return V.takeError();
    mlir::Type T = Red.getType();
    auto ScanAttr = Red->getAttrOfType<mlir::StringAttr>("crucible.scan_kind");
    StringRef Scan = ScanAttr ? ScanAttr.getValue() : "";
    bool IsSigned = false;
    StringRef Fn;
    switch (Red.getOp()) {
    case mlir::gpu::AllReduceOperation::ADD:
      Fn = Scan == "exclusive"   ? "simd_prefix_exclusive_sum"
           : Scan == "inclusive" ? "simd_prefix_inclusive_sum"
                                 : "simd_sum";
      break;
    case mlir::gpu::AllReduceOperation::MUL:
      Fn = Scan == "exclusive"   ? "simd_prefix_exclusive_product"
           : Scan == "inclusive" ? "simd_prefix_inclusive_product"
                                 : "simd_product";
      break;
    case mlir::gpu::AllReduceOperation::MINUI:
    case mlir::gpu::AllReduceOperation::MINNUMF:
    case mlir::gpu::AllReduceOperation::MINIMUMF:
      Fn = "simd_min";
      break;
    case mlir::gpu::AllReduceOperation::MINSI:
      Fn = "simd_min";
      IsSigned = true;
      break;
    case mlir::gpu::AllReduceOperation::MAXUI:
    case mlir::gpu::AllReduceOperation::MAXNUMF:
    case mlir::gpu::AllReduceOperation::MAXIMUMF:
      Fn = "simd_max";
      break;
    case mlir::gpu::AllReduceOperation::MAXSI:
      Fn = "simd_max";
      IsSigned = true;
      break;
    case mlir::gpu::AllReduceOperation::AND:
      Fn = "simd_and";
      break;
    case mlir::gpu::AllReduceOperation::OR:
      Fn = "simd_or";
      break;
    case mlir::gpu::AllReduceOperation::XOR:
      Fn = "simd_xor";
      break;
    }
    bool IsFloat = mlir::isa<mlir::FloatType>(T);
    bool IsAdd = Red.getOp() == mlir::gpu::AllReduceOperation::ADD;
    bool IsMul = Red.getOp() == mlir::gpu::AllReduceOperation::MUL;
    bool IsBitwise = Red.getOp() == mlir::gpu::AllReduceOperation::AND ||
                     Red.getOp() == mlir::gpu::AllReduceOperation::OR ||
                     Red.getOp() == mlir::gpu::AllReduceOperation::XOR;
    bool IsExclusive = Scan == "exclusive";
    bool IsInclusive = Scan == "inclusive";
    if (IsBitwise && T.isInteger(64))
      return Expr{"as_type<ulong>(" + Fn.str() + "(as_type<uint2>(" + V->Text +
                      ")))",
                  PrecPrimary};
    if (!IsBitwise && (T.isF64() || T.isInteger(64))) {
      int WideKind = IsAdd ? 0
                   : IsMul ? 1
                   : (Fn == "simd_min") ? 2
                                        : 3;
      if (T.isF64()) {
        requireHelper("crucible_f64");
        requireHelper("crucible_wave_wide_f64");
        UsesLane |= IsExclusive;
        return Expr{"crucible_wave_wide_f64<" + std::to_string(WideKind) +
                        ", " + (IsExclusive ? "true" : "false") + ">(" +
                        V->Text + ", " + (IsExclusive ? "lane" : "0u") + ")",
                    PrecPrimary};
      }
      requireHelper("crucible_wave_wide");
      UsesLane |= IsExclusive;
      std::string Text =
          std::string("crucible_wave_wide<") + (IsSigned ? "long" : "ulong") +
          ", " + std::to_string(WideKind) + ", " +
          (IsExclusive ? "true" : "false") + ">(" +
          (IsSigned ? signedView(T, V->Text) : V->Text) + ", " +
          (IsExclusive ? "lane" : "0u") + ")";
      return Expr{IsSigned ? unsignedView(T, Text) : Text, PrecPrimary};
    }
    if (IsSigned)
      return Expr{unsignedView(T, (Fn + "(" + signedView(T, V->Text) + ")").str()),
                  PrecPrimary};
    if (IsExclusive && Fn == "simd_prefix_exclusive_sum" && IsFloat) {
      requireHelper("crucible_wave_prefix_sum");
      Fn = "crucible_wave_prefix_sum";
    }
    if (T.isInteger(16) && IsAdd && !IsInclusive) {
      // Some Metal compilers merge identical narrow reductions across
      // different control-flow masks. Capture the participating lanes.
      requireHelper("crucible_wave_wide");
      UsesLane |= IsExclusive;
      return Expr{std::string("ushort(crucible_wave_wide<ulong, 0, ") +
                      (IsExclusive ? "true>(ulong(" : "false>(ulong(") +
                      V->Text + "), " + (IsExclusive ? "lane" : "0u") + "))",
                  PrecPrimary};
    }
    if (T.isInteger(16) && IsMul)
      return Expr{"ushort(" + Fn.str() + "(uint(" + V->Text + ")))",
                  PrecPrimary};
    if (T.isF16() && (IsAdd || IsMul))
      return Expr{"half(" + Fn.str() + "(float(" + V->Text + ")))",
                  PrecPrimary};
    return Expr{(Fn + "(" + V->Text + ")").str(), PrecPrimary};
  }
  if (auto BC = mlir::dyn_cast<mlir::gpu::SubgroupBroadcastOp>(Op)) {
    auto V = exprFor(BC.getSrc(), PrecUnknown);
    if (!V)
      return V.takeError();
    mlir::Type T = BC.getSrc().getType();
    std::string Fn, LaneArg;
    if (BC.getBroadcastType() == mlir::gpu::BroadcastType::first_active_lane) {
      Fn = "simd_broadcast_first";
    } else {
      auto Idx = exprFor(BC->getOperand(1), PrecUnknown);
      if (!Idx)
        return Idx.takeError();
      Fn = "simd_broadcast";
      LaneArg = ", ushort(" + Idx->Text + ")";
    }
    if (T.isInteger(1))
      return Expr{"bool(" + Fn + "(uint(" + V->Text + ")" + LaneArg + "))",
                  PrecPrimary};
    if (T.isInteger(64) || T.isF64())
      return Expr{"as_type<ulong>(" + Fn + "(as_type<uint2>(" + V->Text + ")" +
                      LaneArg + "))",
                  PrecPrimary};
    return Expr{Fn + "(" + V->Text + LaneArg + ")", PrecPrimary};
  }
  if (auto Bal = mlir::dyn_cast<mlir::gpu::BallotOp>(Op)) {
    auto Pred = exprFor(Bal.getPredicate(), PrecUnknown);
    if (!Pred)
      return Pred.takeError();
    auto DstTy = mslType(Bal.getType());
    if (!DstTy)
      return DstTy.takeError();
    return Expr{*DstTy + "(uint64_t(simd_ballot(" + Pred->Text + ")))",
                PrecPrimary};
  }
  if (auto Shuf = mlir::dyn_cast<mlir::gpu::ShuffleOp>(Op)) {
    auto V = exprFor(Shuf.getValue(), PrecUnknown);
    auto Off = exprFor(Shuf.getOffset(), PrecUnknown);
    if (!V || !Off)
      return makeError("invalid gpu.shuffle operand");
    StringRef Fn;
    switch (Shuf.getMode()) {
    case mlir::gpu::ShuffleMode::IDX:
      Fn = "simd_shuffle";
      break;
    case mlir::gpu::ShuffleMode::XOR:
      Fn = "simd_shuffle_xor";
      break;
    case mlir::gpu::ShuffleMode::UP:
      Fn = "simd_shuffle_up";
      break;
    case mlir::gpu::ShuffleMode::DOWN:
      Fn = "simd_shuffle_down";
      break;
    }
    mlir::Type T = Shuf.getValue().getType();
    std::string LaneArg = ", ushort(" + Off->Text + ")";
    if (T.isInteger(1))
      return Expr{"bool(" + Fn.str() + "(uint(" + V->Text + ")" + LaneArg +
                      "))",
                  PrecPrimary};
    if (T.isInteger(64) || T.isF64())
      return Expr{"as_type<ulong>(" + Fn.str() + "(as_type<uint2>(" + V->Text +
                      ")" + LaneArg + "))",
                  PrecPrimary};
    return Expr{Fn.str() + "(" + V->Text + LaneArg + ")", PrecPrimary};
  }

  // Layer 1 math & LLVM dialect intrinsics (with software f64 emulation).
  if (Opts.EmulateF64 &&
      ((Op->getNumResults() == 1 && Op->getResult(0).getType().isF64()) ||
       (Op->getNumOperands() >= 1 && Op->getOperand(0).getType().isF64()))) {
    if (mlir::isa<mlir::LLVM::FAbsOp, mlir::math::AbsFOp>(Op)) {
      auto A = exprFor(Op->getOperand(0), PrecAnd);
      if (!A)
        return A.takeError();
      return Expr{A->Text + " & 0x7ffffffffffffffful", PrecAnd};
    }
    if (mlir::isa<mlir::math::IsNaNOp, mlir::math::IsInfOp,
                  mlir::math::IsFiniteOp, mlir::LLVM::SqrtOp, mlir::math::SqrtOp,
                  mlir::LLVM::TruncOp, mlir::math::TruncOp,
                  mlir::LLVM::FFloorOp, mlir::math::FloorOp,
                  mlir::LLVM::FCeilOp, mlir::math::CeilOp,
                  mlir::LLVM::RoundEvenOp, mlir::LLVM::RintOp,
                  mlir::math::RoundEvenOp>(Op)) {
      requireHelper("crucible_f64");
      auto A = exprFor(Op->getOperand(0), PrecUnknown);
      if (!A)
        return A.takeError();
      if (mlir::isa<mlir::math::IsNaNOp>(Op))
        return Expr{"crucible_f64_isnan(" + A->Text + ")", PrecPrimary};
      if (mlir::isa<mlir::math::IsInfOp>(Op))
        return Expr{"crucible_f64_isinf(" + A->Text + ")", PrecPrimary};
      if (mlir::isa<mlir::math::IsFiniteOp>(Op))
        return Expr{"crucible_f64_isfinite(" + A->Text + ")", PrecPrimary};
      if (mlir::isa<mlir::LLVM::SqrtOp, mlir::math::SqrtOp>(Op))
        return Expr{"crucible_f64_sqrt(" + A->Text + ")", PrecPrimary};
      if (mlir::isa<mlir::LLVM::TruncOp, mlir::math::TruncOp>(Op))
        return Expr{"crucible_f64_round(" + A->Text + ", 0)", PrecPrimary};
      if (mlir::isa<mlir::LLVM::FFloorOp, mlir::math::FloorOp>(Op))
        return Expr{"crucible_f64_round(" + A->Text + ", 1)", PrecPrimary};
      if (mlir::isa<mlir::LLVM::FCeilOp, mlir::math::CeilOp>(Op))
        return Expr{"crucible_f64_round(" + A->Text + ", 2)", PrecPrimary};
      return Expr{"crucible_f64_round(" + A->Text + ", 3)", PrecPrimary};
    }
    if (mlir::isa<mlir::LLVM::MinNumOp, mlir::LLVM::MinimumOp,
                  mlir::LLVM::MaxNumOp, mlir::LLVM::MaximumOp>(Op)) {
      requireHelper("crucible_f64");
      auto A = exprFor(Op->getOperand(0), PrecUnknown);
      auto B = exprFor(Op->getOperand(1), PrecUnknown);
      if (!A || !B)
        return makeError("invalid f64 min/max operand");
      bool IsMin = mlir::isa<mlir::LLVM::MinNumOp, mlir::LLVM::MinimumOp>(Op);
      return Expr{std::string(IsMin ? "crucible_f64_min(" : "crucible_f64_max(") +
                      A->Text + ", " + B->Text + ")",
                  PrecPrimary};
    }
    if (mlir::isa<mlir::LLVM::FMAOp, mlir::LLVM::FMulAddOp, mlir::math::FmaOp>(
            Op)) {
      requireHelper("crucible_f64");
      auto A = exprFor(Op->getOperand(0), PrecUnknown);
      auto B = exprFor(Op->getOperand(1), PrecUnknown);
      auto C = exprFor(Op->getOperand(2), PrecUnknown);
      if (!A || !B || !C)
        return makeError("invalid f64 fma operand");
      return Expr{"crucible_f64_fma(" + A->Text + ", " + B->Text + ", " +
                      C->Text + ")",
                  PrecPrimary};
    }
  }

  auto isF32OrF32Vec = [](mlir::Type T) {
    if (T.isF32())
      return true;
    if (auto VT = mlir::dyn_cast<mlir::VectorType>(T))
      return VT.getElementType().isF32() && VT.getNumElements() >= 2 &&
             VT.getNumElements() <= 4;
    return false;
  };
  auto mathFunctionName = [&](StringRef Fn) {
    bool Precise = R.RequiresStrictFP &&
                   isF32OrF32Vec(Op->getResult(0).getType()) &&
                   (Fn == "acosh" || Fn == "cosh" || Fn == "exp" ||
                    Fn == "log2" || Fn == "atan2");
    return Precise ? ("precise::" + Fn).str() : Fn.str();
  };
  auto emitUnaryCall = [&](StringRef Fn) -> Expected<Expr> {
    auto A = exprFor(Op->getOperand(0), PrecUnknown);
    if (!A)
      return A.takeError();
    std::string Name = mathFunctionName(Fn);
    if (Opts.FlushDenormsF32 && isF32OrF32Vec(Op->getResult(0).getType())) {
      requireHelper("crucible_ftz");
      return Expr{"crucible_ftz(" + Name + "(crucible_ftz(" + A->Text + ")))",
                  PrecPrimary};
    }
    return Expr{Name + "(" + A->Text + ")", PrecPrimary};
  };
  auto emitBinaryCall = [&](StringRef Fn) -> Expected<Expr> {
    auto A = exprFor(Op->getOperand(0), PrecUnknown);
    auto B = exprFor(Op->getOperand(1), PrecUnknown);
    if (!A || !B)
      return makeError("invalid binary math operand");
    return Expr{mathFunctionName(Fn) + "(" + A->Text + ", " + B->Text + ")",
                PrecPrimary};
  };
  if (auto Scale = mlir::dyn_cast<mlir::LLVM::LoadExpOp>(Op)) {
    mlir::Type Ty = Scale.getType(), ExpTy = Scale.getExp().getType();
    if (!isF32OrF32Vec(Ty))
      return makeError("ldexp supports only scalar/vector float32");
    auto VT = mlir::dyn_cast<mlir::VectorType>(Ty);
    auto EV = mlir::dyn_cast<mlir::VectorType>(ExpTy);
    if ((VT && (!EV || VT.getNumElements() != EV.getNumElements())) ||
        (!VT && EV))
      return makeError("ldexp operand shapes must match");
    mlir::Type Integer = EV ? EV.getElementType() : ExpTy;
    if (!Integer.isInteger(32) && !Integer.isInteger(64))
      return makeError("ldexp requires integer32/64 exponents");
    auto X = exprFor(Scale.getVal(), PrecUnknown);
    auto E = exprFor(Scale.getExp(), PrecUnknown);
    if (!X)
      return X.takeError();
    if (!E)
      return E.takeError();
    std::string Suffix = VT ? std::to_string(VT.getNumElements()) : "";
    std::string Int = "int" + Suffix;
    std::string Signed = (Integer.isInteger(64) ? "long" : "int") + Suffix;
    // Beyond +/-512 every finite nonzero float32 over/underflows, even
    // subnormals. Clamp while still signed/full-width, before narrowing.
    std::string Exponent = "clamp(" + signedView(ExpTy, E->Text) + ", " +
                           Signed + "(-512), " + Signed + "(512))";
    if (Integer.isInteger(64))
      Exponent = Int + "(" + Exponent + ")";
    requireHelper("crucible_ldexp");
    std::string Value = X->Text;
    if (Opts.FlushDenormsF32) {
      requireHelper("crucible_ftz");
      Value = "crucible_ftz(" + Value + ")";
    }
    std::string Call = "crucible_ldexp(" + Value + ", " + Exponent + ")";
    if (Opts.FlushDenormsF32)
      Call = "crucible_ftz(" + Call + ")";
    return Expr{Call, PrecPrimary};
  }
  if (mlir::isa<mlir::math::AcoshOp>(Op))
    return emitUnaryCall("acosh");
  if (mlir::isa<mlir::LLVM::FAbsOp, mlir::math::AbsFOp>(Op))
    return emitUnaryCall("fabs");
  if (mlir::isa<mlir::math::IsNaNOp>(Op))
    return emitUnaryCall("isnan");
  if (mlir::isa<mlir::math::IsInfOp>(Op))
    return emitUnaryCall("isinf");
  if (mlir::isa<mlir::math::IsFiniteOp>(Op))
    return emitUnaryCall("isfinite");
  if (mlir::isa<mlir::LLVM::SqrtOp, mlir::math::SqrtOp>(Op))
    return emitUnaryCall("sqrt");
  if (mlir::isa<mlir::math::RsqrtOp>(Op))
    return emitUnaryCall("rsqrt");
  if (mlir::isa<mlir::LLVM::SinOp, mlir::math::SinOp>(Op))
    return emitUnaryCall("sin");
  if (mlir::isa<mlir::LLVM::CosOp, mlir::math::CosOp>(Op))
    return emitUnaryCall("cos");
  if (mlir::isa<mlir::LLVM::TanOp, mlir::math::TanOp>(Op))
    return emitUnaryCall("tan");
  if (mlir::isa<mlir::LLVM::ASinOp, mlir::math::AsinOp>(Op))
    return emitUnaryCall("asin");
  if (mlir::isa<mlir::LLVM::ACosOp, mlir::math::AcosOp>(Op))
    return emitUnaryCall("acos");
  if (mlir::isa<mlir::LLVM::ATanOp, mlir::math::AtanOp>(Op))
    return emitUnaryCall("atan");
  if (mlir::isa<mlir::LLVM::SinhOp, mlir::math::SinhOp>(Op))
    return emitUnaryCall("sinh");
  if (mlir::isa<mlir::LLVM::CoshOp, mlir::math::CoshOp>(Op))
    return emitUnaryCall("cosh");
  if (mlir::isa<mlir::LLVM::TanhOp, mlir::math::TanhOp>(Op))
    return emitUnaryCall("tanh");
  if (mlir::isa<mlir::LLVM::ExpOp, mlir::math::ExpOp>(Op))
    return emitUnaryCall("exp");
  if (mlir::isa<mlir::LLVM::Exp2Op, mlir::math::Exp2Op>(Op)) {
    // Strength-reduce DXC's lowered `pow(x, C)` -> `exp2(log2(x) * C)` for
    // small integer and half-integer constants, replacing pairs of SFU
    // transcendentals with FPU multiplies / sqrt (outside strict FP mode).
    mlir::Type ResTy = Op->getResult(0).getType();
    if (!R.RequiresStrictFP && !Opts.EmulateF64 &&
        (ResTy.isF32() || ResTy.isF16())) {
      if (auto Mul = Op->getOperand(0).getDefiningOp<FMulOp>()) {
        auto MatchLog2AndConst = [](mlir::Value A, mlir::Value B,
                                    mlir::Value &OutX, double &OutC) -> bool {
          auto *LogDef = A.getDefiningOp();
          if (!LogDef || !mlir::isa<mlir::LLVM::Log2Op, mlir::math::Log2Op>(LogDef))
            return false;
          auto Cst = B.getDefiningOp<ConstantOp>();
          if (!Cst)
            return false;
          auto FA = mlir::dyn_cast<mlir::FloatAttr>(Cst.getValue());
          if (!FA)
            return false;
          OutX = LogDef->getOperand(0);
          OutC = FA.getValueAsDouble();
          return true;
        };
        mlir::Value BaseX;
        double ExpC = 0.0;
        if (MatchLog2AndConst(Mul.getLhs(), Mul.getRhs(), BaseX, ExpC) ||
            MatchLog2AndConst(Mul.getRhs(), Mul.getLhs(), BaseX, ExpC)) {
          if (ExpC == 0.5) {
            auto XE = exprFor(BaseX, PrecUnknown);
            if (!XE)
              return XE.takeError();
            return Expr{"sqrt(" + XE->Text + ")", PrecPrimary};
          }
          if (ExpC == -0.5) {
            auto XE = exprFor(BaseX, PrecUnknown);
            if (!XE)
              return XE.takeError();
            return Expr{"rsqrt(" + XE->Text + ")", PrecPrimary};
          }
          if (ExpC == 2.5) {
            auto XE = exprFor(BaseX, PrecUnknown);
            if (!XE)
              return XE.takeError();
            requireHelper("crucible_pow2_5");
            return Expr{"crucible_pow2_5(" + XE->Text + ")", PrecPrimary};
          }
          int IntExp = static_cast<int>(ExpC);
          if (ExpC == static_cast<double>(IntExp) && IntExp >= 2 &&
              IntExp <= 16) {
            auto XE = exprFor(BaseX, PrecUnknown);
            if (!XE)
              return XE.takeError();
            requireHelper("crucible_powi");
            return Expr{"crucible_powi<" + std::to_string(IntExp) + ">(" +
                            XE->Text + ")",
                        PrecPrimary};
          }
        }
      }
    }
    return emitUnaryCall("exp2");
  }
  if (mlir::isa<mlir::LLVM::LogOp, mlir::math::LogOp>(Op))
    return emitUnaryCall("log");
  if (mlir::isa<mlir::LLVM::Log2Op, mlir::math::Log2Op>(Op))
    return emitUnaryCall("log2");
  if (mlir::isa<mlir::LLVM::FFloorOp, mlir::math::FloorOp>(Op))
    return emitUnaryCall("floor");
  if (mlir::isa<mlir::LLVM::FCeilOp, mlir::math::CeilOp>(Op))
    return emitUnaryCall("ceil");
  if (mlir::isa<mlir::LLVM::FTruncOp, mlir::math::TruncOp>(Op))
    return emitUnaryCall("trunc");
  if (mlir::isa<mlir::LLVM::RoundEvenOp, mlir::LLVM::RintOp,
                mlir::math::RoundEvenOp>(Op))
    return emitUnaryCall("rint");
  if (mlir::isa<mlir::LLVM::RoundOp, mlir::math::RoundOp>(Op))
    return emitUnaryCall("round");
  if (mlir::isa<mlir::LLVM::MinNumOp, mlir::LLVM::MinimumOp>(Op))
    return emitBinaryCall("min");
  if (mlir::isa<mlir::LLVM::MaxNumOp, mlir::LLVM::MaximumOp>(Op))
    return emitBinaryCall("max");
  if (mlir::isa<mlir::LLVM::PowOp, mlir::math::PowFOp>(Op))
    return emitBinaryCall("pow");
  if (mlir::isa<mlir::LLVM::ATan2Op, mlir::math::Atan2Op>(Op))
    return emitBinaryCall("atan2");
  if (mlir::isa<mlir::LLVM::CopySignOp, mlir::math::CopySignOp>(Op))
    return emitBinaryCall("copysign");
  if (mlir::isa<mlir::LLVM::AbsOp>(Op)) {
    auto A = exprFor(Op->getOperand(0), PrecUnknown);
    if (!A)
      return A.takeError();
    mlir::Type T = Op->getResult(0).getType();
    return Expr{unsignedView(T, "abs(" + signedView(T, A->Text) + ")"),
                PrecPrimary};
  }
  if (mlir::isa<mlir::LLVM::FMAOp, mlir::LLVM::FMulAddOp, mlir::math::FmaOp>(
          Op)) {
    auto A = exprFor(Op->getOperand(0), PrecUnknown);
    auto B = exprFor(Op->getOperand(1), PrecUnknown);
    auto C = exprFor(Op->getOperand(2), PrecUnknown);
    if (!A || !B || !C)
      return makeError("invalid fma operand");
    return Expr{"fma(" + A->Text + ", " + B->Text + ", " + C->Text + ")",
                PrecPrimary};
  }
  if (mlir::isa<mlir::LLVM::UMinOp>(Op))
    return emitBinaryCall("min");
  if (mlir::isa<mlir::LLVM::UMaxOp>(Op))
    return emitBinaryCall("max");
  if (mlir::isa<mlir::LLVM::SMinOp, mlir::LLVM::SMaxOp>(Op)) {
    auto A = exprFor(Op->getOperand(0), PrecUnknown);
    auto B = exprFor(Op->getOperand(1), PrecUnknown);
    if (!A || !B)
      return makeError("invalid signed min/max operand");
    mlir::Type T = Op->getResult(0).getType();
    StringRef Fn = mlir::isa<mlir::LLVM::SMinOp>(Op) ? "min" : "max";
    return Expr{unsignedView(T, (Fn + "(" + signedView(T, A->Text) + ", " +
                                 signedView(T, B->Text) + ")")
                                    .str()),
                PrecPrimary};
  }
  if (mlir::isa<mlir::LLVM::CtPopOp, mlir::math::CtPopOp>(Op))
    return emitUnaryCall("popcount");
  if (mlir::isa<mlir::LLVM::BitReverseOp>(Op))
    return emitUnaryCall("reverse_bits");
  if (mlir::isa<mlir::LLVM::CountLeadingZerosOp,
                mlir::math::CountLeadingZerosOp>(Op))
    return emitUnaryCall("clz");
  if (mlir::isa<mlir::LLVM::CountTrailingZerosOp,
                mlir::math::CountTrailingZerosOp>(Op))
    return emitUnaryCall("ctz");

  // Crucible dialect ops (frontend-agnostic shader IR).
  if (auto SV = mlir::dyn_cast<ir::SysValOp>(Op)) {
    switch (SV.getKind()) {
    case ir::SysValKind::FlattenedThreadIdInGroup:
      UsesGidx = true;
      return Expr{"gidx", PrecPrimary};
    case ir::SysValKind::PrimitiveId:
      if (R.Geometry)
        return Expr{"crucible_prim_id", PrecPrimary};
      return Expr{"crucible_patch_id", PrecPrimary};
    case ir::SysValKind::OutputControlPointId:
      return Expr{"crucible_cp_id", PrecPrimary};
    case ir::SysValKind::DomainLocation: {
      uint32_t Lane = SV.getComponent().value_or(0);
      unsigned Dim =
          (R.Domain && R.Domain->Domain == TessellationDomain::Triangle) ? 3
                                                                         : 2;
      if (Lane >= Dim)
        return makeError("invalid domain location component");
      return Expr{"crucible_domain_location[" + std::to_string(Lane) + "]",
                  PrecPrimary};
    }
    case ir::SysValKind::DispatchRaysIndex: {
      uint32_t Axis = SV.getComponent().value_or(0);
      if (Axis > 2)
        return makeError("invalid DispatchRaysIndex axis");
      UsesDTid = true;
      static const char *const Axes[] = {"dtid.x", "dtid.y", "dtid.z"};
      return Expr{Axes[Axis], PrecPrimary};
    }
    case ir::SysValKind::DispatchRaysDimensions: {
      uint32_t Axis = SV.getComponent().value_or(0);
      if (Axis > 2)
        return makeError("invalid DispatchRaysDimensions axis");
      UsesGridSize = true;
      static const char *const Axes[] = {"grid_size.x", "grid_size.y",
                                         "grid_size.z"};
      return Expr{Axes[Axis], PrecPrimary};
    }
    case ir::SysValKind::WorldRayOrigin: {
      uint32_t Axis = SV.getComponent().value_or(0);
      if (Axis > 2)
        return makeError("invalid WorldRayOrigin axis");
      static const char *const Axes[] = {"crucible_ray_state.world_origin.x",
                                         "crucible_ray_state.world_origin.y",
                                         "crucible_ray_state.world_origin.z"};
      return Expr{Axes[Axis], PrecPrimary};
    }
    case ir::SysValKind::WorldRayDirection: {
      uint32_t Axis = SV.getComponent().value_or(0);
      if (Axis > 2)
        return makeError("invalid WorldRayDirection axis");
      static const char *const Axes[] = {"crucible_ray_state.world_dir.x",
                                         "crucible_ray_state.world_dir.y",
                                         "crucible_ray_state.world_dir.z"};
      return Expr{Axes[Axis], PrecPrimary};
    }
    case ir::SysValKind::RayTCurrent:
      return Expr{"crucible_ray_state.t_current", PrecPrimary};
    case ir::SysValKind::PrimitiveIndex:
      return Expr{"crucible_ray_state.primitive_index", PrecPrimary};
    default:
      return makeError("unsupported crucible.sysval kind");
    }
  }
  if (auto Sat = mlir::dyn_cast<ir::SaturateOp>(Op))
    return emitUnaryCall("saturate");
  if (auto Frc = mlir::dyn_cast<ir::FrcOp>(Op)) {
    if (Opts.EmulateF64 && Frc.getResult().getType().isF64()) {
      requireHelper("crucible_f64");
      auto A = exprFor(Frc.getValue(), PrecUnknown);
      if (!A)
        return A.takeError();
      return Expr{"crucible_f64_sub(" + A->Text + ", crucible_f64_round(" +
                      A->Text + ", 1))",
                  PrecPrimary};
    }
    auto A = exprFor(Frc.getValue(), PrecAdd);
    auto AF = exprFor(Frc.getValue(), PrecUnknown);
    if (!A || !AF)
      return makeError("invalid crucible.frc operand");
    return Expr{A->Text + " - floor(" + AF->Text + ")", PrecAdd};
  }
  if (auto FB = mlir::dyn_cast<ir::FirstBitOp>(Op)) {
    const char *Helper = FB.getKind() == ir::FirstBitKind::Lo
                             ? "crucible_firstbitlo"
                         : FB.getKind() == ir::FirstBitKind::Hi
                             ? "crucible_firstbithi"
                             : "crucible_firstbitshi";
    requireHelper(Helper);
    auto A = exprFor(FB.getValue(), PrecUnknown);
    if (!A)
      return A.takeError();
    return Expr{std::string(Helper) + "(" + A->Text + ")", PrecPrimary};
  }
  if (auto Bfe = mlir::dyn_cast<ir::BfeOp>(Op)) {
    const char *Helper = Bfe.getIsSigned() ? "crucible_ibfe" : "crucible_ubfe";
    requireHelper(Helper);
    auto W = exprFor(Bfe.getWidth(), PrecUnknown);
    auto O = exprFor(Bfe.getOffset(), PrecUnknown);
    auto V = exprFor(Bfe.getValue(), PrecUnknown);
    if (!W || !O || !V)
      return makeError("invalid crucible.bfe operand");
    return Expr{std::string(Helper) + "(" + W->Text + ", " + O->Text + ", " +
                    V->Text + ")",
                PrecPrimary};
  }
  if (auto Bfi = mlir::dyn_cast<ir::BfiOp>(Op)) {
    requireHelper("crucible_bfi");
    auto W = exprFor(Bfi.getWidth(), PrecUnknown);
    auto O = exprFor(Bfi.getOffset(), PrecUnknown);
    auto Ins = exprFor(Bfi.getInsert(), PrecUnknown);
    auto Base = exprFor(Bfi.getBase(), PrecUnknown);
    if (!W || !O || !Ins || !Base)
      return makeError("invalid crucible.bfi operand");
    return Expr{"crucible_bfi(" + W->Text + ", " + O->Text + ", " + Ins->Text +
                    ", " + Base->Text + ")",
                PrecPrimary};
  }
  if (auto Dot = mlir::dyn_cast<ir::DotOp>(Op)) {
    unsigned N = Dot.getLhs().size();
    std::string VecTy = (Dot.getResult().getType().isF16() ? "half" : "float") +
                        std::to_string(N);
    auto BuildVec = [&](mlir::OperandRange Elems) -> Expected<std::string> {
      std::string V = VecTy + "(";
      for (unsigned I = 0; I < N; ++I) {
        auto A = exprFor(Elems[I], PrecUnknown);
        if (!A)
          return A.takeError();
        V += (I ? ", " : "") + A->Text;
      }
      return V + ")";
    };
    auto LHS = BuildVec(Dot.getLhs());
    if (!LHS)
      return LHS.takeError();
    auto RHS = BuildVec(Dot.getRhs());
    if (!RHS)
      return RHS.takeError();
    return Expr{"dot(" + *LHS + ", " + *RHS + ")", PrecPrimary};
  }
  if (auto DP = mlir::dyn_cast<ir::DotPackedOp>(Op)) {
    auto Acc = exprFor(DP.getAcc(), PrecUnknown);
    if (!Acc)
      return Acc.takeError();
    auto Inputs = DP.getInputs();
    if (DP.getKind() == ir::DotPackedKind::F16x2) {
      std::string Dots;
      for (unsigned I = 0; I < 2; ++I) {
        auto A = exprFor(Inputs[I], PrecUnknown);
        auto B = exprFor(Inputs[2 + I], PrecUnknown);
        if (!A || !B)
          return makeError("invalid crucible.dot_packed operand");
        Dots += " + float(" + A->Text + ") * float(" + B->Text + ")";
      }
      return Expr{Acc->Text + Dots, PrecAdd};
    }
    const char *Helper = DP.getKind() == ir::DotPackedKind::I8x4
                             ? "crucible_dot4add_i8packed"
                             : "crucible_dot4add_u8packed";
    requireHelper(Helper);
    auto A = exprFor(Inputs[0], PrecUnknown);
    auto B = exprFor(Inputs[1], PrecUnknown);
    if (!A || !B)
      return makeError("invalid crucible.dot_packed operand");
    return Expr{std::string(Helper) + "(" + Acc->Text + ", " + A->Text + ", " +
                    B->Text + ")",
                PrecPrimary};
  }
  if (auto PK = mlir::dyn_cast<ir::Pack4x8Op>(Op)) {
    uint8_t Mode = static_cast<uint8_t>(PK.getMode());
    if (Mode > 2)
      return makeError("unsupported pack4x8 mode");
    requireHelper("crucible_pack4x8");
    bool Is16 = PK.getV0().getType().isInteger(16);
    std::string Cast = Is16 ? "as_type<short>" : "as_type<int>";
    mlir::Value Vals[4] = {PK.getV0(), PK.getV1(), PK.getV2(), PK.getV3()};
    std::string Comps;
    for (unsigned I = 0; I < 4; ++I) {
      auto V = exprFor(Vals[I], PrecUnknown);
      if (!V)
        return V.takeError();
      if (I)
        Comps += ", ";
      Comps += Cast + "(" + V->Text + ")";
    }
    return Expr{"crucible_pack4x8<" + std::to_string(Mode) + "u>(int4(" +
                    Comps + "))",
                PrecPrimary};
  }
  if (auto F32To16 = mlir::dyn_cast<ir::LegacyF32ToF16Op>(Op)) {
    auto A = exprFor(F32To16.getValue(), PrecUnknown);
    if (!A)
      return A.takeError();
    return Expr{"uint(as_type<ushort>(half(" + A->Text + ")))", PrecPrimary};
  }
  if (auto F16To32 = mlir::dyn_cast<ir::LegacyF16ToF32Op>(Op)) {
    auto A = exprFor(F16To32.getValue(), PrecUnknown);
    if (!A)
      return A.takeError();
    return Expr{"float(as_type<half>(ushort(" + A->Text + ")))", PrecPrimary};
  }
  if (auto MD = mlir::dyn_cast<ir::MakeDoubleOp>(Op)) {
    if (!Opts.EmulateF64)
      return makeError("MSL has no double type; use float in the shader");
    auto Lo = exprFor(MD.getLo(), PrecUnknown);
    auto Hi = exprFor(MD.getHi(), PrecUnknown);
    if (!Lo || !Hi)
      return makeError("invalid crucible.make_double operand");
    return Expr{"((ulong(" + Hi->Text + ") << 32ul) | ulong(" + Lo->Text + "))",
                PrecPrimary};
  }
  if (mlir::isa<ir::WaveIsFirstLaneOp>(Op))
    return Expr{"simd_is_first()", PrecPrimary};
  if (auto WV = mlir::dyn_cast<ir::WaveVoteOp>(Op)) {
    auto A = exprFor(WV.getValue(), PrecUnknown);
    if (!A)
      return A.takeError();
    if (WV.getKind() == ir::WaveVoteKind::Any)
      return Expr{"simd_any(" + A->Text + ")", PrecPrimary};
    if (WV.getKind() == ir::WaveVoteKind::All)
      return Expr{"simd_all(" + A->Text + ")", PrecPrimary};
    mlir::Type T = WV.getValue().getType();
    if (T.isInteger(1))
      return Expr{"(simd_all(" + A->Text + ") || !simd_any(" + A->Text + "))",
                  PrecPrimary};
    if (T.isF64()) {
      requireHelper("crucible_f64");
      std::string First = "as_type<ulong>(simd_broadcast_first(as_type<uint2>(" +
                          A->Text + ")))";
      return Expr{"simd_all(crucible_f64_cmp_eq(" + A->Text + ", " + First +
                      "))",
                  PrecPrimary};
    }
    std::string First =
        T.isInteger(64)
            ? "as_type<ulong>(simd_broadcast_first(as_type<uint2>(" + A->Text +
                  ")))"
            : "simd_broadcast_first(" + A->Text + ")";
    return Expr{"simd_all((" + A->Text + ") == " + First + ")", PrecPrimary};
  }
  if (auto BC = mlir::dyn_cast<ir::WaveBitCountOp>(Op)) {
    auto A = exprFor(BC.getCond(), PrecUnknown);
    if (!A)
      return A.takeError();
    if (BC.getKind() == ir::WaveBitCountKind::All)
      return Expr{"simd_sum(uint(" + A->Text + "))", PrecPrimary};
    if (BC.getKind() == ir::WaveBitCountKind::Prefix)
      return Expr{"simd_prefix_exclusive_sum(uint(" + A->Text + "))",
                  PrecPrimary};
    auto M0 = exprFor(BC.getMask()[0], PrecUnknown);
    auto M1 = exprFor(BC.getMask()[1], PrecUnknown);
    if (!M0 || !M1)
      return makeError("invalid crucible.wave_bit_count mask operand");
    requireHelper("crucible_wave_multi_prefix");
    UsesLane = true;
    return Expr{"crucible_wave_multi_prefix_arith<uint, 0>(uint(" + A->Text +
                    "), " + M0->Text + ", " + M1->Text + ", lane)",
                PrecPrimary};
  }
  if (auto MP = mlir::dyn_cast<ir::WaveMultiPrefixOp>(Op)) {
    auto OpKind = MP.getOp();
    uint32_t KindNum = static_cast<uint32_t>(OpKind);
    bool IsBitwise = OpKind == ir::WaveMultiPrefixOpKind::And ||
                     OpKind == ir::WaveMultiPrefixOpKind::Or ||
                     OpKind == ir::WaveMultiPrefixOpKind::Xor;
    mlir::Type T = MP.getResult().getType();
    if (IsBitwise && !mlir::isa<mlir::IntegerType>(T))
      return makeError("bitwise wave multi-prefix op requires integer type");
    auto V = exprFor(MP.getValue(), PrecUnknown);
    auto M0 = exprFor(MP.getMask()[0], PrecUnknown);
    auto M1 = exprFor(MP.getMask()[1], PrecUnknown);
    if (!V || !M0 || !M1)
      return makeError("invalid crucible.wave_multi_prefix operand");
    requireHelper("crucible_wave_multi_prefix");
    UsesLane = true;
    if (T.isF64()) {
      requireHelper("crucible_f64");
      requireHelper("crucible_wave_multi_prefix_f64");
      return Expr{"crucible_wave_multi_prefix_f64<" + std::to_string(KindNum) +
                      ">(" + V->Text + ", " + M0->Text + ", " + M1->Text +
                      ", lane)",
                  PrecPrimary};
    }
    if (T.isF16())
      return Expr{"half(crucible_wave_multi_prefix_arith<float, " +
                      std::to_string(KindNum) + ">(float(" + V->Text + "), " +
                      M0->Text + ", " + M1->Text + ", lane))",
                  PrecPrimary};
    auto Ty = mslType(T);
    if (!Ty)
      return Ty.takeError();
    StringRef Helper = IsBitwise ? "crucible_wave_multi_prefix_bit"
                                 : "crucible_wave_multi_prefix_arith";
    return Expr{Helper.str() + "<" + *Ty + ", " + std::to_string(KindNum) +
                    ">(" + V->Text + ", " + M0->Text + ", " + M1->Text +
                    ", lane)",
                PrecPrimary};
  }
  if (auto QR = mlir::dyn_cast<ir::QuadReadLaneAtOp>(Op)) {
    auto V = exprFor(QR.getValue(), PrecUnknown);
    auto L = exprFor(QR.getQuadLane(), PrecUnknown);
    if (!V || !L)
      return makeError("invalid crucible.quad_read_lane_at operand");
    mlir::Type T = QR.getValue().getType();
    std::string LaneArg = ", ushort(" + L->Text + ")";
    if (T.isInteger(1))
      return Expr{"bool(quad_shuffle(uint(" + V->Text + ")" + LaneArg + "))",
                  PrecPrimary};
    if (T.isInteger(64) || T.isF64())
      return Expr{"as_type<ulong>(quad_shuffle(as_type<uint2>(" + V->Text +
                      ")" + LaneArg + "))",
                  PrecPrimary};
    return Expr{"quad_shuffle(" + V->Text + LaneArg + ")", PrecPrimary};
  }
  if (auto QO = mlir::dyn_cast<ir::QuadOp>(Op)) {
    auto V = exprFor(QO.getValue(), PrecUnknown);
    if (!V)
      return V.takeError();
    mlir::Type T = QO.getValue().getType();
    std::string LaneArg =
        ", " + std::to_string(static_cast<uint32_t>(QO.getKind()) + 1) + "u";
    if (T.isInteger(1))
      return Expr{"bool(quad_shuffle_xor(uint(" + V->Text + ")" + LaneArg +
                      "))",
                  PrecPrimary};
    if (T.isInteger(64) || T.isF64())
      return Expr{"as_type<ulong>(quad_shuffle_xor(as_type<uint2>(" + V->Text +
                      ")" + LaneArg + "))",
                  PrecPrimary};
    return Expr{"quad_shuffle_xor(" + V->Text + LaneArg + ")", PrecPrimary};
  }

  if (auto LI = mlir::dyn_cast<ir::LoadInputOp>(Op)) {
    uint32_t ID = static_cast<uint32_t>(LI.getInputId());
    uint32_t Col = static_cast<uint32_t>(static_cast<uint8_t>(LI.getCol()));
    if (R.Geometry)
      return geometryLoad(ID, LI.getRow(), Col, LI.getVertexId(),
                          LI.getResult().getType());
    if (R.Hull)
      return hullLoad(ID, LI.getRow(), Col, LI.getVertexId(),
                      LI.getResult().getType(), /*IsOutputCP=*/false);
    if (R.Domain)
      return domainLoad(ID, LI.getRow(), Col, LI.getVertexId(),
                        LI.getResult().getType(), /*IsConstant=*/false);
    auto Access = stageElement(ID, LI.getRow(), Col, /*Input=*/true);
    if (!Access)
      return Access.takeError();
    auto T = mslType(LI.getResult().getType());
    if (!T)
      return T.takeError();
    return Expr{*T + "(" + *Access + ")", PrecPrimary};
  }
  if (auto LC = mlir::dyn_cast<ir::LoadControlPointOp>(Op)) {
    uint32_t ID = static_cast<uint32_t>(LC.getInputId());
    uint32_t Col = static_cast<uint32_t>(static_cast<uint8_t>(LC.getCol()));
    return hullLoad(ID, LC.getRow(), Col, LC.getCpId(),
                    LC.getResult().getType(), /*IsOutputCP=*/true);
  }
  if (auto LP = mlir::dyn_cast<ir::LoadPatchConstantOp>(Op)) {
    uint32_t ID = static_cast<uint32_t>(LP.getInputId());
    uint32_t Col = static_cast<uint32_t>(static_cast<uint8_t>(LP.getCol()));
    return domainLoad(ID, LP.getRow(), Col, mlir::Value(),
                      LP.getResult().getType(), /*IsConstant=*/true);
  }
  if (auto Deriv = mlir::dyn_cast<ir::DerivOp>(Op)) {
    if (R.Stage != ShaderStage::Pixel)
      return makeError("derivatives currently require a fragment shader");
    auto V = exprFor(Deriv.getValue(), PrecUnknown);
    if (!V)
      return V.takeError();
    bool X = Deriv.getAxis() == ir::DerivAxis::X;
    std::string Text = std::string(X ? "dfdx(" : "dfdy(") + V->Text + ")";
    if (Deriv.getQuality() == ir::DerivQuality::Coarse)
      Text = "quad_broadcast(" + Text + ", 0u)";
    return Expr{Text, PrecPrimary};
  }

  if (auto RQ = mlir::dyn_cast<ir::RayQueryGetOp>(Op)) {
    auto Q = exprFor(RQ.getQuery(), PrecUnknown);
    if (!Q)
      return Q.takeError();
    auto CompExpr = [&]() -> Expected<std::string> {
      if (!RQ.getComponent())
        return makeError("missing crucible.ray_query_get component");
      auto C = exprFor(RQ.getComponent(), PrecUnknown);
      if (!C)
        return C.takeError();
      return C->Text;
    };
    switch (RQ.getField()) {
    case ir::RayQueryField::CommittedStatus:
      return Expr{"uint(" + Q->Text + ".get_committed_intersection_type())",
                  PrecPrimary};
    case ir::RayQueryField::CandidateType:
      return Expr{"(" + Q->Text +
                      ".get_candidate_intersection_type() == "
                      "intersection_type::bounding_box ? 1u : 0u)",
                  PrecPrimary};
    case ir::RayQueryField::CommittedTriangleFrontFace:
      return Expr{Q->Text + ".is_committed_triangle_front_facing()",
                  PrecPrimary};
    case ir::RayQueryField::CommittedTriangleBarycentrics: {
      auto C = CompExpr();
      if (!C)
        return C.takeError();
      return Expr{Q->Text + ".get_committed_triangle_barycentric_coord()[" +
                      *C + "]",
                  PrecPrimary};
    }
    case ir::RayQueryField::RayFlags:
      return Expr{Q->Text + "_flags", PrecPrimary};
    case ir::RayQueryField::WorldRayOrigin: {
      auto C = CompExpr();
      if (!C)
        return C.takeError();
      return Expr{Q->Text + ".get_world_space_ray_origin()[" + *C + "]",
                  PrecPrimary};
    }
    case ir::RayQueryField::WorldRayDirection: {
      auto C = CompExpr();
      if (!C)
        return C.takeError();
      return Expr{Q->Text + ".get_world_space_ray_direction()[" + *C + "]",
                  PrecPrimary};
    }
    case ir::RayQueryField::RayTMin:
      return Expr{Q->Text + ".get_ray_min_distance()", PrecPrimary};
    case ir::RayQueryField::CommittedRayT:
      return Expr{Q->Text + ".get_committed_distance()", PrecPrimary};
    case ir::RayQueryField::CommittedInstanceIndex:
      return Expr{Q->Text + ".get_committed_instance_id()", PrecPrimary};
    case ir::RayQueryField::CommittedInstanceId:
      return Expr{Q->Text + ".get_committed_user_instance_id()", PrecPrimary};
    case ir::RayQueryField::CommittedGeometryIndex:
      return Expr{Q->Text + ".get_committed_geometry_id()", PrecPrimary};
    case ir::RayQueryField::CommittedPrimitiveIndex:
      return Expr{Q->Text + ".get_committed_primitive_id()", PrecPrimary};
    case ir::RayQueryField::CommittedInstanceContrib:
      return Expr{"crucible_inst_contrib[" + Q->Text +
                      ".get_committed_instance_id()]",
                  PrecPrimary};
    }
  }

  return makeError("unsupported operation '{0}'",
                   Op->getName().getStringRef());
}

Expected<std::string> Emitter::bufferIndexExpr(ResState &RS, mlir::Value Coord0,
                                               mlir::Value Coord1,
                                               unsigned Comp) {
  std::string Index;
  uint64_t ConstOffset = 0; // in scalar elements
  const unsigned EB = elemByteSize(RS.ElemType);
  const unsigned Shift = EB == 1 ? 0 : EB == 2 ? 1 : EB == 8 ? 3 : 2;

  switch (RS.RB->Kind) {
  case ResourceKind::StructuredBuffer: {
    uint32_t Stride = RS.RB->ElementStride;
    if (Stride == 0 || Stride % EB != 0)
      return makeError("structured buffer '{0}' has unsupported stride {1}",
                       RS.ParamName, Stride);
    uint32_t K = Stride / EB;
    auto E = exprFor(Coord0, K > 1 ? PrecMul : PrecAdd);
    if (!E)
      return E.takeError();
    Index = E->Text;
    if (K > 1)
      Index += " * " + std::to_string(K) + "u";
    if (!isUndefLike(Coord1)) {
      if (auto COff = constInt(Coord1))
        ConstOffset += *COff / EB;
      else {
        auto Off = exprFor(Coord1, PrecShift);
        if (!Off)
          return Off.takeError();
        Index += " + (" + Off->Text + " >> " + std::to_string(Shift) + "u)";
      }
    }
    break;
  }
  case ResourceKind::RawBuffer: {
    // ByteAddressBuffer: coord0 is a byte offset.
    if (auto CByte = constInt(Coord0)) {
      Index = std::to_string(*CByte / EB) + "u";
    } else {
      auto E = exprFor(Coord0, PrecShift);
      if (!E)
        return E.takeError();
      Index = "(" + E->Text + " >> " + std::to_string(Shift) + "u)";
    }
    break;
  }
  case ResourceKind::TypedBuffer: {
    unsigned Width = RS.RB->TypedComponents;
    auto E = exprFor(Coord0, Width > 1 ? PrecMul : PrecAdd);
    if (!E)
      return E.takeError();
    Index = E->Text;
    if (Width > 1)
      Index += " * " + std::to_string(Width) + "u";
    break;
  }
  default:
    return makeError("unsupported buffer kind for '{0}'", RS.ParamName);
  }

  ConstOffset += Comp;
  if (ConstOffset)
    Index += " + " + std::to_string(ConstOffset) + "u";
  return Index;
}

Expected<std::string> Emitter::bufferDimExpr(ResState &RS,
                                             mlir::Value Handle) {
  auto It = HandleMap.find(Handle.getDefiningOp());
  if (It == HandleMap.end())
    return makeError("unresolved resource handle");
  std::string Slot = std::to_string(RS.RB->MetalIndex) + "u";
  if (RS.RB->RangeSize > 1) {
    if (auto C = constInt(It->second.RegIdx)) {
      uint32_t Offset = static_cast<uint32_t>(*C) - RS.RB->LowerBound;
      Slot = std::to_string(RS.RB->MetalIndex + Offset) + "u";
    } else {
      auto E = exprFor(It->second.RegIdx, PrecAdd);
      if (!E)
        return E.takeError();
      Slot = std::to_string(RS.RB->MetalIndex) + "u + (" + E->Text + " - " +
             std::to_string(RS.RB->LowerBound) + "u)";
    }
  }
  return "crucible_buf_dims[" + Slot + "]";
}

Expected<std::string> Emitter::bufferBoundExpr(ResState &RS,
                                               mlir::Value Handle) {
  auto Dim = bufferDimExpr(RS, Handle);
  if (!Dim)
    return Dim.takeError();
  const unsigned EB = elemByteSize(RS.ElemType);
  const unsigned Shift = EB == 1 ? 0 : EB == 2 ? 1 : EB == 8 ? 3 : 2;
  switch (RS.RB->Kind) {
  case ResourceKind::StructuredBuffer: {
    uint32_t K = RS.RB->ElementStride / EB;
    return K > 1 ? "(" + *Dim + " * " + std::to_string(K) + "u)" : *Dim;
  }
  case ResourceKind::RawBuffer:
    return "(" + *Dim + " >> " + std::to_string(Shift) + "u)";
  case ResourceKind::TypedBuffer: {
    unsigned Width = RS.RB->TypedComponents;
    return Width > 1 ? "(" + *Dim + " * " + std::to_string(Width) + "u)" : *Dim;
  }
  default:
    return makeError("unsupported buffer kind for '{0}'", RS.ParamName);
  }
}

Expected<Expr> Emitter::crucibleExtractExpr(mlir::Operation *Producer,
                                            unsigned Comp, mlir::Type ElemTy) {
  if (auto GD = mlir::dyn_cast<ir::TexGetDimensionsOp>(Producer)) {
    ResState *RS = stateFor(GD.getHandle());
    if (!RS)
      return makeError("getDimensions through unresolved handle");
    if (!isTextureKind(RS->RB->Kind)) {
      if (Comp != 0)
        return makeError("buffer getDimensions only supports component 0");
      auto Dim = bufferDimExpr(*RS, GD.getHandle());
      if (!Dim)
        return Dim.takeError();
      return Expr{*Dim, PrecPrimary};
    }
    auto RName = resourceExpr(GD.getHandle());
    if (!RName)
      return RName.takeError();
    std::string MipArg;
    if (GD.getMipLevel()) {
      auto M = exprFor(GD.getMipLevel(), PrecUnknown);
      if (!M)
        return M.takeError();
      MipArg = M->Text;
    }
    ResourceKind K = RS->RB->Kind;
    bool SupportsMipArg =
        K == ResourceKind::Texture1D || K == ResourceKind::Texture1DArray ||
        K == ResourceKind::Texture2D || K == ResourceKind::Texture2DArray ||
        K == ResourceKind::Texture3D || K == ResourceKind::TextureCube ||
        K == ResourceKind::TextureCubeArray;
    std::string MipCall = SupportsMipArg ? MipArg : std::string();
    switch (Comp) {
    case 0:
      return Expr{*RName + ".get_width(" + MipCall + ")", PrecPrimary};
    case 1:
      if (K == ResourceKind::Texture1DArray)
        return Expr{*RName + ".get_array_size()", PrecPrimary};
      return Expr{*RName + ".get_height(" + MipCall + ")", PrecPrimary};
    case 2:
      if (K == ResourceKind::Texture3D)
        return Expr{*RName + ".get_depth(" + MipCall + ")", PrecPrimary};
      return Expr{*RName + ".get_array_size()", PrecPrimary};
    case 3:
      if (K == ResourceKind::Texture2DMS || K == ResourceKind::Texture2DMSArray)
        return Expr{*RName + ".get_num_samples()", PrecPrimary};
      return Expr{*RName + ".get_num_mip_levels()", PrecPrimary};
    default:
      return makeError("getDimensions component out of range");
    }
  }

  if (auto BL = mlir::dyn_cast<ir::BufferLoadOp>(Producer)) {
    if (Comp > 3)
      return makeError("invalid buffer load component");
    ResState *RS = stateFor(BL.getHandle());
    if (!RS)
      return makeError("buffer load through unresolved handle");
    auto RName = resourceExpr(BL.getHandle());
    if (!RName)
      return RName.takeError();
    auto Index = bufferIndexExpr(*RS, BL.getIndex(), BL.getElementOffset(), Comp);
    if (!Index)
      return Index.takeError();
    std::string Base = *RName;
    if (BL->hasAttr("crucible.volatile"))
      Base = "((volatile device " + RS->ElemType + " *)" + Base + ")";
    std::string Element = Base + "[" + *Index + "]";
    if (Opts.BoundsCheckBuffers && R.BufferSizesMetalIndex != UINT32_MAX) {
      auto Bound = bufferBoundExpr(*RS, BL.getHandle());
      if (!Bound)
        return Bound.takeError();
      Element = "(uint(" + *Index + ") < " + *Bound + " ? " + Element +
                " : " + RS->ElemType + "(0))";
    }
    auto LoadTy = mslType(ElemTy);
    if (!LoadTy)
      return LoadTy.takeError();
    if (*LoadTy != RS->ElemType)
      return Expr{"as_type<" + *LoadTy + ">(" + Element + ")", PrecPrimary};
    return Expr{Element, PrecPrimary};
  }

  if (auto CB = mlir::dyn_cast<ir::CBufferLoadOp>(Producer)) {
    ResState *RS = stateFor(CB.getHandle());
    if (!RS)
      return makeError("cbuffer load through unresolved handle");
    auto RName = resourceExpr(CB.getHandle());
    if (!RName)
      return RName.takeError();
    auto Row = exprFor(CB.getRegIndex(), PrecUnknown);
    if (!Row)
      return Row.takeError();
    std::string RowText = *RName + "[" + Row->Text + "]";
    auto VecTy = mlir::cast<mlir::VectorType>(CB.getResult().getType());
    mlir::Type T = VecTy.getElementType();
    bool RowIsFloat = RS->ElemType == "float4";
    if (Comp >= VecTy.getNumElements())
      return makeError("cbuffer component out of range");
    if (T.isInteger(64) || (Opts.EmulateF64 && T.isF64()))
      return Expr{"as_type<ulong2>(" + RowText + ")[" +
                      std::to_string(Comp) + "]",
                  PrecPrimary};
    if (T.isF16() || T.isInteger(16)) {
      bool Packed = VecTy.getNumElements() == 8;
      std::string Word = RowText + "." + "xyzw"[Packed ? Comp / 2 : Comp];
      if (RowIsFloat)
        Word = "as_type<uint>(" + Word + ")";
      if (Packed && Comp % 2)
        Word += " >> 16u";
      std::string Bits = "ushort(" + Word + ")";
      return Expr{T.isF16() ? "as_type<half>(" + Bits + ")" : Bits,
                  PrecPrimary};
    }
    std::string Element = RowText + "." + "xyzw"[Comp];
    bool LoadIsFloat = T.isF32();
    if (LoadIsFloat != RowIsFloat)
      Element = std::string("as_type<") + (LoadIsFloat ? "float" : "uint") +
                ">(" + Element + ")";
    return Expr{Element, PrecPrimary};
  }

  return makeError("unsupported crucible vector extract producer");
}

Expected<Expr> Emitter::bufferElementExpr(mlir::LLVM::ExtractValueOp EV) {
  mlir::Operation *Producer = EV.getContainer().getDefiningOp();
  auto Pos = EV.getPosition();
  unsigned Comp = Pos.empty() ? 0 : static_cast<unsigned>(Pos[0]);

  // Compare-exchange aggregates: {original, success}.
  if (auto CX = mlir::dyn_cast_or_null<mlir::LLVM::AtomicCmpXchgOp>(Producer)) {
    auto It = ValueNames.find(CX.getResult());
    if (It == ValueNames.end())
      return makeError("use of cmpxchg result before materialization");
    if (Comp == 0)
      return Expr{It->second, PrecPrimary};
    auto Cmp = exprFor(CX.getCmp(), PrecEq);
    if (!Cmp)
      return Cmp.takeError();
    return Expr{It->second + " == " + Cmp->Text, PrecEq};
  }

  if (mlir::isa<mlir::LLVM::LLVMStructType, mlir::LLVM::LLVMArrayType>(
          EV.getContainer().getType())) {
    auto Value = exprFor(EV.getContainer(), PrecPrimary);
    auto Path = aggregateMemberPath(EV.getContainer().getType(), Pos);
    if (!Value)
      return Value.takeError();
    if (!Path)
      return Path.takeError();
    return Expr{Value->Text + *Path, PrecPrimary};
  }

  if (auto Ld = mlir::dyn_cast_or_null<mlir::LLVM::LoadOp>(Producer)) {
    auto LV = lvalueFor(Ld.getAddr(), /*DecayArray=*/false);
    if (!LV)
      return LV.takeError();
    std::string Text = LV->Text;
    for (int64_t Idx : Pos)
      Text += "[" + std::to_string(Idx) + "]";
    return Expr{std::move(Text), PrecPrimary};
  }

  return makeError("unsupported aggregate producer for extractvalue");
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

Expected<std::string> Emitter::ensureNamed(mlir::Value V, unsigned Ind,
                                           bool CopyBlockArgs) {
  auto It = ValueNames.find(V);
  if (It != ValueNames.end()) {
    bool IsBlockArg = !V.getDefiningOp();
    if (IsBlockArg && CopyBlockArgs) {
      auto Ty = mslType(V.getType());
      if (!Ty)
        return Ty.takeError();
      std::string Name = freshName();
      line(Ind, *Ty + " " + Name + " = " + It->second + ";");
      return Name;
    }
    return It->second;
  }
  if (mlir::Operation *Def = V.getDefiningOp()) {
    if (isConstantLike(Def)) {
      auto CE = constant(Def);
      if (!CE)
        return CE.takeError();
      return CE->Text;
    }
    if (Error E = emitTemp(Def, Ind))
      return std::move(E);
    return ValueNames[V];
  }
  return makeError("cannot name an untracked block argument");
}

Error Emitter::emitTemp(mlir::Operation *Op, unsigned Ind) {
  if (Op->getNumResults() != 1)
    return makeError("cannot materialize multi-result op '{0}'",
                     Op->getName().getStringRef());
  if (auto Load = mlir::dyn_cast<mlir::LLVM::LoadOp>(Op);
      Load && mlir::isa<mlir::LLVM::LLVMArrayType>(Load.getType())) {
    if (Load.getVolatile_())
      return makeError("volatile LLVM array loads are unsupported");
    auto Source = lvalueFor(Load.getAddr(), /*DecayArray=*/false);
    auto Name = freshName();
    auto Decl = localDeclaration(Load.getType(), Name);
    if (!Source)
      return Source.takeError();
    if (!Decl)
      return Decl.takeError();
    line(Ind, *Decl + ";");
    emitAggregateCopy(Load.getType(), Name, Source->Text, Ind);
    ValueNames[Load.getResult()] = Name;
    return Error::success();
  }
  auto Ty = mslType(Op->getResult(0).getType());
  if (!Ty)
    return Ty.takeError();
  auto E = opExpr(Op);
  if (!E)
    return E.takeError();
  std::string Name = freshName();
  line(Ind, *Ty + " " + Name + " = " + E->Text + ";");
  ValueNames[Op->getResult(0)] = Name;
  return Error::success();
}

bool Emitter::canVectorizeBufferAccess(ResState &RS, mlir::Value Coord0,
                                       mlir::Value Coord1, unsigned Width) {
  if (Width != 2 && Width != 4)
    return false;
  if (RS.ElemType != "float" && RS.ElemType != "uint")
    return false;
  if (Opts.BoundsCheckBuffers && R.BufferSizesMetalIndex != UINT32_MAX)
    return false;
  uint64_t AlignBytes = static_cast<uint64_t>(Width) * 4u;
  switch (RS.RB->Kind) {
  case ResourceKind::StructuredBuffer: {
    uint32_t Stride = RS.RB->ElementStride;
    if (Stride == 0 || (Stride % AlignBytes) != 0)
      return false;
    if (isUndefLike(Coord1))
      return true;
    auto COff = constInt(Coord1);
    return COff && (*COff % AlignBytes) == 0;
  }
  case ResourceKind::TypedBuffer:
    return RS.RB->TypedComponents == Width;
  case ResourceKind::RawBuffer: {
    auto CByte = constInt(Coord0);
    return CByte && (*CByte % AlignBytes) == 0;
  }
  default:
    return false;
  }
}

Error Emitter::emitBufferStore(ir::BufferStoreOp Store, unsigned Ind) {
  ResState *RS = stateFor(Store.getHandle());
  if (!RS)
    return makeError("buffer store through unresolved handle");
  auto RName = resourceExpr(Store.getHandle());
  if (!RName)
    return RName.takeError();
  if (Store->hasAttr("crucible.volatile"))
    *RName = "((volatile device " + RS->ElemType + " *)" + *RName + ")";
  uint32_t Mask = static_cast<uint32_t>(Store.getMask());
  auto Vals = Store.getValues();

  // Emit a single 128-bit (float4/uint4) or 64-bit (float2/uint2) vector store
  // when all components are contiguous from 0 and base alignment is guaranteed.
  uint32_t EffectiveMask = Mask;
  if (RS->RB->Kind == ResourceKind::TypedBuffer &&
      RS->RB->TypedComponents > 0 && RS->RB->TypedComponents < 4)
    EffectiveMask &= (1u << RS->RB->TypedComponents) - 1u;
  unsigned VecWidth = (EffectiveMask == 0xFu && Vals.size() >= 4)   ? 4
                      : (EffectiveMask == 0x3u && Vals.size() >= 2) ? 2
                                                                    : 0;
  if (VecWidth > 0 && !Store->hasAttr("crucible.volatile") &&
      canVectorizeBufferAccess(*RS, Store.getIndex(), Store.getElementOffset(),
                               VecWidth)) {
    auto BaseIndex =
        bufferIndexExpr(*RS, Store.getIndex(), Store.getElementOffset(), 0);
    if (!BaseIndex)
      return BaseIndex.takeError();
    std::string VecTy = RS->ElemType + std::to_string(VecWidth);
    std::string Args;
    for (unsigned Comp = 0; Comp < VecWidth; ++Comp) {
      auto V = exprFor(Vals[Comp], PrecUnknown);
      if (!V)
        return V.takeError();
      std::string Value = V->Text;
      auto StoreTy = mslType(Vals[Comp].getType());
      if (!StoreTy)
        return StoreTy.takeError();
      if (*StoreTy != RS->ElemType)
        Value = "as_type<" + RS->ElemType + ">(" + Value + ")";
      if (Comp > 0)
        Args += ", ";
      Args += Value;
    }
    line(Ind, "*(device " + VecTy + " *)(&" + *RName + "[" + *BaseIndex +
                  "]) = " + VecTy + "(" + Args + ");");
    return Error::success();
  }
  for (unsigned Comp = 0; Comp < 4 && Comp < Vals.size(); ++Comp) {
    if (RS->RB->Kind == ResourceKind::TypedBuffer &&
        RS->RB->TypedComponents != 0 && Comp >= RS->RB->TypedComponents)
      break;
    if (!(Mask & (1u << Comp)))
      continue;
    auto Index =
        bufferIndexExpr(*RS, Store.getIndex(), Store.getElementOffset(), Comp);
    if (!Index)
      return Index.takeError();
    auto V = exprFor(Vals[Comp], PrecUnknown);
    if (!V)
      return V.takeError();
    std::string Value = V->Text;
    auto StoreTy = mslType(Vals[Comp].getType());
    if (!StoreTy)
      return StoreTy.takeError();
    if (*StoreTy != RS->ElemType)
      Value = "as_type<" + RS->ElemType + ">(" + Value + ")";
    if (Opts.BoundsCheckBuffers && R.BufferSizesMetalIndex != UINT32_MAX) {
      auto Bound = bufferBoundExpr(*RS, Store.getHandle());
      if (!Bound)
        return Bound.takeError();
      line(Ind, "if (uint(" + *Index + ") < " + *Bound + ") " + *RName + "[" +
                    *Index + "] = " + Value + ";");
    } else {
      line(Ind, *RName + "[" + *Index + "] = " + Value + ";");
    }
  }
  return Error::success();
}

Error Emitter::emitTextureLoad(ir::TexLoadOp Load, unsigned Ind) {
  ResState *RS = stateFor(Load.getHandle());
  if (!RS)
    return makeError("texture load through unresolved handle");

  auto Coords = Load.getCoords();
  auto Offsets = Load.getOffsets();
  auto CoordComp = [&](unsigned I) -> Expected<std::string> {
    if (I >= Coords.size())
      return makeError("missing texture load coordinate");
    auto C = exprFor(Coords[I], PrecAdd);
    if (!C)
      return C.takeError();
    if (I >= Offsets.size() || isUndefLike(Offsets[I]))
      return C->Text;
    if (auto COff = constInt(Offsets[I]); COff && *COff == 0)
      return C->Text;
    auto O = exprFor(Offsets[I], PrecAdd);
    if (!O)
      return O.takeError();
    return C->Text + " + " + O->Text;
  };

  auto C0 = CoordComp(0);
  if (!C0)
    return C0.takeError();

  auto RName = resourceExpr(Load.getHandle());
  if (!RName)
    return RName.takeError();
  std::string CallExpr = *RName + ".read(";
  mlir::Value MipVal = Load.getMipOrSample();
  bool HasMip = MipVal && !isUndefLike(MipVal);
  bool IsUAV = RS->RB->Class == ResourceClass::UAV;

  switch (RS->RB->Kind) {
  case ResourceKind::Texture1D:
    CallExpr += "uint2(" + *C0 + ", 0u)";
    if (!IsUAV && HasMip) {
      auto M = exprFor(MipVal, PrecUnknown);
      if (!M)
        return M.takeError();
      CallExpr += ", uint(" + M->Text + ")";
    }
    break;
  case ResourceKind::Texture1DArray: {
    auto C1 = exprFor(Coords[1], PrecUnknown);
    if (!C1)
      return C1.takeError();
    CallExpr += "uint2(" + *C0 + ", 0u), uint(" + C1->Text + ")";
    if (!IsUAV && HasMip) {
      auto M = exprFor(MipVal, PrecUnknown);
      if (!M)
        return M.takeError();
      CallExpr += ", uint(" + M->Text + ")";
    }
    break;
  }
  case ResourceKind::Texture2D: {
    auto C1 = CoordComp(1);
    if (!C1)
      return C1.takeError();
    CallExpr += "uint2(" + *C0 + ", " + *C1 + ")";
    if (!IsUAV && HasMip) {
      auto M = exprFor(MipVal, PrecUnknown);
      if (!M)
        return M.takeError();
      CallExpr += ", uint(" + M->Text + ")";
    }
    break;
  }
  case ResourceKind::Texture2DArray: {
    auto C1 = CoordComp(1);
    if (!C1)
      return C1.takeError();
    auto C2 = exprFor(Coords[2], PrecUnknown);
    if (!C2)
      return C2.takeError();
    CallExpr += "uint2(" + *C0 + ", " + *C1 + "), uint(" + C2->Text + ")";
    if (!IsUAV && HasMip) {
      auto M = exprFor(MipVal, PrecUnknown);
      if (!M)
        return M.takeError();
      CallExpr += ", uint(" + M->Text + ")";
    }
    break;
  }
  case ResourceKind::Texture3D: {
    auto C1 = CoordComp(1);
    if (!C1)
      return C1.takeError();
    auto C2 = CoordComp(2);
    if (!C2)
      return C2.takeError();
    CallExpr += "uint3(" + *C0 + ", " + *C1 + ", " + *C2 + ")";
    if (!IsUAV && HasMip) {
      auto M = exprFor(MipVal, PrecUnknown);
      if (!M)
        return M.takeError();
      CallExpr += ", uint(" + M->Text + ")";
    }
    break;
  }
  case ResourceKind::Texture2DMS: {
    auto C1 = CoordComp(1);
    if (!C1)
      return C1.takeError();
    auto S = exprFor(MipVal, PrecUnknown);
    if (!S)
      return S.takeError();
    CallExpr += "uint2(" + *C0 + ", " + *C1 + "), uint(" + S->Text + ")";
    break;
  }
  case ResourceKind::Texture2DMSArray: {
    auto C1 = CoordComp(1);
    if (!C1)
      return C1.takeError();
    auto C2 = exprFor(Coords[2], PrecUnknown);
    if (!C2)
      return C2.takeError();
    auto S = exprFor(MipVal, PrecUnknown);
    if (!S)
      return S.takeError();
    CallExpr += "uint2(" + *C0 + ", " + *C1 + "), uint(" + C2->Text +
                "), uint(" + S->Text + ")";
    break;
  }
  default:
    return makeError("unsupported texture kind for load on '{0}'",
                     RS->ParamName);
  }
  CallExpr += ")";

  auto VecTy = mlir::cast<mlir::VectorType>(Load.getResult().getType());
  auto RetElemTy = mslType(VecTy.getElementType());
  if (!RetElemTy)
    return RetElemTy.takeError();
  std::string MslVec = *RetElemTy + "4";
  if (*RetElemTy != RS->ElemType)
    CallExpr = "as_type<" + MslVec + ">(" + CallExpr + ")";

  std::string Name = freshName();
  line(Ind, MslVec + " " + Name + " = " + CallExpr + ";");
  ValueNames[Load.getResult()] = Name;
  if (Load.getStatus())
    ValueNames[Load.getStatus()] = "1u";
  return Error::success();
}

Error Emitter::emitTextureStore(ir::TexStoreOp Store, unsigned Ind) {
  ResState *RS = stateFor(Store.getHandle());
  if (!RS)
    return makeError("texture store through unresolved handle");

  auto StoreVals = Store.getValues();
  std::string Vals[4];
  for (unsigned I = 0; I < 4; ++I) {
    auto V = exprFor(StoreVals[I], PrecUnknown);
    if (!V)
      return V.takeError();
    Vals[I] = V->Text;
  }
  auto ValElemTy = mslType(StoreVals[0].getType());
  if (!ValElemTy)
    return ValElemTy.takeError();
  std::string VecExpr = *ValElemTy + "4(" + Vals[0] + ", " + Vals[1] + ", " +
                        Vals[2] + ", " + Vals[3] + ")";
  if (*ValElemTy != RS->ElemType)
    VecExpr = "as_type<" + RS->ElemType + "4>(" + VecExpr + ")";

  auto Coords = Store.getCoords();
  auto C0 = exprFor(Coords[0], PrecUnknown);
  if (!C0)
    return C0.takeError();

  std::string CoordArgs;
  switch (RS->RB->Kind) {
  case ResourceKind::Texture1D:
    CoordArgs = "uint2(" + C0->Text + ", 0u)";
    break;
  case ResourceKind::Texture1DArray: {
    auto C1 = exprFor(Coords[1], PrecUnknown);
    if (!C1)
      return C1.takeError();
    CoordArgs = "uint2(" + C0->Text + ", 0u), uint(" + C1->Text + ")";
    break;
  }
  case ResourceKind::Texture2D: {
    auto C1 = exprFor(Coords[1], PrecUnknown);
    if (!C1)
      return C1.takeError();
    CoordArgs = "uint2(" + C0->Text + ", " + C1->Text + ")";
    break;
  }
  case ResourceKind::Texture2DArray: {
    auto C1 = exprFor(Coords[1], PrecUnknown);
    if (!C1)
      return C1.takeError();
    auto C2 = exprFor(Coords[2], PrecUnknown);
    if (!C2)
      return C2.takeError();
    CoordArgs =
        "uint2(" + C0->Text + ", " + C1->Text + "), uint(" + C2->Text + ")";
    break;
  }
  case ResourceKind::Texture3D: {
    auto C1 = exprFor(Coords[1], PrecUnknown);
    if (!C1)
      return C1.takeError();
    auto C2 = exprFor(Coords[2], PrecUnknown);
    if (!C2)
      return C2.takeError();
    CoordArgs =
        "uint3(" + C0->Text + ", " + C1->Text + ", " + C2->Text + ")";
    break;
  }
  default:
    return makeError("unsupported texture kind for store on '{0}'",
                     RS->ParamName);
  }

  auto RName = resourceExpr(Store.getHandle());
  if (!RName)
    return RName.takeError();
  line(Ind, *RName + ".write(" + VecExpr + ", " + CoordArgs + ");");
  line(Ind, *RName + ".fence();");
  return Error::success();
}

Error Emitter::emitTextureSample(ir::TexSampleOp Sample, unsigned Ind) {
  ResState *RS = stateFor(Sample.getTex());
  if (!RS)
    return makeError("texture sample through unresolved texture handle");
  ResState *Samp = stateFor(Sample.getSampler());
  if (!Samp)
    return makeError("texture sample through unresolved sampler handle");

  ResourceKind K = RS->RB->Kind;
  auto Coords = Sample.getCoords();
  auto C0 = exprFor(Coords[0], PrecUnknown);
  if (!C0)
    return C0.takeError();

  auto sliceIndex = [](const std::string &S) {
    return "uint(max(rint(" + S + "), 0.0f))";
  };
  std::string Coord, Slice;
  switch (K) {
  case ResourceKind::Texture1D:
    Coord = "float2(" + C0->Text + ", 0.5f)";
    break;
  case ResourceKind::Texture1DArray: {
    auto C1 = exprFor(Coords[1], PrecUnknown);
    if (!C1)
      return C1.takeError();
    Coord = "float2(" + C0->Text + ", 0.5f)";
    Slice = sliceIndex(C1->Text);
    break;
  }
  case ResourceKind::Texture2D: {
    auto C1 = exprFor(Coords[1], PrecUnknown);
    if (!C1)
      return C1.takeError();
    Coord = "float2(" + C0->Text + ", " + C1->Text + ")";
    break;
  }
  case ResourceKind::Texture2DArray: {
    auto C1 = exprFor(Coords[1], PrecUnknown);
    if (!C1)
      return C1.takeError();
    auto C2 = exprFor(Coords[2], PrecUnknown);
    if (!C2)
      return C2.takeError();
    Coord = "float2(" + C0->Text + ", " + C1->Text + ")";
    Slice = sliceIndex(C2->Text);
    break;
  }
  case ResourceKind::Texture3D:
  case ResourceKind::TextureCube: {
    auto C1 = exprFor(Coords[1], PrecUnknown);
    if (!C1)
      return C1.takeError();
    auto C2 = exprFor(Coords[2], PrecUnknown);
    if (!C2)
      return C2.takeError();
    Coord = "float3(" + C0->Text + ", " + C1->Text + ", " + C2->Text + ")";
    break;
  }
  case ResourceKind::TextureCubeArray: {
    auto C1 = exprFor(Coords[1], PrecUnknown);
    if (!C1)
      return C1.takeError();
    auto C2 = exprFor(Coords[2], PrecUnknown);
    if (!C2)
      return C2.takeError();
    auto C3 = exprFor(Coords[3], PrecUnknown);
    if (!C3)
      return C3.takeError();
    Coord = "float3(" + C0->Text + ", " + C1->Text + ", " + C2->Text + ")";
    Slice = sliceIndex(C3->Text);
    break;
  }
  default:
    return makeError("unsupported texture kind for sampling '{0}'",
                     RS->ParamName);
  }

  auto Offs = Sample.getOffsets();
  auto IsZeroOrMissing = [&](unsigned I) -> bool {
    if (I >= Offs.size() || isUndefLike(Offs[I]))
      return true;
    auto C = constInt(Offs[I]);
    return C && *C == 0;
  };
  auto IntStr = [&](unsigned I) -> Expected<std::string> {
    if (I >= Offs.size() || isUndefLike(Offs[I]))
      return std::string("0");
    if (auto C = constInt(Offs[I]))
      return std::to_string(static_cast<int32_t>(*C));
    auto E = exprFor(Offs[I], PrecUnknown);
    if (!E)
      return E.takeError();
    return "int(" + E->Text + ")";
  };

  std::string Offset;
  bool SupportsOffset =
      K != ResourceKind::TextureCube && K != ResourceKind::TextureCubeArray;
  if (SupportsOffset) {
    bool HasOff0 = !IsZeroOrMissing(0);
    bool HasOff1 = (K == ResourceKind::Texture2D ||
                    K == ResourceKind::Texture2DArray ||
                    K == ResourceKind::Texture3D) &&
                   !IsZeroOrMissing(1);
    bool HasOff2 = K == ResourceKind::Texture3D && !IsZeroOrMissing(2);
    if (HasOff0 || HasOff1 || HasOff2) {
      auto O0 = IntStr(0);
      if (!O0)
        return O0.takeError();
      if (K == ResourceKind::Texture1D || K == ResourceKind::Texture1DArray) {
        Offset = "int2(" + *O0 + ", 0)";
      } else {
        auto O1 = IntStr(1);
        if (!O1)
          return O1.takeError();
        if (K == ResourceKind::Texture3D) {
          auto O2 = IntStr(2);
          if (!O2)
            return O2.takeError();
          Offset = "int3(" + *O0 + ", " + *O1 + ", " + *O2 + ")";
        } else {
          Offset = "int2(" + *O0 + ", " + *O1 + ")";
        }
      }
    }
  }

  auto RName = resourceExpr(Sample.getTex());
  if (!RName)
    return RName.takeError();
  auto SName = resourceExpr(Sample.getSampler());
  if (!SName)
    return SName.takeError();

  std::string CallExpr = *RName + ".";
  auto Mode = Sample.getMode();
  if (Mode == ir::SampleMode::Implicit || Mode == ir::SampleMode::Bias) {
    CallExpr += "sample(" + *SName + ", " + Coord;
    if (!Slice.empty())
      CallExpr += ", " + Slice;
    if (Mode == ir::SampleMode::Bias) {
      auto Bias = exprFor(Sample.getLodOrBias(), PrecUnknown);
      if (!Bias)
        return Bias.takeError();
      CallExpr += ", bias(" + Bias->Text + ")";
    }
    if (Sample.getClamp()) {
      auto Clamp = exprFor(Sample.getClamp(), PrecUnknown);
      if (!Clamp)
        return Clamp.takeError();
      CallExpr += ", min_lod_clamp(" + Clamp->Text + ")";
    }
    if (!Offset.empty())
      CallExpr += ", " + Offset;
    CallExpr += ")";
  } else if (Mode == ir::SampleMode::Level) {
    auto Lod = exprFor(Sample.getLodOrBias(), PrecUnknown);
    if (!Lod)
      return Lod.takeError();
    CallExpr += "sample(" + *SName + ", " + Coord;
    if (!Slice.empty())
      CallExpr += ", " + Slice;
    CallExpr += ", level(" + Lod->Text + ")";
    if (!Offset.empty())
      CallExpr += ", " + Offset;
    CallExpr += ")";
  } else if (Mode == ir::SampleMode::Grad) {
    auto Ddx = Sample.getDdx();
    auto Ddy = Sample.getDdy();
    auto DX0 = exprFor(Ddx[0], PrecUnknown);
    auto DY0 = exprFor(Ddy[0], PrecUnknown);
    if (!DX0 || !DY0)
      return makeError("invalid sampleGrad derivative operand");
    std::string GradKind, DDX, DDY;
    if (K == ResourceKind::Texture1D || K == ResourceKind::Texture1DArray) {
      GradKind = "gradient2d";
      DDX = "float2(" + DX0->Text + ", 0.0f)";
      DDY = "float2(" + DY0->Text + ", 0.0f)";
    } else if (K == ResourceKind::Texture2D ||
               K == ResourceKind::Texture2DArray) {
      auto DX1 = exprFor(Ddx[1], PrecUnknown);
      auto DY1 = exprFor(Ddy[1], PrecUnknown);
      if (!DX1 || !DY1)
        return makeError("invalid sampleGrad derivative operand");
      GradKind = "gradient2d";
      DDX = "float2(" + DX0->Text + ", " + DX1->Text + ")";
      DDY = "float2(" + DY0->Text + ", " + DY1->Text + ")";
    } else {
      auto DX1 = exprFor(Ddx[1], PrecUnknown);
      auto DX2 = exprFor(Ddx[2], PrecUnknown);
      auto DY1 = exprFor(Ddy[1], PrecUnknown);
      auto DY2 = exprFor(Ddy[2], PrecUnknown);
      if (!DX1 || !DX2 || !DY1 || !DY2)
        return makeError("invalid sampleGrad derivative operand");
      GradKind =
          K == ResourceKind::Texture3D ? "gradient3d" : "gradientcube";
      DDX = "float3(" + DX0->Text + ", " + DX1->Text + ", " + DX2->Text + ")";
      DDY = "float3(" + DY0->Text + ", " + DY1->Text + ", " + DY2->Text + ")";
    }
    CallExpr += "sample(" + *SName + ", " + Coord;
    if (!Slice.empty())
      CallExpr += ", " + Slice;
    CallExpr += ", " + GradKind + "(" + DDX + ", " + DDY + ")";
    if (Sample.getClamp()) {
      auto Clamp = exprFor(Sample.getClamp(), PrecUnknown);
      if (!Clamp)
        return Clamp.takeError();
      CallExpr += ", min_lod_clamp(" + Clamp->Text + ")";
    }
    if (!Offset.empty())
      CallExpr += ", " + Offset;
    CallExpr += ")";
  } else {
    auto Cmp = exprFor(Sample.getCmpVal(), PrecUnknown);
    if (!Cmp)
      return Cmp.takeError();
    CallExpr = "float4(" + *RName + ".sample_compare(" + *SName + ", " + Coord;
    if (!Slice.empty())
      CallExpr += ", " + Slice;
    CallExpr += ", " + Cmp->Text;
    if (Mode == ir::SampleMode::CmpLevelZero) {
      CallExpr += ", level(0)";
    } else if (Sample.getClamp()) {
      auto Clamp = exprFor(Sample.getClamp(), PrecUnknown);
      if (!Clamp)
        return Clamp.takeError();
      CallExpr += ", min_lod_clamp(" + Clamp->Text + ")";
    }
    if (!Offset.empty())
      CallExpr += ", " + Offset;
    CallExpr += "), 0.0f, 0.0f, 0.0f)";
  }

  auto VecTy = mlir::cast<mlir::VectorType>(Sample.getResult().getType());
  auto RetElemTy = mslType(VecTy.getElementType());
  if (!RetElemTy)
    return RetElemTy.takeError();
  std::string MslVec = *RetElemTy + "4";
  if (*RetElemTy != RS->ElemType && !RS->IsDepthSampled)
    CallExpr = "as_type<" + MslVec + ">(" + CallExpr + ")";

  std::string Name = freshName();
  line(Ind, MslVec + " " + Name + " = " + CallExpr + ";");
  ValueNames[Sample.getResult()] = Name;
  if (Sample.getStatus())
    ValueNames[Sample.getStatus()] = "1u";
  return Error::success();
}

Error Emitter::emitTextureGather(ir::TexGatherOp Gather, unsigned Ind) {
  ResState *RS = stateFor(Gather.getTex());
  if (!RS)
    return makeError("texture gather through unresolved texture handle");
  ResState *Samp = stateFor(Gather.getSampler());
  if (!Samp)
    return makeError("texture gather through unresolved sampler handle");

  ResourceKind K = RS->RB->Kind;
  auto Coords = Gather.getCoords();
  auto C0 = exprFor(Coords[0], PrecUnknown);
  if (!C0)
    return C0.takeError();
  auto sliceIndex = [](const std::string &S) {
    return "uint(max(rint(" + S + "), 0.0f))";
  };
  std::string Coord, Slice;
  switch (K) {
  case ResourceKind::Texture2D: {
    auto C1 = exprFor(Coords[1], PrecUnknown);
    if (!C1)
      return C1.takeError();
    Coord = "float2(" + C0->Text + ", " + C1->Text + ")";
    break;
  }
  case ResourceKind::Texture2DArray: {
    auto C1 = exprFor(Coords[1], PrecUnknown);
    auto C2 = exprFor(Coords[2], PrecUnknown);
    if (!C1 || !C2)
      return makeError("invalid gather2DArray coordinate");
    Coord = "float2(" + C0->Text + ", " + C1->Text + ")";
    Slice = sliceIndex(C2->Text);
    break;
  }
  case ResourceKind::TextureCube: {
    auto C1 = exprFor(Coords[1], PrecUnknown);
    auto C2 = exprFor(Coords[2], PrecUnknown);
    if (!C1 || !C2)
      return makeError("invalid gatherCube coordinate");
    Coord = "float3(" + C0->Text + ", " + C1->Text + ", " + C2->Text + ")";
    break;
  }
  case ResourceKind::TextureCubeArray: {
    auto C1 = exprFor(Coords[1], PrecUnknown);
    auto C2 = exprFor(Coords[2], PrecUnknown);
    auto C3 = exprFor(Coords[3], PrecUnknown);
    if (!C1 || !C2 || !C3)
      return makeError("invalid gatherCubeArray coordinate");
    Coord = "float3(" + C0->Text + ", " + C1->Text + ", " + C2->Text + ")";
    Slice = sliceIndex(C3->Text);
    break;
  }
  default:
    return makeError("unsupported texture kind for gather '{0}'",
                     RS->ParamName);
  }

  std::string Offset;
  auto Offs = Gather.getOffsets();
  if (K != ResourceKind::TextureCube && K != ResourceKind::TextureCubeArray &&
      !Offs.empty()) {
    auto IntStr = [&](unsigned I) -> Expected<std::string> {
      if (I >= Offs.size() || isUndefLike(Offs[I]))
        return std::string("0");
      if (auto C = constInt(Offs[I]))
        return std::to_string(static_cast<int32_t>(*C));
      auto E = exprFor(Offs[I], PrecUnknown);
      if (!E)
        return E.takeError();
      return "int(" + E->Text + ")";
    };
    auto O0 = IntStr(0);
    auto O1 = IntStr(1);
    if (!O0 || !O1)
      return makeError("invalid gather offset");
    if (*O0 != "0" || *O1 != "0")
      Offset = "int2(" + *O0 + ", " + *O1 + ")";
  }

  auto RName = resourceExpr(Gather.getTex());
  if (!RName)
    return RName.takeError();
  auto SName = resourceExpr(Gather.getSampler());
  if (!SName)
    return SName.takeError();

  std::string CallExpr = *RName + ".";
  if (Gather.getCmpVal()) {
    auto Cmp = exprFor(Gather.getCmpVal(), PrecUnknown);
    if (!Cmp)
      return Cmp.takeError();
    CallExpr += "gather_compare(" + *SName + ", " + Coord;
    if (!Slice.empty())
      CallExpr += ", " + Slice;
    CallExpr += ", " + Cmp->Text;
    if (!Offset.empty())
      CallExpr += ", " + Offset;
    CallExpr += ")";
  } else {
    uint32_t Chan = static_cast<uint32_t>(Gather.getChannel());
    if (Chan > 3)
      return makeError("invalid textureGather channel");
    static const char *const Comps[4] = {"component::x", "component::y",
                                         "component::z", "component::w"};
    CallExpr += "gather(" + *SName + ", " + Coord;
    if (!Slice.empty())
      CallExpr += ", " + Slice;
    if (!Offset.empty())
      CallExpr += ", " + Offset;
    else if (K != ResourceKind::TextureCube &&
             K != ResourceKind::TextureCubeArray)
      CallExpr += ", int2(0)";
    CallExpr += std::string(", ") + Comps[Chan] + ")";
  }

  auto VecTy = mlir::cast<mlir::VectorType>(Gather.getResult().getType());
  auto RetElemTy = mslType(VecTy.getElementType());
  if (!RetElemTy)
    return RetElemTy.takeError();
  std::string MslVec = *RetElemTy + "4";
  if (*RetElemTy != RS->ElemType && !RS->IsDepthSampled)
    CallExpr = "as_type<" + MslVec + ">(" + CallExpr + ")";

  std::string Name = freshName();
  line(Ind, MslVec + " " + Name + " = " + CallExpr + ";");
  ValueNames[Gather.getResult()] = Name;
  if (Gather.getStatus())
    ValueNames[Gather.getStatus()] = "1u";
  return Error::success();
}

Error Emitter::emitAtomic(mlir::Operation *Op, unsigned Ind) {
  auto BinOp = mlir::dyn_cast<ir::BufferAtomicOp>(Op);
  auto CmpOp = mlir::dyn_cast<ir::BufferAtomicCmpXchgOp>(Op);
  mlir::Value Handle = BinOp ? BinOp.getHandle() : CmpOp.getHandle();
  mlir::Value ValOp = BinOp ? BinOp.getValue() : CmpOp.getValue();
  auto Coords = BinOp ? BinOp.getCoords() : CmpOp.getCoords();
  mlir::Value Res = BinOp ? BinOp.getResult() : CmpOp.getResult();

  bool Is64 = ValOp.getType().isInteger(64);
  ResState *RS = stateFor(Handle);
  if (!RS)
    return makeError("atomic through unresolved handle");
  if (RS->RB->Class != ResourceClass::UAV ||
      (!ValOp.getType().isInteger(32) && !Is64))
    return makeError("resource atomics require a 32-bit or 64-bit integer UAV");

  auto RName = resourceExpr(Handle);
  if (!RName)
    return RName.takeError();
  bool IsTexture = isTextureKind(RS->RB->Kind);
  if (IsTexture && Is64)
    return makeError("64-bit texture atomics are not supported on Metal");
  std::string Coord, LV;
  if (IsTexture) {
    if (RS->RB->Kind != ResourceKind::Texture2D ||
        (RS->ElemType != "int" && RS->ElemType != "uint"))
      return makeError("texture atomics require an integer RWTexture2D");
    auto X = exprFor(Coords[0], PrecUnknown);
    auto Y = exprFor(Coords[1], PrecUnknown);
    if (!X || !Y)
      return makeError("invalid texture atomic coordinate");
    Coord = "uint2(" + X->Text + ", " + Y->Text + ")";
  } else {
    mlir::Value C1 = Coords.size() > 1 ? Coords[1] : mlir::Value();
    auto Index = bufferIndexExpr(*RS, Coords[0], C1, 0);
    if (!Index)
      return Index.takeError();
    LV = *RName + "[" + *Index + "]";
  }
  auto TextureOperand = [&](StringRef V) -> std::string {
    return RS->ElemType == "int" ? ("as_type<int>(" + V + ")").str()
                                 : V.str();
  };
  auto TextureResult = [&](std::string V) {
    return RS->ElemType == "int" ? "as_type<uint>(" + V + ")" : V;
  };

  std::string Text;
  if (BinOp) {
    auto V = exprFor(ValOp, PrecUnknown);
    if (!V)
      return V.takeError();
    StringRef Fn;
    bool Signed = false;
    switch (BinOp.getOp()) {
    case ir::AtomicOp::Add:  Fn = "atomic_fetch_add"; break;
    case ir::AtomicOp::And:  Fn = "atomic_fetch_and"; break;
    case ir::AtomicOp::Or:   Fn = "atomic_fetch_or"; break;
    case ir::AtomicOp::Xor:  Fn = "atomic_fetch_xor"; break;
    case ir::AtomicOp::SMin: Fn = "atomic_fetch_min"; Signed = true; break;
    case ir::AtomicOp::SMax: Fn = "atomic_fetch_max"; Signed = true; break;
    case ir::AtomicOp::UMin: Fn = "atomic_fetch_min"; break;
    case ir::AtomicOp::UMax: Fn = "atomic_fetch_max"; break;
    case ir::AtomicOp::Xchg: Fn = "atomic_exchange"; break;
    }
    if (IsTexture) {
      if ((Fn == "atomic_fetch_min" || Fn == "atomic_fetch_max") &&
          Signed != (RS->ElemType == "int"))
        return makeError("texture atomic min/max signedness does not match the texture");
      Text = TextureResult(*RName + "." + Fn.str() + "(" + Coord +
                           ", " + RS->ElemType + "4(" +
                           TextureOperand(V->Text) + ")).x");
    } else if (Is64) {
      if (Signed) {
        bool IsMin = BinOp.getOp() == ir::AtomicOp::SMin;
        requireHelper(IsMin ? "crucible_atomic_imin_device_i64"
                            : "crucible_atomic_imax_device_i64");
        Text = std::string(IsMin ? "crucible_atomic_imin"
                                 : "crucible_atomic_imax") +
               "((device atomic_ulong *)&" + LV + ", " + V->Text + ")";
      } else {
        Text = Fn.str() + "_explicit((device atomic_ulong *)&" + LV + ", " +
               V->Text + ", memory_order_relaxed)";
      }
    } else if (Signed)
      Text = "as_type<uint>(" + Fn.str() + "_explicit((device atomic_int *)&" +
             LV + ", as_type<int>(" + V->Text + "), memory_order_relaxed))";
    else
      Text = Fn.str() + "_explicit((device atomic_uint *)&" + LV + ", " +
             V->Text + ", memory_order_relaxed)";
  } else {
    auto Cmp = exprFor(CmpOp.getCompare(), PrecUnknown);
    auto V = exprFor(ValOp, PrecUnknown);
    if (!Cmp || !V)
      return makeError("invalid cmpxchg operand");
    if (IsTexture) {
      requireHelper("crucible_atomic_cmpxchg_texture2d");
      Text = TextureResult("crucible_atomic_cmpxchg_texture2d(" +
                           *RName + ", " + Coord + ", " +
                           TextureOperand(Cmp->Text) + ", " +
                           TextureOperand(V->Text) + ")");
    } else if (Is64) {
      requireHelper("crucible_atomic_cmpxchg_device_i64");
      Text = "crucible_atomic_cmpxchg((device atomic_ulong *)&" + LV + ", " +
             Cmp->Text + ", " + V->Text + ")";
    } else {
      requireHelper("crucible_atomic_cmpxchg_device");
      Text = "crucible_atomic_cmpxchg((device atomic_uint *)&" + LV + ", " +
             Cmp->Text + ", " + V->Text + ")";
    }
  }

  if (IsTexture)
    RS->RB->RequiresTextureAtomics = true;
  if (!Res.use_empty()) {
    std::string Name = freshName();
    line(Ind, std::string(Is64 ? "ulong " : "uint ") + Name + " = " + Text +
                  ";");
    ValueNames[Res] = Name;
  } else {
    line(Ind, Text + ";");
  }
  return Error::success();
}

Error Emitter::emitIf(mlir::scf::IfOp If, unsigned Ind) {
  // Results become mutable locals assigned by each branch's yield.
  SmallVector<std::string, 2> ResultNames;
  for (mlir::Value Res : If.getResults()) {
    auto Ty = mslType(Res.getType());
    if (!Ty)
      return Ty.takeError();
    std::string Name = freshName();
    line(Ind, *Ty + " " + Name + ";");
    ValueNames[Res] = Name;
    ResultNames.push_back(Name);
  }

  auto Cond = exprFor(If.getCondition(), PrecUnknown);
  if (!Cond)
    return Cond.takeError();
  line(Ind, "if (" + Cond->Text + ") {");

  auto EmitRegionWithYield = [&](mlir::Region &Reg) -> Error {
    mlir::Block &B = Reg.front();
    if (Error E = emitBlock(B, Ind + 1))
      return E;
    auto Yield = mlir::cast<mlir::scf::YieldOp>(B.getTerminator());
    for (auto [Name, V] : zip(ResultNames, Yield.getOperands())) {
      auto E = exprFor(V, PrecUnknown);
      if (!E)
        return E.takeError();
      line(Ind + 1, Name + " = " + E->Text + ";");
    }
    return Error::success();
  };

  if (Error E = EmitRegionWithYield(If.getThenRegion()))
    return E;
  if (!If.getElseRegion().empty()) {
    line(Ind, "} else {");
    if (Error E = EmitRegionWithYield(If.getElseRegion()))
      return E;
  }
  line(Ind, "}");
  return Error::success();
}

Error Emitter::emitSwitch(mlir::scf::IndexSwitchOp Switch, unsigned Ind) {
  SmallVector<std::string, 2> ResultNames;
  for (mlir::Value Res : Switch.getResults()) {
    auto Ty = mslType(Res.getType());
    if (!Ty)
      return Ty.takeError();
    std::string Name = freshName();
    line(Ind, *Ty + " " + Name + ";");
    ValueNames[Res] = Name;
    ResultNames.push_back(Name);
  }
  // Avoid 64-bit switch discriminants on GPU when the index selector comes
  // from a 32-bit integer and all case labels fit in 32 bits.
  mlir::Value SelVal = Switch.getArg();
  if (auto *Def = SelVal.getDefiningOp();
      Def && mlir::isa<mlir::arith::IndexCastOp, mlir::arith::IndexCastUIOp>(Def) &&
      Def->getOperand(0).getType().isInteger(32)) {
    bool IsUnsigned = mlir::isa<mlir::arith::IndexCastUIOp>(Def);
    bool Fits32 = true;
    for (int64_t V : Switch.getCases()) {
      if (IsUnsigned ? (V < 0 || V > int64_t(UINT32_MAX))
                     : (V < INT32_MIN || V > INT32_MAX)) {
        Fits32 = false;
        break;
      }
    }
    if (Fits32)
      SelVal = Def->getOperand(0);
  }
  auto Selector = exprFor(SelVal, PrecUnknown);
  if (!Selector)
    return Selector.takeError();
  if (SelVal != Switch.getArg() &&
      mlir::isa<mlir::arith::IndexCastOp>(Switch.getArg().getDefiningOp()))
    Selector->Text = signedView(SelVal.getType(), Selector->Text);
  line(Ind, "switch (" + Selector->Text + ") {");
  auto EmitCase = [&](mlir::Block &B, const std::string &Label) -> Error {
    line(Ind + 1, Label + ": {");
    if (Error E = emitBlock(B, Ind + 2))
      return E;
    auto Yield = mlir::cast<mlir::scf::YieldOp>(B.getTerminator());
    for (auto [Name, V] : zip(ResultNames, Yield.getOperands())) {
      auto E = exprFor(V, PrecUnknown);
      if (!E)
        return E.takeError();
      line(Ind + 2, Name + " = " + E->Text + ";");
    }
    line(Ind + 2, "break;");
    line(Ind + 1, "}");
    return Error::success();
  };
  for (auto [I, Value] : enumerate(Switch.getCases()))
    if (Error E = EmitCase(Switch.getCaseBlock(I),
                           "case " + std::to_string(Value)))
      return E;
  if (Error E = EmitCase(Switch.getDefaultBlock(), "default"))
    return E;
  line(Ind, "}");
  return Error::success();
}

// Returns true when the after-region only forwards its arguments 1:1 to
// scf.yield -- the rotated do-while shape dxc commonly produces.
static bool isTrivialAfterRegion(mlir::scf::WhileOp While) {
  mlir::Block &After = While.getAfter().front();
  if (&After.front() != After.getTerminator())
    return false;
  auto Yield = mlir::cast<mlir::scf::YieldOp>(After.getTerminator());
  if (Yield.getNumOperands() != After.getNumArguments())
    return false;
  for (auto [Op, Arg] : zip(Yield.getOperands(), After.getArguments()))
    if (Op != Arg)
      return false;
  return true;
}

Error Emitter::emitWhile(mlir::scf::WhileOp While, unsigned Ind) {
  mlir::Block &Before = While.getBefore().front();
  auto Condition =
      mlir::cast<mlir::scf::ConditionOp>(Before.getTerminator());

  // Loop-carried variables, initialized from the init operands.
  SmallVector<std::string, 4> CarriedNames;
  for (auto [Arg, Init] : zip(Before.getArguments(), While.getInits())) {
    auto Ty = mslType(Arg.getType());
    if (!Ty)
      return Ty.takeError();
    auto InitE = exprFor(Init, PrecUnknown);
    if (!InitE)
      return InitE.takeError();
    std::string Name = freshName();
    line(Ind, *Ty + " " + Name + " = " + InitE->Text + ";");
    ValueNames[Arg] = Name;
    CarriedNames.push_back(Name);
  }

  if (isTrivialAfterRegion(While)) {
    // do { <before>; carried = condArgs; } while (cond);
    // Results alias the carried variables (after trivially forwards them).
    for (auto [Res, Name] : zip(While.getResults(), CarriedNames))
      ValueNames[Res] = Name;

    // The do-while condition is evaluated outside the body's scope, so it
    // lives in a variable declared before the loop and assigned inside --
    // *before* the carried variables are mutated (parallel-copy hazard).
    std::string CondVar = freshName();
    line(Ind, "bool " + CondVar + ";");
    line(Ind, "do {");
    if (Error E = emitBlock(Before, Ind + 1))
      return E;

    auto CondE = exprFor(Condition.getCondition(), PrecUnknown);
    if (!CondE)
      return CondE.takeError();
    line(Ind + 1, CondVar + " = " + CondE->Text + ";");
    SmallVector<std::string, 4> NewVals;
    for (mlir::Value V : Condition.getArgs()) {
      auto Name = ensureNamed(V, Ind + 1, /*CopyBlockArgs=*/true);
      if (!Name)
        return Name.takeError();
      NewVals.push_back(*Name);
    }
    for (auto [Name, NewVal] : zip(CarriedNames, NewVals))
      if (Name != NewVal)
        line(Ind + 1, Name + " = " + NewVal + ";");
    line(Ind, "} while (" + CondVar + ");");
    return Error::success();
  }

  mlir::Block &After = While.getAfter().front();

  // Check for the LiftControlFlowToSCF early-exit loop shape:
  //   - `After` is a pass-through block (only `scf.yield` of its block args),
  //   - `Before` ends with an `scf.if` (plus optional `arith.trunci` to `i1`)
  //     that feeds `scf.condition`'s continuation flag and early-exit results.
  // Lowering this directly to `if (exit) { res = ...; break; }` eliminates
  // synthetic `uint` continue flags, dummy zero assignments, and nested `else`
  // blocks so Metal's AGX compiler sees clean structured early-exit loops.
  auto tryEmitEarlyExitWhile = [&]() -> Expected<bool> {
    if (&After.front() != After.getTerminator())
      return false;
    auto AfterYield = mlir::cast<mlir::scf::YieldOp>(After.getTerminator());
    if (AfterYield.getNumOperands() != CarriedNames.size())
      return false;
    SmallVector<unsigned, 4> YieldArgIdx;
    for (mlir::Value V : AfterYield.getOperands()) {
      auto BA = mlir::dyn_cast<mlir::BlockArgument>(V);
      if (!BA || BA.getOwner() != &After)
        return false;
      YieldArgIdx.push_back(BA.getArgNumber());
    }

    mlir::Value CondVal = Condition.getCondition();
    mlir::Operation *TruncOp = nullptr;
    mlir::Value FlagVal = CondVal;
    if (auto *Def = CondVal.getDefiningOp();
        Def && Def->getBlock() == &Before &&
        mlir::isa<mlir::arith::TruncIOp, mlir::LLVM::TruncOp>(Def) &&
        Def->getNextNode() == Condition.getOperation()) {
      TruncOp = Def;
      FlagVal = Def->getOperand(0);
    }
    auto ExitIf = FlagVal.getDefiningOp<mlir::scf::IfOp>();
    if (!ExitIf || ExitIf->getBlock() != &Before ||
        ExitIf.getElseRegion().empty())
      return false;
    mlir::Operation *ExpectedNext =
        TruncOp ? TruncOp : Condition.getOperation();
    if (ExitIf->getNextNode() != ExpectedNext)
      return false;

    // Every result of ExitIf must be consumed only by TruncOp or Condition.
    for (mlir::Value Res : ExitIf.getResults()) {
      for (mlir::Operation *U : Res.getUsers())
        if (U != TruncOp && U != Condition.getOperation())
          return false;
    }
    unsigned RootFlagIdx =
        mlir::cast<mlir::OpResult>(FlagVal).getResultNumber();

    // Only declare result variables that are actually used after the loop.
    SmallVector<std::string, 4> LiveResultNames(While.getNumResults());
    for (unsigned I = 0, E = While.getNumResults(); I < E; ++I) {
      mlir::Value Res = While.getResult(I);
      if (Res.use_empty())
        continue;
      auto Ty = mslType(Res.getType());
      if (!Ty)
        return Ty.takeError();
      std::string Name = freshName();
      line(Ind, *Ty + " " + Name + ";");
      ValueNames[Res] = Name;
      LiveResultNames[I] = Name;
    }

    auto EmitExitAssigns = [&](ArrayRef<mlir::Value> SubArgs,
                               unsigned BInd) -> Error {
      for (unsigned I = 0, E = LiveResultNames.size(); I < E; ++I) {
        if (LiveResultNames[I].empty())
          continue;
        auto ValE = exprFor(SubArgs[I], PrecUnknown);
        if (!ValE)
          return ValE.takeError();
        line(BInd, LiveResultNames[I] + " = " + ValE->Text + ";");
      }
      return Error::success();
    };

    auto EmitExitTree = [&](auto &Self, mlir::scf::IfOp IfOp, unsigned FlagIdx,
                            SmallVector<mlir::Value, 8> CurArgs,
                            unsigned CurInd) -> Error {
      auto EmitBranch = [&](mlir::Block &BB, SmallVector<mlir::Value, 8> SubArgs,
                            mlir::Value BranchFlag, unsigned BInd) -> Error {
        // Case 1: Unconditional break branch (flag == 0).
        if (auto C = constInt(BranchFlag); C && *C == 0) {
          if (Error E = emitBlock(BB, BInd))
            return E;
          if (Error E = EmitExitAssigns(SubArgs, BInd))
            return E;
          line(BInd, "break;");
          return Error::success();
        }

        // Case 2: Nested early-exit scf.if at the end of BB.
        auto Term = mlir::cast<mlir::scf::YieldOp>(BB.getTerminator());
        if (auto NestedIf = BranchFlag.getDefiningOp<mlir::scf::IfOp>();
            NestedIf && NestedIf->getBlock() == &BB &&
            !NestedIf.getElseRegion().empty() &&
            NestedIf->getNextNode() == Term.getOperation()) {
          bool OnlyUsedByYield = true;
          for (mlir::Value Res : NestedIf.getResults()) {
            for (mlir::Operation *U : Res.getUsers())
              if (U != Term.getOperation()) {
                OnlyUsedByYield = false;
                break;
              }
            if (!OnlyUsedByYield)
              break;
          }
          if (OnlyUsedByYield) {
            if (Error E = emitBlock(BB, BInd, NestedIf.getOperation()))
              return E;
            unsigned NestedFlagIdx =
                mlir::cast<mlir::OpResult>(BranchFlag).getResultNumber();
            return Self(Self, NestedIf, NestedFlagIdx, std::move(SubArgs),
                        BInd);
          }
        }

        // Case 3: Continuation leaf -- evaluate continuation condition, break
        // if false, otherwise update loop-carried variables.
        if (Error E = emitBlock(BB, BInd))
          return E;

        // Unwrap zext/extui from i1 so we emit `if (!(cond))` directly.
        mlir::Value BoolFlag = BranchFlag;
        if (auto *Def = BranchFlag.getDefiningOp();
            Def && mlir::isa<mlir::arith::ExtUIOp, mlir::LLVM::ZExtOp>(Def) &&
            Def->getOperand(0).getType().isInteger(1)) {
          BoolFlag = Def->getOperand(0);
        }
        std::string ContText;
        if (BoolFlag.getType().isInteger(1)) {
          auto CE = exprFor(BoolFlag, PrecUnary);
          if (!CE)
            return CE.takeError();
          ContText = CE->Text;
        } else {
          auto CE = exprFor(BoolFlag, PrecUnknown);
          if (!CE)
            return CE.takeError();
          ContText = "bool((" + CE->Text + ") & 1u)";
        }

        bool HasLiveResults = llvm::any_of(
            LiveResultNames, [](const std::string &S) { return !S.empty(); });
        if (HasLiveResults) {
          line(BInd, "if (!" + ContText + ") {");
          if (Error E = EmitExitAssigns(SubArgs, BInd + 1))
            return E;
          line(BInd + 1, "break;");
          line(BInd, "}");
        } else {
          line(BInd, "if (!" + ContText + ") break;");
        }

        SmallVector<std::string, 4> NewVals;
        for (unsigned Idx : YieldArgIdx) {
          auto Name = ensureNamed(SubArgs[Idx], BInd, /*CopyBlockArgs=*/true);
          if (!Name)
            return Name.takeError();
          NewVals.push_back(*Name);
        }
        for (auto [Name, NewVal] : zip(CarriedNames, NewVals))
          if (Name != NewVal)
            line(BInd, Name + " = " + NewVal + ";");
        return Error::success();
      };

      auto Substitute = [&](mlir::scf::YieldOp Y) {
        SmallVector<mlir::Value, 8> Out = CurArgs;
        for (mlir::Value &V : Out) {
          if (auto OpRes = mlir::dyn_cast<mlir::OpResult>(V);
              OpRes && OpRes.getOwner() == IfOp.getOperation()) {
            V = Y.getOperand(OpRes.getResultNumber());
          }
        }
        return Out;
      };

      mlir::Block &ThenB = IfOp.getThenRegion().front();
      mlir::Block &ElseB = IfOp.getElseRegion().front();
      auto ThenY = mlir::cast<mlir::scf::YieldOp>(ThenB.getTerminator());
      auto ElseY = mlir::cast<mlir::scf::YieldOp>(ElseB.getTerminator());
      auto ThenArgs = Substitute(ThenY);
      auto ElseArgs = Substitute(ElseY);
      mlir::Value ThenFlag = ThenY.getOperand(FlagIdx);
      mlir::Value ElseFlag = ElseY.getOperand(FlagIdx);
      bool ThenBreaks = constInt(ThenFlag).value_or(1) == 0;
      bool ElseBreaks = constInt(ElseFlag).value_or(1) == 0;

      if (ThenBreaks && !ElseBreaks) {
        auto CondE = exprFor(IfOp.getCondition(), PrecUnknown);
        if (!CondE)
          return CondE.takeError();
        line(CurInd, "if (" + CondE->Text + ") {");
        if (Error E = EmitBranch(ThenB, std::move(ThenArgs), ThenFlag,
                                 CurInd + 1))
          return E;
        line(CurInd, "}");
        return EmitBranch(ElseB, std::move(ElseArgs), ElseFlag, CurInd);
      }
      if (ElseBreaks && !ThenBreaks) {
        auto CondE = exprFor(IfOp.getCondition(), PrecUnary);
        if (!CondE)
          return CondE.takeError();
        line(CurInd, "if (!" + CondE->Text + ") {");
        if (Error E = EmitBranch(ElseB, std::move(ElseArgs), ElseFlag,
                                 CurInd + 1))
          return E;
        line(CurInd, "}");
        return EmitBranch(ThenB, std::move(ThenArgs), ThenFlag, CurInd);
      }
      auto CondE = exprFor(IfOp.getCondition(), PrecUnknown);
      if (!CondE)
        return CondE.takeError();
      line(CurInd, "if (" + CondE->Text + ") {");
      if (Error E = EmitBranch(ThenB, std::move(ThenArgs), ThenFlag,
                               CurInd + 1))
        return E;
      line(CurInd, "} else {");
      if (Error E = EmitBranch(ElseB, std::move(ElseArgs), ElseFlag,
                               CurInd + 1))
        return E;
      line(CurInd, "}");
      return Error::success();
    };

    line(Ind, "while (true) {");
    if (Error E = emitBlock(Before, Ind + 1, ExitIf.getOperation()))
      return E;
    SmallVector<mlir::Value, 8> RootArgs(Condition.getArgs().begin(),
                                         Condition.getArgs().end());
    if (Error E = EmitExitTree(EmitExitTree, ExitIf, RootFlagIdx,
                               std::move(RootArgs), Ind + 1))
      return E;
    line(Ind, "}");
    return true;
  };

  auto EarlyExit = tryEmitEarlyExitWhile();
  if (!EarlyExit)
    return EarlyExit.takeError();
  if (*EarlyExit)
    return Error::success();

  // General form:
  //   while (true) { <before>; r = condArgs; if (!cond) break; <after>;
  //                  carried = yields; }
  SmallVector<std::string, 4> ResultNames;
  for (auto [Res, Arg] : zip(While.getResults(), After.getArguments())) {
    auto Ty = mslType(Res.getType());
    if (!Ty)
      return Ty.takeError();
    std::string Name = freshName();
    line(Ind, *Ty + " " + Name + ";");
    ValueNames[Res] = Name;
    ValueNames[Arg] = Name;
    ResultNames.push_back(Name);
  }

  line(Ind, "while (true) {");
  if (Error E = emitBlock(Before, Ind + 1))
    return E;
  for (auto [Name, V] : zip(ResultNames, Condition.getArgs())) {
    auto E = exprFor(V, PrecUnknown);
    if (!E)
      return E.takeError();
    line(Ind + 1, Name + " = " + E->Text + ";");
  }
  auto Cond = exprFor(Condition.getCondition(), PrecUnary);
  if (!Cond)
    return Cond.takeError();
  line(Ind + 1, "if (!" + Cond->Text + ") break;");

  if (Error E = emitBlock(After, Ind + 1))
    return E;
  auto Yield = mlir::cast<mlir::scf::YieldOp>(After.getTerminator());
  for (auto [Name, V] : zip(CarriedNames, Yield.getOperands())) {
    auto E = exprFor(V, PrecUnknown);
    if (!E)
      return E.takeError();
    if (E->Text != Name)
      line(Ind + 1, Name + " = " + E->Text + ";");
  }
  line(Ind, "}");
  return Error::success();
}

Error Emitter::emitStatement(mlir::Operation *Op, unsigned Ind) {
  // Structure ops.
  if (auto If = mlir::dyn_cast<mlir::scf::IfOp>(Op))
    return emitIf(If, Ind);
  if (auto While = mlir::dyn_cast<mlir::scf::WhileOp>(Op))
    return emitWhile(While, Ind);
  if (auto Switch = mlir::dyn_cast<mlir::scf::IndexSwitchOp>(Op))
    return emitSwitch(Switch, Ind);
  if (auto Barrier = mlir::dyn_cast<mlir::gpu::BarrierOp>(Op)) {
    std::string Flags;
    if (!Barrier->hasAttr("crucible.no_tgsm_fence"))
      Flags = "mem_flags::mem_threadgroup";
    if (Barrier->hasAttr("crucible.device_fence")) {
      Flags = Flags.empty() ? "mem_flags::mem_device"
                            : Flags + " | mem_flags::mem_device";
      if (any_of(ResStates, [](const ResState &RS) {
            return RS.RB->Class == ResourceClass::UAV &&
                   isTextureKind(RS.RB->Kind);
          }))
        Flags += " | mem_flags::mem_texture";
    }
    if (Flags.empty())
      Flags = "mem_flags::mem_none";
    line(Ind, "threadgroup_barrier(" + Flags + ");");
    return Error::success();
  }
  if (auto Bar = mlir::dyn_cast<ir::BarrierOp>(Op)) {
    std::string Flags;
    if (Bar.getHandle()) {
      ResState *RS = stateFor(Bar.getHandle());
      if (!RS)
        return makeError("barrier through unresolved handle");
      Flags = isTextureKind(RS->RB->Kind) ? "mem_flags::mem_texture"
                                          : "mem_flags::mem_device";
    } else {
      if (Bar.getTgsmFence())
        Flags = "mem_flags::mem_threadgroup";
      if (Bar.getDeviceFence()) {
        Flags = Flags.empty() ? "mem_flags::mem_device"
                              : Flags + " | mem_flags::mem_device";
        if (any_of(ResStates, [](const ResState &RS) {
              return RS.RB->Class == ResourceClass::UAV &&
                     isTextureKind(RS.RB->Kind);
            }))
          Flags += " | mem_flags::mem_texture";
      }
    }
    if (Flags.empty()) {
      if (!Bar.getExecSync())
        return Error::success();
      Flags = "mem_flags::mem_none";
    }
    line(Ind, "threadgroup_barrier(" + Flags + ");");
    return Error::success();
  }
  if (auto WB = mlir::dyn_cast<ir::WaveBallotOp>(Op)) {
    requireHelper("crucible_wave_ballot");
    auto Cond = exprFor(WB.getCond(), PrecUnknown);
    if (!Cond)
      return Cond.takeError();
    std::string Name = freshName();
    line(Ind, "uint4 " + Name + " = crucible_wave_ballot(" + Cond->Text + ");");
    ValueNames[WB.getResult()] = Name;
    return Error::success();
  }
  if (auto UP = mlir::dyn_cast<ir::Unpack4x8Op>(Op)) {
    uint8_t Mode = static_cast<uint8_t>(UP.getMode());
    if (Mode != 0 && Mode != 1)
      return makeError("unsupported unpack4x8 mode");
    auto PK = exprFor(UP.getPacked(), PrecUnknown);
    if (!PK)
      return PK.takeError();
    auto VecTy = mlir::cast<mlir::VectorType>(UP.getResult().getType());
    bool Is16 = VecTy.getElementType().isInteger(16);
    std::string MslVec = Is16 ? "ushort4" : "uint4";
    std::string CastExpr;
    if (Mode == 0) {
      CastExpr = MslVec + "(as_type<uchar4>(" + PK->Text + "))";
    } else {
      std::string SignedVec = Is16 ? "short4" : "int4";
      CastExpr = "as_type<" + MslVec + ">(" + SignedVec + "(as_type<char4>(" +
                 PK->Text + ")))";
    }
    std::string Name = freshName();
    line(Ind, MslVec + " " + Name + " = " + CastExpr + ";");
    ValueNames[UP.getResult()] = Name;
    return Error::success();
  }
  if (auto SD = mlir::dyn_cast<ir::SplitDoubleOp>(Op)) {
    if (!Opts.EmulateF64)
      return makeError("MSL has no double type; use float in the shader");
    auto A = exprFor(SD.getValue(), PrecUnknown);
    if (!A)
      return A.takeError();
    std::string LoName = freshName();
    std::string HiName = freshName();
    line(Ind, "uint " + LoName + " = uint((" + A->Text + ") & 0xfffffffful);");
    line(Ind, "uint " + HiName + " = uint((" + A->Text + ") >> 32ul);");
    ValueNames[SD.getLo()] = LoName;
    ValueNames[SD.getHi()] = HiName;
    return Error::success();
  }
  if (auto UA = mlir::dyn_cast<ir::UAddcOp>(Op)) {
    auto A = exprFor(UA.getLhs(), PrecUnknown);
    auto B = exprFor(UA.getRhs(), PrecUnknown);
    if (!A || !B)
      return makeError("invalid crucible.uaddc operand");
    std::string Tmp = freshName();
    std::string Sum = freshName();
    std::string Carry = freshName();
    line(Ind, "ulong " + Tmp + " = ulong(" + A->Text + ") + ulong(" + B->Text +
                  ");");
    line(Ind, "uint " + Sum + " = uint(" + Tmp + ");");
    line(Ind, "bool " + Carry + " = (" + Tmp + " >> 32ul) != 0ul;");
    ValueNames[UA.getSum()] = Sum;
    ValueNames[UA.getCarry()] = Carry;
    return Error::success();
  }
  if (auto Shuf = mlir::dyn_cast<mlir::gpu::ShuffleOp>(Op)) {
    auto Ty = mslType(Shuf.getShuffleResult().getType());
    if (!Ty)
      return Ty.takeError();
    auto E = opExpr(Shuf);
    if (!E)
      return E.takeError();
    std::string Name = freshName();
    line(Ind, *Ty + " " + Name + " = " + E->Text + ";");
    ValueNames[Shuf.getShuffleResult()] = Name;
    ValueNames[Shuf.getValid()] = "true";
    return Error::success();
  }
  if (mlir::isa<mlir::scf::ForOp>(Op))
    return makeError("scf.for emission is not implemented yet");

  // Region terminators are handled by their parents.
  if (mlir::isa<mlir::scf::YieldOp, mlir::scf::ConditionOp>(Op))
    return Error::success();

  if (auto Ret = mlir::dyn_cast<mlir::func::ReturnOp>(Op)) {
    if (Ret.getNumOperands() != 0)
      return makeError("non-void entry points are not supported");
    return Error::success();
  }

  if (isConstantLike(Op))
    return Error::success();

  if (auto SO = mlir::dyn_cast<ir::StoreOutputOp>(Op)) {
    uint32_t ID = static_cast<uint32_t>(SO.getOutputId());
    uint32_t Col = static_cast<uint32_t>(static_cast<uint8_t>(SO.getCol()));
    if (R.Hull)
      return hullStore(ID, SO.getRow(), Col, SO.getValue(),
                       /*IsPatchConstant=*/false, Ind);
    auto Access = stageElement(ID, SO.getRow(), Col, /*Input=*/false);
    if (!Access)
      return Access.takeError();
    auto Value = exprFor(SO.getValue(), PrecUnknown);
    if (!Value)
      return Value.takeError();
    line(Ind, *Access + " = " + Value->Text + ";");
    return Error::success();
  }
  if (auto SP = mlir::dyn_cast<ir::StorePatchConstantOp>(Op)) {
    uint32_t ID = static_cast<uint32_t>(SP.getOutputId());
    uint32_t Col = static_cast<uint32_t>(static_cast<uint8_t>(SP.getCol()));
    return hullStore(ID, SP.getRow(), Col, SP.getValue(),
                     /*IsPatchConstant=*/true, Ind);
  }
  if (auto D = mlir::dyn_cast<ir::DiscardOp>(Op)) {
    if (R.Stage != ShaderStage::Pixel)
      return makeError("discard requires a fragment shader");
    auto Cond = exprFor(D.getCond(), PrecUnknown);
    if (!Cond)
      return Cond.takeError();
    line(Ind, "if (" + Cond->Text + ") discard_fragment();");
    return Error::success();
  }
  if (mlir::isa<ir::EmitStreamOp>(Op)) {
    if (R.Geometry)
      return geometryEmit(Ind, true, false);
    return Error::success();
  }
  if (mlir::isa<ir::CutStreamOp>(Op)) {
    if (R.Geometry)
      return geometryEmit(Ind, false, true);
    return Error::success();
  }

  if (auto BL = mlir::dyn_cast<ir::BufferLoadOp>(Op);
      BL && BL->hasAttr("crucible.volatile")) {
    if (BL.getStatus())
      return makeError("volatile buffer load with residency is unsupported");
    auto VecTy = mlir::cast<mlir::VectorType>(BL.getResult().getType());
    auto Ty = mslType(VecTy);
    if (!Ty)
      return Ty.takeError();
    std::string Name = freshName();
    line(Ind, *Ty + " " + Name + " = " + *Ty + "(0);");
    uint32_t Mask = BL.getMask().value_or(15);
    for (unsigned I = 0; I < VecTy.getNumElements() && I < 4; ++I) {
      if (!(Mask & (1u << I)))
        continue;
      auto E = crucibleExtractExpr(BL, I, VecTy.getElementType());
      if (!E)
        return E.takeError();
      line(Ind, Name + "." + std::string(1, "xyzw"[I]) + " = " + E->Text + ";");
    }
    ValueNames[BL.getResult()] = Name;
    return Error::success();
  }

  if (auto BL = mlir::dyn_cast<ir::BufferLoadOp>(Op);
      BL && !BL->hasAttr("crucible.volatile") && !BL.getStatus() &&
      !BL.getResult().use_empty()) {
    ResState *RS = stateFor(BL.getHandle());
    auto VecTy = mlir::dyn_cast<mlir::VectorType>(BL.getResult().getType());
    if (RS && VecTy) {
      uint32_t UsedMask = 0;
      bool AllConstExtract = true;
      for (mlir::Operation *U : BL.getResult().getUsers()) {
        auto EE = mlir::dyn_cast<mlir::LLVM::ExtractElementOp>(U);
        if (!EE) {
          AllConstExtract = false;
          break;
        }
        auto Pos = constInt(EE.getPosition());
        if (!Pos || *Pos >= 4) {
          AllConstExtract = false;
          break;
        }
        UsedMask |= (1u << static_cast<unsigned>(*Pos));
      }
      unsigned UsedCount = llvm::popcount(UsedMask);
      unsigned VecWidth = (UsedMask & 0xCu) ? 4 : 2;
      if (AllConstExtract && UsedCount >= 2 &&
          canVectorizeBufferAccess(*RS, BL.getIndex(), BL.getElementOffset(),
                                   VecWidth)) {
        auto RName = resourceExpr(BL.getHandle());
        if (!RName)
          return RName.takeError();
        auto BaseIndex =
            bufferIndexExpr(*RS, BL.getIndex(), BL.getElementOffset(), 0);
        if (!BaseIndex)
          return BaseIndex.takeError();
        auto LoadElemTy = mslType(VecTy.getElementType());
        if (!LoadElemTy)
          return LoadElemTy.takeError();
        if (elemByteSize(*LoadElemTy) == 4) {
          std::string LoadVecTy = *LoadElemTy + std::to_string(VecWidth);
          std::string Qual = RS->RB->Class == ResourceClass::UAV
                                 ? "device "
                                 : "device const ";
          std::string Name = freshName();
          line(Ind, LoadVecTy + " " + Name + " = *(" + Qual + LoadVecTy +
                        " *)(&" + *RName + "[" + *BaseIndex + "]);");
          ValueNames[BL.getResult()] = Name;
          return Error::success();
        }
      }
    }
  }

  if (auto CB = mlir::dyn_cast<ir::CBufferLoadOp>(Op);
      CB && !CB.getResult().use_empty() && !constInt(CB.getRegIndex())) {
    ResState *RS = stateFor(CB.getHandle());
    auto VecTy = mlir::dyn_cast<mlir::VectorType>(CB.getResult().getType());
    if (RS && VecTy && VecTy.getNumElements() == 4 &&
        VecTy.getElementType().isF32() && RS->ElemType == "float4") {
      uint32_t UsedMask = 0;
      bool AllConstExtract = true;
      for (mlir::Operation *U : CB.getResult().getUsers()) {
        auto EE = mlir::dyn_cast<mlir::LLVM::ExtractElementOp>(U);
        if (!EE) {
          AllConstExtract = false;
          break;
        }
        auto Pos = constInt(EE.getPosition());
        if (!Pos || *Pos >= 4) {
          AllConstExtract = false;
          break;
        }
        UsedMask |= (1u << static_cast<unsigned>(*Pos));
      }
      if (AllConstExtract && llvm::popcount(UsedMask) >= 2) {
        auto RName = resourceExpr(CB.getHandle());
        if (!RName)
          return RName.takeError();
        auto Row = exprFor(CB.getRegIndex(), PrecUnknown);
        if (!Row)
          return Row.takeError();
        std::string Name = freshName();
        line(Ind, "float4 " + Name + " = " + *RName + "[" + Row->Text + "];");
        ValueNames[CB.getResult()] = Name;
        return Error::success();
      }
    }
  }

  if (mlir::isa<ir::BindResourceOp, ir::HeapResourceOp, ir::BufferLoadOp,
                ir::CBufferLoadOp, ir::TexGetDimensionsOp,
                ir::GetMeshPayloadOp>(Op))
    return Error::success(); // folded into bindings / extractelement / lvalue

  if (auto BS = mlir::dyn_cast<ir::BufferStoreOp>(Op))
    return emitBufferStore(BS, Ind);
  if (auto BUC = mlir::dyn_cast<ir::BufferUpdateCounterOp>(Op)) {
    ResState *RS = stateFor(BUC.getHandle());
    if (!RS || !RS->RB->HasCounter)
      return makeError("bufferUpdateCounter on resource without counter");
    int8_t IncVal = BUC.getInc();
    if (IncVal != 1 && IncVal != -1)
      return makeError("unsupported bufferUpdateCounter increment {0}",
                       static_cast<int>(IncVal));
    auto CPtr = resourceExpr(BUC.getHandle(), "_counter");
    if (!CPtr)
      return CPtr.takeError();
    std::string ExprText =
        IncVal == 1
            ? "atomic_fetch_add_explicit(" + *CPtr +
                  ", 1u, memory_order_relaxed)"
            : "(atomic_fetch_sub_explicit(" + *CPtr +
                  ", 1u, memory_order_relaxed) - 1u)";
    if (!BUC.getResult().use_empty()) {
      std::string Name = freshName();
      line(Ind, "uint " + Name + " = " + ExprText + ";");
      ValueNames[BUC.getResult()] = Name;
    } else {
      line(Ind, ExprText + ";");
    }
    return Error::success();
  }
  if (auto TL = mlir::dyn_cast<ir::TexLoadOp>(Op))
    return emitTextureLoad(TL, Ind);
  if (auto TS = mlir::dyn_cast<ir::TexStoreOp>(Op))
    return emitTextureStore(TS, Ind);
  if (auto Samp = mlir::dyn_cast<ir::TexSampleOp>(Op))
    return emitTextureSample(Samp, Ind);
  if (auto Gath = mlir::dyn_cast<ir::TexGatherOp>(Op))
    return emitTextureGather(Gath, Ind);
  if (auto Lod = mlir::dyn_cast<ir::TexCalcLodOp>(Op)) {
    ResState *RS = stateFor(Lod.getTex());
    if (!RS)
      return makeError("calculateLOD through unresolved texture handle");
    auto RName = resourceExpr(Lod.getTex());
    auto SName = resourceExpr(Lod.getSampler());
    if (!RName || !SName)
      return makeError("calculateLOD through unresolved handle");
    auto Coords = Lod.getCoords();
    auto C0 = exprFor(Coords[0], PrecUnknown);
    if (!C0)
      return C0.takeError();
    std::string Coord;
    ResourceKind K = RS->RB->Kind;
    if (K == ResourceKind::Texture1D || K == ResourceKind::Texture1DArray) {
      Coord = "float2(" + C0->Text + ", 0.5f)";
    } else if (K == ResourceKind::Texture2D ||
               K == ResourceKind::Texture2DArray) {
      auto C1 = exprFor(Coords[1], PrecUnknown);
      if (!C1)
        return C1.takeError();
      Coord = "float2(" + C0->Text + ", " + C1->Text + ")";
    } else {
      auto C1 = exprFor(Coords[1], PrecUnknown);
      auto C2 = exprFor(Coords[2], PrecUnknown);
      if (!C1 || !C2)
        return makeError("invalid calculateLOD coordinate");
      Coord = "float3(" + C0->Text + ", " + C1->Text + ", " + C2->Text + ")";
    }
    std::string Fn = Lod.getClamped() ? "calculate_clamped_lod"
                                      : "calculate_unclamped_lod";
    std::string Name = freshName();
    line(Ind, "float " + Name + " = " + *RName + "." + Fn + "(" + *SName +
                  ", " + Coord + ");");
    ValueNames[Lod.getResult()] = Name;
    return Error::success();
  }
  if (mlir::isa<ir::BufferAtomicOp, ir::BufferAtomicCmpXchgOp>(Op))
    return emitAtomic(Op, Ind);
  if (auto SMOC = mlir::dyn_cast<ir::SetMeshOutputCountsOp>(Op)) {
    auto Prims = exprFor(SMOC.getNumPrimitives(), PrecUnknown);
    if (!Prims)
      return Prims.takeError();
    line(Ind, "crucible_mesh.set_primitive_count(" + Prims->Text + ");");
    return Error::success();
  }
  if (auto EI = mlir::dyn_cast<ir::EmitIndicesOp>(Op)) {
    auto P = exprFor(EI.getPrimitiveIndex(), PrecUnknown);
    auto Idxs = EI.getVertexIndices();
    auto V0 = exprFor(Idxs[0], PrecUnknown);
    auto V1 = exprFor(Idxs[1], PrecUnknown);
    if (!P || !V0 || !V1)
      return makeError("invalid emitIndices operand");
    if (R.Mesh && R.Mesh->Topology == MeshOutputTopology::Line) {
      line(Ind, "crucible_mesh.set_index(" + P->Text + " * 2u + 0u, " +
                    V0->Text + ");");
      line(Ind, "crucible_mesh.set_index(" + P->Text + " * 2u + 1u, " +
                    V1->Text + ");");
    } else {
      auto V2 = exprFor(Idxs[2], PrecUnknown);
      if (!V2)
        return V2.takeError();
      line(Ind, "crucible_mesh.set_index(" + P->Text + " * 3u + 0u, " +
                    V0->Text + ");");
      line(Ind, "crucible_mesh.set_index(" + P->Text + " * 3u + 1u, " +
                    V1->Text + ");");
      line(Ind, "crucible_mesh.set_index(" + P->Text + " * 3u + 2u, " +
                    V2->Text + ");");
    }
    return Error::success();
  }
  if (auto SVO = mlir::dyn_cast<ir::StoreVertexOutputOp>(Op)) {
    uint32_t ID = static_cast<uint32_t>(SVO.getOutputId());
    auto Row = constInt(SVO.getRow());
    uint32_t Col = static_cast<uint32_t>(static_cast<uint8_t>(SVO.getCol()));
    auto VIdx = exprFor(SVO.getVertexIndex(), PrecUnknown);
    auto Val = exprFor(SVO.getValue(), PrecUnknown);
    if (!Row || !VIdx || !Val)
      return makeError("invalid storeVertexOutput operand");
    auto P = llvm::find_if(R.Outputs, [&](const StageParameter &S) {
      return S.ID == ID;
    });
    if (P == R.Outputs.end() || *Row >= P->Rows || Col >= P->Columns)
      return makeError("unsupported mesh vertex output signature");
    std::string Target = "crucible_verts[" + VIdx->Text + "]." +
                         signatureField(*P, *Row);
    if (P->Columns > 1)
      Target += "[" + std::to_string(Col) + "]";
    line(Ind, Target + " = " + Val->Text + ";");
    return Error::success();
  }
  if (auto DM = mlir::dyn_cast<ir::DispatchMeshOp>(Op)) {
    auto X = exprFor(DM.getTgX(), PrecUnknown);
    auto Y = exprFor(DM.getTgY(), PrecUnknown);
    auto Z = exprFor(DM.getTgZ(), PrecUnknown);
    if (!X || !Y || !Z)
      return makeError("invalid dispatchMesh grid operand");
    if (!MeshPayloadMSLType.empty()) {
      auto LV = lvalueFor(DM.getPayload(), /*DecayArray=*/false);
      if (!LV)
        return LV.takeError();
      line(Ind, "crucible_payload = " + LV->Text + ";");
    }
    line(Ind, "crucible_mgp.set_threadgroups_per_grid(uint3(" + X->Text +
                  ", " + Y->Text + ", " + Z->Text + "));");
    return Error::success();
  }
  if (auto RQA = mlir::dyn_cast<ir::RayQueryAllocOp>(Op)) {
    UsesRaytracing = true;
    auto Flags = exprFor(RQA.getConstRayFlags(), PrecUnknown);
    if (!Flags)
      return Flags.takeError();
    std::string Name = "rq" + std::to_string(NextTemp++);
    line(Ind, "intersection_query<triangle_data, instancing> " + Name + ";");
    line(Ind, "uint " + Name + "_flags = " + Flags->Text + ";");
    ValueNames[RQA.getResult()] = Name;
    return Error::success();
  }
  if (auto RQTI = mlir::dyn_cast<ir::RayQueryTraceInlineOp>(Op)) {
    UsesRaytracing = true;
    requireHelper("crucible_ray_params");
    auto Q = exprFor(RQTI.getQuery(), PrecUnknown);
    auto AS = resourceExpr(RQTI.getAccel());
    auto Flags = exprFor(RQTI.getRayFlags(), PrecUnknown);
    auto Mask = exprFor(RQTI.getInstanceMask(), PrecUnknown);
    auto OX = exprFor(RQTI.getOriginX(), PrecUnknown);
    auto OY = exprFor(RQTI.getOriginY(), PrecUnknown);
    auto OZ = exprFor(RQTI.getOriginZ(), PrecUnknown);
    auto TMin = exprFor(RQTI.getTMin(), PrecUnknown);
    auto DX = exprFor(RQTI.getDirX(), PrecUnknown);
    auto DY = exprFor(RQTI.getDirY(), PrecUnknown);
    auto DZ = exprFor(RQTI.getDirZ(), PrecUnknown);
    auto TMax = exprFor(RQTI.getTMax(), PrecUnknown);
    if (!Q || !AS || !Flags || !Mask || !OX || !OY || !OZ || !TMin ||
        !DX || !DY || !DZ || !TMax)
      return makeError("invalid traceRayInline operand");
    line(Ind, Q->Text + "_flags = " + Q->Text + "_flags | " + Flags->Text + ";");
    line(Ind, Q->Text + ".reset(ray(float3(" + OX->Text + ", " + OY->Text +
                  ", " + OZ->Text + "), float3(" + DX->Text + ", " +
                  DY->Text + ", " + DZ->Text + "), " + TMin->Text + ", " +
                  TMax->Text + "), " + *AS + ", " + Mask->Text +
                  ", crucible_ray_params(" + Q->Text + "_flags));");
    return Error::success();
  }
  if (auto RQP = mlir::dyn_cast<ir::RayQueryProceedOp>(Op)) {
    auto Q = exprFor(RQP.getQuery(), PrecUnknown);
    if (!Q)
      return Q.takeError();
    if (RQP.getResult().use_empty()) {
      line(Ind, Q->Text + ".next();");
    } else {
      std::string Name = freshName();
      line(Ind, "bool " + Name + " = " + Q->Text + ".next();");
      ValueNames[RQP.getResult()] = Name;
    }
    return Error::success();
  }
  if (auto RQAb = mlir::dyn_cast<ir::RayQueryAbortOp>(Op)) {
    auto Q = exprFor(RQAb.getQuery(), PrecUnknown);
    if (!Q)
      return Q.takeError();
    line(Ind, Q->Text + ".abort();");
    return Error::success();
  }
  if (auto RQC = mlir::dyn_cast<ir::RayQueryCommitOp>(Op)) {
    auto Q = exprFor(RQC.getQuery(), PrecUnknown);
    if (!Q)
      return Q.takeError();
    if (RQC.getKind() == ir::RayQueryCommitKind::Triangle) {
      line(Ind, Q->Text + ".commit_triangle_intersection();");
    } else {
      auto T = exprFor(RQC.getTHit(), PrecUnknown);
      if (!T)
        return T.takeError();
      line(Ind, Q->Text + ".commit_bounding_box_intersection(" + T->Text + ");");
    }
    return Error::success();
  }
  if (auto TR = mlir::dyn_cast<ir::TraceRayOp>(Op)) {
    UsesRaytracing = true;
    requireHelper("crucible_ray_params");
    auto AS = resourceExpr(TR.getAccel());
    auto Flags = exprFor(TR.getRayFlags(), PrecUnknown);
    auto Mask = exprFor(TR.getInstanceMask(), PrecUnknown);
    auto RayContrib = exprFor(TR.getHitGroupOffset(), PrecUnknown);
    auto GeomMult = exprFor(TR.getHitGroupStride(), PrecUnknown);
    auto MissIdx = exprFor(TR.getMissIndex(), PrecUnknown);
    auto OX = exprFor(TR.getOriginX(), PrecUnknown);
    auto OY = exprFor(TR.getOriginY(), PrecUnknown);
    auto OZ = exprFor(TR.getOriginZ(), PrecUnknown);
    auto TMin = exprFor(TR.getTMin(), PrecUnknown);
    auto DX = exprFor(TR.getDirX(), PrecUnknown);
    auto DY = exprFor(TR.getDirY(), PrecUnknown);
    auto DZ = exprFor(TR.getDirZ(), PrecUnknown);
    auto TMax = exprFor(TR.getTMax(), PrecUnknown);
    auto PayloadLV = lvalueFor(TR.getPayload(), /*DecayArray=*/false);
    if (!AS || !Flags || !Mask || !RayContrib || !GeomMult || !MissIdx ||
        !OX || !OY || !OZ || !TMin || !DX || !DY || !DZ || !TMax ||
        !PayloadLV)
      return makeError("invalid TraceRay operand");
    std::string ID = std::to_string(NextTemp++);
    std::string RQ = "rq_" + ID;
    std::string RS = "rs_" + ID;
    line(Ind, "{");
    line(Ind + 1, "intersection_query<triangle_data, instancing> " + RQ + ";");
    line(Ind + 1, RQ + ".reset(ray(float3(" + OX->Text + ", " + OY->Text +
                      ", " + OZ->Text + "), float3(" + DX->Text + ", " +
                      DY->Text + ", " + DZ->Text + "), " + TMin->Text + ", " +
                      TMax->Text + "), " + *AS + ", " + Mask->Text +
                      ", crucible_ray_params(" + Flags->Text + "));");
    line(Ind + 1, RQ + ".next();");
    line(Ind + 1, "CrucibleRayState " + RS + ";");
    line(Ind + 1, RS + ".world_origin = float3(" + OX->Text + ", " + OY->Text +
                      ", " + OZ->Text + ");");
    line(Ind + 1, RS + ".world_dir = float3(" + DX->Text + ", " + DY->Text +
                      ", " + DZ->Text + ");");
    line(Ind + 1, "if (" + RQ + ".get_committed_intersection_type() != "
                      "intersection_type::none) {");
    line(Ind + 2, RS + ".t_current = " + RQ + ".get_committed_distance();");
    line(Ind + 2,
         RS + ".primitive_index = " + RQ + ".get_committed_primitive_id();");
    line(Ind + 2, RS + ".barycentrics = " + RQ +
                      ".get_committed_triangle_barycentric_coord();");
    std::vector<std::string> CHOrder = Opts.HitGroupClosestHits;
    if (CHOrder.empty()) {
      for (const RTShaderInfo &RT : R.RTShaders)
        if (RT.Kind == 10)
          CHOrder.push_back(RT.Name);
    }
    if (!CHOrder.empty()) {
      line(Ind + 2, "uint hg_" + ID + " = " + RayContrib->Text + " + " +
                        GeomMult->Text + " * " + RQ +
                        ".get_committed_geometry_id();");
      line(Ind + 2, "switch (hg_" + ID + ") {");
      for (size_t H = 0; H < CHOrder.size(); ++H) {
        auto It = RTFuncMap.find(CHOrder[H]);
        if (It == RTFuncMap.end())
          continue;
        const EmittedRTFunc &FInfo = It->second;
        line(Ind + 3, "case " + std::to_string(H) + "u: {");
        if (FInfo.AttrTy) {
          auto ATy = mslType(FInfo.AttrTy);
          if (!ATy)
            return ATy.takeError();
          std::string AttrVar = "attr_" + ID + "_" + std::to_string(H);
          line(Ind + 4, *ATy + " " + AttrVar + ";");
          if (auto ST = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(FInfo.AttrTy)) {
            if (ST.getBody().size() == 1 &&
                mlir::isa<mlir::VectorType>(ST.getBody()[0])) {
              line(Ind + 4, AttrVar + ".f0 = " + RS + ".barycentrics;");
            } else if (ST.getBody().size() >= 2) {
              line(Ind + 4, AttrVar + ".f0 = " + RS + ".barycentrics.x;");
              line(Ind + 4, AttrVar + ".f1 = " + RS + ".barycentrics.y;");
            }
          }
          line(Ind + 4, FInfo.MSLName + "(" + PayloadLV->Text + ", " +
                            AttrVar + ", " + RS + ");");
        } else {
          line(Ind + 4,
               FInfo.MSLName + "(" + PayloadLV->Text + ", " + RS + ");");
        }
        line(Ind + 4, "break;");
        line(Ind + 3, "}");
      }
      line(Ind + 2, "}");
    }
    line(Ind + 1, "} else {");
    line(Ind + 2, RS + ".t_current = " + TMax->Text + ";");
    line(Ind + 2, RS + ".primitive_index = 0u;");
    line(Ind + 2, RS + ".barycentrics = float2(0.0f);");
    std::vector<std::string> MissOrder = Opts.MissShaders;
    if (MissOrder.empty()) {
      for (const RTShaderInfo &RT : R.RTShaders)
        if (RT.Kind == 11)
          MissOrder.push_back(RT.Name);
    }
    if (!MissOrder.empty()) {
      line(Ind + 2, "switch (" + MissIdx->Text + ") {");
      for (size_t M = 0; M < MissOrder.size(); ++M) {
        auto It = RTFuncMap.find(MissOrder[M]);
        if (It == RTFuncMap.end())
          continue;
        line(Ind + 3, "case " + std::to_string(M) + "u: {");
        line(Ind + 4,
             It->second.MSLName + "(" + PayloadLV->Text + ", " + RS + ");");
        line(Ind + 4, "break;");
        line(Ind + 3, "}");
      }
      line(Ind + 2, "}");
    }
    line(Ind + 1, "}");
    line(Ind, "}");
    return Error::success();
  }

  if (auto IV = mlir::dyn_cast<mlir::LLVM::InsertValueOp>(Op)) {
    // Binding aggregates are consumed by resource reflection, not Metal code.
    auto ST = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(IV.getType());
    if (ST && ST.isIdentified() && ST.getName() == "dx.types.ResBind")
      return Error::success();
    if (!ST && !mlir::isa<mlir::LLVM::LLVMArrayType>(IV.getType()))
      return makeError(
          "non-aggregate insertvalue is unsupported in Metal emission");
    auto Path = aggregateMemberPath(IV.getType(), IV.getPosition());
    if (!Path)
      return Path.takeError();
    std::string Name = freshName();
    auto Decl = localDeclaration(IV.getType(), Name);
    if (!Decl)
      return Decl.takeError();
    line(Ind, *Decl + " = {};");
    if (!isUndefLike(IV.getContainer())) {
      auto Base = exprFor(IV.getContainer(), PrecUnknown);
      if (!Base)
        return Base.takeError();
      emitAggregateCopy(IV.getType(), Name, Base->Text, Ind);
    }
    auto Value = exprFor(IV.getValue(), PrecUnknown);
    if (!Value)
      return Value.takeError();
    emitAggregateCopy(IV.getValue().getType(), Name + *Path, Value->Text, Ind);
    ValueNames[IV.getResult()] = Name;
    return Error::success();
  }

  // Metal locals do not need LLVM's stack lifetime hints.
  if (mlir::isa<mlir::LLVM::LifetimeStartOp, mlir::LLVM::LifetimeEndOp>(Op))
    return Error::success();

  // Local (thread address space) arrays.
  if (auto AL = mlir::dyn_cast<mlir::LLVM::AllocaOp>(Op)) {
    mlir::Type T = AL.getElemType();
    uint64_t N = 1;
    bool IsArray = false;
    if (auto Arr = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(T)) {
      N = Arr.getNumElements();
      T = Arr.getElementType();
      IsArray = true;
    } else if (auto Vec = mlir::dyn_cast<mlir::VectorType>(T)) {
      if (Vec.isScalable() || Vec.getRank() != 1 ||
          constInt(AL.getArraySize()) != 1)
        return makeError("unsupported vector alloca shape");
      N = Vec.getNumElements();
      T = Vec.getElementType();
      IsArray = true;
    }
    if (auto Count = constInt(AL.getArraySize()); Count && *Count != 1) {
      N *= *Count;
      IsArray = true;
    }
    auto ETy = mslType(T);
    if (!ETy)
      return ETy.takeError();
    std::string Name = freshName();
    line(Ind, *ETy + " " + Name +
                  (IsArray ? "[" + std::to_string(N) + "]" : "") + ";");
    ValueNames[AL.getResult()] = Name;
    return Error::success();
  }

  if (auto Store = mlir::dyn_cast<mlir::LLVM::StoreOp>(Op)) {
    auto LV = lvalueFor(Store.getAddr(), !mlir::isa<mlir::LLVM::LLVMArrayType>(
                                             Store.getValue().getType()));
    if (!LV)
      return LV.takeError();
    auto V = exprFor(Store.getValue(), PrecUnknown);
    if (!V)
      return V.takeError();
    if (Store.getVolatile_()) {
      auto AS = addrSpaceOf(Store.getAddr());
      auto Ty = mslType(Store.getValue().getType());
      if (!AS)
        return AS.takeError();
      if (!Ty)
        return Ty.takeError();
      if (mlir::isa<mlir::VectorType>(Store.getValue().getType()))
        return makeError("volatile LLVM vector stores are unsupported");
      line(Ind, "*((volatile " + *AS + " " + *Ty + " *)&" + LV->Text +
                    ") = " + V->Text + ";");
      return Error::success();
    }
    mlir::Type PointeeTy;
    if (auto AO = Store.getAddr().getDefiningOp<mlir::LLVM::AddressOfOp>()) {
      auto Mod = Entry->getParentOfType<mlir::ModuleOp>();
      if (auto G =
              Mod.lookupSymbol<mlir::LLVM::GlobalOp>(AO.getGlobalNameAttr())) {
        PointeeTy = G.getGlobalType();
        if (G.getAddrSpace() == 3) {
          if (auto ValVec =
                  mlir::dyn_cast<mlir::VectorType>(Store.getValue().getType());
              ValVec && mlir::isa<mlir::VectorType>(PointeeTy)) {
            std::string BaseName = GlobalNames.lookup(G);
            for (int64_t I = 0; I < ValVec.getNumElements(); ++I)
              line(Ind, BaseName + "[" + std::to_string(I) + "] = (" + V->Text +
                            ")[" + std::to_string(I) + "];");
            return Error::success();
          }
        }
      }
    } else if (auto GEP = Store.getAddr().getDefiningOp<mlir::LLVM::GEPOp>()) {
      if (auto AO = GEP.getBase().getDefiningOp<mlir::LLVM::AddressOfOp>()) {
        auto Mod = Entry->getParentOfType<mlir::ModuleOp>();
        if (auto G = Mod.lookupSymbol<mlir::LLVM::GlobalOp>(
                AO.getGlobalNameAttr());
            G && G.getAddrSpace() == 3) {
          if (auto Arr =
                  mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(G.getGlobalType());
              Arr && mlir::isa<mlir::VectorType>(Arr.getElementType())) {
            if (auto ValVec = mlir::dyn_cast<mlir::VectorType>(
                    Store.getValue().getType())) {
              for (int64_t I = 0; I < ValVec.getNumElements(); ++I)
                line(Ind, LV->Text + "[" + std::to_string(I) + "] = (" +
                              V->Text + ")[" + std::to_string(I) + "];");
              return Error::success();
            }
          }
        }
      }
    } else if (auto AL = Store.getAddr().getDefiningOp<mlir::LLVM::AllocaOp>()) {
      PointeeTy = AL.getElemType();
      if (auto ValVec =
              mlir::dyn_cast<mlir::VectorType>(Store.getValue().getType());
          ValVec && mlir::isa<mlir::VectorType>(PointeeTy)) {
        std::string BaseName = ValueNames.lookup(AL.getResult());
        for (int64_t I = 0; I < ValVec.getNumElements(); ++I)
          line(Ind, BaseName + "[" + std::to_string(I) + "] = (" + V->Text +
                        ")[" + std::to_string(I) + "];");
        return Error::success();
      }
    }
    if (PointeeTy &&
        mlir::isa<mlir::LLVM::LLVMStructType, mlir::LLVM::LLVMArrayType>(
            PointeeTy) &&
        !mlir::isa<mlir::LLVM::LLVMStructType, mlir::LLVM::LLVMArrayType>(
            Store.getValue().getType())) {
      auto AS = addrSpaceOf(Store.getAddr());
      auto VT = mslType(Store.getValue().getType());
      if (!AS || !VT)
        return makeError("invalid store to aggregate");
      line(Ind, "*(" + *AS + " " + *VT + " *)&" + LV->Text + " = " + V->Text +
                    ";");
      return Error::success();
    }
    emitAggregateCopy(Store.getValue().getType(), LV->Text, V->Text, Ind);
    return Error::success();
  }

  if (auto RMW = mlir::dyn_cast<mlir::LLVM::AtomicRMWOp>(Op)) {
    bool Is64 = RMW.getVal().getType().isInteger(64);
    if (!RMW.getVal().getType().isInteger(32) && !Is64)
      return makeError("atomicrmw requires a 32-bit or 64-bit integer");
    auto LV = lvalueFor(RMW.getPtr());
    if (!LV)
      return LV.takeError();
    auto AS = addrSpaceOf(RMW.getPtr());
    if (!AS)
      return AS.takeError();
    auto V = exprFor(RMW.getVal(), PrecUnknown);
    if (!V)
      return V.takeError();
    const char *Fn = nullptr;
    bool Signed = false;
    switch (RMW.getBinOp()) {
    case mlir::LLVM::AtomicBinOp::add:
      Fn = "atomic_fetch_add_explicit";
      break;
    case mlir::LLVM::AtomicBinOp::sub:
      Fn = "atomic_fetch_sub_explicit";
      break;
    case mlir::LLVM::AtomicBinOp::_and:
      Fn = "atomic_fetch_and_explicit";
      break;
    case mlir::LLVM::AtomicBinOp::_or:
      Fn = "atomic_fetch_or_explicit";
      break;
    case mlir::LLVM::AtomicBinOp::_xor:
      Fn = "atomic_fetch_xor_explicit";
      break;
    case mlir::LLVM::AtomicBinOp::min:
      Fn = "atomic_fetch_min_explicit";
      Signed = true;
      break;
    case mlir::LLVM::AtomicBinOp::max:
      Fn = "atomic_fetch_max_explicit";
      Signed = true;
      break;
    case mlir::LLVM::AtomicBinOp::umin:
      Fn = "atomic_fetch_min_explicit";
      break;
    case mlir::LLVM::AtomicBinOp::umax:
      Fn = "atomic_fetch_max_explicit";
      break;
    case mlir::LLVM::AtomicBinOp::xchg:
      Fn = "atomic_exchange_explicit";
      break;
    default:
      return makeError("unsupported atomicrmw kind");
    }
    std::string Text;
    if (Is64) {
      if (Signed) {
        bool IsMin = RMW.getBinOp() == mlir::LLVM::AtomicBinOp::min;
        requireHelper(*AS == "threadgroup"
                          ? (IsMin ? "crucible_atomic_imin_threadgroup_i64"
                                   : "crucible_atomic_imax_threadgroup_i64")
                          : (IsMin ? "crucible_atomic_imin_device_i64"
                                   : "crucible_atomic_imax_device_i64"));
        Text = std::string(IsMin ? "crucible_atomic_imin(("
                                 : "crucible_atomic_imax((") +
               *AS + " atomic_ulong *)&" + LV->Text + ", " + V->Text + ")";
      } else {
        Text = std::string(Fn) + "((" + *AS + " atomic_ulong *)&" + LV->Text +
               ", " + V->Text + ", memory_order_relaxed)";
      }
    } else if (Signed)
      Text = std::string("as_type<uint>(") + Fn + "((" + *AS +
             " atomic_int *)&" + LV->Text + ", as_type<int>(" + V->Text +
             "), memory_order_relaxed))";
    else
      Text = std::string(Fn) + "((" + *AS + " atomic_uint *)&" + LV->Text +
             ", " + V->Text + ", memory_order_relaxed)";
    if (!RMW.getResult().use_empty()) {
      std::string Name = freshName();
      line(Ind, std::string(Is64 ? "ulong " : "uint ") + Name + " = " + Text +
                    ";");
      ValueNames[RMW.getResult()] = Name;
    } else {
      line(Ind, Text + ";");
    }
    return Error::success();
  }

  if (auto CX = mlir::dyn_cast<mlir::LLVM::AtomicCmpXchgOp>(Op)) {
    bool Is64 = CX.getVal().getType().isInteger(64);
    if (!CX.getVal().getType().isInteger(32) && !Is64)
      return makeError("cmpxchg requires a 32-bit or 64-bit integer");
    auto LV = lvalueFor(CX.getPtr());
    if (!LV)
      return LV.takeError();
    auto AS = addrSpaceOf(CX.getPtr());
    if (!AS)
      return AS.takeError();
    if (Is64)
      requireHelper(*AS == "threadgroup"
                        ? "crucible_atomic_cmpxchg_threadgroup_i64"
                        : "crucible_atomic_cmpxchg_device_i64");
    else
      requireHelper(*AS == "threadgroup"
                        ? "crucible_atomic_cmpxchg_threadgroup"
                        : "crucible_atomic_cmpxchg_device");
    auto Cmp = exprFor(CX.getCmp(), PrecUnknown);
    if (!Cmp)
      return Cmp.takeError();
    auto V = exprFor(CX.getVal(), PrecUnknown);
    if (!V)
      return V.takeError();
    std::string Name = freshName();
    line(Ind, std::string(Is64 ? "ulong " : "uint ") + Name +
                  " = crucible_atomic_cmpxchg((" + *AS +
                  (Is64 ? " atomic_ulong *)&" : " atomic_uint *)&") +
                  LV->Text + ", " + Cmp->Text + ", " + V->Text + ");");
    ValueNames[CX.getResult()] = Name;
    return Error::success();
  }

  // Pure value: inline single-use values; declare temporaries otherwise.
  if (shouldInline(Op))
    return Error::success();
  if (Op->getNumResults() == 1 && Op->getResult(0).use_empty() &&
      !(mlir::isa<mlir::LLVM::LoadOp>(Op) &&
        mlir::cast<mlir::LLVM::LoadOp>(Op).getVolatile_()))
    return Error::success(); // dead
  return emitTemp(Op, Ind);
}

Error Emitter::emitBlock(mlir::Block &B, unsigned Ind,
                         mlir::Operation *StopBefore) {
  // Pair same-block sin(x) and cos(x) on the same operand into Metal's
  // sincos(x, c) builtin (skipped under strict FP mode).
  if (!R.RequiresStrictFP && !Opts.EmulateF64) {
    DenseMap<mlir::Value, mlir::Operation *> PendingSin;
    DenseMap<mlir::Value, mlir::Operation *> PendingCos;
    for (mlir::Operation &Op : B) {
      if (&Op == StopBefore)
        break;
      if (Op.getNumResults() != 1 || Op.getResult(0).use_empty())
        continue;
      mlir::Type Ty = Op.getResult(0).getType();
      mlir::Type ElemTy = Ty;
      if (auto VT = mlir::dyn_cast<mlir::VectorType>(Ty))
        ElemTy = VT.getElementType();
      if (!ElemTy.isF32() && !ElemTy.isF16())
        continue;
      if (mlir::isa<mlir::LLVM::SinOp, mlir::math::SinOp>(&Op)) {
        mlir::Value X = Op.getOperand(0);
        if (auto It = PendingCos.find(X); It != PendingCos.end()) {
          SinCosPartner[&Op] = It->second;
          SinCosPartner[It->second] = &Op;
          PendingCos.erase(It);
        } else {
          PendingSin.try_emplace(X, &Op);
        }
      } else if (mlir::isa<mlir::LLVM::CosOp, mlir::math::CosOp>(&Op)) {
        mlir::Value X = Op.getOperand(0);
        if (auto It = PendingSin.find(X); It != PendingSin.end()) {
          SinCosPartner[&Op] = It->second;
          SinCosPartner[It->second] = &Op;
          PendingSin.erase(It);
        } else {
          PendingCos.try_emplace(X, &Op);
        }
      }
    }
  }

  for (mlir::Operation &Op : B) {
    if (&Op == StopBefore)
      break;
    if (auto It = SinCosPartner.find(&Op); It != SinCosPartner.end()) {
      if (ValueNames.count(Op.getResult(0)))
        continue; // already emitted by partner
      mlir::Operation *Partner = It->second;
      bool ThisIsSin = mlir::isa<mlir::LLVM::SinOp, mlir::math::SinOp>(&Op);
      mlir::Operation *SinOp = ThisIsSin ? &Op : Partner;
      mlir::Operation *CosOp = ThisIsSin ? Partner : &Op;
      auto Ty = mslType(SinOp->getResult(0).getType());
      if (!Ty)
        return Ty.takeError();
      auto Arg = exprFor(SinOp->getOperand(0), PrecUnknown);
      if (!Arg)
        return Arg.takeError();
      std::string CosName = freshName();
      std::string SinName = freshName();
      line(Ind, *Ty + " " + CosName + ";");
      line(Ind, *Ty + " " + SinName + " = sincos(" + Arg->Text + ", " +
                    CosName + ");");
      ValueNames[SinOp->getResult(0)] = SinName;
      ValueNames[CosOp->getResult(0)] = CosName;
      continue;
    }
    if (Error E = emitStatement(&Op, Ind))
      return E;
  }
  return Error::success();
}

//===----------------------------------------------------------------------===//
// Kernel signature
//===----------------------------------------------------------------------===//

Expected<std::string> Emitter::kernelSignature() {
  std::string Sig;
  raw_string_ostream OS(Sig);
  bool IsComputeKernel = R.Stage == ShaderStage::Compute ||
                         R.Stage == ShaderStage::RayGeneration ||
                         R.Stage == ShaderStage::Hull ||
                         R.Stage == ShaderStage::Geometry ||
                         (Opts.StageAsCompute &&
                          (R.Stage == ShaderStage::Vertex || R.Domain));
  if (R.Stage == ShaderStage::Mesh)
    OS << "[[mesh]] void ";
  else if (R.Stage == ShaderStage::Amplification)
    OS << "[[object]] void ";
  else if (IsComputeKernel) {
    if (R.Stage == ShaderStage::Compute) {
      uint32_t TotalThreads =
          R.NumThreads[0] * R.NumThreads[1] * R.NumThreads[2];
      if (TotalThreads > 1 && TotalThreads <= 1024)
        OS << "[[max_total_threads_per_threadgroup(" << TotalThreads << ")]]\n";
    }
    OS << "kernel void ";
  }
  else {
    if (R.Domain)
      OS << "[[patch("
         << (R.Domain->Domain == TessellationDomain::Triangle ? "triangle" : "quad")
         << ", " << R.Domain->ControlPointCount << ")]]\n";
    OS << (R.Stage == ShaderStage::Pixel ? "fragment " : "vertex ")
       << "CrucibleStageOut ";
  }
  OS << R.MSLEntryName << "(\n";

  std::vector<const ResState *> Sorted;
  for (const ResState &RS : ResStates)
    Sorted.push_back(&RS);
  std::sort(Sorted.begin(), Sorted.end(),
            [](const ResState *A, const ResState *B) {
              return A->RB->MetalIndex < B->RB->MetalIndex;
            });

  bool First = true;
  auto Sep = [&]() {
    if (!First)
      OS << ",\n";
    First = false;
  };

  bool HasInput = false;
  if (R.Stage == ShaderStage::Mesh) {
    Sep();
    OS << "    crucible_mesh_t crucible_mesh";
    if (!MeshPayloadMSLType.empty()) {
      Sep();
      OS << "    const object_data " << MeshPayloadMSLType
         << "& crucible_payload [[payload]]";
    }
  } else if (R.Stage == ShaderStage::Amplification) {
    if (!MeshPayloadMSLType.empty()) {
      Sep();
      OS << "    object_data " << MeshPayloadMSLType
         << "& crucible_payload [[payload]]";
    }
    Sep();
    OS << "    mesh_grid_properties crucible_mgp";
  } else if (R.Geometry) {
    Sep();
    OS << "    uint crucible_prim_id [[thread_position_in_grid]]";
    Sep();
    OS << "    const device uint* crucible_gs_in [[buffer("
       << GeometryInterface::InputDataMetalIndex << ")]]";
    Sep();
    OS << "    device atomic_uint* crucible_gs_counter [[buffer("
       << GeometryInterface::CounterMetalIndex << ")]]";
    Sep();
    OS << "    device uint* crucible_gs_out [[buffer("
       << GeometryInterface::OutputDataMetalIndex << ")]]";
  } else if (R.Hull) {
    bool UseFloatTF =
        R.Hull->Domain == TessellationDomain::Isoline ||
        R.Hull->OutputPrimitive == TessellationOutputPrimitive::Point;
    Sep();
    OS << "    uint crucible_patch_id [[thread_position_in_grid]]";
    Sep();
    OS << "    const device uint* crucible_hs_in [[buffer("
       << HullInterface::InputDataMetalIndex << ")]]";
    Sep();
    OS << "    device " << (UseFloatTF ? "float" : "half")
       << "* crucible_tess_factors [[buffer("
       << HullInterface::TessFactorMetalIndex << ")]]";
    Sep();
    OS << "    device uint* crucible_patch_data [[buffer("
       << HullInterface::PatchDataMetalIndex << ")]]";
  } else if (R.Domain && Opts.StageAsCompute) {
    Sep();
    OS << "    uint crucible_domain_item_id [[thread_position_in_grid]]";
    Sep();
    OS << "    device uint* crucible_stage_out [[buffer(28)]]";
    Sep();
    OS << "    const device float4* crucible_domain_items [[buffer(29)]]";
    Sep();
    OS << "    const device uint* crucible_patch_data [[buffer("
       << DomainInterface::DataMetalIndex << ")]]";
  } else if (R.Domain) {
    Sep();
    OS << "    float"
       << (R.Domain->Domain == TessellationDomain::Triangle ? "3" : "2")
       << " crucible_domain_location [[position_in_patch]]";
    Sep();
    OS << "    uint crucible_patch_id [[patch_id]]";
    Sep();
    OS << "    const device uint* crucible_patch_data [[buffer("
       << DomainInterface::DataMetalIndex << ")]]";
  } else if (R.Stage == ShaderStage::Vertex && Opts.StageAsCompute) {
    Sep();
    OS << "    uint crucible_vertex_id [[thread_position_in_grid]]";
    Sep();
    OS << "    const device uchar* crucible_stage_in [[buffer(28)]]";
    Sep();
    OS << "    device uint* crucible_stage_out [[buffer(30)]]";
  }
  for (const StageParameter &P :
       (R.Domain || R.Hull || R.Geometry || R.Stage == ShaderStage::Mesh ||
        R.Stage == ShaderStage::Amplification ||
        (R.Stage == ShaderStage::Vertex && Opts.StageAsCompute))
           ? ArrayRef<StageParameter>()
           : ArrayRef(R.Inputs)) {
    if (R.Stage == ShaderStage::Vertex && P.Kind == VertexID) {
      Sep();
      OS << "    uint crucible_vertex_id [[vertex_id]]";
    } else if (R.Stage == ShaderStage::Vertex && P.Kind == InstanceID) {
      Sep();
      OS << "    uint crucible_instance_id [[instance_id]]";
    } else if (R.Stage == ShaderStage::Pixel && P.Kind == FrontFace) {
      Sep();
      OS << "    bool crucible_front_face [[front_facing]]";
    } else if (R.Stage == ShaderStage::Pixel && P.Kind == PrimitiveID) {
      Sep();
      OS << "    uint crucible_primitive_id [[primitive_id]]";
    } else if (R.Stage == ShaderStage::Pixel && P.Kind == Barycentrics) {
      Sep();
      bool NoPersp = P.Interpolation == 4 || P.Interpolation == 5 ||
                     P.Interpolation == 7;
      OS << "    float3 crucible_barycentrics [[barycentric_coord"
         << (NoPersp ? ", center_no_perspective" : "") << "]]";
    } else {
      HasInput = true;
    }
  }
  if (HasInput) {
    Sep();
    OS << "    CrucibleStageIn stageIn [[stage_in]]";
  }
  bool HasPhysicalPointers = false;
  Entry.walk([&](mlir::LLVM::IntToPtrOp Cast) {
    HasPhysicalPointers |= Cast->hasAttr("crucible.physical_type");
  });
  for (const ResState *RS : Sorted) {
    const ResourceBinding &RB = *RS->RB;
    Sep();
    if (isTextureKind(RB.Kind)) {
      const char *TexType = nullptr;
      if (RS->IsDepthSampled) {
        switch (RB.Kind) {
        case ResourceKind::Texture1D:
        case ResourceKind::Texture2D:
          TexType = "depth2d";
          break;
        case ResourceKind::Texture1DArray:
        case ResourceKind::Texture2DArray:
          TexType = "depth2d_array";
          break;
        case ResourceKind::TextureCube:
          TexType = "depthcube";
          break;
        case ResourceKind::TextureCubeArray:
          TexType = "depthcube_array";
          break;
        default:
          return makeError("unsupported depth texture kind for '{0}'",
                           RS->ParamName);
        }
        std::string Elem = std::string(TexType) + "<float, access::sample>";
        if (RB.RangeSize > 1)
          OS << "    array<" << Elem << ", " << RB.RangeSize << "> "
             << RS->ParamName << " [[texture(" << RB.MetalIndex << ")]]";
        else
          OS << "    " << Elem << " " << RS->ParamName << " [[texture("
             << RB.MetalIndex << ")]]";
        continue;
      }
      switch (RB.Kind) {
      case ResourceKind::Texture1D:
        TexType = "texture2d";
        break;
      case ResourceKind::Texture1DArray:
        TexType = "texture2d_array";
        break;
      case ResourceKind::Texture2D:
        TexType = "texture2d";
        break;
      case ResourceKind::Texture2DArray:
        TexType = "texture2d_array";
        break;
      case ResourceKind::Texture2DMS:
        TexType = "texture2d_ms";
        break;
      case ResourceKind::Texture2DMSArray:
        TexType = "texture2d_ms_array";
        break;
      case ResourceKind::Texture3D:
        TexType = "texture3d";
        break;
      case ResourceKind::TextureCube:
        TexType = "texturecube";
        break;
      case ResourceKind::TextureCubeArray:
        TexType = "texturecube_array";
        break;
      default:
        return makeError("unsupported texture kind for '{0}'", RS->ParamName);
      }
      const char *Access = RB.Class == ResourceClass::UAV ? "access::read_write"
                           : RS->IsSampled                ? "access::sample"
                                                          : "access::read";
      std::string Elem =
          std::string(TexType) + "<" + RS->ElemType + ", " + Access + ">";
      if (RB.RangeSize > 1)
        OS << "    array<" << Elem << ", " << RB.RangeSize << "> "
           << RS->ParamName << " [[texture(" << RB.MetalIndex << ")]]";
      else
        OS << "    " << Elem << " " << RS->ParamName << " [[texture("
           << RB.MetalIndex << ")]]";
      continue;
    }
    if (RB.Class == ResourceClass::Sampler) {
      if (RB.RangeSize > 1)
        OS << "    array<sampler, " << RB.RangeSize << "> " << RS->ParamName
           << " [[sampler(" << RB.MetalIndex << ")]]";
      else
        OS << "    sampler " << RS->ParamName << " [[sampler(" << RB.MetalIndex
           << ")]]";
      continue;
    }
    if (RB.Kind == ResourceKind::RTAccelerationStructure) {
      UsesRaytracing = true;
      if (RB.RangeSize == 1) {
        OS << "    instance_acceleration_structure " << RS->ParamName
           << " [[buffer(" << RB.MetalIndex << ")]]";
      } else {
        for (uint32_t I = 0; I < RB.RangeSize; ++I) {
          if (I)
            OS << ",\n";
          OS << "    instance_acceleration_structure " << RS->ParamName << "_"
             << I << " [[buffer(" << (RB.MetalIndex + I) << ")]]";
        }
      }
      continue;
    }
    const char *PtrPrefix = RB.Class == ResourceClass::SRV   ? "device const "
                            : RB.Class == ResourceClass::UAV ? "device "
                                                             : "constant ";
    const char *RestrictQual = (RB.Class != ResourceClass::CBV &&
                                !RB.GloballyCoherent && !HasPhysicalPointers)
                                   ? "__restrict "
                                   : "";
    if (RB.RangeSize == 1) {
      OS << "    " << PtrPrefix << RS->ElemType << "* " << RestrictQual
         << RS->ParamName << " [[buffer(" << RB.MetalIndex << ")]]";
    } else {
      for (uint32_t I = 0; I < RB.RangeSize; ++I) {
        if (I)
          OS << ",\n";
        OS << "    " << PtrPrefix << RS->ElemType << "* " << RestrictQual
           << RS->ParamName << "_" << I << " [[buffer(" << (RB.MetalIndex + I)
           << ")]]";
      }
    }
    if (RB.HasCounter) {
      if (RB.RangeSize == 1) {
        OS << ",\n    device atomic_uint* " << RS->ParamName
           << "_counter [[buffer(" << RB.CounterMetalIndex << ")]]";
      } else {
        for (uint32_t I = 0; I < RB.RangeSize; ++I) {
          OS << ",\n    device atomic_uint* " << RS->ParamName
             << "_counter_" << I << " [[buffer(" << (RB.CounterMetalIndex + I)
             << ")]]";
        }
      }
    }
  }
  for (const KernelPtrArg &KA : KernelPtrArgs) {
    Sep();
    OS << "    device " << KA.ElemType << "* " << KA.Name << " [[buffer("
       << KA.BufferIndex << ")]]";
  }
  if (R.BufferSizesMetalIndex != UINT32_MAX) {
    Sep();
    OS << "    constant uint* crucible_buf_dims [[buffer("
       << R.BufferSizesMetalIndex << ")]]";
  }
  if (R.InstanceContributionMetalIndex != UINT32_MAX) {
    Sep();
    OS << "    constant uint* crucible_inst_contrib [[buffer("
       << R.InstanceContributionMetalIndex << ")]]";
  }
  if (R.ResourceHeapMetalIndex != UINT32_MAX) {
    Sep();
    OS << "    device const ulong* crucible_res_heap [[buffer("
       << R.ResourceHeapMetalIndex << ")]]";
  }
  if (R.CounterHeapMetalIndex != UINT32_MAX) {
    Sep();
    OS << "    device const ulong* crucible_counter_heap [[buffer("
       << R.CounterHeapMetalIndex << ")]]";
  }
  if (R.SamplerHeapMetalIndex != UINT32_MAX) {
    Sep();
    OS << "    device const ulong* crucible_sampler_heap [[buffer("
       << R.SamplerHeapMetalIndex << ")]]";
  }

  if (UsesDTid) {
    Sep();
    OS << "    uint3 dtid [[thread_position_in_grid]]";
  }
  if (UsesGTid) {
    Sep();
    OS << "    uint3 gtid [[thread_position_in_threadgroup]]";
  }
  if (UsesGid) {
    Sep();
    OS << "    uint3 gid [[threadgroup_position_in_grid]]";
  }
  if (UsesGidx) {
    Sep();
    OS << "    uint gidx [[thread_index_in_threadgroup]]";
  }
  if (UsesLane) {
    Sep();
    OS << "    uint lane [[thread_index_in_simdgroup]]";
  }
  if (UsesLaneCount) {
    Sep();
    OS << "    uint lanecount [[threads_per_simdgroup]]";
  }
  if (UsesGridSize) {
    Sep();
    OS << "    uint3 grid_size [[threads_per_grid]]";
  }
  OS << ")";
  return Sig;
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

Expected<std::string> Emitter::run() {
  bool MeshStage = R.Stage == ShaderStage::Mesh;
  bool AmpStage = R.Stage == ShaderStage::Amplification;
  bool Graphics = R.Stage == ShaderStage::Vertex || R.Stage == ShaderStage::Pixel ||
                  R.Domain.has_value() || R.Geometry.has_value();
  if (!Graphics && !MeshStage && !AmpStage && R.Stage != ShaderStage::Compute &&
      R.Stage != ShaderStage::RayGeneration && R.Stage != ShaderStage::Hull)
    return makeError("unsupported shader stage (expected compute, vertex, pixel, hull, geometry, domain, mesh, amplification or raygeneration)");
  std::string Input, Output;
  if (Graphics || MeshStage) {
    if (Graphics && !R.Domain && !R.Geometry) {
      auto In = stageStruct(true);
      if (!In)
        return In.takeError();
      Input = std::move(*In);
    }
    auto Out = stageStruct(false);
    if (!Out)
      return Out.takeError();
    if (Out->empty())
      return makeError("graphics shaders require a supported stage output");
    Output = std::move(*Out);
    if (MeshStage) {
      if (!R.Mesh)
        return makeError("missing mesh shader interface metadata");
      Output += "using crucible_mesh_t = metal::mesh<CrucibleStageOut, void, " +
                std::to_string(R.Mesh->MaxVertexCount) + ", " +
                std::to_string(R.Mesh->MaxPrimitiveCount) + ", " +
                (R.Mesh->Topology == MeshOutputTopology::Line
                     ? "metal::topology::line"
                     : "metal::topology::triangle") +
                ">;\n\n";
      UsesGidx = true;
    }
  }
  if (!Opts.MSLEntryPoint.empty() &&
      sanitizeIdentifier(Opts.MSLEntryPoint, "kernel_main") != Opts.MSLEntryPoint)
    return makeError("invalid Metal entry point name '{0}'", Opts.MSLEntryPoint);
  R.MSLEntryName = Opts.MSLEntryPoint.empty()
                       ? sanitizeIdentifier(R.EntryName, "kernel_main")
                       : Opts.MSLEntryPoint;

  if (Error E = resolveHandles())
    return std::move(E);
  if (Error E = inferElementTypes())
    return std::move(E);
  if (Error E = collectGroupshared())
    return std::move(E);
  if (Error E = collectConstantGlobals())
    return std::move(E);

  // Layer 0 addrspace(1) pointer arguments (PTX / compute IR kernel ABI).
  if (!Entry.getBody().empty()) {
    uint32_t NextBuf = 0;
    for (const ResourceBinding &RB : R.Resources) {
      if (RB.Class != ResourceClass::Sampler && !isTextureKind(RB.Kind))
        NextBuf = std::max(NextBuf, RB.MetalIndex + RB.RangeSize);
    }
    for (mlir::BlockArgument Arg : Entry.getBody().front().getArguments()) {
      auto PT = mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(Arg.getType());
      if (!PT || PT.getAddressSpace() != 1)
        continue;
      mlir::Type ElemTy;
      for (mlir::Operation *U : Arg.getUsers()) {
        if (auto GEP = mlir::dyn_cast<mlir::LLVM::GEPOp>(U)) {
          ElemTy = GEP.getElemType();
          break;
        }
        if (auto Ld = mlir::dyn_cast<mlir::LLVM::LoadOp>(U)) {
          ElemTy = Ld.getType();
          break;
        }
        if (auto St = mlir::dyn_cast<mlir::LLVM::StoreOp>(U)) {
          ElemTy = St.getValue().getType();
          break;
        }
      }
      std::string ElemStr = "uint";
      if (ElemTy) {
        auto MT = mslType(ElemTy);
        if (!MT)
          return MT.takeError();
        ElemStr = *MT;
      }
      std::string Name = "arg" + std::to_string(KernelPtrArgs.size());
      ValueNames[Arg] = Name;
      KernelPtrArgs.push_back(KernelPtrArg{Name, ElemStr, NextBuf++});
    }
  }

  if (MeshStage || AmpStage) {
    mlir::Type PayloadTy;
    Entry.walk([&](mlir::Operation *Op) {
      if (auto DM = mlir::dyn_cast<ir::DispatchMeshOp>(Op); AmpStage && DM) {
        mlir::Value Ptr = DM.getPayload();
        if (auto AO = Ptr.getDefiningOp<mlir::LLVM::AddressOfOp>()) {
          auto Mod = Entry->getParentOfType<mlir::ModuleOp>();
          if (auto G = Mod.lookupSymbol<mlir::LLVM::GlobalOp>(AO.getGlobalNameAttr()))
            PayloadTy = G.getGlobalType();
        } else if (auto AL = Ptr.getDefiningOp<mlir::LLVM::AllocaOp>()) {
          PayloadTy = AL.getElemType();
        }
      } else if (auto GMP = mlir::dyn_cast<ir::GetMeshPayloadOp>(Op);
                 MeshStage && GMP) {
        for (mlir::Operation *U : GMP.getResult().getUsers()) {
          if (auto GEP = mlir::dyn_cast<mlir::LLVM::GEPOp>(U)) {
            PayloadTy = GEP.getElemType();
            break;
          }
          if (auto Ld = mlir::dyn_cast<mlir::LLVM::LoadOp>(U)) {
            PayloadTy = Ld.getType();
            break;
          }
        }
      }
    });
    if (PayloadTy) {
      auto MT = mslType(PayloadTy);
      if (!MT)
        return MT.takeError();
      MeshPayloadMSLType = *MT;
    }
  }

  if (R.Stage == ShaderStage::RayGeneration) {
    StructDecls += "struct CrucibleRayState {\n"
                   "  float3 world_origin;\n"
                   "  float3 world_dir;\n"
                   "  float t_current;\n"
                   "  uint primitive_index;\n"
                   "  float2 barycentrics;\n"
                   "};\n\n";
    auto Mod = Entry->getParentOfType<mlir::ModuleOp>();
    for (const RTShaderInfo &RT : R.RTShaders) {
      auto RTFn = Mod.lookupSymbol<mlir::func::FuncOp>(RT.SymbolName);
      if (!RTFn)
        continue;
      auto FindPtrElemTy = [](mlir::Value Arg) -> mlir::Type {
        for (mlir::Operation *U : Arg.getUsers()) {
          if (auto GEP = mlir::dyn_cast<mlir::LLVM::GEPOp>(U))
            return GEP.getElemType();
        }
        return nullptr;
      };
      if (RTFn.getNumArguments() < 1)
        continue;
      mlir::Type PayloadTy = FindPtrElemTy(RTFn.getArgument(0));
      if (!PayloadTy)
        continue;
      auto PayloadMTy = mslType(PayloadTy);
      if (!PayloadMTy)
        return PayloadMTy.takeError();
      mlir::Type AttrTy;
      std::string AttrMTy;
      if (RTFn.getNumArguments() >= 2) {
        AttrTy = FindPtrElemTy(RTFn.getArgument(1));
        if (AttrTy) {
          auto AM = mslType(AttrTy);
          if (!AM)
            return AM.takeError();
          AttrMTy = *AM;
        }
      }
      std::string SavedBody = std::move(Body);
      auto SavedNames = std::move(ValueNames);
      Body.clear();
      ValueNames.clear();
      ValueNames[RTFn.getArgument(0)] = "rt_arg0";
      if (RTFn.getNumArguments() >= 2)
        ValueNames[RTFn.getArgument(1)] = "rt_arg1";
      if (Error E = emitBlock(RTFn.getBody().front(), 1))
        return std::move(E);
      std::string MSLName =
          "crucible_rt_" + sanitizeIdentifier(RT.Name, "rt_fn");
      RTFuncsDecl += "static inline void " + MSLName + "(thread " +
                     *PayloadMTy + "& rt_arg0";
      if (AttrTy)
        RTFuncsDecl += ", thread " + AttrMTy + "& rt_arg1";
      RTFuncsDecl +=
          ", thread const CrucibleRayState& crucible_ray_state) {\n" + Body +
          "}\n\n";
      RTFuncMap[RT.Name] = {MSLName, PayloadTy, AttrTy, RT.Kind};
      Body = std::move(SavedBody);
      ValueNames = std::move(SavedNames);
    }
  }

  if (R.Hull) {
    line(1, "for (uint crucible_cp_id = 0u; crucible_cp_id < " +
                std::to_string(R.Hull->OutputControlPointCount) +
                "u; ++crucible_cp_id) {");
    if (Error E = emitBlock(Entry.getBody().front(), 2))
      return std::move(E);
    line(1, "}");
    auto Mod = Entry->getParentOfType<mlir::ModuleOp>();
    auto PCF = Mod.lookupSymbol<mlir::func::FuncOp>(R.Hull->PatchConstantFuncName);
    if (!PCF)
      return makeError("missing hull patch-constant function '{0}'",
                       R.Hull->PatchConstantFuncName);
    ValueNames.clear();
    if (Error E = emitBlock(PCF.getBody().front(), 1))
      return std::move(E);
  } else {
    if (Error E = emitBlock(Entry.getBody().front(), 1))
      return std::move(E);
  }

  auto Sig = kernelSignature();
  if (!Sig)
    return Sig.takeError();

  std::string Out;
  raw_string_ostream OS(Out);
  OS << "// Generated by crucible from DXIL (" << R.EntryName << ", "
     << (R.Stage == ShaderStage::Vertex ? "vs_" :
         R.Stage == ShaderStage::Pixel ? "ps_" :
         R.Stage == ShaderStage::Hull ? "hs_" :
         R.Stage == ShaderStage::Geometry ? "gs_" :
         R.Stage == ShaderStage::Domain ? "ds_" :
         R.Stage == ShaderStage::Mesh ? "ms_" :
         R.Stage == ShaderStage::Amplification ? "as_" : "cs_")
     << R.ShaderModelMajor << "_" << R.ShaderModelMinor << ")\n"
     << "#include <metal_stdlib>\n"
     << "using namespace metal;\n"
     << (UsesRaytracing ? "using namespace metal::raytracing;\n\n" : "\n");
  if (R.RequiresStrictFP)
    OS << "#pragma METAL fp math_mode(safe)\n"
       << "#pragma METAL fp contract(off)\n\n";
  for (const std::string &H : HelperOrder)
    OS << helperSource(H) << "\n";
  if (!StructDecls.empty())
    OS << StructDecls << "\n";
  if (!RTFuncsDecl.empty())
    OS << RTFuncsDecl;
  std::string ArrayDecls;
  for (const ResState &RS : ResStates) {
    const ResourceBinding &RB = *RS.RB;
    if (RB.RangeSize <= 1 || isTextureKind(RB.Kind) ||
        RB.Class == ResourceClass::Sampler)
      continue;
    if (RB.Kind == ResourceKind::RTAccelerationStructure) {
      ArrayDecls += "  instance_acceleration_structure " + RS.ParamName + "[" +
                    std::to_string(RB.RangeSize) + "] = {";
      for (uint32_t I = 0; I < RB.RangeSize; ++I) {
        if (I)
          ArrayDecls += ", ";
        ArrayDecls += RS.ParamName + "_" + std::to_string(I);
      }
      ArrayDecls += "};\n";
      continue;
    }
    const char *PtrPrefix = RB.Class == ResourceClass::SRV   ? "device const "
                            : RB.Class == ResourceClass::UAV ? "device "
                                                             : "constant ";
    ArrayDecls += "  " + std::string(PtrPrefix) + RS.ElemType + "* " +
                  RS.ParamName + "[" + std::to_string(RB.RangeSize) + "] = {";
    for (uint32_t I = 0; I < RB.RangeSize; ++I) {
      if (I)
        ArrayDecls += ", ";
      ArrayDecls += RS.ParamName + "_" + std::to_string(I);
    }
    ArrayDecls += "};\n";
    if (RB.HasCounter) {
      ArrayDecls += "  device atomic_uint* " + RS.ParamName + "_counter[" +
                    std::to_string(RB.RangeSize) + "] = {";
      for (uint32_t I = 0; I < RB.RangeSize; ++I) {
        if (I)
          ArrayDecls += ", ";
        ArrayDecls += RS.ParamName + "_counter_" + std::to_string(I);
      }
      ArrayDecls += "};\n";
    }
  }

  OS << ConstantGlobalDecls << Input << Output << *Sig << " {\n"
     << ArrayDecls << GroupsharedDecls;
  if (Graphics && Opts.StageAsCompute) {
    if (R.Stage == ShaderStage::Vertex && !Input.empty()) {
      OS << "  CrucibleStageIn stageIn = {};\n";
      for (const StageParameter &P : R.Inputs) {
        if (P.Kind == VertexID) {
          continue;
        }
        std::string Indexed = P.Semantic + std::to_string(P.SemanticIndex);
        auto Attr = llvm::find_if(Opts.VertexInputLayout, [&](const auto &A) {
          return StringRef(A.Name).equals_insensitive(Indexed) ||
                 (P.SemanticIndex == 0 &&
                  StringRef(A.Name).equals_insensitive(P.Semantic));
        });
        if (Attr == Opts.VertexInputLayout.end())
          continue;
        const char *ElemTy =
            Attr->Type == Options::VertexInputAttribute::ElementType::Int32
                ? "int"
            : Attr->Type == Options::VertexInputAttribute::ElementType::UInt32
                ? "uint"
                : "float";
        std::string Ptr = "((const device " + std::string(ElemTy) +
                          "*)(crucible_stage_in + crucible_vertex_id * " +
                          std::to_string(Opts.VertexInputStride) + "u + " +
                          std::to_string(Attr->Offset) + "u))";
        for (uint32_t RIdx = 0; RIdx < P.Rows; ++RIdx) {
          std::string Field = "stageIn." + signatureField(P, RIdx);
          for (uint32_t CIdx = 0; CIdx < P.Columns; ++CIdx) {
            uint32_t Linear = RIdx * P.Columns + CIdx;
            std::string Target = P.Columns > 1
                                     ? Field + "[" + std::to_string(CIdx) + "]"
                                     : Field;
            if (Linear < Attr->Channels)
              OS << "  " << Target << " = " << Ptr << "[" << Linear << "];\n";
            else
              OS << "  " << Target << " = " << (CIdx == 3 ? "1" : "0") << ";\n";
          }
        }
      }
    } else if (R.Domain) {
      OS << "  float"
         << (R.Domain->Domain == TessellationDomain::Triangle ? "3" : "2")
         << " crucible_domain_location = crucible_domain_items[crucible_domain_item_id]."
         << (R.Domain->Domain == TessellationDomain::Triangle ? "xyz" : "xy")
         << ";\n"
         << "  uint crucible_patch_id = as_type<uint>(crucible_domain_items[crucible_domain_item_id].w);\n";
    }
    OS << "  CrucibleStageOut stageOut = {};\n";
    OS << Body;
    unsigned OutWords = signatureWordCount(R.Outputs);
    std::string OutId = R.Domain ? "crucible_domain_item_id" : "crucible_vertex_id";
    unsigned WordOff = 0;
    for (const StageParameter &P : R.Outputs) {
      for (uint32_t RIdx = 0; RIdx < P.Rows; ++RIdx) {
        std::string Field = "stageOut." + signatureField(P, RIdx);
        for (uint32_t CIdx = 0; CIdx < P.Columns; ++CIdx) {
          std::string Src = P.Columns > 1
                                ? Field + "[" + std::to_string(CIdx) + "]"
                                : Field;
          OS << "  crucible_stage_out[" << OutId << " * " << OutWords << "u + "
             << WordOff << "u] = as_type<uint>(" << Src << ");\n";
          ++WordOff;
        }
      }
    }
  } else if (Graphics && R.Geometry) {
    OS << "  CrucibleStageOut stageOut = {};\n"
       << "  CrucibleStageOut crucible_prev[2] = {};\n"
       << "  uint crucible_strip_len = 0u;\n";
    OS << Body;
  } else if (MeshStage) {
    OS << "  threadgroup CrucibleStageOut crucible_verts["
       << R.Mesh->MaxVertexCount << "];\n";
    OS << Body;
    uint32_t TGSize = R.NumThreads[0] * R.NumThreads[1] * R.NumThreads[2];
    OS << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << "  for (uint crucible_vi = gidx; crucible_vi < "
       << R.Mesh->MaxVertexCount << "u; crucible_vi += " << TGSize << "u)\n"
       << "    crucible_mesh.set_vertex(crucible_vi, crucible_verts[crucible_vi]);\n";
  } else {
    if (Graphics)
      OS << "  CrucibleStageOut stageOut = {};\n";
    OS << Body;
    if (Graphics)
      OS << "  return stageOut;\n";
  }
  OS << "}\n";
  return Out;
}

} // namespace

static void scalarizeEmulatedF64Vectors(mlir::func::FuncOp Entry) {
  auto isF64Vec = [](mlir::Type T) {
    auto VT = mlir::dyn_cast<mlir::VectorType>(T);
    return VT && !VT.isScalable() && VT.getRank() == 1 &&
           VT.getElementType().isF64();
  };
  SmallVector<mlir::Operation *, 32> Worklist;
  Entry.walk([&](mlir::Operation *Op) {
    if (Op->getNumResults() != 1 || Op->getNumRegions() != 0)
      return;
    auto ResVT = mlir::dyn_cast<mlir::VectorType>(Op->getResult(0).getType());
    if (!ResVT || ResVT.isScalable() || ResVT.getRank() != 1)
      return;
    bool TouchesF64 = isF64Vec(ResVT);
    if (!TouchesF64) {
      for (mlir::Value V : Op->getOperands())
        if (isF64Vec(V.getType()))
          TouchesF64 = true;
    }
    if (!TouchesF64)
      return;
    if (mlir::isa<mlir::LLVM::FAddOp, mlir::LLVM::FSubOp, mlir::LLVM::FMulOp,
                  mlir::LLVM::FDivOp, mlir::LLVM::FNegOp, mlir::LLVM::FCmpOp,
                  mlir::LLVM::FAbsOp, mlir::math::AbsFOp, mlir::LLVM::SqrtOp,
                  mlir::math::SqrtOp, mlir::LLVM::FFloorOp, mlir::math::FloorOp,
                  mlir::LLVM::FCeilOp, mlir::math::CeilOp, mlir::LLVM::FTruncOp,
                  mlir::math::TruncOp, mlir::LLVM::RoundEvenOp,
                  mlir::LLVM::RintOp, mlir::math::RoundEvenOp,
                  mlir::LLVM::MinNumOp, mlir::LLVM::MinimumOp,
                  mlir::LLVM::MaxNumOp, mlir::LLVM::MaximumOp,
                  mlir::LLVM::FMAOp, mlir::LLVM::FMulAddOp, mlir::math::FmaOp,
                  mlir::math::IsNaNOp, mlir::math::IsInfOp,
                  mlir::math::IsFiniteOp, mlir::LLVM::FPExtOp,
                  mlir::LLVM::FPTruncOp, mlir::LLVM::SIToFPOp,
                  mlir::LLVM::UIToFPOp, mlir::LLVM::FPToSIOp,
                  mlir::LLVM::FPToUIOp, ir::FrcOp>(Op))
      Worklist.push_back(Op);
  });

  mlir::Type I32Ty = mlir::IntegerType::get(Entry.getContext(), 32);
  for (mlir::Operation *Op : Worklist) {
    mlir::OpBuilder B(Op);
    mlir::Location Loc = Op->getLoc();
    auto ResVT = mlir::cast<mlir::VectorType>(Op->getResult(0).getType());
    int64_t N = ResVT.getNumElements();
    mlir::Type ScalarResTy = ResVT.getElementType();
    mlir::Value Acc = mlir::LLVM::UndefOp::create(B, Loc, ResVT);
    for (int64_t I = 0; I < N; ++I) {
      auto IdxC = mlir::LLVM::ConstantOp::create(
          B, Loc, I32Ty, static_cast<int32_t>(I));
      SmallVector<mlir::Value, 4> ScalarOperands;
      for (mlir::Value Operand : Op->getOperands()) {
        if (auto OVT = mlir::dyn_cast<mlir::VectorType>(Operand.getType());
            OVT && OVT.getNumElements() == N) {
          ScalarOperands.push_back(
              mlir::LLVM::ExtractElementOp::create(B, Loc, Operand, IdxC));
        } else {
          ScalarOperands.push_back(Operand);
        }
      }
      mlir::OperationState State(Loc, Op->getName());
      State.addOperands(ScalarOperands);
      State.addTypes(ScalarResTy);
      State.addAttributes(Op->getAttrs());
      mlir::Operation *ScalarOp = B.create(State);
      Acc = mlir::LLVM::InsertElementOp::create(
          B, Loc, Acc, ScalarOp->getResult(0), IdxC);
    }
    Op->getResult(0).replaceAllUsesWith(Acc);
    Op->erase();
  }
}

static void scalarizeVectorSubgroupOps(mlir::func::FuncOp Entry) {
  auto isStaticVec = [](mlir::Type T) -> mlir::VectorType {
    auto VT = mlir::dyn_cast<mlir::VectorType>(T);
    if (VT && !VT.isScalable() && VT.getRank() == 1)
      return VT;
    return {};
  };

  SmallVector<mlir::Operation *, 16> Work;
  Entry.walk([&](mlir::Operation *Op) {
    if (auto Red = mlir::dyn_cast<mlir::gpu::SubgroupReduceOp>(Op)) {
      if (isStaticVec(Red.getType()))
        Work.push_back(Op);
    } else if (auto BC = mlir::dyn_cast<mlir::gpu::SubgroupBroadcastOp>(Op)) {
      if (isStaticVec(BC.getType()))
        Work.push_back(Op);
    } else if (auto Shuf = mlir::dyn_cast<mlir::gpu::ShuffleOp>(Op)) {
      if (isStaticVec(Shuf.getShuffleResult().getType()))
        Work.push_back(Op);
    }
  });

  mlir::Type I32Ty = mlir::IntegerType::get(Entry.getContext(), 32);
  for (mlir::Operation *Op : Work) {
    mlir::OpBuilder B(Op);
    mlir::Location Loc = Op->getLoc();
    if (auto Red = mlir::dyn_cast<mlir::gpu::SubgroupReduceOp>(Op)) {
      auto VecTy = mlir::cast<mlir::VectorType>(Red.getType());
      mlir::Value Acc = mlir::LLVM::PoisonOp::create(B, Loc, VecTy);
      auto ScanAttr = Red->getAttr("crucible.scan_kind");
      for (int64_t I = 0; I < VecTy.getNumElements(); ++I) {
        auto Idx = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty,
                                                  static_cast<int32_t>(I));
        auto Elem = mlir::LLVM::ExtractElementOp::create(B, Loc, Red.getValue(),
                                                         Idx);
        auto SRed = mlir::gpu::SubgroupReduceOp::create(
            B, Loc, Elem, Red.getOp(), Red.getUniform());
        if (ScanAttr)
          SRed->setAttr("crucible.scan_kind", ScanAttr);
        Acc = mlir::LLVM::InsertElementOp::create(B, Loc, Acc, SRed.getResult(),
                                                  Idx);
      }
      Red.replaceAllUsesWith(Acc);
      Red.erase();
    } else if (auto BC = mlir::dyn_cast<mlir::gpu::SubgroupBroadcastOp>(Op)) {
      auto VecTy = mlir::cast<mlir::VectorType>(BC.getType());
      mlir::Value Acc = mlir::LLVM::PoisonOp::create(B, Loc, VecTy);
      for (int64_t I = 0; I < VecTy.getNumElements(); ++I) {
        auto Idx = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty,
                                                  static_cast<int32_t>(I));
        auto Elem =
            mlir::LLVM::ExtractElementOp::create(B, Loc, BC.getSrc(), Idx);
        mlir::Value IdxVal =
            BC->getNumOperands() > 1 ? BC->getOperand(1) : mlir::Value();
        auto SBC = mlir::gpu::SubgroupBroadcastOp::create(
            B, Loc, Elem, IdxVal, BC.getBroadcastType());
        Acc = mlir::LLVM::InsertElementOp::create(B, Loc, Acc, SBC.getResult(),
                                                  Idx);
      }
      BC.replaceAllUsesWith(Acc);
      BC.erase();
    } else if (auto Shuf = mlir::dyn_cast<mlir::gpu::ShuffleOp>(Op)) {
      auto VecTy =
          mlir::cast<mlir::VectorType>(Shuf.getShuffleResult().getType());
      mlir::Value Acc = mlir::LLVM::PoisonOp::create(B, Loc, VecTy);
      mlir::Value LastValid;
      for (int64_t I = 0; I < VecTy.getNumElements(); ++I) {
        auto Idx = mlir::LLVM::ConstantOp::create(B, Loc, I32Ty,
                                                  static_cast<int32_t>(I));
        auto Elem =
            mlir::LLVM::ExtractElementOp::create(B, Loc, Shuf.getValue(), Idx);
        auto SShuf = mlir::gpu::ShuffleOp::create(
            B, Loc, Elem, Shuf.getOffset(), Shuf.getWidth(), Shuf.getMode());
        Acc = mlir::LLVM::InsertElementOp::create(
            B, Loc, Acc, SShuf.getShuffleResult(), Idx);
        LastValid = SShuf.getValid();
      }
      Shuf.getShuffleResult().replaceAllUsesWith(Acc);
      if (LastValid)
        Shuf.getValid().replaceAllUsesWith(LastValid);
      Shuf.erase();
    }
  }
}

Expected<std::string> emitMSL(mlir::func::FuncOp Entry, Reflection &R,
                              const Options &Opts) {
  scalarizeVectorSubgroupOps(Entry);
  if (Opts.EmulateF64)
    scalarizeEmulatedF64Vectors(Entry);
  return Emitter(Entry, R, Opts).run();
}

} // namespace crucible

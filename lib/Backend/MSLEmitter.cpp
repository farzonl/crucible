//===- MSLEmitter.cpp - Structured MLIR -> Metal Shading Language -*- C++ -*-=//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Prints a structurized shader (nested scf.if / scf.while regions over
// LLVM-dialect ops, including the original @dx.op.* calls) as clean,
// idiomatic MSL. Pure single-use values are inlined into their consumer
// expression so the output reads like hand-written Metal; multi-use values
// become typed local temporaries; scf region results become mutable locals
// (PHI demotion falls out of SCF value yields).
//
//===----------------------------------------------------------------------===//

#include "MSLEmitter.h"
#include "IntrinsicMap.h"
#include "crucible/Diagnostic.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
  StringRef Ty = T.isInteger(16) ? "short" : T.isInteger(64) ? "long" : "int";
  return ("as_type<" + Ty + ">(" + Value + ")").str();
}

std::string unsignedView(mlir::Type T, StringRef Value) {
  StringRef Ty = T.isInteger(16) ? "ushort" : T.isInteger(64) ? "ulong" : "uint";
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
  if (Name == "crucible_wave_prefix_sum")
    return "template <typename T> static inline T crucible_wave_prefix_sum(T v) {\n"
           "  T sum = simd_prefix_exclusive_sum(v);\n"
           "  return simd_is_first() ? T(0) : sum;\n"
           "}\n";
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

/// dx.op opcode of a direct call, if it is one.
std::optional<DXILOp> dxOpcode(mlir::Operation *Op) {
  auto Call = mlir::dyn_cast_or_null<mlir::LLVM::CallOp>(Op);
  if (!Call)
    return std::nullopt;
  auto Callee = Call.getCallee();
  if (!Callee || !Callee->starts_with("dx.op."))
    return std::nullopt;
  mlir::Value First = Call.getArgOperands().front();
  if (auto C = mlir::dyn_cast_or_null<mlir::LLVM::ConstantOp>(
          First.getDefiningOp()))
    if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue()))
      return static_cast<DXILOp>(IA.getValue().getZExtValue());
  return std::nullopt;
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
  };

  std::vector<ResState> ResStates; // parallel to R.Resources
  DenseMap<mlir::Operation *, ResState *> HandleMap; // handle def -> resource
  DenseMap<mlir::Value, std::string> ValueNames;     // declared locals
  unsigned NextTemp = 0;
  bool UsesDTid = false, UsesGTid = false, UsesGid = false, UsesGidx = false;
  bool UsesLane = false, UsesLaneCount = false;
  // MSL helper functions required by the emitted code, in emission order.
  std::vector<std::string> HelperOrder;
  llvm::StringSet<> HelperSet;
  std::string GroupsharedDecls; // threadgroup declarations (kernel prologue)

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
  Expected<std::string> mslType(mlir::Type T);
  Expected<Expr> constant(mlir::Operation *Def);

  // --- resource discovery ---
  Error resolveHandles();
  ResState *stateFor(mlir::Value Handle) {
    auto It = HandleMap.find(Handle.getDefiningOp());
    return It == HandleMap.end() ? nullptr : It->second;
  }
  Error inferElementTypes();

  // --- expressions ---
  Expected<Expr> exprFor(mlir::Value V, int ParentPrec);
  Expected<Expr> opExpr(mlir::Operation *Op);
  Expected<Expr> dxOpExpr(mlir::LLVM::CallOp Call, DXILOp Op);
  Expected<Expr> bufferElementExpr(mlir::LLVM::ExtractValueOp EV);
  Expected<std::string> bufferIndexExpr(ResState &RS, mlir::Value Coord0,
                                        mlir::Value Coord1, unsigned Comp);
  /// Byte size of one scalar element of an MSL type name.
  static unsigned elemByteSize(StringRef MSLTy) {
    if (MSLTy == "half" || MSLTy == "ushort")
      return 2;
    if (MSLTy == "ulong")
      return 8;
    return 4;
  }
  bool shouldInline(mlir::Operation *Op);

  // --- pointers (groupshared / local arrays) ---
  Error collectGroupshared();
  /// Render a pointer-typed SSA value as an MSL lvalue (e.g. "gs[idx]").
  /// When the pointer is a bare array base and DecayArray is set, the first
  /// element is addressed (C array-to-pointer decay in reverse).
  Expected<Expr> lvalueFor(mlir::Value Ptr, bool DecayArray = true);
  /// MSL address space of a pointer value ("threadgroup", "device", "thread").
  Expected<std::string> addrSpaceOf(mlir::Value Ptr);
  DenseMap<mlir::Operation *, std::string> GlobalNames; // llvm.mlir.global
  DenseMap<mlir::Operation *, std::string> GlobalTypes; // element MSL type

  // --- statements ---
  Error emitBlock(mlir::Block &B, unsigned Ind);
  Error emitSwitch(mlir::scf::IndexSwitchOp Switch, unsigned Ind);
  Error emitStatement(mlir::Operation *Op, unsigned Ind);
  Error emitIf(mlir::scf::IfOp If, unsigned Ind);
  Error emitWhile(mlir::scf::WhileOp While, unsigned Ind);
  Error emitBufferStore(mlir::LLVM::CallOp Call, unsigned Ind);
  Error emitBarrier(mlir::LLVM::CallOp Call, unsigned Ind);
  Error emitDXAtomic(mlir::LLVM::CallOp Call, DXILOp Op, unsigned Ind);
  Error emitTemp(mlir::Operation *Op, unsigned Ind);

  /// Guarantee V is referable by a stable name (or literal) and return it.
  Expected<std::string> ensureNamed(mlir::Value V, unsigned Ind,
                                    bool CopyBlockArgs = false);

  Expected<std::string> kernelSignature();
};

//===----------------------------------------------------------------------===//
// Types and constants
//===----------------------------------------------------------------------===//

Expected<std::string> Emitter::mslType(mlir::Type T) {
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
  if (T.isF64())
    return makeError("MSL has no double type; use float in the shader");
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

  if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(ValAttr)) {
    if (IA.getType().isInteger(1))
      return Expr{IA.getValue().isZero() ? "false" : "true", PrecPrimary};
    std::string S = std::to_string(IA.getValue().getZExtValue());
    S += IA.getType().isInteger(64) ? "ul" : "u";
    if (IA.getType().isInteger(16))
      S = "ushort(" + S + ")";
    return Expr{S, PrecPrimary};
  }
  if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(ValAttr))
    return Expr{formatFloatLiteral(FA.getValue(), FA.getType().isF16()),
                PrecPrimary};
  return makeError("unsupported constant attribute");
}

//===----------------------------------------------------------------------===//
// Resource discovery
//===----------------------------------------------------------------------===//

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

  Entry.walk([&](mlir::LLVM::CallOp Call) {
    if (!FirstError.empty())
      return;
    auto Op = dxOpcode(Call);
    if (!Op)
      return;
    auto Args = Call.getArgOperands();
    switch (*Op) {
    case DXILOp::CreateHandle: {
      // (op, class:i8, rangeID:i32, index:i32, nonUniform:i1)
      auto ClassC = constInt(Args[1]);
      auto RangeC = constInt(Args[2]);
      if (!ClassC || !RangeC) {
        FirstError = "dynamic createHandle operands are not supported";
        return;
      }
      ResState *RS = FindBinding(static_cast<ResourceClass>(*ClassC),
                                 static_cast<uint32_t>(*RangeC));
      if (!RS) {
        FirstError = "createHandle references unknown resource range";
        return;
      }
      HandleMap[Call] = RS;
      break;
    }
    case DXILOp::CreateHandleFromBinding: {
      // (op, resBind:{lb,ub,space,class}, index:i32, nonUniform:i1)
      auto Index = constInt(Args[2]);
      if (!Index) {
        FirstError =
            "dynamic createHandleFromBinding operands are not supported";
        return;
      }
      uint32_t LB = 0, Space = 0, Class = 0;
      if (auto *Def = Args[1].getDefiningOp()) {
        // The importer materializes the ResBind aggregate as a chain of
        // insertvalues over undef; walk it.
        mlir::Operation *Cur = Def;
        while (auto IV = mlir::dyn_cast<mlir::LLVM::InsertValueOp>(Cur)) {
          auto Pos = IV.getPosition();
          if (auto CV = constInt(IV.getValue())) {
            if (Pos.size() == 1) {
              if (Pos[0] == 0)
                LB = static_cast<uint32_t>(*CV);
              else if (Pos[0] == 2)
                Space = static_cast<uint32_t>(*CV);
              else if (Pos[0] == 3)
                Class = static_cast<uint32_t>(*CV);
            }
          }
          Cur = IV.getContainer().getDefiningOp();
          if (!Cur)
            break;
        }
      }
      (void)LB;
      ResState *RS = FindByRegister(static_cast<ResourceClass>(Class), Space,
                                    static_cast<uint32_t>(*Index));
      if (!RS) {
        FirstError = "createHandleFromBinding references unknown binding";
        return;
      }
      HandleMap[Call] = RS;
      break;
    }
    default:
      break;
    }
  });
  if (!FirstError.empty())
    return makeError("{0}", FirstError);

  // Forward annotateHandle chains.
  bool Changed = true;
  while (Changed) {
    Changed = false;
    Entry.walk([&](mlir::LLVM::CallOp Call) {
      if (HandleMap.count(Call))
        return;
      auto Op = dxOpcode(Call);
      if (!Op || *Op != DXILOp::AnnotateHandle)
        return;
      if (ResState *RS = stateFor(Call.getArgOperands()[1])) {
        HandleMap[Call] = RS;
        Changed = true;
      }
    });
  }
  return Error::success();
}

Error Emitter::inferElementTypes() {
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

  Entry.walk([&](mlir::LLVM::CallOp Call) {
    if (!FirstError.empty())
      return;
    auto Op = dxOpcode(Call);
    if (!Op)
      return;
    auto Args = Call.getArgOperands();
    switch (*Op) {
    case DXILOp::BufferLoad:
    case DXILOp::RawBufferLoad: {
      auto Ret = mlir::cast<mlir::LLVM::LLVMStructType>(
          Call.getResult().getType());
      Record(Args[1], Ret.getBody()[0]);
      break;
    }
    case DXILOp::BufferStore:
    case DXILOp::RawBufferStore:
      Record(Args[1], Args[4].getType());
      break;
    case DXILOp::AtomicBinOp:
    case DXILOp::AtomicCompareExchange:
      // Atomics operate on 32-bit integer views.
      Record(Args[1], Args[6].getType());
      break;
    case DXILOp::CBufferLoadLegacy: {
      auto Ret = mlir::cast<mlir::LLVM::LLVMStructType>(
          Call.getResult().getType());
      mlir::Type Elem = Ret.getBody()[0];
      ResState *RS = stateFor(Args[1]);
      if (!RS) {
        FirstError = "cbuffer access through unresolved handle";
        return;
      }
      std::string Row;
      if (Elem.isF32())
        Row = "float4";
      else if (Elem.isInteger(16) || Elem.isInteger(32) ||
               Elem.isInteger(64) || Elem.isF16())
        Row = "uint4";
      else {
        FirstError = "cbuffer '" + RS->ParamName +
                     "' uses an unsupported element type";
        return;
      }
      // Mixed float/uint rows canonicalize to float4; integer components are
      // reinterpreted with as_type at each load site.
      if (RS->ElemType.empty())
        RS->ElemType = Row;
      else if (RS->ElemType != Row)
        RS->ElemType = "float4";
      break;
    }
    default:
      break;
    }
  });
  if (!FirstError.empty())
    return makeError("{0}", FirstError);

  for (ResState &RS : ResStates)
    if (RS.ElemType.empty())
      RS.ElemType = RS.RB->Class == ResourceClass::CBV ? "float4" : "uint";
  return Error::success();
}

//===----------------------------------------------------------------------===//
// Groupshared memory, local arrays, and pointer lvalues
//===----------------------------------------------------------------------===//

Error Emitter::collectGroupshared() {
  auto Mod = Entry->getParentOfType<mlir::ModuleOp>();
  unsigned Idx = 0;
  for (auto G : Mod.getOps<mlir::LLVM::GlobalOp>()) {
    if (G.getAddrSpace() != 3)
      continue;
    mlir::Type T = G.getGlobalType();
    uint64_t N = 1;
    mlir::Type Elem = T;
    bool IsArray = false;
    if (auto Arr = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(T)) {
      N = Arr.getNumElements();
      Elem = Arr.getElementType();
      IsArray = true;
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
                        (IsArray ? "[" + std::to_string(N) + "]" : "") + ";\n";
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
  if (!Def)
    return makeError("pointer block arguments are not supported");

  if (auto AO = mlir::dyn_cast<mlir::LLVM::AddressOfOp>(Def)) {
    auto Mod = Entry->getParentOfType<mlir::ModuleOp>();
    auto G = Mod.lookupSymbol<mlir::LLVM::GlobalOp>(AO.getGlobalNameAttr());
    auto It = G ? GlobalNames.find(G) : GlobalNames.end();
    if (It == GlobalNames.end())
      return makeError("reference to unsupported global '{0}'",
                       AO.getGlobalName());
    bool IsArray = mlir::isa<mlir::LLVM::LLVMArrayType>(G.getGlobalType());
    return Expr{It->second + (IsArray && DecayArray ? "[0]" : ""),
                PrecPrimary};
  }

  if (auto AL = mlir::dyn_cast<mlir::LLVM::AllocaOp>(Def)) {
    auto It = ValueNames.find(AL.getResult());
    if (It == ValueNames.end())
      return makeError("use of alloca before declaration");
    bool IsArray = mlir::isa<mlir::LLVM::LLVMArrayType>(AL.getElemType());
    return Expr{It->second + (IsArray && DecayArray ? "[0]" : ""),
                PrecPrimary};
  }

  if (auto GEP = mlir::dyn_cast<mlir::LLVM::GEPOp>(Def)) {
    auto Base = lvalueFor(GEP.getBase(), /*DecayArray=*/false);
    if (!Base)
      return Base.takeError();
    std::string Text = Base->Text;
    auto Indices = GEP.getIndices();
    for (unsigned I = 0; I < Indices.size(); ++I) {
      std::string IdxText;
      if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(Indices[I])) {
        if (I == 0) {
          if (IA.getValue().getZExtValue() != 0)
            return makeError("pointer-arithmetic GEPs are not supported");
          continue; // leading zero: address of the aggregate itself
        }
        IdxText = std::to_string(IA.getValue().getZExtValue()) + "u";
      } else {
        auto V = mlir::cast<mlir::Value>(Indices[I]);
        auto E = exprFor(V, PrecUnknown);
        if (!E)
          return E.takeError();
        if (I == 0)
          return makeError("pointer-arithmetic GEPs are not supported");
        IdxText = E->Text;
      }
      Text += "[" + IdxText + "]";
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
  if (Op->getNumRegions() > 0)
    return true; // conservative: nested control flow may contain writes
  if (auto DXOp = dxOpcode(Op)) {
    switch (*DXOp) {
    case DXILOp::BufferStore:
    case DXILOp::RawBufferStore:
    case DXILOp::Barrier:
    case DXILOp::AtomicBinOp:
    case DXILOp::AtomicCompareExchange:
      return true;
    default:
      return false;
    }
  }
  if (mlir::isa<mlir::LLVM::StoreOp, mlir::LLVM::AtomicRMWOp,
                mlir::LLVM::AtomicCmpXchgOp>(Op))
    return true;
  if (mlir::isa<mlir::LLVM::CallOp>(Op))
    return true; // unknown call
  return false;  // the pure ops we support
}

bool Emitter::shouldInline(mlir::Operation *Op) {
  if (isConstantLike(Op))
    return true;

  // Pointer producers are always folded into the lvalue they feed.
  if (mlir::isa<mlir::LLVM::GEPOp, mlir::LLVM::AddressOfOp>(Op))
    return true;

  // Side-effecting or aggregate-producing ops are materialized as statements.
  if (mlir::isa<mlir::LLVM::StoreOp, mlir::LLVM::AtomicRMWOp,
                mlir::LLVM::AtomicCmpXchgOp, mlir::LLVM::AllocaOp>(Op))
    return false;

  if (auto DXOp = dxOpcode(Op)) {
    switch (*DXOp) {
    case DXILOp::ThreadId:
    case DXILOp::GroupId:
    case DXILOp::ThreadIdInGroup:
    case DXILOp::FlattenedThreadIdInGroup:
      // Trivial builtin reads: always re-spell at each use.
      return true;
    case DXILOp::CreateHandle:
    case DXILOp::CreateHandleFromBinding:
    case DXILOp::AnnotateHandle:
    case DXILOp::BufferLoad:
    case DXILOp::RawBufferLoad:
    case DXILOp::CBufferLoadLegacy:
      // Never materialized directly; consumed via handles/extractvalue.
      return true;
    case DXILOp::AtomicBinOp:
    case DXILOp::AtomicCompareExchange:
    case DXILOp::WaveActiveBallot:
      return false; // materialized as statements
    default:
      break;
    }
  }

  if (Op->getNumResults() != 1 || !Op->getResult(0).hasOneUse())
    return false;

  mlir::Operation *User = *Op->getResult(0).getUsers().begin();

  // A memory read must not be folded forward across a write or barrier,
  // since the inlined load would observe the mutation.
  auto PinAgainstWrites = [&]() {
    if (User->getBlock() != Op->getBlock())
      return false; // conservative across region boundaries
    for (mlir::Operation *P = Op->getNextNode(); P && P != User;
         P = P->getNextNode())
      if (mayWriteMemory(P))
        return false;
    return true;
  };

  if (mlir::isa<mlir::LLVM::LoadOp>(Op))
    return PinAgainstWrites();

  if (auto EV = mlir::dyn_cast<mlir::LLVM::ExtractValueOp>(Op)) {
    mlir::Operation *Load = EV.getContainer().getDefiningOp();
    auto LoadOp = dxOpcode(Load);
    if (LoadOp &&
        (*LoadOp == DXILOp::BufferLoad || *LoadOp == DXILOp::RawBufferLoad)) {
      auto Call = mlir::cast<mlir::LLVM::CallOp>(Load);
      ResState *RS = stateFor(Call.getArgOperands()[1]);
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
    if (isConstantLike(Def)) {
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

  if (auto Info = binOpInfo(Op)) {
    mlir::Type T = Op->getResult(0).getType();
    int LPrec = Info->Prec == PrecShift ? PrecMul : Info->Prec;
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
    auto C = exprFor(Sel.getCondition(), PrecLOr);
    if (!C)
      return C.takeError();
    auto T = exprFor(Sel.getTrueValue(), PrecTernary);
    if (!T)
      return T.takeError();
    auto F = exprFor(Sel.getFalseValue(), PrecTernary);
    if (!F)
      return F.takeError();
    return Expr{C->Text + " ? " + T->Text + " : " + F->Text, PrecTernary};
  }
  if (auto Sel = mlir::dyn_cast<mlir::arith::SelectOp>(Op)) {
    auto C = exprFor(Sel.getCondition(), PrecLOr);
    if (!C)
      return C.takeError();
    auto T = exprFor(Sel.getTrueValue(), PrecTernary);
    if (!T)
      return T.takeError();
    auto F = exprFor(Sel.getFalseValue(), PrecTernary);
    if (!F)
      return F.takeError();
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
    if (!T.isIndex() && mlir::isa<mlir::arith::IndexCastOp>(Op))
      Src->Text = signedView(T, Src->Text);
    return Expr{*DstTy + "(" + Src->Text + ")", PrecPrimary};
  }
  if (mlir::isa<TruncOp, ZExtOp, FPTruncOp, FPExtOp, FPToUIOp, UIToFPOp,
                mlir::arith::ExtUIOp, mlir::arith::TruncIOp>(Op)) {
    auto Src = exprFor(Op->getOperand(0), PrecUnknown);
    if (!Src)
      return Src.takeError();
    auto DstTy = mslType(Op->getResult(0).getType());
    if (!DstTy)
      return DstTy.takeError();
    if (mlir::isa<TruncOp, mlir::arith::TruncIOp>(Op) &&
        Op->getResult(0).getType().isInteger(1))
      return Expr{"bool((" + Src->Text + ") & 1u)", PrecPrimary};
    return Expr{*DstTy + "(" + Src->Text + ")", PrecPrimary};
  }
  if (mlir::isa<SExtOp, SIToFPOp, mlir::arith::ExtSIOp>(Op)) {
    auto Src = exprFor(Op->getOperand(0), PrecUnknown);
    if (!Src)
      return Src.takeError();
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
    StringRef Ty = T.isInteger(16) ? "short" : T.isInteger(64) ? "long" : "int";
    return Expr{unsignedView(T, (Ty + "(" + Src->Text + ")").str()),
                PrecPrimary};
  }
  if (mlir::isa<BitcastOp>(Op)) {
    auto Src = exprFor(Op->getOperand(0), PrecUnknown);
    if (!Src)
      return Src.takeError();
    auto DstTy = mslType(Op->getResult(0).getType());
    if (!DstTy)
      return DstTy.takeError();
    return Expr{"as_type<" + *DstTy + ">(" + Src->Text + ")", PrecPrimary};
  }
  if (auto Freeze = mlir::dyn_cast<FreezeOp>(Op))
    return exprFor(Freeze.getOperand(), PrecUnknown);

  if (auto Load = mlir::dyn_cast<LoadOp>(Op))
    return lvalueFor(Load.getAddr());

  if (auto EV = mlir::dyn_cast<ExtractValueOp>(Op))
    return bufferElementExpr(EV);

  if (auto Call = mlir::dyn_cast<CallOp>(Op)) {
    auto DXOp = dxOpcode(Call);
    if (!DXOp)
      return makeError("call to unsupported function '{0}'",
                       Call.getCallee() ? *Call.getCallee() : "<indirect>");
    return dxOpExpr(Call, *DXOp);
  }

  return makeError("unsupported operation '{0}'",
                   Op->getName().getStringRef());
}

Expected<Expr> Emitter::dxOpExpr(mlir::LLVM::CallOp Call, DXILOp Op) {
  auto Args = Call.getArgOperands();
  auto Component = [&](const char *Base) -> Expected<Expr> {
    auto Comp = constInt(Args[1]);
    if (!Comp)
      return makeError("dynamic thread-id component is not supported");
    static const char Swizzle[] = {'x', 'y', 'z'};
    if (*Comp > 2)
      return makeError("thread-id component out of range");
    return Expr{std::string(Base) + "." + Swizzle[*Comp], PrecPrimary};
  };

  switch (Op) {
  case DXILOp::ThreadId:
    UsesDTid = true;
    return Component("dtid");
  case DXILOp::GroupId:
    UsesGid = true;
    return Component("gid");
  case DXILOp::ThreadIdInGroup:
    UsesGTid = true;
    return Component("gtid");
  case DXILOp::FlattenedThreadIdInGroup:
    UsesGidx = true;
    return Expr{"gidx", PrecPrimary};
  default:
    break;
  }

  bool IsFloat = Call.getNumResults() == 1 &&
                 mlir::isa<mlir::FloatType>(Call.getResult().getType());

  // DXIL Frc must propagate NaN for infinite inputs (metal::fract clamps),
  // so spell it as x - floor(x).
  if (Op == DXILOp::Frc) {
    auto A = exprFor(Args[1], PrecAdd);
    if (!A)
      return A.takeError();
    auto AF = exprFor(Args[1], PrecUnknown);
    if (!AF)
      return AF.takeError();
    return Expr{A->Text + " - floor(" + AF->Text + ")", PrecAdd};
  }

  if (StringRef Fn = mslUnaryFunction(Op, IsFloat); !Fn.empty()) {
    auto A = exprFor(Args[1], PrecUnknown);
    if (!A)
      return A.takeError();
    return Expr{(Fn + "(" + A->Text + ")").str(), PrecPrimary};
  }

  if (StringRef Fn = mslBinaryFunction(Op); !Fn.empty()) {
    bool Signed = Op == DXILOp::IMax || Op == DXILOp::IMin;
    auto A = exprFor(Args[1], PrecUnknown);
    if (!A)
      return A.takeError();
    auto B = exprFor(Args[2], PrecUnknown);
    if (!B)
      return B.takeError();
    if (Signed)
      return Expr{unsignedView(Args[1].getType(),
                               Fn.str() + "(" +
                                   signedView(Args[1].getType(), A->Text) +
                                   ", " +
                                   signedView(Args[2].getType(), B->Text) + ")"),
                  PrecPrimary};
    return Expr{Fn.str() + "(" + A->Text + ", " + B->Text + ")", PrecPrimary};
  }

  if (Op == DXILOp::FMad || Op == DXILOp::Fma || Op == DXILOp::IMad ||
      Op == DXILOp::UMad) {
    auto A = exprFor(Args[1], PrecUnknown);
    if (!A)
      return A.takeError();
    auto B = exprFor(Args[2], PrecUnknown);
    if (!B)
      return B.takeError();
    auto C = exprFor(Args[3], PrecUnknown);
    if (!C)
      return C.takeError();
    if (Op == DXILOp::FMad || Op == DXILOp::Fma)
      return Expr{"fma(" + A->Text + ", " + B->Text + ", " + C->Text + ")",
                  PrecPrimary};
    mlir::Type T = Call.getResult().getType();
    if (T.isInteger(16))
      return Expr{"ushort(uint(" + A->Text + ") * uint(" + B->Text +
                      ") + uint(" + C->Text + "))",
                  PrecPrimary};
    return Expr{"(" + A->Text + ") * (" + B->Text + ") + (" + C->Text + ")",
                PrecAdd};
  }

  if (Op == DXILOp::LegacyF32ToF16 || Op == DXILOp::LegacyF16ToF32) {
    auto A = exprFor(Args[1], PrecUnknown);
    if (!A)
      return A.takeError();
    return Expr{Op == DXILOp::LegacyF32ToF16
                    ? "uint(as_type<ushort>(half(" + A->Text + ")))"
                    : "float(as_type<half>(ushort(" + A->Text + ")))",
                PrecPrimary};
  }

  // Bit scans with DXIL's -1-on-zero semantics.
  if (Op == DXILOp::FirstbitLo || Op == DXILOp::FirstbitHi ||
      Op == DXILOp::FirstbitSHi) {
    const char *Helper = Op == DXILOp::FirstbitLo    ? "crucible_firstbitlo"
                         : Op == DXILOp::FirstbitHi ? "crucible_firstbithi"
                                                     : "crucible_firstbitshi";
    requireHelper(Helper);
    auto A = exprFor(Args[1], PrecUnknown);
    if (!A)
      return A.takeError();
    return Expr{std::string(Helper) + "(" + A->Text + ")", PrecPrimary};
  }

  // Bitfield ops with DXIL's operand masking semantics.
  if (Op == DXILOp::Ubfe || Op == DXILOp::Ibfe || Op == DXILOp::Bfi) {
    const char *Helper = Op == DXILOp::Ubfe   ? "crucible_ubfe"
                         : Op == DXILOp::Ibfe ? "crucible_ibfe"
                                              : "crucible_bfi";
    requireHelper(Helper);
    std::string Text = std::string(Helper) + "(";
    for (unsigned I = 1; I < Args.size(); ++I) {
      auto A = exprFor(Args[I], PrecUnknown);
      if (!A)
        return A.takeError();
      Text += (I > 1 ? ", " : "") + A->Text;
    }
    return Expr{Text + ")", PrecPrimary};
  }

  // Float vector dot products: scalarized operands -> vector constructor.
  if (Op == DXILOp::Dot2 || Op == DXILOp::Dot3 || Op == DXILOp::Dot4) {
    unsigned N = Op == DXILOp::Dot2 ? 2 : Op == DXILOp::Dot3 ? 3 : 4;
    std::string VecTy =
        (Call.getResult().getType().isF16() ? "half" : "float") +
        std::to_string(N);
    auto Operand = [&](unsigned Base) -> Expected<std::string> {
      std::string V = VecTy + "(";
      for (unsigned I = 0; I < N; ++I) {
        auto A = exprFor(Args[Base + I], PrecUnknown);
        if (!A)
          return A.takeError();
        V += (I ? ", " : "") + A->Text;
      }
      return V + ")";
    };
    auto LHS = Operand(1);
    if (!LHS)
      return LHS.takeError();
    auto RHS = Operand(1 + N);
    if (!RHS)
      return RHS.takeError();
    return Expr{"dot(" + *LHS + ", " + *RHS + ")", PrecPrimary};
  }

  // SM 6.4 packed dot products.
  if (Op == DXILOp::Dot4AddI8Packed || Op == DXILOp::Dot4AddU8Packed) {
    const char *Helper = Op == DXILOp::Dot4AddI8Packed
                             ? "crucible_dot4add_i8packed"
                             : "crucible_dot4add_u8packed";
    requireHelper(Helper);
    auto Acc = exprFor(Args[1], PrecUnknown);
    if (!Acc)
      return Acc.takeError();
    auto A = exprFor(Args[2], PrecUnknown);
    if (!A)
      return A.takeError();
    auto B = exprFor(Args[3], PrecUnknown);
    if (!B)
      return B.takeError();
    return Expr{std::string(Helper) + "(" + Acc->Text + ", " + A->Text + ", " +
                    B->Text + ")",
                PrecPrimary};
  }
  if (Op == DXILOp::Dot2AddHalf) {
    // (acc:f32, ax, ay, bx, by : f16) -> f32
    auto Acc = exprFor(Args[1], PrecUnknown);
    if (!Acc)
      return Acc.takeError();
    std::string Dots;
    for (unsigned I = 0; I < 2; ++I) {
      auto A = exprFor(Args[2 + I], PrecUnknown);
      if (!A)
        return A.takeError();
      auto B = exprFor(Args[4 + I], PrecUnknown);
      if (!B)
        return B.takeError();
      Dots += " + float(" + A->Text + ") * float(" + B->Text + ")";
    }
    return Expr{Acc->Text + Dots, PrecAdd};
  }

  // Wave operations -> Metal simdgroup functions.
  switch (Op) {
  case DXILOp::WaveIsFirstLane:
    return Expr{"simd_prefix_exclusive_sum(1u) == 0u", PrecEq};
  case DXILOp::WaveGetLaneIndex:
    UsesLane = true;
    return Expr{"lane", PrecPrimary};
  case DXILOp::WaveGetLaneCount:
    UsesLaneCount = true;
    return Expr{"lanecount", PrecPrimary};
  case DXILOp::WaveAnyTrue:
  case DXILOp::WaveAllTrue: {
    auto A = exprFor(Args[1], PrecUnknown);
    if (!A)
      return A.takeError();
    return Expr{std::string(Op == DXILOp::WaveAnyTrue ? "simd_any"
                                                      : "simd_all") +
                    "(" + A->Text + ")",
                PrecPrimary};
  }
  case DXILOp::WaveAllBitCount:
  case DXILOp::WavePrefixBitCount: {
    auto A = exprFor(Args[1], PrecUnknown);
    if (!A)
      return A.takeError();
    return Expr{Op == DXILOp::WaveAllBitCount
                    ? "simd_sum(uint(" + A->Text + "))"
                    : "simd_prefix_exclusive_sum(uint(" + A->Text + "))",
                PrecPrimary};
  }
  case DXILOp::WaveActiveAllEqual: {
    auto A = exprFor(Args[1], PrecUnknown);
    if (!A)
      return A.takeError();
    std::string First;
    if (Args[1].getType().isInteger(1))
      return Expr{"(simd_all(" + A->Text + ") || !simd_any(" + A->Text + "))",
                  PrecPrimary};
    if (Args[1].getType().isInteger(64)) {
      First = "as_type<ulong>(simd_broadcast_first(as_type<uint2>(" +
              A->Text + ")))";
    } else {
      First = "simd_broadcast_first(" + A->Text + ")";
    }
    return Expr{"simd_all((" + A->Text + ") == " + First + ")", PrecPrimary};
  }
  case DXILOp::WaveReadLaneAt:
  case DXILOp::QuadReadLaneAt:
  case DXILOp::QuadOp:
  case DXILOp::WaveReadLaneFirst: {
    auto V = exprFor(Args[1], PrecUnknown);
    if (!V)
      return V.takeError();
    StringRef Fn;
    std::string LaneArg;
    if (Op == DXILOp::WaveReadLaneFirst) {
      Fn = "simd_broadcast_first";
    } else if (Op == DXILOp::QuadOp) {
      auto Kind = constInt(Args[2]);
      if (!Kind || *Kind > 2)
        return makeError("invalid DXIL quad operation kind");
      Fn = "quad_shuffle_xor";
      LaneArg = ", " + std::to_string(*Kind + 1) + "u";
    } else {
      auto L = exprFor(Args[2], PrecUnknown);
      if (!L)
        return L.takeError();
      Fn = Op == DXILOp::WaveReadLaneAt ? "simd_shuffle" : "quad_shuffle";
      LaneArg = ", ushort(" + L->Text + ")";
    }
    mlir::Type T = Args[1].getType();
    if (T.isInteger(1))
      return Expr{"bool(" + Fn.str() + "(uint(" + V->Text + ")" + LaneArg +
                      "))",
                  PrecPrimary};
    if (T.isInteger(64))
      return Expr{"as_type<ulong>(" + Fn.str() + "(as_type<uint2>(" +
                      V->Text + ")" + LaneArg + "))",
                  PrecPrimary};
    return Expr{Fn.str() + "(" + V->Text + LaneArg + ")", PrecPrimary};
  }
  case DXILOp::WaveActiveOp:
  case DXILOp::WavePrefixOp: {
    auto Kind = constInt(Args[2]);
    auto SignedOp = constInt(Args[3]);
    if (!Kind || !SignedOp)
      return makeError("dynamic wave op kind is not supported");
    bool Prefix = Op == DXILOp::WavePrefixOp;
    StringRef Fn;
    switch (static_cast<DXILWaveOpKind>(*Kind)) {
    case DXILWaveOpKind::Sum:
      Fn = Prefix ? "simd_prefix_exclusive_sum" : "simd_sum";
      break;
    case DXILWaveOpKind::Product:
      Fn = Prefix ? "simd_prefix_exclusive_product" : "simd_product";
      break;
    case DXILWaveOpKind::Min:
      Fn = "simd_min";
      break;
    case DXILWaveOpKind::Max:
      Fn = "simd_max";
      break;
    }
    if (Prefix && (Fn == "simd_min" || Fn == "simd_max"))
      return makeError("wave prefix min/max is not supported");
    if (Fn.empty())
      return makeError("invalid DXIL wave operation kind");
    auto V = exprFor(Args[1], PrecUnknown);
    if (!V)
      return V.takeError();
    mlir::Type T = Call.getResult().getType();
    bool IsSigned = *SignedOp == 0 && mlir::isa<mlir::IntegerType>(T);
    // Signed min/max (and ordering-sensitive prefix ops) need signed views;
    // sums/products are bit-identical for signed and unsigned.
    bool SignedOrder = IsSigned && (Fn == "simd_min" || Fn == "simd_max");
    if (T.isInteger(64)) {
      requireHelper("crucible_wave_wide");
      UsesLane |= Prefix;
      std::string Text =
          std::string("crucible_wave_wide<") + (SignedOrder ? "long" : "ulong") +
          ", " + std::to_string(*Kind) + ", " + (Prefix ? "true" : "false") +
          ">(" + (SignedOrder ? signedView(T, V->Text) : V->Text) + ", " +
          (Prefix ? "lane" : "0u") + ")";
      return Expr{SignedOrder ? unsignedView(T, Text) : Text, PrecPrimary};
    }
    if (SignedOrder)
      return Expr{unsignedView(T, Fn.str() + "(" + signedView(T, V->Text) +
                                     ")"),
                  PrecPrimary};
    if (Prefix && Fn == "simd_prefix_exclusive_sum" && IsFloat) {
      requireHelper("crucible_wave_prefix_sum");
      Fn = "crucible_wave_prefix_sum";
    }
    if (T.isInteger(16) && *Kind == 0) {
      // Some Metal compilers merge identical narrow reductions across
      // different control-flow masks. Capture the participating lanes.
      requireHelper("crucible_wave_wide");
      UsesLane |= Prefix;
      return Expr{std::string("ushort(crucible_wave_wide<ulong, 0, ") +
                      (Prefix ? "true>(ulong(" : "false>(ulong(") + V->Text +
                      "), " + (Prefix ? "lane" : "0u") + "))",
                  PrecPrimary};
    }
    if (T.isInteger(16) && *Kind == 1)
      return Expr{"ushort(" + Fn.str() + "(uint(" + V->Text + ")))",
                  PrecPrimary};
    if (T.isF16() && (*Kind == 0 || *Kind == 1))
      return Expr{"half(" + Fn.str() + "(float(" + V->Text + ")))",
                  PrecPrimary};
    return Expr{Fn.str() + "(" + V->Text + ")", PrecPrimary};
  }
  case DXILOp::WaveActiveBit: {
    auto Kind = constInt(Args[2]);
    if (!Kind)
      return makeError("dynamic wave bit op kind is not supported");
    StringRef Fn;
    switch (static_cast<DXILWaveBitOpKind>(*Kind)) {
    case DXILWaveBitOpKind::And:
      Fn = "simd_and";
      break;
    case DXILWaveBitOpKind::Or:
      Fn = "simd_or";
      break;
    case DXILWaveBitOpKind::Xor:
      Fn = "simd_xor";
      break;
    }
    auto V = exprFor(Args[1], PrecUnknown);
    if (!V)
      return V.takeError();
    if (Fn.empty())
      return makeError("invalid DXIL wave bit operation kind");
    if (Args[1].getType().isInteger(64))
      return Expr{"as_type<ulong>(" + Fn.str() + "(as_type<uint2>(" +
                      V->Text + ")))",
                  PrecPrimary};
    return Expr{Fn.str() + "(" + V->Text + ")", PrecPrimary};
  }
  default:
    break;
  }

  return makeError("unsupported DXIL op {0}", static_cast<uint32_t>(Op));
}

Expected<std::string> Emitter::bufferIndexExpr(ResState &RS, mlir::Value Coord0,
                                               mlir::Value Coord1,
                                               unsigned Comp) {
  std::string Index;
  uint64_t ConstOffset = 0; // in scalar elements
  const unsigned EB = elemByteSize(RS.ElemType);
  const unsigned Shift = EB == 2 ? 1 : EB == 8 ? 3 : 2;

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

  auto Call = mlir::dyn_cast_or_null<mlir::LLVM::CallOp>(Producer);
  if (!Call)
    return makeError("extractvalue from a non-resource aggregate");
  auto Op = dxOpcode(Call);
  if (!Op)
    return makeError("extractvalue from unsupported call");
  auto Args = Call.getArgOperands();

  if (*Op == DXILOp::UAddc) {
    auto It = ValueNames.find(Call.getResult());
    if (It == ValueNames.end() || Comp > 1)
      return makeError("invalid extractvalue from uaddc");
    return Expr{Comp == 0 ? "uint(" + It->second + ")"
                          : "bool(" + It->second + " >> 32u)",
                PrecPrimary};
  }

  // Ballot aggregates were materialized as uint4 temporaries.
  if (*Op == DXILOp::WaveActiveBallot) {
    auto It = ValueNames.find(Call.getResult());
    if (It == ValueNames.end())
      return makeError("use of ballot result before materialization");
    return Expr{It->second + "." + std::string(1, "xyzw"[Comp & 3]),
                PrecPrimary};
  }

  if (*Op == DXILOp::BufferLoad || *Op == DXILOp::RawBufferLoad) {
    if (Comp > 3)
      return makeError("buffer load status queries are not supported yet");
    ResState *RS = stateFor(Args[1]);
    if (!RS)
      return makeError("buffer load through unresolved handle");
    auto Index = bufferIndexExpr(*RS, Args[2], Args[3], Comp);
    if (!Index)
      return Index.takeError();
    std::string Element = RS->ParamName + "[" + *Index + "]";
    // Reinterpret when this load's type differs from the canonical element
    // type (same-size mixed-type access).
    auto Ret =
        mlir::cast<mlir::LLVM::LLVMStructType>(Call.getResult().getType());
    auto LoadTy = mslType(Ret.getBody()[0]);
    if (!LoadTy)
      return LoadTy.takeError();
    if (*LoadTy != RS->ElemType)
      return Expr{"as_type<" + *LoadTy + ">(" + Element + ")", PrecPrimary};
    return Expr{Element, PrecPrimary};
  }

  if (*Op == DXILOp::CBufferLoadLegacy) {
    ResState *RS = stateFor(Args[1]);
    if (!RS)
      return makeError("cbuffer load through unresolved handle");
    auto Row = exprFor(Args[2], PrecUnknown);
    if (!Row)
      return Row.takeError();
    std::string RowText = RS->ParamName + "[" + Row->Text + "]";
    // Reinterpret when this load's element type differs from the canonical
    // row type (mixed float/uint cbuffers).
    auto Ret =
        mlir::cast<mlir::LLVM::LLVMStructType>(Call.getResult().getType());
    mlir::Type T = Ret.getBody()[0];
    bool RowIsFloat = RS->ElemType == "float4";
    if (Comp >= Ret.getBody().size())
      return makeError("cbuffer component out of range");
    if (T.isInteger(64))
      return Expr{"as_type<ulong2>(" + RowText + ")[" +
                      std::to_string(Comp) + "]",
                  PrecPrimary};
    if (T.isF16() || T.isInteger(16)) {
      bool Packed = Ret.getBody().size() == 8;
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

Error Emitter::emitBufferStore(mlir::LLVM::CallOp Call, unsigned Ind) {
  // (op, handle, coord0, coord1, v0, v1, v2, v3, mask:i8[, alignment])
  auto Args = Call.getArgOperands();
  ResState *RS = stateFor(Args[1]);
  if (!RS)
    return makeError("buffer store through unresolved handle");
  auto Mask = constInt(Args[8]);
  if (!Mask)
    return makeError("dynamic store mask is not supported");
  for (unsigned Comp = 0; Comp < 4; ++Comp) {
    if (RS->RB->Kind == ResourceKind::TypedBuffer &&
        RS->RB->TypedComponents != 0 && Comp >= RS->RB->TypedComponents)
      break;
    if (!(*Mask & (1u << Comp)))
      continue;
    auto Index = bufferIndexExpr(*RS, Args[2], Args[3], Comp);
    if (!Index)
      return Index.takeError();
    auto V = exprFor(Args[4 + Comp], PrecUnknown);
    if (!V)
      return V.takeError();
    std::string Value = V->Text;
    auto StoreTy = mslType(Args[4 + Comp].getType());
    if (!StoreTy)
      return StoreTy.takeError();
    if (*StoreTy != RS->ElemType)
      Value = "as_type<" + RS->ElemType + ">(" + Value + ")";
    line(Ind, RS->ParamName + "[" + *Index + "] = " + Value + ";");
  }
  return Error::success();
}

Error Emitter::emitBarrier(mlir::LLVM::CallOp Call, unsigned Ind) {
  auto Mode = constInt(Call.getArgOperands()[1]);
  if (!Mode)
    return makeError("dynamic barrier mode is not supported");
  // DXIL barrier modes: 1 = sync threadgroup, 2 = UAV fence global,
  // 4 = UAV fence threadgroup, 8 = TGSM fence.
  std::string Flags;
  if (*Mode & 8)
    Flags = "mem_flags::mem_threadgroup";
  if (*Mode & (2 | 4))
    Flags = Flags.empty() ? "mem_flags::mem_device"
                          : Flags + " | mem_flags::mem_device";
  if (Flags.empty())
    Flags = "mem_flags::mem_none";
  line(Ind, "threadgroup_barrier(" + Flags + ");");
  return Error::success();
}

// Lower dx.op.atomicBinOp / atomicCompareExchange on buffer UAVs to Metal
// atomic_* functions over reinterpreted atomic views.
Error Emitter::emitDXAtomic(mlir::LLVM::CallOp Call, DXILOp Op, unsigned Ind) {
  auto Args = Call.getArgOperands();
  ResState *RS = stateFor(Args[1]);
  if (!RS)
    return makeError("atomic through unresolved handle");

  const unsigned Coord0 = Op == DXILOp::AtomicBinOp ? 3 : 2;
  if (!isUndefLike(Args[Coord0 + 2]))
    return makeError("texture atomics are not supported yet");
  auto Index = bufferIndexExpr(*RS, Args[Coord0], Args[Coord0 + 1], 0);
  if (!Index)
    return Index.takeError();
  const std::string LV = RS->ParamName + "[" + *Index + "]";

  std::string Text;
  if (Op == DXILOp::AtomicBinOp) {
    auto AOp = constInt(Args[2]);
    if (!AOp)
      return makeError("dynamic atomic op kind is not supported");
    auto V = exprFor(Args[6], PrecUnknown);
    if (!V)
      return V.takeError();
    const char *Fn = nullptr;
    bool Signed = false;
    switch (static_cast<DXILAtomicBinOp>(*AOp)) {
    case DXILAtomicBinOp::Add:
      Fn = "atomic_fetch_add_explicit";
      break;
    case DXILAtomicBinOp::And:
      Fn = "atomic_fetch_and_explicit";
      break;
    case DXILAtomicBinOp::Or:
      Fn = "atomic_fetch_or_explicit";
      break;
    case DXILAtomicBinOp::Xor:
      Fn = "atomic_fetch_xor_explicit";
      break;
    case DXILAtomicBinOp::IMin:
      Fn = "atomic_fetch_min_explicit";
      Signed = true;
      break;
    case DXILAtomicBinOp::IMax:
      Fn = "atomic_fetch_max_explicit";
      Signed = true;
      break;
    case DXILAtomicBinOp::UMin:
      Fn = "atomic_fetch_min_explicit";
      break;
    case DXILAtomicBinOp::UMax:
      Fn = "atomic_fetch_max_explicit";
      break;
    case DXILAtomicBinOp::Exchange:
      Fn = "atomic_exchange_explicit";
      break;
    default:
      return makeError("unsupported atomic op kind {0}", *AOp);
    }
    if (Signed)
      Text = std::string("as_type<uint>(") + Fn + "((device atomic_int *)&" +
             LV + ", as_type<int>(" + V->Text + "), memory_order_relaxed))";
    else
      Text = std::string(Fn) + "((device atomic_uint *)&" + LV + ", " +
             V->Text + ", memory_order_relaxed)";
  } else {
    requireHelper("crucible_atomic_cmpxchg_device");
    auto Cmp = exprFor(Args[5], PrecUnknown);
    if (!Cmp)
      return Cmp.takeError();
    auto V = exprFor(Args[6], PrecUnknown);
    if (!V)
      return V.takeError();
    Text = "crucible_atomic_cmpxchg((device atomic_uint *)&" + LV + ", " +
           Cmp->Text + ", " + V->Text + ")";
  }

  if (Call.getNumResults() == 1 && !Call.getResult().use_empty()) {
    std::string Name = freshName();
    line(Ind, "uint " + Name + " = " + Text + ";");
    ValueNames[Call.getResult()] = Name;
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
  auto Selector = exprFor(Switch.getArg(), PrecUnknown);
  if (!Selector)
    return Selector.takeError();
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

  // General form:
  //   while (true) { <before>; r = condArgs; if (!cond) break; <after>;
  //                  carried = yields; }
  SmallVector<std::string, 4> ResultNames;
  mlir::Block &After = While.getAfter().front();
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

  if (auto Call = mlir::dyn_cast<mlir::LLVM::CallOp>(Op)) {
    auto DXOp = dxOpcode(Call);
    if (!DXOp)
      return makeError("call to unsupported function '{0}'",
                       Call.getCallee() ? *Call.getCallee() : "<indirect>");
    switch (*DXOp) {
    case DXILOp::CreateHandle:
    case DXILOp::CreateHandleFromBinding:
    case DXILOp::AnnotateHandle:
      return Error::success(); // folded into bindings
    case DXILOp::BufferStore:
    case DXILOp::RawBufferStore:
      return emitBufferStore(Call, Ind);
    case DXILOp::Barrier:
      return emitBarrier(Call, Ind);
    case DXILOp::AtomicBinOp:
    case DXILOp::AtomicCompareExchange:
      return emitDXAtomic(Call, *DXOp, Ind);
    case DXILOp::UAddc: {
      auto Args = Call.getArgOperands();
      auto A = exprFor(Args[1], PrecUnknown);
      if (!A)
        return A.takeError();
      auto B = exprFor(Args[2], PrecUnknown);
      if (!B)
        return B.takeError();
      std::string Name = freshName();
      line(Ind, "ulong " + Name + " = ulong(" + A->Text + ") + ulong(" +
                    B->Text + ");");
      ValueNames[Call.getResult()] = Name;
      return Error::success();
    }
    case DXILOp::WaveActiveBallot: {
      // Aggregate result; materialize as a uint4 so extractvalue can swizzle.
      requireHelper("crucible_wave_ballot");
      auto Cond = exprFor(Call.getArgOperands()[1], PrecUnknown);
      if (!Cond)
        return Cond.takeError();
      std::string Name = freshName();
      line(Ind, "uint4 " + Name + " = crucible_wave_ballot(" + Cond->Text +
                    ");");
      ValueNames[Call.getResult()] = Name;
      return Error::success();
    }
    case DXILOp::BufferLoad:
    case DXILOp::RawBufferLoad:
    case DXILOp::CBufferLoadLegacy:
      return Error::success(); // materialized at extractvalue sites
    default:
      break;
    }
  }

  // The ResBind aggregate feeding createHandleFromBinding.
  if (mlir::isa<mlir::LLVM::InsertValueOp>(Op))
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
    auto LV = lvalueFor(Store.getAddr());
    if (!LV)
      return LV.takeError();
    auto V = exprFor(Store.getValue(), PrecUnknown);
    if (!V)
      return V.takeError();
    line(Ind, LV->Text + " = " + V->Text + ";");
    return Error::success();
  }

  if (auto RMW = mlir::dyn_cast<mlir::LLVM::AtomicRMWOp>(Op)) {
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
    if (Signed)
      Text = std::string("as_type<uint>(") + Fn + "((" + *AS +
             " atomic_int *)&" + LV->Text + ", as_type<int>(" + V->Text +
             "), memory_order_relaxed))";
    else
      Text = std::string(Fn) + "((" + *AS + " atomic_uint *)&" + LV->Text +
             ", " + V->Text + ", memory_order_relaxed)";
    if (!RMW.getResult().use_empty()) {
      std::string Name = freshName();
      line(Ind, "uint " + Name + " = " + Text + ";");
      ValueNames[RMW.getResult()] = Name;
    } else {
      line(Ind, Text + ";");
    }
    return Error::success();
  }

  if (auto CX = mlir::dyn_cast<mlir::LLVM::AtomicCmpXchgOp>(Op)) {
    auto LV = lvalueFor(CX.getPtr());
    if (!LV)
      return LV.takeError();
    auto AS = addrSpaceOf(CX.getPtr());
    if (!AS)
      return AS.takeError();
    requireHelper(*AS == "threadgroup" ? "crucible_atomic_cmpxchg_threadgroup"
                                       : "crucible_atomic_cmpxchg_device");
    auto Cmp = exprFor(CX.getCmp(), PrecUnknown);
    if (!Cmp)
      return Cmp.takeError();
    auto V = exprFor(CX.getVal(), PrecUnknown);
    if (!V)
      return V.takeError();
    std::string Name = freshName();
    line(Ind, "uint " + Name + " = crucible_atomic_cmpxchg((" + *AS +
                  " atomic_uint *)&" + LV->Text + ", " + Cmp->Text + ", " +
                  V->Text + ");");
    ValueNames[CX.getResult()] = Name;
    return Error::success();
  }

  // Pure value: inline single-use values; declare temporaries otherwise.
  if (shouldInline(Op))
    return Error::success();
  if (Op->getNumResults() == 1 && Op->getResult(0).use_empty())
    return Error::success(); // dead
  return emitTemp(Op, Ind);
}

Error Emitter::emitBlock(mlir::Block &B, unsigned Ind) {
  for (mlir::Operation &Op : B)
    if (Error E = emitStatement(&Op, Ind))
      return E;
  return Error::success();
}

//===----------------------------------------------------------------------===//
// Kernel signature
//===----------------------------------------------------------------------===//

Expected<std::string> Emitter::kernelSignature() {
  if (R.Stage != ShaderStage::Compute)
    return makeError("only compute shaders are supported so far");

  std::string Sig;
  raw_string_ostream OS(Sig);
  OS << "kernel void " << R.MSLEntryName << "(\n";

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

  for (const ResState *RS : Sorted) {
    const ResourceBinding &RB = *RS->RB;
    Sep();
    switch (RB.Class) {
    case ResourceClass::SRV:
      OS << "    device const " << RS->ElemType << "* " << RS->ParamName
         << " [[buffer(" << RB.MetalIndex << ")]]";
      break;
    case ResourceClass::UAV:
      OS << "    device " << RS->ElemType << "* " << RS->ParamName
         << " [[buffer(" << RB.MetalIndex << ")]]";
      break;
    case ResourceClass::CBV:
      OS << "    constant " << RS->ElemType << "* " << RS->ParamName
         << " [[buffer(" << RB.MetalIndex << ")]]";
      break;
    case ResourceClass::Sampler:
      return makeError("samplers are not supported yet");
    }
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
  OS << ")";
  return Sig;
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

Expected<std::string> Emitter::run() {
  R.MSLEntryName = sanitizeIdentifier(R.EntryName, "kernel_main");

  if (Error E = resolveHandles())
    return std::move(E);
  if (Error E = inferElementTypes())
    return std::move(E);
  if (Error E = collectGroupshared())
    return std::move(E);
  if (Error E = emitBlock(Entry.getBody().front(), 1))
    return std::move(E);

  auto Sig = kernelSignature();
  if (!Sig)
    return Sig.takeError();

  std::string Out;
  raw_string_ostream OS(Out);
  OS << "// Generated by crucible from DXIL (" << R.EntryName << ", cs_"
     << R.ShaderModelMajor << "_" << R.ShaderModelMinor << ")\n"
     << "#include <metal_stdlib>\n"
     << "using namespace metal;\n\n";
  for (const std::string &H : HelperOrder)
    OS << helperSource(H) << "\n";
  OS << *Sig << " {\n" << GroupsharedDecls << Body << "}\n";
  return Out;
}

} // namespace

Expected<std::string> emitMSL(mlir::func::FuncOp Entry, Reflection &R,
                              const Options &Opts) {
  return Emitter(Entry, R, Opts).run();
}

} // namespace crucible

//===- MSLEmitter.cpp - LLVM IR -> Metal Shading Language --------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Prints upgraded DXIL (modern LLVM IR plus @dx.op.* calls) as clean,
// idiomatic MSL. Pure single-use values are inlined into their consumer
// expression so the output reads like hand-written Metal rather than a
// register dump; multi-use values become typed local temporaries.
//
//===----------------------------------------------------------------------===//

#include "MSLEmitter.h"
#include "IntrinsicMap.h"
#include "crucible/Diagnostic.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"

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

std::string sanitizeIdentifier(StringRef Name, StringRef Fallback) {
  std::string Out;
  for (char C : Name)
    Out += (isalnum(static_cast<unsigned char>(C)) || C == '_') ? C : '_';
  if (Out.empty() || isdigit(static_cast<unsigned char>(Out[0])))
    return Fallback.str();
  // Reserved in MSL.
  static const char *Reserved[] = {"main", "kernel", "vertex", "fragment",
                                   "constant", "device", "thread", "threadgroup"};
  for (const char *R : Reserved)
    if (Out == R)
      return (Out + "0");
  return Out;
}

//===----------------------------------------------------------------------===//
// Emitter
//===----------------------------------------------------------------------===//

class Emitter {
public:
  Emitter(LoadedShader &Shader, const Options &Opts)
      : Shader(Shader), Opts(Opts) {}

  Expected<std::string> run();

private:
  LoadedShader &Shader;
  const Options &Opts;

  struct ResState {
    ResourceBinding *RB = nullptr;
    std::string ParamName;
    std::string ElemType; // scalar MSL type for buffers; row type for CBVs
  };

  std::vector<ResState> ResStates;                 // parallel to Reflect.Resources
  DenseMap<const Value *, ResState *> HandleMap;   // handle SSA value -> resource
  DenseMap<const Value *, std::string> ValueNames; // declared temps
  unsigned NextTemp = 0;
  // Compute builtin usage.
  bool UsesDTid = false, UsesGTid = false, UsesGid = false, UsesGidx = false;

  std::string Body;
  Error pendingError = Error::success();

  Error err(Error E) { return E; }

  // --- type & constant printing ---
  Expected<std::string> mslType(Type *T);
  Expected<Expr> constant(const Constant *C);

  // --- resource discovery ---
  Error resolveHandles(Function &F);
  ResState *stateFor(const Value *Handle) {
    auto It = HandleMap.find(Handle);
    return It == HandleMap.end() ? nullptr : It->second;
  }
  Error inferElementTypes(Function &F);

  // --- expression emission ---
  Expected<Expr> exprFor(const Value *V, int ParentPrec);
  Expected<Expr> instExpr(const Instruction *I);
  Expected<Expr> dxOpExpr(const CallInst *CI, DXILOp Op);
  Expected<Expr> bufferElementExpr(const ExtractValueInst *EV);
  Expected<std::string> bufferIndexExpr(ResState &RS, const Value *Coord0,
                                        const Value *Coord1, unsigned Comp);
  bool shouldInline(const Instruction *I);

  // --- statement emission ---
  Error emitBody(Function &F);
  Error emitStatement(const Instruction *I);
  Error emitBufferStore(const CallInst *CI);
  Error emitBarrier(const CallInst *CI);

  Expected<std::string> kernelSignature();
};

//===----------------------------------------------------------------------===//
// Types and constants
//===----------------------------------------------------------------------===//

Expected<std::string> Emitter::mslType(Type *T) {
  if (T->isFloatTy())
    return std::string("float");
  if (T->isHalfTy())
    return std::string("half");
  if (T->isIntegerTy(1))
    return std::string("bool");
  if (T->isIntegerTy(16))
    return std::string("ushort");
  if (T->isIntegerTy(32))
    return std::string("uint");
  if (T->isIntegerTy(64))
    return std::string("ulong");
  if (T->isDoubleTy())
    return makeError("MSL has no double type; use float in the shader");
  std::string S;
  raw_string_ostream OS(S);
  T->print(OS);
  return makeError("unsupported type '{0}'", S);
}

static std::string formatFloatLiteral(const APFloat &APF, bool IsHalf) {
  SmallString<32> Str;
  APFloat V = APF;
  bool LosesInfo = false;
  V.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &LosesInfo);
  double D = V.convertToDouble();
  raw_svector_ostream OS(Str);
  OS << format("%.9g", D);
  std::string S(Str.str());
  if (S.find('.') == std::string::npos && S.find('e') == std::string::npos &&
      S.find("inf") == std::string::npos && S.find("nan") == std::string::npos)
    S += ".0";
  if (S.find("inf") != std::string::npos)
    return IsHalf ? "HUGE_VALH" : "HUGE_VALF";
  S += IsHalf ? "h" : "f";
  return S;
}

Expected<Expr> Emitter::constant(const Constant *C) {
  if (const auto *CI = dyn_cast<ConstantInt>(C)) {
    if (CI->getType()->isIntegerTy(1))
      return Expr{CI->isZero() ? "false" : "true", PrecPrimary};
    std::string S = std::to_string(CI->getZExtValue());
    if (CI->getType()->getIntegerBitWidth() == 64)
      S += "ul";
    else
      S += "u";
    return Expr{S, PrecPrimary};
  }
  if (const auto *CF = dyn_cast<ConstantFP>(C))
    return Expr{formatFloatLiteral(CF->getValueAPF(),
                                   CF->getType()->isHalfTy()),
                PrecPrimary};
  if (isa<UndefValue>(C) || isa<PoisonValue>(C))
    return Expr{"0", PrecPrimary};
  return makeError("unsupported constant");
}

//===----------------------------------------------------------------------===//
// Resource discovery
//===----------------------------------------------------------------------===//

static std::optional<DXILOp> dxOpcode(const CallInst *CI) {
  const Function *F = CI->getCalledFunction();
  if (!F || !F->getName().starts_with("dx.op."))
    return std::nullopt;
  if (const auto *Op = dyn_cast<ConstantInt>(CI->getArgOperand(0)))
    return static_cast<DXILOp>(Op->getZExtValue());
  return std::nullopt;
}

Error Emitter::resolveHandles(Function &F) {
  // Prepare per-resource state with stable parameter names.
  auto &Rs = Shader.Reflect.Resources;
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

  for (Instruction &I : instructions(F)) {
    const auto *CI = dyn_cast<CallInst>(&I);
    if (!CI)
      continue;
    auto Op = dxOpcode(CI);
    if (!Op)
      continue;
    switch (*Op) {
    case DXILOp::CreateHandle: {
      // (op, class:i8, rangeID:i32, index:i32, nonUniform:i1)
      const auto *ClassC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
      const auto *RangeC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
      if (!ClassC || !RangeC)
        return makeError("dynamic createHandle operands are not supported");
      ResState *RS = FindBinding(
          static_cast<ResourceClass>(ClassC->getZExtValue()),
          static_cast<uint32_t>(RangeC->getZExtValue()));
      if (!RS)
        return makeError("createHandle references unknown resource range {0}",
                         RangeC->getZExtValue());
      HandleMap[CI] = RS;
      break;
    }
    case DXILOp::CreateHandleFromBinding: {
      // (op, resBind:{lb,ub,space,class}, index:i32, nonUniform:i1)
      const auto *Bind = dyn_cast<Constant>(CI->getArgOperand(1));
      const auto *Index = dyn_cast<ConstantInt>(CI->getArgOperand(2));
      if (!Bind || !Index)
        return makeError(
            "dynamic createHandleFromBinding operands are not supported");
      uint32_t LB = 0, Space = 0;
      uint32_t Class = 0;
      if (const auto *CS = dyn_cast<ConstantStruct>(Bind)) {
        LB = cast<ConstantInt>(CS->getOperand(0))->getZExtValue();
        Space = cast<ConstantInt>(CS->getOperand(2))->getZExtValue();
        Class = cast<ConstantInt>(CS->getOperand(3))->getZExtValue();
      }
      ResState *RS = FindByRegister(static_cast<ResourceClass>(Class), Space,
                                    static_cast<uint32_t>(Index->getZExtValue()));
      if (!RS)
        return makeError("createHandleFromBinding references unknown binding "
                         "(space {0}, register {1})",
                         Space, Index->getZExtValue());
      HandleMap[CI] = RS;
      break;
    }
    case DXILOp::AnnotateHandle:
      // Forwarded below once operands are mapped; handled lazily in stateFor
      // via a second pass because the source handle may appear later.
      break;
    default:
      break;
    }
  }

  // Forward annotateHandle chains.
  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (Instruction &I : instructions(F)) {
      const auto *CI = dyn_cast<CallInst>(&I);
      if (!CI || HandleMap.count(CI))
        continue;
      auto Op = dxOpcode(CI);
      if (!Op || *Op != DXILOp::AnnotateHandle)
        continue;
      if (ResState *RS = stateFor(CI->getArgOperand(1))) {
        HandleMap[CI] = RS;
        Changed = true;
      }
    }
  }
  return Error::success();
}

Error Emitter::inferElementTypes(Function &F) {
  for (Instruction &I : instructions(F)) {
    const auto *CI = dyn_cast<CallInst>(&I);
    if (!CI)
      continue;
    auto Op = dxOpcode(CI);
    if (!Op)
      continue;

    auto Record = [&](const Value *Handle, Type *Elem) -> Error {
      ResState *RS = stateFor(Handle);
      if (!RS)
        return makeError("resource access through unresolved handle");
      auto TyOrErr = mslType(Elem);
      if (!TyOrErr)
        return TyOrErr.takeError();
      if (RS->ElemType.empty())
        RS->ElemType = *TyOrErr;
      else if (RS->ElemType != *TyOrErr)
        return makeError("resource '{0}' is accessed as both {1} and {2}; "
                         "mixed element types are not supported yet",
                         RS->ParamName, RS->ElemType, *TyOrErr);
      return Error::success();
    };

    switch (*Op) {
    case DXILOp::BufferLoad: {
      auto *Ret = cast<StructType>(CI->getType());
      if (Error E = Record(CI->getArgOperand(1), Ret->getElementType(0)))
        return E;
      break;
    }
    case DXILOp::BufferStore:
      if (Error E = Record(CI->getArgOperand(1),
                           CI->getArgOperand(4)->getType()))
        return E;
      break;
    case DXILOp::CBufferLoadLegacy: {
      auto *Ret = cast<StructType>(CI->getType());
      Type *Elem = Ret->getElementType(0);
      ResState *RS = stateFor(CI->getArgOperand(1));
      if (!RS)
        return makeError("cbuffer access through unresolved handle");
      std::string Row = Elem->isFloatTy()  ? "float4"
                        : Elem->isHalfTy() ? "half4"
                                           : "uint4";
      if (RS->ElemType.empty())
        RS->ElemType = Row;
      else if (RS->ElemType != Row)
        return makeError("cbuffer '{0}' loaded with mixed row types",
                         RS->ParamName);
      break;
    }
    default:
      break;
    }
  }
  // Defaults for untouched resources.
  for (ResState &RS : ResStates)
    if (RS.ElemType.empty())
      RS.ElemType = RS.RB->Class == ResourceClass::CBV ? "float4" : "uint";
  return Error::success();
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

bool Emitter::shouldInline(const Instruction *I) {
  if (const auto *CI = dyn_cast<CallInst>(I)) {
    auto Op = dxOpcode(CI);
    if (!Op)
      return false;
    switch (*Op) {
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
    case DXILOp::CBufferLoadLegacy:
      // Never materialized directly; consumed via handles/extractvalue.
      return true;
    default:
      return I->hasOneUse();
    }
  }

  if (!I->hasOneUse())
    return false;

  // A UAV read must not be folded forward across a write or barrier, since
  // the inlined load would observe the mutation.
  if (const auto *EV = dyn_cast<ExtractValueInst>(I)) {
    const auto *Load = dyn_cast<CallInst>(EV->getAggregateOperand());
    if (Load && dxOpcode(Load) == DXILOp::BufferLoad) {
      ResState *RS = stateFor(Load->getArgOperand(1));
      if (RS && RS->RB->Class == ResourceClass::UAV) {
        const auto *User = cast<Instruction>(*I->user_begin());
        for (const Instruction *P = I->getNextNode(); P && P != User;
             P = P->getNextNode())
          if (P->mayWriteToMemory())
            return false;
      }
    }
    return true;
  }

  return !I->mayHaveSideEffects();
}

Expected<Expr> Emitter::exprFor(const Value *V, int ParentPrec) {
  Expr E;
  if (const auto *C = dyn_cast<Constant>(V)) {
    auto CE = constant(C);
    if (!CE)
      return CE.takeError();
    E = *CE;
  } else if (const auto *I = dyn_cast<Instruction>(V)) {
    auto It = ValueNames.find(I);
    if (It != ValueNames.end()) {
      E = Expr{It->second, PrecPrimary};
    } else {
      auto IE = instExpr(I);
      if (!IE)
        return IE.takeError();
      E = *IE;
    }
  } else {
    return makeError("unsupported value kind in expression");
  }
  if (E.Prec < ParentPrec)
    return Expr{"(" + E.Text + ")", PrecPrimary};
  return E;
}

// Render a binary operator, inserting signed reinterpretation where the LLVM
// opcode implies signed semantics over our default-unsigned integers.
static const char *binOpSpelling(unsigned Opcode) {
  switch (Opcode) {
  case Instruction::FAdd:
  case Instruction::Add:
    return "+";
  case Instruction::FSub:
  case Instruction::Sub:
    return "-";
  case Instruction::FMul:
  case Instruction::Mul:
    return "*";
  case Instruction::FDiv:
  case Instruction::UDiv:
  case Instruction::SDiv:
    return "/";
  case Instruction::FRem:
  case Instruction::URem:
  case Instruction::SRem:
    return "%";
  case Instruction::Shl:
    return "<<";
  case Instruction::LShr:
  case Instruction::AShr:
    return ">>";
  case Instruction::And:
    return "&";
  case Instruction::Or:
    return "|";
  case Instruction::Xor:
    return "^";
  default:
    return nullptr;
  }
}

static int binOpPrec(const char *Spelling) {
  StringRef S(Spelling);
  if (S == "*" || S == "/" || S == "%")
    return PrecMul;
  if (S == "+" || S == "-")
    return PrecAdd;
  if (S == "<<" || S == ">>")
    return PrecShift;
  if (S == "&")
    return PrecAnd;
  if (S == "^")
    return PrecXor;
  return PrecOr;
}

Expected<Expr> Emitter::instExpr(const Instruction *I) {
  if (const auto *BO = dyn_cast<BinaryOperator>(I)) {
    const char *Op = binOpSpelling(BO->getOpcode());
    if (!Op)
      return makeError("unsupported binary operator '{0}'",
                       BO->getOpcodeName());
    bool Signed = BO->getOpcode() == Instruction::SDiv ||
                  BO->getOpcode() == Instruction::SRem ||
                  BO->getOpcode() == Instruction::AShr;
    int Prec = binOpPrec(Op);
    auto L = exprFor(BO->getOperand(0), Signed ? PrecUnknown : Prec);
    if (!L)
      return L.takeError();
    auto R = exprFor(BO->getOperand(1), Signed ? PrecUnknown : Prec + 1);
    if (!R)
      return R.takeError();
    if (Signed) {
      // Signed semantics: reinterpret, operate, reinterpret back.
      std::string Text = "as_type<uint>(as_type<int>(" + L->Text + ") " + Op +
                         " as_type<int>(" + R->Text + "))";
      // AShr right operand stays an unsigned shift count.
      if (BO->getOpcode() == Instruction::AShr)
        Text = "as_type<uint>(as_type<int>(" + L->Text + ") >> " + R->Text +
               ")";
      return Expr{Text, PrecPrimary};
    }
    return Expr{L->Text + " " + Op + " " + R->Text, Prec};
  }

  if (const auto *Cmp = dyn_cast<ICmpInst>(I)) {
    const char *Op = nullptr;
    bool Signed = Cmp->isSigned();
    switch (Cmp->getUnsignedPredicate()) {
    case CmpInst::ICMP_EQ:
      Op = "==";
      break;
    case CmpInst::ICMP_NE:
      Op = "!=";
      break;
    case CmpInst::ICMP_ULT:
      Op = "<";
      break;
    case CmpInst::ICMP_ULE:
      Op = "<=";
      break;
    case CmpInst::ICMP_UGT:
      Op = ">";
      break;
    case CmpInst::ICMP_UGE:
      Op = ">=";
      break;
    default:
      return makeError("unsupported icmp predicate");
    }
    int Prec = (StringRef(Op) == "==" || StringRef(Op) == "!=") ? PrecEq
                                                                : PrecRel;
    auto L = exprFor(Cmp->getOperand(0), Signed ? PrecUnknown : Prec);
    if (!L)
      return L.takeError();
    auto R = exprFor(Cmp->getOperand(1), Signed ? PrecUnknown : Prec + 1);
    if (!R)
      return R.takeError();
    if (Signed)
      return Expr{"as_type<int>(" + L->Text + ") " + Op + " as_type<int>(" +
                      R->Text + ")",
                  Prec};
    return Expr{L->Text + " " + Op + " " + R->Text, Prec};
  }

  if (const auto *FC = dyn_cast<FCmpInst>(I)) {
    const char *Op = nullptr;
    switch (FC->getPredicate()) {
    case CmpInst::FCMP_OEQ:
    case CmpInst::FCMP_UEQ:
      Op = "==";
      break;
    case CmpInst::FCMP_ONE:
    case CmpInst::FCMP_UNE:
      Op = "!=";
      break;
    case CmpInst::FCMP_OLT:
    case CmpInst::FCMP_ULT:
      Op = "<";
      break;
    case CmpInst::FCMP_OLE:
    case CmpInst::FCMP_ULE:
      Op = "<=";
      break;
    case CmpInst::FCMP_OGT:
    case CmpInst::FCMP_UGT:
      Op = ">";
      break;
    case CmpInst::FCMP_OGE:
    case CmpInst::FCMP_UGE:
      Op = ">=";
      break;
    default:
      return makeError("unsupported fcmp predicate");
    }
    int Prec = (StringRef(Op) == "==" || StringRef(Op) == "!=") ? PrecEq
                                                                : PrecRel;
    auto L = exprFor(FC->getOperand(0), Prec);
    if (!L)
      return L.takeError();
    auto R = exprFor(FC->getOperand(1), Prec + 1);
    if (!R)
      return R.takeError();
    return Expr{L->Text + " " + Op + " " + R->Text, Prec};
  }

  if (const auto *Sel = dyn_cast<SelectInst>(I)) {
    auto C = exprFor(Sel->getCondition(), PrecLOr);
    if (!C)
      return C.takeError();
    auto T = exprFor(Sel->getTrueValue(), PrecTernary);
    if (!T)
      return T.takeError();
    auto F = exprFor(Sel->getFalseValue(), PrecTernary);
    if (!F)
      return F.takeError();
    return Expr{C->Text + " ? " + T->Text + " : " + F->Text, PrecTernary};
  }

  if (const auto *Cast = dyn_cast<CastInst>(I)) {
    auto Src = exprFor(Cast->getOperand(0), PrecUnknown);
    if (!Src)
      return Src.takeError();
    auto DstTy = mslType(Cast->getDestTy());
    if (!DstTy)
      return DstTy.takeError();
    switch (Cast->getOpcode()) {
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::FPToUI:
    case Instruction::UIToFP:
      return Expr{*DstTy + "(" + Src->Text + ")", PrecPrimary};
    case Instruction::SExt:
      return Expr{*DstTy + "(as_type<int>(" + Src->Text + "))", PrecPrimary};
    case Instruction::FPToSI:
      return Expr{"as_type<uint>(int(" + Src->Text + "))", PrecPrimary};
    case Instruction::SIToFP:
      return Expr{*DstTy + "(as_type<int>(" + Src->Text + "))", PrecPrimary};
    case Instruction::BitCast:
      return Expr{"as_type<" + *DstTy + ">(" + Src->Text + ")", PrecPrimary};
    default:
      return makeError("unsupported cast '{0}'", Cast->getOpcodeName());
    }
  }

  if (const auto *EV = dyn_cast<ExtractValueInst>(I))
    return bufferElementExpr(EV);

  if (const auto *CI = dyn_cast<CallInst>(I)) {
    auto Op = dxOpcode(CI);
    if (!Op)
      return makeError("call to unsupported function '{0}'",
                       CI->getCalledFunction()
                           ? CI->getCalledFunction()->getName()
                           : "<indirect>");
    return dxOpExpr(CI, *Op);
  }

  return makeError("unsupported instruction '{0}'", I->getOpcodeName());
}

Expected<Expr> Emitter::dxOpExpr(const CallInst *CI, DXILOp Op) {
  auto Component = [&](const char *Base) -> Expected<Expr> {
    const auto *Comp = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    if (!Comp)
      return makeError("dynamic thread-id component is not supported");
    static const char Swizzle[] = {'x', 'y', 'z'};
    uint64_t Idx = Comp->getZExtValue();
    if (Idx > 2)
      return makeError("thread-id component out of range");
    return Expr{std::string(Base) + "." + Swizzle[Idx], PrecPrimary};
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

  // Unary metal_stdlib math.
  if (StringRef Fn = mslUnaryFunction(Op, CI->getType()->isFloatingPointTy());
      !Fn.empty()) {
    auto A = exprFor(CI->getArgOperand(1), PrecUnknown);
    if (!A)
      return A.takeError();
    return Expr{(Fn + "(" + A->Text + ")").str(), PrecPrimary};
  }

  // Binary min/max family.
  if (StringRef Fn = mslBinaryFunction(Op); !Fn.empty()) {
    bool Signed = Op == DXILOp::IMax || Op == DXILOp::IMin;
    auto A = exprFor(CI->getArgOperand(1), PrecUnknown);
    if (!A)
      return A.takeError();
    auto B = exprFor(CI->getArgOperand(2), PrecUnknown);
    if (!B)
      return B.takeError();
    if (Signed)
      return Expr{"as_type<uint>(" + Fn.str() + "(as_type<int>(" + A->Text +
                      "), as_type<int>(" + B->Text + ")))",
                  PrecPrimary};
    return Expr{Fn.str() + "(" + A->Text + ", " + B->Text + ")", PrecPrimary};
  }

  // Mad family.
  if (Op == DXILOp::FMad || Op == DXILOp::Fma || Op == DXILOp::IMad ||
      Op == DXILOp::UMad) {
    auto A = exprFor(CI->getArgOperand(1), PrecUnknown);
    if (!A)
      return A.takeError();
    auto B = exprFor(CI->getArgOperand(2), PrecUnknown);
    if (!B)
      return B.takeError();
    auto C = exprFor(CI->getArgOperand(3), PrecUnknown);
    if (!C)
      return C.takeError();
    if (Op == DXILOp::FMad || Op == DXILOp::Fma)
      return Expr{"fma(" + A->Text + ", " + B->Text + ", " + C->Text + ")",
                  PrecPrimary};
    return Expr{A->Text + " * " + B->Text + " + " + C->Text, PrecAdd};
  }

  return makeError("unsupported DXIL op {0}", static_cast<uint32_t>(Op));
}

Expected<std::string> Emitter::bufferIndexExpr(ResState &RS,
                                               const Value *Coord0,
                                               const Value *Coord1,
                                               unsigned Comp) {
  auto IsUndef = [](const Value *V) {
    return isa<UndefValue>(V) || isa<PoisonValue>(V);
  };

  std::string Index;
  uint64_t ConstOffset = 0; // in scalar elements

  switch (RS.RB->Kind) {
  case ResourceKind::StructuredBuffer: {
    uint32_t Stride = RS.RB->ElementStride;
    if (Stride == 0 || Stride % 4 != 0)
      return makeError("structured buffer '{0}' has unsupported stride {1}",
                       RS.ParamName, Stride);
    uint32_t K = Stride / 4;
    auto E = exprFor(Coord0, K > 1 ? PrecMul : PrecAdd);
    if (!E)
      return E.takeError();
    Index = E->Text;
    if (K > 1)
      Index += " * " + std::to_string(K) + "u";
    if (!IsUndef(Coord1)) {
      if (const auto *COff = dyn_cast<ConstantInt>(Coord1))
        ConstOffset += COff->getZExtValue() / 4;
      else {
        auto Off = exprFor(Coord1, PrecShift);
        if (!Off)
          return Off.takeError();
        Index += " + (" + Off->Text + " >> 2u)";
      }
    }
    break;
  }
  case ResourceKind::RawBuffer: {
    // ByteAddressBuffer: coord0 is a byte offset.
    if (const auto *CByte = dyn_cast<ConstantInt>(Coord0)) {
      Index = std::to_string(CByte->getZExtValue() / 4) + "u";
    } else {
      auto E = exprFor(Coord0, PrecShift);
      if (!E)
        return E.takeError();
      Index = "(" + E->Text + " >> 2u)";
    }
    break;
  }
  case ResourceKind::TypedBuffer: {
    auto E = exprFor(Coord0, PrecAdd);
    if (!E)
      return E.takeError();
    Index = E->Text;
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

Expected<Expr> Emitter::bufferElementExpr(const ExtractValueInst *EV) {
  const auto *CI = dyn_cast<CallInst>(EV->getAggregateOperand());
  if (!CI)
    return makeError("extractvalue from a non-resource aggregate");
  auto Op = dxOpcode(CI);
  if (!Op)
    return makeError("extractvalue from unsupported call");
  unsigned Comp = EV->getIndices()[0];

  if (*Op == DXILOp::BufferLoad) {
    if (Comp > 3)
      return makeError("buffer load status queries are not supported yet");
    ResState *RS = stateFor(CI->getArgOperand(1));
    if (!RS)
      return makeError("buffer load through unresolved handle");
    auto Index = bufferIndexExpr(*RS, CI->getArgOperand(2),
                                 CI->getArgOperand(3), Comp);
    if (!Index)
      return Index.takeError();
    return Expr{RS->ParamName + "[" + *Index + "]", PrecPrimary};
  }

  if (*Op == DXILOp::CBufferLoadLegacy) {
    ResState *RS = stateFor(CI->getArgOperand(1));
    if (!RS)
      return makeError("cbuffer load through unresolved handle");
    auto Row = exprFor(CI->getArgOperand(2), PrecUnknown);
    if (!Row)
      return Row.takeError();
    return Expr{RS->ParamName + "[" + Row->Text + "]." +
                    std::string(1, "xyzw"[Comp & 3]),
                PrecPrimary};
  }

  return makeError("unsupported aggregate producer for extractvalue");
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

Error Emitter::emitBufferStore(const CallInst *CI) {
  // (op, handle, coord0, coord1, v0, v1, v2, v3, mask:i8)
  ResState *RS = stateFor(CI->getArgOperand(1));
  if (!RS)
    return makeError("buffer store through unresolved handle");
  const auto *Mask = dyn_cast<ConstantInt>(CI->getArgOperand(8));
  if (!Mask)
    return makeError("dynamic store mask is not supported");
  uint64_t M = Mask->getZExtValue();
  for (unsigned Comp = 0; Comp < 4; ++Comp) {
    if (!(M & (1u << Comp)))
      continue;
    auto Index = bufferIndexExpr(*RS, CI->getArgOperand(2),
                                 CI->getArgOperand(3), Comp);
    if (!Index)
      return Index.takeError();
    auto V = exprFor(CI->getArgOperand(4 + Comp), PrecUnknown);
    if (!V)
      return V.takeError();
    Body += "  " + RS->ParamName + "[" + *Index + "] = " + V->Text + ";\n";
  }
  return Error::success();
}

Error Emitter::emitBarrier(const CallInst *CI) {
  const auto *Mode = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  if (!Mode)
    return makeError("dynamic barrier mode is not supported");
  uint64_t M = Mode->getZExtValue();
  // DXIL barrier modes: 1 = sync threadgroup, 2 = UAV fence global,
  // 4 = UAV fence threadgroup, 8 = TGSM fence.
  std::string Flags;
  if (M & 8)
    Flags = "mem_flags::mem_threadgroup";
  if (M & (2 | 4))
    Flags = Flags.empty() ? "mem_flags::mem_device"
                          : Flags + " | mem_flags::mem_device";
  if (Flags.empty())
    Flags = "mem_flags::mem_none";
  Body += "  threadgroup_barrier(" + Flags + ");\n";
  return Error::success();
}

Error Emitter::emitStatement(const Instruction *I) {
  if (isa<ReturnInst>(I))
    return Error::success(); // kernel epilogue; nothing to print

  if (const auto *CI = dyn_cast<CallInst>(I)) {
    auto Op = dxOpcode(CI);
    if (!Op)
      return makeError("call to unsupported function '{0}'",
                       CI->getCalledFunction()
                           ? CI->getCalledFunction()->getName()
                           : "<indirect>");
    switch (*Op) {
    case DXILOp::CreateHandle:
    case DXILOp::CreateHandleFromBinding:
    case DXILOp::AnnotateHandle:
      return Error::success(); // folded into bindings
    case DXILOp::BufferStore:
      return emitBufferStore(CI);
    case DXILOp::Barrier:
      return emitBarrier(CI);
    case DXILOp::BufferLoad:
    case DXILOp::CBufferLoadLegacy:
      return Error::success(); // materialized at extractvalue sites
    default:
      break;
    }
  }

  // Pure value: inline single-use values; declare temporaries otherwise.
  if (shouldInline(I))
    return Error::success();
  if (I->use_empty())
    return Error::success(); // dead
  auto TyOrErr = mslType(I->getType());
  if (!TyOrErr)
    return TyOrErr.takeError();
  auto E = instExpr(I);
  if (!E)
    return E.takeError();
  std::string Name = "v" + std::to_string(NextTemp++);
  Body += "  " + *TyOrErr + " " + Name + " = " + E->Text + ";\n";
  ValueNames[I] = Name;
  return Error::success();
}

Error Emitter::emitBody(Function &F) {
  if (F.size() != 1)
    return makeError("control flow structurization is not implemented yet "
                     "(entry has {0} basic blocks)",
                     F.size());
  for (const Instruction &I : F.getEntryBlock())
    if (Error E = emitStatement(&I))
      return E;
  return Error::success();
}

//===----------------------------------------------------------------------===//
// Kernel signature
//===----------------------------------------------------------------------===//

Expected<std::string> Emitter::kernelSignature() {
  const Reflection &R = Shader.Reflect;
  if (R.Stage != ShaderStage::Compute)
    return makeError("only compute shaders are supported in Milestone 1");

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
  OS << ")";
  return Sig;
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

Expected<std::string> Emitter::run() {
  Function &F = *Shader.Entry;
  Reflection &R = Shader.Reflect;
  R.MSLEntryName = sanitizeIdentifier(R.EntryName, "kernel_main");

  if (Error E = resolveHandles(F))
    return std::move(E);
  if (Error E = inferElementTypes(F))
    return std::move(E);
  if (Error E = emitBody(F))
    return std::move(E);

  auto Sig = kernelSignature();
  if (!Sig)
    return Sig.takeError();

  std::string Out;
  raw_string_ostream OS(Out);
  OS << "// Generated by crucible from DXIL (" << R.EntryName << ", cs_"
     << R.ShaderModelMajor << "_" << R.ShaderModelMinor << ")\n"
     << "#include <metal_stdlib>\n"
     << "using namespace metal;\n\n"
     << *Sig << " {\n"
     << Body << "}\n";
  return Out;
}

} // namespace

Expected<std::string> emitMSL(LoadedShader &Shader, const Options &Opts) {
  return Emitter(Shader, Opts).run();
}

} // namespace crucible

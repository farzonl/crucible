//===- main.cpp - cruciblec CLI -----------------------------------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// cruciblec input.dxil -o output.metal [--reflect out.json]
//                      [--metallib out.metallib]
//
//===----------------------------------------------------------------------===//

#include "crucible/Compiler.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::OptionCategory CrucibleCat("cruciblec options");

static cl::opt<std::string> InputFile(cl::Positional, cl::Required,
                                      cl::desc("<input .dxil>"),
                                      cl::cat(CrucibleCat));
static cl::opt<std::string> OutputFile("o", cl::desc("Output file (.metal or .spv, defaults to stdout)"),
                                       cl::value_desc("path"),
                                       cl::cat(CrucibleCat));
static cl::opt<std::string> TargetOpt("target",
                                      cl::desc("Target backend: msl (default) or spirv"),
                                      cl::init("msl"), cl::cat(CrucibleCat));
static cl::opt<bool> EmitSPIRVAsm("spv-asm",
                                  cl::desc("Emit SPIR-V assembly text instead of binary words (--target=spirv)"),
                                  cl::cat(CrucibleCat));
static cl::opt<std::string> VulkanVersion("vulkan-version",
                                          cl::desc("Vulkan target environment for SPIR-V backend (default vulkan1.3)"),
                                          cl::init("vulkan1.3"), cl::cat(CrucibleCat));
static cl::list<std::string> VkBShifts("vk-b-shift",
                                       cl::desc("Vulkan CBV register shift: <shift> or <shift>,<space>"),
                                       cl::cat(CrucibleCat));
static cl::list<std::string> VkTShifts("vk-t-shift",
                                       cl::desc("Vulkan SRV register shift: <shift> or <shift>,<space>"),
                                       cl::cat(CrucibleCat));
static cl::list<std::string> VkUShifts("vk-u-shift",
                                       cl::desc("Vulkan UAV register shift: <shift> or <shift>,<space>"),
                                       cl::cat(CrucibleCat));
static cl::list<std::string> VkSShifts("vk-s-shift",
                                       cl::desc("Vulkan Sampler register shift: <shift> or <shift>,<space>"),
                                       cl::cat(CrucibleCat));
static cl::opt<std::string> ReflectFile("reflect",
                                        cl::desc("Write binding reflection JSON"),
                                        cl::value_desc("path"),
                                        cl::cat(CrucibleCat));
static cl::opt<std::string> MetallibFile("metallib",
                                         cl::desc("Compile the MSL to a .metallib via xcrun"),
                                         cl::value_desc("path"),
                                         cl::cat(CrucibleCat));
static cl::opt<std::string> EntryPoint("entry",
                                       cl::desc("Entry point to compile"),
                                       cl::value_desc("name"),
                                       cl::cat(CrucibleCat));
static cl::opt<std::string> MSLStd("msl-std",
                                   cl::desc("MSL language standard (default metal3.1)"),
                                   cl::init("metal3.1"), cl::cat(CrucibleCat));
static cl::opt<bool> FlushDenormsF32("ftz",
                                     cl::desc("Flush f32 denormals to zero around math builtins"),
                                     cl::cat(CrucibleCat));
static cl::opt<bool> EmulateF64("emulate-f64",
                                cl::desc("Emulate double (f64) operations using software IEEE-754 binary64"),
                                cl::cat(CrucibleCat));

static int reportError(Error E) {
  logAllUnhandledErrors(std::move(E), errs(), "cruciblec: error: ");
  return 1;
}

static int reportError(const Twine &Msg) {
  errs() << "cruciblec: error: " << Msg << "\n";
  return 1;
}

static Error writeFile(StringRef Path, StringRef Contents) {
  std::error_code EC;
  raw_fd_ostream OS(Path, EC, sys::fs::OF_None);
  if (EC)
    return createStringError(EC, "cannot write '%s'", Path.str().c_str());
  OS << Contents;
  return Error::success();
}

// Invoke `xcrun -sdk macosx metal` and `metallib`, surfacing stderr so bad
// MSL emission fails loudly with the Apple compiler's diagnostics.
static Error buildMetallib(StringRef MSLPath, StringRef OutPath,
                           StringRef Std) {
  auto Xcrun = sys::findProgramByName("xcrun");
  if (!Xcrun)
    return createStringError(Xcrun.getError(),
                             "xcrun not found; install Xcode command line tools");

  SmallString<128> AirPath(OutPath);
  sys::path::replace_extension(AirPath, ".air");

  std::string StdFlag = (Twine("-std=") + Std).str();
  std::string ErrMsg;

  std::vector<StringRef> MetalArgs = {*Xcrun,  "-sdk",   "macosx", "metal",
                                      StdFlag, "-Werror", "-O2",   "-c",
                                      MSLPath, "-o",      AirPath};
  if (sys::ExecuteAndWait(*Xcrun, MetalArgs, std::nullopt, {}, 0, 0, &ErrMsg))
    return createStringError(inconvertibleErrorCode(),
                             "metal front-end failed: %s", ErrMsg.c_str());

  std::vector<StringRef> LibArgs = {*Xcrun, "-sdk", "macosx", "metallib",
                                    AirPath, "-o", OutPath};
  if (sys::ExecuteAndWait(*Xcrun, LibArgs, std::nullopt, {}, 0, 0, &ErrMsg))
    return createStringError(inconvertibleErrorCode(),
                             "metallib failed: %s", ErrMsg.c_str());
  return Error::success();
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::HideUnrelatedOptions(CrucibleCat);
  cl::ParseCommandLineOptions(
      argc, argv, "crucible: DXIL to Metal Shading Language compiler\n");

  auto BufOrErr = MemoryBuffer::getFileOrSTDIN(InputFile);
  if (!BufOrErr)
    return reportError("cannot read '" + InputFile + "': " +
                       BufOrErr.getError().message());

  crucible::Options Opts;
  Opts.EntryPoint = EntryPoint;
  Opts.MSLStd = MSLStd;
  Opts.FlushDenormsF32 = FlushDenormsF32;
  Opts.EmulateF64 = EmulateF64;
  Opts.EmitSPIRVAsm = EmitSPIRVAsm;
  Opts.VulkanVersion = VulkanVersion;

  if (TargetOpt == "spirv") {
    Opts.Target = crucible::TargetBackend::SPIRV;
    auto parseShifts = [&](const cl::list<std::string> &List,
                           crucible::ResourceClass RC) -> Error {
      for (const std::string &Spec : List) {
        StringRef S(Spec);
        auto [ShiftStr, SpaceStr] = S.split(',');
        uint32_t ShiftVal = 0;
        if (ShiftStr.getAsInteger(10, ShiftVal))
          return createStringError(inconvertibleErrorCode(),
                                   "invalid shift value '%s'", Spec.c_str());
        crucible::VulkanRegisterShift Entry;
        Entry.Class = RC;
        Entry.Shift = ShiftVal;
        if (!SpaceStr.empty()) {
          uint32_t SpaceVal = 0;
          if (SpaceStr.getAsInteger(10, SpaceVal))
            return createStringError(inconvertibleErrorCode(),
                                     "invalid space value in '%s'", Spec.c_str());
          Entry.Space = SpaceVal;
          Entry.AllSpaces = false;
        } else {
          Entry.AllSpaces = true;
        }
        Opts.VulkanRegisterShifts.push_back(Entry);
      }
      return Error::success();
    };
    if (Error E = parseShifts(VkBShifts, crucible::ResourceClass::CBV))
      return reportError(std::move(E));
    if (Error E = parseShifts(VkTShifts, crucible::ResourceClass::SRV))
      return reportError(std::move(E));
    if (Error E = parseShifts(VkUShifts, crucible::ResourceClass::UAV))
      return reportError(std::move(E));
    if (Error E = parseShifts(VkSShifts, crucible::ResourceClass::Sampler))
      return reportError(std::move(E));

    if (!MetallibFile.empty())
      return reportError("--metallib cannot be used with --target=spirv");

    auto Result =
        crucible::compileDXILToSPIRV((*BufOrErr)->getMemBufferRef(), Opts);
    if (!Result)
      return reportError(Result.takeError());

    if (OutputFile.empty() || OutputFile == "-") {
      outs().write(Result->Output.data(), Result->Output.size());
    } else if (Error E = writeFile(OutputFile, Result->Output)) {
      return reportError(std::move(E));
    }

    if (!ReflectFile.empty())
      if (Error E = writeFile(ReflectFile, Result->Reflect.toJSON()))
        return reportError(std::move(E));
    return 0;
  }

  if (TargetOpt != "msl")
    return reportError("unknown --target '" + TargetOpt +
                       "' (expected 'msl' or 'spirv')");

  auto Result =
      crucible::compileDXILToMSL((*BufOrErr)->getMemBufferRef(), Opts);
  if (!Result)
    return reportError(Result.takeError());

  if (OutputFile.empty() || OutputFile == "-") {
    outs() << Result->MSL;
  } else if (Error E = writeFile(OutputFile, Result->MSL)) {
    return reportError(std::move(E));
  }

  if (!ReflectFile.empty())
    if (Error E = writeFile(ReflectFile, Result->Reflect.toJSON()))
      return reportError(std::move(E));

  if (!MetallibFile.empty()) {
    if (OutputFile.empty() || OutputFile == "-")
      return reportError("--metallib requires -o <file.metal>");
    if (Error E = buildMetallib(OutputFile, MetallibFile, MSLStd))
      return reportError(std::move(E));
  }
  return 0;
}

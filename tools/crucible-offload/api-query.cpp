// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Support/HostInfo.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/raw_ostream.h"

int main() {
  const llvm::ExitOnError ExitOnErr("crucible-api-query: error: ");
  const auto Host = ExitOnErr(offloadtest::HostInfo::query());
  auto &OS = llvm::outs();
  OS << "Host:\n";
  auto PrintField = [&](llvm::StringRef Key, llvm::StringRef Value) {
    OS << "  " << Key << ": \"" << llvm::yaml::escape(Value, false) << "\"\n";
  };
  PrintField("Operating System", Host.OperatingSystem);
  PrintField("OS Version", Host.OSVersion);
  PrintField("Architecture", Host.Architecture);
}

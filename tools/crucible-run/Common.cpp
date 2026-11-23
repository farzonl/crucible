//===- Common.cpp - Shared helpers for crucible-run --------------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL/SPIR-V compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace crucible_run {

[[noreturn]] void fail(const std::string &Msg) {
  fprintf(stderr, "crucible-run: error: %s\n", Msg.c_str());
  exit(1);
}

bool parseTriple(const char *S, uint32_t Out[3]) {
  return sscanf(S, "%u,%u,%u", &Out[0], &Out[1], &Out[2]) == 3;
}

std::vector<std::string> split(const std::string &S, char Sep) {
  std::vector<std::string> Parts;
  size_t Start = 0;
  while (true) {
    size_t Pos = S.find(Sep, Start);
    Parts.push_back(S.substr(Start, Pos - Start));
    if (Pos == std::string::npos)
      break;
    Start = Pos + 1;
  }
  return Parts;
}

BufferSpec parseBufferSpec(const std::string &Arg, bool IsUniform) {
  auto Parts = split(Arg, ':');
  if (Parts.size() != 4)
    fail("bad buffer spec '" + Arg + "' (want index:type:count:init)");
  BufferSpec Spec;
  Spec.Index = static_cast<uint32_t>(strtoul(Parts[0].c_str(), nullptr, 10));
  Spec.Type = Parts[1];
  Spec.Count = strtoull(Parts[2].c_str(), nullptr, 10);
  Spec.Init = Parts[3];
  Spec.IsUniform = IsUniform;
  if (Spec.Type != "f32" && Spec.Type != "u32" && Spec.Type != "i32")
    fail("unsupported buffer type '" + Spec.Type + "'");
  if (Spec.Count == 0)
    fail("buffer count must be nonzero");
  return Spec;
}

void fillBuffer(const BufferSpec &Spec, void *Data) {
  std::vector<std::string> Values;
  bool Iota = Spec.Init == "iota";
  bool Zero = Spec.Init == "zero";
  if (!Iota && !Zero)
    Values = split(Spec.Init, ',');
  for (size_t I = 0; I < Spec.Count; ++I) {
    // "0x"-prefixed values are raw 32-bit patterns regardless of buffer type
    // (useful for packing uint fields into float-typed cbuffers).
    if (!Iota && !Zero && I < Values.size() &&
        Values[I].rfind("0x", 0) == 0) {
      uint32_t Bits =
          static_cast<uint32_t>(strtoul(Values[I].c_str(), nullptr, 16));
      memcpy(static_cast<uint32_t *>(Data) + I, &Bits, 4);
      continue;
    }
    double V = 0;
    if (Iota)
      V = static_cast<double>(I);
    else if (!Zero)
      V = I < Values.size() ? strtod(Values[I].c_str(), nullptr) : 0;
    if (Spec.Type == "f32")
      static_cast<float *>(Data)[I] = static_cast<float>(V);
    else if (Spec.Type == "u32")
      static_cast<uint32_t *>(Data)[I] = static_cast<uint32_t>(V);
    else
      static_cast<int32_t *>(Data)[I] = static_cast<int32_t>(V);
  }
}

void printBuffer(const BufferSpec &Spec, const void *Data) {
  printf("buffer%u:", Spec.Index);
  for (size_t I = 0; I < Spec.Count; ++I) {
    if (Spec.Type == "f32")
      printf(" %g", static_cast<const float *>(Data)[I]);
    else if (Spec.Type == "u32")
      printf(" %u", static_cast<const uint32_t *>(Data)[I]);
    else
      printf(" %d", static_cast<const int32_t *>(Data)[I]);
  }
  printf("\n");
}

} // namespace crucible_run

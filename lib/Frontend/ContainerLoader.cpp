//===- ContainerLoader.cpp - DXContainer extraction -------------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Extracts the DXIL bitcode part from a DXContainer (.dxil) binary using
// llvm::object::DXContainer. Bare bitcode buffers pass straight through so
// unit tests can feed IR without re-wrapping it in a container.
//
//===----------------------------------------------------------------------===//

#include "Frontend.h"
#include "crucible/Diagnostic.h"

#include "llvm/Object/DXContainer.h"

using namespace llvm;

namespace crucible {

static bool isBitcode(StringRef Data) {
  return Data.starts_with("BC\xC0\xDE");
}

static bool isDXContainer(StringRef Data) { return Data.starts_with("DXBC"); }

Expected<StringRef> extractDXILBitcode(MemoryBufferRef Buf) {
  StringRef Data = Buf.getBuffer();
  if (isBitcode(Data))
    return Data;

  if (!isDXContainer(Data))
    return makeError("'{0}': not a DXContainer or LLVM bitcode file",
                     Buf.getBufferIdentifier());

  auto Container = object::DXContainer::create(Buf);
  if (!Container)
    return Container.takeError();

  const auto &DXIL = Container->getDXIL(/*Debug=*/false);
  if (!DXIL)
    return makeError("'{0}': DXContainer has no DXIL part",
                     Buf.getBufferIdentifier());

  const dxbc::ProgramHeader &PH = DXIL->first;
  return StringRef(DXIL->second, PH.Bitcode.Size);
}

} // namespace crucible

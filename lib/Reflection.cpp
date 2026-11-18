//===- Reflection.cpp - Binding reflection JSON ------------------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "crucible/Reflection.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace crucible {

static const char *stageName(ShaderStage S) {
  switch (S) {
  case ShaderStage::Pixel:
    return "pixel";
  case ShaderStage::Vertex:
    return "vertex";
  case ShaderStage::Geometry:
    return "geometry";
  case ShaderStage::Hull:
    return "hull";
  case ShaderStage::Domain:
    return "domain";
  case ShaderStage::Compute:
    return "compute";
  case ShaderStage::Library:
    return "library";
  case ShaderStage::Mesh:
    return "mesh";
  case ShaderStage::Amplification:
    return "amplification";
  default:
    return "unknown";
  }
}

static const char *className(ResourceClass C) {
  switch (C) {
  case ResourceClass::SRV:
    return "srv";
  case ResourceClass::UAV:
    return "uav";
  case ResourceClass::CBV:
    return "cbv";
  case ResourceClass::Sampler:
    return "sampler";
  }
  return "unknown";
}

std::string Reflection::toJSON() const {
  json::Array Bindings;
  for (const ResourceBinding &RB : Resources) {
    Bindings.push_back(json::Object{
        {"class", className(RB.Class)},
        {"kind", static_cast<int64_t>(RB.Kind)},
        {"name", RB.Name},
        {"space", static_cast<int64_t>(RB.Space)},
        {"register", static_cast<int64_t>(RB.LowerBound)},
        {"rangeSize", static_cast<int64_t>(RB.RangeSize)},
        {"elementStride", static_cast<int64_t>(RB.ElementStride)},
        {"metalIndex", static_cast<int64_t>(RB.MetalIndex)},
        {"requiresTextureAtomics", RB.RequiresTextureAtomics},
    });
  }

  json::Object Root{
      {"stage", stageName(Stage)},
      {"shaderModel", json::Object{{"major", static_cast<int64_t>(ShaderModelMajor)},
                                   {"minor", static_cast<int64_t>(ShaderModelMinor)}}},
      {"entry", EntryName},
      {"mslEntry", MSLEntryName},
      {"numThreads", json::Array{static_cast<int64_t>(NumThreads[0]),
                                 static_cast<int64_t>(NumThreads[1]),
                                 static_cast<int64_t>(NumThreads[2])}},
      {"bindings", std::move(Bindings)},
  };

  std::string Out;
  raw_string_ostream OS(Out);
  OS << formatv("{0:2}", json::Value(std::move(Root)));
  return Out;
}

} // namespace crucible

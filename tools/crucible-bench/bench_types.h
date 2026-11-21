//===- bench_types.h - Nacelle benchmark catalog & types ----------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Ported from Nacelle's tools/nacelle-bench/common/bench_types.h and
// bench_common.h so Crucible and metal-shaderconverter run the exact same
// workloads, inputs, dispatch sizes, and checksums.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace crucible_bench {

enum class BenchCategory {
  Compute,
  Vertex,
  Pixel,
  Geometry,
  Tessellation,
  RayTracing,
  Mesh,
};

inline const char *categoryName(BenchCategory c) {
  switch (c) {
  case BenchCategory::Compute:
    return "Compute";
  case BenchCategory::Vertex:
    return "Vertex";
  case BenchCategory::Pixel:
    return "Pixel";
  case BenchCategory::Geometry:
    return "Geometry";
  case BenchCategory::Tessellation:
    return "Tessellation";
  case BenchCategory::RayTracing:
    return "Ray Tracing";
  case BenchCategory::Mesh:
    return "Mesh";
  }
  return "Unknown";
}

enum class PipelineKind {
  Compute,
  GraphicsVSPS,
  GraphicsVSGSPS,
  GraphicsTessGS,
  GraphicsMSPS,
  RayTracingInline,
  RayTracingPipeline,
};

struct BenchmarkSpec {
  std::string id;
  std::string name;
  BenchCategory category = BenchCategory::Compute;
  PipelineKind kind = PipelineKind::Compute;
  std::string entryOrProfile;
  std::string csDxil;
  std::string vsDxil;
  std::string hsDxil;
  std::string dsDxil;
  std::string gsDxil;
  std::string asDxil;
  std::string msDxil;
  std::string psDxil;
  std::string rtDxil;
  bool hasTexture = false;
};

struct MetalBenchResult {
  std::string benchId;
  std::string benchName;
  std::string category;
  std::string backend;
  std::string shaderFormat;
  uint32_t simdWidth = 32;
  double compileMs = 0.0;
  double firstDispatchMs = 0.0;
  double coldMs = 0.0;
  double meanMs = 0.0;
  double medianMs = 0.0;
  double minMs = 0.0;
  double shortJobMs = 0.0;
  double longJobMs = 0.0;
  double checksum = 0.0;
  bool ran = false;
  std::string status = "SKIPPED";
};

inline double computeMedian(std::vector<double> vals) {
  if (vals.empty())
    return 0.0;
  std::sort(vals.begin(), vals.end());
  size_t mid = vals.size() / 2;
  if (vals.size() % 2 == 0)
    return 0.5 * (vals[mid - 1] + vals[mid]);
  return vals[mid];
}

inline void finalizeSamples(MetalBenchResult &res,
                            const std::vector<double> &samples) {
  if (samples.empty())
    return;
  double sum = 0.0;
  for (double s : samples)
    sum += s;
  res.meanMs = sum / static_cast<double>(samples.size());
  res.medianMs = computeMedian(samples);
  res.minMs = *std::min_element(samples.begin(), samples.end());
  res.coldMs = res.compileMs + res.firstDispatchMs;
}

inline bool readBinary(const std::string &path, std::vector<uint8_t> &out) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs)
    return false;
  auto sz = ifs.tellg();
  if (sz <= 0)
    return false;
  out.resize(static_cast<size_t>(sz));
  ifs.seekg(0, std::ios::beg);
  return static_cast<bool>(ifs.read(reinterpret_cast<char *>(out.data()), sz));
}

inline std::string jsonEscape(const std::string &s) {
  std::ostringstream os;
  for (char c : s) {
    if (c == '\\')
      os << "\\\\";
    else if (c == '"')
      os << "\\\"";
    else if (c == '\n')
      os << "\\n";
    else if (c == '\r')
      os << "\\r";
    else if (c == '\t')
      os << "\\t";
    else
      os << c;
  }
  return os.str();
}

struct RaymarchCB {
  uint32_t width;
  uint32_t height;
  uint32_t maxSteps;
  float time;
};

struct MatmulCB {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t pad;
};

struct ShadertoyCB {
  float resX, resY, resZ;
  float iTime;
  float mouseX, mouseY, mouseZ, mouseW;
};

struct VertexStressCB {
  float mvp[16];
  float time;
  uint32_t vertexCount;
  float pad[2];
};

struct GeomStressCB {
  float scale;
  float time;
  uint32_t pointCount;
  float pad;
};

struct MeshBenchVertex {
  float px, py, pz;
  int16_t nx, ny, nz, nw;
  uint16_t u, v;
  uint8_t r, g, b, a;
  uint8_t j0, j1, j2, j3;
};
static_assert(sizeof(MeshBenchVertex) == 32, "MeshBenchVertex must be 32 bytes");

struct alignas(16) MeshBenchConstants {
  float viewProj[16];
  float boneRows[12][4];
  float lightDirAndTime[4];
  uint8_t pad[512 - (16 + 48 + 4) * sizeof(float)];
};
static_assert(sizeof(MeshBenchConstants) == 512,
              "MeshBenchConstants must be 512 bytes");

struct MeshBenchSceneData {
  MeshBenchConstants constants{};
  std::vector<MeshBenchVertex> vertices;
  std::vector<uint32_t> indices;
  uint32_t numTriangles = 0;
};

inline int16_t packMeshSnorm16(float v) {
  float c = std::clamp(v, -1.0f, 1.0f);
  return static_cast<int16_t>(std::lrintf(c * 32767.0f));
}

inline uint16_t packMeshUnorm16(float v) {
  float c = std::clamp(v, 0.0f, 1.0f);
  return static_cast<uint16_t>(std::lrintf(c * 65535.0f));
}

inline uint8_t packMeshUnorm8(float v) {
  float c = std::clamp(v, 0.0f, 1.0f);
  return static_cast<uint8_t>(std::lrintf(c * 255.0f));
}

inline void getMeshBenchGridAndDims(int mode, uint32_t baseW, uint32_t baseH,
                                    uint32_t &outGrid, uint32_t &outRW,
                                    uint32_t &outRH) {
  const bool isQuick = (baseW <= 128);
  if (mode == 0) {
    outGrid = 40;
    outRW = 128;
    outRH = 128;
  } else if (mode == 1) {
    outGrid = isQuick ? 80 : 316;
    outRW = baseW;
    outRH = baseH;
  } else {
    outGrid = isQuick ? 160 : 448;
    outRW = baseW * 2;
    outRH = baseH * 2;
  }
}

inline MeshBenchSceneData buildMeshBenchScene(uint32_t gridX, uint32_t gridY) {
  MeshBenchSceneData scene;
  scene.numTriangles = gridX * gridY * 2u;

  float vp[16] = {
      0.92f, 0.00f, 0.00f, 0.0f, 0.00f, 0.92f, 0.00f, 0.0f,
      0.00f, 0.00f, 0.50f, 0.0f, 0.00f, 0.00f, 0.25f, 1.0f,
  };
  std::memcpy(scene.constants.viewProj, vp, sizeof(vp));

  const float angles[4] = {0.06f, -0.08f, 0.05f, -0.04f};
  const float tx[4] = {0.02f, -0.02f, 0.01f, -0.01f};
  const float ty[4] = {-0.01f, 0.02f, -0.02f, 0.01f};
  const float tz[4] = {0.00f, 0.03f, -0.02f, 0.02f};
  for (uint32_t b = 0; b < 4; ++b) {
    float cs = std::cos(angles[b]);
    float sn = std::sin(angles[b]);
    scene.constants.boneRows[b * 3 + 0][0] = cs;
    scene.constants.boneRows[b * 3 + 0][1] = -sn;
    scene.constants.boneRows[b * 3 + 0][2] = 0.0f;
    scene.constants.boneRows[b * 3 + 0][3] = tx[b];

    scene.constants.boneRows[b * 3 + 1][0] = sn;
    scene.constants.boneRows[b * 3 + 1][1] = cs;
    scene.constants.boneRows[b * 3 + 1][2] = 0.0f;
    scene.constants.boneRows[b * 3 + 1][3] = ty[b];

    scene.constants.boneRows[b * 3 + 2][0] = 0.0f;
    scene.constants.boneRows[b * 3 + 2][1] = 0.0f;
    scene.constants.boneRows[b * 3 + 2][2] = 1.0f;
    scene.constants.boneRows[b * 3 + 2][3] = tz[b];
  }
  scene.constants.lightDirAndTime[0] = 0.45f;
  scene.constants.lightDirAndTime[1] = 0.65f;
  scene.constants.lightDirAndTime[2] = 0.61f;
  scene.constants.lightDirAndTime[3] = 0.35f;

  scene.vertices.resize(static_cast<size_t>(gridX + 1) * (gridY + 1));
  for (uint32_t y = 0; y <= gridY; ++y) {
    for (uint32_t x = 0; x <= gridX; ++x) {
      float u = static_cast<float>(x) / static_cast<float>(gridX);
      float v = static_cast<float>(y) / static_cast<float>(gridY);
      float px = u * 1.84f - 0.92f;
      float py = v * 1.84f - 0.92f;
      float pz = 0.35f + 0.12f * std::sin(u * 6.28318f) * std::cos(v * 6.28318f);

      float nx = -0.25f * std::cos(u * 6.28318f);
      float ny = -0.25f * std::sin(v * 6.28318f);
      float nz = 1.0f;
      float nLen = std::sqrt(nx * nx + ny * ny + nz * nz);
      nx /= nLen;
      ny /= nLen;
      nz /= nLen;

      MeshBenchVertex vert{};
      vert.px = px;
      vert.py = py;
      vert.pz = pz;
      vert.nx = packMeshSnorm16(nx);
      vert.ny = packMeshSnorm16(ny);
      vert.nz = packMeshSnorm16(nz);
      vert.nw = packMeshSnorm16(u - v);
      vert.u = packMeshUnorm16(u);
      vert.v = packMeshUnorm16(v);
      vert.r = packMeshUnorm8(0.25f + 0.65f * u);
      vert.g = packMeshUnorm8(0.30f + 0.55f * v);
      vert.b = packMeshUnorm8(0.85f - 0.45f * u);
      vert.a = packMeshUnorm8(0.20f + 0.70f * (0.5f * (u + v)));
      uint8_t baseJoint = static_cast<uint8_t>((x + y) & 3u);
      vert.j0 = baseJoint;
      vert.j1 = static_cast<uint8_t>((baseJoint + 1u) & 3u);
      vert.j2 = static_cast<uint8_t>((baseJoint + 2u) & 3u);
      vert.j3 = static_cast<uint8_t>((baseJoint + 3u) & 3u);
      scene.vertices[static_cast<size_t>(y) * (gridX + 1) + x] = vert;
    }
  }

  scene.indices.reserve(static_cast<size_t>(gridX) * gridY * 6u);
  for (uint32_t y = 0; y < gridY; ++y) {
    for (uint32_t x = 0; x < gridX; ++x) {
      uint32_t i0 = y * (gridX + 1) + x;
      uint32_t i1 = i0 + 1;
      uint32_t i2 = i0 + (gridX + 1);
      uint32_t i3 = i2 + 1;
      if (((x + y) & 1u) == 0u) {
        scene.indices.push_back(i0);
        scene.indices.push_back(i2);
        scene.indices.push_back(i1);
        scene.indices.push_back(i1);
        scene.indices.push_back(i2);
        scene.indices.push_back(i3);
      } else {
        scene.indices.push_back(i0);
        scene.indices.push_back(i1);
        scene.indices.push_back(i2);
        scene.indices.push_back(i1);
        scene.indices.push_back(i3);
        scene.indices.push_back(i2);
      }
    }
  }
  return scene;
}

inline std::vector<uint32_t> buildProceduralTexture256Mip(uint32_t mip) {
  uint32_t mw = std::max(1u, 256u >> mip);
  uint32_t mh = std::max(1u, 256u >> mip);
  std::vector<uint32_t> pixels(static_cast<size_t>(mw) * mh);
  for (uint32_t y = 0; y < mh; ++y) {
    for (uint32_t x = 0; x < mw; ++x) {
      uint32_t r = ((x * 255u) / mw) ^ (((x >> 4) ^ (y >> 4)) & 1u ? 0x40u : 0u);
      uint32_t g = ((y * 255u) / mh);
      uint32_t b = (((x + y) * 127u) / std::max(1u, mw + mh));
      pixels[static_cast<size_t>(y) * mw + x] =
          (r & 0xFFu) | ((g & 0xFFu) << 8) | ((b & 0xFFu) << 16) | 0xFF000000u;
    }
  }
  return pixels;
}

inline std::vector<BenchmarkSpec> buildBenchmarkCatalog() {
  std::vector<BenchmarkSpec> cat;

  auto addCompute = [&](const std::string &id, const std::string &name,
                        const std::string &entry, const std::string &dxil) {
    BenchmarkSpec s;
    s.id = id;
    s.name = name;
    s.category = BenchCategory::Compute;
    s.kind = PipelineKind::Compute;
    s.entryOrProfile = entry;
    s.csDxil = dxil;
    cat.push_back(std::move(s));
  };

  addCompute("bench_raymarch", "Raymarcher (ALU+CF)", "main",
             "bench_raymarch.dxil");
  addCompute("matmul", "MatMul 128x128 (FMA+Mem)", "CSMain", "matmul.dxil");
  addCompute("tiled_matmul", "Tiled MatMul 128x128 (Shared+Mem)", "CSMain",
             "tiled_matmul.dxil");
  addCompute("tiled_matmul_large", "Tiled MatMul 512x512 (Shared+Mem)", "CSMain",
             "tiled_matmul.dxil");
  addCompute("linalg_matmul", "LinAlg MatMul 128x128 (CoopMat)", "CSMain",
             "linalg_matmul.dxil");
  addCompute("prefix_sum_groupshared", "Prefix Sum (Groupshared)", "CSMain",
             "prefix_sum_groupshared.dxil");
  addCompute("divergent_cf_atomics", "Divergent CF + Atomics", "CSMain",
             "divergent_cf_atomics.dxil");
  addCompute("wave_intrinsics", "Wave Intrinsics (Subgroup)", "CSMain",
             "wave_intrinsics.dxil");
  addCompute("gdeflate", "GDeflate Decompress (TileStream)", "CSMain",
             "gdeflate.dxil");
  addCompute("ethash_mine", "Ethash Mining (Keccak+DAG)", "main",
             "ethash_mine.dxil");
  addCompute("coalesced_uniform_cs", "Coalesced UnitStride + Uniform", "CSMain",
             "bench_coalesced_uniform.dxil");

  auto addVSPS = [&](const std::string &id, const std::string &name,
                     BenchCategory c, const std::string &vs,
                     const std::string &ps, bool hasTex = false) {
    BenchmarkSpec s;
    s.id = id;
    s.name = name;
    s.category = c;
    s.kind = PipelineKind::GraphicsVSPS;
    s.vsDxil = vs;
    s.psDxil = ps;
    s.hasTexture = hasTex;
    cat.push_back(std::move(s));
  };

  addVSPS("bench_vertex_stress", "Vertex Transform Stress",
          BenchCategory::Vertex, "bench_vertex_stress_vs.dxil",
          "bench_vertex_stress_ps.dxil");
  addVSPS("bench_small_tri_ps", "Small Triangle PS Stress",
          BenchCategory::Vertex, "bench_small_tri_ps_vs.dxil",
          "bench_small_tri_ps_ps.dxil");
  addVSPS("bench_vertex_mesh", "Skinned Mesh (VB+IB+Cull)",
          BenchCategory::Vertex, "bench_vertex_mesh_vs.dxil",
          "bench_vertex_mesh_ps.dxil");

  addVSPS("light_cycle", "Light Cycle", BenchCategory::Pixel,
          "fullscreen_quad_vs.dxil", "light_cycle.dxil");
  addVSPS("aurora_paint", "Aurora Paint", BenchCategory::Pixel,
          "fullscreen_quad_vs.dxil", "aurora_paint.dxil");
  addVSPS("sunflower_fields", "Sunflower Fields", BenchCategory::Pixel,
          "fullscreen_quad_vs.dxil", "sunflower_fields.dxil");
  addVSPS("mt_fuji_watercolor", "Mt Fuji Watercolor", BenchCategory::Pixel,
          "fullscreen_quad_vs.dxil", "mt_fuji_watercolor.dxil");
  addVSPS("disco_maze", "Disco Maze", BenchCategory::Pixel,
          "fullscreen_quad_vs.dxil", "disco_maze.dxil");
  addVSPS("aetherial_cathedral", "Aetherial Cathedral", BenchCategory::Pixel,
          "fullscreen_quad_vs.dxil", "aetherial_cathedral.dxil");
  addVSPS("state_heavy_ps", "State-Heavy PS (BCn+Blend+DS+MSAA)",
          BenchCategory::Pixel, "bench_state_heavy_ps_vs.dxil",
          "bench_state_heavy_ps_ps.dxil");
  addVSPS("textured_filter", "Textured Filter (8-tap)", BenchCategory::Pixel,
          "fullscreen_quad_vs.dxil", "textured_filter.dxil", true);
  addVSPS("textured_pbr_quad", "Textured PBR Quad", BenchCategory::Pixel,
          "fullscreen_quad_vs.dxil", "textured_pbr_quad.dxil", true);

  {
    BenchmarkSpec s;
    s.id = "bench_geometry_stress";
    s.name = "Geometry Star Expansion";
    s.category = BenchCategory::Geometry;
    s.kind = PipelineKind::GraphicsVSGSPS;
    s.vsDxil = "bench_geometry_stress_vs.dxil";
    s.gsDxil = "bench_geometry_stress_gs.dxil";
    s.psDxil = "bench_geometry_stress_ps.dxil";
    cat.push_back(std::move(s));
  }

  {
    BenchmarkSpec s;
    s.id = "bench_tess_tri";
    s.name = "Tessellation (VS+HS+DS+PS)";
    s.category = BenchCategory::Tessellation;
    s.kind = PipelineKind::GraphicsTessGS;
    s.vsDxil = "bench_tess_tri_vs.dxil";
    s.hsDxil = "bench_tess_tri_hs.dxil";
    s.dsDxil = "bench_tess_tri_ds.dxil";
    s.psDxil = "bench_tess_tri_ps.dxil";
    cat.push_back(std::move(s));
  }

  {
    BenchmarkSpec s;
    s.id = "tess_heavy";
    s.name = "Tessellation Heavy (16x VS+HS+DS+PS)";
    s.category = BenchCategory::Tessellation;
    s.kind = PipelineKind::GraphicsTessGS;
    s.vsDxil = "bench_tess_heavy_vs.dxil";
    s.hsDxil = "bench_tess_heavy_hs.dxil";
    s.dsDxil = "bench_tess_heavy_ds.dxil";
    s.psDxil = "bench_tess_heavy_ps.dxil";
    cat.push_back(std::move(s));
  }

  {
    BenchmarkSpec s;
    s.id = "tess_point_gs";
    s.name = "Tessellation (VS+HS+DS+GS+PS)";
    s.category = BenchCategory::Tessellation;
    s.kind = PipelineKind::GraphicsTessGS;
    s.vsDxil = "gfx_tess_point_vs.dxil";
    s.hsDxil = "gfx_tess_point_hs.dxil";
    s.dsDxil = "gfx_tess_point_ds.dxil";
    s.gsDxil = "gfx_tess_point_gs.dxil";
    s.psDxil = "gfx_tess_point_ps.dxil";
    cat.push_back(std::move(s));
  }

  {
    BenchmarkSpec s;
    s.id = "raytracing_inline";
    s.name = "RayTracing Inline (RayQuery)";
    s.category = BenchCategory::RayTracing;
    s.kind = PipelineKind::RayTracingInline;
    s.rtDxil = "raytracing_inline_cs.dxil";
    cat.push_back(std::move(s));
  }

  {
    BenchmarkSpec s;
    s.id = "raytracing_pipeline";
    s.name = "RayTracing Pipeline (DispatchRays)";
    s.category = BenchCategory::RayTracing;
    s.kind = PipelineKind::RayTracingPipeline;
    s.rtDxil = "raytracing_pipeline.dxil";
    cat.push_back(std::move(s));
  }

  {
    BenchmarkSpec s;
    s.id = "rt_inline_heavy";
    s.name = "RayQuery Heavy (4-Bounce+Shadow 64x64BVH)";
    s.category = BenchCategory::RayTracing;
    s.kind = PipelineKind::RayTracingInline;
    s.rtDxil = "bench_rt_inline_heavy_cs.dxil";
    cat.push_back(std::move(s));
  }

  {
    BenchmarkSpec s;
    s.id = "rt_pipeline_heavy";
    s.name = "DispatchRays Heavy (Recursive+Shadow 64x64BVH)";
    s.category = BenchCategory::RayTracing;
    s.kind = PipelineKind::RayTracingPipeline;
    s.rtDxil = "bench_rt_pipeline_heavy.dxil";
    cat.push_back(std::move(s));
  }

  {
    BenchmarkSpec s;
    s.id = "bench_mesh_stress";
    s.name = "Meshlet Grid Expansion";
    s.category = BenchCategory::Mesh;
    s.kind = PipelineKind::GraphicsMSPS;
    s.msDxil = "bench_mesh_stress_ms.dxil";
    s.psDxil = "bench_mesh_stress_ps.dxil";
    cat.push_back(std::move(s));
  }

  {
    BenchmarkSpec s;
    s.id = "bench_mesh_cull";
    s.name = "Adaptive Meshlet Culling (AS+MS+PS)";
    s.category = BenchCategory::Mesh;
    s.kind = PipelineKind::GraphicsMSPS;
    s.asDxil = "bench_mesh_cull_as.dxil";
    s.msDxil = "bench_mesh_cull_ms.dxil";
    s.psDxil = "bench_mesh_cull_ps.dxil";
    cat.push_back(std::move(s));
  }

  return cat;
}

// Runner declarations (implemented in crucible_runner.mm and irconverter_runner.mm)
MetalBenchResult runCrucibleComputeBench(const std::string &benchId,
                                         const std::vector<uint8_t> &dxil,
                                         uint32_t width, uint32_t height,
                                         uint32_t maxSteps, uint32_t iters,
                                         uint32_t warmup);

MetalBenchResult runCrucibleGraphicsBench(
    const std::string &benchId, const std::string &category,
    const std::vector<uint8_t> &vsDxil, const std::vector<uint8_t> &psDxil,
    uint32_t width, uint32_t height, uint32_t iters, uint32_t warmup,
    const std::vector<uint8_t> &hsDxil, const std::vector<uint8_t> &dsDxil,
    const std::vector<uint8_t> &gsDxil, bool hasTexture);

MetalBenchResult runCrucibleMeshBench(const std::string &benchId,
                                      const std::vector<uint8_t> &asDxil,
                                      const std::vector<uint8_t> &msDxil,
                                      const std::vector<uint8_t> &psDxil,
                                      uint32_t iters, uint32_t warmup);

MetalBenchResult runCrucibleRaytracingBench(const std::string &benchId,
                                            bool isInline,
                                            const std::vector<uint8_t> &dxil,
                                            uint32_t width, uint32_t height,
                                            uint32_t iters, uint32_t warmup);

MetalBenchResult runIRConverterComputeBench(const std::string &benchId,
                                            const std::vector<uint8_t> &dxil,
                                            uint32_t width, uint32_t height,
                                            uint32_t maxSteps, uint32_t iters,
                                            uint32_t warmup);

MetalBenchResult runIRConverterGraphicsBench(
    const std::string &benchId, const std::string &category,
    const std::vector<uint8_t> &vsDxil, const std::vector<uint8_t> &psDxil,
    uint32_t width, uint32_t height, uint32_t iters, uint32_t warmup,
    const std::vector<uint8_t> &hsDxil, const std::vector<uint8_t> &dsDxil,
    const std::vector<uint8_t> &gsDxil, bool hasTexture);

MetalBenchResult runIRConverterMeshBench(const std::string &benchId,
                                         const std::vector<uint8_t> &msDxil,
                                         const std::vector<uint8_t> &psDxil,
                                         uint32_t iters, uint32_t warmup);

MetalBenchResult runIRConverterRaytracingBench(const std::string &benchId,
                                               bool isInline,
                                               const std::vector<uint8_t> &dxil,
                                               uint32_t width, uint32_t height,
                                               uint32_t iters, uint32_t warmup);

} // namespace crucible_bench

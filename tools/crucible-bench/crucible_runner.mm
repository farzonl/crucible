//===- crucible_runner.mm - Crucible in-process Metal bench runner --------===//
//
// Part of Crucible, an open-source DXIL-to-MSL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Compiles DXIL -> MSL in-process via libcrucible and -> Metal PSO via
// [MTLDevice newLibraryWithSource:options:error:], running the exact same
// workloads, inputs, dispatch sizes, and checksums as irconverter_runner.mm.
//
//===----------------------------------------------------------------------===//

#include "bench_types.h"
#include "metal_helpers.h"
#include "crucible/Compiler.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <simd/simd.h>

#include <algorithm>
#include <chrono>
#include <cstring>

namespace crucible_bench {

namespace {

struct CompiledCrucibleStage {
  id<MTLLibrary> lib = nil;
  id<MTLFunction> fn = nil;
  crucible::Reflection reflect;
};

bool compileStageWithCrucible(id<MTLDevice> device,
                              const std::vector<uint8_t> &dxil,
                              const crucible::Options &opts,
                              CompiledCrucibleStage &out,
                              std::string &errOut) {
  llvm::MemoryBufferRef buf(
      llvm::StringRef(reinterpret_cast<const char *>(dxil.data()), dxil.size()),
      "shader.dxil");
  auto compiled = crucible::compileDXILToMSL(buf, opts);
  if (!compiled) {
    errOut = "crucible: " + llvm::toString(compiled.takeError());
    return false;
  }

  MTLCompileOptions *co = [[MTLCompileOptions alloc] init];
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
  if (@available(macOS 15.0, *)) {
    co.mathMode = MTLMathModeFast;
  } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    co.fastMathEnabled = YES;
#pragma clang diagnostic pop
  }
#else
  co.fastMathEnabled = YES;
#endif

  NSString *src = [NSString stringWithUTF8String:compiled->MSL.c_str()];
  NSError *mtlErr = nil;
  out.lib = [device newLibraryWithSource:src options:co error:&mtlErr];
  if (!out.lib) {
    std::string msg =
        mtlErr ? [[mtlErr localizedDescription] UTF8String] : "unknown";
    auto nl = msg.find('\n');
    if (nl != std::string::npos)
      msg = msg.substr(0, nl);
    errOut = "MSL compile: " + msg;
    return false;
  }

  NSString *entry =
      [NSString stringWithUTF8String:compiled->Reflect.MSLEntryName.c_str()];
  out.fn = [out.lib newFunctionWithName:entry];
  if (!out.fn) {
    errOut = "entry '" + compiled->Reflect.MSLEntryName + "' not found";
    return false;
  }
  out.reflect = std::move(compiled->Reflect);
  return true;
}

bool isTextureResource(crucible::ResourceKind k) {
  switch (k) {
  case crucible::ResourceKind::Texture1D:
  case crucible::ResourceKind::Texture2D:
  case crucible::ResourceKind::Texture2DMS:
  case crucible::ResourceKind::Texture3D:
  case crucible::ResourceKind::TextureCube:
  case crucible::ResourceKind::Texture1DArray:
  case crucible::ResourceKind::Texture2DArray:
  case crucible::ResourceKind::Texture2DMSArray:
  case crucible::ResourceKind::TextureCubeArray:
    return true;
  default:
    return false;
  }
}

} // namespace

MetalBenchResult runCrucibleComputeBench(const std::string &benchId,
                                         const std::vector<uint8_t> &dxil,
                                         uint32_t width, uint32_t height,
                                         uint32_t maxSteps, uint32_t iters,
                                         uint32_t warmup) {
  MetalBenchResult res;
  res.backend = "Metal (crucible)";
  res.shaderFormat = "DXIL->MSL";

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      res.status = "SKIPPED (no Metal device)";
      return res;
    }
    res.backend =
        std::string("Metal (crucible): ") + [[device name] UTF8String];

    auto t0 = std::chrono::steady_clock::now();

    crucible::Options opts;
    CompiledCrucibleStage csStage;
    std::string errStr;
    if (!compileStageWithCrucible(device, dxil, opts, csStage, errStr)) {
      res.status = "FAILED (" + errStr + ")";
      return res;
    }

    NSError *mtlErr = nil;
    MTLComputePipelineDescriptor *cpDesc =
        [[MTLComputePipelineDescriptor alloc] init];
    cpDesc.computeFunction = csStage.fn;
    uint32_t tgThreads = csStage.reflect.NumThreads[0] *
                         csStage.reflect.NumThreads[1] *
                         csStage.reflect.NumThreads[2];
    if (tgThreads > 0 && (tgThreads % 32u) == 0u)
      cpDesc.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithDescriptor:cpDesc
                                              options:MTLPipelineOptionNone
                                           reflection:nil
                                                error:&mtlErr];

    auto t1 = std::chrono::steady_clock::now();
    res.compileMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!pso) {
      std::string msg =
          mtlErr ? [[mtlErr localizedDescription] UTF8String] : "unknown";
      res.status = "FAILED (PSO: " + msg + ")";
      return res;
    }

    uint32_t longW = width * 2;
    uint32_t longH = height * 2;
    size_t maxOutBytes = 16 * 1024 * 1024;
    if (benchId == "bench_raymarch")
      maxOutBytes = std::max<size_t>(maxOutBytes,
                                     static_cast<size_t>(longW) * longH * 16);
    size_t auxBytes = 4 * 1024 * 1024;

    id<MTLBuffer> cbBuf =
        [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    id<MTLBuffer> srv0 =
        [device newBufferWithLength:auxBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> srv1 =
        [device newBufferWithLength:auxBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> uav0 =
        [device newBufferWithLength:maxOutBytes
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> uav1 =
        [device newBufferWithLength:auxBytes options:MTLResourceStorageModeShared];

    std::memset([srv0 contents], 0, auxBytes);
    std::memset([srv1 contents], 0, auxBytes);
    std::memset([uav0 contents], 0, maxOutBytes);
    std::memset([uav1 contents], 0, auxBytes);

    if (benchId == "bench_raymarch") {
      float spheres[16] = {
          0.0f,  0.2f, 0.5f, 0.8f, -1.3f, 0.0f, 1.0f, 0.6f,
          1.3f,  0.1f, 0.9f, 0.65f, 0.0f, 1.1f, 1.5f, 0.5f,
      };
      std::memcpy([srv0 contents], spheres, sizeof(spheres));
    } else if (benchId == "matmul" || benchId == "tiled_matmul" ||
               benchId == "tiled_matmul_large" || benchId == "linalg_matmul") {
      auto *a = static_cast<float *>([srv0 contents]);
      auto *b = static_cast<float *>([srv1 contents]);
      const size_t initCount =
          (benchId == "tiled_matmul_large") ? (1024u * 1024u) : (256u * 256u);
      for (size_t i = 0; i < initCount; ++i) {
        a[i] = static_cast<float>((i % 17) + 1) * 0.01f;
        b[i] = static_cast<float>((i % 13) + 1) * 0.01f;
      }
    } else if (benchId == "prefix_sum_groupshared") {
      auto *p = static_cast<uint32_t *>([srv0 contents]);
      for (size_t i = 0; i < 65536 * 4; ++i)
        p[i] = static_cast<uint32_t>((i & 63u) + 1u);
    }

    // Pre-resolve direct Metal buffer bindings from Crucible reflection.
    struct BufBind {
      uint32_t slot;
      __unsafe_unretained id<MTLBuffer> buf;
    };
    std::vector<BufBind> bindings;
    for (const auto &rb : csStage.reflect.Resources) {
      if (rb.Class == crucible::ResourceClass::CBV && rb.LowerBound == 0)
        bindings.push_back({rb.MetalIndex, cbBuf});
      else if (rb.Class == crucible::ResourceClass::SRV && rb.LowerBound == 0)
        bindings.push_back({rb.MetalIndex, srv0});
      else if (rb.Class == crucible::ResourceClass::SRV && rb.LowerBound == 1)
        bindings.push_back({rb.MetalIndex, srv1});
      else if (rb.Class == crucible::ResourceClass::UAV && rb.LowerBound == 0)
        bindings.push_back({rb.MetalIndex, uav0});
      else if (rb.Class == crucible::ResourceClass::UAV && rb.LowerBound == 1)
        bindings.push_back({rb.MetalIndex, uav1});
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    bool is2D = (benchId == "bench_raymarch" || benchId == "matmul" ||
                 benchId == "tiled_matmul" || benchId == "tiled_matmul_large");
    MTLSize tgSize =
        (benchId == "linalg_matmul")
            ? MTLSizeMake(32, 1, 1)
            : (benchId == "tiled_matmul" || benchId == "tiled_matmul_large")
                  ? MTLSizeMake(16, 16, 1)
                  : (is2D ? MTLSizeMake(8, 8, 1) : MTLSizeMake(64, 1, 1));

    auto getDims = [&](int mode, uint32_t &gx, uint32_t &gy, uint32_t &cbW,
                       uint32_t &cbH, uint32_t &cbSteps) {
      if (benchId == "bench_raymarch") {
        cbW = (mode == 0) ? 64 : (mode == 1 ? width : width * 2);
        cbH = (mode == 0) ? 64 : (mode == 1 ? height : height * 2);
        cbSteps = (mode == 0) ? 16 : (mode == 1 ? maxSteps : maxSteps * 2);
        gx = (cbW + 7) / 8;
        gy = (cbH + 7) / 8;
      } else if (benchId == "matmul") {
        uint32_t n = (mode == 0) ? 64 : (mode == 1 ? 128 : 256);
        cbW = cbH = cbSteps = n;
        gx = (n + 7) / 8;
        gy = (n + 7) / 8;
      } else if (benchId == "tiled_matmul") {
        uint32_t n = (mode == 0) ? 64 : (mode == 1 ? 128 : 256);
        cbW = cbH = cbSteps = n;
        gx = (n + 15) / 16;
        gy = (n + 15) / 16;
      } else if (benchId == "tiled_matmul_large") {
        uint32_t n = (mode == 0) ? 256 : (mode == 1 ? 512 : 1024);
        cbW = cbH = cbSteps = n;
        gx = (n + 15) / 16;
        gy = (n + 15) / 16;
      } else if (benchId == "linalg_matmul") {
        uint32_t n = (mode == 0) ? 64 : (mode == 1 ? 128 : 256);
        cbW = cbH = cbSteps = n;
        gx = (n + 31) / 32;
        gy = (n + 31) / 32;
      } else {
        gx = (mode == 0) ? 64 : (mode == 1 ? 512 : 4096);
        gy = 1;
        cbW = gx * 64;
        cbH = 1;
        cbSteps = 1;
      }
    };

    auto dispatchOnce = [&](uint32_t gx, uint32_t gy, uint32_t cbW,
                            uint32_t cbH, uint32_t cbSteps) -> double {
      if (benchId == "bench_raymarch") {
        RaymarchCB cb{cbW, cbH, cbSteps, 0.0f};
        std::memcpy([cbBuf contents], &cb, sizeof(cb));
      } else if (benchId == "matmul" || benchId == "tiled_matmul" ||
                 benchId == "tiled_matmul_large" ||
                 benchId == "linalg_matmul") {
        MatmulCB cb{cbW, cbH, cbSteps, 0};
        std::memcpy([cbBuf contents], &cb, sizeof(cb));
      }
      auto s0 = std::chrono::steady_clock::now();
      id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      [enc setComputePipelineState:pso];
      for (const auto &b : bindings)
        [enc setBuffer:b.buf offset:0 atIndex:b.slot];
      [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
          threadsPerThreadgroup:tgSize];
      [enc endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
      auto s1 = std::chrono::steady_clock::now();
      return measureCmdMs(cmdBuf, s0, s1);
    };

    uint32_t gx = 1, gy = 1, cbW = 1, cbH = 1, cbSteps = 1;
    getDims(1, gx, gy, cbW, cbH, cbSteps);

    res.firstDispatchMs = dispatchOnce(gx, gy, cbW, cbH, cbSteps);
    runWarmup(warmup, [&]() { return dispatchOnce(gx, gy, cbW, cbH, cbSteps); });

    std::vector<double> samples;
    samples.reserve(iters);
    for (uint32_t i = 0; i < iters; ++i)
      samples.push_back(dispatchOnce(gx, gy, cbW, cbH, cbSteps));

    const auto *outPtr = static_cast<const float *>([uav0 contents]);
    double sum = 0.0;
    size_t floatCount =
        (benchId == "bench_raymarch")
            ? static_cast<size_t>(width) * height * 4
            : std::min<size_t>(4096, maxOutBytes / sizeof(float));
    for (size_t i = 0; i < floatCount; ++i)
      sum += outPtr[i];

    getDims(0, gx, gy, cbW, cbH, cbSteps);
    res.shortJobMs = dispatchOnce(gx, gy, cbW, cbH, cbSteps);
    getDims(2, gx, gy, cbW, cbH, cbSteps);
    res.longJobMs = dispatchOnce(gx, gy, cbW, cbH, cbSteps);

    finalizeSamples(res, samples);
    res.checksum = sum;
    res.ran = true;
    res.status = "OK";
  }
  return res;
}

MetalBenchResult runCrucibleGraphicsBench(
    const std::string &benchId, const std::string &category,
    const std::vector<uint8_t> &vsDxil, const std::vector<uint8_t> &psDxil,
    uint32_t width, uint32_t height, uint32_t iters, uint32_t warmup,
    const std::vector<uint8_t> &hsDxil, const std::vector<uint8_t> &dsDxil,
    const std::vector<uint8_t> &gsDxil, bool hasTexture) {
  MetalBenchResult res;
  res.backend = "Metal (crucible)";
  res.shaderFormat = "DXIL->MSL";

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      res.status = "SKIPPED (no Metal device)";
      return res;
    }
    res.backend =
        std::string("Metal (crucible): ") + [[device name] UTF8String];

    const bool hasTess = (!hsDxil.empty() && !dsDxil.empty());
    const bool hasGS = !gsDxil.empty();
    if (hasTess || hasGS) {
      // Verify all stages compile through Crucible + Metal, and report
      // multi-stage compute-lowered GS/Tess dispatch status clearly.
      auto t0 = std::chrono::steady_clock::now();
      crucible::Options opts;
      opts.StageAsCompute = true;
      crucible::Options psOpts;
      CompiledCrucibleStage vsS, hsS, dsS, gsS, psS;
      std::string errStr;
      bool ok = compileStageWithCrucible(device, vsDxil, opts, vsS, errStr) &&
                compileStageWithCrucible(device, psDxil, psOpts, psS, errStr);
      if (ok && hasTess) {
        ok = compileStageWithCrucible(device, hsDxil, opts, hsS, errStr) &&
             compileStageWithCrucible(device, dsDxil, opts, dsS, errStr);
      }
      if (ok && hasGS)
        ok = compileStageWithCrucible(device, gsDxil, opts, gsS, errStr);
      auto t1 = std::chrono::steady_clock::now();
      res.compileMs =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
      if (!ok) {
        res.status = "FAILED (" + errStr + ")";
      } else {
        res.status = "SKIPPED (multi-stage GS/Tess harness TBD; compiled OK)";
      }
      return res;
    }

    auto t0 = std::chrono::steady_clock::now();

    crucible::Options opts;
    CompiledCrucibleStage vsStage, psStage;
    std::string errStr;
    if (!compileStageWithCrucible(device, vsDxil, opts, vsStage, errStr) ||
        !compileStageWithCrucible(device, psDxil, opts, psStage, errStr)) {
      res.status = "FAILED (" + errStr + ")";
      return res;
    }

    bool isVertexMesh = (benchId == "bench_vertex_mesh");
    bool isPixel = (category == "Pixel");
    constexpr uint32_t kVertexBufSlot = 30;

    MTLRenderPipelineDescriptor *psoDesc =
        [[MTLRenderPipelineDescriptor alloc] init];
    psoDesc.vertexFunction = vsStage.fn;
    psoDesc.fragmentFunction = psStage.fn;
    psoDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;

    id<MTLDepthStencilState> dsState = nil;
    if (isVertexMesh) {
      psoDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
      MTLVertexDescriptor *vDesc = [MTLVertexDescriptor vertexDescriptor];
      vDesc.attributes[0].format = MTLVertexFormatFloat3;
      vDesc.attributes[0].offset = 0;
      vDesc.attributes[0].bufferIndex = kVertexBufSlot;
      vDesc.attributes[1].format = MTLVertexFormatShort4Normalized;
      vDesc.attributes[1].offset = 12;
      vDesc.attributes[1].bufferIndex = kVertexBufSlot;
      vDesc.attributes[2].format = MTLVertexFormatUShort2Normalized;
      vDesc.attributes[2].offset = 20;
      vDesc.attributes[2].bufferIndex = kVertexBufSlot;
      vDesc.attributes[3].format = MTLVertexFormatUChar4Normalized;
      vDesc.attributes[3].offset = 24;
      vDesc.attributes[3].bufferIndex = kVertexBufSlot;
      vDesc.attributes[4].format = MTLVertexFormatUChar4;
      vDesc.attributes[4].offset = 28;
      vDesc.attributes[4].bufferIndex = kVertexBufSlot;
      vDesc.layouts[kVertexBufSlot].stride = sizeof(MeshBenchVertex);
      vDesc.layouts[kVertexBufSlot].stepFunction =
          MTLVertexStepFunctionPerVertex;
      psoDesc.vertexDescriptor = vDesc;

      MTLDepthStencilDescriptor *dsd = [[MTLDepthStencilDescriptor alloc] init];
      dsd.depthCompareFunction = MTLCompareFunctionLess;
      dsd.depthWriteEnabled = YES;
      dsState = [device newDepthStencilStateWithDescriptor:dsd];
    } else if (isPixel && !vsStage.reflect.Inputs.empty()) {
      MTLVertexDescriptor *vDesc = [MTLVertexDescriptor vertexDescriptor];
      vDesc.attributes[0].format = MTLVertexFormatFloat3;
      vDesc.attributes[0].offset = 0;
      vDesc.attributes[0].bufferIndex = kVertexBufSlot;
      vDesc.layouts[kVertexBufSlot].stride = 3 * sizeof(float);
      vDesc.layouts[kVertexBufSlot].stepFunction =
          MTLVertexStepFunctionPerVertex;
      psoDesc.vertexDescriptor = vDesc;
    }

    NSError *mtlErr = nil;
    id<MTLRenderPipelineState> pso =
        [device newRenderPipelineStateWithDescriptor:psoDesc error:&mtlErr];

    auto t1 = std::chrono::steady_clock::now();
    res.compileMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!pso) {
      res.status =
          mtlErr ? std::string("FAILED (PSO: ") +
                       [[mtlErr localizedDescription] UTF8String] + ")"
                 : "FAILED (Metal Render PSO creation)";
      return res;
    }

    uint32_t maxW = (isPixel || isVertexMesh) ? std::max(128u, width * 2) : 128;
    uint32_t maxH =
        (isPixel || isVertexMesh) ? std::max(128u, height * 2) : 128;
    MTLTextureDescriptor *texDesc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:maxW
                                    height:maxH
                                 mipmapped:NO];
    texDesc.usage = MTLTextureUsageRenderTarget;
    texDesc.storageMode = MTLStorageModeShared;
    id<MTLTexture> rtTex = [device newTextureWithDescriptor:texDesc];

    id<MTLTexture> depthTex = nil;
    MeshBenchSceneData meshScenes[3];
    id<MTLBuffer> meshVBs[3] = {nil, nil, nil};
    id<MTLBuffer> meshIBs[3] = {nil, nil, nil};
    if (isVertexMesh) {
      MTLTextureDescriptor *dDesc = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                       width:maxW
                                      height:maxH
                                   mipmapped:NO];
      dDesc.usage = MTLTextureUsageRenderTarget;
      dDesc.storageMode = MTLStorageModePrivate;
      depthTex = [device newTextureWithDescriptor:dDesc];
      for (int m = 0; m < 3; ++m) {
        uint32_t g = 0, mrw = 0, mrh = 0;
        getMeshBenchGridAndDims(m, width, height, g, mrw, mrh);
        meshScenes[m] = buildMeshBenchScene(g, g);
        meshVBs[m] = [device
            newBufferWithBytes:meshScenes[m].vertices.data()
                        length:meshScenes[m].vertices.size() *
                               sizeof(MeshBenchVertex)
                       options:MTLResourceStorageModeShared];
        meshIBs[m] =
            [device newBufferWithBytes:meshScenes[m].indices.data()
                                length:meshScenes[m].indices.size() *
                                       sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
      }
    }

    size_t cbBytes = isVertexMesh ? sizeof(MeshBenchConstants) : 256;
    id<MTLBuffer> cbBuf =
        [device newBufferWithLength:cbBytes options:MTLResourceStorageModeShared];
    const float quadVerts[18] = {
        -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f,  1.0f, 0.0f,
        -1.0f, -1.0f, 0.0f, 1.0f, 1.0f,  0.0f, -1.0f, 1.0f, 0.0f,
    };
    id<MTLBuffer> vbBuf =
        [device newBufferWithBytes:quadVerts
                            length:sizeof(quadVerts)
                           options:MTLResourceStorageModeShared];

    id<MTLTexture> colorTex = nil;
    id<MTLSamplerState> samplerState = nil;
    if (hasTexture) {
      MTLTextureDescriptor *td = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                       width:256
                                      height:256
                                   mipmapped:YES];
      td.mipmapLevelCount = 9;
      td.usage = MTLTextureUsageShaderRead;
      td.storageMode = MTLStorageModeShared;
      colorTex = [device newTextureWithDescriptor:td];
      for (uint32_t mip = 0; mip < 9; ++mip) {
        uint32_t mw = std::max(1u, 256u >> mip);
        uint32_t mh = std::max(1u, 256u >> mip);
        auto pixels = buildProceduralTexture256Mip(mip);
        [colorTex replaceRegion:MTLRegionMake2D(0, 0, mw, mh)
                    mipmapLevel:mip
                      withBytes:pixels.data()
                    bytesPerRow:mw * 4u];
      }
      MTLSamplerDescriptor *sd = [[MTLSamplerDescriptor alloc] init];
      sd.minFilter = MTLSamplerMinMagFilterLinear;
      sd.magFilter = MTLSamplerMinMagFilterLinear;
      sd.mipFilter = MTLSamplerMipFilterLinear;
      sd.sAddressMode = MTLSamplerAddressModeRepeat;
      sd.tAddressMode = MTLSamplerAddressModeRepeat;
      sd.rAddressMode = MTLSamplerAddressModeRepeat;
      samplerState = [device newSamplerStateWithDescriptor:sd];
    }

    // Collect direct Metal slots for VS and PS from Crucible reflection.
    std::vector<uint32_t> vsCBSlots, psCBSlots, psTexSlots, psSampSlots;
    for (const auto &rb : vsStage.reflect.Resources) {
      if (rb.Class == crucible::ResourceClass::CBV && rb.LowerBound == 0)
        vsCBSlots.push_back(rb.MetalIndex);
    }
    for (const auto &rb : psStage.reflect.Resources) {
      if (rb.Class == crucible::ResourceClass::CBV && rb.LowerBound == 0)
        psCBSlots.push_back(rb.MetalIndex);
      else if (rb.Class == crucible::ResourceClass::SRV &&
               isTextureResource(rb.Kind))
        psTexSlots.push_back(rb.MetalIndex);
      else if (rb.Class == crucible::ResourceClass::Sampler)
        psSampSlots.push_back(rb.MetalIndex);
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];

    auto getGfxParams = [&](int mode, uint32_t &rw, uint32_t &rh,
                            uint32_t &vc) {
      if (isVertexMesh) {
        uint32_t g = 0;
        getMeshBenchGridAndDims(mode, width, height, g, rw, rh);
        vc = static_cast<uint32_t>(meshScenes[mode].indices.size());
      } else if (isPixel) {
        rw = (mode == 0) ? 64 : (mode == 1 ? width : width * 2);
        rh = (mode == 0) ? 64 : (mode == 1 ? height : height * 2);
        vc = 6;
      } else {
        rw = 128;
        rh = 128;
        vc = (mode == 0) ? 3000 : (mode == 1 ? 24000 : 96000);
      }
    };

    auto drawOnce = [&](int mode, uint32_t rw, uint32_t rh,
                        uint32_t vc) -> double {
      if (isVertexMesh) {
        std::memcpy([cbBuf contents], &meshScenes[mode].constants,
                    sizeof(MeshBenchConstants));
      } else if (isPixel) {
        ShadertoyCB cb{static_cast<float>(rw), static_cast<float>(rh),
                       1.0f,                   1.25f,
                       0.0f,                   0.0f,
                       0.0f,                   0.0f};
        std::memcpy([cbBuf contents], &cb, sizeof(cb));
      } else {
        VertexStressCB cb{};
        cb.mvp[0] = 1.0f;
        cb.mvp[5] = 1.0f;
        cb.mvp[10] = 1.0f;
        cb.mvp[15] = 1.0f;
        cb.time = 0.5f;
        cb.vertexCount = vc;
        std::memcpy([cbBuf contents], &cb, sizeof(cb));
      }

      MTLRenderPassDescriptor *rpd =
          [MTLRenderPassDescriptor renderPassDescriptor];
      rpd.colorAttachments[0].texture = rtTex;
      rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
      rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
      rpd.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
      if (isVertexMesh) {
        rpd.depthAttachment.texture = depthTex;
        rpd.depthAttachment.loadAction = MTLLoadActionClear;
        rpd.depthAttachment.storeAction = MTLStoreActionDontCare;
        rpd.depthAttachment.clearDepth = 1.0;
      }

      auto s0 = std::chrono::steady_clock::now();
      id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
      id<MTLRenderCommandEncoder> enc =
          [cmdBuf renderCommandEncoderWithDescriptor:rpd];
      [enc setRenderPipelineState:pso];
      [enc setViewport:(MTLViewport){0.0, 0.0, static_cast<double>(rw),
                                     static_cast<double>(rh), 0.0, 1.0}];
      [enc setScissorRect:(MTLScissorRect){0, 0, rw, rh}];
      for (uint32_t slot : vsCBSlots)
        [enc setVertexBuffer:cbBuf offset:0 atIndex:slot];
      for (uint32_t slot : psCBSlots)
        [enc setFragmentBuffer:cbBuf offset:0 atIndex:slot];
      if (colorTex) {
        for (uint32_t slot : psTexSlots)
          [enc setFragmentTexture:colorTex atIndex:slot];
      }
      if (samplerState) {
        for (uint32_t slot : psSampSlots)
          [enc setFragmentSamplerState:samplerState atIndex:slot];
      }

      if (isVertexMesh) {
        [enc setDepthStencilState:dsState];
        [enc setFrontFacingWinding:MTLWindingClockwise];
        [enc setCullMode:MTLCullModeBack];
        [enc setVertexBuffer:meshVBs[mode] offset:0 atIndex:kVertexBufSlot];
        [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                        indexCount:vc
                         indexType:MTLIndexTypeUInt32
                       indexBuffer:meshIBs[mode]
                 indexBufferOffset:0];
      } else {
        if (isPixel && !vsStage.reflect.Inputs.empty())
          [enc setVertexBuffer:vbBuf offset:0 atIndex:kVertexBufSlot];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle
                vertexStart:0
                vertexCount:vc];
      }
      [enc endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
      auto s1 = std::chrono::steady_clock::now();
      return measureCmdMs(cmdBuf, s0, s1);
    };

    uint32_t rw = 128, rh = 128, vc = 6;
    getGfxParams(1, rw, rh, vc);

    res.firstDispatchMs = drawOnce(1, rw, rh, vc);
    runWarmup(warmup, [&]() { return drawOnce(1, rw, rh, vc); });

    std::vector<double> samples;
    samples.reserve(iters);
    for (uint32_t i = 0; i < iters; ++i)
      samples.push_back(drawOnce(1, rw, rh, vc));

    std::vector<uint8_t> px(static_cast<size_t>(rw) * rh * 4, 0);
    [rtTex getBytes:px.data()
        bytesPerRow:rw * 4
         fromRegion:MTLRegionMake2D(0, 0, rw, rh)
        mipmapLevel:0];
    double sum = 0.0;
    for (uint8_t v : px)
      sum += v;
    res.checksum = sum;

    getGfxParams(0, rw, rh, vc);
    res.shortJobMs = drawOnce(0, rw, rh, vc);
    getGfxParams(2, rw, rh, vc);
    res.longJobMs = drawOnce(2, rw, rh, vc);

    finalizeSamples(res, samples);
    res.ran = true;
    res.status = "OK";
  }
  return res;
}

MetalBenchResult runCrucibleMeshBench(const std::string &benchId,
                                      const std::vector<uint8_t> &asDxil,
                                      const std::vector<uint8_t> &msDxil,
                                      const std::vector<uint8_t> &psDxil,
                                      uint32_t iters, uint32_t warmup) {
  MetalBenchResult res;
  res.backend = "Metal (crucible)";
  res.shaderFormat = "DXIL->MSL";

  (void)benchId;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      res.status = "SKIPPED (no Metal device)";
      return res;
    }
    res.backend =
        std::string("Metal (crucible): ") + [[device name] UTF8String];
    if (!deviceSupportsMeshShaders(device)) {
      res.status = "SKIPPED (no Metal Mesh Shader support on this GPU)";
      return res;
    }

    auto t0 = std::chrono::steady_clock::now();

    crucible::Options opts;
    CompiledCrucibleStage asStage, msStage, psStage;
    std::string errStr;
    if (!asDxil.empty() &&
        !compileStageWithCrucible(device, asDxil, opts, asStage, errStr)) {
      res.status = "FAILED (" + errStr + ")";
      return res;
    }
    if (!compileStageWithCrucible(device, msDxil, opts, msStage, errStr) ||
        !compileStageWithCrucible(device, psDxil, opts, psStage, errStr)) {
      res.status = "FAILED (" + errStr + ")";
      return res;
    }

    MTLMeshRenderPipelineDescriptor *meshDesc =
        [[MTLMeshRenderPipelineDescriptor alloc] init];
    if (!asDxil.empty())
      meshDesc.objectFunction = asStage.fn;
    meshDesc.meshFunction = msStage.fn;
    meshDesc.fragmentFunction = psStage.fn;
    meshDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    meshDesc.maxTotalThreadsPerMeshThreadgroup = 32;

    NSError *mtlErr = nil;
    id<MTLRenderPipelineState> pso =
        [device newRenderPipelineStateWithMeshDescriptor:meshDesc
                                                 options:MTLPipelineOptionNone
                                              reflection:nil
                                                   error:&mtlErr];
    auto t1 = std::chrono::steady_clock::now();
    res.compileMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!pso) {
      std::string msg =
          mtlErr ? [[mtlErr localizedDescription] UTF8String] : "unknown";
      res.status = "FAILED (MTLMeshPSO: " + msg + ")";
      return res;
    }

    const uint32_t rw = 128, rh = 128;
    MTLTextureDescriptor *td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:rw
                                    height:rh
                                 mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> rtTex = [device newTextureWithDescriptor:td];
    id<MTLBuffer> cbBuf =
        [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];

    std::vector<uint32_t> asCBSlots, msCBSlots, psCBSlots;
    for (const auto &rb : asStage.reflect.Resources)
      if (rb.Class == crucible::ResourceClass::CBV && rb.LowerBound == 0)
        asCBSlots.push_back(rb.MetalIndex);
    for (const auto &rb : msStage.reflect.Resources)
      if (rb.Class == crucible::ResourceClass::CBV && rb.LowerBound == 0)
        msCBSlots.push_back(rb.MetalIndex);
    for (const auto &rb : psStage.reflect.Resources)
      if (rb.Class == crucible::ResourceClass::CBV && rb.LowerBound == 0)
        psCBSlots.push_back(rb.MetalIndex);

    auto drawOnce = [&](uint32_t groups) -> double {
      struct {
        float scale;
        float time;
        uint32_t groupCount;
        float pad;
      } cb{0.05f, 0.25f, groups, 0.0f};
      std::memcpy([cbBuf contents], &cb, sizeof(cb));

      auto s0 = std::chrono::steady_clock::now();
      id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
      MTLRenderPassDescriptor *rpd =
          [MTLRenderPassDescriptor renderPassDescriptor];
      rpd.colorAttachments[0].texture = rtTex;
      rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
      rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
      rpd.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);

      id<MTLRenderCommandEncoder> enc =
          [cmdBuf renderCommandEncoderWithDescriptor:rpd];
      if (!encoderSupportsMeshShaders(enc)) {
        [enc endEncoding];
        return -1.0;
      }
      [enc setRenderPipelineState:pso];
      [enc setViewport:(MTLViewport){0.0, 0.0, static_cast<double>(rw),
                                     static_cast<double>(rh), 0.0, 1.0}];
      for (uint32_t slot : asCBSlots)
        [enc setObjectBuffer:cbBuf offset:0 atIndex:slot];
      for (uint32_t slot : msCBSlots)
        [enc setMeshBuffer:cbBuf offset:0 atIndex:slot];
      for (uint32_t slot : psCBSlots)
        [enc setFragmentBuffer:cbBuf offset:0 atIndex:slot];
      [enc drawMeshThreadgroups:MTLSizeMake(groups, 1, 1)
          threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
            threadsPerMeshThreadgroup:MTLSizeMake(32, 1, 1)];
      [enc endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
      auto s1 = std::chrono::steady_clock::now();
      return measureCmdMs(cmdBuf, s0, s1);
    };

    res.firstDispatchMs = drawOnce(128);
    if (res.firstDispatchMs < 0.0) {
      res.firstDispatchMs = 0.0;
      res.status = "SKIPPED (no Metal Mesh Shader encoder on this GPU)";
      return res;
    }
    runWarmup(warmup, [&]() { return drawOnce(128); });
    std::vector<double> samples;
    samples.reserve(iters);
    for (uint32_t i = 0; i < iters; ++i)
      samples.push_back(drawOnce(128));

    std::vector<uint8_t> px(static_cast<size_t>(rw) * rh * 4, 0);
    [rtTex getBytes:px.data()
        bytesPerRow:rw * 4
         fromRegion:MTLRegionMake2D(0, 0, rw, rh)
        mipmapLevel:0];
    double sum = 0.0;
    for (uint8_t b : px)
      sum += b;
    res.checksum = sum;

    res.shortJobMs = drawOnce(16);
    res.longJobMs = drawOnce(512);
    finalizeSamples(res, samples);
    res.ran = true;
    res.status = "OK";
  }
  return res;
}

MetalBenchResult runCrucibleRaytracingBench(const std::string &benchId,
                                            bool isInline,
                                            const std::vector<uint8_t> &dxil,
                                            uint32_t width, uint32_t height,
                                            uint32_t iters, uint32_t warmup) {
  MetalBenchResult res;
  res.backend = "Metal (crucible)";
  res.shaderFormat = "DXIL->MSL";

  (void)benchId;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      res.status = "SKIPPED (no Metal device)";
      return res;
    }
    if (![device supportsRaytracing]) {
      res.status = "SKIPPED (Metal raytracing unsupported)";
      return res;
    }
    res.backend =
        std::string("Metal (crucible): ") + [[device name] UTF8String];

    id<MTLCommandQueue> queue = [device newCommandQueue];

    const float triVerts[9] = {0.0f, 1.0f, 0.0f, -1.0f, -1.0f,
                               0.0f, 1.0f, -1.0f, 0.0f};
    id<MTLBuffer> vb =
        [device newBufferWithBytes:triVerts
                            length:sizeof(triVerts)
                           options:MTLResourceStorageModeShared];

    MTLAccelerationStructureTriangleGeometryDescriptor *triGeo =
        [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    triGeo.vertexBuffer = vb;
    triGeo.vertexStride = 12;
    triGeo.triangleCount = 1;
    triGeo.opaque = YES;

    MTLPrimitiveAccelerationStructureDescriptor *blasDesc =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    blasDesc.geometryDescriptors = @[ triGeo ];
    MTLAccelerationStructureSizes blasSizes =
        [device accelerationStructureSizesWithDescriptor:blasDesc];
    id<MTLAccelerationStructure> blas = [device
        newAccelerationStructureWithSize:blasSizes.accelerationStructureSize];

    id<MTLBuffer> scratch = [device
        newBufferWithLength:std::max<NSUInteger>(4096,
                                                 blasSizes.buildScratchBufferSize)
                    options:MTLResourceStorageModePrivate];
    id<MTLCommandBuffer> asCmd = [queue commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> asEnc =
        [asCmd accelerationStructureCommandEncoder];
    [asEnc buildAccelerationStructure:blas
                           descriptor:blasDesc
                        scratchBuffer:scratch
                  scratchBufferOffset:0];
    [asEnc endEncoding];
    [asCmd commit];
    [asCmd waitUntilCompleted];

    id<MTLBuffer> instBuf = [device
        newBufferWithLength:sizeof(MTLAccelerationStructureInstanceDescriptor)
                    options:MTLResourceStorageModeShared];
    auto *instDesc =
        static_cast<MTLAccelerationStructureInstanceDescriptor *>(
            [instBuf contents]);
    std::memset(instDesc, 0,
                sizeof(MTLAccelerationStructureInstanceDescriptor));
    instDesc->transformationMatrix.columns[0] = MTLPackedFloat3Make(1, 0, 0);
    instDesc->transformationMatrix.columns[1] = MTLPackedFloat3Make(0, 1, 0);
    instDesc->transformationMatrix.columns[2] = MTLPackedFloat3Make(0, 0, 1);
    instDesc->mask = 0xFF;
    instDesc->options = MTLAccelerationStructureInstanceOptionOpaque;
    instDesc->accelerationStructureIndex = 0;

    MTLInstanceAccelerationStructureDescriptor *tlasDesc =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    tlasDesc.instancedAccelerationStructures = @[ blas ];
    tlasDesc.instanceCount = 1;
    tlasDesc.instanceDescriptorBuffer = instBuf;
    MTLAccelerationStructureSizes tlasSizes =
        [device accelerationStructureSizesWithDescriptor:tlasDesc];
    id<MTLAccelerationStructure> tlas = [device
        newAccelerationStructureWithSize:tlasSizes.accelerationStructureSize];

    id<MTLBuffer> tlasScratch = [device
        newBufferWithLength:std::max<NSUInteger>(4096,
                                                 tlasSizes.buildScratchBufferSize)
                    options:MTLResourceStorageModePrivate];
    id<MTLCommandBuffer> tlasCmd = [queue commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> tlasEnc =
        [tlasCmd accelerationStructureCommandEncoder];
    [tlasEnc buildAccelerationStructure:tlas
                             descriptor:tlasDesc
                          scratchBuffer:tlasScratch
                    scratchBufferOffset:0];
    [tlasEnc endEncoding];
    [tlasCmd commit];
    [tlasCmd waitUntilCompleted];

    auto t0 = std::chrono::steady_clock::now();

    crucible::Options opts;
    if (!isInline) {
      opts.MissShaders = {"MissMain"};
      opts.HitGroupClosestHits = {"ClosestHitMain"};
    }
    CompiledCrucibleStage rtStage;
    std::string errStr;
    if (!compileStageWithCrucible(device, dxil, opts, rtStage, errStr)) {
      res.status = "FAILED (" + errStr + ")";
      return res;
    }

    NSError *mtlErr = nil;
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:rtStage.fn error:&mtlErr];

    auto t1 = std::chrono::steady_clock::now();
    res.compileMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!pso) {
      std::string msg =
          mtlErr ? [[mtlErr localizedDescription] UTF8String] : "unknown";
      res.status = "FAILED (RT PSO: " + msg + ")";
      return res;
    }

    id<MTLBuffer> outBuf =
        [device newBufferWithLength:4 * 1024 * 1024
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> instContribBuf =
        [device newBufferWithLength:sizeof(uint32_t)
                            options:MTLResourceStorageModeShared];
    std::memset([instContribBuf contents], 0, sizeof(uint32_t));

    int32_t asSlot = -1, outSlot = -1;
    for (const auto &rb : rtStage.reflect.Resources) {
      if (rb.Kind == crucible::ResourceKind::RTAccelerationStructure)
        asSlot = static_cast<int32_t>(rb.MetalIndex);
      else if (rb.Class == crucible::ResourceClass::UAV && rb.LowerBound == 0)
        outSlot = static_cast<int32_t>(rb.MetalIndex);
    }

    auto dispatchOnce = [&](uint32_t w, uint32_t h) -> double {
      auto s0 = std::chrono::steady_clock::now();
      id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc useResource:blas usage:MTLResourceUsageRead];
      if (asSlot >= 0)
        [enc setAccelerationStructure:tlas atBufferIndex:asSlot];
      if (outSlot >= 0)
        [enc setBuffer:outBuf offset:0 atIndex:outSlot];
      if (rtStage.reflect.HasInstanceContribution)
        [enc setBuffer:instContribBuf
                offset:0
               atIndex:rtStage.reflect.InstanceContributionMetalIndex];
      if (isInline) {
        [enc dispatchThreadgroups:MTLSizeMake((w + 3) / 4, h, 1)
            threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
      } else {
        [enc dispatchThreadgroups:MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1)
            threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
      }
      [enc endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
      auto s1 = std::chrono::steady_clock::now();
      return measureCmdMs(cmdBuf, s0, s1);
    };

    res.firstDispatchMs = dispatchOnce(width, height);
    runWarmup(warmup, [&]() { return dispatchOnce(width, height); });
    std::vector<double> samples;
    samples.reserve(iters);
    for (uint32_t i = 0; i < iters; ++i)
      samples.push_back(dispatchOnce(width, height));

    const size_t checksumCount = std::min<size_t>(
        1024u * 1024u,
        isInline ? static_cast<size_t>(width) * 4u : static_cast<size_t>(width));
    const auto *up = static_cast<const uint32_t *>([outBuf contents]);
    double sum = 0.0;
    for (size_t i = 0; i < checksumCount; ++i)
      sum += static_cast<double>(up[i] & 0xFFFFu);
    res.checksum = sum;

    res.shortJobMs = dispatchOnce(64, 64);
    res.longJobMs = dispatchOnce(width * 2, height * 2);

    finalizeSamples(res, samples);
    res.ran = true;
    res.status = "OK";
  }
  return res;
}

} // namespace crucible_bench

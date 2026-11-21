//===- VulkanDevice.cpp - Crucible Vulkan offload provider -------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements offloadtest::Device with a DirectX API personality backed by
// Crucible's DXIL -> SPIR-V compiler and Vulkan (Lavapipe/Mesa/MoltenVK).
//
//===----------------------------------------------------------------------===//

#include "Capabilities.h"
#include "Common.h"
#include "crucible/Compiler.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Object/DXContainer.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace llvm;
using namespace offloadtest;
using namespace crucible_offload;

namespace {

Error vkErr(VkResult R, const char *What) {
  return createStringError(inconvertibleErrorCode(),
                           "Vulkan %s failed (VkResult=%d)", What,
                           static_cast<int>(R));
}

VkFilter vkFilter(FilterMode F) {
  return F == FilterMode::Nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
}

VkSamplerAddressMode vkAddress(AddressMode A) {
  switch (A) {
  case AddressMode::Clamp:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  case AddressMode::Repeat:
    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  case AddressMode::Mirror:
    return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  case AddressMode::Border:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  case AddressMode::MirrorOnce:
    return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
  }
  return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
}

VkCompareOp vkCompare(CompareFunction C) {
  switch (C) {
  case CompareFunction::Never:
    return VK_COMPARE_OP_NEVER;
  case CompareFunction::Less:
    return VK_COMPARE_OP_LESS;
  case CompareFunction::Equal:
    return VK_COMPARE_OP_EQUAL;
  case CompareFunction::LessEqual:
    return VK_COMPARE_OP_LESS_OR_EQUAL;
  case CompareFunction::Greater:
    return VK_COMPARE_OP_GREATER;
  case CompareFunction::NotEqual:
    return VK_COMPARE_OP_NOT_EQUAL;
  case CompareFunction::GreaterEqual:
    return VK_COMPARE_OP_GREATER_OR_EQUAL;
  case CompareFunction::Always:
    return VK_COMPARE_OP_ALWAYS;
  }
  return VK_COMPARE_OP_ALWAYS;
}

VkFormat vkFormatForBuffer(const CPUBuffer &Buf) {
  unsigned Ch = Buf.Channels ? Buf.Channels : 1u;
  switch (Buf.Format) {
  case DataFormat::Float32:
    if (Ch == 1) return VK_FORMAT_R32_SFLOAT;
    if (Ch == 2) return VK_FORMAT_R32G32_SFLOAT;
    if (Ch == 3) return VK_FORMAT_R32G32B32_SFLOAT;
    return VK_FORMAT_R32G32B32A32_SFLOAT;
  case DataFormat::Int32:
    if (Ch == 1) return VK_FORMAT_R32_SINT;
    if (Ch == 2) return VK_FORMAT_R32G32_SINT;
    if (Ch == 3) return VK_FORMAT_R32G32B32_SINT;
    return VK_FORMAT_R32G32B32A32_SINT;
  case DataFormat::Hex32:
  case DataFormat::UInt32:
    if (Ch == 1) return VK_FORMAT_R32_UINT;
    if (Ch == 2) return VK_FORMAT_R32G32_UINT;
    if (Ch == 3) return VK_FORMAT_R32G32B32_UINT;
    return VK_FORMAT_R32G32B32A32_UINT;
  case DataFormat::Float16:
    if (Ch == 1) return VK_FORMAT_R16_SFLOAT;
    if (Ch == 2) return VK_FORMAT_R16G16_SFLOAT;
    return VK_FORMAT_R16G16B16A16_SFLOAT;
  case DataFormat::Int16:
    if (Ch == 1) return VK_FORMAT_R16_SINT;
    if (Ch == 2) return VK_FORMAT_R16G16_SINT;
    return VK_FORMAT_R16G16B16A16_SINT;
  case DataFormat::Hex16:
  case DataFormat::UInt16:
    if (Ch == 1) return VK_FORMAT_R16_UINT;
    if (Ch == 2) return VK_FORMAT_R16G16_UINT;
    return VK_FORMAT_R16G16B16A16_UINT;
  case DataFormat::Hex8:
    if (Ch == 1) return VK_FORMAT_R8_UINT;
    if (Ch == 2) return VK_FORMAT_R8G8_UINT;
    return VK_FORMAT_R8G8B8A8_UINT;
  case DataFormat::Int64:
  case DataFormat::UInt64:
  case DataFormat::Hex64:
  case DataFormat::Float64:
    if (Ch == 1) return VK_FORMAT_R32G32_UINT;
    return VK_FORMAT_R32G32B32A32_UINT;
  default:
    return VK_FORMAT_R32G32B32A32_SFLOAT;
  }
}

bool isTextureKind(ResourceKind K) {
  switch (K) {
  case ResourceKind::Texture1D:
  case ResourceKind::Texture1DArray:
  case ResourceKind::Texture2D:
  case ResourceKind::Texture2DArray:
  case ResourceKind::Texture3D:
  case ResourceKind::TextureCube:
  case ResourceKind::TextureCubeArray:
  case ResourceKind::RWTexture1D:
  case ResourceKind::RWTexture1DArray:
  case ResourceKind::RWTexture2D:
  case ResourceKind::RWTexture2DArray:
  case ResourceKind::RWTexture3D:
    return true;
  default:
    return false;
  }
}

bool isTypedBufferKind(ResourceKind K) {
  return K == ResourceKind::Buffer || K == ResourceKind::RWBuffer;
}

bool isUAVKind(ResourceKind K) {
  auto C = classifyResource(K);
  return C && *C == crucible::ResourceClass::UAV;
}

struct VkContext {
  VkInstance Instance = VK_NULL_HANDLE;
  VkPhysicalDevice PhysDev = VK_NULL_HANDLE;
  VkDevice Device = VK_NULL_HANDLE;
  VkQueue Queue = VK_NULL_HANDLE;
  uint32_t ComputeQF = UINT32_MAX;
  uint32_t GraphicsQF = UINT32_MAX;
  VulkanBackendCaps BackendCaps;

  ~VkContext() {
    if (Device)
      vkDestroyDevice(Device, nullptr);
    if (Instance)
      vkDestroyInstance(Instance, nullptr);
  }

  Expected<uint32_t> findMemoryType(uint32_t TypeFilter,
                                    VkMemoryPropertyFlags Props) const {
    VkPhysicalDeviceMemoryProperties MemProps{};
    vkGetPhysicalDeviceMemoryProperties(PhysDev, &MemProps);
    for (uint32_t I = 0; I < MemProps.memoryTypeCount; ++I) {
      if ((TypeFilter & (1u << I)) &&
          (MemProps.memoryTypes[I].propertyFlags & Props) == Props)
        return I;
    }
    return createStringError(inconvertibleErrorCode(),
                             "no suitable Vulkan memory type found");
  }
};

struct VkBufAlloc {
  VkBuffer Buffer = VK_NULL_HANDLE;
  VkDeviceMemory Memory = VK_NULL_HANDLE;
  VkBufferView View = VK_NULL_HANDLE;
  VkDeviceSize SizeBytes = 0;
  bool IsUniform = false;
  bool IsTyped = false;
  bool IsUAV = false;
  bool IsCounter = false;
  CPUBuffer *CPU = nullptr;
  uint32_t Slice = 0;
};

struct VkImgAlloc {
  VkImage Image = VK_NULL_HANDLE;
  VkDeviceMemory Memory = VK_NULL_HANDLE;
  VkImageView View = VK_NULL_HANDLE;
  VkFormat Format = VK_FORMAT_R32G32B32A32_SFLOAT;
  uint32_t Width = 1, Height = 1, Depth = 1, Layers = 1, Mips = 1;
  uint32_t TexelBytes = 16;
  bool IsUAV = false;
  CPUBuffer *CPU = nullptr;
  uint32_t Slice = 0;
};

Error executeVulkanCompute(Pipeline &P, VkContext &VK) {
  if (!P.isCompute())
    return createStringError(
        inconvertibleErrorCode(),
        "Crucible Vulkan runtime currently supports compute pipelines only");

  VkCommandPoolCreateInfo PoolCI{};
  PoolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  PoolCI.queueFamilyIndex = VK.ComputeQF;
  PoolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VkCommandPool CmdPool = VK_NULL_HANDLE;
  if (VkResult R = vkCreateCommandPool(VK.Device, &PoolCI, nullptr, &CmdPool))
    return vkErr(R, "vkCreateCommandPool");

  auto runOneShot = [&](auto &&RecordFn) -> Error {
    VkCommandBufferAllocateInfo AI{};
    AI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    AI.commandPool = CmdPool;
    AI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    AI.commandBufferCount = 1;
    VkCommandBuffer CB = VK_NULL_HANDLE;
    if (VkResult R = vkAllocateCommandBuffers(VK.Device, &AI, &CB))
      return vkErr(R, "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo BI{};
    BI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    BI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (VkResult R = vkBeginCommandBuffer(CB, &BI))
      return vkErr(R, "vkBeginCommandBuffer");
    RecordFn(CB);
    if (VkResult R = vkEndCommandBuffer(CB))
      return vkErr(R, "vkEndCommandBuffer");
    VkSubmitInfo SI{};
    SI.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    SI.commandBufferCount = 1;
    SI.pCommandBuffers = &CB;
    if (VkResult R = vkQueueSubmit(VK.Queue, 1, &SI, VK_NULL_HANDLE))
      return vkErr(R, "vkQueueSubmit");
    if (VkResult R = vkQueueWaitIdle(VK.Queue))
      return vkErr(R, "vkQueueWaitIdle");
    vkFreeCommandBuffers(VK.Device, CmdPool, 1, &CB);
    return Error::success();
  };

  std::vector<VkBufAlloc> AllBufs;
  std::vector<VkImgAlloc> AllImgs;
  std::vector<VkSampler> AllSamplers;

  auto cleanup = [&]() {
    for (VkSampler S : AllSamplers)
      vkDestroySampler(VK.Device, S, nullptr);
    for (auto &Img : AllImgs) {
      if (Img.View) vkDestroyImageView(VK.Device, Img.View, nullptr);
      if (Img.Image) vkDestroyImage(VK.Device, Img.Image, nullptr);
      if (Img.Memory) vkFreeMemory(VK.Device, Img.Memory, nullptr);
    }
    for (auto &Buf : AllBufs) {
      if (Buf.View) vkDestroyBufferView(VK.Device, Buf.View, nullptr);
      if (Buf.Buffer) vkDestroyBuffer(VK.Device, Buf.Buffer, nullptr);
      if (Buf.Memory) vkFreeMemory(VK.Device, Buf.Memory, nullptr);
    }
    if (CmdPool)
      vkDestroyCommandPool(VK.Device, CmdPool, nullptr);
  };

  using BufKey = std::tuple<CPUBuffer *, uint32_t, bool, bool>;
  std::map<BufKey, size_t> BufMap;
  using CounterKey = std::pair<CPUBuffer *, uint32_t>;
  std::map<CounterKey, size_t> CounterBufMap;
  using ImgKey = std::pair<CPUBuffer *, uint32_t>;
  std::map<ImgKey, size_t> ImgMap;
  std::map<YAMLSampler *, VkSampler> SamplerMap;

  auto getOrCreateBuffer = [&](CPUBuffer *CPU, uint32_t Slice, bool IsUniform,
                               bool IsTyped, bool IsUAV) -> Expected<VkBufAlloc *> {
    BufKey Key{CPU, Slice, IsUniform, IsTyped};
    auto It = BufMap.find(Key);
    if (It != BufMap.end()) {
      AllBufs[It->second].IsUAV |= IsUAV;
      return &AllBufs[It->second];
    }

    VkBufAlloc A;
    A.CPU = CPU;
    A.Slice = Slice;
    A.IsUniform = IsUniform;
    A.IsTyped = IsTyped;
    A.IsUAV = IsUAV;
    A.SizeBytes = std::max<VkDeviceSize>(CPU->Size, 16);
    if (IsUniform && (A.SizeBytes % 16) != 0)
      A.SizeBytes = ((A.SizeBytes + 15) / 16) * 16;

    VkBufferCreateInfo BCI{};
    BCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    BCI.size = A.SizeBytes;
    BCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (IsUniform)
      BCI.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    else if (IsTyped)
      BCI.usage |= VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
                   VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
    else
      BCI.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    BCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (VkResult R = vkCreateBuffer(VK.Device, &BCI, nullptr, &A.Buffer))
      return vkErr(R, "vkCreateBuffer");

    VkMemoryRequirements MR{};
    vkGetBufferMemoryRequirements(VK.Device, A.Buffer, &MR);
    auto MemType = VK.findMemoryType(
        MR.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!MemType)
      return MemType.takeError();
    VkMemoryAllocateInfo MAI{};
    MAI.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    MAI.allocationSize = MR.size;
    MAI.memoryTypeIndex = *MemType;
    if (VkResult R = vkAllocateMemory(VK.Device, &MAI, nullptr, &A.Memory))
      return vkErr(R, "vkAllocateMemory");
    if (VkResult R = vkBindBufferMemory(VK.Device, A.Buffer, A.Memory, 0))
      return vkErr(R, "vkBindBufferMemory");

    void *Mapped = nullptr;
    if (VkResult R = vkMapMemory(VK.Device, A.Memory, 0, A.SizeBytes, 0, &Mapped))
      return vkErr(R, "vkMapMemory");
    memset(Mapped, 0, A.SizeBytes);
    if (Slice < CPU->Data.size() && CPU->Data[Slice])
      memcpy(Mapped, CPU->Data[Slice].get(), CPU->Size);
    vkUnmapMemory(VK.Device, A.Memory);

    if (IsTyped) {
      VkBufferViewCreateInfo BVCI{};
      BVCI.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
      BVCI.buffer = A.Buffer;
      BVCI.format = vkFormatForBuffer(*CPU);
      BVCI.offset = 0;
      BVCI.range = A.SizeBytes;
      if (VkResult R = vkCreateBufferView(VK.Device, &BVCI, nullptr, &A.View))
        return vkErr(R, "vkCreateBufferView");
    }

    size_t Idx = AllBufs.size();
    AllBufs.push_back(A);
    BufMap[Key] = Idx;
    return &AllBufs[Idx];
  };

  auto getOrCreateCounterBuffer =
      [&](CPUBuffer *CPU, uint32_t Slice) -> Expected<VkBufAlloc *> {
    CounterKey Key{CPU, Slice};
    auto It = CounterBufMap.find(Key);
    if (It != CounterBufMap.end())
      return &AllBufs[It->second];

    if (CPU->Counters.size() <= Slice)
      CPU->Counters.resize(Slice + 1, 0);

    VkBufAlloc A;
    A.CPU = CPU;
    A.Slice = Slice;
    A.IsUAV = true;
    A.IsCounter = true;
    A.SizeBytes = 16;

    VkBufferCreateInfo BCI{};
    BCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    BCI.size = A.SizeBytes;
    BCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    BCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (VkResult R = vkCreateBuffer(VK.Device, &BCI, nullptr, &A.Buffer))
      return vkErr(R, "vkCreateBuffer (counter)");

    VkMemoryRequirements MR{};
    vkGetBufferMemoryRequirements(VK.Device, A.Buffer, &MR);
    auto MemType = VK.findMemoryType(
        MR.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!MemType)
      return MemType.takeError();
    VkMemoryAllocateInfo MAI{};
    MAI.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    MAI.allocationSize = MR.size;
    MAI.memoryTypeIndex = *MemType;
    if (VkResult R = vkAllocateMemory(VK.Device, &MAI, nullptr, &A.Memory))
      return vkErr(R, "vkAllocateMemory (counter)");
    if (VkResult R = vkBindBufferMemory(VK.Device, A.Buffer, A.Memory, 0))
      return vkErr(R, "vkBindBufferMemory (counter)");

    void *Mapped = nullptr;
    if (VkResult R =
            vkMapMemory(VK.Device, A.Memory, 0, A.SizeBytes, 0, &Mapped))
      return vkErr(R, "vkMapMemory (counter)");
    memset(Mapped, 0, A.SizeBytes);
    uint32_t InitVal = CPU->Counters[Slice];
    memcpy(Mapped, &InitVal, sizeof(InitVal));
    vkUnmapMemory(VK.Device, A.Memory);

    size_t Idx = AllBufs.size();
    AllBufs.push_back(A);
    CounterBufMap[Key] = Idx;
    return &AllBufs[Idx];
  };

  auto getOrCreateImage = [&](CPUBuffer *CPU, uint32_t Slice, ResourceKind Kind,
                              bool IsUAV) -> Expected<VkImgAlloc *> {
    ImgKey Key{CPU, Slice};
    auto It = ImgMap.find(Key);
    if (It != ImgMap.end()) {
      AllImgs[It->second].IsUAV |= IsUAV;
      return &AllImgs[It->second];
    }

    VkImageType ImgType = VK_IMAGE_TYPE_2D;
    VkImageViewType ViewType = VK_IMAGE_VIEW_TYPE_2D;
    VkImageCreateFlags ImgFlags = 0;
    uint32_t W = static_cast<uint32_t>(std::max(1, CPU->OutputProps.Width));
    uint32_t H = static_cast<uint32_t>(std::max(1, CPU->OutputProps.Height));
    uint32_t D = static_cast<uint32_t>(std::max(1, CPU->OutputProps.Depth));
    uint32_t Mips = static_cast<uint32_t>(std::max(1, CPU->OutputProps.MipLevels));
    uint32_t Layers = static_cast<uint32_t>(std::max(1, CPU->OutputProps.ArraySlices));

    switch (Kind) {
    case ResourceKind::Texture1D:
    case ResourceKind::RWTexture1D:
      ImgType = VK_IMAGE_TYPE_1D;
      ViewType = VK_IMAGE_VIEW_TYPE_1D;
      H = 1;
      D = 1;
      Layers = 1;
      break;
    case ResourceKind::Texture1DArray:
    case ResourceKind::RWTexture1DArray:
      ImgType = VK_IMAGE_TYPE_1D;
      ViewType = VK_IMAGE_VIEW_TYPE_1D_ARRAY;
      H = 1;
      D = 1;
      break;
    case ResourceKind::Texture2D:
    case ResourceKind::RWTexture2D:
      ImgType = VK_IMAGE_TYPE_2D;
      ViewType = VK_IMAGE_VIEW_TYPE_2D;
      D = 1;
      Layers = 1;
      break;
    case ResourceKind::Texture2DArray:
    case ResourceKind::RWTexture2DArray:
      ImgType = VK_IMAGE_TYPE_2D;
      ViewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      D = 1;
      break;
    case ResourceKind::Texture3D:
    case ResourceKind::RWTexture3D:
      ImgType = VK_IMAGE_TYPE_3D;
      ViewType = VK_IMAGE_VIEW_TYPE_3D;
      Layers = 1;
      break;
    case ResourceKind::TextureCube:
      ImgType = VK_IMAGE_TYPE_2D;
      ViewType = VK_IMAGE_VIEW_TYPE_CUBE;
      ImgFlags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
      D = 1;
      Layers = 6;
      break;
    case ResourceKind::TextureCubeArray:
      ImgType = VK_IMAGE_TYPE_2D;
      ViewType = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
      ImgFlags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
      D = 1;
      if (Layers < 6)
        Layers = 6;
      break;
    default:
      break;
    }

    VkFormat Fmt = vkFormatForBuffer(*CPU);
    uint32_t TexelBytes = std::max<uint32_t>(
        1u, static_cast<uint32_t>(CPU->getSingleElementSize()) *
                std::max<uint32_t>(1u, CPU->Channels));

    VkImgAlloc A;
    A.Format = Fmt;
    A.Width = W;
    A.Height = H;
    A.Depth = D;
    A.Layers = Layers;
    A.Mips = Mips;
    A.TexelBytes = TexelBytes;
    A.IsUAV = IsUAV;
    A.CPU = CPU;
    A.Slice = Slice;

    VkImageCreateInfo ICI{};
    ICI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ICI.flags = ImgFlags;
    ICI.imageType = ImgType;
    ICI.format = Fmt;
    ICI.extent = {W, H, D};
    ICI.mipLevels = Mips;
    ICI.arrayLayers = Layers;
    ICI.samples = VK_SAMPLE_COUNT_1_BIT;
    ICI.tiling = VK_IMAGE_TILING_OPTIMAL;
    ICI.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ICI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ICI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (VkResult R = vkCreateImage(VK.Device, &ICI, nullptr, &A.Image))
      return vkErr(R, "vkCreateImage (compute texture)");

    VkMemoryRequirements MR{};
    vkGetImageMemoryRequirements(VK.Device, A.Image, &MR);
    auto MemType =
        VK.findMemoryType(MR.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!MemType)
      MemType = VK.findMemoryType(MR.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (!MemType)
      return MemType.takeError();
    VkMemoryAllocateInfo MAI{};
    MAI.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    MAI.allocationSize = MR.size;
    MAI.memoryTypeIndex = *MemType;
    if (VkResult R = vkAllocateMemory(VK.Device, &MAI, nullptr, &A.Memory)) {
      vkDestroyImage(VK.Device, A.Image, nullptr);
      return vkErr(R, "vkAllocateMemory (compute texture)");
    }
    if (VkResult R = vkBindImageMemory(VK.Device, A.Image, A.Memory, 0)) {
      vkFreeMemory(VK.Device, A.Memory, nullptr);
      vkDestroyImage(VK.Device, A.Image, nullptr);
      return vkErr(R, "vkBindImageMemory (compute texture)");
    }

    VkImageViewCreateInfo IVCI{};
    IVCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    IVCI.image = A.Image;
    IVCI.viewType = ViewType;
    IVCI.format = Fmt;
    IVCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, Mips, 0, Layers};
    if (VkResult R = vkCreateImageView(VK.Device, &IVCI, nullptr, &A.View)) {
      vkFreeMemory(VK.Device, A.Memory, nullptr);
      vkDestroyImage(VK.Device, A.Image, nullptr);
      return vkErr(R, "vkCreateImageView (compute texture)");
    }

    // Upload CPU texture data via temporary staging buffer.
    VkDeviceSize StageBytes = std::max<VkDeviceSize>(CPU->Size, 16);
    VkBuffer StageBuf = VK_NULL_HANDLE;
    VkDeviceMemory StageMem = VK_NULL_HANDLE;
    VkBufferCreateInfo SBCI{};
    SBCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    SBCI.size = StageBytes;
    SBCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    SBCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (VkResult R = vkCreateBuffer(VK.Device, &SBCI, nullptr, &StageBuf))
      return vkErr(R, "vkCreateBuffer (tex stage)");
    VkMemoryRequirements SMR{};
    vkGetBufferMemoryRequirements(VK.Device, StageBuf, &SMR);
    auto SMemType = VK.findMemoryType(
        SMR.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!SMemType) {
      vkDestroyBuffer(VK.Device, StageBuf, nullptr);
      return SMemType.takeError();
    }
    VkMemoryAllocateInfo SMAI{};
    SMAI.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    SMAI.allocationSize = SMR.size;
    SMAI.memoryTypeIndex = *SMemType;
    if (VkResult R = vkAllocateMemory(VK.Device, &SMAI, nullptr, &StageMem)) {
      vkDestroyBuffer(VK.Device, StageBuf, nullptr);
      return vkErr(R, "vkAllocateMemory (tex stage)");
    }
    vkBindBufferMemory(VK.Device, StageBuf, StageMem, 0);
    void *Mapped = nullptr;
    vkMapMemory(VK.Device, StageMem, 0, StageBytes, 0, &Mapped);
    memset(Mapped, 0, StageBytes);
    if (Slice < CPU->Data.size() && CPU->Data[Slice])
      memcpy(Mapped, CPU->Data[Slice].get(), CPU->Size);
    vkUnmapMemory(VK.Device, StageMem);

    Error UpErr = runOneShot([&](VkCommandBuffer CB) {
      VkImageMemoryBarrier B1{};
      B1.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      B1.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      B1.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      B1.srcAccessMask = 0;
      B1.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      B1.image = A.Image;
      B1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, Mips, 0, Layers};
      vkCmdPipelineBarrier(CB, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                           nullptr, 1, &B1);

      VkDeviceSize Off = 0;
      for (uint32_t L = 0; L < Layers; ++L) {
        for (uint32_t M = 0; M < Mips; ++M) {
          uint32_t MW = std::max(1u, W >> M);
          uint32_t MH = std::max(1u, H >> M);
          uint32_t MD = std::max(1u, D >> M);
          VkDeviceSize SubBytes =
              static_cast<VkDeviceSize>(MW) * MH * MD * TexelBytes;
          if (Off + SubBytes <= StageBytes) {
            VkBufferImageCopy Reg{};
            Reg.bufferOffset = Off;
            Reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, M, L, 1};
            Reg.imageExtent = {MW, MH, MD};
            vkCmdCopyBufferToImage(CB, StageBuf, A.Image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                   &Reg);
          }
          Off += SubBytes;
        }
      }

      VkImageMemoryBarrier B2{};
      B2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      B2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      B2.newLayout = VK_IMAGE_LAYOUT_GENERAL;
      B2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      B2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      B2.image = A.Image;
      B2.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, Mips, 0, Layers};
      vkCmdPipelineBarrier(CB, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                           0, nullptr, 1, &B2);
    });
    vkDestroyBuffer(VK.Device, StageBuf, nullptr);
    vkFreeMemory(VK.Device, StageMem, nullptr);
    if (UpErr) {
      vkDestroyImageView(VK.Device, A.View, nullptr);
      vkFreeMemory(VK.Device, A.Memory, nullptr);
      vkDestroyImage(VK.Device, A.Image, nullptr);
      return std::move(UpErr);
    }

    size_t Idx = AllImgs.size();
    AllImgs.push_back(A);
    ImgMap[Key] = Idx;
    return &AllImgs[Idx];
  };

  auto getOrCreateSampler = [&](YAMLSampler *S) -> Expected<VkSampler> {
    auto It = SamplerMap.find(S);
    if (It != SamplerMap.end())
      return It->second;
    VkSamplerCreateInfo SCI{};
    SCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    SCI.magFilter = vkFilter(S->MagFilter);
    SCI.minFilter = vkFilter(S->MinFilter);
    SCI.mipmapMode = S->MinFilter == FilterMode::Nearest
                         ? VK_SAMPLER_MIPMAP_MODE_NEAREST
                         : VK_SAMPLER_MIPMAP_MODE_LINEAR;
    SCI.addressModeU = vkAddress(S->Address);
    SCI.addressModeV = vkAddress(S->Address);
    SCI.addressModeW = vkAddress(S->Address);
    SCI.minLod = S->MinLOD;
    SCI.maxLod = S->MaxLOD;
    if (S->Kind == SamplerKind::SamplerComparison) {
      SCI.compareEnable = VK_TRUE;
      SCI.compareOp = vkCompare(S->ComparisonOp);
    }
    VkSampler Samp = VK_NULL_HANDLE;
    if (VkResult R = vkCreateSampler(VK.Device, &SCI, nullptr, &Samp))
      return vkErr(R, "vkCreateSampler");
    AllSamplers.push_back(Samp);
    SamplerMap[S] = Samp;
    return Samp;
  };

  // Compile and dispatch each compute shader in the pipeline.
  for (const Shader &S : P.Shaders) {
    crucible::Options Opts;
    Opts.Target = crucible::TargetBackend::SPIRV;
    Opts.EntryPoint = S.Entry;
    auto CompiledOrErr =
        crucible::compileDXILToSPIRV(S.Shader->getMemBufferRef(), Opts);
    if (!CompiledOrErr) {
      cleanup();
      return CompiledOrErr.takeError();
    }
    const auto &Compiled = *CompiledOrErr;
    const auto &R = Compiled.Reflect;

    if (Compiled.Output.size() % 4 != 0) {
      cleanup();
      return createStringError(inconvertibleErrorCode(),
                               "invalid SPIR-V output size");
    }

    VkShaderModuleCreateInfo SMCI{};
    SMCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    SMCI.codeSize = Compiled.Output.size();
    SMCI.pCode = reinterpret_cast<const uint32_t *>(Compiled.Output.data());
    VkShaderModule ShaderMod = VK_NULL_HANDLE;
    if (VkResult Res =
            vkCreateShaderModule(VK.Device, &SMCI, nullptr, &ShaderMod)) {
      cleanup();
      return vkErr(Res, "vkCreateShaderModule");
    }

    // Group bindings by descriptor set.
    struct BoundResource {
      uint32_t Set = 0;
      uint32_t Binding = 0;
      uint32_t ArrayElem = 0;
      VkDescriptorType DescType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      VkBuffer Buffer = VK_NULL_HANDLE;
      VkDeviceSize BufferSize = 0;
      VkBufferView BufferView = VK_NULL_HANDLE;
      VkSampler Sampler = VK_NULL_HANDLE;
      VkImageView ImageView = VK_NULL_HANDLE;
    };
    std::vector<BoundResource> BoundList;

    auto findReg = [&](crucible::ResourceClass Class, uint32_t Space,
                       uint32_t Reg) -> const crucible::ResourceBinding * {
      for (const crucible::ResourceBinding &RB : R.Resources) {
        if (RB.Class == Class && RB.Space == Space && Reg >= RB.LowerBound &&
            (RB.RangeSize == 0xFFFFFFFFu || Reg - RB.LowerBound < RB.RangeSize))
          return &RB;
      }
      return nullptr;
    };

    for (const DescriptorSet &DS : P.Sets) {
      for (const offloadtest::Resource &Res : DS.Resources) {
        auto Class = classifyResource(Res.Kind);
        if (!Class || !Res.DXBinding)
          continue;
        const auto *Match =
            findReg(*Class, Res.DXBinding->Space, Res.DXBinding->Register);
        if (!Match)
          continue;
        uint32_t SlotOffset = Res.DXBinding->Register - Match->LowerBound;
        uint32_t VkBind = Match->SPIRVBinding;

        if (Res.Kind == ResourceKind::Sampler) {
          if (!Res.SamplerPtr)
            continue;
          auto SampOrErr = getOrCreateSampler(Res.SamplerPtr);
          if (!SampOrErr) {
            vkDestroyShaderModule(VK.Device, ShaderMod, nullptr);
            cleanup();
            return SampOrErr.takeError();
          }
          BoundList.push_back({Match->SPIRVSet, VkBind, SlotOffset,
                               VK_DESCRIPTOR_TYPE_SAMPLER, VK_NULL_HANDLE, 0,
                               VK_NULL_HANDLE, *SampOrErr});
          continue;
        }

        if (!Res.BufferPtr)
          continue;
        uint32_t ArrayCount =
            std::max<uint32_t>(1u, static_cast<uint32_t>(Res.BufferPtr->ArraySize));
        if (Res.BufferPtr->Data.size() > ArrayCount)
          ArrayCount = static_cast<uint32_t>(Res.BufferPtr->Data.size());

        if (isTextureKind(Res.Kind)) {
          bool IsUAV = isUAVKind(Res.Kind);
          for (uint32_t Slice = 0; Slice < ArrayCount; ++Slice) {
            auto ImgOrErr = getOrCreateImage(Res.BufferPtr, Slice, Res.Kind, IsUAV);
            if (!ImgOrErr) {
              vkDestroyShaderModule(VK.Device, ShaderMod, nullptr);
              cleanup();
              return ImgOrErr.takeError();
            }
            VkImgAlloc *IA = *ImgOrErr;
            BoundList.push_back({Match->SPIRVSet, VkBind, SlotOffset + Slice,
                                 IsUAV ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                       : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                 VK_NULL_HANDLE, 0, VK_NULL_HANDLE,
                                 VK_NULL_HANDLE, IA->View});
          }
          continue;
        }

        bool IsUniform = (Res.Kind == ResourceKind::ConstantBuffer);
        bool IsTyped = isTypedBufferKind(Res.Kind);
        bool IsUAV = isUAVKind(Res.Kind);
        VkDescriptorType DT =
            IsUniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
            : IsTyped ? (IsUAV ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER
                               : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER)
                      : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        for (uint32_t Slice = 0; Slice < ArrayCount; ++Slice) {
          auto BufOrErr =
              getOrCreateBuffer(Res.BufferPtr, Slice, IsUniform, IsTyped, IsUAV);
          if (!BufOrErr) {
            vkDestroyShaderModule(VK.Device, ShaderMod, nullptr);
            cleanup();
            return BufOrErr.takeError();
          }
          VkBufAlloc *BA = *BufOrErr;
          BoundList.push_back({Match->SPIRVSet, VkBind, SlotOffset + Slice, DT,
                               BA->Buffer, BA->SizeBytes, BA->View,
                               VK_NULL_HANDLE});
          if (Match->HasCounter) {
            auto CntOrErr = getOrCreateCounterBuffer(Res.BufferPtr, Slice);
            if (!CntOrErr) {
              vkDestroyShaderModule(VK.Device, ShaderMod, nullptr);
              cleanup();
              return CntOrErr.takeError();
            }
            VkBufAlloc *CA = *CntOrErr;
            BoundList.push_back({Match->SPIRVSet, Match->CounterSPIRVBinding,
                                 SlotOffset + Slice,
                                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, CA->Buffer,
                                 CA->SizeBytes, VK_NULL_HANDLE, VK_NULL_HANDLE});
          }
        }
      }
    }

    // Handle DirectX RootConstants and RootDescriptors.
    if (!P.Settings.DX.RootParams.empty()) {
      auto Container = object::DXContainer::create(S.Shader->getMemBufferRef());
      if (!Container) {
        vkDestroyShaderModule(VK.Device, ShaderMod, nullptr);
        cleanup();
        return Container.takeError();
      }
      const auto &RSOpt = Container->getRootSignature();
      if (RSOpt) {
        const auto &RS = *RSOpt;
        size_t Idx = 0;
        for (const auto &Header : RS.param_headers()) {
          if (Idx >= P.Settings.DX.RootParams.size())
            break;
          const auto &RP = P.Settings.DX.RootParams[Idx++];
          if (RP.Kind == dx::RootParamKind::DescriptorTable)
            continue;
          auto ParamOrErr = RS.getParameter(Header);
          if (!ParamOrErr) {
            vkDestroyShaderModule(VK.Device, ShaderMod, nullptr);
            cleanup();
            return ParamOrErr.takeError();
          }
          if (RP.Kind == dx::RootParamKind::Constant) {
            const auto &RC = std::get<dx::RootConstant>(RP.Data);
            if (auto *CV =
                    dyn_cast<object::DirectX::RootConstantView>(&*ParamOrErr)) {
              auto Info = CV->read();
              if (!Info) {
                vkDestroyShaderModule(VK.Device, ShaderMod, nullptr);
                cleanup();
                return Info.takeError();
              }
              if (const auto *Match =
                      findReg(crucible::ResourceClass::CBV, Info->RegisterSpace,
                              Info->ShaderRegister)) {
                auto BufOrErr = getOrCreateBuffer(RC.BufferPtr, 0, true, false,
                                                  false);
                if (!BufOrErr) {
                  vkDestroyShaderModule(VK.Device, ShaderMod, nullptr);
                  cleanup();
                  return BufOrErr.takeError();
                }
                BoundList.push_back(
                    {Match->SPIRVSet, Match->SPIRVBinding,
                     Info->ShaderRegister - Match->LowerBound,
                     VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, (*BufOrErr)->Buffer,
                     (*BufOrErr)->SizeBytes, VK_NULL_HANDLE, VK_NULL_HANDLE});
              }
            }
          } else if (RP.Kind == dx::RootParamKind::RootDescriptor) {
            const auto &RR = std::get<dx::RootResource>(RP.Data);
            auto Class = classifyResource(RR.Kind);
            if (Class) {
              if (auto *DV = dyn_cast<object::DirectX::RootDescriptorView>(
                      &*ParamOrErr)) {
                auto Info = DV->read(RS.getVersion());
                if (!Info) {
                  vkDestroyShaderModule(VK.Device, ShaderMod, nullptr);
                  cleanup();
                  return Info.takeError();
                }
                if (const auto *Match = findReg(*Class, Info->RegisterSpace,
                                                Info->ShaderRegister)) {
                  bool IsUniform = (RR.Kind == ResourceKind::ConstantBuffer);
                  bool IsUAV = isUAVKind(RR.Kind);
                  auto BufOrErr = getOrCreateBuffer(RR.BufferPtr, 0, IsUniform,
                                                    false, IsUAV);
                  if (!BufOrErr) {
                    vkDestroyShaderModule(VK.Device, ShaderMod, nullptr);
                    cleanup();
                    return BufOrErr.takeError();
                  }
                  BoundList.push_back(
                      {Match->SPIRVSet, Match->SPIRVBinding,
                       Info->ShaderRegister - Match->LowerBound,
                       IsUniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                 : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                       (*BufOrErr)->Buffer, (*BufOrErr)->SizeBytes,
                       VK_NULL_HANDLE, VK_NULL_HANDLE});
                }
              }
            }
          }
        }
      }
    }

    uint32_t MaxSet = 0;
    for (const auto &B : BoundList)
      MaxSet = std::max(MaxSet, B.Set + 1u);

    std::vector<VkDescriptorSetLayout> SetLayouts(MaxSet, VK_NULL_HANDLE);
    std::map<VkDescriptorType, uint32_t> PoolCounts;
    for (uint32_t SetIdx = 0; SetIdx < MaxSet; ++SetIdx) {
      struct BindingLayoutInfo {
        VkDescriptorType Type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        uint32_t Count = 0;
      };
      std::map<uint32_t, BindingLayoutInfo> BindingMap;
      for (const auto &B : BoundList) {
        if (B.Set != SetIdx)
          continue;
        auto &Info = BindingMap[B.Binding];
        Info.Type = B.DescType;
        Info.Count = std::max(Info.Count, B.ArrayElem + 1u);
      }
      for (const crucible::ResourceBinding &RB : R.Resources) {
        if (RB.SPIRVSet != SetIdx)
          continue;
        if (RB.RangeSize > 1u && RB.RangeSize != 0xFFFFFFFFu) {
          if (auto It = BindingMap.find(RB.SPIRVBinding); It != BindingMap.end())
            It->second.Count = std::max(It->second.Count, RB.RangeSize);
          if (RB.HasCounter) {
            if (auto It = BindingMap.find(RB.CounterSPIRVBinding);
                It != BindingMap.end())
              It->second.Count = std::max(It->second.Count, RB.RangeSize);
          }
        }
      }
      std::vector<VkDescriptorSetLayoutBinding> LayoutBinds;
      for (const auto &[BindIdx, Info] : BindingMap) {
        VkDescriptorSetLayoutBinding LB{};
        LB.binding = BindIdx;
        LB.descriptorType = Info.Type;
        LB.descriptorCount = Info.Count;
        LB.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        LayoutBinds.push_back(LB);
        PoolCounts[Info.Type] += Info.Count;
      }
      VkDescriptorSetLayoutCreateInfo DSLCI{};
      DSLCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      DSLCI.bindingCount = static_cast<uint32_t>(LayoutBinds.size());
      DSLCI.pBindings = LayoutBinds.data();
      if (VkResult Res = vkCreateDescriptorSetLayout(VK.Device, &DSLCI, nullptr,
                                                     &SetLayouts[SetIdx])) {
        vkDestroyShaderModule(VK.Device, ShaderMod, nullptr);
        cleanup();
        return vkErr(Res, "vkCreateDescriptorSetLayout");
      }
    }

    VkPipelineLayoutCreateInfo PLCI{};
    PLCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    PLCI.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
    PLCI.pSetLayouts = SetLayouts.data();
    VkPipelineLayout PipeLayout = VK_NULL_HANDLE;
    if (VkResult Res =
            vkCreatePipelineLayout(VK.Device, &PLCI, nullptr, &PipeLayout)) {
      for (auto L : SetLayouts)
        vkDestroyDescriptorSetLayout(VK.Device, L, nullptr);
      vkDestroyShaderModule(VK.Device, ShaderMod, nullptr);
      cleanup();
      return vkErr(Res, "vkCreatePipelineLayout");
    }

    VkComputePipelineCreateInfo CPCI{};
    CPCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    CPCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    CPCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    CPCI.stage.module = ShaderMod;
    CPCI.stage.pName = R.EntryName.empty() ? "main" : R.EntryName.c_str();
    CPCI.layout = PipeLayout;
    VkPipeline Pipe = VK_NULL_HANDLE;
    if (VkResult Res = vkCreateComputePipelines(VK.Device, VK_NULL_HANDLE, 1,
                                                &CPCI, nullptr, &Pipe)) {
      vkDestroyPipelineLayout(VK.Device, PipeLayout, nullptr);
      for (auto L : SetLayouts)
        vkDestroyDescriptorSetLayout(VK.Device, L, nullptr);
      vkDestroyShaderModule(VK.Device, ShaderMod, nullptr);
      cleanup();
      return vkErr(Res, "vkCreateComputePipelines");
    }

    VkDescriptorPool DPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> DSets;
    if (MaxSet > 0) {
      std::vector<VkDescriptorPoolSize> PoolSizes;
      for (auto &[DT, Cnt] : PoolCounts)
        PoolSizes.push_back({DT, Cnt});
      if (PoolSizes.empty())
        PoolSizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1});
      VkDescriptorPoolCreateInfo DPCI{};
      DPCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      DPCI.maxSets = MaxSet;
      DPCI.poolSizeCount = static_cast<uint32_t>(PoolSizes.size());
      DPCI.pPoolSizes = PoolSizes.data();
      if (VkResult Res =
              vkCreateDescriptorPool(VK.Device, &DPCI, nullptr, &DPool)) {
        vkDestroyPipeline(VK.Device, Pipe, nullptr);
        vkDestroyPipelineLayout(VK.Device, PipeLayout, nullptr);
        for (auto L : SetLayouts)
          vkDestroyDescriptorSetLayout(VK.Device, L, nullptr);
        vkDestroyShaderModule(VK.Device, ShaderMod, nullptr);
        cleanup();
        return vkErr(Res, "vkCreateDescriptorPool");
      }

      DSets.resize(MaxSet, VK_NULL_HANDLE);
      VkDescriptorSetAllocateInfo DSAI{};
      DSAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      DSAI.descriptorPool = DPool;
      DSAI.descriptorSetCount = MaxSet;
      DSAI.pSetLayouts = SetLayouts.data();
      if (VkResult Res =
              vkAllocateDescriptorSets(VK.Device, &DSAI, DSets.data())) {
        vkDestroyDescriptorPool(VK.Device, DPool, nullptr);
        vkDestroyPipeline(VK.Device, Pipe, nullptr);
        vkDestroyPipelineLayout(VK.Device, PipeLayout, nullptr);
        for (auto L : SetLayouts)
          vkDestroyDescriptorSetLayout(VK.Device, L, nullptr);
        vkDestroyShaderModule(VK.Device, ShaderMod, nullptr);
        cleanup();
        return vkErr(Res, "vkAllocateDescriptorSets");
      }

      std::vector<VkDescriptorBufferInfo> BufInfos(BoundList.size());
      std::vector<VkDescriptorImageInfo> ImgInfos(BoundList.size());
      std::vector<VkWriteDescriptorSet> Writes;
      for (size_t I = 0; I < BoundList.size(); ++I) {
        const auto &B = BoundList[I];
        VkWriteDescriptorSet W{};
        W.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        W.dstSet = DSets[B.Set];
        W.dstBinding = B.Binding;
        W.dstArrayElement = B.ArrayElem;
        W.descriptorCount = 1;
        W.descriptorType = B.DescType;
        if (B.DescType == VK_DESCRIPTOR_TYPE_SAMPLER) {
          ImgInfos[I].sampler = B.Sampler;
          W.pImageInfo = &ImgInfos[I];
        } else if (B.DescType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
                   B.DescType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
          ImgInfos[I].imageView = B.ImageView;
          ImgInfos[I].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
          W.pImageInfo = &ImgInfos[I];
        } else if (B.DescType == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER ||
                   B.DescType == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
          W.pTexelBufferView = &BoundList[I].BufferView;
        } else {
          BufInfos[I] = {B.Buffer, 0, B.BufferSize};
          W.pBufferInfo = &BufInfos[I];
        }
        Writes.push_back(W);
      }
      if (!Writes.empty())
        vkUpdateDescriptorSets(VK.Device, static_cast<uint32_t>(Writes.size()),
                               Writes.data(), 0, nullptr);
    }

    Error DispatchErr = runOneShot([&](VkCommandBuffer CB) {
      vkCmdBindPipeline(CB, VK_PIPELINE_BIND_POINT_COMPUTE, Pipe);
      if (!DSets.empty()) {
        vkCmdBindDescriptorSets(CB, VK_PIPELINE_BIND_POINT_COMPUTE, PipeLayout,
                                0, static_cast<uint32_t>(DSets.size()),
                                DSets.data(), 0, nullptr);
      }
      const auto &G = P.DispatchParameters.DispatchGroupCount;
      vkCmdDispatch(CB, static_cast<uint32_t>(G[0]),
                    static_cast<uint32_t>(G[1]), static_cast<uint32_t>(G[2]));
    });

    if (DPool) vkDestroyDescriptorPool(VK.Device, DPool, nullptr);
    vkDestroyPipeline(VK.Device, Pipe, nullptr);
    vkDestroyPipelineLayout(VK.Device, PipeLayout, nullptr);
    for (auto L : SetLayouts)
      vkDestroyDescriptorSetLayout(VK.Device, L, nullptr);
    vkDestroyShaderModule(VK.Device, ShaderMod, nullptr);

    if (DispatchErr) {
      cleanup();
      return DispatchErr;
    }
  }

  // Copy back UAV buffers and UAV counters to CPU buffers for verification.
  for (const auto &Buf : AllBufs) {
    if (!Buf.IsUAV || !Buf.CPU)
      continue;
    if (Buf.IsCounter) {
      if (Buf.Slice >= Buf.CPU->Counters.size())
        Buf.CPU->Counters.resize(Buf.Slice + 1, 0);
      void *Mapped = nullptr;
      if (VkResult R =
              vkMapMemory(VK.Device, Buf.Memory, 0, Buf.SizeBytes, 0, &Mapped)) {
        cleanup();
        return vkErr(R, "vkMapMemory (counter readback)");
      }
      uint32_t Val = 0;
      memcpy(&Val, Mapped, sizeof(Val));
      Buf.CPU->Counters[Buf.Slice] = Val;
      vkUnmapMemory(VK.Device, Buf.Memory);
      continue;
    }
    if (Buf.Slice >= Buf.CPU->Data.size())
      continue;
    void *Mapped = nullptr;
    if (VkResult R =
            vkMapMemory(VK.Device, Buf.Memory, 0, Buf.SizeBytes, 0, &Mapped)) {
      cleanup();
      return vkErr(R, "vkMapMemory (readback)");
    }
    memcpy(Buf.CPU->Data[Buf.Slice].get(), Mapped, Buf.CPU->Size);
    vkUnmapMemory(VK.Device, Buf.Memory);
  }

  // Copy back UAV textures to CPU buffers.
  for (const auto &Img : AllImgs) {
    if (!Img.IsUAV || !Img.CPU || Img.Slice >= Img.CPU->Data.size())
      continue;
    VkDeviceSize StageBytes = std::max<VkDeviceSize>(Img.CPU->Size, 16);
    VkBuffer StageBuf = VK_NULL_HANDLE;
    VkDeviceMemory StageMem = VK_NULL_HANDLE;
    VkBufferCreateInfo SBCI{};
    SBCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    SBCI.size = StageBytes;
    SBCI.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    SBCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (VkResult R = vkCreateBuffer(VK.Device, &SBCI, nullptr, &StageBuf)) {
      cleanup();
      return vkErr(R, "vkCreateBuffer (tex readback)");
    }
    VkMemoryRequirements SMR{};
    vkGetBufferMemoryRequirements(VK.Device, StageBuf, &SMR);
    auto SMemType = VK.findMemoryType(
        SMR.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!SMemType) {
      vkDestroyBuffer(VK.Device, StageBuf, nullptr);
      cleanup();
      return SMemType.takeError();
    }
    VkMemoryAllocateInfo SMAI{};
    SMAI.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    SMAI.allocationSize = SMR.size;
    SMAI.memoryTypeIndex = *SMemType;
    if (VkResult R = vkAllocateMemory(VK.Device, &SMAI, nullptr, &StageMem)) {
      vkDestroyBuffer(VK.Device, StageBuf, nullptr);
      cleanup();
      return vkErr(R, "vkAllocateMemory (tex readback)");
    }
    vkBindBufferMemory(VK.Device, StageBuf, StageMem, 0);

    Error RbErr = runOneShot([&](VkCommandBuffer CB) {
      VkImageMemoryBarrier B1{};
      B1.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      B1.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
      B1.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      B1.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      B1.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      B1.image = Img.Image;
      B1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, Img.Mips, 0,
                             Img.Layers};
      vkCmdPipelineBarrier(CB, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                           nullptr, 1, &B1);

      VkDeviceSize Off = 0;
      for (uint32_t L = 0; L < Img.Layers; ++L) {
        for (uint32_t M = 0; M < Img.Mips; ++M) {
          uint32_t MW = std::max(1u, Img.Width >> M);
          uint32_t MH = std::max(1u, Img.Height >> M);
          uint32_t MD = std::max(1u, Img.Depth >> M);
          VkDeviceSize SubBytes =
              static_cast<VkDeviceSize>(MW) * MH * MD * Img.TexelBytes;
          if (Off + SubBytes <= StageBytes) {
            VkBufferImageCopy Reg{};
            Reg.bufferOffset = Off;
            Reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, M, L, 1};
            Reg.imageExtent = {MW, MH, MD};
            vkCmdCopyImageToBuffer(CB, Img.Image,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   StageBuf, 1, &Reg);
          }
          Off += SubBytes;
        }
      }
    });
    if (!RbErr) {
      void *Mapped = nullptr;
      if (vkMapMemory(VK.Device, StageMem, 0, StageBytes, 0, &Mapped) ==
          VK_SUCCESS) {
        memcpy(Img.CPU->Data[Img.Slice].get(), Mapped, Img.CPU->Size);
        vkUnmapMemory(VK.Device, StageMem);
      }
    }
    vkDestroyBuffer(VK.Device, StageBuf, nullptr);
    vkFreeMemory(VK.Device, StageMem, nullptr);
    if (RbErr) {
      cleanup();
      return RbErr;
    }
  }

  cleanup();
  return Error::success();
}

VkFormat vkFormatForAttr(DataFormat Fmt, unsigned Ch) {
  CPUBuffer Tmp;
  Tmp.Format = Fmt;
  Tmp.Channels = Ch;
  return vkFormatForBuffer(Tmp);
}

Error executeVulkanRaster(Pipeline &P, VkContext &VK) {
  auto Fail = [](const char *Msg) {
    return createStringError(inconvertibleErrorCode(), "%s", Msg);
  };
  if (VK.GraphicsQF == UINT32_MAX)
    return Fail("no Vulkan graphics queue family available");
  if (!P.isTraditionalRaster() || P.Shaders.size() < 2)
    return Fail("raster execution requires one vertex and one pixel shader");

  const Shader *VS = nullptr, *PS = nullptr;
  for (const Shader &S : P.Shaders) {
    if (S.Stage == Stages::Vertex)
      VS = &S;
    else if (S.Stage == Stages::Pixel)
      PS = &S;
    else
      return Fail("Vulkan raster currently supports VS + PS pipelines only");
  }
  if (!VS || !PS)
    return Fail("raster execution requires one vertex and one pixel shader");

  auto &B = P.Bindings;
  CPUBuffer *Output = B.RTargetBufferPtr;
  if (!Output || Output->OutputProps.Width < 1 ||
      Output->OutputProps.Height < 1 || Output->OutputProps.Depth > 1 ||
      Output->OutputProps.ArraySlices > 1 || Output->OutputProps.MipLevels > 1)
    return Fail("raster execution requires a single 2D color target");

  uint32_t Width = static_cast<uint32_t>(Output->OutputProps.Width);
  uint32_t Height = static_cast<uint32_t>(Output->OutputProps.Height);
  size_t PixelSize = Output->getSingleElementSize() * Output->Channels;
  if (!PixelSize || Width > Output->Size / PixelSize / Height)
    return Fail("color target exceeds its backing storage");
  VkDeviceSize ReadbackBytes = static_cast<VkDeviceSize>(Width) * Height * PixelSize;
  VkFormat ColorFmt = vkFormatForBuffer(*Output);

  crucible::Options VSOpts;
  VSOpts.Target = crucible::TargetBackend::SPIRV;
  VSOpts.EntryPoint = VS->Entry;
  auto VSCompiled =
      crucible::compileDXILToSPIRV(VS->Shader->getMemBufferRef(), VSOpts);
  if (!VSCompiled)
    return VSCompiled.takeError();

  crucible::Options PSOpts;
  PSOpts.Target = crucible::TargetBackend::SPIRV;
  PSOpts.EntryPoint = PS->Entry;
  auto PSCompiled =
      crucible::compileDXILToSPIRV(PS->Shader->getMemBufferRef(), PSOpts);
  if (!PSCompiled)
    return PSCompiled.takeError();

  VkShaderModule VSMod = VK_NULL_HANDLE, PSMod = VK_NULL_HANDLE;
  auto createMod = [&](const std::string &Bytes,
                       VkShaderModule &Out) -> Error {
    VkShaderModuleCreateInfo CI{};
    CI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    CI.codeSize = Bytes.size();
    CI.pCode = reinterpret_cast<const uint32_t *>(Bytes.data());
    if (VkResult R = vkCreateShaderModule(VK.Device, &CI, nullptr, &Out))
      return vkErr(R, "vkCreateShaderModule");
    return Error::success();
  };
  if (Error E = createMod(VSCompiled->Output, VSMod))
    return E;
  if (Error E = createMod(PSCompiled->Output, PSMod)) {
    vkDestroyShaderModule(VK.Device, VSMod, nullptr);
    return E;
  }

  // Offscreen color attachment image + view (+ optional MSAA and depth/stencil).
  bool HasStencil = P.Stencil && P.Stencil->Enabled;
  uint32_t SampleCount = std::max<uint32_t>(1u, B.SampleCount);
  VkSampleCountFlagBits SampleBits = static_cast<VkSampleCountFlagBits>(SampleCount);
  VkFormat DepthFmt = VK_FORMAT_D32_SFLOAT_S8_UINT;

  VkImage ColorImg = VK_NULL_HANDLE;
  VkDeviceMemory ColorMem = VK_NULL_HANDLE;
  VkImageView ColorView = VK_NULL_HANDLE;
  VkImage MSAAImg = VK_NULL_HANDLE;
  VkDeviceMemory MSAAMem = VK_NULL_HANDLE;
  VkImageView MSAAView = VK_NULL_HANDLE;
  VkImage DepthImg = VK_NULL_HANDLE;
  VkDeviceMemory DepthMem = VK_NULL_HANDLE;
  VkImageView DepthView = VK_NULL_HANDLE;
  VkBuffer ReadbackBuf = VK_NULL_HANDLE;
  VkDeviceMemory ReadbackMem = VK_NULL_HANDLE;
  VkBuffer VertBuf = VK_NULL_HANDLE;
  VkDeviceMemory VertMem = VK_NULL_HANDLE;
  VkRenderPass RenderPass = VK_NULL_HANDLE;
  VkFramebuffer FB = VK_NULL_HANDLE;
  VkPipelineLayout PipeLayout = VK_NULL_HANDLE;
  VkPipeline Pipe = VK_NULL_HANDLE;
  VkCommandPool CmdPool = VK_NULL_HANDLE;
  VkDescriptorPool DPool = VK_NULL_HANDLE;
  std::vector<VkDescriptorSetLayout> SetLayouts;

  struct RasterBuf {
    VkBuffer Buffer = VK_NULL_HANDLE;
    VkDeviceMemory Memory = VK_NULL_HANDLE;
    VkBufferView View = VK_NULL_HANDLE;
    VkDeviceSize SizeBytes = 0;
    CPUBuffer *CPU = nullptr;
    uint32_t Slice = 0;
    bool IsUAV = false;
  };
  struct RasterImg {
    VkImage Image = VK_NULL_HANDLE;
    VkDeviceMemory Memory = VK_NULL_HANDLE;
    VkImageView View = VK_NULL_HANDLE;
  };
  std::vector<RasterBuf> RasterBufs;
  std::vector<RasterImg> RasterImgs;
  std::vector<VkSampler> RasterSamplers;

  auto cleanup = [&]() {
    if (DPool) vkDestroyDescriptorPool(VK.Device, DPool, nullptr);
    for (auto L : SetLayouts)
      if (L) vkDestroyDescriptorSetLayout(VK.Device, L, nullptr);
    for (auto S : RasterSamplers)
      if (S) vkDestroySampler(VK.Device, S, nullptr);
    for (auto &RI : RasterImgs) {
      if (RI.View) vkDestroyImageView(VK.Device, RI.View, nullptr);
      if (RI.Image) vkDestroyImage(VK.Device, RI.Image, nullptr);
      if (RI.Memory) vkFreeMemory(VK.Device, RI.Memory, nullptr);
    }
    for (auto &RB : RasterBufs) {
      if (RB.View) vkDestroyBufferView(VK.Device, RB.View, nullptr);
      if (RB.Buffer) vkDestroyBuffer(VK.Device, RB.Buffer, nullptr);
      if (RB.Memory) vkFreeMemory(VK.Device, RB.Memory, nullptr);
    }
    if (CmdPool) vkDestroyCommandPool(VK.Device, CmdPool, nullptr);
    if (Pipe) vkDestroyPipeline(VK.Device, Pipe, nullptr);
    if (PipeLayout) vkDestroyPipelineLayout(VK.Device, PipeLayout, nullptr);
    if (FB) vkDestroyFramebuffer(VK.Device, FB, nullptr);
    if (RenderPass) vkDestroyRenderPass(VK.Device, RenderPass, nullptr);
    if (VertBuf) vkDestroyBuffer(VK.Device, VertBuf, nullptr);
    if (VertMem) vkFreeMemory(VK.Device, VertMem, nullptr);
    if (ReadbackBuf) vkDestroyBuffer(VK.Device, ReadbackBuf, nullptr);
    if (ReadbackMem) vkFreeMemory(VK.Device, ReadbackMem, nullptr);
    if (DepthView) vkDestroyImageView(VK.Device, DepthView, nullptr);
    if (DepthImg) vkDestroyImage(VK.Device, DepthImg, nullptr);
    if (DepthMem) vkFreeMemory(VK.Device, DepthMem, nullptr);
    if (MSAAView) vkDestroyImageView(VK.Device, MSAAView, nullptr);
    if (MSAAImg) vkDestroyImage(VK.Device, MSAAImg, nullptr);
    if (MSAAMem) vkFreeMemory(VK.Device, MSAAMem, nullptr);
    if (ColorView) vkDestroyImageView(VK.Device, ColorView, nullptr);
    if (ColorImg) vkDestroyImage(VK.Device, ColorImg, nullptr);
    if (ColorMem) vkFreeMemory(VK.Device, ColorMem, nullptr);
    if (PSMod) vkDestroyShaderModule(VK.Device, PSMod, nullptr);
    if (VSMod) vkDestroyShaderModule(VK.Device, VSMod, nullptr);
  };

  VkImageCreateInfo ICI{};
  ICI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ICI.imageType = VK_IMAGE_TYPE_2D;
  ICI.format = ColorFmt;
  ICI.extent = {Width, Height, 1};
  ICI.mipLevels = 1;
  ICI.arrayLayers = 1;
  ICI.samples = VK_SAMPLE_COUNT_1_BIT;
  ICI.tiling = VK_IMAGE_TILING_OPTIMAL;
  ICI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  ICI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ICI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (VkResult R = vkCreateImage(VK.Device, &ICI, nullptr, &ColorImg)) {
    cleanup();
    return vkErr(R, "vkCreateImage");
  }
  VkMemoryRequirements IMR{};
  vkGetImageMemoryRequirements(VK.Device, ColorImg, &IMR);
  auto ImgMemType = VK.findMemoryType(IMR.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (!ImgMemType) {
    cleanup();
    return ImgMemType.takeError();
  }
  VkMemoryAllocateInfo IMAI{};
  IMAI.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  IMAI.allocationSize = IMR.size;
  IMAI.memoryTypeIndex = *ImgMemType;
  if (VkResult R = vkAllocateMemory(VK.Device, &IMAI, nullptr, &ColorMem)) {
    cleanup();
    return vkErr(R, "vkAllocateMemory (color image)");
  }
  if (VkResult R = vkBindImageMemory(VK.Device, ColorImg, ColorMem, 0)) {
    cleanup();
    return vkErr(R, "vkBindImageMemory");
  }

  VkImageViewCreateInfo IVCI{};
  IVCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  IVCI.image = ColorImg;
  IVCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
  IVCI.format = ColorFmt;
  IVCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  if (VkResult R = vkCreateImageView(VK.Device, &IVCI, nullptr, &ColorView)) {
    cleanup();
    return vkErr(R, "vkCreateImageView");
  }

  // Host-visible staging buffer for color target readback.
  auto allocHostBuf = [&](VkDeviceSize Size, VkBufferUsageFlags Usage,
                          const void *InitData, VkBuffer &BufOut,
                          VkDeviceMemory &MemOut) -> Error {
    VkBufferCreateInfo BCI{};
    BCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    BCI.size = std::max<VkDeviceSize>(Size, 16);
    BCI.usage = Usage;
    BCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (VkResult R = vkCreateBuffer(VK.Device, &BCI, nullptr, &BufOut))
      return vkErr(R, "vkCreateBuffer");
    VkMemoryRequirements MR{};
    vkGetBufferMemoryRequirements(VK.Device, BufOut, &MR);
    auto MT = VK.findMemoryType(MR.memoryTypeBits,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!MT)
      return MT.takeError();
    VkMemoryAllocateInfo MAI{};
    MAI.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    MAI.allocationSize = MR.size;
    MAI.memoryTypeIndex = *MT;
    if (VkResult R = vkAllocateMemory(VK.Device, &MAI, nullptr, &MemOut))
      return vkErr(R, "vkAllocateMemory");
    if (VkResult R = vkBindBufferMemory(VK.Device, BufOut, MemOut, 0))
      return vkErr(R, "vkBindBufferMemory");
    if (InitData) {
      void *Mapped = nullptr;
      if (VkResult R = vkMapMemory(VK.Device, MemOut, 0, Size, 0, &Mapped))
        return vkErr(R, "vkMapMemory");
      memcpy(Mapped, InitData, Size);
      vkUnmapMemory(VK.Device, MemOut);
    }
    return Error::success();
  };

  if (Error E = allocHostBuf(ReadbackBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             nullptr, ReadbackBuf, ReadbackMem)) {
    cleanup();
    return E;
  }

  uint32_t Stride = B.VertexBufferPtr ? static_cast<uint32_t>(B.VertexBufferPtr->Stride) : 0;
  if (!Stride)
    Stride = static_cast<uint32_t>(B.getVertexStride());
  uint32_t VertexCount = 3;
  if (B.VertexBufferPtr) {
    if (Stride > 0)
      VertexCount = static_cast<uint32_t>(B.VertexBufferPtr->Size / Stride);
    if (Error E = allocHostBuf(B.VertexBufferPtr->Size,
                               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                               B.VertexBufferPtr->Data[0].get(), VertBuf,
                               VertMem)) {
      cleanup();
      return E;
    }
  }

  // RenderPass & Framebuffer.
  // Optional MSAA color target.
  if (SampleCount > 1u) {
    VkImageCreateInfo MCI{};
    MCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    MCI.imageType = VK_IMAGE_TYPE_2D;
    MCI.format = ColorFmt;
    MCI.extent = {Width, Height, 1};
    MCI.mipLevels = 1;
    MCI.arrayLayers = 1;
    MCI.samples = SampleBits;
    MCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    MCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    MCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    MCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (VkResult R = vkCreateImage(VK.Device, &MCI, nullptr, &MSAAImg)) {
      cleanup();
      return vkErr(R, "vkCreateImage (MSAA)");
    }
    VkMemoryRequirements MMR{};
    vkGetImageMemoryRequirements(VK.Device, MSAAImg, &MMR);
    auto MMemType =
        VK.findMemoryType(MMR.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!MMemType)
      MMemType = VK.findMemoryType(MMR.memoryTypeBits,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (!MMemType) {
      cleanup();
      return MMemType.takeError();
    }
    VkMemoryAllocateInfo MMAI{};
    MMAI.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    MMAI.allocationSize = MMR.size;
    MMAI.memoryTypeIndex = *MMemType;
    if (VkResult R = vkAllocateMemory(VK.Device, &MMAI, nullptr, &MSAAMem)) {
      cleanup();
      return vkErr(R, "vkAllocateMemory (MSAA)");
    }
    vkBindImageMemory(VK.Device, MSAAImg, MSAAMem, 0);
    VkImageViewCreateInfo MIVCI{};
    MIVCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    MIVCI.image = MSAAImg;
    MIVCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    MIVCI.format = ColorFmt;
    MIVCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (VkResult R = vkCreateImageView(VK.Device, &MIVCI, nullptr, &MSAAView)) {
      cleanup();
      return vkErr(R, "vkCreateImageView (MSAA)");
    }
  }

  // Depth/stencil attachment image + view.
  {
    VkImageCreateInfo DCI{};
    DCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    DCI.imageType = VK_IMAGE_TYPE_2D;
    DCI.format = DepthFmt;
    DCI.extent = {Width, Height, 1};
    DCI.mipLevels = 1;
    DCI.arrayLayers = 1;
    DCI.samples = SampleBits;
    DCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    DCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    DCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    DCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (VkResult R = vkCreateImage(VK.Device, &DCI, nullptr, &DepthImg)) {
      cleanup();
      return vkErr(R, "vkCreateImage (depth/stencil)");
    }
    VkMemoryRequirements DMR{};
    vkGetImageMemoryRequirements(VK.Device, DepthImg, &DMR);
    auto DMemType =
        VK.findMemoryType(DMR.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!DMemType)
      DMemType = VK.findMemoryType(DMR.memoryTypeBits,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (!DMemType) {
      cleanup();
      return DMemType.takeError();
    }
    VkMemoryAllocateInfo DMAI{};
    DMAI.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    DMAI.allocationSize = DMR.size;
    DMAI.memoryTypeIndex = *DMemType;
    if (VkResult R = vkAllocateMemory(VK.Device, &DMAI, nullptr, &DepthMem)) {
      cleanup();
      return vkErr(R, "vkAllocateMemory (depth/stencil)");
    }
    vkBindImageMemory(VK.Device, DepthImg, DepthMem, 0);
    VkImageViewCreateInfo DIVCI{};
    DIVCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    DIVCI.image = DepthImg;
    DIVCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    DIVCI.format = DepthFmt;
    DIVCI.subresourceRange = {
        VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
    if (VkResult R = vkCreateImageView(VK.Device, &DIVCI, nullptr, &DepthView)) {
      cleanup();
      return vkErr(R, "vkCreateImageView (depth/stencil)");
    }
  }

  std::vector<VkAttachmentDescription> Atts;
  VkAttachmentDescription ColorAtt{};
  ColorAtt.format = ColorFmt;
  ColorAtt.samples = SampleBits;
  ColorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  ColorAtt.storeOp = SampleCount > 1u ? VK_ATTACHMENT_STORE_OP_DONT_CARE
                                      : VK_ATTACHMENT_STORE_OP_STORE;
  ColorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  ColorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  ColorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  ColorAtt.finalLayout = SampleCount > 1u
                             ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                             : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  Atts.push_back(ColorAtt);

  VkAttachmentDescription DepthAtt{};
  DepthAtt.format = DepthFmt;
  DepthAtt.samples = SampleBits;
  DepthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  DepthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  DepthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  DepthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  DepthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  DepthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  Atts.push_back(DepthAtt);

  VkAttachmentReference ColorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkAttachmentReference DepthRef{1,
                                 VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
  VkAttachmentReference ResolveRef{2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

  if (SampleCount > 1u) {
    VkAttachmentDescription ResolveAtt{};
    ResolveAtt.format = ColorFmt;
    ResolveAtt.samples = VK_SAMPLE_COUNT_1_BIT;
    ResolveAtt.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    ResolveAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ResolveAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    ResolveAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    ResolveAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ResolveAtt.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    Atts.push_back(ResolveAtt);
  }

  VkSubpassDescription Subpass{};
  Subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  Subpass.colorAttachmentCount = 1;
  Subpass.pColorAttachments = &ColorRef;
  Subpass.pDepthStencilAttachment = &DepthRef;
  if (SampleCount > 1u)
    Subpass.pResolveAttachments = &ResolveRef;

  VkRenderPassCreateInfo RPCI{};
  RPCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  RPCI.attachmentCount = static_cast<uint32_t>(Atts.size());
  RPCI.pAttachments = Atts.data();
  RPCI.subpassCount = 1;
  RPCI.pSubpasses = &Subpass;
  if (VkResult R = vkCreateRenderPass(VK.Device, &RPCI, nullptr, &RenderPass)) {
    cleanup();
    return vkErr(R, "vkCreateRenderPass");
  }

  SmallVector<VkImageView, 3> FBViews;
  if (SampleCount > 1u) {
    FBViews.push_back(MSAAView);
    FBViews.push_back(DepthView);
    FBViews.push_back(ColorView);
  } else {
    FBViews.push_back(ColorView);
    FBViews.push_back(DepthView);
  }
  VkFramebufferCreateInfo FBCI{};
  FBCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  FBCI.renderPass = RenderPass;
  FBCI.attachmentCount = static_cast<uint32_t>(FBViews.size());
  FBCI.pAttachments = FBViews.data();
  FBCI.width = Width;
  FBCI.height = Height;
  FBCI.layers = 1;
  if (VkResult R = vkCreateFramebuffer(VK.Device, &FBCI, nullptr, &FB)) {
    cleanup();
    return vkErr(R, "vkCreateFramebuffer");
  }

  // Vertex input state matching SPIR-V VS input locations.
  std::vector<VkVertexInputBindingDescription> VBindings;
  std::vector<VkVertexInputAttributeDescription> VAttrs;
  if (B.VertexBufferPtr && Stride > 0) {
    VBindings.push_back({0, Stride, VK_VERTEX_INPUT_RATE_VERTEX});
    uint32_t NextLoc = 0;
    for (const auto &Input : VSCompiled->Reflect.Inputs) {
      if (Input.Kind != 0)
        continue;
      std::string Indexed = Input.Semantic + std::to_string(Input.SemanticIndex);
      auto It = llvm::find_if(B.VertexAttributes, [&](const VertexAttribute &A) {
        return StringRef(A.Name).equals_insensitive(Input.Semantic) ||
               StringRef(A.Name).equals_insensitive(Indexed);
      });
      if (It != B.VertexAttributes.end()) {
        VAttrs.push_back({NextLoc, 0,
                          vkFormatForAttr(It->Format, It->Channels),
                          static_cast<uint32_t>(It->Offset)});
      }
      NextLoc += std::max(1u, Input.Rows);
    }
  }

  // Create command pool early so texture uploads can run before pipeline draw.
  VkCommandPoolCreateInfo PoolCI{};
  PoolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  PoolCI.queueFamilyIndex = VK.GraphicsQF;
  if (VkResult R = vkCreateCommandPool(VK.Device, &PoolCI, nullptr, &CmdPool)) {
    cleanup();
    return vkErr(R, "vkCreateCommandPool");
  }

  auto runRasterOneShot = [&](auto &&Record) -> Error {
    VkCommandBufferAllocateInfo AI{};
    AI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    AI.commandPool = CmdPool;
    AI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    AI.commandBufferCount = 1;
    VkCommandBuffer TmpCB = VK_NULL_HANDLE;
    if (VkResult R = vkAllocateCommandBuffers(VK.Device, &AI, &TmpCB))
      return vkErr(R, "vkAllocateCommandBuffers (raster upload)");
    VkCommandBufferBeginInfo BI{};
    BI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    BI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(TmpCB, &BI);
    Record(TmpCB);
    vkEndCommandBuffer(TmpCB);
    VkSubmitInfo SI{};
    SI.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    SI.commandBufferCount = 1;
    SI.pCommandBuffers = &TmpCB;
    if (VkResult R = vkQueueSubmit(VK.Queue, 1, &SI, VK_NULL_HANDLE)) {
      vkFreeCommandBuffers(VK.Device, CmdPool, 1, &TmpCB);
      return vkErr(R, "vkQueueSubmit (raster upload)");
    }
    vkQueueWaitIdle(VK.Queue);
    vkFreeCommandBuffers(VK.Device, CmdPool, 1, &TmpCB);
    return Error::success();
  };

  struct RasterBound {
    uint32_t Set = 0;
    uint32_t Binding = 0;
    uint32_t ArrayElem = 0;
    VkDescriptorType DescType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    VkBuffer Buffer = VK_NULL_HANDLE;
    VkDeviceSize BufferSize = 0;
    VkBufferView BufferView = VK_NULL_HANDLE;
    VkSampler Sampler = VK_NULL_HANDLE;
    VkImageView ImageView = VK_NULL_HANDLE;
  };
  std::vector<RasterBound> BoundList;

  auto findStageReg = [&](const crucible::Reflection &Ref,
                          crucible::ResourceClass Class, uint32_t Space,
                          uint32_t Reg) -> const crucible::ResourceBinding * {
    for (const crucible::ResourceBinding &RB : Ref.Resources) {
      if (RB.Class == Class && RB.Space == Space && Reg >= RB.LowerBound &&
          (RB.RangeSize == 0xFFFFFFFFu || Reg - RB.LowerBound < RB.RangeSize))
        return &RB;
    }
    return nullptr;
  };

  for (const DescriptorSet &DS : P.Sets) {
    for (const offloadtest::Resource &Res : DS.Resources) {
      auto Class = classifyResource(Res.Kind);
      if (!Class || !Res.DXBinding)
        continue;
      SmallVector<const crucible::ResourceBinding *, 2> Matches;
      if (const auto *M = findStageReg(VSCompiled->Reflect, *Class,
                                       Res.DXBinding->Space,
                                       Res.DXBinding->Register))
        Matches.push_back(M);
      if (const auto *M = findStageReg(PSCompiled->Reflect, *Class,
                                       Res.DXBinding->Space,
                                       Res.DXBinding->Register)) {
        bool Dup = false;
        for (const auto *E : Matches)
          if (E->SPIRVSet == M->SPIRVSet && E->SPIRVBinding == M->SPIRVBinding)
            Dup = true;
        if (!Dup)
          Matches.push_back(M);
      }
      if (Matches.empty())
        continue;

      if (Res.Kind == ResourceKind::Sampler) {
        if (!Res.SamplerPtr)
          continue;
        VkSamplerCreateInfo SCI{};
        SCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        SCI.magFilter = vkFilter(Res.SamplerPtr->MagFilter);
        SCI.minFilter = vkFilter(Res.SamplerPtr->MinFilter);
        SCI.mipmapMode = Res.SamplerPtr->MinFilter == FilterMode::Nearest
                             ? VK_SAMPLER_MIPMAP_MODE_NEAREST
                             : VK_SAMPLER_MIPMAP_MODE_LINEAR;
        SCI.addressModeU = vkAddress(Res.SamplerPtr->Address);
        SCI.addressModeV = vkAddress(Res.SamplerPtr->Address);
        SCI.addressModeW = vkAddress(Res.SamplerPtr->Address);
        SCI.minLod = Res.SamplerPtr->MinLOD;
        SCI.maxLod = Res.SamplerPtr->MaxLOD;
        if (Res.SamplerPtr->Kind == SamplerKind::SamplerComparison) {
          SCI.compareEnable = VK_TRUE;
          SCI.compareOp = vkCompare(Res.SamplerPtr->ComparisonOp);
        }
        VkSampler Samp = VK_NULL_HANDLE;
        if (VkResult R = vkCreateSampler(VK.Device, &SCI, nullptr, &Samp)) {
          cleanup();
          return vkErr(R, "vkCreateSampler (raster)");
        }
        RasterSamplers.push_back(Samp);
        for (const auto *Match : Matches) {
          uint32_t SlotOffset = Res.DXBinding->Register - Match->LowerBound;
          BoundList.push_back({Match->SPIRVSet, Match->SPIRVBinding, SlotOffset,
                               VK_DESCRIPTOR_TYPE_SAMPLER, VK_NULL_HANDLE, 0,
                               VK_NULL_HANDLE, Samp, VK_NULL_HANDLE});
        }
        continue;
      }

      if (!Res.BufferPtr)
        continue;
      uint32_t ArrayCount =
          std::max<uint32_t>(1u, static_cast<uint32_t>(Res.BufferPtr->ArraySize));
      if (Res.BufferPtr->Data.size() > ArrayCount)
        ArrayCount = static_cast<uint32_t>(Res.BufferPtr->Data.size());

      if (isTextureKind(Res.Kind)) {
        bool IsUAV = isUAVKind(Res.Kind);
        for (uint32_t Slice = 0; Slice < ArrayCount; ++Slice) {
          CPUBuffer *CPU = Res.BufferPtr;
          uint32_t TW = std::max(1, CPU->OutputProps.Width);
          uint32_t TH = std::max(1, CPU->OutputProps.Height);
          uint32_t TD = std::max(1, CPU->OutputProps.Depth);
          uint32_t Layers = std::max(1, CPU->OutputProps.ArraySlices);
          uint32_t Mips = std::max(1, CPU->OutputProps.MipLevels);
          VkImageType ImgType = VK_IMAGE_TYPE_2D;
          VkImageViewType ViewType = VK_IMAGE_VIEW_TYPE_2D;
          VkImageCreateFlags ImgFlags = 0;
          switch (Res.Kind) {
          case ResourceKind::Texture1D:
          case ResourceKind::RWTexture1D:
            ImgType = VK_IMAGE_TYPE_1D;
            ViewType = VK_IMAGE_VIEW_TYPE_1D;
            TH = 1; TD = 1; Layers = 1;
            break;
          case ResourceKind::Texture1DArray:
          case ResourceKind::RWTexture1DArray:
            ImgType = VK_IMAGE_TYPE_1D;
            ViewType = VK_IMAGE_VIEW_TYPE_1D_ARRAY;
            TH = 1; TD = 1;
            break;
          case ResourceKind::Texture2D:
          case ResourceKind::RWTexture2D:
            ImgType = VK_IMAGE_TYPE_2D;
            ViewType = VK_IMAGE_VIEW_TYPE_2D;
            TD = 1; Layers = 1;
            break;
          case ResourceKind::Texture2DArray:
          case ResourceKind::RWTexture2DArray:
            ImgType = VK_IMAGE_TYPE_2D;
            ViewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            TD = 1;
            break;
          case ResourceKind::Texture3D:
          case ResourceKind::RWTexture3D:
            ImgType = VK_IMAGE_TYPE_3D;
            ViewType = VK_IMAGE_VIEW_TYPE_3D;
            Layers = 1;
            break;
          case ResourceKind::TextureCube:
            ImgType = VK_IMAGE_TYPE_2D;
            ViewType = VK_IMAGE_VIEW_TYPE_CUBE;
            ImgFlags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
            TD = 1; Layers = 6;
            break;
          case ResourceKind::TextureCubeArray:
            ImgType = VK_IMAGE_TYPE_2D;
            ViewType = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
            ImgFlags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
            TD = 1;
            if (Layers < 6) Layers = 6;
            break;
          default:
            break;
          }
          VkFormat Fmt = vkFormatForBuffer(*CPU);
          uint32_t TexelBytes = std::max<uint32_t>(
              1u, static_cast<uint32_t>(CPU->getSingleElementSize()) *
                      std::max<uint32_t>(1u, CPU->Channels));
          RasterImg RI;
          VkImageCreateInfo TICI{};
          TICI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
          TICI.flags = ImgFlags;
          TICI.imageType = ImgType;
          TICI.format = Fmt;
          TICI.extent = {TW, TH, TD};
          TICI.mipLevels = Mips;
          TICI.arrayLayers = Layers;
          TICI.samples = VK_SAMPLE_COUNT_1_BIT;
          TICI.tiling = VK_IMAGE_TILING_OPTIMAL;
          TICI.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                       (IsUAV ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
          TICI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
          TICI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
          if (VkResult R = vkCreateImage(VK.Device, &TICI, nullptr, &RI.Image)) {
            cleanup();
            return vkErr(R, "vkCreateImage (raster tex)");
          }
          VkMemoryRequirements TMR{};
          vkGetImageMemoryRequirements(VK.Device, RI.Image, &TMR);
          auto TMemType = VK.findMemoryType(TMR.memoryTypeBits,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
          if (!TMemType)
            TMemType = VK.findMemoryType(TMR.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
          if (!TMemType) {
            vkDestroyImage(VK.Device, RI.Image, nullptr);
            cleanup();
            return TMemType.takeError();
          }
          VkMemoryAllocateInfo TMAI{};
          TMAI.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
          TMAI.allocationSize = TMR.size;
          TMAI.memoryTypeIndex = *TMemType;
          if (VkResult R = vkAllocateMemory(VK.Device, &TMAI, nullptr, &RI.Memory)) {
            vkDestroyImage(VK.Device, RI.Image, nullptr);
            cleanup();
            return vkErr(R, "vkAllocateMemory (raster tex)");
          }
          vkBindImageMemory(VK.Device, RI.Image, RI.Memory, 0);
          VkImageViewCreateInfo TIVCI{};
          TIVCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
          TIVCI.image = RI.Image;
          TIVCI.viewType = ViewType;
          TIVCI.format = Fmt;
          TIVCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, Mips, 0, Layers};
          if (VkResult R = vkCreateImageView(VK.Device, &TIVCI, nullptr, &RI.View)) {
            vkFreeMemory(VK.Device, RI.Memory, nullptr);
            vkDestroyImage(VK.Device, RI.Image, nullptr);
            cleanup();
            return vkErr(R, "vkCreateImageView (raster tex)");
          }
          RasterImgs.push_back(RI);

          VkDeviceSize StageBytes = std::max<VkDeviceSize>(CPU->Size, 16);
          VkBuffer StageBuf = VK_NULL_HANDLE;
          VkDeviceMemory StageMem = VK_NULL_HANDLE;
          VkBufferCreateInfo SBCI{};
          SBCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
          SBCI.size = StageBytes;
          SBCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
          SBCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
          vkCreateBuffer(VK.Device, &SBCI, nullptr, &StageBuf);
          VkMemoryRequirements SMR{};
          vkGetBufferMemoryRequirements(VK.Device, StageBuf, &SMR);
          auto SMemType = VK.findMemoryType(
              SMR.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
          if (!SMemType) {
            vkDestroyBuffer(VK.Device, StageBuf, nullptr);
            cleanup();
            return SMemType.takeError();
          }
          VkMemoryAllocateInfo SMAI{};
          SMAI.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
          SMAI.allocationSize = SMR.size;
          SMAI.memoryTypeIndex = *SMemType;
          vkAllocateMemory(VK.Device, &SMAI, nullptr, &StageMem);
          vkBindBufferMemory(VK.Device, StageBuf, StageMem, 0);
          void *SMapped = nullptr;
          vkMapMemory(VK.Device, StageMem, 0, StageBytes, 0, &SMapped);
          memset(SMapped, 0, StageBytes);
          if (Slice < CPU->Data.size() && CPU->Data[Slice])
            memcpy(SMapped, CPU->Data[Slice].get(), CPU->Size);
          vkUnmapMemory(VK.Device, StageMem);

          Error UpErr = runRasterOneShot([&](VkCommandBuffer UCB) {
            VkImageMemoryBarrier B1{};
            B1.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            B1.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            B1.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            B1.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            B1.image = RI.Image;
            B1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, Mips, 0, Layers};
            vkCmdPipelineBarrier(UCB, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                                 0, nullptr, 1, &B1);
            VkDeviceSize Off = 0;
            for (uint32_t L = 0; L < Layers; ++L) {
              for (uint32_t M = 0; M < Mips; ++M) {
                uint32_t MW = std::max(1u, TW >> M);
                uint32_t MH = std::max(1u, TH >> M);
                uint32_t MD = std::max(1u, TD >> M);
                VkDeviceSize SubBytes =
                    static_cast<VkDeviceSize>(MW) * MH * MD * TexelBytes;
                if (Off + SubBytes <= StageBytes) {
                  VkBufferImageCopy Reg{};
                  Reg.bufferOffset = Off;
                  Reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, M, L, 1};
                  Reg.imageExtent = {MW, MH, MD};
                  vkCmdCopyBufferToImage(UCB, StageBuf, RI.Image,
                                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                         &Reg);
                }
                Off += SubBytes;
              }
            }
            VkImageMemoryBarrier B2{};
            B2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            B2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            B2.newLayout = IsUAV ? VK_IMAGE_LAYOUT_GENERAL
                                 : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            B2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            B2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                               (IsUAV ? VK_ACCESS_SHADER_WRITE_BIT : 0);
            B2.image = RI.Image;
            B2.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, Mips, 0, Layers};
            vkCmdPipelineBarrier(UCB, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &B2);
          });
          vkDestroyBuffer(VK.Device, StageBuf, nullptr);
          vkFreeMemory(VK.Device, StageMem, nullptr);
          if (UpErr) {
            cleanup();
            return UpErr;
          }
          for (const auto *Match : Matches) {
            uint32_t SlotOffset = Res.DXBinding->Register - Match->LowerBound;
            BoundList.push_back({Match->SPIRVSet, Match->SPIRVBinding,
                                 SlotOffset + Slice,
                                 IsUAV ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                       : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                 VK_NULL_HANDLE, 0, VK_NULL_HANDLE,
                                 VK_NULL_HANDLE, RI.View});
          }
        }
        continue;
      }

      bool IsUniform = (Res.Kind == ResourceKind::ConstantBuffer);
      bool IsTyped = isTypedBufferKind(Res.Kind);
      bool IsUAV = isUAVKind(Res.Kind);
      VkDescriptorType DT =
          IsUniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
          : IsTyped ? (IsUAV ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER
                             : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER)
                    : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      for (uint32_t Slice = 0; Slice < ArrayCount; ++Slice) {
        CPUBuffer *CPU = Res.BufferPtr;
        RasterBuf RB;
        RB.CPU = CPU;
        RB.Slice = Slice;
        RB.IsUAV = IsUAV;
        RB.SizeBytes = std::max<VkDeviceSize>(CPU->Size, 16);
        if (IsUniform && (RB.SizeBytes % 16) != 0)
          RB.SizeBytes = ((RB.SizeBytes + 15) / 16) * 16;
        VkBufferCreateInfo BCI{};
        BCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        BCI.size = RB.SizeBytes;
        BCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (IsUniform)
          BCI.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        else if (IsTyped)
          BCI.usage |= VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
                       VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
        else
          BCI.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        BCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (VkResult R = vkCreateBuffer(VK.Device, &BCI, nullptr, &RB.Buffer)) {
          cleanup();
          return vkErr(R, "vkCreateBuffer (raster)");
        }
        VkMemoryRequirements BMR{};
        vkGetBufferMemoryRequirements(VK.Device, RB.Buffer, &BMR);
        auto BMemType = VK.findMemoryType(
            BMR.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!BMemType) {
          vkDestroyBuffer(VK.Device, RB.Buffer, nullptr);
          cleanup();
          return BMemType.takeError();
        }
        VkMemoryAllocateInfo BMAI{};
        BMAI.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        BMAI.allocationSize = BMR.size;
        BMAI.memoryTypeIndex = *BMemType;
        if (VkResult R = vkAllocateMemory(VK.Device, &BMAI, nullptr, &RB.Memory)) {
          vkDestroyBuffer(VK.Device, RB.Buffer, nullptr);
          cleanup();
          return vkErr(R, "vkAllocateMemory (raster)");
        }
        vkBindBufferMemory(VK.Device, RB.Buffer, RB.Memory, 0);
        void *BMapped = nullptr;
        vkMapMemory(VK.Device, RB.Memory, 0, RB.SizeBytes, 0, &BMapped);
        memset(BMapped, 0, RB.SizeBytes);
        if (Slice < CPU->Data.size() && CPU->Data[Slice])
          memcpy(BMapped, CPU->Data[Slice].get(), CPU->Size);
        vkUnmapMemory(VK.Device, RB.Memory);
        if (IsTyped) {
          VkBufferViewCreateInfo BVCI{};
          BVCI.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
          BVCI.buffer = RB.Buffer;
          BVCI.format = vkFormatForBuffer(*CPU);
          BVCI.offset = 0;
          BVCI.range = RB.SizeBytes;
          if (VkResult R = vkCreateBufferView(VK.Device, &BVCI, nullptr, &RB.View)) {
            vkFreeMemory(VK.Device, RB.Memory, nullptr);
            vkDestroyBuffer(VK.Device, RB.Buffer, nullptr);
            cleanup();
            return vkErr(R, "vkCreateBufferView (raster)");
          }
        }
        RasterBufs.push_back(RB);
        for (const auto *Match : Matches) {
          uint32_t SlotOffset = Res.DXBinding->Register - Match->LowerBound;
          BoundList.push_back({Match->SPIRVSet, Match->SPIRVBinding,
                               SlotOffset + Slice, DT, RB.Buffer, RB.SizeBytes,
                               RB.View, VK_NULL_HANDLE, VK_NULL_HANDLE});
        }
      }
    }
  }

  uint32_t MaxSet = 0;
  for (const auto &RB : BoundList)
    MaxSet = std::max(MaxSet, RB.Set + 1u);
  SetLayouts.assign(MaxSet, VK_NULL_HANDLE);
  std::map<VkDescriptorType, uint32_t> PoolCounts;
  for (uint32_t SetIdx = 0; SetIdx < MaxSet; ++SetIdx) {
    struct BindInfo {
      VkDescriptorType Type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      uint32_t Count = 0;
    };
    std::map<uint32_t, BindInfo> BMap;
    for (const auto &RB : BoundList) {
      if (RB.Set != SetIdx)
        continue;
      auto &Info = BMap[RB.Binding];
      Info.Type = RB.DescType;
      Info.Count = std::max(Info.Count, RB.ArrayElem + 1u);
    }
    std::vector<VkDescriptorSetLayoutBinding> LayoutBinds;
    for (const auto &[BindIdx, Info] : BMap) {
      VkDescriptorSetLayoutBinding LB{};
      LB.binding = BindIdx;
      LB.descriptorType = Info.Type;
      LB.descriptorCount = Info.Count;
      LB.stageFlags =
          VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
      LayoutBinds.push_back(LB);
      PoolCounts[Info.Type] += Info.Count;
    }
    VkDescriptorSetLayoutCreateInfo DSLCI{};
    DSLCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    DSLCI.bindingCount = static_cast<uint32_t>(LayoutBinds.size());
    DSLCI.pBindings = LayoutBinds.data();
    if (VkResult R = vkCreateDescriptorSetLayout(VK.Device, &DSLCI, nullptr,
                                                 &SetLayouts[SetIdx])) {
      cleanup();
      return vkErr(R, "vkCreateDescriptorSetLayout (raster)");
    }
  }

  std::vector<VkDescriptorSet> DSets;
  if (MaxSet > 0) {
    std::vector<VkDescriptorPoolSize> PoolSizes;
    for (auto &[DT, Cnt] : PoolCounts)
      PoolSizes.push_back({DT, Cnt});
    if (PoolSizes.empty())
      PoolSizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1});
    VkDescriptorPoolCreateInfo DPCI{};
    DPCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    DPCI.maxSets = MaxSet;
    DPCI.poolSizeCount = static_cast<uint32_t>(PoolSizes.size());
    DPCI.pPoolSizes = PoolSizes.data();
    if (VkResult R = vkCreateDescriptorPool(VK.Device, &DPCI, nullptr, &DPool)) {
      cleanup();
      return vkErr(R, "vkCreateDescriptorPool (raster)");
    }
    DSets.resize(MaxSet, VK_NULL_HANDLE);
    VkDescriptorSetAllocateInfo DSAI{};
    DSAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    DSAI.descriptorPool = DPool;
    DSAI.descriptorSetCount = MaxSet;
    DSAI.pSetLayouts = SetLayouts.data();
    if (VkResult R = vkAllocateDescriptorSets(VK.Device, &DSAI, DSets.data())) {
      cleanup();
      return vkErr(R, "vkAllocateDescriptorSets (raster)");
    }
    std::vector<VkDescriptorBufferInfo> BufInfos(BoundList.size());
    std::vector<VkDescriptorImageInfo> ImgInfos(BoundList.size());
    std::vector<VkWriteDescriptorSet> Writes;
    for (size_t I = 0; I < BoundList.size(); ++I) {
      const auto &RB = BoundList[I];
      VkWriteDescriptorSet W{};
      W.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      W.dstSet = DSets[RB.Set];
      W.dstBinding = RB.Binding;
      W.dstArrayElement = RB.ArrayElem;
      W.descriptorCount = 1;
      W.descriptorType = RB.DescType;
      if (RB.DescType == VK_DESCRIPTOR_TYPE_SAMPLER) {
        ImgInfos[I].sampler = RB.Sampler;
        W.pImageInfo = &ImgInfos[I];
      } else if (RB.DescType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
                 RB.DescType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
        ImgInfos[I].imageView = RB.ImageView;
        ImgInfos[I].imageLayout =
            (RB.DescType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                ? VK_IMAGE_LAYOUT_GENERAL
                : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        W.pImageInfo = &ImgInfos[I];
      } else if (RB.DescType == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER ||
                 RB.DescType == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
        W.pTexelBufferView = &BoundList[I].BufferView;
      } else {
        BufInfos[I] = {RB.Buffer, 0, RB.BufferSize};
        W.pBufferInfo = &BufInfos[I];
      }
      Writes.push_back(W);
    }
    if (!Writes.empty())
      vkUpdateDescriptorSets(VK.Device, static_cast<uint32_t>(Writes.size()),
                             Writes.data(), 0, nullptr);
  }

  VkPipelineLayoutCreateInfo PLCI{};
  PLCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  PLCI.setLayoutCount = static_cast<uint32_t>(SetLayouts.size());
  PLCI.pSetLayouts = SetLayouts.data();
  if (VkResult R = vkCreatePipelineLayout(VK.Device, &PLCI, nullptr, &PipeLayout)) {
    cleanup();
    return vkErr(R, "vkCreatePipelineLayout");
  }

  VkPipelineShaderStageCreateInfo StagesCI[2]{};
  StagesCI[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  StagesCI[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  StagesCI[0].module = VSMod;
  StagesCI[0].pName =
      VSCompiled->Reflect.EntryName.empty() ? "main" : VSCompiled->Reflect.EntryName.c_str();
  StagesCI[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  StagesCI[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  StagesCI[1].module = PSMod;
  StagesCI[1].pName =
      PSCompiled->Reflect.EntryName.empty() ? "main" : PSCompiled->Reflect.EntryName.c_str();

  VkPipelineVertexInputStateCreateInfo VISCI{};
  VISCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  VISCI.vertexBindingDescriptionCount = static_cast<uint32_t>(VBindings.size());
  VISCI.pVertexBindingDescriptions = VBindings.data();
  VISCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(VAttrs.size());
  VISCI.pVertexAttributeDescriptions = VAttrs.data();

  VkPipelineInputAssemblyStateCreateInfo IASCI{};
  IASCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  IASCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  // Flip Vulkan viewport Y to match DirectX top-left clip-space convention.
  VkViewport VP{0.0f, static_cast<float>(Height), static_cast<float>(Width),
                -static_cast<float>(Height), 0.0f, 1.0f};
  VkRect2D Scissor{{0, 0}, {Width, Height}};
  VkPipelineViewportStateCreateInfo VPSCI{};
  VPSCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  VPSCI.viewportCount = 1;
  VPSCI.pViewports = &VP;
  VPSCI.scissorCount = 1;
  VPSCI.pScissors = &Scissor;

  VkPipelineRasterizationStateCreateInfo RSCI{};
  RSCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  RSCI.polygonMode = VK_POLYGON_MODE_FILL;
  RSCI.cullMode = VK_CULL_MODE_NONE;
  // Viewport Y is flipped above to match DX clip space, which reverses
  // screen-space winding; use CCW so SV_IsFrontFace matches DirectX.
  RSCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  RSCI.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo MSSCI{};
  MSSCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  MSSCI.rasterizationSamples = SampleBits;

  VkPipelineDepthStencilStateCreateInfo DSSCI{};
  DSSCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  DSSCI.depthTestEnable = VK_TRUE;
  DSSCI.depthWriteEnable = VK_TRUE;
  DSSCI.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
  if (HasStencil) {
    DSSCI.stencilTestEnable = VK_TRUE;
    VkStencilOpState StencilOp{};
    StencilOp.failOp = VK_STENCIL_OP_KEEP;
    StencilOp.passOp = VK_STENCIL_OP_KEEP;
    StencilOp.depthFailOp = VK_STENCIL_OP_KEEP;
    StencilOp.compareOp = vkCompare(P.Stencil->ComparisonOp);
    StencilOp.compareMask = 0xFFu;
    StencilOp.writeMask = 0xFFu;
    StencilOp.reference = 0u;
    DSSCI.front = StencilOp;
    DSSCI.back = StencilOp;
  }

  VkPipelineColorBlendAttachmentState CBAtt{};
  CBAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo CBSCI{};
  CBSCI.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  CBSCI.attachmentCount = 1;
  CBSCI.pAttachments = &CBAtt;

  VkGraphicsPipelineCreateInfo GPCI{};
  GPCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  GPCI.stageCount = 2;
  GPCI.pStages = StagesCI;
  GPCI.pVertexInputState = &VISCI;
  GPCI.pInputAssemblyState = &IASCI;
  GPCI.pViewportState = &VPSCI;
  GPCI.pRasterizationState = &RSCI;
  GPCI.pMultisampleState = &MSSCI;
  GPCI.pDepthStencilState = &DSSCI;
  GPCI.pColorBlendState = &CBSCI;
  GPCI.layout = PipeLayout;
  GPCI.renderPass = RenderPass;
  GPCI.subpass = 0;

  if (VkResult R = vkCreateGraphicsPipelines(VK.Device, VK_NULL_HANDLE, 1,
                                             &GPCI, nullptr, &Pipe)) {
    cleanup();
    return vkErr(R, "vkCreateGraphicsPipelines");
  }

  VkCommandBufferAllocateInfo CBAI{};
  CBAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  CBAI.commandPool = CmdPool;
  CBAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  CBAI.commandBufferCount = 1;
  VkCommandBuffer CB = VK_NULL_HANDLE;
  if (VkResult R = vkAllocateCommandBuffers(VK.Device, &CBAI, &CB)) {
    cleanup();
    return vkErr(R, "vkAllocateCommandBuffers");
  }

  VkCommandBufferBeginInfo CBBI{};
  CBBI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  CBBI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(CB, &CBBI);

  VkClearValue ClearVals[2]{};
  ClearVals[1].depthStencil = {
      1.0f, HasStencil ? static_cast<uint32_t>(P.Stencil->ClearValue) : 0u};
  VkRenderPassBeginInfo RPBI{};
  RPBI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  RPBI.renderPass = RenderPass;
  RPBI.framebuffer = FB;
  RPBI.renderArea = {{0, 0}, {Width, Height}};
  RPBI.clearValueCount = 2;
  RPBI.pClearValues = ClearVals;
  vkCmdBeginRenderPass(CB, &RPBI, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdBindPipeline(CB, VK_PIPELINE_BIND_POINT_GRAPHICS, Pipe);
  if (!DSets.empty()) {
    vkCmdBindDescriptorSets(CB, VK_PIPELINE_BIND_POINT_GRAPHICS, PipeLayout, 0,
                            static_cast<uint32_t>(DSets.size()), DSets.data(),
                            0, nullptr);
  }
  if (VertBuf) {
    VkDeviceSize Off = 0;
    vkCmdBindVertexBuffers(CB, 0, 1, &VertBuf, &Off);
  }
  vkCmdDraw(CB, VertexCount, 1, 0, 0);
  vkCmdEndRenderPass(CB);

  VkBufferImageCopy CopyRegion{};
  CopyRegion.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  CopyRegion.imageExtent = {Width, Height, 1};
  vkCmdCopyImageToBuffer(CB, ColorImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         ReadbackBuf, 1, &CopyRegion);
  vkEndCommandBuffer(CB);

  VkSubmitInfo SI{};
  SI.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  SI.commandBufferCount = 1;
  SI.pCommandBuffers = &CB;
  if (VkResult R = vkQueueSubmit(VK.Queue, 1, &SI, VK_NULL_HANDLE)) {
    cleanup();
    return vkErr(R, "vkQueueSubmit (raster)");
  }
  if (VkResult R = vkQueueWaitIdle(VK.Queue)) {
    cleanup();
    return vkErr(R, "vkQueueWaitIdle (raster)");
  }

  void *Mapped = nullptr;
  if (VkResult R =
          vkMapMemory(VK.Device, ReadbackMem, 0, ReadbackBytes, 0, &Mapped)) {
    cleanup();
    return vkErr(R, "vkMapMemory (raster readback)");
  }
  memcpy(Output->Data[0].get(), Mapped, ReadbackBytes);
  vkUnmapMemory(VK.Device, ReadbackMem);

  // Copy back UAV buffers written by VS/PS shaders.
  for (const auto &RB : RasterBufs) {
    if (!RB.IsUAV || !RB.CPU || RB.Slice >= RB.CPU->Data.size())
      continue;
    void *BMapped = nullptr;
    if (VkResult R =
            vkMapMemory(VK.Device, RB.Memory, 0, RB.SizeBytes, 0, &BMapped)) {
      cleanup();
      return vkErr(R, "vkMapMemory (raster UAV readback)");
    }
    memcpy(RB.CPU->Data[RB.Slice].get(), BMapped, RB.CPU->Size);
    vkUnmapMemory(VK.Device, RB.Memory);
  }

  cleanup();
  return Error::success();
}

class CrucibleVulkanDevice final : public offloadtest::Device {
  std::unique_ptr<VkContext> VK;
  Capabilities Caps;

  static Error unsupported() {
    return createStringError(std::errc::not_supported,
                             "Crucible supports only pipeline execution");
  }

public:
  explicit CrucibleVulkanDevice(std::unique_ptr<VkContext> VK)
      : VK(std::move(VK)) {
    const VulkanBackendCaps &BC = this->VK->BackendCaps;
    Description = formatVulkanDeviceDescription(BC.DeviceName, BC.DriverName);
    DriverName = "Crucible";
    if (!BC.DriverInfo.empty()) {
      DriverVersion = BC.DriverInfo;
    } else {
      DriverVersion =
          formatv("{0}.{1}.{2}", VK_API_VERSION_MAJOR(BC.DriverVersion),
                  VK_API_VERSION_MINOR(BC.DriverVersion),
                  VK_API_VERSION_PATCH(BC.DriverVersion));
    }
    if (StringRef(BC.DeviceName).starts_with("Apple "))
      GPUGeneration =
          StringRef(BC.DeviceName).split(' ').second.split(' ').first.str();
    else
      GPUGeneration = "Unknown";

    VulkanRuntimeCaps RuntimeCaps;
    RuntimeCaps.SupportsMeshShader = false;
    RuntimeCaps.SupportsRaytracing = false;

    CrucibleEmulationCaps Emul;
    Emul.EmulateF64 = false;
    Emul.EmulateInt64Atomics = false;
    Emul.HighestShaderModel = "6.6";

    Caps = buildDirectXCapsFromVulkan(BC, RuntimeCaps, Emul);
  }

  const Capabilities &getCapabilities() override { return Caps; }
  StringRef getAPIName() const override { return "DirectX"; }
  GPUAPI getAPI() const override { return GPUAPI::DirectX; }
  void printExtra(raw_ostream &OS) override {
    OS << "  Backend: Vulkan\n";
    if (!VK->BackendCaps.DriverName.empty())
      OS << "  VulkanDriver: " << VK->BackendCaps.DriverName << '\n';
    OS << "  VendorID: 0x" << Twine::utohexstr(VK->BackendCaps.VendorID) << '\n'
       << "  DeviceID: 0x" << Twine::utohexstr(VK->BackendCaps.DeviceID)
       << '\n';
  }
  Error executeProgram(Pipeline &P) override {
    if (auto Err = validateRootParameters(P))
      return Err;
    return P.isTraditionalRaster() ? executeVulkanRaster(P, *VK)
                                   : executeVulkanCompute(P, *VK);
  }

  Queue &getGraphicsQueue() override {
    report_fatal_error("Crucible does not implement the low-level graphics queue");
  }
  Expected<std::unique_ptr<PipelineState>>
  createPipelineCs(StringRef, const BindingsDesc &, ShaderContainer) override {
    return unsupported();
  }
  Expected<std::unique_ptr<PipelineState>> createTraditionalRasterPipeline(
      StringRef, const BindingsDesc &,
      const TraditionalRasterPipelineCreateDesc &) override {
    return unsupported();
  }
  Expected<std::unique_ptr<PipelineState>> createMeshShaderRasterPipeline(
      StringRef, const BindingsDesc &,
      const MeshShaderRasterPipelineCreateDesc &) override {
    return unsupported();
  }
  Expected<std::unique_ptr<PipelineState>> createPipelineRT(
      StringRef, const BindingsDesc &,
      const RayTracingPipelineCreateDesc &) override {
    return unsupported();
  }
  Expected<std::unique_ptr<ShaderBindingTable>> createShaderBindingTable(
      const PipelineState &, const ShaderBindingTableDesc &) override {
    return unsupported();
  }
  Expected<std::unique_ptr<Fence>> createFence(StringRef) override {
    return unsupported();
  }
  Expected<std::unique_ptr<Buffer>>
  createBuffer(std::string, const BufferCreateDesc &, size_t) override {
    return unsupported();
  }
  Expected<std::unique_ptr<Texture>>
  createTexture(std::string, const TextureCreateDesc &) override {
    return unsupported();
  }
  Expected<std::unique_ptr<Sampler>>
  createSampler(std::string, const SamplerCreateDesc &) override {
    return unsupported();
  }
  Expected<std::unique_ptr<MemoryHeap>>
  createMemoryHeap(std::string, size_t) override {
    return unsupported();
  }
  uint32_t getTextureUploadRowStrideInBytes(
      const TextureCreateDesc &) const override {
    report_fatal_error("Crucible does not implement low-level texture upload");
  }
  TextureUploadLayout getTextureUploadLayout(
      const TextureCreateDesc &) const override {
    report_fatal_error("Crucible does not implement low-level texture upload");
  }
  Expected<std::unique_ptr<RenderPass>>
  createRenderPass(const RenderPassDesc &) override { return unsupported(); }
  Expected<std::unique_ptr<CommandBuffer>> createCommandBuffer() override {
    return unsupported();
  }
  Expected<AccelerationStructureSizes>
  getBLASBuildSizes(ArrayRef<TriangleGeometryDesc>) override {
    return unsupported();
  }
  Expected<AccelerationStructureSizes>
  getBLASBuildSizes(ArrayRef<AABBGeometryDesc>) override {
    return unsupported();
  }
  Expected<AccelerationStructureSizes> getTLASBuildSizes(uint32_t) override {
    return unsupported();
  }
  Expected<std::unique_ptr<AccelerationStructure>>
  createBLAS(const AccelerationStructureSizes &) override {
    return unsupported();
  }
  Expected<std::unique_ptr<AccelerationStructure>>
  createTLAS(const AccelerationStructureSizes &) override {
    return unsupported();
  }
};

} // namespace

namespace crucible_offload {

Expected<SmallVector<std::unique_ptr<offloadtest::Device>>>
createVulkanDevices(const DeviceConfig &Config) {
  (void)Config;
  auto VK = std::make_unique<VkContext>();

  VkApplicationInfo AppInfo{};
  AppInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  AppInfo.pApplicationName = "crucible-offload";
  AppInfo.apiVersion = VK_API_VERSION_1_2;

  std::vector<const char *> InstanceExts;
  VkInstanceCreateFlags InstanceFlags = 0;
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
  uint32_t ExtCount = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &ExtCount, nullptr);
  std::vector<VkExtensionProperties> AvailExts(ExtCount);
  vkEnumerateInstanceExtensionProperties(nullptr, &ExtCount, AvailExts.data());
  for (const auto &E : AvailExts) {
    if (strcmp(E.extensionName,
               VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0) {
      InstanceExts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
      InstanceFlags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
      break;
    }
  }
#endif

  VkInstanceCreateInfo InstCI{};
  InstCI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  InstCI.flags = InstanceFlags;
  InstCI.pApplicationInfo = &AppInfo;
  InstCI.enabledExtensionCount = static_cast<uint32_t>(InstanceExts.size());
  InstCI.ppEnabledExtensionNames = InstanceExts.data();

  if (VkResult R = vkCreateInstance(&InstCI, nullptr, &VK->Instance))
    return vkErr(R, "vkCreateInstance");

  uint32_t DevCount = 0;
  if (VkResult R = vkEnumeratePhysicalDevices(VK->Instance, &DevCount, nullptr))
    return vkErr(R, "vkEnumeratePhysicalDevices");
  if (DevCount == 0)
    return createStringError(std::errc::no_such_device,
                             "no Vulkan physical device available");
  std::vector<VkPhysicalDevice> Devs(DevCount);
  if (VkResult R =
          vkEnumeratePhysicalDevices(VK->Instance, &DevCount, Devs.data()))
    return vkErr(R, "vkEnumeratePhysicalDevices");

  auto queryDriverName = [](VkPhysicalDevice PD,
                            VkPhysicalDeviceProperties &BaseProps) -> std::string {
    vkGetPhysicalDeviceProperties(PD, &BaseProps);
    if (BaseProps.apiVersion >= VK_API_VERSION_1_2) {
      VkPhysicalDeviceDriverProperties DriverProps{};
      DriverProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
      VkPhysicalDeviceProperties2 Props2{};
      Props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
      Props2.pNext = &DriverProps;
      vkGetPhysicalDeviceProperties2(PD, &Props2);
      return std::string(
          DriverProps.driverName,
          strnlen(DriverProps.driverName, VK_MAX_DRIVER_NAME_SIZE));
    }
    return {};
  };

  VK->PhysDev = Devs[0];
  if (const char *WantedDriver = std::getenv("CRUCIBLE_OFFLOAD_VK_DRIVER")) {
    StringRef Filter(WantedDriver);
    bool Matched = false;
    for (VkPhysicalDevice PD : Devs) {
      VkPhysicalDeviceProperties BaseProps{};
      std::string DrvName = queryDriverName(PD, BaseProps);
      if (StringRef(DrvName).contains_insensitive(Filter) ||
          StringRef(BaseProps.deviceName).contains_insensitive(Filter)) {
        VK->PhysDev = PD;
        Matched = true;
        break;
      }
    }
    if (!Matched)
      return createStringError(
          std::errc::no_such_device,
          "no Vulkan physical device matched CRUCIBLE_OFFLOAD_VK_DRIVER='%s'",
          WantedDriver);
  }

  uint32_t DevExtCount = 0;
  vkEnumerateDeviceExtensionProperties(VK->PhysDev, nullptr, &DevExtCount,
                                       nullptr);
  std::vector<VkExtensionProperties> DevExts(DevExtCount);
  if (DevExtCount > 0)
    vkEnumerateDeviceExtensionProperties(VK->PhysDev, nullptr, &DevExtCount,
                                         DevExts.data());
  StringSet<> AvailDevExts;
  for (const auto &E : DevExts)
    AvailDevExts.insert(E.extensionName);

  VkPhysicalDeviceProperties BaseProps{};
  vkGetPhysicalDeviceProperties(VK->PhysDev, &BaseProps);
  const bool HasVk12 = BaseProps.apiVersion >= VK_API_VERSION_1_2;

  VkPhysicalDeviceSubgroupProperties SubgroupProps{};
  SubgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  VkPhysicalDeviceDriverProperties DriverProps{};
  DriverProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
  VkPhysicalDeviceProperties2 Props2{};
  Props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  Props2.pNext = &SubgroupProps;
  if (HasVk12 || AvailDevExts.contains(VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME))
    SubgroupProps.pNext = &DriverProps;
  vkGetPhysicalDeviceProperties2(VK->PhysDev, &Props2);

  VkPhysicalDeviceVulkan11Features SupportedVk11{};
  SupportedVk11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
  VkPhysicalDeviceVulkan12Features SupportedVk12{};
  SupportedVk12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  VkPhysicalDeviceFeatures2 SupportedFeats2{};
  SupportedFeats2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  if (HasVk12) {
    SupportedFeats2.pNext = &SupportedVk11;
    SupportedVk11.pNext = &SupportedVk12;
  }
  vkGetPhysicalDeviceFeatures2(VK->PhysDev, &SupportedFeats2);

  VulkanBackendCaps &BC = VK->BackendCaps;
  BC.DeviceName = std::string(
      BaseProps.deviceName,
      strnlen(BaseProps.deviceName, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE));
  if (HasVk12 || AvailDevExts.contains(VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME)) {
    BC.DriverName = std::string(
        DriverProps.driverName,
        strnlen(DriverProps.driverName, VK_MAX_DRIVER_NAME_SIZE));
    BC.DriverInfo = std::string(
        DriverProps.driverInfo,
        strnlen(DriverProps.driverInfo, VK_MAX_DRIVER_INFO_SIZE));
  }
  BC.VendorID = BaseProps.vendorID;
  BC.DeviceID = BaseProps.deviceID;
  BC.ApiVersion = BaseProps.apiVersion;
  BC.DriverVersion = BaseProps.driverVersion;
  BC.SubgroupSize = SubgroupProps.subgroupSize;
  BC.SubgroupCompute =
      (SubgroupProps.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
  BC.ShaderFloat64 = SupportedFeats2.features.shaderFloat64 != 0;
  BC.ShaderInt64 = SupportedFeats2.features.shaderInt64 != 0;
  BC.ShaderInt16 = SupportedFeats2.features.shaderInt16 != 0;
  BC.ShaderResourceMinLod = SupportedFeats2.features.shaderResourceMinLod != 0;
  BC.MultiViewport = SupportedFeats2.features.multiViewport != 0;
  if (HasVk12) {
    BC.ShaderFloat16 = SupportedVk12.shaderFloat16 != 0;
    BC.ShaderOutputViewportIndex = SupportedVk12.shaderOutputViewportIndex != 0;
    BC.DescriptorIndexing = SupportedVk12.descriptorIndexing != 0 &&
                            SupportedVk12.runtimeDescriptorArray != 0;
    BC.ShaderBufferInt64Atomics = SupportedVk12.shaderBufferInt64Atomics != 0;
    BC.ShaderSharedInt64Atomics = SupportedVk12.shaderSharedInt64Atomics != 0;
  }
  if (!BC.ShaderOutputViewportIndex &&
      AvailDevExts.contains("VK_EXT_shader_viewport_index_layer"))
    BC.ShaderOutputViewportIndex = true;
  BC.ShaderStencilExport = AvailDevExts.contains("VK_EXT_shader_stencil_export");
  BC.FragmentShaderBarycentric =
      AvailDevExts.contains("VK_KHR_fragment_shader_barycentric") ||
      AvailDevExts.contains("VK_NV_fragment_shader_barycentric");
  BC.MeshShader = AvailDevExts.contains("VK_EXT_mesh_shader");
  BC.RayTracing = AvailDevExts.contains("VK_KHR_ray_tracing_pipeline");

  uint32_t QFCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(VK->PhysDev, &QFCount, nullptr);
  std::vector<VkQueueFamilyProperties> QFs(QFCount);
  vkGetPhysicalDeviceQueueFamilyProperties(VK->PhysDev, &QFCount, QFs.data());
  for (uint32_t I = 0; I < QFCount; ++I) {
    if ((QFs[I].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
        (QFs[I].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
      VK->ComputeQF = I;
      VK->GraphicsQF = I;
      break;
    }
  }
  if (VK->ComputeQF == UINT32_MAX) {
    for (uint32_t I = 0; I < QFCount; ++I) {
      if (QFs[I].queueFlags & VK_QUEUE_COMPUTE_BIT)
        VK->ComputeQF = I;
      if (QFs[I].queueFlags & VK_QUEUE_GRAPHICS_BIT)
        VK->GraphicsQF = I;
    }
  }
  if (VK->ComputeQF == UINT32_MAX)
    return createStringError(std::errc::no_such_device,
                             "no Vulkan compute queue family found");

  std::vector<const char *> EnabledDevExts;
  auto enableIfAvail = [&](const char *ExtName) {
    if (AvailDevExts.contains(ExtName))
      EnabledDevExts.push_back(ExtName);
  };
  enableIfAvail("VK_KHR_portability_subset");
  enableIfAvail("VK_EXT_shader_stencil_export");
  enableIfAvail("VK_EXT_shader_viewport_index_layer");
  if (AvailDevExts.contains("VK_KHR_fragment_shader_barycentric"))
    EnabledDevExts.push_back("VK_KHR_fragment_shader_barycentric");

  VkPhysicalDeviceFeatures EnabledFeats{};
  EnabledFeats.shaderInt64 = SupportedFeats2.features.shaderInt64;
  EnabledFeats.shaderFloat64 = SupportedFeats2.features.shaderFloat64;
  EnabledFeats.shaderInt16 = SupportedFeats2.features.shaderInt16;
  EnabledFeats.fragmentStoresAndAtomics =
      SupportedFeats2.features.fragmentStoresAndAtomics;
  EnabledFeats.vertexPipelineStoresAndAtomics =
      SupportedFeats2.features.vertexPipelineStoresAndAtomics;
  EnabledFeats.imageCubeArray = SupportedFeats2.features.imageCubeArray;
  EnabledFeats.shaderClipDistance = SupportedFeats2.features.shaderClipDistance;
  EnabledFeats.shaderCullDistance = SupportedFeats2.features.shaderCullDistance;
  EnabledFeats.shaderResourceMinLod =
      SupportedFeats2.features.shaderResourceMinLod;
  EnabledFeats.multiViewport = SupportedFeats2.features.multiViewport;

  VkPhysicalDeviceVulkan11Features EnabledVk11{};
  EnabledVk11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
  VkPhysicalDeviceVulkan12Features EnabledVk12{};
  EnabledVk12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  if (HasVk12) {
    EnabledVk11.storageBuffer16BitAccess =
        SupportedVk11.storageBuffer16BitAccess;
    EnabledVk11.uniformAndStorageBuffer16BitAccess =
        SupportedVk11.uniformAndStorageBuffer16BitAccess;
    EnabledVk12.shaderFloat16 = SupportedVk12.shaderFloat16;
    EnabledVk12.shaderInt8 = SupportedVk12.shaderInt8;
    EnabledVk12.storageBuffer8BitAccess = SupportedVk12.storageBuffer8BitAccess;
    EnabledVk12.uniformAndStorageBuffer8BitAccess =
        SupportedVk12.uniformAndStorageBuffer8BitAccess;
    EnabledVk12.shaderOutputViewportIndex =
        SupportedVk12.shaderOutputViewportIndex;
    EnabledVk12.descriptorIndexing = SupportedVk12.descriptorIndexing;
    EnabledVk12.runtimeDescriptorArray = SupportedVk12.runtimeDescriptorArray;
    EnabledVk12.descriptorBindingPartiallyBound =
        SupportedVk12.descriptorBindingPartiallyBound;
    EnabledVk12.shaderStorageBufferArrayNonUniformIndexing =
        SupportedVk12.shaderStorageBufferArrayNonUniformIndexing;
    EnabledVk12.shaderUniformBufferArrayNonUniformIndexing =
        SupportedVk12.shaderUniformBufferArrayNonUniformIndexing;
    EnabledVk12.shaderSampledImageArrayNonUniformIndexing =
        SupportedVk12.shaderSampledImageArrayNonUniformIndexing;
    EnabledVk12.shaderStorageImageArrayNonUniformIndexing =
        SupportedVk12.shaderStorageImageArrayNonUniformIndexing;
    EnabledVk12.shaderStorageTexelBufferArrayNonUniformIndexing =
        SupportedVk12.shaderStorageTexelBufferArrayNonUniformIndexing;
    EnabledVk12.shaderUniformTexelBufferArrayNonUniformIndexing =
        SupportedVk12.shaderUniformTexelBufferArrayNonUniformIndexing;
    EnabledVk12.shaderBufferInt64Atomics =
        SupportedVk12.shaderBufferInt64Atomics;
    EnabledVk12.shaderSharedInt64Atomics =
        SupportedVk12.shaderSharedInt64Atomics;
    EnabledVk11.pNext = &EnabledVk12;
  }

  float Prio = 1.0f;
  VkDeviceQueueCreateInfo QCI{};
  QCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  QCI.queueFamilyIndex = VK->ComputeQF;
  QCI.queueCount = 1;
  QCI.pQueuePriorities = &Prio;

  VkDeviceCreateInfo DCI{};
  DCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  DCI.pNext = HasVk12 ? &EnabledVk11 : nullptr;
  DCI.queueCreateInfoCount = 1;
  DCI.pQueueCreateInfos = &QCI;
  DCI.enabledExtensionCount = static_cast<uint32_t>(EnabledDevExts.size());
  DCI.ppEnabledExtensionNames = EnabledDevExts.data();
  DCI.pEnabledFeatures = &EnabledFeats;

  if (VkResult R = vkCreateDevice(VK->PhysDev, &DCI, nullptr, &VK->Device))
    return vkErr(R, "vkCreateDevice");

  vkGetDeviceQueue(VK->Device, VK->ComputeQF, 0, &VK->Queue);

  SmallVector<std::unique_ptr<offloadtest::Device>> Result;
  Result.push_back(std::make_unique<CrucibleVulkanDevice>(std::move(VK)));
  return Result;
}

} // namespace crucible_offload

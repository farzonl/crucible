//===- VulkanDevice.cpp - Crucible Vulkan offload provider -------*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements offloadtest::Device with a DirectX API personality backed by
// Crucible's DXIL -> SPIR-V compiler and Vulkan (Lavapipe/Mesa/MoltenVK).
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "crucible/Compiler.h"

#include "llvm/Object/DXContainer.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <vulkan/vulkan.h>

#include <algorithm>
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
  std::string DeviceName;
  uint32_t VendorID = 0;
  uint32_t DeviceID = 0;
  uint32_t DriverVersion = 0;

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
  CPUBuffer *CPU = nullptr;
  uint32_t Slice = 0;
};

struct VkImgAlloc {
  VkImage Image = VK_NULL_HANDLE;
  VkDeviceMemory Memory = VK_NULL_HANDLE;
  VkImageView View = VK_NULL_HANDLE;
  uint32_t Width = 1, Height = 1, Depth = 1, Layers = 1;
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
      VkDescriptorType DescType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      VkBuffer Buffer = VK_NULL_HANDLE;
      VkDeviceSize BufferSize = 0;
      VkBufferView BufferView = VK_NULL_HANDLE;
      VkSampler Sampler = VK_NULL_HANDLE;
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
        uint32_t VkBind = Match->SPIRVBinding + SlotOffset;

        if (Res.Kind == ResourceKind::Sampler) {
          if (!Res.SamplerPtr)
            continue;
          auto SampOrErr = getOrCreateSampler(Res.SamplerPtr);
          if (!SampOrErr) {
            vkDestroyShaderModule(VK.Device, ShaderMod, nullptr);
            cleanup();
            return SampOrErr.takeError();
          }
          BoundList.push_back({Match->SPIRVSet, VkBind,
                               VK_DESCRIPTOR_TYPE_SAMPLER, VK_NULL_HANDLE, 0,
                               VK_NULL_HANDLE, *SampOrErr});
          continue;
        }

        if (isTextureKind(Res.Kind)) {
          vkDestroyShaderModule(VK.Device, ShaderMod, nullptr);
          cleanup();
          return createStringError(
              inconvertibleErrorCode(),
              "Crucible Vulkan offload provider: texture resources not yet "
              "wired in compute dispatch");
        }

        if (!Res.BufferPtr)
          continue;
        bool IsUniform = (Res.Kind == ResourceKind::ConstantBuffer);
        bool IsTyped = isTypedBufferKind(Res.Kind);
        bool IsUAV = isUAVKind(Res.Kind);
        auto BufOrErr =
            getOrCreateBuffer(Res.BufferPtr, 0, IsUniform, IsTyped, IsUAV);
        if (!BufOrErr) {
          vkDestroyShaderModule(VK.Device, ShaderMod, nullptr);
          cleanup();
          return BufOrErr.takeError();
        }
        VkBufAlloc *BA = *BufOrErr;
        VkDescriptorType DT =
            IsUniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
            : IsTyped ? (IsUAV ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER
                               : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER)
                      : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        BoundList.push_back({Match->SPIRVSet, VkBind, DT, BA->Buffer,
                             BA->SizeBytes, BA->View, VK_NULL_HANDLE});
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
                    {Match->SPIRVSet,
                     Match->SPIRVBinding +
                         (Info->ShaderRegister - Match->LowerBound),
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
                      {Match->SPIRVSet,
                       Match->SPIRVBinding +
                           (Info->ShaderRegister - Match->LowerBound),
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
      std::vector<VkDescriptorSetLayoutBinding> LayoutBinds;
      for (const auto &B : BoundList) {
        if (B.Set != SetIdx)
          continue;
        VkDescriptorSetLayoutBinding LB{};
        LB.binding = B.Binding;
        LB.descriptorType = B.DescType;
        LB.descriptorCount = 1;
        LB.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        LayoutBinds.push_back(LB);
        PoolCounts[B.DescType]++;
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
        W.descriptorCount = 1;
        W.descriptorType = B.DescType;
        if (B.DescType == VK_DESCRIPTOR_TYPE_SAMPLER) {
          ImgInfos[I].sampler = B.Sampler;
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

  // Copy back UAV buffers to CPU buffers for offload-test-suite verification.
  for (const auto &Buf : AllBufs) {
    if (!Buf.IsUAV || !Buf.CPU || Buf.Slice >= Buf.CPU->Data.size())
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

  // Offscreen color attachment image + view.
  VkImage ColorImg = VK_NULL_HANDLE;
  VkDeviceMemory ColorMem = VK_NULL_HANDLE;
  VkImageView ColorView = VK_NULL_HANDLE;
  VkBuffer ReadbackBuf = VK_NULL_HANDLE;
  VkDeviceMemory ReadbackMem = VK_NULL_HANDLE;
  VkBuffer VertBuf = VK_NULL_HANDLE;
  VkDeviceMemory VertMem = VK_NULL_HANDLE;
  VkRenderPass RenderPass = VK_NULL_HANDLE;
  VkFramebuffer FB = VK_NULL_HANDLE;
  VkPipelineLayout PipeLayout = VK_NULL_HANDLE;
  VkPipeline Pipe = VK_NULL_HANDLE;
  VkCommandPool CmdPool = VK_NULL_HANDLE;

  auto cleanup = [&]() {
    if (CmdPool) vkDestroyCommandPool(VK.Device, CmdPool, nullptr);
    if (Pipe) vkDestroyPipeline(VK.Device, Pipe, nullptr);
    if (PipeLayout) vkDestroyPipelineLayout(VK.Device, PipeLayout, nullptr);
    if (FB) vkDestroyFramebuffer(VK.Device, FB, nullptr);
    if (RenderPass) vkDestroyRenderPass(VK.Device, RenderPass, nullptr);
    if (VertBuf) vkDestroyBuffer(VK.Device, VertBuf, nullptr);
    if (VertMem) vkFreeMemory(VK.Device, VertMem, nullptr);
    if (ReadbackBuf) vkDestroyBuffer(VK.Device, ReadbackBuf, nullptr);
    if (ReadbackMem) vkFreeMemory(VK.Device, ReadbackMem, nullptr);
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
  VkAttachmentDescription ColorAtt{};
  ColorAtt.format = ColorFmt;
  ColorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
  ColorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  ColorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  ColorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  ColorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  ColorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  ColorAtt.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

  VkAttachmentReference ColorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkSubpassDescription Subpass{};
  Subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  Subpass.colorAttachmentCount = 1;
  Subpass.pColorAttachments = &ColorRef;

  VkRenderPassCreateInfo RPCI{};
  RPCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  RPCI.attachmentCount = 1;
  RPCI.pAttachments = &ColorAtt;
  RPCI.subpassCount = 1;
  RPCI.pSubpasses = &Subpass;
  if (VkResult R = vkCreateRenderPass(VK.Device, &RPCI, nullptr, &RenderPass)) {
    cleanup();
    return vkErr(R, "vkCreateRenderPass");
  }

  VkFramebufferCreateInfo FBCI{};
  FBCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  FBCI.renderPass = RenderPass;
  FBCI.attachmentCount = 1;
  FBCI.pAttachments = &ColorView;
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

  VkPipelineLayoutCreateInfo PLCI{};
  PLCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
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
  RSCI.frontFace = VK_FRONT_FACE_CLOCKWISE;
  RSCI.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo MSSCI{};
  MSSCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  MSSCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

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
  GPCI.pColorBlendState = &CBSCI;
  GPCI.layout = PipeLayout;
  GPCI.renderPass = RenderPass;
  GPCI.subpass = 0;

  if (VkResult R = vkCreateGraphicsPipelines(VK.Device, VK_NULL_HANDLE, 1,
                                             &GPCI, nullptr, &Pipe)) {
    cleanup();
    return vkErr(R, "vkCreateGraphicsPipelines");
  }

  VkCommandPoolCreateInfo PoolCI{};
  PoolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  PoolCI.queueFamilyIndex = VK.GraphicsQF;
  if (VkResult R = vkCreateCommandPool(VK.Device, &PoolCI, nullptr, &CmdPool)) {
    cleanup();
    return vkErr(R, "vkCreateCommandPool");
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

  VkClearValue ClearVal{};
  VkRenderPassBeginInfo RPBI{};
  RPBI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  RPBI.renderPass = RenderPass;
  RPBI.framebuffer = FB;
  RPBI.renderArea = {{0, 0}, {Width, Height}};
  RPBI.clearValueCount = 1;
  RPBI.pClearValues = &ClearVal;
  vkCmdBeginRenderPass(CB, &RPBI, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdBindPipeline(CB, VK_PIPELINE_BIND_POINT_GRAPHICS, Pipe);
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
    Description = this->VK->DeviceName;
    DriverName = "Crucible";

    VkPhysicalDeviceSubgroupProperties SubgroupProps{};
    SubgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 Props2{};
    Props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    Props2.pNext = &SubgroupProps;
    vkGetPhysicalDeviceProperties2(this->VK->PhysDev, &Props2);

    VkPhysicalDeviceFeatures2 Feats2{};
    Feats2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    vkGetPhysicalDeviceFeatures2(this->VK->PhysDev, &Feats2);

    Caps.try_emplace("HighestShaderModel",
                     makeStringCapability("HighestShaderModel", "6.6"));
    Caps.try_emplace("DoublePrecisionFloatShaderOps",
                     makeCapability("DoublePrecisionFloatShaderOps",
                                    Feats2.features.shaderFloat64 != 0));
    Caps.try_emplace("Int64ShaderOps",
                     makeCapability("Int64ShaderOps",
                                    Feats2.features.shaderInt64 != 0));
    Caps.try_emplace("Native16BitShaderOpsSupported",
                     makeCapability("Native16BitShaderOpsSupported",
                                    Feats2.features.shaderInt16 != 0));
    bool HasWave = (SubgroupProps.supportedStages &
                    VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
                   SubgroupProps.subgroupSize > 0;
    Caps.try_emplace("WaveOps", makeCapability("WaveOps", HasWave));
    Caps.try_emplace("WaveLaneCountMin",
                     makeCapability<uint32_t>("WaveLaneCountMin",
                                              SubgroupProps.subgroupSize));
    Caps.try_emplace("WaveLaneCountMax",
                     makeCapability<uint32_t>("WaveLaneCountMax",
                                              SubgroupProps.subgroupSize));
    Caps.try_emplace("ResourceBindingTier",
                     makeCapability<uint32_t>("ResourceBindingTier", 3u));
    Caps.try_emplace("TiledResourcesTier",
                     makeCapability<uint32_t>("TiledResourcesTier", 0u));
    Caps.try_emplace("MeshShaderTier",
                     makeStringCapability("MeshShaderTier", "NotSupported"));
    Caps.try_emplace("RaytracingTier",
                     makeStringCapability("RaytracingTier", "NotSupported"));
  }

  const Capabilities &getCapabilities() override { return Caps; }
  StringRef getAPIName() const override { return "DirectX"; }
  GPUAPI getAPI() const override { return GPUAPI::DirectX; }
  void printExtra(raw_ostream &OS) override {
    OS << "  Backend: Vulkan\n"
       << "  VendorID: 0x" << Twine::utohexstr(VK->VendorID) << '\n'
       << "  DeviceID: 0x" << Twine::utohexstr(VK->DeviceID) << '\n';
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
  VK->PhysDev = Devs[0];

  VkPhysicalDeviceProperties Props{};
  vkGetPhysicalDeviceProperties(VK->PhysDev, &Props);
  VK->DeviceName = Props.deviceName;
  VK->VendorID = Props.vendorID;
  VK->DeviceID = Props.deviceID;
  VK->DriverVersion = Props.driverVersion;

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

  VkPhysicalDeviceFeatures SupportedFeats{};
  vkGetPhysicalDeviceFeatures(VK->PhysDev, &SupportedFeats);
  VkPhysicalDeviceFeatures EnabledFeats{};
  EnabledFeats.shaderInt64 = SupportedFeats.shaderInt64;
  EnabledFeats.shaderFloat64 = SupportedFeats.shaderFloat64;
  EnabledFeats.shaderInt16 = SupportedFeats.shaderInt16;

  float Prio = 1.0f;
  VkDeviceQueueCreateInfo QCI{};
  QCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  QCI.queueFamilyIndex = VK->ComputeQF;
  QCI.queueCount = 1;
  QCI.pQueuePriorities = &Prio;

  VkDeviceCreateInfo DCI{};
  DCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  DCI.queueCreateInfoCount = 1;
  DCI.pQueueCreateInfos = &QCI;
  DCI.pEnabledFeatures = &EnabledFeats;

  if (VkResult R = vkCreateDevice(VK->PhysDev, &DCI, nullptr, &VK->Device))
    return vkErr(R, "vkCreateDevice");

  vkGetDeviceQueue(VK->Device, VK->ComputeQF, 0, &VK->Queue);

  SmallVector<std::unique_ptr<offloadtest::Device>> Result;
  Result.push_back(std::make_unique<CrucibleVulkanDevice>(std::move(VK)));
  return Result;
}

} // namespace crucible_offload

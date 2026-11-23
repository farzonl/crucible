//===- VulkanBackend.cpp - Vulkan compute backend for crucible-run -*- C++ -*-===//
//
// Part of Crucible, an open-source DXIL/SPIR-V compiler.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace crucible_run {
namespace {

struct SpecializationSpec {
  std::vector<VkSpecializationMapEntry> Entries;
  std::vector<uint8_t> Data{0};
};

void checkVk(VkResult R, const char *What) {
  if (R != VK_SUCCESS)
    fail(std::string(What) + " failed (VkResult=" + std::to_string(R) + ")");
}

SpecializationSpec parseSpecialization(const std::string &Arg) {
  SpecializationSpec Spec;
  if (Arg == "default")
    return Spec;
  for (const auto &Part : split(Arg, ',')) {
    auto Pair = split(Part, ':');
    char *End = nullptr;
    unsigned long ID = strtoul(Pair[0].c_str(), &End, 10);
    if (Pair.size() != 2 || Pair[0].empty() || *End || ID > UINT32_MAX ||
        Pair[1].empty() || Pair[1].size() % 2)
      fail("bad specialization (want id:hexbytes[,id:hexbytes] or default)");
    VkSpecializationMapEntry Entry{static_cast<uint32_t>(ID),
                                   static_cast<uint32_t>(Spec.Data.size()),
                                   Pair[1].size() / 2};
    for (size_t I = 0; I < Pair[1].size(); I += 2) {
      auto Byte = Pair[1].substr(I, 2);
      if (Byte.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos)
        fail("bad specialization byte '" + Byte + "'");
      Spec.Data.push_back(
          static_cast<uint8_t>(strtoul(Byte.c_str(), nullptr, 16)));
    }
    Spec.Entries.push_back(Entry);
    Spec.Data.push_back(0);
  }
  return Spec;
}

std::vector<uint32_t> readSPIRV(const std::string &Path) {
  std::ifstream F(Path, std::ios::binary | std::ios::ate);
  if (!F)
    fail("cannot open SPIR-V file '" + Path + "'");
  std::streamsize Size = F.tellg();
  if (Size <= 0 || (Size % 4) != 0)
    fail("invalid SPIR-V file size for '" + Path + "'");
  F.seekg(0, std::ios::beg);
  std::vector<uint32_t> Words(static_cast<size_t>(Size) / 4);
  if (!F.read(reinterpret_cast<char *>(Words.data()), Size))
    fail("failed reading '" + Path + "'");
  return Words;
}

uint32_t findMemoryType(VkPhysicalDevice PhysDev, uint32_t TypeFilter,
                        VkMemoryPropertyFlags Props) {
  VkPhysicalDeviceMemoryProperties MemProps;
  vkGetPhysicalDeviceMemoryProperties(PhysDev, &MemProps);
  for (uint32_t I = 0; I < MemProps.memoryTypeCount; ++I) {
    if ((TypeFilter & (1u << I)) &&
        (MemProps.memoryTypes[I].propertyFlags & Props) == Props)
      return I;
  }
  fail("no suitable host-visible Vulkan memory type found");
}

struct AllocatedBuffer {
  BufferSpec Spec;
  VkBuffer Buffer = VK_NULL_HANDLE;
  VkDeviceMemory Memory = VK_NULL_HANDLE;
  VkDeviceSize SizeBytes = 0;
};

} // namespace

int runVulkan(const RunnerOptions &Opts) {
  if (Opts.SpvPath.empty())
    fail("--spv <file.spv> is required for vulkan backend");

  std::string Entry = Opts.Entry.empty() ? "main" : Opts.Entry;
  std::vector<SpecializationSpec> Specializations;
  Specializations.reserve(Opts.SpecializationArgs.size());
  for (const std::string &Arg : Opts.SpecializationArgs)
    Specializations.push_back(parseSpecialization(Arg));

  std::vector<uint32_t> Code = readSPIRV(Opts.SpvPath);

  VkApplicationInfo AppInfo{};
  AppInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  AppInfo.pApplicationName = "crucible-run";
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

  VkInstance Instance = VK_NULL_HANDLE;
  checkVk(vkCreateInstance(&InstCI, nullptr, &Instance), "vkCreateInstance");

  uint32_t DevCount = 0;
  checkVk(vkEnumeratePhysicalDevices(Instance, &DevCount, nullptr),
          "vkEnumeratePhysicalDevices");
  if (DevCount == 0)
    fail("no Vulkan physical devices available");
  std::vector<VkPhysicalDevice> PhysDevs(DevCount);
  checkVk(vkEnumeratePhysicalDevices(Instance, &DevCount, PhysDevs.data()),
          "vkEnumeratePhysicalDevices");
  VkPhysicalDevice PhysDev = PhysDevs[0];

  uint32_t QFCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(PhysDev, &QFCount, nullptr);
  std::vector<VkQueueFamilyProperties> QFProps(QFCount);
  vkGetPhysicalDeviceQueueFamilyProperties(PhysDev, &QFCount, QFProps.data());
  uint32_t ComputeQF = UINT32_MAX;
  for (uint32_t I = 0; I < QFCount; ++I) {
    if (QFProps[I].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      ComputeQF = I;
      break;
    }
  }
  if (ComputeQF == UINT32_MAX)
    fail("no Vulkan compute queue family found");

  float QueuePrio = 1.0f;
  VkDeviceQueueCreateInfo QueueCI{};
  QueueCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  QueueCI.queueFamilyIndex = ComputeQF;
  QueueCI.queueCount = 1;
  QueueCI.pQueuePriorities = &QueuePrio;

  VkDeviceCreateInfo DevCI{};
  DevCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  DevCI.queueCreateInfoCount = 1;
  DevCI.pQueueCreateInfos = &QueueCI;

  VkDevice Device = VK_NULL_HANDLE;
  checkVk(vkCreateDevice(PhysDev, &DevCI, nullptr, &Device), "vkCreateDevice");

  VkQueue Queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(Device, ComputeQF, 0, &Queue);

  // Allocate buffers and populate initial contents.
  std::vector<AllocatedBuffer> Bufs;
  std::vector<VkDescriptorSetLayoutBinding> Bindings;
  uint32_t NumStorage = 0, NumUniform = 0;

  for (const BufferSpec &Spec : Opts.Buffers) {
    AllocatedBuffer AB;
    AB.Spec = Spec;
    // Pad uniform buffers to 16-byte multiples (std140 vec4 rows).
    AB.SizeBytes = Spec.Count * 4;
    if (Spec.IsUniform && (AB.SizeBytes % 16) != 0)
      AB.SizeBytes = ((AB.SizeBytes + 15) / 16) * 16;

    VkBufferCreateInfo BCI{};
    BCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    BCI.size = AB.SizeBytes;
    BCI.usage = Spec.IsUniform ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                               : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    BCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    checkVk(vkCreateBuffer(Device, &BCI, nullptr, &AB.Buffer), "vkCreateBuffer");

    VkMemoryRequirements MemReq{};
    vkGetBufferMemoryRequirements(Device, AB.Buffer, &MemReq);
    VkMemoryAllocateInfo MAI{};
    MAI.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    MAI.allocationSize = MemReq.size;
    MAI.memoryTypeIndex = findMemoryType(
        PhysDev, MemReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    checkVk(vkAllocateMemory(Device, &MAI, nullptr, &AB.Memory),
            "vkAllocateMemory");
    checkVk(vkBindBufferMemory(Device, AB.Buffer, AB.Memory, 0),
            "vkBindBufferMemory");

    void *Mapped = nullptr;
    checkVk(vkMapMemory(Device, AB.Memory, 0, AB.SizeBytes, 0, &Mapped),
            "vkMapMemory");
    memset(Mapped, 0, AB.SizeBytes);
    fillBuffer(Spec, Mapped);
    vkUnmapMemory(Device, AB.Memory);

    VkDescriptorSetLayoutBinding B{};
    B.binding = Spec.Index;
    B.descriptorType = Spec.IsUniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                      : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    B.descriptorCount = 1;
    B.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    Bindings.push_back(B);
    if (Spec.IsUniform)
      ++NumUniform;
    else
      ++NumStorage;
    Bufs.push_back(AB);
  }

  VkDescriptorSetLayoutCreateInfo DSLCI{};
  DSLCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  DSLCI.bindingCount = static_cast<uint32_t>(Bindings.size());
  DSLCI.pBindings = Bindings.data();
  VkDescriptorSetLayout DSL = VK_NULL_HANDLE;
  checkVk(vkCreateDescriptorSetLayout(Device, &DSLCI, nullptr, &DSL),
          "vkCreateDescriptorSetLayout");

  VkPipelineLayoutCreateInfo PLCI{};
  PLCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  PLCI.setLayoutCount = 1;
  PLCI.pSetLayouts = &DSL;
  VkPipelineLayout PipeLayout = VK_NULL_HANDLE;
  checkVk(vkCreatePipelineLayout(Device, &PLCI, nullptr, &PipeLayout),
          "vkCreatePipelineLayout");

  VkShaderModuleCreateInfo SMCI{};
  SMCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  SMCI.codeSize = Code.size() * sizeof(uint32_t);
  SMCI.pCode = Code.data();
  VkShaderModule ShaderMod = VK_NULL_HANDLE;
  checkVk(vkCreateShaderModule(Device, &SMCI, nullptr, &ShaderMod),
          "vkCreateShaderModule");

  VkComputePipelineCreateInfo CPCI{};
  CPCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  CPCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  CPCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  CPCI.stage.module = ShaderMod;
  CPCI.stage.pName = Entry.c_str();
  CPCI.layout = PipeLayout;
  VkPipelineCache PipelineCache = VK_NULL_HANDLE;
  if (!Specializations.empty()) {
    VkPipelineCacheCreateInfo CacheCI{};
    CacheCI.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    checkVk(vkCreatePipelineCache(Device, &CacheCI, nullptr, &PipelineCache),
            "vkCreatePipelineCache");
  }
  std::vector<VkPipeline> Pipelines;
  size_t Variants = Specializations.empty() ? 1 : Specializations.size();
  for (size_t I = 0; I < Variants; ++I) {
    VkSpecializationInfo Info{};
    if (!Specializations.empty()) {
      auto &Spec = Specializations[I];
      Info = {static_cast<uint32_t>(Spec.Entries.size()), Spec.Entries.data(),
              Spec.Data.size(), Spec.Data.data()};
      CPCI.stage.pSpecializationInfo = &Info;
    }
    VkPipeline Pipeline = VK_NULL_HANDLE;
    checkVk(vkCreateComputePipelines(Device, PipelineCache, 1, &CPCI, nullptr,
                                     &Pipeline),
            "vkCreateComputePipelines");
    Pipelines.push_back(Pipeline);
  }

  std::vector<VkDescriptorPoolSize> PoolSizes;
  if (NumStorage > 0)
    PoolSizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NumStorage});
  if (NumUniform > 0)
    PoolSizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, NumUniform});

  VkDescriptorPoolCreateInfo DPCI{};
  DPCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  DPCI.maxSets = 1;
  DPCI.poolSizeCount = static_cast<uint32_t>(PoolSizes.size());
  DPCI.pPoolSizes = PoolSizes.data();
  VkDescriptorPool DPool = VK_NULL_HANDLE;
  checkVk(vkCreateDescriptorPool(Device, &DPCI, nullptr, &DPool),
          "vkCreateDescriptorPool");

  VkDescriptorSetAllocateInfo DSAI{};
  DSAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  DSAI.descriptorPool = DPool;
  DSAI.descriptorSetCount = 1;
  DSAI.pSetLayouts = &DSL;
  VkDescriptorSet DSet = VK_NULL_HANDLE;
  checkVk(vkAllocateDescriptorSets(Device, &DSAI, &DSet),
          "vkAllocateDescriptorSets");

  std::vector<VkDescriptorBufferInfo> DBInfos(Bufs.size());
  std::vector<VkWriteDescriptorSet> Writes(Bufs.size());
  for (size_t I = 0; I < Bufs.size(); ++I) {
    DBInfos[I] = {Bufs[I].Buffer, 0, Bufs[I].SizeBytes};
    Writes[I] = {};
    Writes[I].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    Writes[I].dstSet = DSet;
    Writes[I].dstBinding = Bufs[I].Spec.Index;
    Writes[I].descriptorCount = 1;
    Writes[I].descriptorType = Bufs[I].Spec.IsUniform
                                   ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                   : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    Writes[I].pBufferInfo = &DBInfos[I];
  }
  if (!Writes.empty())
    vkUpdateDescriptorSets(Device, static_cast<uint32_t>(Writes.size()),
                           Writes.data(), 0, nullptr);

  VkCommandPoolCreateInfo CmdPoolCI{};
  CmdPoolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  CmdPoolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  CmdPoolCI.queueFamilyIndex = ComputeQF;
  VkCommandPool CmdPool = VK_NULL_HANDLE;
  checkVk(vkCreateCommandPool(Device, &CmdPoolCI, nullptr, &CmdPool),
          "vkCreateCommandPool");

  VkCommandBufferAllocateInfo CBAI{};
  CBAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  CBAI.commandPool = CmdPool;
  CBAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  CBAI.commandBufferCount = 1;
  VkCommandBuffer CmdBuf = VK_NULL_HANDLE;
  checkVk(vkAllocateCommandBuffers(Device, &CBAI, &CmdBuf),
          "vkAllocateCommandBuffers");

  for (size_t I = 0; I < Pipelines.size(); ++I) {
    if (I) {
      checkVk(vkResetCommandBuffer(CmdBuf, 0), "vkResetCommandBuffer");
      for (const auto &AB : Bufs) {
        void *Mapped = nullptr;
        checkVk(vkMapMemory(Device, AB.Memory, 0, AB.SizeBytes, 0, &Mapped),
                "vkMapMemory");
        fillBuffer(AB.Spec, Mapped);
        vkUnmapMemory(Device, AB.Memory);
      }
    }
    if (!Specializations.empty())
      printf("variant%zu:\n", I);
    VkCommandBufferBeginInfo CBBI{};
    CBBI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    CBBI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    checkVk(vkBeginCommandBuffer(CmdBuf, &CBBI), "vkBeginCommandBuffer");
    vkCmdBindPipeline(CmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, Pipelines[I]);
    vkCmdBindDescriptorSets(CmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, PipeLayout,
                            0, 1, &DSet, 0, nullptr);
    vkCmdDispatch(CmdBuf, Opts.Groups[0], Opts.Groups[1], Opts.Groups[2]);
    checkVk(vkEndCommandBuffer(CmdBuf), "vkEndCommandBuffer");

    VkSubmitInfo SI{};
    SI.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    SI.commandBufferCount = 1;
    SI.pCommandBuffers = &CmdBuf;
    checkVk(vkQueueSubmit(Queue, 1, &SI, VK_NULL_HANDLE), "vkQueueSubmit");
    checkVk(vkQueueWaitIdle(Queue), "vkQueueWaitIdle");

    for (const AllocatedBuffer &AB : Bufs) {
      if (AB.Spec.IsUniform)
        continue;
      void *Mapped = nullptr;
      checkVk(vkMapMemory(Device, AB.Memory, 0, AB.SizeBytes, 0, &Mapped),
              "vkMapMemory");
      printBuffer(AB.Spec, Mapped);
      vkUnmapMemory(Device, AB.Memory);
    }
  }

  vkDestroyCommandPool(Device, CmdPool, nullptr);
  vkDestroyDescriptorPool(Device, DPool, nullptr);
  for (auto Pipeline : Pipelines)
    vkDestroyPipeline(Device, Pipeline, nullptr);
  if (PipelineCache)
    vkDestroyPipelineCache(Device, PipelineCache, nullptr);
  vkDestroyShaderModule(Device, ShaderMod, nullptr);
  vkDestroyPipelineLayout(Device, PipeLayout, nullptr);
  vkDestroyDescriptorSetLayout(Device, DSL, nullptr);
  for (const AllocatedBuffer &AB : Bufs) {
    vkDestroyBuffer(Device, AB.Buffer, nullptr);
    vkFreeMemory(Device, AB.Memory, nullptr);
  }
  vkDestroyDevice(Device, nullptr);
  vkDestroyInstance(Instance, nullptr);
  return 0;
}

} // namespace crucible_run

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Standalone adapter regression using the pinned upstream converter headers.
#include <vulkan/vulkan.h>
#include <SPIRVToMSLConverter.h>
#include "../../scripts/cts_moltenvk_hook.h"
#include <cassert>

namespace mvk {
SPIRVToMSLConversionOptions::SPIRVToMSLConversionOptions() {}
MSLResourceBinding::MSLResourceBinding() {}
}

int main(int argc, char **argv) {
    assert(argc == 6); // cruciblec, trace, image, specialization, physical SPIR-V
    // Valid SPIR-V 1.3: main writes 1.0 to storage buffer set 0, binding 0.
    std::vector<uint32_t> spirv = {
        0x07230203, 0x00010300, 0, 15, 0,
        (2u << 16) | 17u, 1,
        (3u << 16) | 14u, 0, 1,
        (5u << 16) | 15u, 5, 8, 0x6e69616d, 0,
        (6u << 16) | 16u, 8, 17, 64, 1, 1,
        (4u << 16) | 71u, 4, 6, 4,
        (3u << 16) | 71u, 5, 2,
        (5u << 16) | 72u, 5, 0, 35, 0,
        (4u << 16) | 71u, 7, 34, 0,
        (4u << 16) | 71u, 7, 33, 0,
        (2u << 16) | 19u, 1,
        (3u << 16) | 33u, 2, 1,
        (3u << 16) | 22u, 3, 32,
        (3u << 16) | 29u, 4, 3,
        (3u << 16) | 30u, 5, 4,
        (4u << 16) | 32u, 6, 12, 5,
        (4u << 16) | 59u, 6, 7, 12,
        (4u << 16) | 21u, 9, 32, 0,
        (4u << 16) | 43u, 9, 10, 0,
        (4u << 16) | 32u, 11, 12, 3,
        (4u << 16) | 43u, 3, 12, 0x3f800000,
        (5u << 16) | 54u, 1, 8, 0, 2,
        (2u << 16) | 248u, 13,
        (6u << 16) | 65u, 11, 14, 7, 10, 10,
        (3u << 16) | 62u, 14, 12,
        (1u << 16) | 253u,
        (1u << 16) | 56u,
    };
    mvk::SPIRVToMSLConversionConfiguration config;
    config.options.entryPointStage = spv::ExecutionModelGLCompute;
    config.options.entryPointName = "main";
    config.options.mslOptions.msl_version = 30100;
    config.options.mslOptions.argument_buffers = false;
    mvk::MSLResourceBinding resource;
    resource.resourceBinding.stage = spv::ExecutionModelGLCompute;
    resource.resourceBinding.desc_set = 0;
    resource.resourceBinding.binding = 0;
    resource.resourceBinding.msl_buffer = 7;
    config.resourceBindings.push_back(resource);
    mvk::SPIRVToMSLConversionResult result;
    result.resultInfo.entryPoint.mtlFunctionName = "cts_main";
    result.msl = "native sentinel";
    unsetenv("CRUCIBLE_CTS_COMPILER");
    assert(crucibleCTSCompile(spirv, config, result));
    assert(result.msl == "native sentinel");
    setenv("CRUCIBLE_CTS_COMPILER", argv[1], 1);
    setenv("CRUCIBLE_CTS_TRACE", argv[2], 1);
    assert(crucibleCTSCompile(spirv, config, result));
    assert(result.msl.find("kernel void cts_main(") != std::string::npos);
    assert(result.msl.find("[[buffer(7)]]") != std::string::npos);
    assert(result.msl.find("native sentinel") == std::string::npos);
    config.resourceBindings[0].resourceBinding.basetype = SPIRV_CROSS_NAMESPACE::SPIRType::AccelerationStructure;
    result.msl = "native sentinel";
    assert(!crucibleCTSCompile(spirv, config, result));
    assert(result.msl.empty());
    config.resourceBindings[0].resourceBinding.basetype = SPIRV_CROSS_NAMESPACE::SPIRType::Struct;
    config.resourceBindings[0].resourceBinding.count = 2;
    assert(!crucibleCTSCompile(spirv, config, result));
    assert(result.msl.empty());
    config.resourceBindings[0].resourceBinding.count = 1;
    for (auto scalar : {std::pair<uint32_t, uint32_t>{20, 1}, {21, 8}, {21, 16},
                        {21, 64}, {22, 16}, {22, 32}, {22, 64}}) {
        uint32_t opcode = scalar.first, width = scalar.second;
        auto source = spirv;
        source[3] = 17;
        uint32_t capability = opcode == 20 || width == 32 ? 1 :
                              opcode == 22 ? (width == 16 ? 9 : 10) :
                              width == 8 ? 39 : width == 16 ? 22 : 11;
        if (capability != 1)
            source.insert(source.begin() + 5, {(2u << 16) | 17u, capability});
        // Decorations precede type declarations.
        size_t type = 5;
        while ((source[type] & 0xffff) != 19)
            type += source[type] >> 16;
        source.insert(source.begin() + type, {(4u << 16) | 71u, 16, 1, 42});
        size_t function = 5;
        while ((source[function] & 0xffff) != 54)
            function += source[function] >> 16;
        std::vector<uint32_t> declarations;
        uint32_t typeID = width == 32 ? 3 : 15;
        if (width != 32) {
            declarations = {((opcode == 20 ? 2u : opcode == 21 ? 4u : 3u) << 16) | opcode, 15};
            if (opcode != 20) { declarations.push_back(width); }
            if (opcode == 21) { declarations.push_back(1); }
        }
        if (width == 1)
            declarations.insert(declarations.end(), {(3u << 16) | 49u, typeID, 16});
        else {
            declarations.insert(declarations.end(),
                                {((width == 64 ? 5u : 4u) << 16) | 50u, typeID, 16, 0});
            if (width == 64) { declarations.push_back(0); }
        }
        source.insert(source.begin() + function, declarations.begin(), declarations.end());
        uint64_t bits = width == 1 ? 1 : width == 64 ? 0xbff4000012345678ull :
                        width == 32 ? 0xc0200000 : width == 16 ? 0xbc00 : 0x80;
        uint8_t bytes[9] = {};
        size_t size = width == 1 ? 4 : width / 8;
        std::memcpy(bytes + 1, &bits, size);
        VkSpecializationMapEntry entry{42, 1, size};
        VkSpecializationInfo info{1, &entry, sizeof(bytes), bytes};
        std::vector<uint32_t> specialized;
        std::string log;
        if (!crucibleCTSSpecialize(source, &info, specialized, log)) {
            std::fprintf(stderr, "scalar opcode %u width %u: %s", opcode, width, log.c_str());
            std::abort();
        }
        bool found = false;
        for (size_t i = 5; i < specialized.size(); i += specialized[i] >> 16) {
            uint32_t op = specialized[i] & 0xffff;
            if ((op == 41 || op == 43) && specialized[i + 2] == 16) {
                if (width == 1)
                    assert(op == 41);
                else {
                    uint64_t actual = specialized[i + 3];
                    if (width == 64) { actual |= uint64_t(specialized[i + 4]) << 32; }
                    // Signed narrow integers may have their unused high bits extended.
                    uint64_t mask = width == 64 ? ~uint64_t(0) : (uint64_t(1) << width) - 1;
                    assert((actual & mask) == bits);
                }
                found = true;
            }
        }
        assert(found);
    }
    {
        std::ifstream stream(argv[4], std::ios::binary | std::ios::ate);
        assert(stream);
        auto bytes = stream.tellg();
        assert(bytes > 0 && bytes % sizeof(uint32_t) == 0);
        std::vector<uint32_t> source(bytes / sizeof(uint32_t)), defaults, variant;
        stream.seekg(0);
        stream.read(reinterpret_cast<char*>(source.data()), bytes);
        assert(stream);
        std::string log;
        assert(crucibleCTSHasSpecialization(source));
        assert(crucibleCTSSpecialize(source, nullptr, defaults, log));
        assert(!crucibleCTSHasSpecialization(defaults));
        assert(crucibleCTSCompile(defaults, config, result));
        assert(result.msl.find("22u") != std::string::npos);
        // Deliberately unaligned offsets and shuffled IDs, including an unused ID.
        uint8_t data[17] = {};
        uint32_t value = 40, size = 4;
        std::memcpy(data + 1, &value, 4);
        std::memcpy(data + 6, &size, 4);
        VkSpecializationMapEntry entries[] = {{9, 11, 4}, {8, 6, 4},
                                             {7, 1, 4}, {1234, 16, 1}};
        VkSpecializationInfo info{4, entries, sizeof(data), data};
        assert(crucibleCTSSpecialize(source, &info, variant, log));
        assert(variant != defaults);
        assert(crucibleCTSCompile(variant, config, result));
        assert(result.msl.find("40u") != std::string::npos);
        assert(crucibleCTSSpecialize(source, nullptr, variant, log));
        assert(variant == defaults);
        entries[0].size = 1;
        assert(!crucibleCTSSpecialize(source, &info, variant, log));
        assert(variant.empty());
        entries[0].size = 4;
        entries[0].offset = 15;
        assert(!crucibleCTSSpecialize(source, &info, variant, log));
        entries[0].offset = 11;
        entries[0].constantID = 7;
        assert(!crucibleCTSSpecialize(source, &info, variant, log));
        setenv("CRUCIBLE_CTS_SPIRV_OPT", "/bin/false", 1);
        assert(!crucibleCTSSpecialize(source, nullptr, variant, log));
        assert(variant.empty());
        unsetenv("CRUCIBLE_CTS_SPIRV_OPT");
    }
    {
        std::ifstream stream(argv[3], std::ios::binary | std::ios::ate);
        assert(stream);
        auto bytes = stream.tellg();
        assert(bytes > 0 && bytes % sizeof(uint32_t) == 0);
        std::vector<uint32_t> imageSpirv(bytes / sizeof(uint32_t));
        stream.seekg(0);
        stream.read(reinterpret_cast<char *>(imageSpirv.data()), bytes);
        assert(stream);
        auto imageConfig = config;
        imageConfig.resourceBindings.clear();
        using Base = SPIRV_CROSS_NAMESPACE::SPIRType;
        for (uint32_t binding = 0; binding < 5; ++binding) {
            auto imageResource = resource;
            auto& b = imageResource.resourceBinding;
            b.binding = binding;
            b.count = 1;
            b.basetype = binding == 0 ? Base::SampledImage :
                         binding == 1 || binding == 3 ? Base::Image :
                         binding == 2 ? Base::Sampler : Base::Struct;
            b.msl_texture = binding == 0 ? 5 : binding == 1 ? 6 : 7;
            b.msl_sampler = binding == 0 ? 3 : 4;
            b.msl_buffer = 7;
            imageConfig.resourceBindings.push_back(imageResource);
        }
        assert(crucibleCTSCompile(imageSpirv, imageConfig, result));
        for (auto attribute : {"texture(5)", "texture(6)", "texture(7)",
                               "sampler(3)", "sampler(4)", "buffer(7)"})
            assert(result.msl.find(attribute) != std::string::npos);
        imageConfig.resourceBindings[0].requiresConstExprSampler = true;
        assert(!crucibleCTSCompile(imageSpirv, imageConfig, result));
        assert(result.msl.empty());
    }
    {
        std::ifstream stream(argv[5], std::ios::binary | std::ios::ate);
        assert(stream);
        auto bytes = stream.tellg();
        assert(bytes > 0 && bytes % sizeof(uint32_t) == 0);
        std::vector<uint32_t> physicalSpirv(bytes / sizeof(uint32_t));
        stream.seekg(0);
        stream.read(reinterpret_cast<char *>(physicalSpirv.data()), bytes);
        assert(stream);
        result.resultInfo.usesPhysicalStorageBufferAddressesCapability = true;
        result.msl = "native sentinel";
        assert(crucibleCTSCompile(physicalSpirv, config, result));
        assert(result.resultInfo.usesPhysicalStorageBufferAddressesCapability);
        assert(result.msl.find("[[buffer(7)]]") != std::string::npos);
        assert(result.msl.find("device float") != std::string::npos);
        assert(result.msl.find("native sentinel") == std::string::npos);
        result.resultInfo.usesPhysicalStorageBufferAddressesCapability = false;
    }
    result.msl = "native sentinel";
    setenv("CRUCIBLE_CTS_COMPILER", "/bin/false", 1);
    assert(!crucibleCTSCompile(spirv, config, result));
    assert(result.msl.empty());
    config.options.entryPointStage = spv::ExecutionModelVertex;
    result.msl = "native sentinel";
    assert(!crucibleCTSCompile(spirv, config, result));
    assert(result.msl.empty());
    config.options.entryPointStage = spv::ExecutionModelGLCompute;
    std::vector<uint32_t> specialized = spirv;
    specialized.push_back((1u << 16) | 50u);
    assert(!crucibleCTSCompile(specialized, config, result));
    assert(result.msl.empty());
    setenv("CRUCIBLE_CTS_COMPILER", "", 1);
    assert(!crucibleCTSCompile(spirv, config, result));
    assert(result.msl.empty());
}

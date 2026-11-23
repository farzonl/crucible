// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Standalone adapter regression using the pinned upstream converter headers.
#include <SPIRVToMSLConverter.h>
#include "../../scripts/cts_moltenvk_hook.h"
#include <cassert>

namespace mvk {
SPIRVToMSLConversionOptions::SPIRVToMSLConversionOptions() {}
MSLResourceBinding::MSLResourceBinding() {}
}

int main(int argc, char **argv) {
    assert(argc == 4); // cruciblec, trace path, image SPIR-V
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

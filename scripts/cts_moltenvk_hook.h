// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Included only in the pinned CTS build of MoltenVK, not in production drivers.
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iterator>
#include <map>
#include <spawn.h>
#include <signal.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char **environ;

static bool crucibleCTSTrace(const char *event);

static int crucibleCTSRun(std::vector<std::string> arguments,
                          const std::string& diagnostics, bool compiler) {
    std::vector<char *> argv;
    for (auto& argument : arguments) { argv.push_back(&argument[0]); }
    argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    int error = posix_spawn_file_actions_init(&actions);
    if (error) { return error; }
    error = posix_spawn_file_actions_addopen(
        &actions, STDERR_FILENO, diagnostics.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (!error) {
        error = posix_spawn_file_actions_adddup2(&actions, STDERR_FILENO, STDOUT_FILENO);
    }
    pid_t pid = -1;
    if (!error && compiler && !crucibleCTSTrace("attempt")) { error = EIO; }
    if (!error) {
        error = posix_spawnp(&pid, argv[0], &actions, nullptr, argv.data(), environ);
    }
    posix_spawn_file_actions_destroy(&actions);
    if (error) { return error; }
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    int status = 0;
    while (true) {
        pid_t waited = waitpid(pid, &status, WNOHANG);
        if (waited == pid) {
            return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : EINVAL;
        }
        if (waited < 0 && errno != EINTR) { return errno; }
        if (std::chrono::steady_clock::now() >= deadline) {
            kill(pid, SIGKILL);
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
            return ETIMEDOUT;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

static bool crucibleCTSTrace(const char *event) {
    const char *path = std::getenv("CRUCIBLE_CTS_TRACE");
    if (!path || !*path) { return false; }
    FILE *file = std::fopen(path, "a");
    if (!file) { return false; }
    bool ok = std::fprintf(file, "%s\n", event) > 0;
    return std::fclose(file) == 0 && ok;
}

static bool crucibleCTSHasSpecialization(const std::vector<uint32_t>& spirv) {
    for (size_t i = 5; i < spirv.size();) {
        uint32_t count = spirv[i] >> 16, opcode = spirv[i] & 0xffff;
        if (!count || count > spirv.size() - i) { return true; }
        if (opcode >= 48 && opcode <= 52) { return true; }
        i += count;
    }
    return false;
}

static bool crucibleCTSSpecialize(const std::vector<uint32_t>& spirv,
                                  const VkSpecializationInfo* info,
                                  std::vector<uint32_t>& specialized,
                                  std::string& diagnostics) {
    specialized.clear();
    auto fail = [&](const char* message) {
        specialized.clear();
        diagnostics += std::string("\nCrucible CTS specialization: ") + message + "\n";
        if (!crucibleCTSTrace("adapter_failed"))
            diagnostics += "Cannot write compiler attribution trace\n";
        return false;
    };
    if (spirv.size() < 5 || spirv[0] != 0x07230203) {
        return fail("invalid SPIR-V header");
    }
    if (info && ((info->mapEntryCount && !info->pMapEntries) ||
                 (info->dataSize && !info->pData))) {
        return fail("invalid specialization pointers");
    }
    std::map<uint32_t, const VkSpecializationMapEntry*> overrides;
    if (info) {
        for (uint32_t i = 0; i < info->mapEntryCount; ++i) {
            const auto& entry = info->pMapEntries[i];
            if (!entry.size || entry.offset > info->dataSize ||
                entry.size > info->dataSize - entry.offset ||
                !overrides.emplace(entry.constantID, &entry).second) {
                return fail("invalid specialization range or duplicate constant ID");
            }
        }
    }
    std::map<uint32_t, uint32_t> specIDs, widths;
    std::map<uint32_t, bool> signedTypes;
    for (size_t i = 5; i < spirv.size();) {
        uint32_t count = spirv[i] >> 16, opcode = spirv[i] & 0xffff;
        if (!count || count > spirv.size() - i) {
            return fail("malformed SPIR-V instruction");
        }
        if (opcode == 71 && count >= 4 && spirv[i + 2] == 1) {
            specIDs[spirv[i + 1]] = spirv[i + 3];
        } else if (opcode == 20 && count == 2) {
            widths[spirv[i + 1]] = 1;
        } else if ((opcode == 21 && count == 4) || (opcode == 22 && count == 3)) {
            widths[spirv[i + 1]] = spirv[i + 2];
            if (opcode == 21) { signedTypes[spirv[i + 1]] = spirv[i + 3] != 0; }
        } else if (opcode == 73) {
            return fail("decoration groups are not supported for specialization");
        }
        i += count;
    }
    specialized = spirv;
    for (size_t i = 5; i < specialized.size();) {
        uint32_t count = specialized[i] >> 16, opcode = specialized[i] & 0xffff;
        if (opcode >= 48 && opcode <= 50) {
            if (count < 3) { return fail("malformed specialization constant"); }
            auto id = specIDs.find(specialized[i + 2]);
            auto entry = id == specIDs.end() ? overrides.end() : overrides.find(id->second);
            if (entry != overrides.end()) {
                auto type = widths.find(specialized[i + 1]);
                if (type == widths.end() ||
                    (type->second != 1 && type->second != 8 && type->second != 16 &&
                     type->second != 32 && type->second != 64)) {
                    return fail("unsupported specialization constant type");
                }
                uint32_t width = type->second;
                size_t size = width == 1 ? sizeof(VkBool32) : width / 8;
                if (entry->second->size != size ||
                    (width == 1 ? count != 3 : count != 3 + (width + 31) / 32)) {
                    return fail("specialization constant size does not match its type");
                }
                const char* data = static_cast<const char*>(info->pData) + entry->second->offset;
                if (width == 1) {
                    VkBool32 value;
                    std::memcpy(&value, data, size);
                    specialized[i] = (count << 16) | (value ? 48u : 49u);
                } else {
                    std::memset(&specialized[i + 3], 0, (count - 3) * sizeof(uint32_t));
                    std::memcpy(&specialized[i + 3], data, size);
                    if (width < 32 && signedTypes[specialized[i + 1]] &&
                        (specialized[i + 3] & (1u << (width - 1)))) {
                        specialized[i + 3] |= ~((1u << width) - 1);
                    }
                }
            }
        }
        i += count;
    }
    char directory[] = "/tmp/crucible-cts-spec-XXXXXX";
    if (!mkdtemp(directory)) { return fail("cannot create specialization directory"); }
    std::string input = std::string(directory) + "/input.spv";
    std::string output = std::string(directory) + "/output.spv";
    std::string logPath = std::string(directory) + "/diagnostics.txt";
    auto cleanup = [&]() {
        unlink(input.c_str());
        unlink(output.c_str());
        unlink(logPath.c_str());
        rmdir(directory);
    };
    {
        std::ofstream stream(input, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(specialized.data()),
                     specialized.size() * sizeof(uint32_t));
        stream.close();
        if (!stream) {
            cleanup();
            return fail("cannot write specialization input");
        }
    }
    const char* optimizer = std::getenv("CRUCIBLE_CTS_SPIRV_OPT");
    int error = crucibleCTSRun(
        {optimizer ? optimizer : "spirv-opt", input, "-o", output,
         "--freeze-spec-const", "--fold-spec-const-op-composite"}, logPath, false);
    std::ifstream log(logPath);
    diagnostics.append(std::istreambuf_iterator<char>(log), {});
    std::ifstream stream(output, std::ios::binary | std::ios::ate);
    std::streamoff bytes = stream ? std::streamoff(stream.tellg()) : 0;
    if (error || bytes < 20 || bytes % sizeof(uint32_t)) {
        cleanup();
        return fail("specialization optimizer failed or produced invalid output");
    }
    specialized.resize(bytes / sizeof(uint32_t));
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(specialized.data()), bytes);
    cleanup();
    if (!stream || crucibleCTSHasSpecialization(specialized)) {
        return fail("specialization could not be fully resolved");
    }
    return true;
}

static bool crucibleCTSCompile(const std::vector<uint32_t>& spirv,
                               const mvk::SPIRVToMSLConversionConfiguration& config,
                               mvk::SPIRVToMSLConversionResult& result) {
    const char *compiler = std::getenv("CRUCIBLE_CTS_COMPILER");
    if (!compiler) { return true; } // Matched native reference.
    result.msl.clear(); // Never allow native MSL to escape the Crucible track.
    auto fail = [&](const char *message, const char *event) {
        result.resultLog += std::string("\nCrucible CTS: ") + message + "\n";
        if (!crucibleCTSTrace(event)) {
            result.resultLog += "Cannot write compiler attribution trace\n";
        }
        return false;
    };
    if (!*compiler) { return fail("empty compiler path", "failed"); }
    const auto& info = result.resultInfo;
    if (config.options.entryPointStage != spv::ExecutionModelGLCompute ||
        config.options.mslOptions.argument_buffers ||
        info.needsSwizzleBuffer || info.needsOutputBuffer || info.needsPatchOutputBuffer ||
        info.needsBufferSizeBuffer || info.needsDynamicOffsetBuffer ||
        info.needsInputThreadgroupMem || info.needsDispatchBaseBuffer ||
        info.needsViewRangeBuffer || info.needsDrawId ||
        !info.specializationMacros.empty()) {
        return fail("unsupported runtime ABI (initial adapter is direct-binding compute)",
                    "adapter_failed");
    }
    for (size_t i = 5; i < spirv.size();) {
        uint32_t count = spirv[i] >> 16;
        uint32_t opcode = spirv[i] & 0xffff;
        if (!count || count > spirv.size() - i) {
            return fail("malformed SPIR-V instruction", "adapter_failed");
        }
        if (opcode >= 48 && opcode <= 52) {
            return fail("pipeline specialization must be resolved before compilation",
                        "adapter_failed");
        }
        i += count;
    }
    char directory[] = "/tmp/crucible-cts-XXXXXX";
    if (!mkdtemp(directory)) { return fail("cannot create shader directory", "failed"); }
    std::string input = std::string(directory) + "/shader.spv";
    std::string output = std::string(directory) + "/shader.metal";
    std::string diagnostics = std::string(directory) + "/diagnostics.txt";
    auto cleanup = [&]() {
        unlink(input.c_str());
        unlink(output.c_str());
        unlink(diagnostics.c_str());
        rmdir(directory);
    };
    {
        std::ofstream stream(input, std::ios::binary);
        stream.write(reinterpret_cast<const char *>(spirv.data()),
                     spirv.size() * sizeof(uint32_t));
        stream.close();
        if (!stream) {
            cleanup();
            return fail("cannot write shader input", "failed");
        }
    }
    std::vector<std::string> arguments = {
        compiler, input, "-o", output, "--target=msl", "--require-msl-buffer-bindings",
        "--entry=" + config.options.entryPointName,
        "--msl-entry=" + info.entryPoint.mtlFunctionName,
    };
    uint32_t version = config.options.mslOptions.msl_version;
    arguments.push_back("--msl-std=metal" + std::to_string(version / 10000) + "." +
                        std::to_string((version / 100) % 100));
    for (const auto& resource : config.resourceBindings) {
        const auto& b = resource.resourceBinding;
        if (b.stage != config.options.entryPointStage) { continue; }
        using BaseType = SPIRV_CROSS_NAMESPACE::SPIRType;
        if (resource.requiresConstExprSampler || b.count > 1 ||
            b.basetype == BaseType::AccelerationStructure) {
            cleanup();
            return fail("descriptor arrays, constexpr samplers and acceleration structures are not supported by the adapter",
                        "adapter_failed");
        }
        // Crucible's SPIR-V loader represents PushConstant with space ~1u.
        uint32_t space = b.desc_set == ~0u ? ~1u : b.desc_set;
        auto binding = std::to_string(space) + "," + std::to_string(b.binding) + ",";
        if (b.basetype == BaseType::Image || b.basetype == BaseType::SampledImage)
            arguments.push_back("--msl-texture-binding=" + binding + std::to_string(b.msl_texture));
        if (b.basetype == BaseType::Sampler || b.basetype == BaseType::SampledImage)
            arguments.push_back("--msl-sampler-binding=" + binding + std::to_string(b.msl_sampler));
        if (b.basetype != BaseType::Image && b.basetype != BaseType::SampledImage &&
            b.basetype != BaseType::Sampler)
            arguments.push_back("--msl-buffer-binding=" + binding + std::to_string(b.msl_buffer));
    }
    int error = crucibleCTSRun(arguments, diagnostics, true);
    std::ifstream log(diagnostics);
    result.resultLog.append(std::istreambuf_iterator<char>(log), {});
    if (error) {
        cleanup();
        return fail("compiler process failed or timed out", "failed");
    }
    std::ifstream source(output);
    if (!source) {
        cleanup();
        return fail("compiler produced no MSL", "failed");
    }
    result.msl.assign(std::istreambuf_iterator<char>(source), {});
    cleanup();
    if (result.msl.empty() || !crucibleCTSTrace("compiled")) {
        result.msl.clear();
        return fail("empty MSL or missing attribution trace", "failed");
    }
    return true;
}

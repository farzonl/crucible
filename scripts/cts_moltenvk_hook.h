// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Included only in the pinned CTS build of MoltenVK, not in production drivers.
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iterator>
#include <spawn.h>
#include <signal.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char **environ;

static bool crucibleCTSTrace(const char *event) {
    const char *path = std::getenv("CRUCIBLE_CTS_TRACE");
    if (!path || !*path) { return false; }
    FILE *file = std::fopen(path, "a");
    if (!file) { return false; }
    bool ok = std::fprintf(file, "%s\n", event) > 0;
    return std::fclose(file) == 0 && ok;
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
        info.usesPhysicalStorageBufferAddressesCapability ||
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
        // Specialization is a pipeline-time operation; the initial adapter cannot
        // bake module defaults and pretend that VkSpecializationInfo was honored.
        if (opcode >= 48 && opcode <= 52) {
            return fail("specialization constants are not supported by the adapter",
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
        if (resource.requiresConstExprSampler) {
            cleanup();
            return fail("immutable samplers are not supported by the adapter", "adapter_failed");
        }
        // Crucible's SPIR-V loader represents PushConstant with space ~1u.
        uint32_t space = b.desc_set == ~0u ? ~1u : b.desc_set;
        arguments.push_back("--msl-buffer-binding=" + std::to_string(space) + "," +
                            std::to_string(b.binding) + "," + std::to_string(b.msl_buffer));
    }
    std::vector<char *> argv;
    for (auto& argument : arguments) { argv.push_back(&argument[0]); }
    argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    int error = posix_spawn_file_actions_init(&actions);
    if (!error) {
        error = posix_spawn_file_actions_addopen(
            &actions, STDERR_FILENO, diagnostics.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (!error) {
            error = posix_spawn_file_actions_adddup2(&actions, STDERR_FILENO, STDOUT_FILENO);
        }
        pid_t pid = -1;
        if (!error && !crucibleCTSTrace("attempt")) { error = EIO; }
        if (!error) { error = posix_spawn(&pid, compiler, &actions, nullptr, argv.data(), environ); }
        posix_spawn_file_actions_destroy(&actions);
        if (!error) {
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            int status = 0;
            while (true) {
                pid_t waited = waitpid(pid, &status, WNOHANG);
                if (waited == pid) {
                    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) { error = EINVAL; }
                    break;
                }
                if (waited < 0 && errno != EINTR) { error = errno; break; }
                if (std::chrono::steady_clock::now() >= deadline) {
                    kill(pid, SIGKILL);
                    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
                    error = ETIMEDOUT;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    }
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

# -*- Python -*-
# Lit configuration running llvm/offload-test-suite's test tree against
# crucible-offload (DXIL -> crucible -> MSL -> Metal, direct bindings).

import os
import platform

import lit.formats

config.name = "Crucible-Offload"
config.test_format = lit.formats.ShTest()
config.suffixes = [".test", ".yaml"]
config.excludes = ["Inputs", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

# The suite caps individual tests so one GPU hang doesn't wedge the run.
if lit_config.maxIndividualTestTime == 0:
    lit_config.maxIndividualTestTime = 300

config.test_source_root = os.path.join(config.offload_suite_dir, "test")
config.test_exec_root = config.crucible_offload_exec_root

# Feature set — crucible advertises as a *DirectX* implementation whose
# backend happens to be Metal (the MoltenVK model: identity is the API you
# implement, not the one you use). This inherits the D3D12 contract's test
# annotations (e.g. REQUIRES: DirectX root-signature tests run and fail
# honestly until implemented) and avoids metal-shaderconverter's Metal
# XFAIL/UNSUPPORTED waivers, which describe MSC's backend, not ours.
# "Crucible" is the driver-level feature (like WARP/NV/AMD/Intel) for
# annotations that target us specifically.
config.available_features.add("DirectX")
config.available_features.add("Crucible")
config.available_features.add("DXC")
config.available_features.add("Darwin")
if platform.machine().lower() in ("arm64", "aarch64"):
    config.available_features.add("host-arm64")
# Scalar shader operations, not the separate 64-bit atomic capabilities.
config.available_features.update(["Half", "Int16", "Int64"])

# Substitution order matters: %dxc_target_lib before %dxc_target.
config.substitutions.append(("%dxc_target_lib", config.dxc))
config.substitutions.append(("%dxc_target", config.dxc))
config.substitutions.append(("%offloader", config.crucible_offload))
if config.crucible_api_query:
    config.substitutions.append(("%api_query", config.crucible_api_query))
for tool in ["FileCheck", "split-file", "obj2yaml", "not"]:
    config.substitutions.append(
        (rf"(?<!\S){tool}(?=\s|$)", os.path.join(config.llvm_tools_dir, tool)))

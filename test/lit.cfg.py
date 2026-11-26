# -*- Python -*-
# Lit configuration for the crucible test suite.

import os
import platform
import shutil
import subprocess

import lit.formats

config.name = "crucible"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".test", ".hlsl", ".ll"]
config.test_source_root = os.path.dirname(__file__)
# "offload" hosts the offload-test-suite harness (its own lit suite rooted
# elsewhere); keep this suite from descending into it.
config.excludes = ["Inputs", "offload"]

# Tool substitutions. Paths are provided by lit.site.cfg.py.
config.substitutions.append(("%dxc", config.dxc))
config.substitutions.append(("%cruciblec", config.cruciblec))
config.substitutions.append(("FileCheck", config.filecheck))

if config.crucible_run:
    config.substitutions.append(("%crucible-run", config.crucible_run))

if getattr(config, "crucible_has_vulkan", False) and config.crucible_run and os.path.exists(config.crucible_run):
    config.available_features.add("vulkan-gpu")

if getattr(config, "crucible_has_directx", False):
    config.available_features.add("directx-backend")

if getattr(config, "nacelle_compile", "") and os.path.exists(config.nacelle_compile):
    config.substitutions.append(("%nacelle-compile", config.nacelle_compile))
    config.available_features.add("nacelle-compile")

if getattr(config, "crucible_has_nacelle", False) and config.crucible_run and os.path.exists(config.crucible_run):
    config.available_features.add("nacelle-d3d12")

if platform.system() == "Windows" and config.crucible_run and os.path.exists(config.crucible_run):
    config.available_features.add("windows-d3d12")
    if os.environ.get("CRUCIBLE_D3D12_WARP"):
        config.environment["CRUCIBLE_D3D12_WARP"] = os.environ["CRUCIBLE_D3D12_WARP"]
    if config.dxc:
        dxil_dll = os.path.join(os.path.dirname(config.dxc), "dxil.dll")
        if os.path.exists(dxil_dll):
            config.environment["CRUCIBLE_DXIL_DLL"] = dxil_dll

if config.crucible_offload:
    config.substitutions.append(("%offloader", f'"{config.crucible_offload}" -api=dx'))
    config.substitutions.append(("%api_query", f'"{config.crucible_api_query}"'))
    config.available_features.add("offload-runner")

if getattr(config, "spirv_val", "") and os.path.exists(config.spirv_val):
    config.substitutions.append(("%spirv-val", config.spirv_val))
    if getattr(config, "spirv_dis", "") and os.path.exists(config.spirv_dis):
        config.substitutions.append(("%spirv-dis", config.spirv_dis))
        config.available_features.add("spirv-tools")
    if getattr(config, "spirv_as", "") and os.path.exists(config.spirv_as):
        config.substitutions.append(("%spirv-as", config.spirv_as))
        config.available_features.add("spirv-assembler")
        for capability, feature in (
            ("FMAKHR", "spirv-fma-khr"),
            ("FloatControls2", "spirv-float-controls2"),
        ):
            if subprocess.run(
                [config.spirv_as, "-o", os.devnull],
                input=f"OpCapability {capability}\n",
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0:
                config.available_features.add(feature)

# Feature flags:
#   metal-tools -- Apple's `xcrun metal` toolchain is available, so emitted
#                  MSL can be compiled with -Werror.
#   gpu         -- a Metal device is available for end-to-end execution.
if platform.system() == "Darwin" and shutil.which("xcrun"):
    if os.system("xcrun --find metal > /dev/null 2>&1") == 0:
        config.available_features.add("metal-tools")
    if config.crucible_run and os.path.exists(config.crucible_run):
        config.available_features.add("gpu")

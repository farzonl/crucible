# -*- Python -*-
# Lit configuration running llvm/offload-test-suite's test tree against
# crucible-offload (DXIL -> crucible -> MSL -> Metal, direct bindings).

import os
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from crucible_lit_format import CrucibleOffloadShTest

import lit.formats
try:
    import yaml
except ImportError:
    lit_config.fatal("Crucible offload discovery requires PyYAML (pip install pyyaml)")

config.name = "Crucible-Offload"
config.test_format = lit.formats.ShTest()
config.suffixes = [".test", ".yaml"]
config.excludes = ["Inputs", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

# The suite caps individual tests so one GPU hang doesn't wedge the run.
if lit_config.maxIndividualTestTime == 0:
    lit_config.maxIndividualTestTime = 300

config.test_source_root = os.path.join(config.offload_suite_dir, "test")
config.test_exec_root = config.crucible_offload_exec_root

# Propagate explicit runtime selection (metal|vulkan|d3d12|warp) to both
# api-query and offloader child processes.
offload_runtime = lit_config.params.get(
    "offload_runtime", os.environ.get("CRUCIBLE_OFFLOAD_RUNTIME")
)
api_query_env = os.environ.copy()
if offload_runtime:
    config.environment["CRUCIBLE_OFFLOAD_RUNTIME"] = offload_runtime
    api_query_env["CRUCIBLE_OFFLOAD_RUNTIME"] = offload_runtime
warp_env = lit_config.params.get(
    "d3d12_warp", os.environ.get("CRUCIBLE_D3D12_WARP")
)
if warp_env:
    config.environment["CRUCIBLE_D3D12_WARP"] = str(warp_env)
    api_query_env["CRUCIBLE_D3D12_WARP"] = str(warp_env)

try:
    report = yaml.safe_load(subprocess.check_output(
        [config.crucible_api_query], text=True, env=api_query_env
    ))
except (OSError, subprocess.CalledProcessError, yaml.YAMLError) as error:
    lit_config.fatal(f"Crucible device discovery failed: {error}")
if lit_config.params.get("spirv_frontend"):
    vulkan_driver = lit_config.params.get("vulkan_driver", "Crucible")
    targets = [
        device for device in report["Devices"]
        if device["API"] == "Vulkan" and device["Driver"] == vulkan_driver
    ]
else:
    targets = [
        device for device in report["Devices"]
        if device["API"] == "DirectX" and device["Driver"] == "Crucible"
    ]
if len(targets) != 1:
    lit_config.fatal(
        f"Expected one matching offload adapter, found {len(targets)}"
    )
target = targets[0]
host = report["Host"]
_is_paravirtual = "paravirtual" in str(target.get("Description", "")).lower()
_use_spirv_frontend = bool(lit_config.params.get("spirv_frontend"))
if _is_paravirtual:
    config.available_features.add("apple-paravirtual")
if _is_paravirtual or _use_spirv_frontend:
    config.test_format = CrucibleOffloadShTest(
        is_paravirtual=_is_paravirtual,
        clear_spirv_xfails=_use_spirv_frontend,
    )

# Use the advertised API contract for test annotations, not the execution
# backend.
config.available_features.update(
    [target["API"], target["Driver"], "DXC", host["Operating System"]]
)
if host["Architecture"].lower() in ("arm64", "aarch64"):
    config.available_features.add("host-arm64")
features = target["Features"]
if target["API"] == "Vulkan":
    for extension in target.get("Extensions", []):
        config.available_features.add(extension["ExtensionName"])
    if features.get("shaderFloat16", False):
        config.available_features.add("Half")
    if features.get("shaderInt16", False):
        config.available_features.add("Int16")
    if features.get("shaderInt64", False):
        config.available_features.add("Int64")
    if features.get("shaderFloat64", False):
        config.available_features.add("Double")
    if features.get("shaderResourceMinLod", False):
        config.available_features.add("LodClamp")
    if features.get("runtimeDescriptorArray", False):
        config.available_features.add("VulkanRuntimeDescriptorArray")
    if features.get("descriptorBindingPartiallyBound", False):
        config.available_features.add("VulkanDescriptorBindingPartiallyBound")
    if (features.get("multiViewport", False) and
            features.get("shaderOutputViewportIndex", False)):
        config.available_features.add("multi-viewport")
    if "VK_EXT_mesh_shader" in config.available_features:
        config.available_features.add("MeshShader")
    config.available_features.add("WaveMultiPrefix")
    config.available_features.add("WaveSize_32")
    config.available_features.update(f"SM_6_{minor}" for minor in range(10))
else:
    if features.get("Native16BitShaderOpsSupported", False):
        config.available_features.update(["Half", "Int16"])
    if features.get("Int64ShaderOps", False):
        config.available_features.add("Int64")
    if features.get("DoublePrecisionFloatShaderOps", False):
        config.available_features.add("Double")
    if features.get("WaveOps", False):
        config.available_features.add("WaveMultiPrefix")
        if features.get("WaveLaneCountMin", 0) <= 32 <= features.get("WaveLaneCountMax", 0):
            config.available_features.add("WaveSize_32")
    if features.get("ResourceBindingTier", 0) >= 3:
        config.available_features.add("ResourceBindingTier3")
    if features.get("TiledResourcesTier", 0) >= 2:
        config.available_features.add("LodClamp")
    if features.get("MeshShaderTier", "NotSupported") != "NotSupported":
        config.available_features.add("MeshShader")
    if features.get("PSSpecifiedStencilRefSupported", False):
        config.available_features.add("ShaderStencilRef")
    if features.get("VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation", False):
        config.available_features.add("multi-viewport")
    if features.get("RaytracingTier", "NotSupported") != "NotSupported":
        config.available_features.update(["acceleration-structure", "raytracing-pipeline"])
    # SM_6_6 is withheld until 64-bit buffer atomics are supported on Metal.
    sm = str(features.get("HighestShaderModel", "6.0"))
    if sm >= "6.8":
        config.available_features.add("SM_6_8")
    if sm >= "6.9":
        config.available_features.add("SM_6_9")
config.available_features.add("Float32GroupSharedAtomics")

# Substitution order matters: %dxc_target_lib before %dxc_target.
if lit_config.params.get("spirv_frontend"):
    dxc_cmd = f"{config.dxc} -spirv -fspv-target-env=vulkan1.3"
    config.substitutions.append(("%dxc_target_lib", dxc_cmd))
    config.substitutions.append(("%dxc_target", dxc_cmd))
else:
    config.substitutions.append(("%dxc_target_lib", config.dxc))
    config.substitutions.append(("%dxc_target", config.dxc))
offload_api = "vk" if lit_config.params.get("spirv_frontend") else "dx"
def _quote_cmd(path):
    return f'"{path}"' if os.name == "nt" else shlex.quote(path)

offloader = f"{_quote_cmd(config.crucible_offload)} -api={offload_api}"
if lit_config.params.get("vulkan_driver"):
    driver = _quote_cmd(lit_config.params["vulkan_driver"])
    offloader = f"env CRUCIBLE_OFFLOAD_DRIVER={driver} {offloader}"
config.substitutions.append(("%offloader", offloader))
config.substitutions.append(("%api_query", _quote_cmd(config.crucible_api_query)))
exe_suffix = ".exe" if os.name == "nt" else ""
for tool in ["FileCheck", "split-file", "obj2yaml", "not"]:
    config.substitutions.append(
        (rf"(?<!\S){tool}(?=\s|$)", os.path.join(config.llvm_tools_dir, tool + exe_suffix)))

if getattr(config, "goldenimage_dir", "") and os.path.exists(config.goldenimage_dir) and getattr(config, "crucible_imgdiff", ""):
    config.substitutions.append(("%goldenimage_dir", config.goldenimage_dir))
    config.substitutions.append((r"(?<!\S)imgdiff(?=\s|$)", config.crucible_imgdiff))
    config.available_features.add("goldenimage")

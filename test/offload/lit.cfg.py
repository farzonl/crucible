# -*- Python -*-
# Lit configuration running llvm/offload-test-suite's test tree against
# crucible-offload (DXIL -> crucible -> MSL -> Metal, direct bindings).

import os
import shlex
import subprocess

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

# Propagate explicit runtime selection (metal|vulkan) to both api-query and
# offloader child processes.
if "CRUCIBLE_OFFLOAD_RUNTIME" in os.environ:
    config.environment["CRUCIBLE_OFFLOAD_RUNTIME"] = os.environ["CRUCIBLE_OFFLOAD_RUNTIME"]

try:
    report = yaml.safe_load(subprocess.check_output(
        [config.crucible_api_query], text=True
    ))
except (OSError, subprocess.CalledProcessError, yaml.YAMLError) as error:
    lit_config.fatal(f"Crucible device discovery failed: {error}")
targets = [
    device for device in report["Devices"]
    if device["API"] == "DirectX" and device["Driver"] == "Crucible"
]
if len(targets) != 1:
    lit_config.fatal(f"Expected one DirectX/Crucible adapter, found {len(targets)}")
target = targets[0]
host = report["Host"]
# Use the DirectX contract for test annotations, not the Metal backend.
config.available_features.update(
    [target["API"], target["Driver"], "DXC", host["Operating System"]]
)
if host["Architecture"].lower() in ("arm64", "aarch64"):
    config.available_features.add("host-arm64")
features = target["Features"]
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
config.available_features.add("Float32GroupSharedAtomics")
# SM_6_6 is withheld until 64-bit buffer atomics are supported on Metal.
sm = str(features.get("HighestShaderModel", "6.0"))
if sm >= "6.8":
    config.available_features.add("SM_6_8")
if sm >= "6.9":
    config.available_features.add("SM_6_9")

# Substitution order matters: %dxc_target_lib before %dxc_target.
if lit_config.params.get("spirv_frontend"):
    dxc_cmd = (
        f"{config.dxc} -spirv -fspv-target-env=vulkan1.3 "
        "-fvk-t-shift 10000 all -fvk-u-shift 20000 all -fvk-s-shift 30000 all"
    )
    config.substitutions.append(("%dxc_target_lib", dxc_cmd))
    config.substitutions.append(("%dxc_target", dxc_cmd))
else:
    config.substitutions.append(("%dxc_target_lib", config.dxc))
    config.substitutions.append(("%dxc_target", config.dxc))
config.substitutions.append(("%offloader", f"{shlex.quote(config.crucible_offload)} -api=dx"))
config.substitutions.append(("%api_query", shlex.quote(config.crucible_api_query)))
for tool in ["FileCheck", "split-file", "obj2yaml", "not"]:
    config.substitutions.append(
        (rf"(?<!\S){tool}(?=\s|$)", os.path.join(config.llvm_tools_dir, tool)))

if getattr(config, "goldenimage_dir", "") and os.path.exists(config.goldenimage_dir) and getattr(config, "crucible_imgdiff", ""):
    config.substitutions.append(("%goldenimage_dir", config.goldenimage_dir))
    config.substitutions.append((r"(?<!\S)imgdiff(?=\s|$)", config.crucible_imgdiff))
    config.available_features.add("goldenimage")

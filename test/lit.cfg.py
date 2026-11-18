# -*- Python -*-
# Lit configuration for the crucible test suite.

import os
import platform
import shutil

import lit.formats

config.name = "crucible"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".test", ".hlsl", ".ll"]
config.test_source_root = os.path.dirname(__file__)
config.excludes = ["Inputs"]

# Tool substitutions. Paths are provided by lit.site.cfg.py.
config.substitutions.append(("%dxc", config.dxc))
config.substitutions.append(("%cruciblec", config.cruciblec))
config.substitutions.append(("FileCheck", config.filecheck))

if config.crucible_run:
    config.substitutions.append(("%crucible-run", config.crucible_run))

# Feature flags:
#   metal-tools -- Apple's `xcrun metal` toolchain is available, so emitted
#                  MSL can be compiled with -Werror.
#   gpu         -- a Metal device is available for end-to-end execution.
if platform.system() == "Darwin" and shutil.which("xcrun"):
    if os.system("xcrun --find metal > /dev/null 2>&1") == 0:
        config.available_features.add("metal-tools")
    if config.crucible_run and os.path.exists(config.crucible_run):
        config.available_features.add("gpu")

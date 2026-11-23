#!/usr/bin/env python3
"""Install the fail-closed compiler hook in the pinned CTS-only MoltenVK source."""

import argparse
import shutil
from pathlib import Path

REVISION = "db66022459ffb663aa2b50f6b018bc2e124f5edf"


def prepare(source: Path) -> None:
    target = source / "MoltenVK/MoltenVK/GPUObjects/MVKShaderModule.mm"
    text = target.read_text(encoding="utf-8")
    include = '#include "MVKShaderModule.h"\n'
    call = ("bool wasConverted = _spvConverter.convert(*pShaderConfig, conversionResult, "
            "shouldLogCode, shouldLogCode, shouldLogEstimatedGLSL);")
    if ("cts_moltenvk_hook.h" in text or
            text.count(include) != 1 or text.count(call) != 1):
        raise ValueError("Pinned MoltenVK shader-module integration points changed")
    text = text.replace(include, include + '#include "cts_moltenvk_hook.h"\n')
    text = text.replace(call, call + """
    if (std::getenv("CRUCIBLE_CTS_COMPILER")) {
        if (wasConverted) {
            wasConverted = crucibleCTSCompile(_spvConverter.getSPIRV(), *pShaderConfig, conversionResult);
        } else {
            conversionResult.msl.clear();
            if (!crucibleCTSTrace("metadata_failed")) {
                conversionResult.resultLog += "\\nCannot write compiler attribution trace\\n";
            }
        }
    }""")
    shutil.copyfile(Path(__file__).with_name("cts_moltenvk_hook.h"),
                    target.with_name("cts_moltenvk_hook.h"))
    target.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    prepare(parser.parse_args().source)

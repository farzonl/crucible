#!/usr/bin/env python3
"""Correct four redundant template qualifiers in CTS 1.3.9 for modern Clang."""

import argparse
from pathlib import Path


def prepare(source: Path) -> None:
    target = source / (
        "external/vulkancts/modules/vulkan/shaderexecutor/"
        "vktShaderBuiltinPrecisionTests.cpp")
    text = target.read_text(encoding="utf-8")
    replacements = {
        "FloatTypeSelector<GenType>::type::template convert(": 3,
        "SamplingT::SelectorT::template conv(": 1,
    }
    for original, count in replacements.items():
        if text.count(original) != count:
            raise ValueError("Pinned CTS Clang compatibility patch no longer matches")
        text = text.replace(original, original.replace("template ", ""))
    target.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    prepare(parser.parse_args().source)

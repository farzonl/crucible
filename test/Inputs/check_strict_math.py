from __future__ import annotations

import math
import struct
import sys


def decode(bits: int, width: int) -> float:
    return struct.unpack("<e" if width == 16 else "<f", bits.to_bytes(width // 8, "little"))[0]


def encode(value: float, width: int) -> int:
    try:
        packed = struct.pack("<e" if width == 16 else "<f", value)
    except OverflowError:
        packed = struct.pack("<e" if width == 16 else "<f", math.copysign(math.inf, value))
    return int.from_bytes(packed, "little")


def reference(operation: str, a: float, b: float, width: int = 32) -> float:
    if operation in ("Determinant", "Reflect", "ReflectY"):
        def rounded(value: float) -> float:
            return decode(encode(value, width), width)

        if operation == "Determinant":
            return rounded(rounded(a * a) - rounded(b * b))
        if operation == "ReflectY":
            dot = rounded(rounded(0.0 * a) + rounded(-1.0 * b))
            return rounded(b - rounded(rounded(2.0 * dot) * -1.0))
        dot = rounded(b * a)
        return rounded(a - rounded(rounded(2.0 * dot) * b))
    if operation == "Atan2":
        return math.atan2(a, b)
    if math.isnan(a):
        return a
    if operation == "Acosh":
        return math.nan if a < 1 else math.acosh(a)
    if operation == "Log2":
        if a == 0:
            return -math.inf
        return math.nan if a < 0 else math.log2(a)
    if operation not in ("Exp", "Cosh"):
        raise ValueError(f"unknown operation: {operation}")
    try:
        return math.exp(a) if operation == "Exp" else math.cosh(a)
    except OverflowError:
        return math.inf


def check_value(actual: int, expected: float, width: int, tolerance: int) -> None:
    sign = 1 << (width - 1)
    mask = sign - 1
    infinity = 0x7C00 if width == 16 else 0x7F800000
    assert 0 <= actual < sign * 2, (actual, width)
    if math.isnan(expected):
        assert actual & mask > infinity, (actual, expected, width)
        return
    rounded = encode(expected, width)
    if rounded & mask in (0, infinity):
        assert actual == rounded, (actual, rounded, width)
        return
    # No DenormPreserve mode is requested; Metal may flush subnormal results.
    minimum_normal = 0x0400 if width == 16 else 0x00800000
    if 0 < rounded & mask < minimum_normal and actual == rounded & sign:
        return
    assert actual & mask < infinity, (actual, rounded, width)

    def ordered(bits: int) -> int:
        return (~bits & (sign * 2 - 1)) if bits & sign else bits | sign

    distance = abs(ordered(actual) - ordered(rounded))
    assert distance <= tolerance, (actual, rounded, width, distance, tolerance)


def check(operation: str, a_bits: list[int], b_bits: list[int], values: list[int]) -> None:
    assert len(a_bits) == len(b_bits) and len(values) == len(a_bits) * 10
    for i, (ab, bb) in enumerate(zip(a_bits, b_bits)):
        a, b = decode(ab, 32), decode(bb, 32)
        ah, bh = decode(ab & 0xFFFF, 16), decode(bb & 0xFFFF, 16)
        nan = math.isnan(a) or math.isnan(b)
        hnan = math.isnan(ah) or math.isnan(bh)
        v = values[i * 10 : (i + 1) * 10]
        assert [v[j] for j in (0, 1, 2, 3, 5, 6, 8)] == [
            int(a == b), int(a == b or nan), int(a != b and not nan),
            int(a != b or nan), int(ah == bh), int(ah != bh or hnan), int(a == b),
        ], (i, v)
        if operation in ("Determinant", "Reflect"):
            tolerance = 0
        else:
            tolerance = 6 if operation == "Atan2" else 4
        check_value(v[4], reference(operation, a, b), 32, tolerance)
        check_value(
            v[7], reference(operation, ah, bh, 16), 16,
            0 if tolerance == 0 else 1,
        )
        if operation == "Determinant":
            vector_result = reference(operation, a, b)
        elif operation == "Reflect":
            vector_result = reference("ReflectY", a, b)
        else:
            vector_result = reference(operation, b, a)
        check_value(v[9], vector_result, 32, tolerance)


if __name__ == "__main__":
    buffers: dict[str, list[int]] = {}
    for line in sys.stdin:
        if line.startswith(("buffer0:", "buffer1:", "buffer2:")):
            name, _, data = line.partition(":")
            assert name not in buffers, name
            buffers[name] = [int(value) for value in data.split()]
    assert set(buffers) == {"buffer0", "buffer1", "buffer2"}, buffers.keys()
    check(sys.argv[1], buffers["buffer0"], buffers["buffer1"], buffers["buffer2"])

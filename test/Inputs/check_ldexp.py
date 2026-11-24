import math
import sys

from check_strict_math import check_value, decode


def reference(value: float, exponent: int) -> float:
    if value == 0 or not math.isfinite(value):
        return value
    # Bound the host double's range, not the emitter's float32 clamp.
    if exponent > 1024:
        return math.copysign(math.inf, value)
    if exponent < -1075:
        return math.copysign(0.0, value)
    try:
        return math.ldexp(value, exponent)
    except OverflowError:
        return math.copysign(math.inf, value)


def verify(inputs: list[int], words: list[int], results: list[int]) -> None:
    assert len(words) == len(inputs) * 2 and len(results) == len(inputs) * 6
    for i, bits in enumerate(inputs):
        low, high = words[2 * i : 2 * i + 2]
        raw = low | (high << 32)
        exponent64 = raw if raw < 1 << 63 else raw - (1 << 64)
        exponent32 = low if low < 1 << 31 else low - (1 << 32)
        value = decode(bits, 32)
        expected = [
            reference(value, exponent64), reference(value, exponent64),
            reference(value, -2), reference(value, exponent32),
            reference(value, exponent32), reference(value, -2),
        ]
        for actual, wanted in zip(results[i * 6 : (i + 1) * 6], expected):
            check_value(actual, wanted, 32, 0)


if __name__ == "__main__":
    buffers: dict[str, list[int]] = {}
    for line in sys.stdin:
        if line.startswith(("buffer0:", "buffer1:", "buffer2:")):
            name, _, data = line.partition(":")
            assert name not in buffers, name
            buffers[name] = [int(value) for value in data.split()]
    assert set(buffers) == {"buffer0", "buffer1", "buffer2"}, buffers.keys()
    verify(buffers["buffer0"], buffers["buffer1"], buffers["buffer2"])

RWStructuredBuffer<uint> Out : register(u0);

cbuffer Params : register(b0) {
  half scale;
  int16_t divisor;
  int64_t base;
};

[numthreads(8, 1, 1)]
void main(uint lane : SV_GroupIndex) {
  int16_t value = int16_t(lane) * divisor;
  int64_t wide = base + int64_t(value / int16_t(2));
  uint result = uint(int(wide >> 32) + int(float(half(lane) * scale)));
  if ((lane & 1u) == 0u) {
    uint64_t x = (uint64_t(1) << 32u) | uint64_t(lane + 1u);
    uint64_t sum = WaveActiveSum(x);
    uint64_t prefix = WavePrefixSum(x);
    bool first = WaveReadLaneAt(lane < 2u, 0);
    result += uint(sum >> 32u) + uint(sum) + uint(prefix) +
              WaveActiveCountBits(lane < 4u) + uint(first);
  }
  Out[lane] = result;
}

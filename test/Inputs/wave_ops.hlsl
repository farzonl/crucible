// Wave intrinsics mapped to Metal simdgroup functions.
RWStructuredBuffer<uint> Out : register(u0);

[numthreads(32, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  uint total = WaveActiveSum(id.x);     // 496 across one 32-lane wave
  uint first = WaveReadLaneFirst(id.x); // 0
  uint prefix = WavePrefixSum(1u);      // exclusive sum of 1 == lane index
  Out[id.x] = total + first + prefix;
}

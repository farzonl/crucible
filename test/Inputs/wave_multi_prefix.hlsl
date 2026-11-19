// SM 6.5 wave multi-prefix intrinsics mapped to Metal simdgroup helpers.
RWStructuredBuffer<uint> Out : register(u0);

[numthreads(32, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  uint lane = WaveGetLaneIndex();
  uint4 mask = (lane & 1u) == 0u ? uint4(0x55555555u, 0u, 0u, 0u)
                                 : uint4(0xAAAAAAAAu, 0u, 0u, 0u);
  uint s = WaveMultiPrefixSum(lane + 1u, mask);
  uint p = WaveMultiPrefixProduct((lane & 1u) + 2u, mask);
  uint c = WaveMultiPrefixCountBits((lane & 2u) == 0u, mask);
  uint a = WaveMultiPrefixBitAnd(0xFu, mask);
  uint o = WaveMultiPrefixBitOr(1u << (lane & 3u), mask);
  uint x = WaveMultiPrefixBitXor(1u << (lane & 1u), mask);
  Out[id.x] = s + p + c + (a & 0xFFu) + o + x;
}

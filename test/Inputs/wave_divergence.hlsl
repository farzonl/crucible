RWStructuredBuffer<uint> Out : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_GroupThreadID) {
  for (uint i = 0; i < 64; ++i) {
    if (i == tid.x) {
      Out[tid.x] = WaveActiveCountBits(true);
      break;
    }
  }
}

[numthreads(64, 1, 1)]
void collectives(uint3 tid : SV_GroupThreadID) {
  for (uint i = 0; i < 64; ++i) {
    if (i == tid.x) {
      uint4 ballot = WaveActiveBallot(true);
      uint4 mask = uint4(~0u, ~0u, ~0u, ~0u);
      Out[tid.x] = uint(WaveIsFirstLane()) +
                  uint(WaveActiveAnyTrue(true)) +
                  uint(WaveActiveAllTrue(true)) +
                  countbits(ballot.x) + countbits(ballot.y) +
                  countbits(ballot.z) + countbits(ballot.w) +
                  WaveMultiPrefixSum(1u, mask);
      break;
    }
  }
}

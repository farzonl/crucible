// Groupshared memory + barriers + shared atomics: threadgroup reduction.
RWStructuredBuffer<uint> Out : register(u0);
groupshared uint gsSum[1];

[numthreads(64, 1, 1)]
void main(uint gidx : SV_GroupIndex) {
  if (gidx == 0)
    gsSum[0] = 0;
  GroupMemoryBarrierWithGroupSync();
  InterlockedAdd(gsSum[0], gidx);
  GroupMemoryBarrierWithGroupSync();
  if (gidx == 0)
    Out[0] = gsSum[0];
}

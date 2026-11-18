// Barrier-heavy: groupshared tree reduction + global atomic accumulate.
StructuredBuffer<float> In : register(t0);
RWStructuredBuffer<uint> Out : register(u0);
groupshared float tile[256];
[RootSignature("SRV(t0),UAV(u0)")]
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
  tile[gi] = In[id.x];
  GroupMemoryBarrierWithGroupSync();
  for (uint s = 128u; s > 0u; s >>= 1u) {
    if (gi < s)
      tile[gi] += tile[gi + s];
    GroupMemoryBarrierWithGroupSync();
  }
  if (gi == 0)
    InterlockedAdd(Out[0], uint(tile[0]));
}

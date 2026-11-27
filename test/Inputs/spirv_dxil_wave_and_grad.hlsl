// Exercises DXIL wave lowerings (#183):
// - crucible.wave_multi_prefix (WaveMultiPrefixSum)
// - gpu.shuffle (WaveReadLaneAt)
StructuredBuffer<uint> InVal : register(t0);
StructuredBuffer<uint4> InMask : register(t1);
RWStructuredBuffer<uint> OutU : register(u0);

[numthreads(4, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint v = InVal[tid.x];
  uint4 mask = InMask[tid.x];
  uint s = WaveMultiPrefixSum(v, mask);
  uint r = WaveReadLaneAt(v, 0);
  OutU[tid.x] = s + r;
}

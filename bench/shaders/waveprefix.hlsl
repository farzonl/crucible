// Simdgroup-bound: wave prefix + reductions.
RWStructuredBuffer<uint> Out : register(u0);
[RootSignature("UAV(u0)")]
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  uint v = Out[id.x];
  uint p = WavePrefixSum(v);
  uint s = WaveActiveSum(v);
  uint m = WaveActiveMax(v);
  Out[id.x] = p + s + m;
}

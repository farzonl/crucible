// Memory-bound streaming FMA.
RWStructuredBuffer<float> Out : register(u0);
[RootSignature("UAV(u0)")]
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  float v = Out[id.x];
  Out[id.x] = mad(v, 1.000001f, 0.5f);
}

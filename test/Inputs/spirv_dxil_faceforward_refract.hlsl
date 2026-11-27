RWStructuredBuffer<float3> Out : register(u0);
StructuredBuffer<float3> InA : register(t0);
StructuredBuffer<float3> InB : register(t1);
StructuredBuffer<float3> InC : register(t2);

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  float3 ff = faceforward(InA[tid.x], InB[tid.x], InC[tid.x]);
  float3 rf = refract(InA[tid.x], InB[tid.x], 0.75f);
  Out[tid.x] = ff + rf;
}

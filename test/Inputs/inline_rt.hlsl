RaytracingAccelerationStructure Scene : register(t0);
RWStructuredBuffer<uint> Output : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  RayDesc Ray;
  Ray.Origin = float3(0, 0, 1);
  Ray.Direction = float3(0, 0, -1);
  Ray.TMin = 0.0;
  Ray.TMax = 100.0;
  RayQuery<RAY_FLAG_NONE> Q;
  Q.TraceRayInline(Scene, RAY_FLAG_NONE, 0xFF, Ray);
  Q.Proceed();
  Output[tid.x] = Q.CommittedStatus() == COMMITTED_TRIANGLE_HIT
                      ? Q.CommittedInstanceID() + Q.CommittedPrimitiveIndex()
                      : 0xFFFFFFFF;
}

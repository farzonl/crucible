struct Payload {
  uint Value;
};

RaytracingAccelerationStructure Scene : register(t0);
RWStructuredBuffer<uint> Output : register(u0);

[shader("raygeneration")]
void RayGen() {
  Payload P;
  P.Value = 0;
  RayDesc Ray;
  Ray.Origin = float3(0, 0, 1);
  Ray.Direction = float3(0, 0, -1);
  Ray.TMin = 0.0;
  Ray.TMax = 100.0;
  TraceRay(Scene, RAY_FLAG_NONE, 0xFF, 0, 1, 0, Ray, P);
  Output[DispatchRaysIndex().x] = P.Value + DispatchRaysDimensions().x;
}

[shader("miss")]
void MissMain(inout Payload P) {
  P.Value = 0xDEAD;
}

[shader("closesthit")]
void ClosestHitMain(inout Payload P,
                    in BuiltInTriangleIntersectionAttributes Attr) {
  P.Value = PrimitiveIndex() + uint(RayTCurrent() + WorldRayOrigin().z + WorldRayDirection().z);
}

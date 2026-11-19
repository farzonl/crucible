struct ControlPoint {
  float4 position : POSITION;
};

struct PatchConstants {
#ifdef QUAD
  float edges[4] : SV_TessFactor;
  float inside[2] : SV_InsideTessFactor;
#else
  float edges[3] : SV_TessFactor;
  float inside : SV_InsideTessFactor;
#endif
  uint label : LABEL;
};

struct DomainOutput {
  float4 position : SV_Position;
  nointerpolation uint label : LABEL;
};

#ifdef QUAD
[domain("quad")]
DomainOutput main(PatchConstants constants, float2 uv : SV_DomainLocation,
                  const OutputPatch<ControlPoint, 4> points,
                  uint patch : SV_PrimitiveID) {
  float factor = constants.inside[1];
#else
[domain("tri")]
DomainOutput main(PatchConstants constants, float3 uv : SV_DomainLocation,
                  const OutputPatch<ControlPoint, 3> points,
                  uint patch : SV_PrimitiveID) {
  float factor = constants.inside;
#endif
  DomainOutput output;
  output.position = lerp(points[0].position, points[1].position, uv.y) * factor;
  output.label = constants.label + patch;
  return output;
}

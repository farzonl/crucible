struct Varyings {
  float4 position : SV_Position;
  float2 values[2] : TEXCOORD3;
};

struct VertexOutput {
  Varyings varyings;
  float2 clip : SV_ClipDistance0;
  float clipExtra : SV_ClipDistance1;
};

VertexOutput vs(float4 position : POSITION) {
  VertexOutput output;
  output.varyings.position = position;
  output.varyings.values[0] = position.xy;
  output.varyings.values[1] = position.zw;
  output.clip = position.xy;
  output.clipExtra = position.z;
  return output;
}

#ifndef DEPTH_SEMANTIC
#define DEPTH_SEMANTIC SV_Depth
#endif

struct PixelOutput {
  float4 color : SV_Target;
  float depth : DEPTH_SEMANTIC;
};

PixelOutput ps(centroid noperspective float4 position : SV_Position,
               float2 values[2] : texcoord3,
               uint primitive : SV_PrimitiveID) {
  PixelOutput output;
  output.color = float4(values[0], values[1] + float(primitive));
  output.depth = position.z;
  return output;
}

struct VSOutput {
  float4 Pos : SV_POSITION;
  float4 Color : COLOR;
};

VSOutput mainVS(float4 Pos : POSITION, float4 Color : COLOR) {
  VSOutput O;
  O.Pos = Pos;
  O.Color = Color;
  return O;
}

VSOutput mainVSId(uint Vid : SV_VertexID) {
  VSOutput O;
  float2 P = float2((Vid == 1) ? 1.0 : -1.0, (Vid == 2) ? 1.0 : -1.0);
  O.Pos = float4(P, 0.0, 1.0);
  O.Color = float4(1.0, 0.0, 0.0, 1.0);
  return O;
}

float4 mainPS(VSOutput I) : SV_TARGET {
  return I.Color;
}

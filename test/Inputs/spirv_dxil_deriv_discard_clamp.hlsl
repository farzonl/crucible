[[vk::binding(0, 0)]] Texture2D<float4> Tex2D : register(t0);
[[vk::binding(1, 0)]] SamplerState Samp : register(s0);

float4 mainDerivDiscard(float4 Pos : SV_POSITION) : SV_TARGET {
  if (Pos.x < 1.0)
    discard;
  float dx = ddx_fine(Pos.x);
  float dy = ddy_coarse(Pos.y);
  float fw = fwidth(Pos.x + Pos.y);
  return float4(dx, dy, fw, 1.0);
}

float4 mainSampleClamp(float4 Pos : SV_POSITION) : SV_TARGET {
  return Tex2D.SampleBias(Samp, Pos.xy * 0.01, -0.5, int2(0, 0), 1.0);
}

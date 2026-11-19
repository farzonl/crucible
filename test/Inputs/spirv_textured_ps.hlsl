Texture2D<float4> Albedo : register(t0);
SamplerState LinearSamp : register(s0);

float4 ps(float2 uv : TEXCOORD0) : SV_Target {
  return Albedo.Sample(LinearSamp, uv);
}

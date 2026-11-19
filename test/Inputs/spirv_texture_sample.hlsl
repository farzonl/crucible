Texture2D<float4> Src : register(t0);
SamplerState Samp : register(s0);
RWTexture2D<float4> Dst : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
  uint w, h, levels;
  Src.GetDimensions(0, w, h, levels);
  float2 uv = float2(dtid.xy) / float2(float(w), float(h));
  float4 sampled = Src.SampleLevel(Samp, uv, 0.0f);
  float4 loaded = Src.Load(int3(dtid.xy, 0));
  Dst[dtid.xy] = sampled + loaded;
}

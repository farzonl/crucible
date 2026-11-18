Texture2D<float4> InTex : register(t0);
RWTexture2D<float4> OutTex : register(u0);
RWBuffer<uint> Dims : register(u1);

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
  uint w, h, mips;
  InTex.GetDimensions(0, w, h, mips);
  if (dtid.x == 0 && dtid.y == 0) {
    Dims[0] = w;
    Dims[1] = h;
    Dims[2] = mips;
  }
  float4 c = InTex.Load(int3(dtid.xy, 0), int2(1, 0));
  OutTex[dtid.xy] = c.bgra;
}

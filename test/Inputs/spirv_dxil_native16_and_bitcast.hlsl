RWStructuredBuffer<float16_t> Out16 : register(u0);
RWTexture2D<float> OutTex : register(u1);
RWStructuredBuffer<uint2> OutU2 : register(u2);
StructuredBuffer<double> InD : register(t0);

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  float16_t h = Out16[tid.x];
  int16_t bits = asint16(h);
  Out16[tid.x] = exp(asfloat16(bits));
  OutTex[uint2(0, 0)] = f16tof32((uint)(uint16_t)bits);
  uint lo, hi;
  asuint(InD[tid.x], lo, hi);
  OutU2[tid.x] = uint2(lo, hi);
}

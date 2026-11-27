// Middle-end capability-driven math legalization test:
// exercises atan2, firstbithigh/firstbitlow (ctlz/cttz), and fma/mad.
RWStructuredBuffer<float> OutF : register(u0);
RWStructuredBuffer<uint> OutU : register(u1);
StructuredBuffer<float> InA : register(t0);
StructuredBuffer<float> InB : register(t1);
StructuredBuffer<uint> InU : register(t2);

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  float a = InA[i];
  float b = InB[i];
  float angle = atan2(a, b);
  float m = mad(a, b, angle);
  OutF[i] = m;

  uint u = InU[i];
  OutU[i] = firstbithigh(u) + firstbitlow(u);
}

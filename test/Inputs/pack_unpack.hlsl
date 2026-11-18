StructuredBuffer<uint> InU : register(t0);
StructuredBuffer<int4> InI4 : register(t1);
RWStructuredBuffer<uint> OutU : register(u0);
RWStructuredBuffer<int4> OutI4 : register(u1);

[numthreads(1, 1, 1)]
void main() {
  uint pk = InU[0];
  int4 u = unpack_s8s32(pk);
  OutI4[0] = u;
  OutU[0] = pack_clamp_u8(InI4[0]);
  OutU[1] = pack_clamp_s8(InI4[0]);
  OutU[2] = pack_u8(uint4(u));
}

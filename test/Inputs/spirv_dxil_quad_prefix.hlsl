// Exercises SPIR-V -> DXIL/MSL quad swap, wave prefix scan, and WaveIsFirstLane:
// - spirv.GroupNonUniformQuadSwap -> crucible.quad_op (#71)
// - gpu.subgroup_reduce with crucible.scan_kind="exclusive" -> dx.op.wavePrefixOp (#186)
// - WaveIsFirstLane via exclusive scan (#186)
StructuredBuffer<uint> In : register(t0);
RWStructuredBuffer<uint> Out : register(u0);

[numthreads(4, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint v = In[tid.x];
  uint ps = WavePrefixSum(v);
  uint pp = WavePrefixProduct(v);
  uint qx = QuadReadAcrossX(v);
  uint qy = QuadReadAcrossY(v);
  uint qd = QuadReadAcrossDiagonal(v);
  uint fl = WaveIsFirstLane() ? 1u : 0u;
  Out[tid.x] = ps + pp + qx + qy + qd + fl;
}

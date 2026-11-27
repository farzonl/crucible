// Exercises DXIL compliance fixes:
// - TypedBuffer UAV store without undef lanes (#180)
// - 16-bit StructuredBuffer without i8 array crash/OOM (#185)
// - half FP ops without !dx.precise metadata (#181)
RWBuffer<uint> TypedOut : register(u0);
RWStructuredBuffer<float16_t> HalfOut : register(u1);
StructuredBuffer<float16_t> HalfIn : register(t0);

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  TypedOut[tid.x] = tid.x + 1u;
  HalfOut[tid.x] = abs(HalfIn[tid.x]);
}

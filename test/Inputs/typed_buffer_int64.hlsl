Buffer<int64_t> ScalarIn : register(t0);
Buffer<uint64_t2> VectorIn : register(t1);
RWBuffer<int64_t> ScalarOut : register(u0);
RWBuffer<uint64_t2> VectorOut : register(u1);

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  ScalarOut[id.x] = ScalarIn[id.x];
  VectorOut[id.x] = VectorIn[id.x];
}

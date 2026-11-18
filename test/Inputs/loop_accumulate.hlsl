// Loop with a nested condition and continue; dxc rotates this into a
// zero-trip-guarded do-while.
RWStructuredBuffer<float> Data : register(u0);

[numthreads(8, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  float acc = 0.0f;
  for (uint i = 0; i < id.x; ++i) {
    if (Data[i] > 4.0f) continue;
    acc += Data[i];
  }
  Data[id.x] = acc;
}

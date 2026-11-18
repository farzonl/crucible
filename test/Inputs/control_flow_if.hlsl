// Real divergent branch: dxc cannot speculate the dx.op min call, so this
// produces an actual CFG diamond (not a select).
RWStructuredBuffer<float> Data : register(u0);

[numthreads(8, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  float x = Data[id.x];
  Data[id.x] = (x < 2.0f) ? x * 3.0f : min(x, 10.0f);
}

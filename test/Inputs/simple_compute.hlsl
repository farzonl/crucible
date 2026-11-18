// Milestone 1 reference shader: RWStructuredBuffer multiply-add.
RWStructuredBuffer<float> Out : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  Out[id.x] = Out[id.x] * 2.0f + 1.0f;
}

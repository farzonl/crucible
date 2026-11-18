// Constant buffer with mixed float/uint fields.
cbuffer Params : register(b0) {
  float scale;
  float bias;
  uint count;
};
RWStructuredBuffer<float> Data : register(u0);

[numthreads(8, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  if (id.x < count)
    Data[id.x] = Data[id.x] * scale + bias;
}

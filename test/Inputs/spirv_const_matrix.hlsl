// Exercises constant matrix (OpConstantComposite %mat2v2float) lowering in
// SPIRVLoader for both DXIL and Metal backends (#185).
StructuredBuffer<float2> In : register(t0);
RWStructuredBuffer<float2> Out : register(u0);

[numthreads(1, 1, 1)]
void main() {
  float2x2 m = float2x2(1.25, 2.9, 3.5, 4.1) * In[0].x;
  Out[0] = m[0];
}

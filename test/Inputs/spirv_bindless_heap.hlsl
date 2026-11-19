StructuredBuffer<float> UnboundedInputs[] : register(t0, space0);
RWStructuredBuffer<float> Out : register(u0, space1);

[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
  StructuredBuffer<float> heapBuf = ResourceDescriptorHeap[NonUniformResourceIndex(dtid.x)];
  float a = heapBuf[0];
  float b = UnboundedInputs[NonUniformResourceIndex(dtid.x)][0];
  Out[dtid.x] = a + b;
}

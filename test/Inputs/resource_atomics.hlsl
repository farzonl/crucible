RWTexture2D<uint> Unsigned : register(u0);
RWTexture2D<int> Signed : register(u1);
RWByteAddressBuffer Raw : register(u2);
RWStructuredBuffer<uint> Out : register(u3);

[numthreads(1, 1, 1)]
void main(uint2 p : SV_DispatchThreadID) {
  uint previous;
  int signedPrevious;
  InterlockedExchange(Unsigned[p], 17u, previous);
  Out[0] = previous;
  InterlockedCompareExchange(Unsigned[p], 17u, 42u, previous);
  Out[1] = previous;
  InterlockedMin(Signed[p], -7, signedPrevious);
  Out[2] = asuint(signedPrevious);
  Raw.InterlockedXor(4, 3u, previous);
  Out[3] = previous;
}

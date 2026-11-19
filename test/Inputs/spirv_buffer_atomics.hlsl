RWByteAddressBuffer Raw : register(u0);
RWStructuredBuffer<uint> Out : register(u1);

[numthreads(1, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
  uint prev;
  Raw.InterlockedAdd(0, 5u, prev);
  Out[0] = prev;
  Raw.InterlockedXor(4, 3u, prev);
  Out[1] = prev;
  Raw.InterlockedExchange(8, 20u, prev);
  Out[2] = prev;
}

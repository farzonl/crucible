#ifdef __HLSL_ENABLE_16_BIT
StructuredBuffer<uint16_t> In16 : register(t0);
RWStructuredBuffer<uint16_t> Out16 : register(u1);
RWStructuredBuffer<uint> Dimensions : register(u2);
StructuredBuffer<float16_t> InHalf : register(t3);
RWStructuredBuffer<float16_t> OutHalf : register(u4);

[numthreads(1, 1, 1)]
void nativeTail() {
  uint count, stride;
  In16.GetDimensions(count, stride);
  Dimensions[0] = count;
  Dimensions[1] = stride;
  Out16.GetDimensions(count, stride);
  Dimensions[2] = count;
  Dimensions[3] = stride;
  InHalf.GetDimensions(count, stride);
  Dimensions[4] = count;
  Dimensions[5] = stride;
  OutHalf.GetDimensions(count, stride);
  Dimensions[6] = count;
  Dimensions[7] = stride;
  uint status;
  In16.Load(8, status);
  Dimensions[8] = CheckAccessFullyMapped(status);
  In16.Load(9, status);
  Dimensions[9] = CheckAccessFullyMapped(status);
  for (uint i = 0; i < 9; ++i) {
    Out16[i] = In16[i] * uint16_t(17);
    OutHalf[i] = InHalf[i];
  }
}
#endif

StructuredBuffer<int4> Prefix : register(t10);
StructuredBuffer<int4> Empty : register(t11);
RWStructuredBuffer<int> Result : register(u12);
ByteAddressBuffer RawPrefix : register(t13);
Buffer<uint> TypedPrefix : register(t14);

[numthreads(1, 1, 1)]
void residency() {
  uint status;
  Result[0] = Prefix.Load(0, status).x;
  Result[1] = CheckAccessFullyMapped(status);
  Result[2] = Prefix.Load(4095, status).x;
  Result[3] = CheckAccessFullyMapped(status);
  Result[4] = Prefix.Load(4096, status).x;
  Result[5] = CheckAccessFullyMapped(status);
  Result[6] = Empty.Load(0, status).x;
  Result[7] = CheckAccessFullyMapped(status);
  Result[8] = RawPrefix.Load(65532, status);
  Result[9] = CheckAccessFullyMapped(status);
  Result[10] = RawPrefix.Load(65536, status);
  Result[11] = CheckAccessFullyMapped(status);
  Prefix.Load(0x10000000, status);
  Result[12] = CheckAccessFullyMapped(status);
  uint bytes;
  RawPrefix.GetDimensions(bytes);
  Result[13] = bytes;
  Result[14] = TypedPrefix.Load(16383, status);
  Result[15] = CheckAccessFullyMapped(status);
  Result[16] = TypedPrefix.Load(16384, status);
  Result[17] = CheckAccessFullyMapped(status);
}

RWBuffer<uint> Scalar : register(u0);
RWBuffer<uint2> Vector : register(u1);

[numthreads(2, 1, 1)]
void main(uint lane : SV_GroupIndex) {
  Scalar[lane] = lane + 10;
  Vector[lane] = uint2(lane + 20, lane + 30);
}

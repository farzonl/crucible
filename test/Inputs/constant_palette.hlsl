static const float Palette[4] = {0.0, 0.25, 0.5, 1.0};
RWBuffer<float> Output : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  Output[id.x] = Palette[id.x & 3];
}

// Exercises math intrinsics, min/max, selects, and casts.
RWStructuredBuffer<float> Data : register(u0);

[numthreads(8, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  float x = Data[id.x];
  float s = sqrt(abs(x));
  float m = max(s, 0.5f);
  uint bits = id.x;
  float scaled = m * float(bits + 1u);
  float limit = (x < 2.0f) ? 1.0e9f : 10.0f;
  Data[id.x] = min(scaled, limit);
}

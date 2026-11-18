// ALU/loop-bound: mandelbrot escape iteration over a 2048x2048 grid.
RWStructuredBuffer<float> Out : register(u0);
[RootSignature("UAV(u0)")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  uint x = id.x % 2048u;
  uint y = id.x / 2048u;
  float cx = (float(x) / 2048.0f) * 3.5f - 2.5f;
  float cy = (float(y) / 2048.0f) * 2.0f - 1.0f;
  float zx = 0.0f, zy = 0.0f;
  uint i = 0;
  for (; i < 256u; ++i) {
    float nx = zx * zx - zy * zy + cx;
    zy = 2.0f * zx * zy + cy;
    zx = nx;
    if (zx * zx + zy * zy > 4.0f)
      break;
  }
  Out[id.x] = float(i);
}

RWStructuredBuffer<double> Buf : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  double a = Buf[0];
  double b = Buf[1];
  double c = asdouble(1u, 2u);
  double sum = a + b;
  double prod = a * b;
  double fma_val = fma(a, b, c);
  double div_val = a / b;
  double mm = min(abs(a), b) + max(a, b);
  bool lt = a < b;
  float f32 = (float)sum;
  double from_f32 = (double)f32;
  Buf[2] = lt ? fma_val : (prod + div_val + mm + from_f32);
}

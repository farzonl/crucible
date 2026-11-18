StructuredBuffer<int> Input : register(t0);
RWStructuredBuffer<uint> Output : register(u0);

[numthreads(1, 1, 1)]
void main() {
  vector<int, 5> a, b;
  [loop]
  for (uint i = 0; i < 5; ++i) {
    int x = Input[i];
    a[i] = x - 3;
    b[i] = x + 1;
  }
  vector<int, 5> magnitude = abs(a);
  Output[0] = uint(dot(a, b));
  Output[1] = dot(vector<uint, 5>(a + 3), vector<uint, 5>(b));
  Output[2] = uint(2.0 * dot(vector<float, 5>(a) + 3.5,
                            vector<float, 5>(b)));
  [loop]
  for (uint i = 0; i < 5; ++i)
    Output[3 + i] = uint(magnitude[i]);
}

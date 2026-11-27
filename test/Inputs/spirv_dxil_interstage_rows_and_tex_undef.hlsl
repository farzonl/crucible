Texture2DArray<float4> Tex2DArr : register(t0);
RWStructuredBuffer<float4> Out : register(u0);

[numthreads(1, 1, 1)]
void mainLoad(uint3 tid : SV_DispatchThreadID) {
  Out[0] = Tex2DArr.Load(int4(0, 0, 0, 0), int2(1, -1));
}

struct VSOut {
  float4 Pos : SV_POSITION;
  float4 Custom : USER_SEMANTIC;
};

VSOut mainVS(float4 Pos : POSITION) {
  VSOut O;
  O.Pos = Pos;
  O.Custom = float4(0.5, 1.0, 0.0, 1.0);
  return O;
}

float4 mainPS(float4 Custom : USER_SEMANTIC, bool Front : SV_IsFrontFace) : SV_TARGET {
  return Front ? Custom : float4(0.0, 0.0, 0.0, 1.0);
}

struct VSOut {
  float4 Pos : SV_POSITION;
  float Clip : SV_ClipDistance0;
  float Arr[2] : MY_ARR;
  float2x2 Mat : MY_MAT;
};

VSOut mainVS(float4 Pos : POSITION) {
  VSOut O;
  O.Pos = Pos;
  O.Clip = Pos.x;
  O.Arr[0] = 1.0;
  O.Arr[1] = 2.0;
  O.Mat = float2x2(1.0, 2.0, 3.0, 4.0);
  return O;
}

float4 mainPS(VSOut I) : SV_TARGET {
  return float4(I.Arr[0], I.Arr[1], I.Mat[0][0], I.Mat[1][1]);
}

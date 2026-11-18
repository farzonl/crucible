struct Varyings {
  float4 position : SV_Position;
  float2 value : COLOR2;
};

Varyings vs(float2 position : POSITION) {
  Varyings result;
  result.position = float4(position, 0, 1);
  result.value = position * 0.25 + 0.5;
  return result;
}

float4 ps(float2 value : color2, float4 position : SV_Position) : SV_Target {
  return float4(value.x, value.y, ddx_fine(position.x), ddy_coarse(position.y));
}

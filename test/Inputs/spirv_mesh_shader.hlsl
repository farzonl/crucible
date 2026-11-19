struct Vert {
  float4 pos : SV_Position;
};

[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void main(uint gtid : SV_GroupThreadID,
          out vertices Vert verts[3],
          out indices uint3 tris[1]) {
  SetMeshOutputCounts(3, 1);
  verts[0].pos = float4(0.0, 0.5, 0.0, 1.0);
  verts[1].pos = float4(0.5, -0.5, 0.0, 1.0);
  verts[2].pos = float4(-0.5, -0.5, 0.0, 1.0);
  tris[0] = uint3(0, 1, 2);
}

#version 330 core
out vec4 FragColor;
flat in vec3 colours[3];
flat in vec3 vert[3];
in vec3 coord;

uniform vec4 weight;
uniform int mode;
const int constant_color = 0;
const int bilinear_interpolation = 1;
const int step_cutoff = 2;
const int step_smooth = 3;

// uniform sampler1D texture1;
// layout (std140) uniform variables1 {
uniform variables1 {
  float red_color[20000];
  // float green_color[20000];
  // float blue_color[20000];
};

void main()
{
  vec3 color = vec3(0.0, 0.0, 0.0);
  vec3 normal_coor = coord.x * vert[0] + coord.y * vert[1] + coord.z * vert[2];

  if (mode == constant_color)
  {
    // ---------------------------------------------------------------------
    color = colours[0];
    float valuee = red_color[gl_PrimitiveID];
    color = vec3(valuee, 0.0, 0.0);
  }
  else if (mode == bilinear_interpolation)
  {
    // ---------------------------------------------------------------------
    color = coord.x * colours[0] + coord.y * colours[1] + coord.z * colours[2];
  }
  else if (mode == step_cutoff)
  {
    // ---------------------------------------------------------------------
    vec2 dir1 = weight.zw - weight.xy;
    dir1 = vec2(dir1.y, -dir1.x); // rotate 90 degrees
    vec2 dir2 = normal_coor.xy - weight.xy;
    float result = dot(dir1, dir2); // vecors don't need to be normalized -> only need to know the sign

    color = step(0.0f, result) * colours[0];

    if (length(normal_coor.xy - weight.xy) < 0.01) { color = colours[1]; }
    else if (length(normal_coor.xy - weight.zw) < 0.01) { color = colours[2]; }
  }
  else if (mode == step_smooth)
  {
    // ---------------------------------------------------------------------
    vec2 dir1 = weight.zw - weight.xy;
    dir1 = vec2(dir1.y, -dir1.x); // rotate 90 degrees
    vec2 dir2 = normal_coor.xy - weight.xy;
    float result = dot(dir1, dir2); // vecors don't need to be normalized -> only need to know the sign

    color = smoothstep(0.0f, 0.01f, result) * colours[0];

    if (length(normal_coor.xy - weight.xy) < 0.01) { color = colours[1]; }
    else if (length(normal_coor.xy - weight.zw) < 0.01) { color = colours[2]; }
  }
  else
  {
    // ---------------------------------------------------------------------
    vec2 lenn = vec2(weight.x - normal_coor.x, weight.y - normal_coor.y);
    if ((lenn.x * lenn.x) + (lenn.y * lenn.y) < 0.1)
    {
      color = colours[0];
    }
  }

  FragColor = vec4(color, 1.0);
}

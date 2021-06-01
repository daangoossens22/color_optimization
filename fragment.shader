#version 330 core
out vec4 FragColor;
flat in vec3 colours[3];
flat in vec3 vert[3];
in vec3 coord;

uniform vec4 weight;
uniform int mode;
const int constant_color_avg = 0;
const int constant_color_center = 1;
const int bilinear_interpolation_no_opt = 2;
const int bilinear_interpolation_opt = 3;
const int bicubic_interpolation = 4;
const int step_cutoff = 5;
const int step_smooth = 6;
const int testing = 7;

const int triangles_per_side = 52;

uniform variables1 {
  float var1[triangles_per_side * triangles_per_side * 2 * 3];
};
uniform variables2 {
  float var2[triangles_per_side * triangles_per_side * 2 * 3];
};
// layout (std140) uniform variables1 {
// uniform variables1 {
//   float red_color[100*100];
//   float green_color[100*100];
//   float blue_color[100*100];
// };

void main()
{
  vec3 color = vec3(0.0, 0.0, 0.0);
  vec3 normal_coor = coord.x * vert[0] + coord.y * vert[1] + coord.z * vert[2];

  // set output color of pixel according to which coloring mode is selected
  switch (mode)
  {
    case constant_color_avg:
    case constant_color_center:
      color = vec3(var1[gl_PrimitiveID * 3], var1[(gl_PrimitiveID * 3) + 1], var1[(gl_PrimitiveID * 3) + 2]);
      break;

    // ---------------------------------------------------------------------
    case bilinear_interpolation_no_opt:
    case bilinear_interpolation_opt:
      color = coord.x * colours[0] + coord.y * colours[1] + coord.z * colours[2];
      break;

    // ---------------------------------------------------------------------
    case bicubic_interpolation:
      break;
    // ---------------------------------------------------------------------
    case step_cutoff:
      {
        vec2 dir1 = weight.zw - weight.xy;
        dir1 = vec2(dir1.y, -dir1.x); // rotate 90 degrees
        vec2 dir2 = normal_coor.xy - weight.xy;
        float result = dot(dir1, dir2); // vecors don't need to be normalized -> only need to know the sign

        color = step(0.0f, result) * colours[0];

        if (length(normal_coor.xy - weight.xy) < 0.01) { color = colours[1]; }
        else if (length(normal_coor.xy - weight.zw) < 0.01) { color = colours[2]; }
      }
      break;

    // ---------------------------------------------------------------------
    case step_smooth:
      {
        vec2 dir1 = weight.zw - weight.xy;
        dir1 = vec2(dir1.y, -dir1.x); // rotate 90 degrees
        vec2 dir2 = normal_coor.xy - weight.xy;
        float result = dot(dir1, dir2); // vecors don't need to be normalized -> only need to know the sign

        color = smoothstep(0.0f, 0.01f, result) * colours[0];

        if (length(normal_coor.xy - weight.xy) < 0.01) { color = colours[1]; }
        else if (length(normal_coor.xy - weight.zw) < 0.01) { color = colours[2]; }
      }
      break;

    // ---------------------------------------------------------------------
    case testing:
      vec2 lenn = vec2(weight.x - normal_coor.x, weight.y - normal_coor.y);
      if ((lenn.x * lenn.x) + (lenn.y * lenn.y) < 0.1)
      {
        color = colours[0];
      }
      break;
  }


  FragColor = vec4(color, 1.0);
}

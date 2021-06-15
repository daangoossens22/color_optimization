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
const int step_constant = 4;
const int step_bilinear = 5;
const int quadratic_step = 6;
const int step_smooth = 7;
const int testing = 8;

const int triangles_per_side = 52;

// main color (r, g, b)
uniform variables1 {
  float var1[triangles_per_side * triangles_per_side * 2 * 3];
};
// secondairy color (r, g, b)
uniform variables2 {
  float var2[triangles_per_side * triangles_per_side * 2 * 3];
};
// 3 variables per triangle
uniform variables3 {
  float var3[triangles_per_side * triangles_per_side * 2 * 3];
};
// layout (std140) uniform variables1 {
// uniform variables1 {
//   float red_color[100*100];
//   float green_color[100*100];
//   float blue_color[100*100];
// };

vec2 float_to_coor(in float val)
{
  vec2 point = vec2(0.0f, 0.0f);
  if (val <= 1.0f)
  {
    vec2 dir = vert[1].xy - vert[0].xy;
    point = vert[0].xy + dir * val;
  }
  else if (val <= 2.0f)
  {
    vec2 dir = vert[2].xy - vert[1].xy;
    point = vert[1].xy + dir * (val - 1.0f);
  }
  else
  {
    vec2 dir = vert[0].xy - vert[2].xy;
    point = vert[2].xy + dir * (val - 2.0f);
  }
  return(point);
}

void main()
{
  vec3 color = vec3(0.0, 0.0, 0.0);
  vec3 normal_coor = coord.x * vert[0] + coord.y * vert[1] + coord.z * vert[2];
  vec2 origin_coor = vec2(0.0f, 0.0f);
  float length_x = 0.0f;
  float length_y = 0.0f;
  if (vert[0].x < vert[1].x)
  {
    origin_coor = vert[0].xy;
    length_x = length(vert[1].xy - origin_coor.xy);
    length_y = length(vert[2].xy - origin_coor.xy);
  }
  else
  {
    origin_coor = vec2(vert[1].x, vert[0].y);
    length_x = length(vert[0].xy - origin_coor.xy);
    length_y = length(vert[1].xy - origin_coor.xy);
  }
  vec2 normal_coord2 = vec2(normal_coor.xy - origin_coor);
  normal_coord2.x = normal_coord2.x / length_x;
  normal_coord2.y = normal_coord2.y / length_y;

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
    case step_constant:
      {
        float c0 = var3[(gl_PrimitiveID * 3) + 0];
        float c1 = var3[(gl_PrimitiveID * 3) + 1];
        float c2 = var3[(gl_PrimitiveID * 3) + 2];

        vec3 color1 = vec3(var1[gl_PrimitiveID * 3], var1[(gl_PrimitiveID * 3) + 1], var1[(gl_PrimitiveID * 3) + 2]);
        vec3 color2 = vec3(var2[gl_PrimitiveID * 3], var2[(gl_PrimitiveID * 3) + 1], var2[(gl_PrimitiveID * 3) + 2]);

        // vertical line (inf slope)
        if (c2 >= 0.5f && c2 <= 1.5f)
        {
          if (normal_coord2.x < c0)
          {
            color = color1;
          }
          else
          {
            color = color2;
          }
        }
        else
        {
          if (normal_coord2.x * c1 + c0 > normal_coord2.y)
          {
            color = color1;
          }
          else
          {
            color = color2;
          }
        }
      }
      break;
    // ---------------------------------------------------------------------
    // case step_bilinear:
    //   {
    //     float c0 = var3[(gl_PrimitiveID * 3) + 0];
    //     float c1 = var3[(gl_PrimitiveID * 3) + 1];
    //     float c2 = var3[(gl_PrimitiveID * 3) + 2];

    //     vec3 color1 = vec3(var1[gl_PrimitiveID * 3], var1[(gl_PrimitiveID * 3) + 1], var1[(gl_PrimitiveID * 3) + 2]);
    //     vec3 color2 = vec3(var2[gl_PrimitiveID * 3], var2[(gl_PrimitiveID * 3) + 1], var2[(gl_PrimitiveID * 3) + 2]);

    //     if (!(point1 > -0.1f && point1 <= 1.0f && point2 > -0.1f && point2 <= 1.0f))
    //     {
    //       if (result < 0.0f)
    //       {
    //         color = vec3(var1[gl_PrimitiveID * 3], var1[(gl_PrimitiveID * 3) + 1], var1[(gl_PrimitiveID * 3) + 2]);
    //       }
    //       else
    //       {
    //         color = vec3(var2[gl_PrimitiveID * 3], var2[(gl_PrimitiveID * 3) + 1], var2[(gl_PrimitiveID * 3) + 2]);
    //       }
    //     }
    //     else
    //     {
    //       color = coord.x * colours[0] + coord.y * colours[1] + coord.z * colours[2];
    //     }
    //   }
    //   break;
    case quadratic_step:
      {
        // change to take the quadratic polynomial -> take derivative vector -> take dot product
        float c0 = var3[(gl_PrimitiveID * 3) + 0];
        float c1 = var3[(gl_PrimitiveID * 3) + 1];
        float c2 = var3[(gl_PrimitiveID * 3) + 2];

        vec3 color1 = vec3(var1[gl_PrimitiveID * 3], var1[(gl_PrimitiveID * 3) + 1], var1[(gl_PrimitiveID * 3) + 2]);
        vec3 color2 = vec3(var2[gl_PrimitiveID * 3], var2[(gl_PrimitiveID * 3) + 1], var2[(gl_PrimitiveID * 3) + 2]);

        if (normal_coord2.x * normal_coord2.x * c2 + normal_coord2.x * c1 + c0 >= normal_coord2.y)
        {
          color = color1;
        }
        else
        {
          color = color2;
        }
      }
      break;
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
    {
      float point1 = weight.x;
      float point2 = weight.y;

      vec2 p1 = float_to_coor(point1);
      vec2 p2 = float_to_coor(point2);

      float dis1 = length(normal_coor.xy - p1);
      float dis2 = length(normal_coor.xy - p2);
      if (dis1 < 0.1f)
      {
        color = vec3(1.0f, 0.0f, 0.0f);
      }
      else if (dis2 < 0.1f)
      {
        color = vec3(0.0f, 0.0f, 1.0f);
      }
      else
      {
        color = vec3(var1[gl_PrimitiveID * 3], var1[(gl_PrimitiveID * 3) + 1], var1[(gl_PrimitiveID * 3) + 2]);
      }

    }
    break;
  }

  FragColor = vec4(color, 1.0);
}



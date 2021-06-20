#version 330 core
out vec4 FragColor;
flat in vec3 colours[3];
flat in vec3 vert[3];
in vec3 coord;

// uniform vec4 weight;
uniform int mode;
const int constant_color_avg = 0;
const int constant_color_center = 1;
const int bilinear_interpolation_no_opt = 2;
const int linear_split_constant = 3;
const int quadratic_split_constant = 4;
const int bilinear_interpolation_opt = 5;
const int biquadratic_interpolation = 6;
const int bicubic_interpolation = 7;
const int biquartic_interpolation = 8;

const int triangles_per_side = 52;

// main color (r, g, b)
uniform variables1 {
  float var1[triangles_per_side * triangles_per_side * 2 * 3];
};
// secondairy color (r, g, b)
uniform variables2 {
  float var2[triangles_per_side * triangles_per_side * 2 * 3];
};
uniform variables3 {
  float var3[triangles_per_side * triangles_per_side * 2 * 3];
};
uniform variables4 {
  float var4[triangles_per_side * triangles_per_side * 2 * 3];
};
uniform variables5 {
  float var5[triangles_per_side * triangles_per_side * 2 * 3];
};
uniform variables6 {
  float var6[triangles_per_side * triangles_per_side * 2 * 3];
};
uniform variables7 {
  float var7[triangles_per_side * triangles_per_side * 2 * 3];
};
uniform variables8 {
  float var8[triangles_per_side * triangles_per_side * 2 * 3];
};
uniform variables9 {
  float var9[triangles_per_side * triangles_per_side * 2 * 3];
};
uniform variables10 {
  float var10[triangles_per_side * triangles_per_side * 2 * 3];
};
uniform variables11 {
  float var11[triangles_per_side * triangles_per_side * 2 * 3];
};
uniform variables12 {
  float var12[triangles_per_side * triangles_per_side * 2 * 3];
};
uniform variables13 {
  float var13[triangles_per_side * triangles_per_side * 2 * 3];
};
uniform variables14 {
  float var14[triangles_per_side * triangles_per_side * 2 * 3];
};
uniform variables15 {
  float var15[triangles_per_side * triangles_per_side * 2 * 3];
};

vec2 get_coords_triangle_space(in vec2 normal_coor)
{
  // calculate the coord in the "triangle space" where the origin is at the bottom left of the bounding box surrounding the triangle
  // and all vertices lie in either one of these coords (0,0) (1,0) (0,1) (1,1)
  vec2 origin_coor = vec2(0.0f, 0.0f);
  float length_x = 0.0f;
  float length_y = 0.0f;
  if (vert[0].x < vert[1].x)
  {
    origin_coor = vert[0].xy;
    length_x = length(vert[1].xy - origin_coor);
    length_y = length(vert[2].xy - origin_coor);
  }
  else
  {
    origin_coor = vec2(vert[1].x, vert[0].y);
    length_x = length(vert[0].xy - origin_coor);
    length_y = length(vert[1].xy - origin_coor);
  }
  vec2 normal_coord2 = vec2(normal_coor - origin_coor);
  normal_coord2.x = normal_coord2.x / length_x;
  normal_coord2.y = normal_coord2.y / length_y;

  return normal_coord2;
}


vec3 compute_general_interpolation(in int n);

void main()
{
  vec3 color = vec3(0.0, 0.0, 0.0);
  vec3 normal_coor = coord.x * vert[0] + coord.y * vert[1] + coord.z * vert[2]; // coords in the clip space
  vec2 triangle_coor = get_coords_triangle_space(normal_coor.xy);

  // set output color of pixel according to which coloring mode is selected
  switch (mode)
  {
    case constant_color_avg:
    case constant_color_center:
      color = vec3(var1[gl_PrimitiveID * 3], var1[(gl_PrimitiveID * 3) + 1], var1[(gl_PrimitiveID * 3) + 2]);
      break;
    // ---------------------------------------------------------------------
    case bilinear_interpolation_no_opt:
      color = coord.x * colours[0] + coord.y * colours[1] + coord.z * colours[2];
      break;
    // ---------------------------------------------------------------------
    case linear_split_constant:
      {
        float c0 = var3[(gl_PrimitiveID * 3) + 0];
        float c1 = var3[(gl_PrimitiveID * 3) + 1];
        float c2 = var3[(gl_PrimitiveID * 3) + 2];

        vec3 color1 = vec3(var1[gl_PrimitiveID * 3], var1[(gl_PrimitiveID * 3) + 1], var1[(gl_PrimitiveID * 3) + 2]);
        vec3 color2 = vec3(var2[gl_PrimitiveID * 3], var2[(gl_PrimitiveID * 3) + 1], var2[(gl_PrimitiveID * 3) + 2]);

        // vertical line (inf slope)
        if (c2 >= 0.5f && c2 <= 1.5f)
        {
          if (triangle_coor.x < c0)
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
          if (triangle_coor.x * c1 + c0 > triangle_coor.y)
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
    case quadratic_split_constant:
      {
        // change to take the quadratic polynomial -> take derivative vector -> take dot product
        float c0 = var3[(gl_PrimitiveID * 3) + 0];
        float c1 = var3[(gl_PrimitiveID * 3) + 1];
        float c2 = var3[(gl_PrimitiveID * 3) + 2];

        vec3 color1 = vec3(var1[gl_PrimitiveID * 3], var1[(gl_PrimitiveID * 3) + 1], var1[(gl_PrimitiveID * 3) + 2]);
        vec3 color2 = vec3(var2[gl_PrimitiveID * 3], var2[(gl_PrimitiveID * 3) + 1], var2[(gl_PrimitiveID * 3) + 2]);

        if (triangle_coor.x * triangle_coor.x * c2 + triangle_coor.x * c1 + c0 >= triangle_coor.y)
        {
          color = color1;
        }
        else
        {
          color = color2;
        }
      }
      break;
    // ---------------------------------------------------------------------
    case bilinear_interpolation_opt:
      {
        color = compute_general_interpolation(1);
      }
      break;
    // ---------------------------------------------------------------------
    case biquadratic_interpolation:
      {
        color = compute_general_interpolation(2);
      }
      break;
    // ---------------------------------------------------------------------
    case bicubic_interpolation:
      {
        color = compute_general_interpolation(3);
      }
      break;
    case biquartic_interpolation:
      {
        color = compute_general_interpolation(4);
      }
      break;
  }
  FragColor = vec4(color, 1.0);
}

float fact(in int n)
{
  float result = 1;
  for (int i = 1; i <= n; ++i)
  {
    result *= i;
  }
  return result;
}
// pow function of glsl give weird artifacts for intperpolation because casting the exponent to float gives weird errors
// also this is probably faster than the general case where exponent is a float
float pow2(in float base, in int exponent)
{
  float res = 1.0f;
  for (int i = 0; i < exponent; ++i)
  {
    res *= base;
  }
  return res;
}
vec3 compute_general_interpolation(in int n)
{
  vec3 control_points[15];
  control_points[0] = vec3(var1[gl_PrimitiveID * 3], var1[(gl_PrimitiveID * 3) + 1], var1[(gl_PrimitiveID * 3) + 2]);
  control_points[1] = vec3(var2[gl_PrimitiveID * 3], var2[(gl_PrimitiveID * 3) + 1], var2[(gl_PrimitiveID * 3) + 2]);
  control_points[2] = vec3(var3[gl_PrimitiveID * 3], var3[(gl_PrimitiveID * 3) + 1], var3[(gl_PrimitiveID * 3) + 2]);
  control_points[3] = vec3(var4[gl_PrimitiveID * 3], var4[(gl_PrimitiveID * 3) + 1], var4[(gl_PrimitiveID * 3) + 2]);
  control_points[4] = vec3(var5[gl_PrimitiveID * 3], var5[(gl_PrimitiveID * 3) + 1], var5[(gl_PrimitiveID * 3) + 2]);
  control_points[5] = vec3(var6[gl_PrimitiveID * 3], var6[(gl_PrimitiveID * 3) + 1], var6[(gl_PrimitiveID * 3) + 2]);
  control_points[6] = vec3(var7[gl_PrimitiveID * 3], var7[(gl_PrimitiveID * 3) + 1], var7[(gl_PrimitiveID * 3) + 2]);
  control_points[7] = vec3(var8[gl_PrimitiveID * 3], var8[(gl_PrimitiveID * 3) + 1], var8[(gl_PrimitiveID * 3) + 2]);
  control_points[8] = vec3(var9[gl_PrimitiveID * 3], var9[(gl_PrimitiveID * 3) + 1], var9[(gl_PrimitiveID * 3) + 2]);
  control_points[9] = vec3(var10[gl_PrimitiveID * 3], var10[(gl_PrimitiveID * 3) + 1], var10[(gl_PrimitiveID * 3) + 2]);
  control_points[10] = vec3(var11[gl_PrimitiveID * 3], var11[(gl_PrimitiveID * 3) + 1], var11[(gl_PrimitiveID * 3) + 2]);
  control_points[11] = vec3(var12[gl_PrimitiveID * 3], var12[(gl_PrimitiveID * 3) + 1], var12[(gl_PrimitiveID * 3) + 2]);
  control_points[12] = vec3(var13[gl_PrimitiveID * 3], var13[(gl_PrimitiveID * 3) + 1], var13[(gl_PrimitiveID * 3) + 2]);
  control_points[13] = vec3(var14[gl_PrimitiveID * 3], var14[(gl_PrimitiveID * 3) + 1], var14[(gl_PrimitiveID * 3) + 2]);
  control_points[14] = vec3(var15[gl_PrimitiveID * 3], var15[(gl_PrimitiveID * 3) + 1], var15[(gl_PrimitiveID * 3) + 2]);

  int index = 0;
  float numerator = fact(n);
  vec3 res_color = vec3(0.0f, 0.0f, 0.0f);
  for (int i = 0; i <= n; ++i)
  {
      for (int j = 0; i+j <= n; ++j)
      {
          int k = n - i - j;
          float multiplier = (numerator / (fact(i) * fact(j) * fact(k)));
          res_color += control_points[index] * multiplier * pow2(coord.x, i) * pow2(coord.y, j) * pow2(coord.z, k);
          ++index;
      }
  }
  return res_color;
}
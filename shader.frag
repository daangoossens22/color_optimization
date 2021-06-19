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

void main()
{
  vec3 color = vec3(0.0, 0.0, 0.0);
  vec3 normal_coor = coord.x * vert[0] + coord.y * vert[1] + coord.z * vert[2]; // coords in the clip space

  // calculate the coord in the "triangle space" where the origin is at the bottom left of the bounding box surrounding the triangle
  // and all vertices lie in either one of these coords (0,0) (1,0) (0,1) (1,1)
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
    case quadratic_split_constant:
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
    // ---------------------------------------------------------------------
    case bilinear_interpolation_opt:
      {
        // get all the color values of the control points
        vec3 color_p100 = vec3(var1[gl_PrimitiveID * 3], var1[(gl_PrimitiveID * 3) + 1], var1[(gl_PrimitiveID * 3) + 2]);
        vec3 color_p010 = vec3(var2[gl_PrimitiveID * 3], var2[(gl_PrimitiveID * 3) + 1], var2[(gl_PrimitiveID * 3) + 2]);
        vec3 color_p001 = vec3(var3[gl_PrimitiveID * 3], var3[(gl_PrimitiveID * 3) + 1], var3[(gl_PrimitiveID * 3) + 2]);

        color = coord.x * color_p100 +
                coord.y * color_p010 +
                coord.z * color_p001;
      }
      break;
    // ---------------------------------------------------------------------
    case biquadratic_interpolation:
      {
        // get all the color values of the control points
        vec3 color_p101 = vec3(var1[gl_PrimitiveID * 3], var1[(gl_PrimitiveID * 3) + 1], var1[(gl_PrimitiveID * 3) + 2]);
        vec3 color_p011 = vec3(var2[gl_PrimitiveID * 3], var2[(gl_PrimitiveID * 3) + 1], var2[(gl_PrimitiveID * 3) + 2]);
        vec3 color_p110 = vec3(var3[gl_PrimitiveID * 3], var3[(gl_PrimitiveID * 3) + 1], var3[(gl_PrimitiveID * 3) + 2]);
        vec3 color_p200 = vec3(var4[gl_PrimitiveID * 3], var4[(gl_PrimitiveID * 3) + 1], var4[(gl_PrimitiveID * 3) + 2]);
        vec3 color_p020 = vec3(var5[gl_PrimitiveID * 3], var5[(gl_PrimitiveID * 3) + 1], var5[(gl_PrimitiveID * 3) + 2]);
        vec3 color_p002 = vec3(var6[gl_PrimitiveID * 3], var6[(gl_PrimitiveID * 3) + 1], var6[(gl_PrimitiveID * 3) + 2]);

        // compute the output color
        color = coord.x * coord.x * color_p200 +
                2.0f * coord.x * coord.y * color_p110 +
                2.0f * coord.x * coord.z * color_p101 +
                coord.y * coord.y * color_p020 +
                2.0f * coord.y * coord.z * color_p011 +
                coord.z * coord.z * color_p002;
      }
      break;
    // ---------------------------------------------------------------------
    case bicubic_interpolation:
      {
        // get all the color values of the control points
        vec3 p300 = vec3(var1[gl_PrimitiveID * 3], var1[(gl_PrimitiveID * 3) + 1], var1[(gl_PrimitiveID * 3) + 2]);
        vec3 p210 = vec3(var2[gl_PrimitiveID * 3], var2[(gl_PrimitiveID * 3) + 1], var2[(gl_PrimitiveID * 3) + 2]);
        vec3 p201 = vec3(var3[gl_PrimitiveID * 3], var3[(gl_PrimitiveID * 3) + 1], var3[(gl_PrimitiveID * 3) + 2]);
        vec3 p120 = vec3(var4[gl_PrimitiveID * 3], var4[(gl_PrimitiveID * 3) + 1], var4[(gl_PrimitiveID * 3) + 2]);
        vec3 p111 = vec3(var5[gl_PrimitiveID * 3], var5[(gl_PrimitiveID * 3) + 1], var5[(gl_PrimitiveID * 3) + 2]);
        vec3 p102 = vec3(var6[gl_PrimitiveID * 3], var6[(gl_PrimitiveID * 3) + 1], var6[(gl_PrimitiveID * 3) + 2]);
        vec3 p030 = vec3(var7[gl_PrimitiveID * 3], var7[(gl_PrimitiveID * 3) + 1], var7[(gl_PrimitiveID * 3) + 2]);
        vec3 p021 = vec3(var8[gl_PrimitiveID * 3], var8[(gl_PrimitiveID * 3) + 1], var8[(gl_PrimitiveID * 3) + 2]);
        vec3 p012 = vec3(var9[gl_PrimitiveID * 3], var9[(gl_PrimitiveID * 3) + 1], var9[(gl_PrimitiveID * 3) + 2]);
        vec3 p003 = vec3(var10[gl_PrimitiveID * 3], var10[(gl_PrimitiveID * 3) + 1], var10[(gl_PrimitiveID * 3) + 2]);

        // compute the output color
        color = p300 * coord.x * coord.x * coord.x +
                3.0f * p210 * coord.x * coord.x * coord.y +
                3.0f * p201 * coord.x * coord.x * coord.z +
                3.0f * p120 * coord.x * coord.y * coord.y +
                6.0f * p111 * coord.x * coord.y * coord.z +
                3.0f * p102 * coord.x * coord.z * coord.z +
                p030 * coord.y * coord.y * coord.y +
                3.0f * p021 * coord.y * coord.y * coord.z +
                3.0f * p012 * coord.y * coord.z * coord.z +
                p003 * coord.z * coord.z * coord.z;
      }
      break;
  }
  FragColor = vec4(color, 1.0);
}



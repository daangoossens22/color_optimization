#version 330 core
out vec4 FragColor;
flat in vec3 colours[3];
flat in vec3 vert[3];
in vec3 coord;

uniform vec4 weight;
uniform int mode;
const int constant_color_avg = 0;
const int constant_color_center = 1;
const int bilinear_interpolation = 2;
const int step_constant = 3;
const int step_bilinear = 4;
const int quadratic_step = 5;
const int quadratic_interpolation = 6;

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
    case bilinear_interpolation:
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
    case step_bilinear:
      {
        float c0 = var3[(gl_PrimitiveID * 3) + 0];
        float c1 = var3[(gl_PrimitiveID * 3) + 1];
        float c2 = var3[(gl_PrimitiveID * 3) + 2];

        vec3 color1 = vec3(var1[gl_PrimitiveID * 3], var1[(gl_PrimitiveID * 3) + 1], var1[(gl_PrimitiveID * 3) + 2]);
        vec3 color2 = vec3(var2[gl_PrimitiveID * 3], var2[(gl_PrimitiveID * 3) + 1], var2[(gl_PrimitiveID * 3) + 2]);

        // c2 = 1 -> vertical line (inf slope)
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
        else if (c2 >= -0.5f && c2 <= 0.5f) // c2 = 0 -> normal case y = c0 + c1*x
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
        else
        {
          color = coord.x * colours[0] + coord.y * colours[1] + coord.z * colours[2];
        }
      }
      break;
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
    case quadratic_interpolation:
      {
        // get all the color values of the control points
        // vec3 color_p200 = colours[0];
        // vec3 color_p002 = colours[1];
        // vec3 color_p020 = colours[2];
        // vec3 color_p200 = colours[0];
        // vec3 color_p002 = colours[2];
        // vec3 color_p020 = colours[1];
        vec3 color_p101 = vec3(var1[gl_PrimitiveID * 3], var1[(gl_PrimitiveID * 3) + 1], var1[(gl_PrimitiveID * 3) + 2]);
        vec3 color_p011 = vec3(var2[gl_PrimitiveID * 3], var2[(gl_PrimitiveID * 3) + 1], var2[(gl_PrimitiveID * 3) + 2]);
        vec3 color_p110 = vec3(var3[gl_PrimitiveID * 3], var3[(gl_PrimitiveID * 3) + 1], var3[(gl_PrimitiveID * 3) + 2]);
        vec3 color_p200 = vec3(var4[gl_PrimitiveID * 3], var4[(gl_PrimitiveID * 3) + 1], var4[(gl_PrimitiveID * 3) + 2]);
        vec3 color_p020 = vec3(var5[gl_PrimitiveID * 3], var5[(gl_PrimitiveID * 3) + 1], var5[(gl_PrimitiveID * 3) + 2]);
        vec3 color_p002 = vec3(var6[gl_PrimitiveID * 3], var6[(gl_PrimitiveID * 3) + 1], var6[(gl_PrimitiveID * 3) + 2]);

        // compute the output color
        color = coord.x * coord.x * color_p200 +
                2 * coord.x * coord.y * color_p110 +
                2 * coord.x * coord.z * color_p101 +
                coord.y * coord.y * color_p020 +
                2 * coord.y * coord.z * color_p011 +
                coord.z * coord.z * color_p002;
      }
      break;
  }

  FragColor = vec4(color, 1.0);
}



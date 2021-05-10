#version 330 core
out vec4 FragColor;
// in vec3 colour;
flat in vec3 colours[3];
flat in vec3 vert[3];
in vec3 coord;

uniform vec3 weight;
uniform int mode;
const int constant_color = 0;
const int bilinear_interpolation = 1;

void main()
{
  vec3 color = vec3(0.0, 0.0, 0.0);
  vec3 normal_coor = coord.x * vert[0] + coord.y * vert[1] + coord.z * vert[2];

  if (mode == constant_color)
  {
    color = colours[0];
  }
  else if (mode == bilinear_interpolation)
  {
    color = coord.x * colours[0] + coord.y * colours[1] + coord.z * colours[2];
  }
  else
  {
    if (length(normal_coor) < 0.48)
    {
      color = colours[0];
    }
    // color = step(weight, coord);
    // if (coord.x > weight.x || coord.y > weight.y)
    // {
    //   color = colours[1];
    // }
  }
  FragColor = vec4(color, 1.0);
}

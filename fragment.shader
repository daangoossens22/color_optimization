#version 330 core
out vec4 FragColor;
// in vec3 colour;
flat in vec3 colours[3];
in vec3 coord;
void main()
{
  // int i = (coord.x > coord.y && coord.x > coord.z) ? 0 : ((coord.y > coord.z) ? 1 : 2);
  // int i = (coord.x > 0.1 && coord.x < 0.9) ? 0 : 1;
  // FragColor = vec4(colours[i], 1.0f);
  FragColor = vec4(coord.x * colours[0] + coord.y * colours[1] + coord.z * colours[2], 1.0);
  // FragColor = vec4(colour, 1.0);
}

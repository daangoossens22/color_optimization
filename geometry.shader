#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;
in vec3 colour[];
flat out vec3 colours[3];
out vec3 coord;
void main()
{
  for (int i = 0; i < 3; i++) { colours[i] = colour[i]; }
  for (int i = 0; i < 3; i++)
  {
    coord = vec3(0.0);
    coord[i] = 1.0;
    gl_Position = gl_in[i].gl_Position;
    EmitVertex();
  }
}

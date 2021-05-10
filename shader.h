#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include <glad/gl.h>

class Shader
{
    public:
        unsigned int ID;
    
        Shader(const char* vertex_path, const char* geometry_path, const char* fragment_path)
        {
            unsigned int vertexShader = load_shader(vertex_path, GL_VERTEX_SHADER);
            unsigned int geometryShader = load_shader(geometry_path, GL_GEOMETRY_SHADER);
            unsigned int fragmentShader = load_shader(fragment_path, GL_FRAGMENT_SHADER);
        
            ID = glCreateProgram();
            glAttachShader(ID, vertexShader);
            glAttachShader(ID, geometryShader);
            glAttachShader(ID, fragmentShader);
            glLinkProgram(ID);
        
            int success;
            char infoLog[512];
            glGetShaderiv(ID, GL_LINK_STATUS, &success);
            if (!success)
            {
                glGetShaderInfoLog(ID, 512, NULL, infoLog);
                std::cout << "Linking shaders failed\n" << infoLog << std::endl;
            }
        
            glDeleteShader(vertexShader);
            glDeleteShader(geometryShader);
            glDeleteShader(fragmentShader);
        }
    
        void use()
        {
            glUseProgram(ID);
        }
    
    private:
        unsigned int load_shader(std::string path, GLenum type)
        {
            std::ifstream t (path);
            std::stringstream buffer;
            buffer << t.rdbuf();
            std::string source_string = buffer.str();
            const char* source_code = source_string.c_str();
            unsigned int shader;
            shader = glCreateShader(type);
            glShaderSource(shader, 1, &source_code, NULL);
            glCompileShader(shader);
        
            int success;
            char infoLog[512];
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success)
            {
                glGetShaderInfoLog(shader, 512, NULL, infoLog);
                std::cout << path << " compilation failed\n" << infoLog << std::endl;
            }
            return shader;
        }
  
};

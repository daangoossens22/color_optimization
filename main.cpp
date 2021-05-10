#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#include <cmath>

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include "shader.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int main(int, char**)
{
    float random_colors[101*101*3];
    for (int i = 0; i < 101*101*3; i++)
    {
        random_colors[i] = (float)std::rand() / (RAND_MAX + 1.0f);
    }

    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // GL 3.3 + GLSL 330
    const char* glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_FALSE);
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only

    // Create window with graphics context
    // GLFWwindow* window = glfwCreateWindow(900, 900, "Test", NULL, NULL);
    GLFWwindow* window = glfwCreateWindow(1600, 900, "Test", NULL, NULL);
    if (window == NULL)
    {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize OpenGL loader
    bool err = gladLoadGL(glfwGetProcAddress) == 0; // glad2 recommend using the windowing library loader instead of the (optionally) bundled one.
    if (err)
    {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        return 1;
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // ---------------------------------------------------
    Shader shader ("vertex.shader", "geometry.shader", "fragment.shader");
    
    // ---------------------------------------------------
    float vertices[] = {
       0.0f,  0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // bottom left
       1.0f,  0.0f, 0.0f, 0.0f, 1.0f, 0.0f, // bottom right
       0.0f,  1.0f, 0.0f, 0.0f, 0.0f, 1.0f, // top left
       1.0f,  1.0f, 0.0f, 0.0f, 0.0f, 1.0f  // top right
    };

    unsigned int indices[] = {
        0, 1, 2,
        1, 2, 3
    };

    unsigned int VAO, VBO, EBO;

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*) 0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*) (3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // imgui variables
    bool show_demo_window = false;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    // ImVec2 num_triangles_dimensions = ImVec2(1, 1);
    int num_triangles_dimensions[2] = { 2, 2 };
    int mode = 2;
    ImVec4 vcolor1 = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
    ImVec4 vcolor2 = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
    ImVec4 vcolor3 = ImVec4(0.0f, 0.0f, 1.0f, 1.0f);
    float weightx = 0.0f;
    float weighty = 0.0f;
    float weightz = 0.0f;
    float weightw = 0.0f;


    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            ImGui::Begin("Main window");
            // ImGui::Text("This is some useful text.");

            ImGui::Checkbox("Demo Window", &show_demo_window);
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::ColorEdit3("clear color", (float*)&clear_color);

            ImGui::SliderInt2("# triangles width x height", num_triangles_dimensions, 1, 100);

            ImGui::Combo("mode", &mode, "constant\0bilinear\0step\0smooth step\0testing\0\0");

            ImGui::ColorEdit3("vertex 1", (float*)&vcolor1);
            ImGui::ColorEdit3("vertex 2", (float*)&vcolor2);
            ImGui::ColorEdit3("vertex 3", (float*)&vcolor3);
            ImGui::SliderFloat("float 1", &weightx, -1.0f, 1.0f);
            ImGui::SliderFloat("float 2", &weighty, -1.0f, 1.0f);
            ImGui::SliderFloat("float 3", &weightz, -1.0f, 1.0f);
            ImGui::SliderFloat("float 4", &weightw, -1.0f, 1.0f);

            // if (ImGui::Button("Button"))
            //     counter++;
            // ImGui::SameLine();
            // ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }

        // TODO change this into windows for different interpolations for the triangles (constant, linear, non-linear)
        if (show_another_window)
        {
            ImGui::Begin("Another Window", &show_another_window);
            ImGui::End();
        }
        // demo window that displays most functionality
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);


        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(display_w - display_h, 0, display_h, display_h); // have a square viewport where the imgui stuff can be on the left
        glClearColor(clear_color.x, clear_color.y, clear_color.z, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);


        shader.use();

        // set/update shader parameters
        int x_max = num_triangles_dimensions[0] + 1;
        int y_max = num_triangles_dimensions[1] + 1;
        float x_step = 1.0f / ((float)x_max - 1.0f);
        float y_step = 1.0f / ((float)y_max - 1.0f);
        
        float vertices[x_max * y_max * 6]; // 6 elements per vertex
        for (int y = 0; y < y_max; y++)
        {
            for (int x = 0; x < x_max; x++)
            {
                int base_index = (x + y * x_max) * 6;
                vertices[base_index + 0] = x * x_step;
                vertices[base_index + 1] = y * y_step;
                vertices[base_index + 2] = 0.0f;
                // random colors
                vertices[base_index + 3] = random_colors[base_index / 2];
                vertices[base_index + 4] = random_colors[base_index / 2 + 1];
                vertices[base_index + 5] = random_colors[base_index / 2 + 2];
            }
        }

        x_max--;
        y_max--;
        int num_vertices = x_max * y_max * 2 * 3; // 2 triangles, 3 values per triangle
        unsigned int indices[num_vertices];
        for (int y = 0; y < y_max; y++)
        {
            for (int x = 0; x < x_max; x++)
            {
                unsigned int bottom_left = (x_max + 1) * y + x;
                unsigned int bottom_right = (x_max + 1) * y + x + 1;
                unsigned int top_left = (x_max + 1) * (y + 1) + x;
                unsigned int top_right = (x_max + 1) * (y + 1) + x + 1;

                int base_index = 2 * 3 * (x + y * x_max);
                // triangle 1
                indices[base_index + 0] = bottom_left;
                indices[base_index + 1] = bottom_right;
                indices[base_index + 2] = top_left;
                // triangle 2
                indices[base_index + 3] = bottom_right;
                indices[base_index + 4] = top_left;
                indices[base_index + 5] = top_right;
            }
        }
        
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*) 0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*) (3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glUniform4f(glGetUniformLocation(shader.ID, "weight"), weightx, weighty, weightz, weightw);
        glUniform1i(glGetUniformLocation(shader.ID, "mode"), mode);
        // glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)display_w / (float)display_h, 0.1f, 100.0f);
        // glm::mat4 proj = glm::ortho((-(float)display_w / (float)display_h) + 1.0f, 1.0f, 0.0f, 1.0f, -10.0f, 10.0f);
        glm::mat4 proj = glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, -10.0f, 10.0f); // have a coordinate system (0, 0) bottom left and (1, 1) top right
        glUniformMatrix4fv(glGetUniformLocation(shader.ID, "projection"), 1, GL_FALSE, glm::value_ptr(proj));

        glBindVertexArray(VAO);
        // glDrawArrays(GL_TRIANGLES, 0, 3);
        glDrawElements(GL_TRIANGLES, num_vertices, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
        // glfwPollEvents();
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shader.ID);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}




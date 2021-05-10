#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#include <cmath>

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"


static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int main(int, char**)
{
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
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Test", NULL, NULL);
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
    // Shader shader ("vertex.shader", "geometry.shader", "fragment.shader");
    Shader shader ("vertex2.shader", "geometry2.shader", "fragment2.shader");
    
    // ---------------------------------------------------
    float vertices[] = {
       0.0f,  0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
       1.0f,  0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
       0.5f,  0.866025f, 0.0f, 0.0f, 0.0f, 1.0f
    };

    unsigned int VAO, VBO;

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*) 0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*) (3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // imgui variables
    bool show_demo_window = false;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
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

            ImGui::Combo("mode", &mode, "constant\0bilinear\0testing\0\0");

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
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x, clear_color.y, clear_color.z, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);


        shader.use();
        vertices[0 * 6 + 3] = vcolor1.x;
        vertices[0 * 6 + 4] = vcolor1.y;
        vertices[0 * 6 + 5] = vcolor1.z;
        vertices[1 * 6 + 3] = vcolor2.x;
        vertices[1 * 6 + 4] = vcolor2.y;
        vertices[1 * 6 + 5] = vcolor2.z;
        vertices[2 * 6 + 3] = vcolor3.x;
        vertices[2 * 6 + 4] = vcolor3.y;
        vertices[2 * 6 + 5] = vcolor3.z;
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
        glUniform4f(glGetUniformLocation(shader.ID, "weight"), weightx, weighty, weightz, weightw);
        glUniform1i(glGetUniformLocation(shader.ID, "mode"), mode);
        // glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)display_w / (float)display_h, 0.1f, 100.0f);
        // glm::mat4 proj = glm::ortho((float)display_h - (float)display_w, (float)display_h, 0.0f, (float)display_h, -10.0f, 10.0f);
        glm::mat4 proj = glm::ortho((-(float)display_w / (float)display_h) + 1.0f, 1.0f, 0.0f, 1.0f, -10.0f, 10.0f);
        glUniformMatrix4fv(glGetUniformLocation(shader.ID, "projection"), 1, GL_FALSE, glm::value_ptr(proj));

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
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




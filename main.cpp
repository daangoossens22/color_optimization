#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <Eigen/Householder>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <gsl/gsl_fit.h>

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include "shader.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

constexpr int max_triangles_per_side = 52; // all the nvidia uniform buffer can handle
constexpr float saliency_bias = 0.1;
constexpr int points_tested_per_side = 4;
enum saliency_method { fine_grained, spectral_residual };

// function pointers for function that are at the bottom
static void glfw_error_callback(int error, const char* description);
void load_picture(cv::Mat& img, const std::string file_name);
void update_saliency_map(const cv::Mat& img, cv::Mat& saliency_map, int saliency_mode);
void get_edges(const cv::Mat& img, cv::Mat& edges, int low_threshold);
GLFWwindow* glfw_setup();

void update_vertex_buffer(int num_triangles_x, int num_triangles_y, float vertices[], const float vertex_colors[]);
void update_index_buffer(int num_triangles_x, int num_triangles_y, unsigned int indices[]);

void update_vertex_colors(int num_triangles_x, int num_triangles_y, const cv::Mat& img, const cv::Mat& saliency_map, bool use_saliency, float vertices[], float vertex_colors[]);
void update_triangle_colors(int num_triangles_x, int num_triangles_y, const cv::Mat& img, const cv::Mat& saliency_map, bool use_saliency, const float vertices[], float triangle_colors1[]);
void update_constant_colors(int num_triangles_x, int num_triangles_y, const cv::Mat& img, const cv::Mat& saliency_map, bool use_saliency, const float vertices[], float triangle_colors1[]);
void get_average_color(float bottom_left_x_pixels, float bottom_left_y_pixels, float diff_x_pixels, float diff_y_pixels, const cv::Mat& img, const cv::Mat& saliency_map, bool use_saliency, float total_1[3], float total_2[3]);
void update_bilinear_colors_opt(int num_triangles_x, int num_triangles_y, const cv::Mat& img, const cv::Mat& saliency_map, bool use_saliency, float vertices[], float vertex_colors[]);
void update_bicubic_variables(int num_triangles_x, int num_triangles_y, const cv::Mat& img, const cv::Mat& saliency_map, bool use_saliency, float vertices[], float vertex_colors[]);
void update_step_constant_color(int num_triangles_x, int num_triangles_y, const cv::Mat& img, const cv::Mat& saliency_map, const cv::Mat& edges, bool use_saliency, const float vertices[], float triangle_colors1[], float triangle_colors2[]);

int main(int argc, const char** argv)
{
    // load image into opencv buffer
    cv::Mat img_temp, img;
    cv::Mat saliency_map;
    // load_picture(img, "lenna.png");
    load_picture(img_temp, "carrot2.png");
    cv::flip(img_temp, img, 0);

    GLFWwindow* window = glfw_setup();
    if (!window) { return 1; };

    Shader shader ("vertex.shader", "geometry.shader", "fragment.shader");
    int num_triangles = max_triangles_per_side * max_triangles_per_side * 2 * 3;
    
    // -----------------------------------------------------------------------------
    // generate opengl buffers
    unsigned int VAO, VBO, EBO;

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    unsigned int variables1, variables2, variables3, variables4;
    glGenBuffers(1, &variables1);
    glBindBuffer(GL_UNIFORM_BUFFER, variables1);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(GLfloat) * num_triangles, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glGenBuffers(1, &variables2);
    glBindBuffer(GL_UNIFORM_BUFFER, variables2);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(GLfloat) * num_triangles, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    unsigned int inddd = glGetUniformBlockIndex(shader.ID, "variables1");
    glUniformBlockBinding(shader.ID, inddd, 0);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, variables1);

    unsigned int inddd2 = glGetUniformBlockIndex(shader.ID, "variables2");
    glUniformBlockBinding(shader.ID, inddd2, 1);
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, variables2);

    // -----------------------------------------------------------------------------
    // imgui variables
    bool show_demo_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    int num_triangles_dimensions[2] = { 11, 11 };
    bool square_grid = true;
    bool use_saliency = true;
    int saliency_mode = 0;
    bool show_saliency_map = false;
    int mode = 0;
    int low_threshold = 130;
    ImVec4 vcolor1 = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
    ImVec4 vcolor2 = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
    ImVec4 vcolor3 = ImVec4(0.0f, 0.0f, 1.0f, 1.0f);
    float weightx = 0.0f;
    float weighty = 0.0f;
    float weightz = 0.0f;
    float weightw = 0.0f;

    // for checking if recalculation is needed
    int old_mode = -1;
    int old_saliency_mode = -1;
    int old_num_triangles_dimensions[2] = { 0, 0};
    bool old_use_saliency = false;

    // make an array for the vertex and triangle colors that can later be loaded into an opengl buffer
    float vertex_colors[(max_triangles_per_side + 1) * (max_triangles_per_side + 1) * 3];
    float triangle_colors1[num_triangles];
    float triangle_colors2[num_triangles];

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();
        
        // -----------------------------------------------------------------------------
        // dear imgui windows
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        {
            ImGui::Begin("Main window");

            ImGui::Checkbox("Demo Window", &show_demo_window);

            ImGui::ColorEdit3("clear color", (float*)&clear_color);

            ImGui::SliderInt2("# triangles width x height", num_triangles_dimensions, 1, max_triangles_per_side);
            ImGui::Checkbox("square grid", &square_grid);

            ImGui::Combo("saliency mode", &saliency_mode, "fine_grained\0spectral_residual\0\0");
            ImGui::Checkbox("use saliency", &use_saliency);
            ImGui::Checkbox("show saliency map (close window by pressing any key)", &show_saliency_map);

            ImGui::Combo("mode", &mode, "constant (avg)\0 constant (center)\0bilinear (no opt)\0bilinear (opt)\0bicubic\0step\0smooth step\0testing\0\0");

            ImGui::SliderInt("theshold edge detection", &low_threshold, 0, 160);

            ImGui::ColorEdit3("vertex 1", (float*)&vcolor1);
            ImGui::ColorEdit3("vertex 2", (float*)&vcolor2);
            ImGui::ColorEdit3("vertex 3", (float*)&vcolor3);
            ImGui::SliderFloat("float 1", &weightx, 0.0f, 3.0f);
            ImGui::SliderFloat("float 2", &weighty, 0.0f, 3.0f);
            ImGui::SliderFloat("float 3", &weightz, 0.0f, 3.0f);
            ImGui::SliderFloat("float 4", &weightw, 0.0f, 3.0f);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }
        if (square_grid) { num_triangles_dimensions[1] = num_triangles_dimensions[0]; }
        // demo window that displays most dear imgui functionality
        if (show_demo_window) { ImGui::ShowDemoWindow(&show_demo_window); }


        // -----------------------------------------------------------------------------
        // Clear values and set opengl viewport
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(display_w - display_h, 0, display_h, display_h); // have a square viewport where the imgui stuff can be on the left
        glClearColor(clear_color.x, clear_color.y, clear_color.z, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);

        shader.use();

        // -----------------------------------------------------------------------------
        // update vertex and index buffers
        float vertices[(num_triangles_dimensions[0] + 1) * (num_triangles_dimensions[1] + 1) * 6]; // 6 elements per vertex
        update_vertex_buffer(num_triangles_dimensions[0], num_triangles_dimensions[1], vertices, vertex_colors);

        int num_vertices = num_triangles_dimensions[0] * num_triangles_dimensions[1] * 2 * 3; // 2 triangles, 3 values per triangle
        unsigned int indices[num_vertices];
        update_index_buffer(num_triangles_dimensions[0], num_triangles_dimensions[1], indices);

        // -----------------------------------------------------------------------------
        // update triangle coloring variables (only when something changed and recalculation is needed)
        if (!(mode == old_mode && 
              num_triangles_dimensions[0] == old_num_triangles_dimensions[0] && 
              num_triangles_dimensions[1] == old_num_triangles_dimensions[1] && 
              use_saliency == old_use_saliency && 
              old_saliency_mode == saliency_mode))
        {
            old_mode = mode;
            old_num_triangles_dimensions[0] = num_triangles_dimensions[0];
            old_num_triangles_dimensions[1] = num_triangles_dimensions[1];
            old_use_saliency = use_saliency;
            old_saliency_mode = saliency_mode;

            update_saliency_map(img, saliency_map, saliency_mode);

            switch (mode)
            {
                case 0:
                    update_constant_colors(num_triangles_dimensions[0], num_triangles_dimensions[1], img, saliency_map, use_saliency, vertices, triangle_colors1);
                    break;
                case 1:
                    update_triangle_colors(num_triangles_dimensions[0], num_triangles_dimensions[1], img, saliency_map, use_saliency, vertices, triangle_colors1);
                    break;
                case 2:
                    update_vertex_colors(num_triangles_dimensions[0], num_triangles_dimensions[1], img, saliency_map, use_saliency, vertices, vertex_colors);
                    break;
                case 3:
                    update_bilinear_colors_opt(num_triangles_dimensions[0], num_triangles_dimensions[1], img, saliency_map, use_saliency, vertices, vertex_colors);
                    break;
                case 4:
                    update_bicubic_variables(num_triangles_dimensions[0], num_triangles_dimensions[1], img, saliency_map, use_saliency, vertices, vertex_colors);
                    break;
                case 5:
                    cv::Mat edges;
                    get_edges(img, edges, low_threshold);

                    update_step_constant_color(num_triangles_dimensions[0], num_triangles_dimensions[1], img, saliency_map, edges, use_saliency, vertices, triangle_colors1, triangle_colors2);
                    break;
            }
        }

        // -----------------------------------------------------------------------------
        // shows the saliency map when the checkbox is selected in the imgui window
        // currently can only be exited by pressing any key (if the close button is pressed it locks the program)
        if (show_saliency_map)
        {
            cv::imshow("saliency map", saliency_map);
            cv::waitKey(0);
            cv::destroyAllWindows();
            show_saliency_map = false;
        }

        // -----------------------------------------------------------------------------
        // put all the buffers on the gpu
        glBindBuffer(GL_UNIFORM_BUFFER, variables1);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, 4 * num_triangles, triangle_colors1);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);

        glBindBuffer(GL_UNIFORM_BUFFER, variables2);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, 4 * num_triangles, triangle_colors2);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
        
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
        glm::mat4 proj = glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, -10.0f, 10.0f); // have a coordinate system (0, 0) bottom left and (1, 1) top right
        glUniformMatrix4fv(glGetUniformLocation(shader.ID, "projection"), 1, GL_FALSE, glm::value_ptr(proj));

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, num_vertices, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // -----------------------------------------------------------------------------
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


// -----------------------------------------------------------------------------
// for each triangle it gets a number of points inside the triangle and gets the color and saliency value of that position in the image
// then it takes a weighted average if saliency is turned on, otherwise it computes the average without the saliency weights
// sets those values in the triangle color array which can be accessed later in the glsl shader by their gl_PrimitiveID (triangle number)
void update_constant_colors(int num_triangles_x, int num_triangles_y, const cv::Mat& img, const cv::Mat& saliency_map, bool use_saliency, const float vertices[], float triangle_colors1[])
{
    int x_max = num_triangles_x;
    int y_max = num_triangles_y;
    float width_triangle_pixels = (float)img.cols / (float)x_max;
    float height_triangle_pixels = (float)img.rows / (float)y_max;
    float diff_x_pixels = width_triangle_pixels / points_tested_per_side;
    float diff_y_pixels = height_triangle_pixels / points_tested_per_side;

    for (int y = 0; y < y_max; y++)
    {
        for (int x = 0; x < x_max; x++)
        {
            // (x_max + 1) becuase the rightmost vertices are already tested in the previous box
            unsigned int bottom_left = (x_max + 1) * y + x;

            // top left = (0, 0), top right = (0, img.cols - 1), bottom left = (img.rows - 1, 0)
            float bottom_left_x_pixels = (float)(vertices[bottom_left * 6] * img.cols);
            float bottom_left_y_pixels = (float)(vertices[bottom_left * 6 + 1] * img.rows);

            float total_1[3] = {0.0, 0.0, 0.0};
            float total_2[3] = {0.0, 0.0, 0.0};
            get_average_color(bottom_left_x_pixels, bottom_left_y_pixels, diff_x_pixels, diff_y_pixels, img, saliency_map, use_saliency, total_1, total_2);

            int basee = (x + (y * x_max)) * 6;
            triangle_colors1[basee + 0] = total_1[2];
            triangle_colors1[basee + 1] = total_1[1];
            triangle_colors1[basee + 2] = total_1[0];
            triangle_colors1[basee + 3] = total_2[2];
            triangle_colors1[basee + 4] = total_2[1];
            triangle_colors1[basee + 5] = total_2[0];
            
            // cv::Vec3b reccc = img.at<cv::Vec3b>(0, img.cols - 1);
        }
    }
}

void get_average_color(float bottom_left_x_pixels, float bottom_left_y_pixels, float diff_x_pixels, float diff_y_pixels, const cv::Mat& img, const cv::Mat& saliency_map, bool use_saliency, float total_1[3], float total_2[3])
{
    float count_1 = 0;
    float count_2 = 0;
    for (int j = 0; j < points_tested_per_side; j++)
    {
        for (int i = 0; i < points_tested_per_side; i++)
        {
            int x2 = std::floor((i * diff_x_pixels) + bottom_left_x_pixels);
            int y2 = std::floor((j * diff_y_pixels) + bottom_left_y_pixels);
            cv::Vec3b val = img.at<cv::Vec3b>(y2, x2);
            float saliency_val = saliency_map.at<float>(y2, x2);
            saliency_val += saliency_bias;

            int vall = i + j + 1;
            if (vall <= points_tested_per_side)
            {
                if (use_saliency)
                {
                    total_1[0] += (val[0] * saliency_val);
                    total_1[1] += (val[1] * saliency_val);
                    total_1[2] += (val[2] * saliency_val);
                    count_1 += saliency_val;
                }
                else
                {
                    total_1[0] += val[0];
                    total_1[1] += val[1];
                    total_1[2] += val[2];
                    count_1 += 1.0;
                }
            }
            if (vall >= points_tested_per_side)
            {
                if (use_saliency)
                {
                    total_2[0] += (val[0] * saliency_val);
                    total_2[1] += (val[1] * saliency_val);
                    total_2[2] += (val[2] * saliency_val);
                    count_2 += saliency_val;
                }
                else
                {
                    total_2[0] += val[0];
                    total_2[1] += val[1];
                    total_2[2] += val[2];
                    count_2 += 1.0;
                }
            }
        }
    }
    // std::cout << total_1 << "\t";
    total_1[0] /= (count_1 * 255.0);
    total_1[1] /= (count_1 * 255.0);
    total_1[2] /= (count_1 * 255.0);
    total_2[0] /= (count_2 * 255.0);
    total_2[1] /= (count_2 * 255.0);
    total_2[2] /= (count_2 * 255.0);
    // std::cout << total_1 << "\t" << count_1 << "\n";
}

// -----------------------------------------------------------------------------
// for each vertex it gets the color at that point in the actual image and updates the vertex color attribute in the vertex buffer
void update_vertex_colors(int num_triangles_x, int num_triangles_y, const cv::Mat& img, const cv::Mat& saliency_map, bool use_saliency, float vertices[], float vertex_colors[])
{
    // set/update shader parameters
    int x_max = num_triangles_x + 1;
    int y_max = num_triangles_y + 1;
    float x_step = 1.0f / ((float)x_max - 1.0f);
    float y_step = 1.0f / ((float)y_max - 1.0f);
    
    // set vertices buffer (vertices + color)
    for (int y = 0; y < y_max; y++)
    {
        for (int x = 0; x < x_max; x++)
        {
            int base_index = (x + y * x_max) * 3;
            int x_coor = std::floor((x * x_step) * img.cols);
            int y_coor = std::floor((y * y_step) * img.rows);
            x_coor = (x_coor == img.cols) ? img.cols - 1 : x_coor;
            y_coor = (y_coor == img.rows) ? img.rows - 1 : y_coor;
            cv::Vec3b val = img.at<cv::Vec3b>(y_coor, x_coor);

            // set colors
            vertex_colors[base_index + 0] = val[2] / 255.0;
            vertex_colors[base_index + 1] = val[1] / 255.0;
            vertex_colors[base_index + 2] = val[0] / 255.0;

            vertices[base_index * 2 + 3] = val[2] / 255.0;
            vertices[base_index * 2 + 4] = val[1] / 255.0;
            vertices[base_index * 2 + 5] = val[0] / 255.0;
        }
    }
}

// -----------------------------------------------------------------------------
// for each triangle it gets the color at the center of the triangle and updates the vertex color attribute in the vertex buffer
void update_triangle_colors(int num_triangles_x, int num_triangles_y, const cv::Mat& img, const cv::Mat& saliency_map, bool use_saliency, const float vertices[], float triangle_colors1[])
{
    int x_max = num_triangles_x;
    int y_max = num_triangles_y;
    float width_triangle_pixels = (float)img.cols / (float)x_max;
    float height_triangle_pixels = (float)img.rows / (float)y_max;

    for (int y = 0; y < y_max; y++)
    {
        for (int x = 0; x < x_max; x++)
        {
            // (x_max + 1) becuase the rightmost vertices are already tested in the previous box
            unsigned int bottom_left = (x_max + 1) * y + x;

            // top left = (0, 0), top right = (0, img.cols - 1), bottom left = (img.rows - 1, 0)
            float bottom_left_x_pixels = (float)(vertices[bottom_left * 6] * img.cols);
            float bottom_left_y_pixels = (float)(vertices[bottom_left * 6 + 1] * img.rows);

            // TODO find center triangle 1
            int x1 = std::floor((0.35355 * width_triangle_pixels) + bottom_left_x_pixels);
            int y1 = std::floor((0.35355 * height_triangle_pixels) + bottom_left_y_pixels);
            // int y1 = std::ceil(bottom_left_y_pixels - (0.35355 * height_triangle_pixels));
            cv::Vec3b val1 = img.at<cv::Vec3b>(y1, x1);
            // TODO find center triangle 2
            int x2 = std::floor(((1-0.35355) * width_triangle_pixels) + bottom_left_x_pixels);
            int y2 = std::floor(((1-0.35355) * height_triangle_pixels) + bottom_left_y_pixels);
            // int y2 = std::ceil(bottom_left_y_pixels - ((1-0.35355) * height_triangle_pixels));
            cv::Vec3b val2 = img.at<cv::Vec3b>(y2, x2);

            int basee = (x + (y * x_max)) * 6;
            triangle_colors1[basee + 0] = val1[2] / 255.0;
            triangle_colors1[basee + 1] = val1[1] / 255.0;
            triangle_colors1[basee + 2] = val1[0] / 255.0;
            triangle_colors1[basee + 3] = val2[2] / 255.0;
            triangle_colors1[basee + 4] = val2[1] / 255.0;
            triangle_colors1[basee + 5] = val2[0] / 255.0;
        }
    }
}

void update_step_constant_color(int num_triangles_x, int num_triangles_y, const cv::Mat& img, const cv::Mat& saliency_map, const cv::Mat& edges, bool use_saliency, const float vertices[], float triangle_colors1[], float triangle_colors2[])
{
    int x_max = num_triangles_x;
    int y_max = num_triangles_y;
    float width_triangle_pixels = (float)img.cols / (float)x_max;
    float height_triangle_pixels = (float)img.rows / (float)y_max;
    float diff_x_pixels = width_triangle_pixels / points_tested_per_side;
    float diff_y_pixels = height_triangle_pixels / points_tested_per_side;

    for (int y = 0; y < y_max; y++)
    {
        for (int x = 0; x < x_max; x++)
        {
            // (x_max + 1) becuase the rightmost vertices are already tested in the previous box
            unsigned int bottom_left = (x_max + 1) * y + x;

            // top left = (0, 0), top right = (0, img.cols - 1), bottom left = (img.rows - 1, 0)
            float bottom_left_x_pixels = (float)(vertices[bottom_left * 6] * img.cols);
            float bottom_left_y_pixels = (float)(vertices[bottom_left * 6 + 1] * img.rows);

            float top_right_x_pixels = bottom_left_x_pixels + width_triangle_pixels;
            float top_right_y_pixels = bottom_left_y_pixels + height_triangle_pixels;

            // set of points for both triangles in the box
            std::vector<double> x_points_1;
            std::vector<double> y_points_1;
            std::vector<double> x_points_2;
            std::vector<double> y_points_2;
            // loop through all pixels in box; add edge pixels to set of points
            for (int y1 = std::floor(bottom_left_y_pixels); y1 < std::floor(top_right_y_pixels); ++y1)
            {
                for (int x1 = std::floor(bottom_left_x_pixels); x1 < std::floor(top_right_x_pixels); ++x1)
                {
                    int val = (int)edges.at<unsigned char>(y1, x1);
                    if (val > 0)
                    {
                        // add to vector // check triangle 1 or 2
                    }
                }
            }
            // if points.empty -> calculate straight line through points
            // else -> constant color
            float total_1[3] = {0.0, 0.0, 0.0};
            float total_2[3] = {0.0, 0.0, 0.0};
            if (x_points_1.size() < 2 || x_points_2.size() < 2)
            {
                get_average_color(bottom_left_x_pixels, bottom_left_y_pixels, diff_x_pixels, diff_y_pixels, img, saliency_map, use_saliency, total_1, total_2);
            }

            int basee = (x + (y * x_max)) * 6;
            if (x_points_1.size() < 2)
            {
                triangle_colors1[basee + 0] = total_1[2];
                triangle_colors1[basee + 1] = total_1[1];
                triangle_colors1[basee + 2] = total_1[0];

                triangle_colors2[basee + 0] = total_1[2];
                triangle_colors2[basee + 1] = total_1[1];
                triangle_colors2[basee + 2] = total_1[0];
            }
            else
            {
            }
            if (x_points_2.size() < 2)
            {
                triangle_colors1[basee + 3] = total_2[2];
                triangle_colors1[basee + 4] = total_2[1];
                triangle_colors1[basee + 5] = total_2[0];

                triangle_colors2[basee + 3] = total_2[2];
                triangle_colors2[basee + 4] = total_2[1];
                triangle_colors2[basee + 5] = total_2[0];
            }
            else
            {
            }
        }
    }
}

// -----------------------------------------------------------------------------
// TODO
void update_bicubic_variables(int num_triangles_x, int num_triangles_y, const cv::Mat& img, const cv::Mat& saliency_map, bool use_saliency, float vertices[], float vertex_colors[])
{
    update_vertex_colors(num_triangles_x, num_triangles_y, img, saliency_map, use_saliency, vertices, vertex_colors);
    // TODO
}


// -----------------------------------------------------------------------------
// tries to optimize the vertex colors for bilinear interpolation
// select points inside each triangle and given the error function (color_image - color_interpolation) * saliency_value
// using the error functions for points inside the triangles it tries to optimize the vertex colors such that the total error is minimize
// problems: the optimization return a lot of NaN and negative color values. only solution is to try and optimize it in another way
void update_bilinear_colors_opt(int num_triangles_x, int num_triangles_y, const cv::Mat& img, const cv::Mat& saliency_map, bool use_saliency, float vertices[], float vertex_colors[])
{
    int x_max = num_triangles_x;
    int y_max = num_triangles_y;
    float width_triangle_pixels = (float)img.cols / (float)x_max;
    float height_triangle_pixels = (float)img.rows / (float)y_max;
    float diff_x_pixels = width_triangle_pixels / points_tested_per_side;
    float diff_y_pixels = height_triangle_pixels / points_tested_per_side;
    float area_per_small_box = 1.0 / (float)((points_tested_per_side - 1) * (points_tested_per_side - 1)); // 1 / num_boxes
    float points_per_box = points_tested_per_side * points_tested_per_side;

    // solve Ax = b as closely as possible
    int num_equations = points_per_box * x_max * y_max;
    int num_vertices = (x_max + 1) * (y_max + 1);
    Eigen::MatrixXf matA_r(num_equations, num_vertices);
    Eigen::VectorXf b_r(num_equations);
    Eigen::MatrixXf matA_g(num_equations, num_vertices);
    Eigen::VectorXf b_g(num_equations);
    Eigen::MatrixXf matA_b(num_equations, num_vertices);
    Eigen::VectorXf b_b(num_equations);

    for (int y = 0; y < y_max; y++)
    {
        for (int x = 0; x < x_max; x++)
        {
            unsigned int bottom_left = (x_max + 1) * y + x;
            unsigned int bottom_right = bottom_left + 1;
            unsigned int top_left = bottom_left + x_max + 1;
            unsigned int top_right = top_left + 1;

            // int base_index = bottom_left * points_per_box;
            int base_index = (y * x_max + x) * points_per_box;

            float bottom_left_x_pixels = (float)(vertices[bottom_left * 6] * (img.cols - 1.0));
            float bottom_left_y_pixels = (float)img.rows - 1.0 - (float)(vertices[bottom_left * 6 + 1] * (img.rows - 1.0));
            for (int j = 0; j < points_tested_per_side; j++)
            {
                for (int i = 0; i < points_tested_per_side; i++)
                {
                    int x2 = std::floor((i * diff_x_pixels) + bottom_left_x_pixels);
                    int y2 = std::ceil(bottom_left_y_pixels - (j * diff_y_pixels));
                    cv::Vec3b val = img.at<cv::Vec3b>(y2, x2);
                    float saliency_val = saliency_map.at<float>(y2, x2);
                    saliency_val += saliency_bias;

                    float area_bl = i * j * area_per_small_box;
                    float area_br = (points_tested_per_side - i) * j * area_per_small_box;
                    float area_tl = i * (points_tested_per_side - j) * area_per_small_box;
                    float area_tr = (points_tested_per_side - i) * (points_tested_per_side - j) * area_per_small_box;

                    if (use_saliency)
                    {
                        val[0] *= saliency_val;
                        val[1] *= saliency_val;
                        val[2] *= saliency_val;

                        area_bl *= saliency_val;
                        area_br *= saliency_val;
                        area_tl *= saliency_val;
                        area_tr *= saliency_val;
                    }

                    int index = base_index + i + (j * points_tested_per_side);

                    b_r(index) = val[2];
                    b_g(index) = val[1];
                    b_b(index) = val[0];

                    matA_r(index, bottom_left) = area_tr;
                    matA_r(index, bottom_right) = area_tl;
                    matA_r(index, top_left) = area_br;
                    matA_r(index, top_right) = area_bl;

                    matA_g(index, bottom_left) = area_tr;
                    matA_g(index, bottom_right) = area_tl;
                    matA_g(index, top_left) = area_br;
                    matA_g(index, top_right) = area_bl;

                    matA_b(index, bottom_left) = area_tr;
                    matA_b(index, bottom_right) = area_tl;
                    matA_b(index, top_left) = area_br;
                    matA_b(index, top_right) = area_bl;
                }
            }
        }
    }
    // Eigen::VectorXf sol_r = matA_r.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_r); // really slow and small number problem
    // Eigen::VectorXf sol_r = matA_r.householderQr().solve(b_r);
    Eigen::VectorXf sol_r = matA_r.colPivHouseholderQr().solve(b_r) / 255.0f;
    Eigen::VectorXf sol_g = matA_g.colPivHouseholderQr().solve(b_g) / 255.0f;
    Eigen::VectorXf sol_b = matA_b.colPivHouseholderQr().solve(b_b) / 255.0f;
    // Eigen::VectorXf sol_r = matA_r.fullPivHouseholderQr().solve(b_r); // 0 problem
    // Eigen::VectorXf sol_r = (matA_r.transpose() * matA_r).ldlt().solve(matA_r.transpose() * b_r); // nan problem

    for (int y = 0; y < y_max; y++)
    {
        for (int x = 0; x < x_max; x++)
        {
            int base_index = (x + y * x_max) * 3;
            int index_color = (x + y * x_max);

            float color_r = sol_r(index_color);
            float color_g = sol_g(index_color);
            float color_b = sol_b(index_color);
            color_r = (std::isnan(color_r) || color_r < 0.0) ? 0.0f : color_r;
            color_g = (std::isnan(color_g) || color_g < 0.0) ? 0.0f : color_g;
            color_b = (std::isnan(color_b) || color_b < 0.0) ? 0.0f : color_b;

            // set colors
            vertex_colors[base_index + 0] = color_r;
            vertex_colors[base_index + 1] = color_g;
            vertex_colors[base_index + 2] = color_b;

            vertices[base_index * 2 + 3] = color_r;
            vertices[base_index * 2 + 4] = color_g;
            vertices[base_index * 2 + 5] = color_b;
        }
    }
}

// -----------------------------------------------------------------------------
// updates the array that defines where the vertices are
// defines the vertex position in such a way that it makes a square grid with the appropriate number of vertices given the number of triangles at each side (selected in imgui)
// (0, 0) is bottom left (1, 1) is top right
void update_vertex_buffer(int num_triangles_x, int num_triangles_y, float vertices[], const float vertex_colors[])
{
    // set/update shader parameters
    int x_max = num_triangles_x + 1;
    int y_max = num_triangles_y + 1;
    float x_step = 1.0f / ((float)x_max - 1.0f);
    float y_step = 1.0f / ((float)y_max - 1.0f);
    
    // set vertices buffer (vertices + color)
    for (int y = 0; y < y_max; y++)
    {
        for (int x = 0; x < x_max; x++)
        {
            int base_index = (x + y * x_max) * 6;
            vertices[base_index + 0] = x * x_step;
            vertices[base_index + 1] = y * y_step;
            vertices[base_index + 2] = 0.0f;
            // random colors
            vertices[base_index + 3] = vertex_colors[base_index / 2];
            vertices[base_index + 4] = vertex_colors[base_index / 2 + 1];
            vertices[base_index + 5] = vertex_colors[base_index / 2 + 2];
        }
    }
}

// -----------------------------------------------------------------------------
// updates the array that defines which vertices form a triangle
// triangle 0 is at bottom left and the last triangle is at the top right (numbering left to right and then bottom to top)
void update_index_buffer(int num_triangles_x, int num_triangles_y, unsigned int indices[])
{
    // set index buffer (array of the vertices that form a triangle)
    int x_max = num_triangles_x;
    int y_max = num_triangles_y;
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
}

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

// -----------------------------------------------------------------------------
// computes the saliency map from the image buffer given the selected saliency mode (from the imgui window)
// and replaces the saliency map buffer with it
void update_saliency_map(const cv::Mat& img, cv::Mat& saliency_map, int saliency_mode)
{
    // compute saliencymap
    switch (saliency_mode)
    {
        case saliency_method::fine_grained:
        {
            auto saliency_alg = cv::saliency::StaticSaliencyFineGrained::create();
            saliency_alg->computeSaliency(img, saliency_map);
            break;
        }
        case saliency_method::spectral_residual:
        {
            auto saliency_alg = cv::saliency::StaticSaliencySpectralResidual::create();
            saliency_alg->computeSaliency(img, saliency_map);
            break;
        }
    }
    // cv::imshow("saliency map", saliency_map);
    // cv::waitKey(0);
    // std::cout << saliency_map << std::endl;
}

void get_edges(const cv::Mat& img, cv::Mat& edges, int low_threshold)
{
    cv::Mat img_gray;
    const int ratios = 3;
    const int kernel_size = 3;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::blur(img_gray, edges, cv::Size(3, 3));
    cv::Canny(edges, edges, low_threshold, low_threshold * ratios, kernel_size);
    // cv::imshow("test", edges);
    // cv::waitKey(0);
    // cv::destroyAllWindows();
}

// -----------------------------------------------------------------------------
// uses opencv to load an image to the image buffer
void load_picture(cv::Mat& img, const std::string file_name)
{
    // load image
    std::string image_path = cv::samples::findFile(file_name);
    img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cout << "error loading image: " << image_path << std::endl;
    }

    // std::cout << img << std::endl;
}

// -----------------------------------------------------------------------------
// create the glfw window and the opengl context within the window
GLFWwindow* glfw_setup()
{
    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 0;

    // GL 3.3 + GLSL 330
    const char* glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_FALSE);

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(1600, 900, "Coloring of Triangulations of an Image", NULL, NULL);
    if (window == NULL)
    {
        glfwTerminate();
        return 0;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize OpenGL loader
    bool err = gladLoadGL(glfwGetProcAddress) == 0; // glad2 recommend using the windowing library loader instead of the (optionally) bundled one.
    if (err)
    {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        return 0;
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

    return window;
}

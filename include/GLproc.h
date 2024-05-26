#include <vector>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

using std::cout;
using std::vector;
using namespace cv;

class GLproc {
private:
    vector<vector<Point>> &ver, vector<vector<Point>> &nver
    // vector<GLfloat> vertices;
    // vector<GLuint> indices;
public:
    GLproc(Mat &img, vector<vector<Point>> &ver, vector<vector<Point>> &nver);

    static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
    }

    void processInput(GLFWwindow *window) {
        if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }
    }

    void getData(vector<vector<Point>> &ver, vector<GLfloat> &vertices, vector<GLuint> &indices);
};

GLproc::GLproc(Mat &img, vector<vector<Point>> &ver, vector<vector<Point>> &nver):
ver(ver), nver(nver) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "Source Mesh", NULL, NULL);
    if (window == NULL) {
        cout << "Failed to create GLFW window" << '\n';
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        cout << "Failed to initialize GLAD" << '\n';
        return;
    }

    glViewport(0, 0, 800, 600);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // 生成并绑定 VAO
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    // 生成并绑定 VBO
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    // 生成并绑定 EBO
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

    while(!glfwWindowShouldClose(window)) {
        processInput(window);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    return;
}

void GLproc::getData(vector<vector<Point>> &ver, vector<GLfloat> &vertices, vector<GLuint> &indices) {

}
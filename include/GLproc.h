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
    vector<vector<Point>> &ver, &nver;
    // vector<GLfloat> vertices;
    // vector<GLuint> indices;
    
    const char *vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        void main() {
           gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
        }
    )";

    /*
    const char* fragmentShaderSource = R"(
        #version 330 core
        out vec4 FragColor;
        in vec2 TexCoord;
        uniform sampler2D texture1;
        void main() {
            FragColor = texture(texture1, TexCoord);
        }
    )";
    */
    
    const char* fragmentShaderSource = R"(
        #version 330 core
        out vec4 FragColor;
        void main() {
            FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
        } 
    )";
public:
    GLproc(Mat &img, vector<vector<Point>> &ver, vector<vector<Point>> &nver);

    static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
    void processInput(GLFWwindow *window);

    void getData(Mat &img, vector<vector<Point>> &ver, vector<GLfloat> &vertices, vector<GLuint> &indices);
    GLuint compileShader(GLenum type, const char* source);
    GLuint createShaderProgram();
};

GLproc::GLproc(Mat &img, vector<vector<Point>> &ver, vector<vector<Point>> &nver):
ver(ver), nver(nver) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    const unsigned int SCR_WIDTH = img.cols;
    const unsigned int SCR_HEIGHT = img.rows;

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Source Mesh", NULL, NULL);
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

    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    vector<GLfloat> vertices;
    vector<GLuint> indices;
    getData(img, ver, vertices, indices);

    GLuint VAO, VBO, EBO;
    // 生成并绑定 VAO
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    // 生成并绑定 VBO
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // 此处尝试一下选择 STATIC_DRAW 还是 DYNAMIC_DRAW
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat), &vertices[0], GL_STATIC_DRAW);

    // 生成并绑定 EBO
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), &indices[0], GL_STATIC_DRAW);

    // 第四个参数代表是否需要标准化
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    GLuint shaderProgram = createShaderProgram();

    while(!glfwWindowShouldClose(window)) {
        processInput(window);
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);

        glBindVertexArray(VAO);
        glDrawElements(GL_LINES, indices.size(), GL_UNSIGNED_INT, 0);


        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);
    glfwTerminate();
    return;
}

void GLproc::framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void GLproc::processInput(GLFWwindow *window) {
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

void GLproc::getData(Mat &img, vector<vector<Point>> &ver, vector<GLfloat> &vertices, vector<GLuint> &indices) {
    for (int i = 0, cnt = 0; i < ver.size(); i++) {
        for (int j = 0; j < ver[i].size(); j++, cnt++) {
            Point2f cur = ver[i][j];
            cur.x /= (float)img.cols;
            cur.y = img.rows - cur.y - 1;
            cur.y /= (float)img.rows;
            cur *= 2.0f;
            cur -= Point2f(1.0f, 1.0f);
            vertices.push_back(cur.x);
            vertices.push_back(cur.y);
            vertices.push_back(0.0f);
            if (i) {
                indices.push_back(cnt);
                indices.push_back(cnt - ver[i].size());
            }
            if (j) {
                indices.push_back(cnt);
                indices.push_back(cnt - 1);
            }
        }
    }
}

GLuint GLproc::compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    return shader;
}

GLuint GLproc::createShaderProgram() {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}
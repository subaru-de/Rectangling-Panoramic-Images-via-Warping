#include <string>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <Rectangling.h>

using std::cout;
using std::string;
using namespace cv;

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv, "{@input| ../img/img5.jpg |}");
    string filename = parser.get<string>("@input");
    if (filename.empty()) {
        std::cout << "\nDurn, empty filename" << std::endl;
        return 1;
    }
    Mat image = cv::imread(cv::samples::findFile(filename), cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cout << "\n Durn, couldn't read image filename " << filename << std::endl;
        return 1;
    }
    Rectangling rc(image);
    return 0;
}
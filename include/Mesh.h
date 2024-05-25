#include <vector>
#include <string>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

class Mesh {
private:
    vector<vector<Point2d>> ver;
public:
    Mesh(Mat &img);
};

Mesh::Mesh(Mat &img) {
    const double totVer = 400;
    double scale = sqrt(img.rows * img.cols * 1.0 / totVer);
    
    // (x / scale) * (y / scale) = totVer
    int totVerR = img.rows / scale;
    if (!totVerR) totVerR++;
    int totVerC = img.cols / scale;
    if (!totVerC) totVerC++;
    ver.resize(totVerR, vector<Point2d>(totVerC, {0, 0}));

    double disR = 1.0 * img.rows / (totVerR + 1.0);
    double disC = 1.0 * img.cols / (totVerC + 1.0);

    
    /* -------- 防止 x, y 重名 -------- */ {
        double x = disR, y = disC;
        for (int i = 1; i <= totVerR; i++, x += disR) {
            for (int j = 1; j <= totVerC; j++, y += disC) {
                ver[i][j] = Point2d(x, y);
            }
        }
    }
}
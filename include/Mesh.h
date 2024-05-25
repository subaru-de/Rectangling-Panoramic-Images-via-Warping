#include <vector>
#include <string>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

class Mesh {
private:
public:
    vector<vector<Point2d>> ver;
    Mesh(Mat &img);
};

Mesh::Mesh(Mat &img) {
    const double totVer = 400;
    double scale = sqrt(img.rows * img.cols * 1.0 / totVer);
    
    // (x / scale) * (y / scale) = totVer
    int totVerR = img.rows / scale;
    int totVerC = img.cols / scale;

    double disR = 1.0 * img.rows / totVerR;
    double disC = 1.0 * img.cols / totVerC;

    // 加上 i == 0 || j == 0
    totVerR++;
    totVerC++;
    ver.resize(totVerR, vector<Point2d>(totVerC, {0, 0}));

    cout << totVerR << ' ' << totVerC << ' ' << disR << ' ' << disC << '\n';

    
    /* -------- 防止 x, y 重名 -------- */ {
        double x = 0, y = 0;
        for (int i = 0; i < totVerR; i++, x += disR) {
            y = 0;
            for (int j = 0; j < totVerC; j++, y += disC) {
                ver[i][j] = Point2d(y, x);
            }
        }
    }

    for (int i = 0; i < ver.size(); i++) {
        for (int j = 0; j < ver[i].size(); j++) {
            cout << ver[i][j] << '\t';
        }
        cout << '\n';
    }
}
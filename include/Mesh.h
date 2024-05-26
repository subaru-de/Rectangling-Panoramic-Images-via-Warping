#include <vector>
#include <string>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

class Mesh {
private:
    vector<vector<Point>> ver, nver;
public:
    Mesh(Mat &img);
    void putMesh(Mat &img);
    void displace(Mat &dispV, Mat &dispH);
    void callGL(Mat &img);
};

Mesh::Mesh(Mat &img) {
    const double totVer = 400;
    double scale = sqrt(img.rows * img.cols * 1.0 / totVer);
    
    // (x / scale) * (y / scale) = totVer
    int totVerR = img.rows / scale;
    int totVerC = img.cols / scale;

    // 保证最后一行最后一列不出界，所以要减一
    double disR = 1.0 * (img.rows - 1.0) / totVerR;
    double disC = 1.0 * (img.cols - 1.0) / totVerC;

    // 加上 i == 0 || j == 0
    totVerR++;
    totVerC++;
    ver.resize(totVerR, vector<Point>(totVerC, {0, 0}));

    cout << totVerR << ' ' << totVerC << ' ' << disR << ' ' << disC << '\n';

    
    /* -------- 防止 x, y 重名 -------- */ {
        double x = 0, y = 0;
        for (int i = 0; i < totVerR; i++, x += disR) {
            y = 0;
            for (int j = 0; j < totVerC; j++, y += disC) {
                ver[i][j] = Point(y, x);
                nver[i][j] = ver[i][j];
            }
        }
    }

    // for (int i = 0; i < ver.size(); i++) {
    //     for (int j = 0; j < ver[i].size(); j++) {
    //         cout << ver[i][j] << '\t';
    //     }
    //     cout << '\n';
    // }
}

void Mesh::putMesh(Mat &img) {
    // put mesh
    for (int i = 0; i < ver.size(); i++) {
        for (int j = 0; j < ver[i].size(); j++) {
            if (i) {
                line(img, ver[i][j], ver[i - 1][j], Green, 1);
            }
            if (j) {
                line(img, ver[i][j], ver[i][j - 1], Green, 1);
            }
        }
    }
}

void Mesh::displace(Mat &dispV, Mat &dispH) {
    for (int i = 0; i < ver.size(); i++) {
        for (int j = 0; j < ver[i].size(); j++) {
            // x 对应 width
            Point de = {0, 0};
            de.x += dispV.at<int>(ver[i][j]);
            de.y += dispH.at<int>(ver[i][j]);
            ver[i][j] += de;
            // cout << de << ' ' << ver[i][j] << ' ';
        }
        cout << '\n';
    }
}

void Mesh::callGL(Mat &img) {
    GLproc glp(img, ver, nver);
}
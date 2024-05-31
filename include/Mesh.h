// #ifndef Mesh
// #define Mesh
#include <vector>
#include <string>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <GLproc.h>
#include <Energy.h>

class Mesh {
private:
    Mat &img, resImg;
    vector<vector<Point>> ver, nver;
public:
    Mesh(Mat &img, Mat resImg);
    void putMesh(Mat &img, bool initial = 1);
    void displace(Mat &dispV, Mat &dispH);
    void callGL();
    void callEnergy();
    pair<double, double> getScale();
    void setRes(Mat resImg) { this -> resImg = resImg; }
    Mat getRes() { return resImg; }
};

Mesh::Mesh(Mat &img, Mat resImg):
img(img) {
    resImg.copyTo(this -> resImg);
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
    nver.resize(totVerR, vector<Point>(totVerC, {0, 0}));

    // cout << totVerR << ' ' << totVerC << ' ' << disR << ' ' << disC << '\n';

    
    /* -------- 防止 x, y 重名 -------- */ {
        double x = 0, y = 0;
        for (int i = 0; i < totVerR; i++, x += disR) {
            y = 0;
            for (int j = 0; j < totVerC; j++, y += disC) {
                ver[i][j] = Point(y + 0.5, x + 0.5);
                nver[i][j] = ver[i][j];
            }
        }
    }
    // cout << "------------- first of all ------------\n";
    // for (int i = 0; i < ver.size(); i++) {
    //     for (int j = 0; j < ver[i].size(); j++) {
    //         cout << nver[i][j] << '\t';
    //     }
    //     cout << '\n';
    // }
}

void Mesh::putMesh(Mat &img, bool showVer) {
    // put mesh
    vecvecP &v = showVer ? ver : nver;
    for (int i = 0; i < v.size(); i++) {
        for (int j = 0; j < v[i].size(); j++) {
            if (i) {
                line(img, v[i][j], v[i - 1][j], Green, 1);
            }
            if (j) {
                line(img, v[i][j], v[i][j - 1], Green, 1);
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
        // cout << '\n';
    }
}

void Mesh::callGL() {
    GLproc glp(img, resImg, ver, nver);
}

void Mesh::callEnergy() {
    // cout << "--------------- before ---------------\n";
    // for (int i = 0; i < nver.size(); i++) {
    //     for (int j = 0; j < nver[i].size(); j++) {
    //         cout << nver[i][j] << ' ';
    //     } cout << '\n';
    // } cout << '\n';
    Energy energy(img, resImg, ver, nver);
}

pair<double, double> Mesh::getScale() {
    double sx = 0.0, sy = 0.0, cnt = 0.0, tmp = 0.0;
    for (int i = 1; i < ver.size(); i++) {
        for (int j = 1; j < ver[i].size(); j++, cnt++) {
            tmp = max(nver[i][j].x, nver[i - 1][j].x) - min(nver[i][j - 1].x, nver[i - 1][j - 1].x);
            tmp /= max(ver[i][j].x, ver[i - 1][j].x) - min(ver[i][j - 1].x, ver[i - 1][j - 1].x);
            sx += tmp;
            tmp = max(nver[i][j].y, nver[i][j - 1].y) - min(nver[i - 1][j].y, nver[i - 1][j - 1].y);
            tmp /= max(ver[i][j].y, ver[i][j - 1].y) - min(ver[i - 1][j].y, ver[i - 1][j - 1].y);
            sy += tmp;
        }
    }
    sx /= cnt;
    sy /= cnt;
    return std::make_pair(sx, sy);
}

// #endif
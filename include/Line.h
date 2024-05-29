#include <string>
#include <fstream>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
// #include <eigen3/Eigen/Core>
// #include <eigen3/Eigen/Sparse>
// #include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigen>
#include <lsd.h>

using std::pair;
using std::cout;
using std::string;
using std::vector;
using namespace cv;
using namespace Eigen;

typedef pair<Point2d, Point2d> lin;
typedef vector<vector<Point>> vecvecP;

class Line {
private:
    Mat &img;
    vecvecP &ver;
public:
    Line(Mat &img, vecvecP &ver);
    double cross(Point2d A, Point2d B, Point2d C);
    bool onSegment(Point2d A, Point2d B, Point2d C);
    bool doIntersect(lin l1, lin l2);
    Point2d getIntersection(lin l1, lin l2);
};

Line::Line(Mat &img, vecvecP &ver):
img(img), ver(ver) {
    Mat grayImg;
    cvtColor(img, grayImg, COLOR_BGR2GRAY);
    imshow("gray", grayImg);
    waitKey(0);
    double* lsdImg = new double[img.rows * img.cols];
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            lsdImg[i * img.cols + j] = grayImg.at<uchar>(i, j);
        }
    }
    int cntLines;
    double* lsdLines = lsd(&cntLines, lsdImg, img.cols, img.rows);

    // for (int i = 0; i < cntLines; i++) {
    //     for (int j = 0; j < 7; j++) {
    //         cout << lsdLines[i * 7 + j] << ' ';
    //     }
    //     cout << "\n";
    // }

    Mat lineImg;
    img.copyTo(lineImg);
    for (int i = 0; i < cntLines; i++) {
        line(lineImg, Point(lsdLines[i * 7], lsdLines[i * 7 + 1]), Point(lsdLines[i * 7 + 2], lsdLines[i * 7 + 3]), Scalar(0, 0, 255), 1);
    }
    imshow("lines", lineImg);
    waitKey(0);

    vector<vector<vector<lin>>> lines;
    lines.resize(ver.size() - 1, vector<vector<lin>>(ver[0].size() - 1, vector<lin>()));
    for (int i = 1; i < ver.size(); i++) {
        for (int j = 1; j < ver[i].size(); j++) {
            vector<Point2d> contour;
            Point2d tl = ver[i - 1][j - 1];
            Point2d tr = ver[i - 1][j];
            Point2d bl = ver[i][j - 1];
            Point2d br = ver[i][j];
            contour.push_back(tl);
            contour.push_back(tr);
            contour.push_back(bl);
            contour.push_back(br);
            for (int k = 0; k < cntLines; k++) {
                double flag1 = pointPolygonTest(contour, Point2d(lsdLines[k * 7], lsdLines[k * 7 + 1]), 0);
                double flag2 = pointPolygonTest(contour, Point2d(lsdLines[k * 7 + 2], lsdLines[k * 7 + 3]), 0);
                
            }
        }
    }
}

// AB times AC
double Line::cross(Point2d A, Point2d B, Point2d C) {
    return (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
}

bool Line::onSegment(Point2d A, Point2d B, Point2d C) {
    return (min(A.x, B.x) <= C.x && C.x <= max(A.x, B.x)) && (min(A.y, B.y) <= C.y && C.y <= max(A.y, B.y));
}

bool Line::doIntersect(lin l1, lin l2) {
    double d1 = cross(l1.first, l1.second, l2.first);
    double d2 = cross(l1.first, l1.second, l2.second);
    double d3 = cross(l2.first, l2.second, l1.first);
    double d4 = cross(l2.first, l2.second, l1.second);
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
        return true;
    }
    if (d1 == 0 && onSegment(l1.first, l1.second, l2.first)) return true;
    if (d2 == 0 && onSegment(l1.first, l1.second, l2.second)) return true;
    if (d3 == 0 && onSegment(l2.first, l2.second, l1.first)) return true;
    if (d4 == 0 && onSegment(l2.first, l2.second, l1.second)) return true;
    return false;
}

Point2d Line::getIntersection(lin l1, lin l2) {
    // ax + by = c
    double a1 = l1.second.y - l1.first.y;
    double b1 = l1.first.x - l1.second.x;
    double c1 = a1 * l1.first.x + b1 * l1.first.y;

    double a2 = l2.second.y - l2.first.y;
    double b2 = l2.first.x - l2.second.x;
    double c2 = a2 * l2.first.x + b2 * l2.first.y;

    // 行列式
    double determinant = a1 * b2 - a2 * b1;
    Point2d intersection;
    intersection.x = (b2 * c1 - b1 * c2) / determinant;
    intersection.y = (a1 * c2 - a2 * c1) / determinant;
    return intersection;
}
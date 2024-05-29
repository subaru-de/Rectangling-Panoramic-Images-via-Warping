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

double dis(Point2d A, Point2d B) {
    Point2d de = B - A;
    return sqrt(de.x * de.x + de.y * de.y);
}

class Line {
private:
    Mat &img;
    vecvecP &ver;
    const double eps = 1e-3;
public:
    Line(Mat &img, vecvecP &ver, vector<vector<vector<lin>>> &lines, vector<vector<vector<MatrixXd>>> &F);
    double cross(Point2d A, Point2d B, Point2d C);
    bool onSegment(Point2d A, Point2d B, Point2d C);
    bool doIntersect(lin l1, lin l2);
    Point2d getIntersection(lin l1, lin l2);
    void checkLines(vector<vector<vector<lin>>> &lines);
    void getF(MatrixXd &F, vector<Point2f> &contour, lin l);
    void processSeg(Point2d &s, Point2d &t);
};

Line::Line(Mat &img, vecvecP &ver, vector<vector<vector<lin>>> &lines, vector<vector<vector<MatrixXd>>> &F):
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
    imshow("lines after LSD", lineImg);
    waitKey(0);

    lines.resize(ver.size(), vector<vector<lin>>(ver[0].size(), vector<lin>()));
    F.resize(ver.size(), vector<vector<MatrixXd>>(ver[0].size(), vector<MatrixXd>()));

    // const double eps = 1e-3;
    for (int i = 1; i < ver.size(); i++) {
        // cout << i << '\n';
        vector<Point2f> contour;
        Point2d tl, tr, br, bl, s, t;
        for (int j = 1; j < ver[i].size(); j++) {
            // cout << '\t' << j << '\n';
            tl = ver[i - 1][j - 1];
            tr = ver[i - 1][j];
            br = ver[i][j];
            bl = ver[i][j - 1];
            contour.clear();
            contour.push_back(tl);
            contour.push_back(tr);
            contour.push_back(br);
            contour.push_back(bl);

            // cout << "\t\t" << j << '\n';
            for (int k = 0; k < cntLines; k++) {
                s = Point2d(lsdLines[k * 7], lsdLines[k * 7 + 1]);
                t = Point2d(lsdLines[k * 7 + 2], lsdLines[k * 7 + 3]);
                double flag1 = pointPolygonTest(contour, s, 1);
                double flag2 = pointPolygonTest(contour, t, 1);
                // 都在内部
                if (flag1 >= eps && flag2 >= eps) {
                    processSeg(s, t);
                    if (dis(s, t) > eps) {
                        lines[i][j].push_back(lin(s, t));
                        MatrixXd M;
                        getF(M, contour, lines[i][j].back());
                        F[i][j].push_back(M);
                    }
                }
                else if (flag1 >= eps || flag2 >= eps) {
                    Point2d ins;
                    if (doIntersect(lin(tl, tr), lin(s, t))) {
                        ins = getIntersection(lin(tl, tr), lin(s, t));
                        processSeg((flag1 >= eps ? s : t), ins);
                        if (dis((flag1 >= eps ? s : t), ins) > eps) {
                            lines[i][j].push_back(lin((flag1 >= eps ? s : t), ins));
                            MatrixXd M;
                            getF(M, contour, lines[i][j].back());
                            F[i][j].push_back(M);
                        }
                    }
                    else if (doIntersect(lin(tr, br), lin(s, t))) {
                        ins = getIntersection(lin(tr, br), lin(s, t));
                        processSeg((flag1 >= eps ? s : t), ins);
                        if (dis((flag1 >= eps ? s : t), ins) > eps) {
                            lines[i][j].push_back(lin(flag1 >= eps ? s : t, ins));
                            MatrixXd M;
                            getF(M, contour, lines[i][j].back());
                            F[i][j].push_back(M);
                        }
                    }
                    else if (doIntersect(lin(bl, br), lin(s, t))) {
                        ins = getIntersection(lin(bl, br), lin(s, t));
                        processSeg((flag1 >= eps ? s : t), ins);
                        if (dis((flag1 >= eps ? s : t), ins) > eps) {
                            lines[i][j].push_back(lin(flag1 >= eps ? s : t, ins));
                            MatrixXd M;
                            getF(M, contour, lines[i][j].back());
                            F[i][j].push_back(M);
                        }
                    }
                    else if (doIntersect(lin(tl, bl), lin(s, t))) {
                        ins = getIntersection(lin(tl, bl), lin(s, t));
                        processSeg((flag1 >= eps ? s : t), ins);
                        if (dis((flag1 >= eps ? s : t), ins) > eps) {
                            lines[i][j].push_back(lin(flag1 >= eps ? s : t, ins));
                            MatrixXd M;
                            getF(M, contour, lines[i][j].back());
                            F[i][j].push_back(M);
                        }
                    }
                    // cout << pointPolygonTest(contour, flag1 >= eps ? s : t, 1) << ' ' << pointPolygonTest(contour, ins, 1) << '\n';
                    // cout << "line:\n" << tl << ' ' << tr << '\n' << bl << ' ' << br << '\n' <<
                    //     (flag1 >= eps ? s : t) << ' ' << ins << '\n';
                    assert(pointPolygonTest(contour, flag1 >= eps ? s : t, 1) >= -eps);
                    assert(pointPolygonTest(contour, ins, 1) >= -eps);
                }
                else {
                    Point2d ins1(-1, -1), ins2(-1, -1), de;
                    if (doIntersect(lin(tl, tr), lin(s, t))) {
                        (ins1.x == -1 ? ins1 : ins2) = getIntersection(lin(tl, tr), lin(s, t));
                    }
                    if (doIntersect(lin(tr, br), lin(s, t))) {
                        (ins1.x == -1 ? ins1 : ins2) = getIntersection(lin(tr, br), lin(s, t));
                    }
                    if (doIntersect(lin(bl, br), lin(s, t))) {
                        (ins1.x == -1 ? ins1 : ins2) = getIntersection(lin(bl, br), lin(s, t));
                    }
                    if (doIntersect(lin(tl, bl), lin(s, t))) {
                        (ins1.x == -1 ? ins1 : ins2) = getIntersection(lin(tl, bl), lin(s, t));
                    }
                    if (ins1.x != -1 && ins2.x != -1) {
                        processSeg(ins1, ins2);
                        if (dis(ins1, ins2) > eps) {
                            lines[i][j].push_back(lin(ins1, ins2));
                            MatrixXd M;
                            getF(M, contour, lines[i][j].back());
                            F[i][j].push_back(M);
                        }
                        // if (pointPolygonTest(contour, ins1, 1) < -eps || pointPolygonTest(contour, ins2, 0) < -eps) {
                        //     cout
                        //     << "line:\n"
                        //     << tl << ' ' << tr << '\n'
                        //     << bl << ' ' << br << '\n'
                        //     << ins1 << ' ' << ins2 << '\n'
                        //     << pointPolygonTest(contour, ins1, 1) << ' ' << pointPolygonTest(contour, ins2, 0) << '\n';
                        // }
                        assert(pointPolygonTest(contour, ins1, 1) >= -eps);
                        assert(pointPolygonTest(contour, ins2, 1) >= -eps);
                    }
                }
            }
        }
    }

    checkLines(lines);
    delete[] lsdImg;
    delete[] lsdLines;
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

void Line::checkLines(vector<vector<vector<lin>>> &lines) {
    Mat lineImg;
    img.copyTo(lineImg);
    for (int i = 0; i < lines.size(); i++) {
        for (int j = 0; j < lines[i].size(); j++) {
            if (i) {
                line(lineImg, ver[i][j], ver[i - 1][j], Vec3b(0, 255, 0), 1);
            }
            if (j) {
                line(lineImg, ver[i][j], ver[i][j - 1], Vec3b(0, 255, 0), 1);
            }
            Scalar randColor = Scalar(rand() % 256, rand() % 256, rand() % 256);
            for (int k = 0; k < lines[i][j].size(); k++) {
                line(lineImg, Point(lines[i][j][k].first + Point2d(0.5, 0.5)), Point(lines[i][j][k].second + Point2d(0.5, 0.5)), randColor, 1);
            }
        }
    }
    imshow("lines after process", lineImg);
    waitKey(0);
}

void Line::getF(MatrixXd &F, vector<Point2f> &contour, lin l) {
    double u[2], v[2];
    for (int i = 0; i < 2; i++) {
        Point2d p0 = contour[1] - contour[0];
        Point2d p1 = contour[3] - contour[0];
        Point2d p2 = contour[0] - contour[1] + contour[2] - contour[3];
        Point2d p3 = (i ? l.second : l.first) - Point2d(contour[0]);
        double A = p2.x * p1.y - p2.y * p1.x;
        double B = p0.x * p1.y - p0.y * p1.x + p3.x * p2.y - p3.y * p2.x;
        double C = p3.x * p0.y - p3.y * p0.x;
        if (A < 1e-3) {
            v[i] = -C / B;
        }
        else {
            double delta = sqrt(B * B - 4 * A * C);
            double tmp = (-B - delta) / A / 2;
            if (tmp >= -1e-6 && tmp <= 1 + 1e-6) {
                v[i] = tmp;
            }
            else {
                v[i] = (-B + delta) / A / 2;
            }
        }
        u[i] = (p3.x - p1.x * v[i]) / (p0.x + p2.x * v[i]);
    }
    // for (auto it : contour) {
    //     cout << it << ' ';
    // } cout << '\n';
    // cout << l.first << ' ' << l.second << '\n';
    // cout << u[0] << ' ' << u[1] << ' ' << v[0] << ' ' << v[1] << '\n';
    // for (auto it : contour) cout << it << ' '; cout << l.first << ' ' << l.second << '\n';
    assert(!isnan(u[0]) && u[0] >= -1e-3 && u[0] <= 1 + 1e-3);
    assert(!isnan(v[0]) && v[0] >= -1e-3 && v[0] <= 1 + 1e-3);
    assert(!isnan(u[1]) && u[1] >= -1e-3 && u[1] <= 1 + 1e-3);
    assert(!isnan(v[1]) && v[1] >= -1e-3 && v[1] <= 1 + 1e-3);
    F.resize(4, 8);
    F <<
        (1 - u[0]) * (1 - v[0]), 0, u[0] * (1 - v[0]), 0, v[0] * (1 - u[0]), 0, u[0] * v[0], 0,
        0, (1 - u[0]) * (1 - v[0]), 0, u[0] * (1 - v[0]), 0, v[0] * (1 - u[0]), 0, u[0] * v[0],
        (1 - u[1]) * (1 - v[1]), 0, u[1] * (1 - v[1]), 0, v[1] * (1 - u[1]), 0, u[1] * v[1], 0,
        0, (1 - u[1]) * (1 - v[1]), 0, u[1] * (1 - v[1]), 0, v[1] * (1 - u[1]), 0, u[1] * v[1];
    // cout << F << '\n';
    // waitKey(0);
    MatrixXd minus(2, 4);
    minus << -1, 0, 1, 0, 0, -1, 0, 1;
    minus /= dis(l.first, l.second);
    F = minus * F;
    return;
}

void Line::processSeg(Point2d &s, Point2d &t) {
    if (s.x > t.x) {
        s.x -= 0.1 * (s.x - t.x);
        t.x += 0.1 * (s.x - t.x);
    }
    else {
        s.x += 0.1 * (t.x - s.x);
        t.x -= 0.1 * (t.x - s.x);
    }
    if (s.y > t.y) {
        s.y -= 0.1 * (s.y - t.y);
        t.y += 0.1 * (s.y - t.y);
    }
    else {
        s.y += 0.1 * (t.y - s.y);
        t.y -= 0.1 * (t.y - s.y);
    }
}
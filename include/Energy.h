// #ifndef Energy
// #define Energy
#include <string>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
// #include <eigen3/Eigen/Core>
// #include <eigen3/Eigen/Sparse>
// #include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigen>

using std::cout;
using std::string;
using namespace cv;
using namespace Eigen;

class Energy {
private:
    const double lambdaLine = 100.0;
    const double lambdaBound = 1e8;
    Mat &img;
    vecvecP &ver, &nver;
    const int MAXN = 20 * 20 * 8;
    SparseMatrix<double> A, B, C;
    MatrixXd V, Y;
public:
    Energy(Mat &img, vecvecP &ver, vecvecP &nver);
    double getEnergy();
    // 为了减少一次遍历，在 shapeTerm 函数中同时执行 getV 与 getA
    double shapeTerm();
    // 在 lineTerm 中执行 getC
    double lineTerm();
    // 在 boundTerm 中执行 getB
    double boundTerm();
};

Energy::Energy(Mat &img, vecvecP &ver, vecvecP &nver):
img(img), ver(ver), nver(nver),
A(MAXN, MAXN), B(MAXN, MAXN), C(MAXN, MAXN),
V(0, 1), Y(0, 1) {}

double Energy::getEnergy() {
    double E = 0.0;
    E += shapeTerm();
    E += lambdaBound * boundTerm();
    // E += lambdaLine * lineTerm();
    return E;
}

double Energy::shapeTerm() {
    // Mat A, AT, ATA, invATA, V, I;
    // I = Mat::eye(8, 8, CV_64F);
    double E = 0;
    int cnt = 0;
    for (int i = 0; i < ver.size(); i++) {
        for (int j = 0; j < ver[i].size(); j++) {
            if (!i || !j) continue;
            MatrixXd Aq(8, 4), Vq(8, 1);
            Aq <<
                ver[i - 1][j - 1].x, -ver[i - 1][j - 1].y, 1, 0,
                ver[i - 1][j - 1].y, ver[i - 1][j - 1].x, 0, 1,
                ver[i - 1][j].x, -ver[i - 1][j].y, 1, 0,
                ver[i - 1][j].y, ver[i - 1][j].x, 0, 1,
                ver[i][j - 1].x, -ver[i][j - 1].y, 1, 0,
                ver[i][j - 1].y, ver[i][j - 1].x, 0, 1,
                ver[i][j].x, -ver[i][j].y, 1, 0,
                ver[i][j].y, ver[i][j].x, 0, 1;
            Vq <<
                nver[i - 1][j - 1].x, nver[i - 1][j - 1].y,
                nver[i - 1][j].x, nver[i - 1][j].y,
                nver[i][j - 1].x, nver[i][j - 1].y,
                nver[i][j].x, nver[i][j].y;
            
            Aq = (Aq * (Aq.transpose() * Aq).inverse() * Aq.transpose() - MatrixXd::Identity(8, 8));
            VectorXd tmp = Aq * Vq;
            E += tmp.dot(tmp);

            Aq = Aq.transpose() * Aq;
            // A.resize(A.rows() + Aq.rows(), A.cols() + Aq.cols());
            // A.block(A.rows() - Aq.rows(), A.cols() - Aq.cols()) << Aq;
            for (int u = 0; u < 8; u++) {
                for (int v = 0; v < 8; v++) {
                    A.insert(cnt * 8 + u, cnt * 8 + v) = Aq(u, v);
                }
            }
            V.resize(V.rows() + Vq.rows(), V.cols());
            V << Vq;
            Y.resize(Y.rows() + Aq.rows(), Y.cols());
            Y << MatrixXd::Zero(8, 1);

            cnt++;
        }
    }
    E /= cnt;
    return E;
}

double Energy::boundTerm() {
    // 这里处理 B 矩阵与 Y
    double E = 0;
    for (int i = 0; i < nver.size(); i++) {
        for (int j = 0; j < nver[i].size(); j++) {
            if (!i || !j) continue;
            MatrixXd Bq(8, 8);
            Bq <<
                nver[i - 1][j - 1].x, -nver[i - 1][j - 1].y, 1, 0,
                nver[i - 1][j - 1].y, nver[i - 1][j - 1].x, 0, 1,
                nver[i - 1][j].x, -nver[i - 1][j].y, 1, 0,
                nver[i - 1][j].y, nver[i - 1][j].x, 0, 1,
                nver[i][j - 1].x, -nver[i][j - 1].y, 1, 0,
                nver[i][j - 1].y, nver[i][j - 1].x, 0, 1,
                nver[i][j].x, -nver[i][j].y, 1, 0,
                nver[i][j].y, nver[i][j].x, 0, 1;
        }
    }
    // for (int j = 0; j < nver[0].size(); j++) {
    //     E += nver[0][j].y * nver[0][j].y;
    //     double de = (nver.back())[j].y - img.rows + 1;
    //     E += de * de;
    // }
    // for (int i = 0; i < nver.size(); i++) {
    //     E += nver[i][0].x * nver[i][0].x;
    //     double de = nver[i].back().x - img.cols + 1;
    //     E += de * de;
    // }
    return E;
}

// #endif
#include <string>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <Rectangling.h>
#include <Eigen/Core>
#include <Eigen/Sparse>

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
    SparseMatrix<double> A, B, C, V(0, 1), Y(0, 1);
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
img(img), ver(ver), nver(nver) {}

double Energy::getEnergy() {
    double E = 0.0;
    E += shapeTerm();
    E += lambdaBound * boundTerm();
    E += lambdaLine * lineTerm();
    return E;
}

double Energy::shapeTerm() {
    // Mat A, AT, ATA, invATA, V, I;
    // I = Mat::eye(8, 8, CV_64F);
    double E = 0, cnt = 0;
    for (int i = 0; i < ver.size(); i++) {
        for (int j = 0; j < ver[i].size(); j++) {
            if (!i || !j) continue;
            // A = (
            //     Mat_<double>(8, 4) <<
            //     ver[i - 1][j - 1].x, -ver[i - 1][j - 1].y, 1, 0,
            //     ver[i - 1][j - 1].y, ver[i - 1][j - 1].x, 0, 1,
                
            //     ver[i - 1][j].x, -ver[i - 1][j].y, 1, 0,
            //     ver[i - 1][j].y, ver[i - 1][j].x, 0, 1,
                
            //     ver[i][j - 1].x, -ver[i][j - 1].y, 1, 0,
            //     ver[i][j - 1].y, ver[i][j - 1].x, 0, 1,
                
            //     ver[i][j].x, -ver[i][j].y, 1, 0,
            //     ver[i][j].y, ver[i][j].x, 0, 1
            // );
            // V = (
            //     Mat_<double>(8, 1) <<
            //     nver[i - 1][j - 1].x, nver[i - 1][j - 1].y,
            //     nver[i - 1][j].x, nver[i - 1][j].y,
            //     nver[i][j - 1].x, nver[i][j - 1].y,
            //     nver[i][j].x, nver[i][j].y
            // );
            // transpose(A, AT);
            // ATA = AT * A;
            // invert(ATA, invATA, DECOMP_LU);
            MatrixXd Aq(8, 4);
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
            
            Aq = (Aq * (Aq.transpose * Aq).inverse() * Aq.transpose() - MatrixXd::Identity(8, 8));
            VectorXd tmp = Aq * Vq;
            E += tmp.dot(tmp);
            cnt += 1.0;

            Aq = Aq.transpose() * Aq;
            A.resize(A.rows() + Aq.rows(), A.cols() + Aq.cols());
            A.block(A.rows() - Aq.rows(), A.cols() - Aq.cols()) << Aq;
            V.resize(V.rows() + Vq.rows(), V.cols());
            V << Vq;
            Y.resize(Y.rows() + Aq.rows(), Y.cols());
            Y << MatrixXd::zero(8, 1);
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
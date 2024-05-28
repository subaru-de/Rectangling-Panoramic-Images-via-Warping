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
    const int MAXV = 21 * 21 * 2;
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
A(MAXN, MAXV), B(MAXN, MAXV), C(MAXN, MAXV),
V(0, 1), Y(0, 1) {
    cout << getEnergy() << '\n';
}

double Energy::getEnergy() {
    getV();
    double E = 0.0;
    E += shapeTerm();
    E += lambdaBound * boundTerm();
    // E += lambdaLine * lineTerm();
    return E;
}

void Energy::getV() {
    for (int i = 0; i < ver.size(); i++) {
        for (int j = 0; j < ver[i].size(); j++) {
            V << nver[i][j].x, nver[i][j].y;
        }
    }
}

void Energy::getA() {
    int Nq = 0;
    for (int i = 0; i < ver.size(); i++) {
        for (int j = 0; j < ver[i].size(); j++, Nq++) {
            if (!i || !j) continue;
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
            Aq = (Aq * (Aq.transpose() * Aq).inverse() * Aq.transpose() - MatrixXd::Identity(8, 8));
            Aq = Aq.transpose() * Aq;
            for (int u = 0, uo, vo; u < 8; u++) {
                uo = Nq * 8 + u;
                vo = ((i - 1) * 21 + j - 1) * 2;
                A.insert(uo, vo) = Aq[u][0];
                A.insert(uo, vo + 1) = Aq[u][1];

                vo = ((i - 1) * 21 + j) * 2;
                A.insert(uo, vo) = Aq[u][2];
                A.insert(uo, vo + 1) = Aq[u][3];
                
                vo = (i * 21 + j - 1) * 2;
                A.insert(uo, vo) = Aq[u][4];
                A.insert(uo, vo + 1) = Aq[u][5];
                
                vo = (i * 21 + j) * 2;
                A.insert(uo, vo) = Aq[u][6];
                A.insert(uo, vo + 1) = Aq[u][7];
            }
            Y.resize(Y.rows() + 8, 1);
            Y << MatrixXd::Zero(8, 1);
        }
    }
}

void Energy::getB() {
    for (int i = 0, cnt = 0; i < nver.size(); i++) {
        for (int j = 0; j < nver[i].size(); j++, cnt++) {
            if (!i || !j) continue;
            int cur = cnt * 8;
            B.insert(cur, cur) = (i == 1);
            B.insert(cur + 1, cur + 1) = (j == 1);
            Y.resize(Y.rows() + 8, 1);
            Y << 0, 0;

            B.insert(cur + 2, cur + 2) = (i == 0);
            B.insert(cur + 3, cur + 3) = (j == nver[i].size() - 1);
            Y << 0, (j == nver[i].size() - 1 ? img.cols : 0);

            B.insert(cur + 4, cur + 4) = (i == nver.size() - 1);
            B.insert(cur + 5, cur + 5) = (j == 0);
            Y << (i == nver.size() - 1 ? img.rows : 0), 0;

            B.insert(cur + 6, cur + 6) = (i == nver.size() - 1);
            B.insert(cur + 7, cur + 7) = (j == nver[i].size() - 1);
            Y << (i == nver.size() - 1 ? img.rows : 0), (j == nver[i].size() - 1 ? img.cols : 0);
        }
    }
}

void Energy::getC() {}

double Energy::shapeTerm() {
    double E = 0;
    int Nq = 0;
    for (int i = 0; i < ver.size(); i++) {
        for (int j = 0; j < ver[i].size(); j++, Nq++) {
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
            for (int u = 0, uo, vo; u < 8; u++) {
                uo = Nq * 8 + u;
                vo = ((i - 1) * 21 + j - 1) * 2;
                A.insert(uo, vo) = Aq[u][0];
                A.insert(uo, vo + 1) = Aq[u][1];

                vo = ((i - 1) * 21 + j) * 2;
                A.insert(uo, vo) = Aq[u][2];
                A.insert(uo, vo + 1) = Aq[u][3];
                
                vo = (i * 21 + j - 1) * 2;
                A.insert(uo, vo) = Aq[u][4];
                A.insert(uo, vo + 1) = Aq[u][5];
                
                vo = (i * 21 + j) * 2;
                A.insert(uo, vo) = Aq[u][6];
                A.insert(uo, vo + 1) = Aq[u][7];
            }
            Y.resize(Y.rows() + 8, 1);
            Y << MatrixXd::Zero(8, 1);
        }
    }
    E /= Nq;
    return E;
}

double Energy::boundTerm() {
    double E = 0;
    for (int i = 0, cnt = 0; i < nver.size(); i++) {
        for (int j = 0; j < nver[i].size(); j++, cnt++) {
            if (!i || !j) continue;
            int cur = cnt * 8;
            B.insert(cur, cur) = (i == 1);
            B.insert(cur + 1, cur + 1) = (j == 1);
            Y.resize(Y.rows() + 8, 1);
            Y << 0, 0;

            B.insert(cur + 2, cur + 2) = (i == 0);
            B.insert(cur + 3, cur + 3) = (j == nver[i].size() - 1);
            Y << 0, (j == nver[i].size() - 1 ? img.cols : 0);

            B.insert(cur + 4, cur + 4) = (i == nver.size() - 1);
            B.insert(cur + 5, cur + 5) = (j == 0);
            Y << (i == nver.size() - 1 ? img.rows : 0), 0;

            B.insert(cur + 6, cur + 6) = (i == nver.size() - 1);
            B.insert(cur + 7, cur + 7) = (j == nver[i].size() - 1);
            Y << (i == nver.size() - 1 ? img.rows : 0), (j == nver[i].size() - 1 ? img.cols : 0);
        }
    }
}

double Energy::lineTerm() {

}

// #endif
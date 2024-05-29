// #ifndef Energy
// #define Energy
#include <cmath>
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
#include <Line.h>

using std::pair;
using std::cout;
using std::string;
using namespace cv;
using namespace Eigen;

class Energy {
private:
    const double lambdaLine = 100.0;
    const double lambdaBound = 1e8;
    
    const int cntBin = 50;
    const double alpha = acos(-1) / cntBin;
    Mat &img;
    vecvecP &ver, &nver;
    const int MAXNq, MAXV, MAXB;
    SparseMatrix<double> A, B, C;
    MatrixXd V, Y;
    vector<vector<vector<lin>>> lines;
    vector<vector<vector<MatrixXd>>> F;
    vector<vector<vector<int>>> bin;
    double theta[cntBin];
    int numInBin[cntBin];
    int totL;
public:
    Energy(Mat &img, vecvecP &ver, vecvecP &nver);
    void init();
    void getV();
    void getA();
    void getB();
    void getC();
    void getTheta(vector<vector<vector<lin>>> &lines, vector<vector<vector<MatrixXd>>> &F);
    double getEnergy();
    void optimize();
    void convertV();
    // 为了减少一次遍历，在 shapeTerm 函数中同时执行 getV 与 getA
    // double shapeTerm();
    // 在 lineTerm 中执行 getC
    // double lineTerm();
    // 在 boundTerm 中执行 getB
    // double boundTerm();
};

Energy::Energy(Mat &img, vecvecP &ver, vecvecP &nver):
MAXNq((ver.size() - 1) * (ver[0].size() - 1) * 8),
MAXV(ver.size() * ver[0].size() * 2), MAXB((ver.size() + ver[0].size()) * 2),
img(img), ver(ver), nver(nver),
A(MAXNq, MAXV), B(MAXB, MAXV), C(0, MAXV),
V(MAXV, 1), Y(0, 1) {
    init();
    optimize();
    cout << getEnergy() << '\n';
}

double Energy::getEnergy() {
    double E = 0.0;
    VectorXd tmp = A * V;
    E += tmp.dot(tmp);
    tmp = C * V;
    E += tmp.dot(tmp);
    tmp = B * V - Y.block(Y.rows() - MAXB, 0, MAXB, 1);
    E += tmp.dot(tmp);
    return E;
}

void Energy::optimize() {
    SparseMatrix<double> AA = A.transpose() * A;
    SparseMatrix<double> CC = C.transpose() * C;
    // SparseMatrix<double> AA = A;
    // std::ofstream outfile;

    for (int iter = 0; iter < 10; iter++) {
        getC();
        cout << "Energy before iteration #" << iter << ": "<< getEnergy() << '\n';
        SparseMatrix<double> L(AA.rows() + CC.rows() + B.rows(), AA.cols());
        MatrixXd YY(Y.rows() + AA.rows() + CC.rows(), 1);
        YY << MatrixXd::Zero(AA.rows() + CC.rows(), 1), Y;
        Y = YY;

        // 竖直方向上 concat 矩阵 AA CC B
        for (int k = 0; k < AA.outerSize(); k++) {
            for (SparseMatrix<double>::InnerIterator it(AA, k); it; ++it) {
                L.insert(it.row(), it.col()) = it.value();
            }
        }
        for (int k = 0; k < CC.outerSize(); k++) {
            for (SparseMatrix<double>::InnerIterator it(CC, k); it; ++it) {
                L.insert(it.row() + AA.rows(), it.col()) = it.value();
            }
        }
        for (int k = 0; k < B.outerSize(); k++) {
            for (SparseMatrix<double>::InnerIterator it(B, k); it; ++it) {
                L.insert(it.row() + AA.rows() + CC.rows(), it.col()) = it.value();
            }
        }
        L.makeCompressed();
        // cout << L.rows() << ' ' << L.cols() << " ---------------\n";
        // outfile.open("/home/nxte/codes/Rectangling-Panoramic-Images-via-Warping/include/matrix_output.txt");
        // if (outfile.is_open()) {
        //     outfile << L << '\n';
        //     outfile.close();
        //     std::cout << "矩阵已成功写入文件 matrix_output.txt" << std::endl;
        // } else {
        //     std::cerr << "无法打开文件" << std::endl;
        // }

        // L * V = Y
        SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> solver;
        solver.compute(L);
        if (solver.info() != Success) {
            std::cerr << "Decomposition failed!" << std::endl;
            return;
        }
        V = solver.solve(Y);
        if (solver.info() != Success) {
            std::cerr << "Solving failed!" << std::endl;
            return;
        }

        // 计算条件数（估计）
        // Eigen::SparseMatrix<double> R = solver.matrixR();
        // Eigen::VectorXd diagR = R.diagonal();
        // double cond_number = diagR.array().abs().maxCoeff() / diagR.array().abs().minCoeff();

        // std::cout << "Estimated condition number: " << cond_number << std::endl;
    }
    convertV();
}

void Energy::convertV() {
    for (int i = 0, cnt = 0; i < nver.size(); i++) {
        for (int j = 0; j < nver[i].size(); j++, cnt += 2) {
            nver[i][j].x = V(cnt, 0) + 0.5;
            nver[i][j].y = V(cnt + 1, 0) + 0.5;
            // cout << nver[i][j] << ' '<< V(cnt, 0) << ' ' << V(cnt + 1, 0) << '\n';
            // cout << nver[i][j] << ' ';
        }
        // cout << '\n';
    }
}

void Energy::getV() {
    for (int i = 0, cnt = 0; i < nver.size(); i++) {
        for (int j = 0; j < nver[i].size(); j++, cnt += 2) {
            V.block<2, 1>(cnt, 0) << nver[i][j].x, nver[i][j].y;
            // cout << nver[i][j] << ' ';
        }
        // cout << '\n';
    }
    // cout << '\n';
    // cout << V.block<10, 1>(0, 0) << '\n';
}

void Energy::getA() {
    double Nq = 0;
    for (int i = 0; i < ver.size(); i++) {
        for (int j = 0; j < ver[i].size(); j++) {
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
            // Aq = Aq.transpose() * Aq;
            for (int u = 0, uo, vo; u < 8; u++) {
                uo = Nq * 8 + u;
                vo = ((i - 1) * ver[i].size() + j - 1) * 2;
                A.insert(uo, vo) = Aq(u, 0);
                A.insert(uo, vo + 1) = Aq(u, 1);

                vo = ((i - 1) * ver[i].size() + j) * 2;
                A.insert(uo, vo) = Aq(u, 2);
                A.insert(uo, vo + 1) = Aq(u, 3);
                
                vo = (i * ver[i].size() + j - 1) * 2;
                A.insert(uo, vo) = Aq(u, 4);
                A.insert(uo, vo + 1) = Aq(u, 5);
                
                vo = (i * ver[i].size() + j) * 2;
                A.insert(uo, vo) = Aq(u, 6);
                A.insert(uo, vo + 1) = Aq(u, 7);
            }
            // Y.resize(Y.rows() + 8, 1);
            // Y.block<8, 1>(Y.rows() - 8, 0) << MatrixXd::Zero(8, 1);
            Nq++;
        }
    }
    A /= Nq;
    A.makeCompressed();
    assert(Nq * 8 == MAXNq);
}

void Energy::getB() {
    Y.resize(Y.rows() + MAXB, 1);
    int cnt = 0;
    for (int i = 0, pcnt = 0; i < nver.size(); i++) {
        for (int j = 0; j < nver[i].size(); j++, pcnt++) {
            if (!i) {
                B.insert(cnt, pcnt * 2 + 1) = lambdaBound;
                Y(Y.rows() - MAXB + cnt) = 0;
                cnt++;
            }
            else if (i + 1 == nver.size()) {
                B.insert(cnt, pcnt * 2 + 1) = lambdaBound;
                Y(Y.rows() - MAXB + cnt) = lambdaBound * (img.rows - 1);
                cnt++;
            }
            if (!j) {
                B.insert(cnt, pcnt * 2) = lambdaBound;
                Y(Y.rows() - MAXB + cnt) = 0;
                cnt++;
            }
            else if (j + 1 == nver[i].size()) {
                B.insert(cnt, pcnt * 2) = lambdaBound;
                Y(Y.rows() - MAXB + cnt) = lambdaBound * (img.cols - 1);
                cnt++;
            }
        }
    }
    assert(cnt == MAXB);
    B.makeCompressed();
}

void Energy::getC() {
    for (int i = 0; i < cntBin; i++) {
        theta[i] = 0;
    }
    getTheta(lines, F);
    C.resize(totL * 2, MAXV);
    for (int i = 0, cnt = 0, Nl = 0; i < ver.size(); i++) {
        for (int j = 0; j < ver[i].size(); j++, cnt++) {
            for (int k = 0; k < lines[i][j].size(); k++, Nl++) {
                lin &l = lines[i][j][k];
                MatrixXd R(2, 2), e(2, 1);
                int &m = bin[i][j][k];
                R << cos(theta[m]), -sin(theta[m]), sin(theta[m]), cos(theta[m]);
                e << l.second.x - l.first.x, l.second.y - l.first.y;
                MatrixXd Cl = R * e * (e.transpose() * e).inverse() * e.transpose() * R.transpose() - MatrixXd::Identity(2, 2);
                // Ce = CF * Vq, C is Cl above
                // Cl(2, 8)
                Cl = Cl * F[i][j][k];
                for (int u = 0, uo, vo; u < 2; u++) {
                    uo = Nl * 2 + u;
                    vo = ((i - 1) * ver[i].size() + j - 1) * 2;
                    C.insert(uo, vo) = Cl[u][0];
                    C.insert(uo, vo + 1) = Cl[u][1];
                    
                    vo = ((i - 1) * ver[i].size() + j) * 2;
                    C.insert(uo, vo) = Cl[u][2];
                    C.insert(uo, vo + 1) = Cl[u][3];

                    vo = (i * ver[i].size() + j - 1) * 2;
                    C.insert(uo, vo) = Cl[u][4];
                    C.insert(uo, vo + 1) = Cl[u][5];

                    vo = (i * ver[i].size() + j) * 2;
                    C.insert(uo, vo) = Cl[u][6];
                    C.insert(uo, vo + 1) = Cl[u][7];
                }
            }
        }
    }
    C *= 1.0 * lambdaLine / totL;
    C.makeCompressed();
}

void Energy::init() {
    totL = 0;
    Line(img, ver, lines, F);
    // asin(e(1, 0)) 为角度
    vector<vector<vector<int>>> bin(ver.size(), vector<vector<int>>(ver[0].size(), vector<int>()));
    for (int i = 0, cnt = 0; i < ver.size(); i++) {
        for (int j = 0; j < ver[i].size(); j++, cnt++) {
            for (int k = 0; k < lines[i][j].size(); k++) {
                totL++;
                double angle = asin(l.second.y - l.first.y);
                bin[i][j].push_back(floor(angle / alpha));
                numInBin[bin[i][j][k]]++;
            }
        }
    }
    getV();
    getA();
    getB();
}

void Energy::getTheta(vector<vector<vector<lin>>> &lines, vector<vector<vector<MatrixXd>>> &F) {
    for (int i = 0, cnt = 0; i < ver.size(); i++) {
        for (int j = 0; j < ver[i].size(); j++, cnt++) {
            V(cnt, 0)
            MatrixXd Vq(8, 1);
            Vq <<
                V((cnt - ver[i].size() - 1) * 2, 0), V((cnt - ver[i].size() - 1) * 2 + 1, 0),
                V((cnt - ver[i].size()) * 2, 0), V((cnt - ver[i].size()) * 2 + 1, 0),
                V((cnt - 1) * 2, 0), V((cnt - 1) * 2 + 1, 0),
                V(cnt * 2, 0), V(cnt * 2 + 1, 0);
            for (int k = 0; k < lines[i][j].size(); k++) {
                lin &l = lines[i][j][k];
                MatrixXd nl = F[i][j][k] * Vq;
                double angle = asin(l.second.y - l.first.y);
                int bin = floor(angle / alpha);
                double nangle = asin(nl(1, 0));
                theta[i] += (nangle - angle) / numInBin[bin];
            }
        }
    }
}

// double Energy::shapeTerm() {
//     double E = 0;
//     int Nq = 0;
//     for (int i = 0; i < ver.size(); i++) {
//         for (int j = 0; j < ver[i].size(); j++, Nq++) {
//             if (!i || !j) continue;
//             MatrixXd Aq(8, 4), Vq(8, 1);
//             Aq <<
//                 ver[i - 1][j - 1].x, -ver[i - 1][j - 1].y, 1, 0,
//                 ver[i - 1][j - 1].y, ver[i - 1][j - 1].x, 0, 1,
//                 ver[i - 1][j].x, -ver[i - 1][j].y, 1, 0,
//                 ver[i - 1][j].y, ver[i - 1][j].x, 0, 1,
//                 ver[i][j - 1].x, -ver[i][j - 1].y, 1, 0,
//                 ver[i][j - 1].y, ver[i][j - 1].x, 0, 1,
//                 ver[i][j].x, -ver[i][j].y, 1, 0,
//                 ver[i][j].y, ver[i][j].x, 0, 1;
//             Vq <<
//                 nver[i - 1][j - 1].x, nver[i - 1][j - 1].y,
//                 nver[i - 1][j].x, nver[i - 1][j].y,
//                 nver[i][j - 1].x, nver[i][j - 1].y,
//                 nver[i][j].x, nver[i][j].y;
//             Aq = (Aq * (Aq.transpose() * Aq).inverse() * Aq.transpose() - MatrixXd::Identity(8, 8));
//             VectorXd tmp = Aq * Vq;
//             E += tmp.dot(tmp);
//             Aq = Aq.transpose() * Aq;
//             for (int u = 0, uo, vo; u < 8; u++) {
//                 uo = Nq * 8 + u;
//                 vo = ((i - 1) * 21 + j - 1) * 2;
//                 A.insert(uo, vo) = Aq(u, 0);
//                 A.insert(uo, vo + 1) = Aq(u, 1);

//                 vo = ((i - 1) * 21 + j) * 2;
//                 A.insert(uo, vo) = Aq(u, 2);
//                 A.insert(uo, vo + 1) = Aq(u, 3);
                
//                 vo = (i * 21 + j - 1) * 2;
//                 A.insert(uo, vo) = Aq(u, 4);
//                 A.insert(uo, vo + 1) = Aq(u, 5);
                
//                 vo = (i * 21 + j) * 2;
//                 A.insert(uo, vo) = Aq(u, 6);
//                 A.insert(uo, vo + 1) = Aq(u, 7);
//             }
//             Y.resize(Y.rows() + 8, 1);
//             Y << MatrixXd::Zero(8, 1);
//         }
//     }
//     E /= Nq;
//     return E;
// }

// double Energy::boundTerm() {
//     double E = 0;
//     for (int i = 0, cnt = 0; i < nver.size(); i++) {
//         for (int j = 0; j < nver[i].size(); j++, cnt++) {
//             if (!i || !j) continue;
//             int cur = cnt * 8;
//             B.insert(cur, cur) = (i == 1);
//             B.insert(cur + 1, cur + 1) = (j == 1);
//             Y.resize(Y.rows() + 8, 1);
//             Y << 0, 0;

//             B.insert(cur + 2, cur + 2) = (i == 0);
//             B.insert(cur + 3, cur + 3) = (j == nver[i].size() - 1);
//             Y << 0, (j == nver[i].size() - 1 ? img.cols : 0);

//             B.insert(cur + 4, cur + 4) = (i == nver.size() - 1);
//             B.insert(cur + 5, cur + 5) = (j == 0);
//             Y << (i == nver.size() - 1 ? img.rows : 0), 0;

//             B.insert(cur + 6, cur + 6) = (i == nver.size() - 1);
//             B.insert(cur + 7, cur + 7) = (j == nver[i].size() - 1);
//             Y << (i == nver.size() - 1 ? img.rows : 0), (j == nver[i].size() - 1 ? img.cols : 0);
//         }
//     }
// }

// double Energy::lineTerm() {

// }

// #endif
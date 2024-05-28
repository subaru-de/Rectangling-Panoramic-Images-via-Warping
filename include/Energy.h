// #ifndef Energy
// #define Energy
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
    const int MAXNq, MAXV, MAXB;
    SparseMatrix<double> A, B, C;
    MatrixXd V, Y;
public:
    Energy(Mat &img, vecvecP &ver, vecvecP &nver);
    void getV();
    void getA();
    void getB();
    void getC();
    double getEnergy();
    void optimize();
    void convertV();
    // 为了减少一次遍历，在 shapeTerm 函数中同时执行 getV 与 getA
    double shapeTerm();
    // 在 lineTerm 中执行 getC
    double lineTerm();
    // 在 boundTerm 中执行 getB
    double boundTerm();
};

Energy::Energy(Mat &img, vecvecP &ver, vecvecP &nver):
MAXNq((ver.size() - 1) * (ver[0].size() - 1) * 8),
MAXV(ver.size() * ver[0].size() * 2), MAXB((ver.size() + ver[0].size()) * 2),
img(img), ver(ver), nver(nver),
A(MAXNq, MAXV), B(MAXB, MAXV), C(MAXNq, MAXV),
V(0, 1), Y(0, 1) {
    cout << getEnergy() << '\n';
    optimize();
    cout << getEnergy() << '\n';
}

double Energy::getEnergy() {
    if (!V.rows()) {
        V.resize(MAXV, 1);
        getV();
        getA();
        getB();
    }
    double E = 0.0;
    // E += (V.transpose() * A.transpose() * A * V)(0, 0);
    VectorXd tmp = A * V;
    E += tmp.dot(tmp) / MAXNq;
    tmp = B * V - Y.block(Y.rows() - MAXB, 0, MAXB, 1);
    E += tmp.dot(tmp);
    // E += lambdaLine * lineTerm();
    return E;
}

void Energy::optimize() {
    SparseMatrix<double> AA = A.transpose() * A;
    std::ofstream outfile;
    // outfile.open("/home/nxte/codes/Rectangling-Panoramic-Images-via-Warping/include/matrix1_output.txt");
    // if (outfile.is_open()) {
    //     outfile << A << '\n';
    //     outfile.close();
    //     std::cout << "矩阵已成功写入文件 matrix1_output.txt" << std::endl;
    // } else {
    //     std::cerr << "无法打开文件" << std::endl;
    // }


    // SparseMatrix<double> AA = A;
    cout << AA.rows() << "qwqwq\n";
    SparseMatrix<double> L(AA.rows() + B.rows(), AA.cols());
    MatrixXd YY(Y.rows() + AA.rows(), 1);
    YY << MatrixXd::Zero(AA.rows(), 1), Y;
    Y = YY;
    // 竖直方向上 concat 矩阵 AA 和 B
    for (int k = 0; k < AA.outerSize(); k++) {
        for (SparseMatrix<double>::InnerIterator it(AA, k); it; ++it) {
            L.insert(it.row(), it.col()) = it.value();
        }
    }
    bool flag = 0;
    for (int k = 0; k < B.outerSize(); k++) {
        for (SparseMatrix<double>::InnerIterator it(B, k); it; ++it) {
            L.insert(it.row() + AA.rows(), it.col()) = it.value();
            if (it.value() != 0) {
                // cout << it.value() << '\t' << it.row() << ' ' << it.col() << '\n';
                flag = 1;
            }
        }
    }
    outfile.open("/home/nxte/codes/Rectangling-Panoramic-Images-via-Warping/include/matrix_output.txt");
    if (outfile.is_open()) {
        outfile << L << '\n';
        outfile.close();
        std::cout << "矩阵已成功写入文件 matrix_output.txt" << std::endl;
    } else {
        std::cerr << "无法打开文件" << std::endl;
    }
    L.makeCompressed();
    // cout << L.rows() << ' ' << L.cols() << " ---------------\n";
    assert(flag == 1);

    // // 创建一个正则化参数
    // double lambda = 0.1;

    // // 创建正则化项
    // Eigen::SparseMatrix<double> I(L.cols(), L.cols());
    // I.setIdentity();

    // // 构造增强矩阵
    // Eigen::SparseMatrix<double> L_reg = Eigen::SparseMatrix<double>(L.rows() + L.cols(), L.cols());
    // Eigen::MatrixXd Y_reg = Eigen::MatrixXd(L.rows() + L.cols(), 1);

    // // 设置上半部分
    // for (int k = 0; k < L.outerSize(); ++k) {
    //     for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
    //         L_reg.insert(it.row(), it.col()) = it.value();
    //     }
    // }

    // // 设置下半部分
    // for (int k = 0; k < I.outerSize(); ++k) {
    //     for (Eigen::SparseMatrix<double>::InnerIterator it(I, k); it; ++it) {
    //         L_reg.insert(L.rows() + it.row(), it.col()) = lambda * it.value();
    //     }
    // }
    // L_reg.makeCompressed();

    // Y_reg.topRows(L.rows()) = Y;
    // Y_reg.bottomRows(L.cols()).setZero();

    // L * V = Y
    SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> solver;
    solver.compute(L);
    // solver.compute(AA);
    if (solver.info() != Success) {
        std::cerr << "Decomposition failed!" << std::endl;
        return;
    }
    V = solver.solve(Y);
    // V = solver.solve(MatrixXd::Zero(AA.rows(), 1));
    if (solver.info() != Success) {
        std::cerr << "Solving failed!" << std::endl;
        return;
    }
    // V = (L.transpose() * L).inverse() * L.transpose();

    // 计算条件数（估计）
    Eigen::SparseMatrix<double> R = solver.matrixR();
    Eigen::VectorXd diagR = R.diagonal();
    double cond_number = diagR.array().abs().maxCoeff() / diagR.array().abs().minCoeff();

    std::cout << "Estimated condition number: " << cond_number << std::endl;

    convertV();
}

void Energy::convertV() {
    for (int i = 0, cnt = 0; i < nver.size(); i++) {
        for (int j = 0; j < nver[i].size(); j++, cnt += 2) {
            nver[i][j].x = V(cnt, 0) + 0.5;
            nver[i][j].y = V(cnt + 1, 0) + 0.5;
            // cout << nver[i][j] << ' '<< V(cnt, 0) << ' ' << V(cnt + 1, 0) << '\n';
            cout << nver[i][j] << ' ';
        } cout << '\n';
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
    A.makeCompressed();
    // A /= Nq;
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

void Energy::getC() {}

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
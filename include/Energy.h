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
    const double lambdaBound = 1e6;
    
    static const int cntBin = 50;
    static constexpr double alpha = acos(-1) / cntBin;
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
    void getTheta();
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
    // cout << "quq\n";
    init();
    // cout << "qnq\n";
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
    // SparseMatrix<double> AA = A;

    double preEnergy = 1e18, E = 1e18;
    for (int iter = 0; iter < 10; iter++) {
        // cout << iter << '\n';
        getC();
        // cout << "quq\n";
        SparseMatrix<double> CC = C.transpose() * C;
        // SparseMatrix<double> CC = C;
        // cout << "qnq\n";

        SparseMatrix<double> L(A.rows() + C.rows() + B.rows(), A.cols());
        // 竖直方向上 concat 矩阵 A C B
        for (int k = 0; k < A.outerSize(); k++) {
            for (SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
                if (it.value() == 0) continue;
                L.insert(it.row(), it.col()) = it.value();
            }
        }
        for (int k = 0; k < C.outerSize(); k++) {
            for (SparseMatrix<double>::InnerIterator it(C, k); it; ++it) {
                if (it.value() == 0) continue;
                L.insert(it.row() + A.rows(), it.col()) = it.value();
            }
        }
        for (int k = 0; k < B.outerSize(); k++) {
            for (SparseMatrix<double>::InnerIterator it(B, k); it; ++it) {
                if (it.value() == 0) continue;
                L.insert(it.row() + A.rows() + C.rows(), it.col()) = it.value();
            }
        }
        L.makeCompressed();
        // SparseMatrix<double> LL(AA.rows() + B.rows(), AA.cols());
        // for (int k = 0; k < L.outerSize(); k++) {
        //     for (SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
        //         LL.insert(it.row(), it.col()) = it.value();
        //     }
        // }

        MatrixXd YY(A.rows() + C.rows() + B.rows(), 1);
        YY << MatrixXd::Zero(A.rows() + C.rows(), 1), Y;

        // std::ofstream outfile;
        // outfile.open("/home/nxte/codes/Rectangling-Panoramic-Images-via-Warping/include/matrix_output.txt");
        // if (outfile.is_open()) {
        //     outfile << L << "\n\n\n";
        //     outfile.close();
        //     std::cout << "矩阵已成功写入文件 matrix_output.txt" << std::endl;
        // } else {
        //     std::cerr << "无法打开文件" << std::endl;
        // }

        // L * V = YY
        SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> solver;
        solver.compute(L);
        if (solver.info() != Success) {
            std::cerr << "Decomposition failed!" << std::endl;
            return;
        }
        V = solver.solve(YY);
        if (solver.info() != Success) {
            std::cerr << "Solving failed!" << std::endl;
            return;
        }

        // 计算条件数（估计）
        // Eigen::SparseMatrix<double> R = solver.matrixR();
        // Eigen::VectorXd diagR = R.diagonal();
        // double cond_number = diagR.array().abs().maxCoeff() / diagR.array().abs().minCoeff();

        // std::cout << "Estimated condition number: " << cond_number << std::endl;
        
        cout << "Energy after iteration #" << iter << ": "<< (E = getEnergy()) << '\t';
        double incRate = (preEnergy - E) / preEnergy;
        cout << "inc_rate: " << incRate << '\n';
        if (incRate < 0.01) break;
        preEnergy = E;
    }
    convertV();
}

void Energy::convertV() {
    for (int i = 0, cnt = 0; i < nver.size(); i++) {
        for (int j = 0; j < nver[i].size(); j++, cnt += 2) {
            nver[i][j].x = V(cnt, 0) + 0.5;
            nver[i][j].y = V(cnt + 1, 0) + 0.5;
            // cout << i << ' ' << j << ' ' << nver[i][j] << '\n';
            assert(nver[i][j].x >= 0 && nver[i][j].x < img.cols);
            assert(nver[i][j].y >= 0 && nver[i][j].y < img.rows);
        }
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
    getTheta();
    C.resize(totL * 2, MAXV);
    int Nl = 0;
    for (int i = 0, cnt = 0; i < ver.size(); i++) {
        for (int j = 0; j < ver[i].size(); j++, cnt++) {
            for (int k = 0; k < lines[i][j].size(); k++, Nl++) {
                // cout << k << "quq\n";
                lin &l = lines[i][j][k];
                Point2d vecl = (l.second - l.first) / dis(l.first, l.second);
                MatrixXd R(2, 2), e(2, 1);
                // cout << k << "qoq\n";
                
                int &m = bin[i][j][k];
                assert(m < cntBin);
                R << cos(theta[m]), -sin(theta[m]), sin(theta[m]), cos(theta[m]);
                // cout << m << ' ' << theta[m] << '\n';
                assert(!isnan(R(0, 0)) && !isnan(R(0, 1)) && !isnan(R(1, 0)) && !isnan(R(0, 1)));
                // cout << "R: \n" << R << '\n';
                e << vecl.x, vecl.y;
                // cout << "e: \n" << e << '\n';
                MatrixXd Cl = R * e * (e.transpose() * e).inverse() * e.transpose() * R.transpose() - MatrixXd::Identity(2, 2);
                // Ce = CF * Vq, C is Cl above
                // Cl(2, 8)
                // cout << "qwqCl: \n" << Cl << '\n';
                Cl = Cl * F[i][j][k];
                // cout << "Cl: \n" << Cl << '\n';
                bool flag = 0;
                for (int u = 0; u < 8; u++) {
                    if (Cl(0, u)) flag = 1;
                }
                assert(flag == 1);
                flag = 0;
                for (int u = 0; u < 8; u++) {
                    if (Cl(0, u)) flag = 1;
                }
                assert(flag == 1);
                for (int u = 0, uo, vo; u < 2; u++) {
                    uo = Nl * 2 + u;
                    vo = ((i - 1) * ver[i].size() + j - 1) * 2;
                    C.insert(uo, vo) = Cl(u, 0);
                    C.insert(uo, vo + 1) = Cl(u, 1);
                    
                    vo = ((i - 1) * ver[i].size() + j) * 2;
                    C.insert(uo, vo) = Cl(u, 2);
                    C.insert(uo, vo + 1) = Cl(u, 3);

                    vo = (i * ver[i].size() + j - 1) * 2;
                    C.insert(uo, vo) = Cl(u, 4);
                    C.insert(uo, vo + 1) = Cl(u, 5);

                    vo = (i * ver[i].size() + j) * 2;
                    C.insert(uo, vo) = Cl(u, 6);
                    C.insert(uo, vo + 1) = Cl(u, 7);
                }
                // cout << k << "qnq\n";
            }
        }
    }
    assert(Nl == totL);
    C *= 1.0 * lambdaLine / totL;
    C.makeCompressed();
}

void Energy::init() {
    totL = 0;
    Line(img, ver, lines, F);
    // acos(e(0, 0)) 为角度
    bin.resize(ver.size(), vector<vector<int>>(ver[0].size(), vector<int>(0)));
    for (int i = 0, cnt = 0; i < ver.size(); i++) {
        // cout << i << '\n';
        for (int j = 0; j < ver[i].size(); j++, cnt++) {
            // cout << '\t' << j << '\n';
            for (int k = 0; k < lines[i][j].size(); k++) {
                // cout << "\t\t" << k << '\n';
                totL++;
                lin &l = lines[i][j][k];
                Point2d vecl = (l.second - l.first) / dis(l.first, l.second);
                assert(vecl.y >= -1 && vecl.y <= 1);
                double angle = acos(vecl.x);
                bin[i][j].push_back(floor(angle / alpha));
                assert(!isnan(bin[i][j][k]) && !isinf(bin[i][j][k]));
                assert(bin[i][j][k] >= 0 && bin[i][j][k] < cntBin);
                numInBin[bin[i][j][k]]++;
            }
        }
    }
    // cout << "bin:\n";
    // for (int i = 0; i < bin.size(); i++) {
    //     cout << i << '\n';
    //     for (int j = 0; j < bin[i].size(); j++) {
    //         cout << "\t" << j << '\n';
    //         for (int k = 0; k < bin[i][j].size(); k++) {
    //             cout << bin[i][j][k] << ' ';
    //         } cout << '\n';
    //     } cout << '\n';
    // } cout << '\n';
    getV();
    getA();
    getB();
}

void Energy::getTheta() {
    for (int i = 0, cnt = 0; i < ver.size(); i++) {
        for (int j = 0; j < ver[i].size(); j++, cnt++) {
            if (!i || !j) continue;
            MatrixXd Vq(8, 1);
            Vq <<
                V((cnt - ver[i].size() - 1) * 2, 0), V((cnt - ver[i].size() - 1) * 2 + 1, 0),
                V((cnt - ver[i].size()) * 2, 0), V((cnt - ver[i].size()) * 2 + 1, 0),
                V((cnt - 1) * 2, 0), V((cnt - 1) * 2 + 1, 0),
                V(cnt * 2, 0), V(cnt * 2 + 1, 0);
            for (int k = 0; k < lines[i][j].size(); k++) {
                lin &l = lines[i][j][k];
                Point2d vecl = (l.second - l.first) / dis(l.first, l.second);
                // cout << k << '\n';
                MatrixXd nl = F[i][j][k] * Vq;
                double lennl = sqrt(nl(0, 0) * nl(0, 0) + nl(1, 0) * nl(1, 0));
                if (lennl) nl /= lennl;
                else {
                    cout << F[i][j][k] << "\n\n";
                    cout << Vq << "\n\n";
                    cout << nl << '\n';
                    assert(0);
                }
                for (int ii = 0; ii < F[i][j][k].rows(); ii++) {
                    for (int jj = 0; jj < F[i][j][k].cols(); jj++) {
                        assert(!isnan(F[i][j][k](ii, jj)));
                    }
                }
                for (int ii = 0; ii < Vq.rows(); ii++) {
                    for (int jj = 0; jj < Vq.cols(); jj++) {
                        assert(!isnan(Vq(ii, jj)));
                    }
                }
                
                // cout << k << '\n';
                double angle = acos(vecl.x);
                int bin = floor(angle / alpha);
                // cout << "nl:\n" << nl << '\n';
                assert(!isnan(nl(0, 0)) && !isnan(nl(1, 0)));
                assert(nl(0, 0) >= -1 && nl(0, 0) <= 1 && nl(1, 0) >= -1 && nl(1, 0) <= 1);
                // cout << nl(0, 0) << '\n';
                double nangle = acos(nl(0, 0));
                // cout << angle << ' ' << nangle << '\n';
                assert(!isnan(angle) && !isnan(nangle));
                if (numInBin[bin]) theta[i] += (nangle - angle) / numInBin[bin];
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
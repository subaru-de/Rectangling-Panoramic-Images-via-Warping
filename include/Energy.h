#include <string>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <Rectangling.h>

using std::cout;
using std::string;
using namespace cv;

class Energy {
private:
    const double lambdaLine = 100.0;
    const double lambdaBound = 1e8;
    Mat &img;
    vecvecP &ver, &nver;
public:
    Energy(Mat &img, vecvecP &ver, vecvecP &nver);
    double getEnergy();
    double shapeTerm();
    double lineTerm();
    double boundTerm();
};

Energy::Energy(Mat &img, vecvecP &ver, vecvecP &nver):
img(img), ver(ver), nver(nver) {}

double Energy::getEnergy() {
    double E = 0.0;
    E += shapeTerm();
    E += lambdaBound * boundTerm();
}

double Energy::shapeTerm() {
    Mat A, AT, ATA, invATA, V, I;
    I = Mat::eye(8, 8, CV_64F);
    double E = 0, cnt = 0;
    for (int i = 0; i < ver.size(); i++) {
        for (int j = 0; j < ver[i].size(); j++) {
            if (!i || !j) continue;
            A = (
                Mat_<double>(8, 4) <<
                ver[i - 1][j - 1].x, -ver[i - 1][j - 1].y, 1, 0,
                ver[i - 1][j - 1].y, ver[i - 1][j - 1].x, 0, 1,
                
                ver[i - 1][j].x, -ver[i - 1][j].y, 1, 0,
                ver[i - 1][j].y, ver[i - 1][j].x, 0, 1,
                
                ver[i][j - 1].x, -ver[i][j - 1].y, 1, 0,
                ver[i][j - 1].y, ver[i][j - 1].x, 0, 1,
                
                ver[i][j].x, -ver[i][j].y, 1, 0,
                ver[i][j].y, ver[i][j].x, 0, 1
            );
            V = (
                Mat_<double>(8, 1) <<
                nver[i - 1][j - 1].x, nver[i - 1][j - 1].y,
                nver[i - 1][j].x, nver[i - 1][j].y,
                nver[i][j - 1].x, nver[i][j - 1].y,
                nver[i][j].x, nver[i][j].y
            );
            transpose(A, AT);
            ATA = AT * A;
            invert(ATA, invATA, DECOMP_LU);

            Mat tmp = (A * invATA * AT - I) * V;
            E += tmp.dot(tmp);
            cnt += 1.0;
        }
    }
    E /= cnt;
    return E;
}

double Energy::boundTerm() {
    double E = 0;
    for (int j = 0; j < nver[0].size(); j++) {
        E += nver[0][j].y * nver[0][j].y;
        double de = (nver.back())[j].y - img.rows + 1;
        E += de * de;
    }
    for (int i = 0; i < nver.size(); i++) {
        E += nver[i][0].x * nver[i][0].x;
        double de = nver[i].back().x - img.cols + 1;
        E += de * de;
    }
    return E;
}
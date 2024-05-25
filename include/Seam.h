#include <cmath>
#include <vector>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

using std::cout;
using std::vector;
using namespace cv;

double INF = 1e8;
enum BorderType {
    Top = 0,
    Bottom = 1,
    Left = 3,
    Right = 4
};

const Vec3b Black = Vec3b(0, 0, 0);
const Vec3b White = Vec3b(255, 255, 255);
const Vec3b Orange = Vec3b(0, 165, 255);
const Vec3b Green = Vec3b(0, 255, 0);

class Seam {
private:
    // Mat E;
public:
    const Mat sobelVer = (Mat_<double>(3, 3) <<
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1
    );
    const Mat sobelHor = (Mat_<double>(3, 3) <<
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    );
    Seam(const Mat &img);
    void insertVertical(Mat &img, Mat &mask, Mat &dispV, Mat &litSeam, BorderType BType);
    void insertHorizontal(Mat &img, Mat &mask, Mat &dispH, Mat &litSeam, BorderType BType);
};

Seam::Seam(const Mat &img) {
    // Mat sobelVer = (Mat_<Vec3d>(3, 3) <<
    //     Vec3d(-1, -1, -1), Vec3b(-2, -2, -2), Vec3b(-1, -1, -1),
    //     Vec3d(0, 0, 0), Vec3d(0, 0, 0), Vec3d(0, 0, 0),
    //     Vec3d(1, 1, 1), Vec3b(2, 2, 2), Vec3b(1, 1, 1)
    // );
    // Mat sobelHor = (Mat_<Vec3d>(3, 3) <<
    //     Vec3d(-1, -1, -1), Vec3d(0, 0, 0), Vec3b(1, 1, 1),
    //     Vec3d(-2, -2, -2), Vec3d(0, 0, 0), Vec3b(2, 2, 2),
    //     Vec3d(-1, -1, -1), Vec3b(0, 0, 0), Vec3b(1, 1, 1)
    // );

    // E.create(img.size(), CV_64FC1);
    // Rect rect = Rect(0, 0, 3, 3);
    // Mat channel[3];
    // for (int i = 1; i < img.rows - 1; i++) {
    //     for (int j = 1; j < img.cols - 1; j++) {
    //         split(img(rect), channel);
    //         Vec3d tmp;
    //         // !!! check broadcast !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //         for (int c = 0; c < 3; c++) {
    //             channel[c].convertTo(channel[c], CV_64F);
    //             tmp[c] = sobelVer.dot(channel[c]);
    //         }
    //         E.at<double>(i, j) = sqrt(tmp.dot(tmp));
            
    //         for (int c = 0; c < 3; c++) {
    //             tmp[c] = sobelHor.dot(channel[c]);
    //         }
    //         E.at<double>(i, j) += sqrt(tmp.dot(tmp));
    //         rect.x++;
    //     }
    //     rect.y++;
    //     rect.x = 0;
    // }
    // for (int i = 0; i < img.rows; i++) {
    //     E.at<double>(i, 0) = E.at<double>(i, img.cols - 1) = 1e8;
    // }
    // for (int j = 0; j < img.cols; j++) {
    //     E.at<double>(0, j) = E.at<double>(img.rows - 1, j) = 1e8;
    // }
    // !!! check if E is right.
}

void Seam::insertVertical(Mat &img, Mat &mask, Mat &dispV, Mat &litSeam, BorderType BType) { 
    // cout << "-------- insert vertical seam --------\n";
    // cout << "sub-image size: " << img.size() << '\n';
    // 保证找出的 seam 在 mask == 1 范围内
    // 考虑如何实现这件事
    // 论文中给出的方法是把 mask == 0 的像素的 cost 都设成 inf = 1e8，以此保证 seam 不经过它们
    // 考虑是否一定存在一条不经过 mask == 0 的 seam
    // 可以使用反证法
    // 在讲 ppt 的时候可以讲一下这个证明
    // 记得更新位移场 disp field
    Mat M, from;
    M.create(img.size(), CV_64FC1); // sub-image
    from.create(img.size(), CV_32SC1); // sub-image

    /* -------- get Energy -------- */
    Mat Energy;
    Energy.create(img.size(), CV_64FC1);
    Mat paddedImg;
    copyMakeBorder(img, paddedImg, 1, 1, 1, 1, BORDER_REPLICATE);
    
    // 初始化梯度矩阵
    Mat gradX(img.size(), CV_64FC3);
    Mat gradY(img.size(), CV_64FC3);
    // 对每个通道分别应用Sobel滤波器
    for (int c = 0; c < 3; ++c) {
        Mat channel = Mat(paddedImg.size(), CV_64F);
        extractChannel(paddedImg, channel, c);

        Mat gradXChannel, gradYChannel;
        filter2D(channel, gradXChannel, CV_64F, sobelHor);
        filter2D(channel, gradYChannel, CV_64F, sobelVer);

        insertChannel(gradXChannel(Rect(1, 1, img.cols, img.rows)), gradX, c);
        insertChannel(gradYChannel(Rect(1, 1, img.cols, img.rows)), gradY, c);
    }
    // 计算能量
    INF = 0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            Vec3d gx = gradX.at<Vec3d>(i, j);
            Vec3d gy = gradY.at<Vec3d>(i, j);
            Energy.at<double>(i, j) = sqrt(gx.dot(gx) + gy.dot(gy));
            INF = max(INF, Energy.at<double>(i, j));
        }
    }
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (mask.at<uchar>(i, j) == 0) {
                Energy.at<double>(i, j) = 1e8;
            }
            else if (mask.at<uchar>(i, j) == 2) {
                Energy.at<double>(i, j) = INF;
            }
        }
    }

    /* -------- Dynamic Programming get M -------- */
    /* -------- Border -------- */
    for (int j = 0; j < img.cols; j++) {
        M.at<double>(0, j) = Energy.at<double>(0, j);
    }
    /* -------- dp -------- */
    for (int i = 1; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            M.at<double>(i, j) = M.at<double>(i - 1, j);
            from.at<int>(i, j) = j;
            if (j > 0 && M.at<double>(i - 1, j - 1) < M.at<double>(i, j)) {
                M.at<double>(i, j) = M.at<double>(i - 1, j - 1);
                from.at<int>(i, j) = j - 1;
            }
            if (j < img.cols - 1 && M.at<double>(i - 1, j + 1) < M.at<double>(i, j)) {
                M.at<double>(i, j) = M.at<double>(i - 1, j + 1);
                from.at<int>(i, j) = j + 1;
            }
            // if (M.at<double>(i, j) < INF || Energy.at<double>(i, j) < INF) {
                M.at<double>(i, j) += Energy.at<double>(i, j);
            // }
        }
    }
    /* -------- store seam -------- */
    vector<Point> verSeam(1, {img.rows - 1, 0});
    double mn = M.at<double>(img.rows - 1, 0);
    for (int j = 1; j < img.cols; j++) {
        // if (M.at<double>(img.rows - 1, j) > INF) {
        //     M.at<double>(img.rows - 1, j) -= INF;
        // }
        if (M.at<double>(img.rows - 1, j) < mn) {
            mn = M.at<double>(img.rows - 1, j);
            verSeam[0] = {img.rows - 1, j};
        }
    }
    cout << "minM: " << mn << '\n';
    for (; verSeam.back().x > 0; ) {
        int &pre = from.at<int>(verSeam.back().x, verSeam.back().y);
        verSeam.push_back({verSeam.back().x - 1, pre});
    }
    reverse(verSeam.begin(), verSeam.end());

    /* -------- Insert Vertical Seam -------- */
    if (BType == Right) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = img.cols - 1; j > verSeam[i].y; j--) {
                // if (!mask.at<uchar>(i, j - 1)) continue;
                img.at<Vec3b>(i, j) = img.at<Vec3b>(i, j - 1);
                mask.at<uchar>(i, j) = mask.at<uchar>(i, j - 1);
                dispV.at<int>(i, j)--;
            }
            litSeam.at<Vec3b>(verSeam[i].x, verSeam[i].y) = Orange;
            if (verSeam[i].y + 1 < img.cols && mask.at<uchar>(verSeam[i].x, verSeam[i].y + 1)) {
                mask.at<uchar>(verSeam[i].x, verSeam[i].y + 1) = 2;
            }
            if (verSeam[i].y > 0) {
                if (mask.at<uchar>(verSeam[i].x, verSeam[i].y)) mask.at<uchar>(verSeam[i].x, verSeam[i].y) = 2;
                if (mask.at<uchar>(verSeam[i].x, verSeam[i].y - 1)) mask.at<uchar>(verSeam[i].x, verSeam[i].y - 1) = 2;
                Vec3d tmp = (Vec3d)img.at<Vec3b>(verSeam[i].x, verSeam[i].y) + (Vec3d)img.at<Vec3b>(verSeam[i].x, verSeam[i].y - 1);
                img.at<Vec3b>(verSeam[i].x, verSeam[i].y) = tmp / 2.0;
                // img.at<Vec3b>(verSeam[i].x, verSeam[i].y) = Orange;
            }
        }
    }
    else { // Left
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < verSeam[i].y; j++) {
                // if (!mask.at<uchar>(i, j + 1)) continue;
                img.at<Vec3b>(i, j) = img.at<Vec3b>(i, j + 1);
                mask.at<uchar>(i, j) = mask.at<uchar>(i, j + 1);
                dispV.at<int>(i, j)++;
            }
            litSeam.at<Vec3b>(verSeam[i].x, verSeam[i].y) = Orange;
            if (verSeam[i].y > 0 && mask.at<uchar>(verSeam[i].x, verSeam[i].y - 1)) {
                mask.at<uchar>(verSeam[i].x, verSeam[i].y - 1) = 2;
            }
            if (verSeam[i].y < img.cols - 1) {
                if (mask.at<uchar>(verSeam[i].x, verSeam[i].y)) mask.at<uchar>(verSeam[i].x, verSeam[i].y) = 2;
                if (mask.at<uchar>(verSeam[i].x, verSeam[i].y + 1)) mask.at<uchar>(verSeam[i].x, verSeam[i].y + 1) = 2;
                Vec3d tmp = (Vec3d)img.at<Vec3b>(verSeam[i].x, verSeam[i].y) + (Vec3d)img.at<Vec3b>(verSeam[i].x, verSeam[i].y + 1);
                img.at<Vec3b>(verSeam[i].x, verSeam[i].y) = tmp / 2.0;
                // img.at<Vec3b>(verSeam[i].x, verSeam[i].y) = Orange;
            }
        }
    }
}

void Seam::insertHorizontal(Mat &img, Mat &mask, Mat &dispH, Mat &litSeam, BorderType BType) {
    // cout << "-------- insert horizontal seam --------\n";
    
    /* -------- Find Horizontal Seam -------- */
    // cout << "sub-image size: " << img.size() << '\n';
    // cout << img.rows << ' ' << img.cols << '\n';
    Mat M, from;
    M.create(img.size(), CV_64FC1);
    from.create(img.size(), CV_32SC1);

    /* -------- get Energy -------- */
    Mat Energy;
    Energy.create(img.size(), CV_64F);
    Mat paddedImg;
    copyMakeBorder(img, paddedImg, 1, 1, 1, 1, BORDER_REPLICATE);
    
    // 初始化梯度矩阵
    Mat gradX(img.size(), CV_64FC3);
    Mat gradY(img.size(), CV_64FC3);
    // 对每个通道分别应用Sobel滤波器
    for (int c = 0; c < 3; ++c) {
        Mat channel = Mat(paddedImg.size(), CV_64F);
        extractChannel(paddedImg, channel, c);

        Mat gradXChannel, gradYChannel;
        filter2D(channel, gradXChannel, CV_64F, sobelHor);
        filter2D(channel, gradYChannel, CV_64F, sobelVer);

        insertChannel(gradXChannel(Rect(1, 1, img.cols, img.rows)), gradX, c);
        insertChannel(gradYChannel(Rect(1, 1, img.cols, img.rows)), gradY, c);
    }
    // 计算能量
    INF = 0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            Vec3d gx = gradX.at<Vec3d>(i, j);
            Vec3d gy = gradY.at<Vec3d>(i, j);
            Energy.at<double>(i, j) = sqrt(gx.dot(gx) + gy.dot(gy));
            INF = max(INF, Energy.at<double>(i, j));
        }
    }
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (mask.at<uchar>(i, j) == 0) {
                Energy.at<double>(i, j) = 1e8;
            }
            else if (mask.at<uchar>(i, j) == 2) {
                Energy.at<double>(i, j) = INF;
            }
        }
    }

    /* -------- Dynamic Programming get M -------- */
    /* -------- Border -------- */
    for (int i = 0; i < img.rows; i++) {
        M.at<double>(i, 0) = Energy.at<double>(i, 0);
    }
    /* -------- dp -------- */
    for (int j = 1; j < img.cols; j++) {
        for (int i = 0; i < img.rows; i++) {
            M.at<double>(i, j) = M.at<double>(i, j - 1);
            from.at<int>(i, j) = i;
            if (i > 0 && M.at<double>(i - 1, j - 1) < M.at<double>(i, j)) {
                M.at<double>(i, j) = M.at<double>(i - 1, j - 1);
                from.at<int>(i, j) = i - 1;
            }
            if (i < img.rows - 1 && M.at<double>(i + 1, j - 1) < M.at<double>(i, j)) {
                M.at<double>(i, j) = M.at<double>(i + 1, j - 1);
                from.at<int>(i, j) = i + 1;
            }
            // if (M.at<double>(i, j) < INF || Energy.at<double>(i, j) < INF) {
                M.at<double>(i, j) += Energy.at<double>(i, j);
            // }
        }
    }
    /* -------- get min and store seam -------- */
    vector<Point> horSeam(1, {0, img.cols - 1});
    double mn = M.at<double>(0, img.cols - 1);
    for (int i = 1; i < img.rows; i++) {
        // if (M.at<double>(i, img.cols - 1) > INF) {
        //     M.at<double>(i, img.cols - 1) -= INF;
        // }
        if (M.at<double>(i, img.cols - 1) < mn) {
            mn = M.at<double>(i, img.cols - 1);
            horSeam[0] = {i, img.cols - 1};
        }
    }
    cout << "minM: " << mn << '\n';
    for (; horSeam.back().y > 0; ) {
        int &pre = from.at<int>(horSeam.back().x, horSeam.back().y);
        horSeam.push_back({pre, horSeam.back().y - 1});
    }
    reverse(horSeam.begin(), horSeam.end());
    
    /* -------- Insert Horizontal Seam -------- */
    if (BType == Bottom) { // Bottom
        for (int j = 0; j < img.cols; j++) {
            for (int i = img.rows - 1; i > horSeam[j].x; i--) {
                // if (!mask.at<uchar>(i - 1, j)) continue;
                img.at<Vec3b>(i, j) = img.at<Vec3b>(i - 1, j);
                mask.at<uchar>(i, j) = mask.at<uchar>(i - 1, j);
                dispH.at<int>(i, j)--;
            }
            litSeam.at<Vec3b>(horSeam[j].x, horSeam[j].y) = Orange;
            if (horSeam[j].x + 1 < img.rows && mask.at<uchar>(horSeam[j].x + 1, horSeam[j].y)) {
                mask.at<uchar>(horSeam[j].x + 1, horSeam[j].y) = 2;
            }
            if (horSeam[j].x > 0) {
                if (mask.at<uchar>(horSeam[j].x, horSeam[j].y)) mask.at<uchar>(horSeam[j].x, horSeam[j].y) = 2;
                if (mask.at<uchar>(horSeam[j].x - 1, horSeam[j].y)) mask.at<uchar>(horSeam[j].x - 1, horSeam[j].y) = 2;
                Vec3d tmp = (Vec3d)img.at<Vec3b>(horSeam[j].x, horSeam[j].y) + (Vec3d)img.at<Vec3b>(horSeam[j].x - 1, horSeam[j].y);
                img.at<Vec3b>(horSeam[j].x, horSeam[j].y) = tmp / 2.0;
                // img.at<Vec3b>(horSeam[j].x, horSeam[j].y) = Orange;
            }
        }
    }
    else { // Top
        for (int j = 0; j < img.cols; j++) {
            for (int i = 0; i < horSeam[j].x; i++) {
                // if (!mask.at<uchar>(i + 1, j)) continue;
                img.at<Vec3b>(i, j) = img.at<Vec3b>(i + 1, j);
                mask.at<uchar>(i, j) = mask.at<uchar>(i + 1, j);
                dispH.at<int>(i, j)++;
            }
            litSeam.at<Vec3b>(horSeam[j].x, horSeam[j].y) = Orange;
            if (horSeam[j].x > 0 && mask.at<uchar>(horSeam[j].x - 1, horSeam[j].y)) {
                mask.at<uchar>(horSeam[j].x - 1, horSeam[j].y) = 2;
            }
            if (horSeam[j].x < img.rows - 1) {
                if (mask.at<uchar>(horSeam[j].x, horSeam[j].y)) mask.at<uchar>(horSeam[j].x, horSeam[j].y) = 2;
                if (mask.at<uchar>(horSeam[j].x + 1, horSeam[j].y)) mask.at<uchar>(horSeam[j].x + 1, horSeam[j].y) = 2;
                Vec3d tmp = (Vec3d)img.at<Vec3b>(horSeam[j].x, horSeam[j].y) + (Vec3d)img.at<Vec3b>(horSeam[j].x + 1, horSeam[j].y);
                img.at<Vec3b>(horSeam[j].x, horSeam[j].y) = tmp / 2.0;
                // img.at<Vec3b>(horSeam[j].x, horSeam[j].y) = Orange;
            }
        }
    }
}
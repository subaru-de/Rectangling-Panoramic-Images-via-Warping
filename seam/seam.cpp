#include <cmath>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

class Seam {
private:
    Mat E;
public:
    void getEnergy(const Mat &img);
    void getVertical(const Mat &img);
    void getHorizontal(const Mat &img);
};

void Seam::getEnergy(const Mat &img) {
    E.create(img.size(), CV_64FC1);
    Mat sobelVer = (Mat_<Vec3d>(3, 3) <<
        Vec3d(-1, -1, -1), Vec3b(-2, -2, -2), Vec3b(-1, -1, -1),
        Vec3d(0, 0, 0), Vec3d(0, 0, 0), Vec3d(0, 0, 0),
        Vec3d(1, 1, 1), Vec3b(2, 2, 2), Vec3b(1, 1, 1)
    );
    Mat sobelHor = (Mat_<Vec3d>(3, 3) <<
        Vec3d(-1, -1, -1), Vec3d(0, 0, 0), Vec3b(1, 1, 1),
        Vec3d(-2, -2, -2), Vec3d(0, 0, 0), Vec3b(2, 2, 2),
        Vec3d(-1, -1, -1), Vec3b(0, 0, 0), Vec3b(1, 1, 1)
    );
    Rect rect = Rect(0, 0, 3, 3);
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1; j++) {
            Vec3d tmp = sobelVer.dot(img(rect));
            // !!! check broadcast !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            E.at<double>(i, j) = sqrt(tmp.dot(tmp));
            tmp = sobelHor.dot(img(rect));
            E.at<double>(i, j) += sqrt(tmp.dot(tmp));
            rect.x++;
        }
        rect.y++;
        rect.x = 0;
    }
    for (int i = 0; i < img.rows; i++) {
        E.at<double>(i, 0) = E.at<double>(i, img.cols - 1) = 1e8;
    }
    for (int j = 0; j < img.cols; j++) {
        E.at<double>(0, j) = E.at<double>(img.rows - 1, j) = 1e8;
    }
    // !!! check if E is right.
}

void Seam::getVertical(const Mat &img) {
    Mat M, from;
    M.create(img.size(), CV_64FC1);
    from.create(img.size(), CV_32SC1);
    for (int j = 0; j < img.cols; j++) {
        M.at<double>(0, j) = E.at<double>(0, j);
    }
    for (int i = 1; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            M.at<double>(i, j) = M.at<double>(i - 1, j);
            from.at<double>(i, j) = j;
            if (j > 0 && M.at<double>(i - 1, j - 1) < M.at<double>(i, j)) {
                M.at<double>(i, j) = M.at<double>(i - 1, j - 1);
                from.at<double>(i, j) = j - 1;
            }
            if (j < img.cols - 1 && M.at<double>(i - 1, j + 1) < M.at<double>(i, j)) {
                M.at<double>(i, j) = M.at<double>(i - 1, j + 1);
                from.at<double>(i, j) = j + 1;
            }
            M.at<double>(i, j) += E.at<double>(i, j);
        }
    }
    vector<Point> verSeam(1, {img.rows - 1, 0});
    int mn = M.at<double>(img.rows - 1, 0);
    for (int j = 1; j < img.cols; j++) {
        if (M.at<double>(img.rows - 1, j) < mn) {
            mn = M.at<double>(img.rows - 1, j);
            verSeam[0] = {img.rows - 1, j};
        }
    }
    for (; verSeam.back().x > 0; ) {
        verSeam.push_back({verSeam.back().x - 1, from.at<double>(verSeam.back())});
    }
    // !!! check vertical seam
}

void Seam::getHorizontal(const Mat &img) {
    Mat M, from;
    M.create(img.size(), CV_64FC1);
    from.create(img.size(), CV_32SC1);
    for (int i = 0; i < img.rows; i++) {
        M.at<double>(i, 0) = E.at<double>(i, 0);
    }
    for (int j = 1; j < img.cols; j++) {
        for (int i = 0; i < img.rows; i++) {
            M.at<double>(i, j) = M.at<double>(i, j - 1);
            from.at<double>(i, j) = i;
            if (i > 0 && M.at<double>(i - 1, j - 1) < M.at<double>(i, j)) {
                M.at<double>(i, j) = M.at<double>(i - 1, j - 1);
                from.at<double>(i, j) = i - 1;
            }
            if (i < img.rows - 1 && M.at<double>(i + 1, j - 1) < M.at<double>(i, j)) {
                M.at<double>(i, j) = M.at<double>(i + 1, j - 1);
                from.at<double>(i, j) = i + 1;
            }
            M.at<double>(i, j) += E.at<double>(i, j);
        }
    }
    // !!! check horizontal seam
}
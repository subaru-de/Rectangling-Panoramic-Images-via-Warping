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
}

void Seam::getVertical(const Mat &img) {
    Mat M, from;
    M.create(img.size(), CV_64FC1);
    from.create(img.size(), CV_32SC1);
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
}

void Seam::getHorizontal(const Mat &img) {
}
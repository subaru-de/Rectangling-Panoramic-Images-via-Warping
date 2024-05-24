#include <vector>
#include <string>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <Seam.h>

using std::cout;
using std::string;
using std::vector;
using namespace cv;

enum DirectionType {
    Vertical = 0,
    Horizontal = 1
};

class Rectangling {
private:
    Mat &img;
    Mat mask;
    vector<vector<Point>> disp; // 位移场
    Vec3b Corner;
public:
    Rectangling(Mat &image);
    void getRect(Rect &rect, DirectionType DType, CornerType CType, int seamLen, int seamEndp);
    void insertSeam();
    void showImg();
};

Rectangling::Rectangling(Mat &image):
img(image) {
    Corner = image.at<Vec3b>(0);
    cout << "Size of the input image: " << img.size() << "\n";
    cout << img.rows << ' ' << img.cols << '\n';
    
    // 初始化 mask，mask 应该在 insertSeam 之后更新
    // 接下来需要保证找到的 seam 在 mask 之内
    mask.create(img.size(), CV_8UC1);
    mask.setTo(Scalar(1));
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img.at<Vec3b>(i, j) == Corner) {
                mask.at<uchar>(i, j) = 0;
            }
            else break;
        }
        for (int j = img.cols - 1; j >= 0; j--) {
            if (img.at<Vec3b>(i, j) == Corner) {
                mask.at<uchar>(i, j) = 0;
            }
            else break;
        }
    }
    for (int j = 0; j < img.cols; j++) {
        for (int i = 0; i < img.rows; i++) {
            if (mask.at<uchar>(i, j) == 0) break;
            else if (img.at<Vec3b>(i, j) == Corner) {
                mask.at<uchar>(i, j) = 0;
            }
            else break;
        }
        for (int i = img.rows - 1; i >= 0; i--) {
            if (mask.at<uchar>(i, j) == 0) break;
            else if (img.at<Vec3b>(i, j) == Corner) {
                mask.at<uchar>(i, j) = 0;
            }
            else break;
        }
    }
    
    // 初始化位移场
    disp.resize(img.rows, vector<Point>(img.cols, {0, 0}));
}

void Rectangling::getRect(Rect &rect, DirectionType DType, CornerType CType, int seamLen, int seamEndp) {
    if (DType == Vertical) { // Vertical
        rect.width = img.cols;
        rect.height = seamLen;
        // roi.x + roi.width <= m.cols
        // roi.y + roi.height <= m.rows
        rect.x = 0;
        rect.y = seamEndp;
    }
    else { // Horizontal
        rect.width = seamLen;
        rect.height = img.rows;
        // roi.x + roi.width <= m.cols
        // roi.y + roi.height <= m.rows
        rect.x = seamEndp;
        rect.y = 0;
    }
}

void Rectangling::insertSeam() {
    int horLen = 0, verLen = 0, horEndp = 0, verEndp = 0;
    BoarderType horType, verType;
    Seam seam(img);
    
    for (int loopCount = 0; ; loopCount++) {
        showImg();
        printf("-------- loopCount: %d --------\n", loopCount);
        /* ------- Vertical --------*/
        int len[2] = {0, 0}, mx[2] = {0, 0}, endp[2] = {0, 0};
        for (int i = 0; i < img.rows; i++) {
            if (mask.at<uchar>(i, 0) == 0) {
                len[0]++;
            }
            else {
                if (len[0] > mx[0]) {
                    mx[0] = len[0];
                    endp[0] = i - len[0];
                }
                len[0] = 0;
            }
            if (mask.at<uchar>(i, img.cols - 1) == 0) {
                len[1]++;
            }
            else {
                if (len[1] > mx[1]) {
                    mx[1] = len[1];
                    endp[1] = i - len[1];
                }
                len[1] = 0;
            }
        }
        // 考虑最后一段
        if (len[0] > mx[0]) {
            mx[0] = len[0];
            endp[0] = img.rows - len[0];
        }
        len[0] = 0;
        if (len[1] > mx[1]) {
            mx[1] = len[1];
            endp[1] = img.rows - len[1];
        }
        len[1] = 0;
        // 更新最长段
        if (mx[0] >= mx[1]) {
            verType = Left;
            verLen = mx[0];
            verEndp = endp[0];
        }
        else {
            verType = Right;
            verLen = mx[1];
            verEndp = endp[1];
        }
        
        /* ------- Horizontal --------*/
        len[0] = len[1] = mx[0] = mx[1] = endp[0] = endp[1] = 0;
        for (int j = 0; j < img.cols; j++) {
            if (mask.at<uchar>(0, j) == 0) {
                len[0]++;
            }
            else {
                if (len[0] > mx[0]) {
                    mx[0] = len[0];
                    endp[0] = j - len[0];
                }
                len[0] = 0;
            }
            if (mask.at<uchar>(img.rows - 1, j) == 0) {
                len[1]++;
            }
            else {
                if (len[1] > mx[1]) {
                    mx[1] = len[1];
                    endp[1] = j - len[1];
                }
                len[1] = 0;
            }
        }
        // 考虑最后一段
        if (len[0] > mx[0]) {
            mx[0] = len[0];
            endp[0] = img.cols - len[0];
        }
        len[0] = 0;
        if (len[1] > mx[1]) {
            mx[1] = len[1];
            endp[1] = img.cols - len[1];
        }
        len[1] = 0;
        // 更新最长段
        if (mx[0] >= mx[1]) {
            horType = Top;
            horLen = mx[0];
            horEndp = endp[0];
        }
        else {
            horType = Bottom;
            horLen = mx[1];
            horEndp = endp[1];
        }
        
        printf("verLen: %d\t verType: %d\t verEndp: %d\n", verLen, verType, verEndp);
        printf("horLen: %d\t horType: %d\t horEndp: %d\n", horLen, horType, horEndp);

        if (verLen == 0 && horLen == 0) break;
        /* -------- choose vertical or horizontal -------- */
        Rect rect;
        // roi.x + roi.width <= m.cols
        // roi.y + roi.height <= m.rows
        Mat tmpImg, tmpMask;
        if (verLen >= horLen) { // Vertical
            // get rect
            getRect(rect, Vertical, verType, verLen, verEndp);
            printf("V rect.x: %d\t rect.y: %d\t rect.width: %d\t rect.height: %d\n", rect.x, rect.y, rect.width, rect.height);
            tmpImg = img(rect);
            tmpMask = mask(rect);
            seam.insertVertical(tmpImg, tmpMask, verType);
        }
        else { // Horizontal
            // get rect
            getRect(rect, Horizontal, horType, horLen, horEndp);
            printf("H rect.x: %d\t rect.y: %d\t rect.width: %d\t rect.height: %d\n", rect.x, rect.y, rect.width, rect.height);
            tmpImg = img(rect);
            tmpMask = mask(rect);
            seam.insertHorizontal(tmpImg, tmpMask, horType);
        }
        img(rect) = tmpImg;
        mask(rect) = tmpMask;
    }
}

void Rectangling::showImg() {
    // cout << "qwqwq " << img.size() << '\n';
    imshow("Image after seam carving", img);
    waitKey(0);
}
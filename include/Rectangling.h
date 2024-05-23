#include <string>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <Seam.h>

using std::cout;
using std::string;
using namespace cv;

enum DirectionType {
    Vertical = 0,
    Horizontal = 1
};

class Rectangling {
private:
    Mat &img;
    Vec3b Corner;
public:
    Rectangling(Mat &image);
    void getRect(Rect &rect, DirectionType DType, CornerType CType, int seamLen);
    void insertSeam();
    void showImg();
};

Rectangling::Rectangling(Mat &image):
img(image) {
    Corner = image.at<Vec3b>(0);
    cout << "Size of the input image: " << img.size() << "\n";
}

void Rectangling::getRect(Rect &rect, DirectionType DType, CornerType CType, int seamLen) {
    if (DType == Vertical) {
        rect.y = 0;
        rect.width = img.cols;
        rect.height = seamLen + 1;
        if (CType == TopLeft || CType == TopRight) {
            rect.x = 0;
        }
        else { // Bottom
            rect.x = img.rows - seamLen;
        }
    }
    else { // Horizontal
        rect.x = 0;
        rect.width = seamLen + 1;
        rect.height = img.rows;
        if (CType == TopLeft || CType == BottomLeft) { // Left
            rect.y = 0;
        }
        else { // Right
            rect.y = img.cols - seamLen;
        }
    }
}

void Rectangling::insertSeam() {
    int horLen = 0, verLen = 0;
    CornerType horType, verType;
    Seam seam(img);
    
    for (int loopCount = 0; ; loopCount++) {
        printf("-------- loopCount: %d --------\n", loopCount);
        /* ------- Vertical --------*/
        int flag[2] = {-1, -1};
        for (int i = 0; i < img.rows; i++) {
            if (flag[0] == i - 1 && img.at<Vec3b>(i, 0) == Corner) {
                flag[0] = i;
            }
            if (flag[1] == i - 1 && img.at<Vec3b>(i, img.cols - 1) == Corner) {
                flag[1] = i;
            }
        }
        if (flag[0] >= flag[1]) {
            verType = TopLeft;
            verLen = flag[0];
        }
        else {
            verType = TopRight;
            verLen = flag[1];
        }
        
        flag[0] = flag[1] = img.rows;
        for (int i = img.rows - 1; i >= 0; i--) {
            if (flag[0] == i + 1 && img.at<Vec3b>(i, 0) == Corner) {
                flag[0] = i;
            }
            if (flag[1] == i + 1 && img.at<Vec3b>(i, img.cols - 1) == Corner) {
                flag[1] = i;
            }
        }
        if (flag[0] <= flag[1] && img.rows - flag[0] > verLen) {
            verType = BottomLeft;
            verLen = img.rows - flag[0];
        }
        else if (img.rows - flag[1] > verLen) {
            verType = BottomRight;
            verLen = img.rows - flag[1];
        }
        // Len is (length - 1)

        /* ------- Horizontal --------*/
        flag[0] = flag[1] = -1;
        for (int j = 0; j < img.cols; j++) {
            if (flag[0] == j - 1 && img.at<Vec3b>(0, j) == Corner) {
                flag[0] = j;
            }
            if (flag[1] == j - 1 && img.at<Vec3b>(img.rows - 1, j) == Corner) {
                flag[1] = j;
            }
        }
        if (flag[0] >= flag[1]) {
            horType = TopLeft;
            horLen = flag[0];
        }
        else {
            horType = BottomLeft;
            horLen = flag[1];
        }
        
        flag[0] = flag[1] = img.cols;
        for (int j = img.cols - 1; j >= 0; j--) {
            if (flag[0] == j + 1 && img.at<Vec3b>(0, j) == Corner) {
                flag[0] = j;
            }
            if (flag[1] == j + 1 && img.at<Vec3b>(img.rows - 1, j) == Corner) {
                flag[1] = j;
            }
        }
        if (flag[0] <= flag[1] && img.cols - flag[0] > horLen) {
            horType = TopRight;
            horLen = img.cols - flag[0];
        }
        else if (img.cols - flag[1] > horLen) {
            horType = BottomRight;
            horLen = img.cols - flag[1];
        }

        if (verLen == -1 && horLen == -1) break;
        /* -------- choose vertical or horizontal -------- */
        Rect rect;
        Mat tmpImg = img(rect);
        if (verLen >= horLen) {
            // get rect
            getRect(rect, Horizontal, verType, verLen);
            seam.insertVertical(tmpImg, verType);
        }
        else {
            // get rect
            getRect(rect, Vertical, horType, horLen);
            seam.insertHorizontal(tmpImg, horType);
        }
        img(rect) = tmpImg;
    }
}

void Rectangling::showImg() {
    // cout << "qwqwq " << img.size() << '\n';
    imshow("Image after seam carving", img);
    waitKey(0);
}
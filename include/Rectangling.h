// #ifndef Rectangling
// #define Rectangling
#include <ctime>
#include <vector>
#include <string>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <Seam.h>
#include <Mesh.h>

using std::cout;
using std::string;
using std::vector;
using namespace cv;

// const Vec3b Black = Vec3b(0, 0, 0);
// const Vec3b White = Vec3b(255, 255, 255);
namespace {
enum DirectionType {
    Vertical = 0,
    Horizontal = 1
};

class Rectangling {
private:
    Mat &img;
    Mat resImg, finImg;
    Mat img_bak;
    Mat mask;
    // x 对应的是 width，y 对应的是 height
    Mat dispV, dispH; // 位移场
    Vec3b Corner;
    Mat litSeam; // 高亮 seam 的位置的图片
public:
    Rectangling(Mat &image);
    void init();
    void getRect(Rect &rect, DirectionType DType, BorderType BType, int seamLen, int seamEndp);
    void insertSeam();
    void showImg();
    void writeImg(string filename);
    void showSeam();
    void showMesh(Mesh &mesh, bool initial, bool showVer);
};

Rectangling::Rectangling(Mat &image):
img(image) {
    imwrite("../output/initial.jpg", img);
    img.copyTo(img_bak);
    clock_t start_t = clock();
    init();
    // showImg();
    
    insertSeam();
    // imshow("mask after", mask);
    // showImg();
    imwrite("../output/after_seam_carving.jpg", img);
    showSeam();

    clock_t mid1_t = clock();
    std::cout << ">>>> Seam Carving used ";
    std::cout << (mid1_t - start_t) * 1000 / CLOCKS_PER_SEC << "ms\n";

    resImg.create(img.rows, img.cols, CV_8UC3);
    Mesh mesh(img, resImg);
    {
        Mat res;
        img.copyTo(res);
        mesh.putMesh(res, 1);
        imwrite("../output/img_with_mesh.jpg", res);
    }
    img_bak.copyTo(img);
    mesh.displace(dispV, dispH);

    {
        Mat res;
        img.copyTo(res);
        mesh.putMesh(res, 1);
        imwrite("../output/initial_img_with_mesh.jpg", res);
    }
    mesh.callEnergy();

    clock_t mid2_t = clock();
    std::cout << ">>>> Energy Optimization used ";
    std::cout << (mid2_t - mid1_t) * 1000 / CLOCKS_PER_SEC << "ms\n";

    mesh.callGL();
    resImg = mesh.getRes();
    {
        Mat res;
        resImg.copyTo(res);
        mesh.putMesh(res, 0);
        imwrite("../output/result_before_with_mesh.jpg", res);
    }
    imwrite("../output/result_before_Post_processing.jpg", resImg);

    clock_t mid3_t = clock();

    pair<double, double> scaleFactor = mesh.getScale();
    // resImg.release();
    finImg.create(img.rows / scaleFactor.second, img.cols / scaleFactor.first, CV_8UC3);
    mesh.setRes(finImg);
    mesh.callEnergy();

    clock_t end_t = clock();
    std::cout << ">>>> Post-Processing used ";
    std::cout << (end_t - mid3_t) * 1000 / CLOCKS_PER_SEC << "ms\n";

    mesh.callGL();
    finImg = mesh.getRes();
    {
        Mat res;
        finImg.copyTo(res);
        mesh.putMesh(res, 0);
        imwrite("../output/result_with_mesh.jpg", res);
    }
    imwrite("../output/result.jpg", finImg);

    std::cout << "Established, used ";
    std::cout << ((end_t - start_t) - (mid3_t - mid2_t)) * 1000 / CLOCKS_PER_SEC << "ms in total.\n";
}

void Rectangling::init() {
    Corner = img.at<Vec3b>(0);
    cout << "Size of the input image: " << img.size() << "\n";
    // cout << img.rows << ' ' << img.cols << '\n';


    /* -------- 有噪声 -------- */
    // for (int j = 0; j < img.cols; j++) {
    //     if (img.at<Vec3b>(0, j) != White) {
    //         cout << j << ' ' << img.at<Vec3b>(0, j) << '\n';
    //     }
    // }
    
    // 初始化 mask，mask 应该在 insertSeam 之后更新
    // 接下来需要保证找到的 seam 在 mask 之内
    /* -------- get mask -------- */
    const double cornerEps = 300;
    mask.create(img.size(), CV_8UC1);
    mask.setTo(Scalar(255));
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            Vec3d de = img.at<Vec3b>(i, j);
            de -= (Vec3d)Corner;
            if (de.dot(de) < cornerEps) {
                mask.at<uchar>(i, j) = 0;
            }
            else break;
        }
        for (int j = img.cols - 1; j >= 0; j--) {
            Vec3d de = img.at<Vec3b>(i, j);
            de -= (Vec3d)Corner;
            if (de.dot(de) < cornerEps) {
                mask.at<uchar>(i, j) = 0;
            }
            else break;
        }
    }
    for (int j = 0; j < img.cols; j++) {
        for (int i = 0; i < img.rows; i++) {
            // if (mask.at<uchar>(i, j) == 0) break;
            // else
            Vec3d de = img.at<Vec3b>(i, j);
            de -= (Vec3d)Corner;
            if (de.dot(de) < cornerEps) {
                mask.at<uchar>(i, j) = 0;
            }
            else break;
        }
        for (int i = img.rows - 1; i >= 0; i--) {
            // if (mask.at<uchar>(i, j) == 0) break;
            // else
            Vec3d de = img.at<Vec3b>(i, j);
            de -= (Vec3d)Corner;
            if (de.dot(de) < cornerEps) {
                mask.at<uchar>(i, j) = 0;
            }
            else break;
        }
    }
    Mat erodeImg, dilateImg;
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    erode(mask, erodeImg, element);
    erode(erodeImg, erodeImg, element);
    // erode(erodeImg, erodeImg, element);
    // erode(erodeImg, erodeImg, element);
    mask = erodeImg;

    /* -------- show mask -------- */
    // Mat outImg;
    // outImg.create(img.size(), CV_8UC3);
    // for (int i = 0; i < img.rows; i++) {
    //     for (int j = 0; j < img.cols; j++) {
    //         if (mask.at<uchar>(i, j)) outImg.at<Vec3b>(i, j) = White;
    //         else outImg.at<Vec3b>(i, j) = Black;
    //     }
    // }
    // imshow("Image", mask);
    // waitKey(0);
    imwrite("../output/mask.jpg", mask);
    
    // 初始化位移场
    dispV.create(img.size(), CV_32SC1);
    dispH.create(img.size(), CV_32SC1);
    dispV.setTo(Scalar(0));
    dispH.setTo(Scalar(0));

    // 初始化 litSeam
    litSeam.create(img.size(), CV_8UC3);
    litSeam.setTo(White);
}

void Rectangling::getRect(Rect &rect, DirectionType DType, BorderType BType, int seamLen, int seamEndp) {
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
    BorderType horType, verType;
    Seam seam(img);
    
    for (int loopCount = 0; ; loopCount++) {
        // showImg();
        // printf("-------- loopCount: %d --------\n", loopCount);
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
        
        // printf("verLen: %d\t verType: %d\t verEndp: %d\n", verLen, verType, verEndp);
        // printf("horLen: %d\t horType: %d\t horEndp: %d\n", horLen, horType, horEndp);

        if (verLen == 0 && horLen == 0) break;
        /* -------- choose vertical or horizontal -------- */
        Rect rect;
        // roi.x + roi.width <= m.cols
        // roi.y + roi.height <= m.rows
        Mat tmpImg, tmpMask, tmpDispV, tmpDispH, tmpLitSeam;
        if (verLen >= horLen) { // Vertical
            // get rect
            getRect(rect, Vertical, verType, verLen, verEndp);
            // printf("V rect.x: %d\t rect.y: %d\t rect.width: %d\t rect.height: %d\n", rect.x, rect.y, rect.width, rect.height);
            tmpImg = img(rect);
            tmpMask = mask(rect);
            tmpDispV = dispV(rect);
            tmpDispH = dispH(rect);
            tmpLitSeam = litSeam(rect);
            seam.insertVertical(tmpImg, tmpMask, tmpDispV, tmpDispH, tmpLitSeam, verType);
            dispV(rect) = tmpDispV;
            dispH(rect) = tmpDispH;
        }
        else { // Horizontal
            // get rect
            getRect(rect, Horizontal, horType, horLen, horEndp);
            // printf("H rect.x: %d\t rect.y: %d\t rect.width: %d\t rect.height: %d\n", rect.x, rect.y, rect.width, rect.height);
            tmpImg = img(rect);
            tmpMask = mask(rect);
            tmpDispV = dispV(rect);
            tmpDispH = dispH(rect);
            tmpLitSeam = litSeam(rect);
            seam.insertHorizontal(tmpImg, tmpMask, tmpDispV, tmpDispH, tmpLitSeam, horType);
            dispV(rect) = tmpDispV;
            dispH(rect) = tmpDispH;
        }
        img(rect) = tmpImg;
        mask(rect) = tmpMask;
        litSeam(rect) = tmpLitSeam;
    }
    // showImg();
}

void Rectangling::showImg() {
    // cout << "qwqwq " << img.size() << '\n';
    imshow("Image", img);
    waitKey(0);
}

void Rectangling::showSeam() {
    Mat res;
    img.copyTo(res);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (litSeam.at<Vec3b>(i, j) != White) {
                res.at<Vec3b>(i, j) = litSeam.at<Vec3b>(i, j);
            }
        }
    }
    // imshow("Image", res);
    // waitKey(0);
    imwrite("../output/img_with_seam.jpg", res);
}

void Rectangling::showMesh(Mesh &mesh, bool initial, bool showVer) {
    Mat res;
    if (!showVer) {
        finImg.copyTo(res);
    }
    else {
        img.copyTo(res);
    }
    mesh.putMesh(res, showVer);
    if (!showVer) imwrite("../output/result_with_mesh.jpg", res);
    else if (!initial) imwrite("../output/img_with_mesh.jpg", res);
    else imwrite("../output/initial_img_with_mesh.jpg", res);
}

void Rectangling::writeImg(string filename) {
    imwrite(filename, img);
}

}
// #endif
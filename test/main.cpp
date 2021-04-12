#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main() {
    std::cout << "Hello, World!" << std::endl;
    VideoCapture cap(0);
    if (!cap.isOpened()) return -1;
    Mat frame;
    namedWindow("camera", 1);
    while (true) {
        cap >> frame;
        imshow("frame", frame);
        if (waitKey(2) > 0) break;
        destroyAllWindows();
        cap.release();
    }
    return 0;
}

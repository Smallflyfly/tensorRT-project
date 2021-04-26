#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
using namespace std;
using namespace cv;

int main() {

    VideoCapture cap;
//    cap.open("person.mp4");
//    cout<<CV_VERSION<<endl;
//    cout<<"fang"<<endl;
    cap.open("person1080.mp4");
    if (!cap.isOpened()) return -1;
    cout<<"fang"<<endl;
    Mat frame;
    while (true) {
        cap >> frame;
        imshow("frame", frame);
        if (waitKey(10) >= 0) break;
    }
    cap.release();
    destroyAllWindows();
//    Mat image = imread("person.jpeg");
//    imshow("image", image);
//    waitKey(0);
//    destroyAllWindows();
//    cout << "Built with OpenCV " << CV_VERSION << endl;
//    Mat image;
//    VideoCapture capture;
//    capture.open(0);
//    if(capture.isOpened())
//    {
//        cout << "Capture is opened" << endl;
//        for(;;)
//        {
//            capture >> image;
//            if(image.empty())
//                break;
////            drawText(image);
//            imshow("Sample", image);
//            if(waitKey(10) >= 0)
//                break;
//        }
//    }
    return 0;
}

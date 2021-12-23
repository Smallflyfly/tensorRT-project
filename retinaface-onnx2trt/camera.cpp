//
// Created by smallflyfly on 2021/12/22.
//

#include <opencv2/opencv.hpp>
#include "retinaFace.h"

using namespace std;
using namespace cv;

VideoCapture open_cam_onboard(int width, int height) {
    string gst_str = "nvarguscamerasrc ! "
                     "video/x-raw(memory:NVMM), "
                     "width=(int)1920, height=(int)1080, "
                     "format=(string)NV12, framerate=(fraction)30/1 ! "
                     "nvvidconv flip-method=0 ! "
                     "video/x-raw, width=(int)800, height=(int)600, "
                     "format=(string)BGRx ! "
                     "videoconvert ! appsink";
    return VideoCapture(gst_str, CAP_GSTREAMER);
}

int main() {
    int height = 600;
    int width = 800;
    VideoCapture cap = open_cam_onboard(width, height);
    if (!cap.isOpened())
        exit(0);
    Mat frame;
    ICudaEngine *engine = readEngine();
    cout << "Load engine done!" << endl;
    while (cap.isOpened()) {
        cap >> frame;
        float scale = 1.0;
        int pw = 0, ph = 0;
        float *data = readImageData(frame, scale, pw, ph);
        IExecutionContext *context = engine->createExecutionContext();
        assert(context != nullptr);
        vector<RetinaFace::Detection> detections;
        vector<vector<float>> anchors = getAnchors();
        auto start = chrono::high_resolution_clock::now();
        detections = doInference(*context, data, anchors);
        auto end = chrono::high_resolution_clock::now();
        float t = chrono::duration<float, milli>(end - start).count();
        cout << "total cost time: " << t << endl;
        showResult(frame, detections, scale, pw, ph);

        imshow("frame", frame);
        if (cvWaitKey(0) == 27) {
            break;
        }
    }
    cap.release();
    destroyAllWindows();
    return 0;
}
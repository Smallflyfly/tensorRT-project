//
// Created by smallflyfly on 2021/11/16.
//

#include <iostream>
#include <fstream>
#include "logging.h"
#include <string>
#include "cuda_runtime.h"
#include <opencv2/opencv.hpp>
#include "yolox.h"

using namespace cv;
using namespace std;

static const char *trtFile = "yolox-helmet.trt";

static sample::Logger gLogger;

int main(int argc, char **argv) {
//    cudaSetDevice(0);
    String testFile = argv[1];
    size_t size{0};
    char *trtStream{nullptr};
    ifstream file(trtFile, ios::binary);
    if(file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtStream = new char[size];
        assert(trtStream != nullptr);
        file.read(trtStream, size);
        file.close();
    }

    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    ICudaEngine *engine = runtime->deserializeCudaEngine(trtStream, size);
    assert(engine != nullptr);

    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    delete[] trtStream;

    //camera or video
//    VideoCapture cap(videoFile);
//    if (!cap.isOpened()) {
//        cerr << "read camera/video error" << endl;
//    }
//    Mat frame;
//    cap >> frame;
//    int h = cap.get(CAP_PROP_FRAME_HEIGHT);
//    int w = cap.get(CAP_PROP_FRAME_WIDTH);
//    int fps = cap.get(CAP_PROP_FPS);
//    cout << h << " " << w << " " << fps << endl;

    vector<Yolox::Detection> detections;

    // image
    Mat image = imread(testFile);
    Mat im = image.clone();
    float scale = 1, pw = 0, ph = 0;
    float *data = prepareImage(image, scale, pw, ph);
    detections = doInference(*context, data);

    context->destroy();
    engine->destroy();
    runtime->destroy();

    fixDetection(detections, scale, pw, ph);
    showResult(detections, im);

    return 1;
}
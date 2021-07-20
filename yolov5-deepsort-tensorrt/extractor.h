//
// Created by smallflyfly on 2021/7/15.
//

#ifndef YOLOV5_ONNX2TRT_EXTRACTOR_H
#define YOLOV5_ONNX2TRT_EXTRACTOR_H

#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include "NvInfer.h"
#include "logging.h"
#include "cuda_runtime.h"
#include <numeric>
#include <vector>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace nvinfer1;

static const int EXTRACTOR_BATCH_SIZE = 1;
static const int EXTRACTOR_INPUT_W = 64;
static const int EXTRACTOR_INPUT_H = 128;
static const char *EXTRACTOR_INPUT_NAME = "input";
static const char *EXTRACTOR_OUTPUT_NAME = "output";
static const int64_t EXTRACTOR_OUTPUT_SIZE = 512;

static Logger gLogger;

#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            cerr << "Cuda failure: " << ret << endl; \
        } \
    } while(0)


ICudaEngine* readEngine(const string &trtFile);

float* readImageData(const cv::Mat &im, float &scale, int &pw, int &ph);

float* doInferenceExtractor(IExecutionContext &context, float *input);


#endif //YOLOV5_ONNX2TRT_EXTRACTOR_H

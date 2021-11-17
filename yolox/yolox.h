//
// Created by smallflyfly on 2021/11/16.
//

#ifndef YOLOX_YOLOX_H
#define YOLOX_YOLOX_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "NvInfer.h"
#include "cuda_runtime.h"

static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const int OUTPUT_SIZE = 8400 * 7;
static const char *INPUT_NAME = "input";
static const char *OUTPUT_NAME = "output";
static const int BATCH_SIZE = 1;

using namespace nvinfer1;

#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            std::cerr << "Cuda failure: " << ret << std::endl; \
        } \
    } while(0)

class Yolox {
public:
    class Detection {
        int classes;
        float x;
        float y;
        float w;
        float h;
        float prob;
        float xmax;
        float ymax;
    };
};

float* prepareImage(cv::Mat image, float &scale, float &pw, float &ph);

std::vector<Yolox::Detection> doInference(IExecutionContext &context, float *input);

#endif //YOLOX_YOLOX_H

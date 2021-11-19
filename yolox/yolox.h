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
static const int NUM_CLASSES = 2;

static const float CONF_THRESHOLD = 0.7;
static const float NMS_THRESHOLD = 0.7;

static const char *CLASSES[2] = {"HELMET", "NO-HELMET"};

static const float SHOW_COLOR[NUM_CLASSES][3] = {
        {0.8, 0.8, 0.9},
        {0.850, 0.8, 0.6}
};

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
    public:
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

std::vector<Yolox::Detection> postProcess(float *output);

void nms(std::vector<Yolox::Detection> &detections);

float iouCalculate(const Yolox::Detection &det1, const Yolox::Detection &det2);

void fixDetection(std::vector<Yolox::Detection> &detections, float scale, float pw, float ph);

void showResult(const std::vector<Yolox::Detection> &detections, cv::Mat &image);

#endif //YOLOX_YOLOX_H

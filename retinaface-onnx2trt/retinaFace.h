//
// Created by smallflyfly on 2021/7/13.
//

#ifndef RETINAFACE_ONNX2TRT_RETINAFACE_H
#define RETINAFACE_ONNX2TRT_RETINAFACE_H

#include <iostream>
#include <vector>
#include "NvInfer.h"
#include "logging.h"
#include <fstream>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"



using namespace nvinfer1;
using namespace std;

#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            cerr << "Cuda failure: " << ret << endl; \
        } \
    } while(0)

static sample::Logger gLogger;
static const char *TRT_FILE = "retinaFace_1output.trt";

static const int BATCH_SIZE = 1;
static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const char *INPUT_NAME = "input";
static const char *OUTPUT_NAME = "output";
static const int64_t OUTPUT_SIZE = 16800 * 16;
static const int64_t OUTPUT_CHANNELS = 16800;
static const float OBJ_THRESHOLD = 0.6;
static const float NMS_THRESHOLD = 0.4;

class RetinaFace {
public:
    class Bbox{
    private:
        float xmin;
        float ymin;
        float xmax;
        float ymax;
    public:
        float getXmin() const;

        void setXmin(float xmin);

        float getYmin() const;

        void setYmin(float ymin);

        float getXmax() const;

        void setXmax(float xmax);

        float getYmax() const;

        void setYmax(float ymax);
    };
    class LandMark{
    private:
        vector<float> landmarks;
    public:
        const vector<float> &getLandmarks() const;

        void setLandmarks(const vector<float> &landmarks);
    };
    class Detection {
        float prob;
        Bbox bbox;
        LandMark landMark;
    public:
        float getProb() const;

        void setProb(float prob);

        const Bbox &getBbox() const;

        void setBbox(const Bbox &bbox);

        const LandMark &getLandMark() const;

        void setLandMark(const LandMark &landMark);
    };
};

ICudaEngine* readEngine();

float iouCalculate(const RetinaFace::Detection &det1, const RetinaFace::Detection &det2);

void nms(vector<RetinaFace::Detection> &detections);

float *readImageData(cv::Mat &im, float &scale, int &pw, int &ph);

vector<vector<float>> getAnchors();

void decode(vector<vector<float>> &bboxes, const vector<vector<float>> &anchors);

vector<RetinaFace::Detection> postProcess(float *output, const vector<vector<float>> &anchors);

vector<RetinaFace::Detection> doInference(IExecutionContext &context, float *input, const vector<vector<float>> &anchors);

void showResult(const cv::Mat &im, const vector<RetinaFace::Detection> &detections, float scale, int pw, int ph);

#endif //RETINAFACE_ONNX2TRT_RETINAFACE_H

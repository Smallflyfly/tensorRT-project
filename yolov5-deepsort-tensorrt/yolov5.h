//
// Created by smallflyfly on 2021/6/28.
//

#ifndef YOLOV5_ONNX2TRT_YOLOV5_H
#define YOLOV5_ONNX2TRT_YOLOV5_H

#include "NvInfer.h"
#include <string>
#include "yaml-cpp/yaml.h"
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include "logging.h"
#include "cuda_runtime.h"
#include <numeric>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc.hpp"

#define max(a, b) a > b ? a : b
#define min(a, b) a < b ? a : b

static const char *ENGINE_FILE = "yolov5s.trt";
static const int YOLO_BATCH_SIZE = 1;
static const int YOLO_INPUT_W = 640;
static const int YOLO_INPUT_H = 640;
static const int64 YOLO_OUTPUT_SIZE = 604800;
static const char *YOLO_INPUT_NAME = "images";
static const char *YOLO_OUTPUT_NAME = "output";

#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            cerr << "Cuda failure: " << ret << endl; \
        } \
    } while(0)

using namespace std;
using namespace cv;
using namespace nvinfer1;


class Yolo {
public:
    class Detection {
    public:
        int getClasses() const {
            return classes;
        }

        void setClasses(int classes) {
            Detection::classes = classes;
        }

        float getX() const {
            return x;
        }

        void setX(float x) {
            Detection::x = x;
        }

        float getY() const {
            return y;
        }

        void setY(float y) {
            Detection::y = y;
        }

        float getW() const {
            return w;
        }

        void setW(float w) {
            Detection::w = w;
        }

        float getH() const {
            return h;
        }

        void setH(float h) {
            Detection::h = h;
        }

        float getProb() const {
            return prob;
        }

        void setProb(float prob) {
            Detection::prob = prob;
        }

        float getXmax() const {
            return xmax;
        }

        void setXmax(float xmax) {
            Detection::xmax = xmax;
        }

        float getYmax() const {
            return ymax;
        }

        void setYmax(float ymax) {
            Detection::ymax = ymax;
        }

        virtual ~Detection() {
        }

    private:
        int classes{};
        float x{};
        float y{};
        float w{};
        float h{};
        float prob{};
        float xmax{};
        float ymax{};
    };
    explicit Yolo(const string &config);

    ~Yolo();

    const vector<int> &getStrides() const;

    const vector<int> &getNumAnchors() const;

    const vector<vector<int>> &getAnchors() const;

    const vector<vector<int>> &getGrids() const;

    int getInputW() const;

    int getInputH() const;

    void readNames();

    int getCategory() const;

    float getObjThreshold() const;

    float getNmsThreshold() const;

    void nms(vector<Detection> &dets);

    float iouCalculate(const Detection &det1, const Detection &det2);

private:
    vector<int> strides;
    vector<int> numAnchors;
    vector<vector<int>> anchors;
    vector<vector<int>> grids;
    int inputW{640};
    int inputH{640};
    string nameFile;
    map<int, string> names;
    int category;
    float objThreshold;
    float nmsThreshold;
};

float *prepareImage(Mat &frame, float &scale, float &pw, float &ph);

vector<Yolo::Detection> postProcess(float *output, const int &outputSize);

vector<Yolo::Detection> doInferenceYolo(IExecutionContext &context, float *input);

void showResultYolo(const vector<Yolo::Detection> &dets, Mat &image);

vector<Yolo::Detection> fixDetections(vector<Yolo::Detection> &detections, float scale, float pw, float ph);

#endif //YOLOV5_ONNX2TRT_YOLOV5_H

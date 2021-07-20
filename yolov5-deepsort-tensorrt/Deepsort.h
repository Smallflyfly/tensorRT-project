//
// Created by smallflyfly on 2021/7/19.
//

#ifndef YOLOV5_ONNX2TRT_DEEPSORT_H
#define YOLOV5_ONNX2TRT_DEEPSORT_H

#include "yolov5.h"
#include "NearestNeighborDistanceMetric.h"
#include "Tracker.h"


class DeepSort {
private:
    float maxDist = 0.2;
    float minConfidence = 0.3;
    float nmsMaxOverlap = 1.0;
    float maxIouDistance = 0.7;
    int maxAge = 70;
    int nInit = 3;
    int nnBudget = 100;
    float maxCosineDistance;
    NearestNeighborDistanceMetric metric;
    Tracker tracker;


public:
    void update(const vector<Yolo::Detection> &detections, const Mat &im, IExecutionContext &extractorContext);
    vector<vector<float>> getFeatures(const vector<Yolo::Detection> &detections, const Mat &im, IExecutionContext &extractorContext);

    DeepSort(float maxDist, float minConfidence, float nmsMaxOverlap, float maxIouDistance, int nInit,
             int nnBudget);
};


#endif //YOLOV5_ONNX2TRT_DEEPSORT_H

//
// Created by smallflyfly on 2021/7/20.
//

#ifndef YOLOV5_ONNX2TRT_TRACKER_H
#define YOLOV5_ONNX2TRT_TRACKER_H

#include "NearestNeighborDistanceMetric.h"
#include "KalmanFilter.h"
#include "Track.h"


class Tracker {
private:
    NearestNeighborDistanceMetric metric;
    float maxIouDistance = 0.7;
    int maxAge = 70;
    int nInit = 3;
    KalmanFilter kf;
    int nextId = 1;
    vector<Track> tracks;

public:
    Tracker();
    explicit Tracker(NearestNeighborDistanceMetric metric, float maxIouDistance = 0.7, int maxAge = 70, int nInit = 3);
    void predict();
};


#endif //YOLOV5_ONNX2TRT_TRACKER_H

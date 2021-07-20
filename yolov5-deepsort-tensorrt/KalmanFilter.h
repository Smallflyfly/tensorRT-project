//
// Created by smallflyfly on 2021/7/20.
//

#ifndef YOLOV5_ONNX2TRT_KALMANFILTER_H
#define YOLOV5_ONNX2TRT_KALMANFILTER_H

#include <vector>

using namespace std;

class KalmanFilter {
private:
    vector<vector<float>> motionMat;
    vector<vector<float>> updateMat;
    float stdWeightPosition = 1.0 / 20;
    float stdWeightVelocity = 1.0 / 160;
public:
    KalmanFilter();
};


#endif //YOLOV5_ONNX2TRT_KALMANFILTER_H

//
// Created by smallflyfly on 2021/7/20.
//

#ifndef YOLOV5_ONNX2TRT_DETECTION_H
#define YOLOV5_ONNX2TRT_DETECTION_H

#include <vector>

using namespace std;

class Detection {
private:
    vector<float> tlwh;
    float confidence;
    vector<float> feature;
public:
    Detection(vector<float> tlwh, float confidence, vector<float> feature);
};


#endif //YOLOV5_ONNX2TRT_DETECTION_H

//
// Created by smallflyfly on 2021/7/19.
//

#ifndef YOLOV5_ONNX2TRT_NEARESTNEIGHBORDISTANCEMETRIC_H
#define YOLOV5_ONNX2TRT_NEARESTNEIGHBORDISTANCEMETRIC_H

#include <iostream>
#include <string>
#include <cassert>
#include <vector>
#include <map>

using namespace std;

class NearestNeighborDistanceMetric {
private:
    float matchingThreshold;
    int budget;
    // target feature
    map<int, vector<float>> samples;
public:
    vector<float> cosineDistance(vector<vector<float>> x, vector<vector<float>> y);
    NearestNeighborDistanceMetric(float matchingThreshold, int budget);
    NearestNeighborDistanceMetric();
};


#endif //YOLOV5_ONNX2TRT_NEARESTNEIGHBORDISTANCEMETRIC_H

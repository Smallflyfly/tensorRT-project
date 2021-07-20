//
// Created by smallflyfly on 2021/7/20.
//

#include "KalmanFilter.h"

KalmanFilter::KalmanFilter() {
    int nDim = 4;
    float dt = 1.0;
    // Create Kalman filter model matrices.
    for (int i = 0; i < 2*nDim; ++i) {
        vector<float> vec;
        for (int j = 0; j < 2*nDim; ++j) {
            if (i == j) {
                vec.push_back(1.0);
            } else {
                vec.push_back(0.0);
            }
        }
        this->motionMat.push_back(vec);
    }
    for (int i = 0; i < nDim; ++i) {
        this->motionMat[i][nDim+1] = dt;
    }
    for (int i = 0; i < nDim; ++i) {
        vector<float> vec;
        for (int j = 0; j < 2*nDim; ++j) {
            if (i == j) {
                vec.push_back(1.0);
            } else {
                vec.push_back(0.0);
            }
        }
        this->updateMat.push_back(vec);
    }
}

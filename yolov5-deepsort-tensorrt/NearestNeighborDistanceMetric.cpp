//
// Created by smallflyfly on 2021/7/19.
//

#include "NearestNeighborDistanceMetric.h"

#include <math.h>


float getMold(const vector<float> &vec) {
    int n = vec.size();
    float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += vec[i] * vec[i];
    }
}

vector<vector<float>> normalizeData(vector<vector<float>> vecs) {
    vector<vector<float>> result;
    for (int i = 0; i < vecs.size(); ++i) {
        vector<float> vec = vecs[i];
        float sum = 0.0;
        for (int j = 0; j < vec.size(); ++j) {
            sum += vec[j] * vec[i];
        }
        float x = sqrt(sum);
        for (int j = 0; j < vec.size(); ++j) {
            vec[j] /= (x + 1e-5);
        }
        result.push_back(vec);
    }
    return result;
}

vector<vector<float>> transpose(vector<vector<float>> x) {
    vector<vector<float>> vecs(x[0].size());
    for (int i = 0; i <x[0].size(); ++i) {
        for (int j = 0; j < x.size(); ++j) {
            vecs[i].push_back(x[j][i]);
        }
    }
}

float dot(const vector<float> &x, const vector<float> &y) {
    assert(x.size() == y.size());
    float d = 0.0;
    for (int i = 0; i < x.size(); ++i) {
        d += x[i] * y[i];
    }
}

vector<float> NearestNeighborDistanceMetric::cosineDistance(vector<vector<float>> x, vector<vector<float>> y) {
    assert(x[0].size() == y[0].size());
    vector<vector<float>> dotResult;
    x = normalizeData(x);
    y = normalizeData(y);
    // 1. - np.dot(a, b.T)
    // x N1 * M  y N2 * M  y.T M * N2
    // np.dot(N1*M, M*N2)
    // y = transpose(y);
    for (int i = 0; i < x.size(); ++i) {
        vector<float> vec;
        for (int j = 0; j < y.size(); ++j) {
            float d = dot(x[i], y[i]);
            vec.push_back(1.0 - d);
        }
        dotResult.push_back(vec);
    }
    vector<float> result;
    for (int i = 0; i < dotResult[0].size(); ++i) {
        float minValue = 99999.0;
        for (int j = 0; j < dotResult.size(); ++j) {
            minValue = min(minValue, dotResult[j][i]);
        }
        result.push_back(minValue);
    }
    return result;
}

NearestNeighborDistanceMetric::NearestNeighborDistanceMetric(float matchingThreshold, int budget) {
    this->matchingThreshold = matchingThreshold;
    this->budget = budget;
}

NearestNeighborDistanceMetric::NearestNeighborDistanceMetric() {

}

//
// Created by smallflyfly on 2021/7/20.
//

#include "Detection.h"

Detection::Detection(vector<float> tlwh, float confidence, vector<float> feature) {
    this->tlwh = tlwh;
    this->confidence = confidence;
    this->feature = feature;
}

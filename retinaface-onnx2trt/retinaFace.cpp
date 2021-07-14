//
// Created by smallflyfly on 2021/7/13.
//

#include "retinaFace.h"

const vector<float> &RetinaFace::LandMark::getLandmarks() const {
    return landmarks;
}

void RetinaFace::LandMark::setLandmarks(const vector<float> &landmarks) {
    LandMark::landmarks = landmarks;
}

float RetinaFace::Detection::getProb() const {
    return prob;
}

void RetinaFace::Detection::setProb(float prob) {
    Detection::prob = prob;
}

const RetinaFace::Bbox &RetinaFace::Detection::getBbox() const {
    return bbox;
}

void RetinaFace::Detection::setBbox(const RetinaFace::Bbox &bbox) {
    Detection::bbox = bbox;
}

const RetinaFace::LandMark &RetinaFace::Detection::getLandMark() const {
    return landMark;
}

void RetinaFace::Detection::setLandMark(const RetinaFace::LandMark &landMark) {
    Detection::landMark = landMark;
}

float RetinaFace::Bbox::getXmin() const {
    return xmin;
}

void RetinaFace::Bbox::setXmin(float xmin) {
    Bbox::xmin = xmin;
}

float RetinaFace::Bbox::getYmin() const {
    return ymin;
}

void RetinaFace::Bbox::setYmin(float ymin) {
    Bbox::ymin = ymin;
}

float RetinaFace::Bbox::getXmax() const {
    return xmax;
}

void RetinaFace::Bbox::setXmax(float xmax) {
    Bbox::xmax = xmax;
}

float RetinaFace::Bbox::getYmax() const {
    return ymax;
}

void RetinaFace::Bbox::setYmax(float ymax) {
    Bbox::ymax = ymax;
}

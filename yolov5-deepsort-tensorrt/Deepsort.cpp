//
// Created by smallflyfly on 2021/7/19.
//

#include "Deepsort.h"
#include "extractor.h"
#include "Detection.h"


void DeepSort::update(const vector<Yolo::Detection> &dets, const Mat &im, IExecutionContext &context) {
    int height = im.rows;
    int width = im.cols;
    vector<vector<float>> features = getFeatures(dets, im, context);
    cout << features.size() << endl;
    vector<Detection> detections;
    for (int i = 0; i < dets.size(); ++i) {
        Yolo::Detection det = dets[i];
        float xmin = det.getX();
        float ymin = det.getY();
        float w = det.getW();
        float h = det.getH();
        float conf = det.getProb();
        vector<float> tlwh{xmin, ymin, w, h};
        Detection detection(tlwh, conf, features[i]);
        detections.push_back(detection);
    }
    this->tracker
}

vector<vector<float>> DeepSort::getFeatures(const vector<Yolo::Detection> &detections, const Mat &im, IExecutionContext &context) {
    vector<vector<float>> features;
    for (int i = 0; i < detections.size(); ++i) {
        Yolo::Detection detection = detections[i];
        int xmin = detection.getX();
        int ymin = detection.getY();
        int w = detection.getW();
        int h = detection.getH();
        Mat imCrop;
        imCrop = im(Rect(xmin, ymin, w, h));
        float escale = 1.0;
        int epw, eph;
        float *input = readImageData(imCrop, escale, epw, eph);
        float *output = (float *) malloc(EXTRACTOR_OUTPUT_SIZE * sizeof(float));
        output = doInferenceExtractor(context, input);
        // feature 512-dim
        vector<float> feature;
        for (int j = 0; j < 512; ++j) {
            feature.push_back(output[i]);
        }
        features.push_back(feature);
    }
    return features;
}

DeepSort::DeepSort(float maxDist, float minConfidence, float nmsMaxOverlap, float maxIouDistance, int nInit,
                   int nnBudget) {
    this->minConfidence = minConfidence;
    this->nmsMaxOverlap = nmsMaxOverlap;
    this->maxCosineDistance = maxDist;
    this->metric = NearestNeighborDistanceMetric(this->maxCosineDistance, nnBudget);
    this->tracker = Tracker(this->metric, maxIouDistance, maxAge, nInit);
}

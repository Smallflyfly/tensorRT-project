//
// Created by smallflyfly on 2021/7/12.
//

#include <iostream>
#include <fstream>
#include "NvInfer.h"
#include "logging.h"
#include "opencv2/opencv.hpp"
#include "retinaFace.h"
#include "cuda_runtime.h"


using namespace std;
using namespace nvinfer1;

static Logger gLogger;

static const int BATCH_SIZE = 1;
static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const char *TRT_FILE = "retinaFace_1output.trt";
static const char *INPUT_NAME = "input";
static const char *OUTPUT_NAME = "output";
static const int64 OUTPUT_SIZE = 16800 * 16;
static const int64 OUTPUT_CHANNELS = 16800;
static const float OBJ_THRESHOLD = 0.6;
static const float NMS_THRESHOLD = 0.4;


ICudaEngine* readEngine() {
    string engineCached;
    ifstream file(TRT_FILE, ios::binary);
    while (file.peek() != EOF) {
        stringstream buffer;
        buffer << file.rdbuf();
        engineCached.append(buffer.str());
    }
    file.close();
    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    ICudaEngine *engine = runtime->deserializeCudaEngine(engineCached.data(), engineCached.size(), nullptr);
    assert(engine != nullptr);
    return engine;
}

float *readImageData(cv::Mat im, float &scale, int &pw, int &ph) {
    int w = im.cols, h = im.rows;
    scale = w > h ? INPUT_W * 1.0 / w : INPUT_H * 1.0 / h;
    cv::Mat resizeIm;
    cv::resize(im, resizeIm, cv::Size(0, 0), scale, scale);
    int w1 = resizeIm.cols, h1 = resizeIm.rows;
    pw = (int)((INPUT_W - w1) / 2), ph = (int)((INPUT_H - h1) / 2);
    cv::Mat newIm = cv::Mat::zeros(cv::Size(INPUT_W, INPUT_H), CV_8UC3);
    float *data = (float *) malloc(BATCH_SIZE * 3 * INPUT_W * INPUT_H * sizeof(float));
    resizeIm.copyTo(newIm(cv::Rect(pw, ph, resizeIm.cols, resizeIm.rows)));
//    cv::imshow("im", newIm);
//    cv::waitKey(0);
//    cv::destroyAllWindows();
    // - (104, 117, 123) n
    newIm.convertTo(newIm, CV_32FC3, 1.0);
    vector<cv::Mat> channels(3);
    vector<float> mean{-104, -117, -123};
    cv::split(newIm, channels);
    for (int i = 0; i < channels.size(); ++i) {
        channels[i].convertTo(channels[i], CV_32FC1, 1.0, mean[i]);
    }
    cv::merge(channels, newIm);
//    cv::imshow("im", newIm);
//    cv::waitKey(0);
//    cv::destroyAllWindows();
    int channelLength = INPUT_H * INPUT_W;
    vector<cv::Mat> splitIm = {
            cv::Mat(INPUT_H, INPUT_W, CV_32FC1, data + channelLength * 2),
            cv::Mat(INPUT_H, INPUT_W, CV_32FC1, data + channelLength * 1),
            cv::Mat(INPUT_H, INPUT_W, CV_32FC1, data + channelLength * 0)
    };

    cv::split(newIm, splitIm);
    return data;
}

vector<vector<float>> getAnchors() {
    vector<vector<int>> minSizes = {{16, 32}, {64, 128}, {256, 512}};
    vector<int> steps = {8, 16, 32};
    vector<vector<int>> featureMaps;
    for (int i = 0; i < steps.size(); ++i) {
        vector<int> featureMap;
        featureMap.push_back(INPUT_H / steps[i]);
        featureMap.push_back(INPUT_W / steps[i]);
        featureMaps.push_back(featureMap);
    }
    vector<vector<float>> anchors;
    // 3
//    int count = 0;
    for (int k = 0; k < featureMaps.size(); ++k) {
        vector<int> minSize = minSizes[k];
        // 80 40  20
        for (int i = 0; i < featureMaps[k][0]; ++i) {
            // 80 40 20
            for (int j = 0; j < featureMaps[k][1]; ++j) {
                // 2
                for (int m = 0; m < minSize.size(); ++m) {
                    float sx = minSize[m] * 1.0 / INPUT_W;
                    float sy = minSize[m] * 1.0 / INPUT_H;
                    float denseCx = (j + 0.5) * steps[k] / INPUT_W;
                    float denseCy = (i + 0.5) * steps[k] / INPUT_H;
                    vector<float> anchor = {denseCx, denseCy, sx, sy};
//                    cout << "anchors:" << endl;
//                    cout << denseCx << " " << denseCy << " " << sx << " " << sy << endl;
//                    count ++;
//                    if (count == 100) {
//                        return anchors;
//                    }
                    anchors.push_back(anchor);
                }
//                break;
//                return anchors;
            }
        }
    }
    return anchors;
}

void decode(vector<vector<float>> &bboxes, const vector<vector<float>> &anchors) {
    float variances0 = 0.1;
    float variances1 = 0.2;
    for (int i = 0; i < bboxes.size(); ++i) {
        bboxes[i][0] = anchors[i][0] + bboxes[i][0] * variances0 * anchors[i][2];
        bboxes[i][1] = anchors[i][1] + bboxes[i][1] * variances0 * anchors[i][3];
        bboxes[i][2] = anchors[i][2] * exp(bboxes[i][2] * variances1);
        bboxes[i][3] = anchors[i][3] * exp(bboxes[i][3] * variances1);

//        cout << bboxes[i][0] << " " << bboxes[i][1] << " " << bboxes[i][2] << " " << bboxes[i][3] << endl;
        // center point w h to xmin ymin xmax ymax
        bboxes[i][0] -= bboxes[i][2] / 2;
        bboxes[i][1] -= bboxes[i][3] / 2;
        bboxes[i][2] += bboxes[i][0];
        bboxes[i][3] += bboxes[i][1];

        bboxes[i][0] *= INPUT_W;
        bboxes[i][1] *= INPUT_H;
        bboxes[i][2] *= INPUT_W;
        bboxes[i][3] *= INPUT_H;

//        cout << bboxes[i][0] << " " << bboxes[i][1] << " " << bboxes[i][2] << " " << bboxes[i][3] << endl;
//        return;
    }
}

float iouCalculate(const RetinaFace::Detection &det1, const RetinaFace::Detection &det2) {
    RetinaFace::Bbox bbox1 = det1.getBbox();
    RetinaFace::Bbox bbox2 = det2.getBbox();
    float x11 = bbox1.getXmin();
    float y11 = bbox1.getYmin();
    float x12 = bbox1.getXmax();
    float y12 = bbox1.getYmax();

    float x21 = bbox2.getXmin();
    float y21 = bbox2.getYmin();
    float x22 = bbox2.getXmax();
    float y22 = bbox2.getYmax();

    float area1 = (x12 - x11) * (y12 - y11);
    float area2 = (x22 - x21) * (y22 - y21);

    float x1 = max(x11, x21);
    float x2 = min(x12, x22);
    float y1 = max(y11, y21);
    float y2 = min(y12, y22);

    if (x1 >= x2 || y1 >= y2) {
        return 0.0;
    } else {
        float overlap = (x2 - x1) * (y2 - y1);
        return (float) (overlap / (area1 + area2 - overlap + 1e-5));
    }
}

void nms(vector<RetinaFace::Detection> &detections) {
    sort(detections.begin(), detections.end(), [=](const RetinaFace::Detection &d1, const RetinaFace::Detection &d2) {
        return d1.getProb() > d2.getProb();
    });
    for (int i = 0; i < detections.size(); ++i) {
        for (int j = i+1; j < detections.size(); ++j) {
            float iou = iouCalculate(detections[i], detections[j]);
            if (iou > NMS_THRESHOLD) {
                detections[j].setProb(0.0);
            }
        }
    }
    detections.erase(remove_if(detections.begin(), detections.end(), [](const RetinaFace::Detection &det)
    { return det.getProb() == 0; }), detections.end());
}

vector<RetinaFace::Detection> postProcess(float *output, const vector<vector<float>> &anchors) {
    cout << "post process" << endl;
    vector<RetinaFace::Detection> detections;
    // 16800 * 16
    vector<vector<float>> probs;
    vector<vector<float>> bboxes;
    vector<vector<float>> landmarks;
    vector<vector<float>> leftAnchors;
    int j = 0;
    for (int64 i = 0; i < OUTPUT_SIZE; i+=16, j++) {
//        cout << output[i+5] << endl;
        if (output[i+5] <= OBJ_THRESHOLD) {
            continue;
        }
        // 0-3 bbox 4-5 conf 6-15 landmarks
        vector<float> bbox = {output[i+0], output[i+1], output[i+2], output[i+3]};
        // i+4 no face conf
        vector<float> prob = {output[i+5]};
        vector<float> landmark = {
                output[i+6], output[i+7], output[i+8], output[i+9], output[i+10], output[i+11],output[i+12], output[i+13],
                output[i+14], output[i+15]
        };
        probs.push_back(prob);
        bboxes.push_back(bbox);
        landmarks.push_back(landmark);
        leftAnchors.push_back(anchors[j]);
    }
//    cout << probs.size() << endl;
//    cout << bboxes.size() << endl;
//    cout << landmarks.size() << endl;
//    cout << anchors.size() << endl;

    decode(bboxes, leftAnchors);
    for (int i = 0; i < bboxes.size(); ++i) {
        float prob = probs[i][0];
//        if (prob <= OBJ_THRESHOLD) {
//            continue;
//        }
        RetinaFace::Detection detection;
        detection.setProb(prob);
        float xmin = bboxes[i][0];
        float ymin = bboxes[i][1];
        float xmax = bboxes[i][2];
        float ymax = bboxes[i][3];
        RetinaFace::Bbox bbox{};
        bbox.setXmin(xmin);
        bbox.setYmin(ymin);
        bbox.setXmax(xmax);
        bbox.setYmax(ymax);
        // ignore landmarks  useless
        detection.setBbox(bbox);
        detections.push_back(detection);
    }
//    cout << "detection size" << endl;
//    cout << detections.size() << endl;
    nms(detections);
    return detections;
}

vector<RetinaFace::Detection> doInference(IExecutionContext &context, float *input, const vector<vector<float>> &anchors) {
    const ICudaEngine &engine = context.getEngine();
//    cout << engine.getNbBindings() << endl;
    // input 1 +  (output 3  merge into 1)
    assert(engine.getNbBindings() == 2);
    void *buffers[2];
    const int inputIndex = engine.getBindingIndex(INPUT_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_NAME);

    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], OUTPUT_SIZE * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(BATCH_SIZE, buffers, stream, nullptr);
    float *output = (float *) malloc(OUTPUT_SIZE * sizeof(float));
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

    auto start = chrono::high_resolution_clock::now();
    vector<RetinaFace::Detection> detections = postProcess(output, anchors);
    auto end = chrono::high_resolution_clock::now();
    float t = chrono::duration<float, milli>(end - start).count();
    cout << "post process cost time: " << t << endl;
    return detections;
}

void showResult(const cv::Mat &im, const vector<RetinaFace::Detection> &detections, float scale, int pw, int ph) {
    // - padding    / scale
    for (int i = 0; i < detections.size(); ++i) {
        RetinaFace::Detection detection = detections[i];
        float xmin = detection.getBbox().getXmin();
        float ymin = detection.getBbox().getYmin();
        float xmax = detection.getBbox().getXmax();
        float ymax = detection.getBbox().getYmax();
        xmin = (xmin - pw * 1.0) / scale;
        ymin = (ymin - ph * 1.0) / scale;
        xmax = (xmax - pw * 1.0) / scale;
        ymax = (ymax - ph * 1.0) / scale;
        float w = xmax - xmin;
        float h = ymax - ymin;
        cv::Rect rec(xmin, ymin, w, h);
        cv::rectangle(im, rec, cv::Scalar(255, 0, 0), 1);
    }
}

int main() {
    ICudaEngine *engine = readEngine();
    cout << "Load engine done!" << endl;
    float scale = 1.0;
    int pw = 0, ph = 0;
    for (int i = 0; i < 10; ++i) {
        cv::Mat im = cv::imread("test.jpg");
        float *data = readImageData(im, scale, pw, ph);
        IExecutionContext *context = engine->createExecutionContext();
        assert(context != nullptr);
        vector<RetinaFace::Detection> detections;
        vector<vector<float>> anchors = getAnchors();
        auto start = chrono::high_resolution_clock::now();
        detections = doInference(*context, data, anchors);
        auto end = chrono::high_resolution_clock::now();
        float t = chrono::duration<float, milli>(end - start).count();
        cout << "total cost time: " << t << endl;
        showResult(im, detections, scale, pw, ph);
        cv::imshow("im", im);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    return 0;

}
//
// Created by fangpf on 2021/4/29.
//

#ifndef RETINAFACE_RETINAFACE_H
#define RETINAFACE_RETINAFACE_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <dirent.h>
#include "decode.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

static inline cv::Mat preprocessImage(cv::Mat &image, int inputW, int inputH) {
    int w, h, x, y;
    float rW = inputW / (image.cols * 1.0);
    float rH = inputH / (image.rows * 1.0);
    if (rH > rW) {
        w = inputW;
        h = rW * image.rows;
        x = 0;
        y = (inputH - h) / 2;
    } else {
        h = inputH;
        w = rH * image.cols;
        x = (inputW - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(image, re, re.size(), 0, 0,cv::INTER_LINEAR);
    cv::Mat out(inputH, inputW, CV_8UC3, cv::Scalar(128, 128, 128));
    // Rect(x, y, w, h)
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

bool cmp(const decodeplugin::Detection &a, const decodeplugin::Detection &b) {
    return a.class_confidence > b.class_confidence;
}

static float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
            std::max(lbox[0], rbox[0]),
            std::min(lbox[2], rbox[2]),
            std::max(lbox[1], rbox[1]),
            std::min(lbox[3], rbox[3])
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1]) return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / ((lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) - interBoxS + 1e-5f);
}

static inline void nms(std::vector<decodeplugin::Detection> &res, float *output, float nms_thresh = 0.4) {
    std::vector<decodeplugin::Detection> dets;
    for (int i = 0; i < output[0]; ++i) {
        if (output[15 * i + 1 + 4] <= 0.1) continue;
        decodeplugin::Detection det;
        memcpy(&det, &output[15 * i + 1], sizeof(decodeplugin::Detection));
        dets.push_back(det);
    }
    std::sort(dets.begin(), dets.end(), cmp);
    for (size_t m = 0; m < dets.size(); m++) {
        auto &item = dets[m];
        res.push_back(item);
        for (size_t n = m + 1; n < dets.size(); n++) {
            if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                dets.erase(dets.begin() + n);
                n--;
            }
        }
    }
}

static inline cv::Rect getRectAdaptLandmark(cv::Mat &img, int inputW, int inputH, float bbox[4], float lmk[10]) {
    int l, r, t, b;
    float rW = inputW / (img.cols * 1.0);
    float rH = inputH / (img.rows * 1.0);
    if (rH > rW) {
        l = bbox[0] / rW;
        r = bbox[2] / rW;
        t = (bbox[1] - (inputH - rW * img.rows) / 2) / rW;
        b = (bbox[3] - (inputH - rW * img.rows) / 2) / rW;
        for (int i = 0; i < 10; i += 2) {
            lmk[i] /= rW;
            lmk[i+1] = (lmk[i+1] - (inputH - rW * img.rows) / 2) / rW;
        }
    } else {
        l = (bbox[0] - (inputW - rH * img.cols) / 2) / rH;
        r = (bbox[2] - (inputW - rH * img.cols) / 2) / rH;
        t = bbox[1] / rH;
        b = bbox[3] / rH;
        for (int i = 0; i < 10; i += 2) {
            lmk[i] = (lmk[i] - (inputW - rH * img.cols) / 2) / rH;
            lmk[i+1] /= rH;
        }
    }
    return cv::Rect(l, t, r-l, b-t);
}

#endif //RETINAFACE_RETINAFACE_H

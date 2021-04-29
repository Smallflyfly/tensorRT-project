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
    cv::resize(image, re, re.size(), cv::INTER_LINEAR);
    cv::Mat out(inputH, inputW, CV_8UC3, cv::Scalar(128, 128, 128));
    // Rect(x, y, w, h)
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}



#endif //RETINAFACE_RETINAFACE_H

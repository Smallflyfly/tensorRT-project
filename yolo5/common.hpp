//
// Created by smallflyfly on 2021/4/22.
//

#ifndef YOLO5_COMMON_HPP
#define YOLO5_COMMON_HPP

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "yololayer.h"


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

using namespace nvinfer1;

IScaleLayer* addBatchNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string layerName, float eps) {
    float *gamma = (float *)weightMap[layerName + ".weight"].values;
    float *beta = (float *)weightMap[layerName + ".bias"].values;
    float *mean = (float *)weightMap[layerName + ".running_mean"].values;
    float *var = (float *)weightMap[layerName + ".running_var"].values;
    int len = weightMap[layerName + ".running_var"].count;

    float *scval = reinterpret_cast<float *>(malloc(sizeof(float ) * len));
    for (int i = 0; i < len; ++i) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float *>(malloc(sizeof(float ) * len));
    for (int i = 0; i < len; ++i) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[layerName + ".scale"] = scale;
    weightMap[layerName + ".shift"] = shift;
    weightMap[layerName + ".power"] = power;
    IScaleLayer *scale1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale1);
    return scale1;
}

ILayer* convBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outCh, int kSize,
                  int s, int g, std::string layerName) {
    Weights emptyWt{DataType::kFLOAT, nullptr, 0};
    int p = kSize / 2;
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outCh, DimsHW{kSize, kSize}, weightMap[layerName+".conv.weight"], emptyWt);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(g);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), layerName+".bn", 1e-5);
    // hard swish = x * hard_sigmoid
    auto hsig = network->addActivation(*bn1->getOutput(0), ActivationType::kHARD_SIGMOID);
    assert(hsig);
    hsig->setAlpha(1.0 / 6.0);
    hsig->setBeta(0.5);
    auto ew = network->addElementWise(*bn1->getOutput(0), *hsig->getOutput(0), ElementWiseOperation::kPROD);
    return ew;
}

ILayer* focus(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int inCh,
              int outCh, int kSize, const std::string& layerName) {
    ISliceLayer* s1 = network->addSlice(input, Dims3{0, 0, 0}, Dims3{inCh, Yolo::INPUT_H / 2,
                                                                     Yolo::INPUT_W / 2}, Dims3{1, 2, 2});
    ISliceLayer* s2 = network->addSlice(input, Dims3{0, 1, 0}, Dims3{inCh,Yolo::INPUT_H / 2,
                                                                     Yolo::INPUT_W / 2}, Dims3{1, 2, 2});
    ISliceLayer* s3 = network->addSlice(input, Dims3{0, 0, 1}, Dims3{inCh, Yolo::INPUT_H / 2,
                                                                     Yolo::INPUT_W / 2}, Dims3{1, 2, 2});
    ISliceLayer* s4 = network->addSlice(input, Dims3{0, 1, 0}, Dims3{inCh, Yolo::INPUT_H / 2,
                                                                     Yolo::INPUT_W / 2}, Dims3{1, 2, 2});
    ITensor* inputTensors[] = {s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);
    auto conv = convBlock(network, weightMap, *cat->getOutput(0), outCh, kSize, 1, 1, layerName+".conv");

    return conv;
}

ILayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1,
                   int c2, bool shortcut, int g, float e, std::string layerName) {
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, layerName + ".cv1");
    auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), c2, 3, 1, g, layerName + ".cv2");
    if (shortcut && c1 == c2) {
        auto ew = network->addElementWise(input, *cv2->getOutput(0), ElementWiseOperation::kSUM);
        return ew;
    }
    return cv2;
}

ILayer* bottleneckCSP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1,
                      int c2, int n, bool shortcut, int g, float e, std::string layerName) {
    Weights emptywt{DataType::kFLOAT, nullptr, 0};
    int c_ = (int)(float)c2 * e;
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, layerName);
    auto cv2 = network->addConvolutionNd(input, c_, DimsHW{1, 1},
                                         weightMap[layerName + ".cv2.weight"], emptywt);
    ITensor *y1 = cv1->getOutput(0);
    for (int i = 0; i < n; ++i) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, layerName+".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }
    auto cv3 = network->addConvolutionNd(*y1, c_, DimsHW{1, 1}, weightMap[layerName + ".cv3.weight"], emptywt);

    ITensor* inputTensors[] = {cv3->getOutput(0), cv2->getOutput(0)};
    auto cat = netwo#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

using namespace nvinfer1;rk->addConcatenation(inputTensors, 2);

    IScaleLayer* bn = addBatchNorm2d(network, weightMap, *cat->getOutput(0), layerName+".bn", 1e-5);
    auto leakReLu = network->addActivation(*bn->getOutput(0), ActivationType::kLEAKY_RELU);
    leakReLu->setAlpha(0.1);

    auto cv4 = convBlock(network, weightMap, *leakReLu->getOutput(0), c2, 1, 1, 1, layerName+".cv4");
    return cv4;
}

ILayer* SPP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int k1, int k2, int k3, std::string layerName) {
    int c_ = c1 / 2;
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, layerName+".cv1");

    auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{k1, k1});
    pool1->setPaddingNd(DimsHW{k1 / 2, k1 / 2});
    pool1->setStrideNd(DimsHW{1, 1});

    auto pool2 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{k2, k2});
    pool2->setPaddingNd(DimsHW{k2 / 2, k2 / 2});
    pool2->setStrideNd(DimsHW{1, 1});

    auto pool3 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{k3, k3});
    pool3->setPaddingNd(DimsHW{k3 / 2, k3 / 2});
    pool3->setStrideNd(DimsHW{1, 1});

    ITensor* inputTensors[] = {cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);

    auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, layerName + ".cv2");

    return cv2;
}

#endif //YOLO5_COMMON_HPP

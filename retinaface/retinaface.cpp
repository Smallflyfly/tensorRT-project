//
// Created by fangpf on 2021/4/27.
//

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"

#define DEVICE 0
#define BATCH_SIZE 1

using namespace std;
using namespace nvinfer1;

static Logger gLogger;
static const char* INPUT_BLOB_NAME = "input";
static const char* OUTPUT_BLOB_NAME = "output";
static const int INPUT_W = 256;
static const int INPUT_H = 128;



map<string, Weights> loadWeight(const string& weightFile) {
    cout << "Loading weights: " << weightFile << endl;

    map<string, Weights> weightMap;

    // open weight file
    ifstream input(weightFile);
    assert(input.is_open() && "can not open weight file");

    // read num of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight file");

    while (count --) {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t  size;

        // read name and type of blob
        string name;
        input >> name >> dec >> size;
        wt.type = DataType::kFLOAT;

        // load blob
        uint32_t *val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (int i = 0; i < size; ++i) {
            input >> hex >> val[i];
        }
        wt.values = val;
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBN(INetworkDefinition *network, map<string, Weights> &weightMap, ITensor &input, const string &layerName, float eps) {
    float *gamma = (float *)weightMap[layerName + ".weight"].values;
    float *beta = (float *)weightMap[layerName + ".bias"].values;
    float *mean = (float *)weightMap[layerName + ".running_mean"].values;
    float *var = (float *)weightMap[layerName + ".running_var"].values;

    int len = weightMap[layerName + ".weight"].count;

    auto *scval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    auto *shval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    auto *pval = reinterpret_cast<float *>(malloc(sizeof(float ) * len));

    for (int i = 0; i < len; ++i) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
        pval[i] = 1.0;
    }
    Weights scale{DataType::kFLOAT, scval, len};
    Weights shift{DataType::kFLOAT, shval, len};
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[layerName + ".scale"] = scale;
    weightMap[layerName + ".shift"] = shift;
    weightMap[layerName + ".power"] = power;

    IScaleLayer *scaleLayer = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scaleLayer);

    return scaleLayer;
}


IActivationLayer* bottleneck(INetworkDefinition *network, map<string, Weights>& weightMap, ITensor &input, int inCh, int outCh, int stride, const string &layerName) {
    Weights emptyWt{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer *conv1 = network->addConvolutionNd(input, 64, DimsHW{1, 1}, weightMap[layerName + ".conv1.weight"], emptyWt);
    assert(conv1);

    IScaleLayer *bn1 = addBN(network, weightMap, *conv1->getOutput(0), layerName + ".bn1", 1e-5);

    IActivationLayer *relu1 =
}


// just use tensorrt api to construct network
ICudaEngine* createEngine(int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dataType) {
    INetworkDefinition *network = builder->createNetworkV2(0u);

    // input
    ITensor *data = network->addInput(INPUT_BLOB_NAME, dataType, Dims{3, INPUT_H, INPUT_W});
    assert(data);

    map<string, Weights> weightMap = loadWeight("retinaface.wts");
    Weights emptyWt{DataType::kFLOAT, nullptr, 0};

    // create resnet50 see resnet50
    IConvolutionLayer *conv1 = network->addConvolutionNd(*data, 64, DimsHW{7, 7}, weightMap["body.conv1.weight"], emptyWt);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});

    IScaleLayer *bn1 = addBN(network, weightMap, *conv1->getOutput(0), "body.layer1.0.bn1", 1e-5);
    assert(bn1);

    // relu
    IActivationLayer *relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IPoolingLayer *pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});

    IActivationLayer *x;
    x = bottleneck(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "body.layer1.0.");





}


void APIToModel(int maxBatchSize, IHostMemory **modelStream) {
    // create builder
    IBuilder *builder = createInferBuilder(gLogger);
    // config
    IBuilderConfig *config = builder->createBuilderConfig();

    // create engine
    ICudaEngine *engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);


}


int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create model user api and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    // -s generate engine file
    if (string(argv[1]) == "-s") {
        IHostMemory *modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream);

    }

}
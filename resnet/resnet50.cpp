//
// Created by fangpf on 2021/4/25.
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


using namespace std;
using namespace nvinfer1;
using namespace sample;

static sample::Logger gLogger;
static const char* INPUT_BLOB_NAME = "input";
static const int INPUT_H = 256;
static const int INPUT_W = 256;


map<string, Weights> loadWeight(const string& weightFile) {
    cout << "Loading weight file " << weightFile << endl;
    map<string, Weights> weightMap;

    // open weight file
    ifstream input(weightFile);
    assert(input.is_open() && "unable load weight file");

    // read number of weight blob
    int count;
    input >> count;
    assert(count > 0 && "Invalid weight map file");

    while (count --) {
        Weights weight{DataType::kFLOAT, nullptr, 0};
        u_int32_t size;
        // read name and type of blob
        string name;
        input >> name >> dec >> size;
        weight.type = DataType::kFLOAT;

        //load blob
        uint32_t *val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (int i = 0; i < size; ++i) {
            input >> hex >> val[i];
        }
        weight.values = val;
        weight.count = size;
        weightMap[name] = weight;
    }
    return weightMap;
}

IScaleLayer* addBN2d(INetworkDefinition *network, map<string, Weights> weightMap, ITensor& input, const string& layerName, float eps) {
    float *gamma = (float *)weightMap[layerName + ".weight"].values;
    float *beta = (float *)weightMap[layerName + ".bias"].values;
    float *mean = (float *)weightMap[layerName + ".running_mean"].values;
    float *var = (float *)weightMap[layerName + ".running_var"].values;

    int len = weightMap[layerName + ".running_var"].count;
    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; ++i) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; ++i) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i <len; ++i) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[layerName + ".scale"] = scale;
    weightMap[layerName + ".shift"] = shift;
    weightMap[layerName + ".power"] = power;

    IScaleLayer *scaleLayer =  network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scaleLayer);

    return scaleLayer;
}

IActivationLayer* bottleneck(INetworkDefinition *network, map<string, Weights>& weightMap, ITensor &input, int inCh, int outCh, int stride, const string &layerName) {
    Weights wtempty{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer *conv1 = network->addConvolutionNd(input, outCh, DimsHW{1, 1}, weightMap[layerName + "conv1.weight"], wtempty);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1, 1});
    conv1->setPaddingNd(DimsHW{0, 0});

    IScaleLayer *bn1 = addBN2d(network, weightMap, *conv1->getOutput(0), layerName + "bn1", 1e-5);
    assert(bn1);

    IActivationLayer *relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer *conv2 = network->addConvolutionNd(*relu1->getOutput(0), 64, DimsHW{3, 3}, weightMap[layerName + "conv2.weight"], wtempty);
    assert(conv2);
    conv2->setStrideNd(DimsHW{1, 1});
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer *bn2 = addBN2d(network, weightMap, *conv2->getOutput(0), layerName + "bn2", 1e-5);
    assert(bn2);

    IActivationLayer *relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer *conv3 = network->addConvolutionNd(*relu2->getOutput(0), 64, DimsHW{1, 1}, weightMap[layerName + "conv3.weight"], wtempty);
    assert(conv3);
    conv3->setStrideNd(DimsHW{1, 1});
    conv3->setPaddingNd(DimsHW{0, 0});

    IScaleLayer *bn3 = addBN2d(network, weightMap, *conv3->getOutput(0), "bn3", 1e-5);
    assert(bn3);


}

// create engine
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig* config, DataType dtype) {
    const auto explictBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explictBatch);
    ITensor* data = network->addInput(INPUT_BLOB_NAME, DataType::kFLOAT, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    map<string, Weights> weightMap = loadWeight("resnet50.wts");

    Weights wtempty{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer *conv1 = network->addConvolutionNd(*data, 64, DimsHW{7, 7}, weightMap["conv1.weight"], wtempty);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setNbGroups(1);
    conv1->setPaddingNd(DimsHW{3, 3});

    IScaleLayer* bn1 = addBN2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);

    IActivationLayer *relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // max pool
    IPoolingLayer *pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});

    IActivationLayer *x = bottleneck(network, weightMap, *pool1->getOutput(0), 64, 64, "layer1.0.");



}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);

    IBuilderConfig* config = builder->createBuilderConfig();

    //create engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "arguments not right" <<endl;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
    }
}
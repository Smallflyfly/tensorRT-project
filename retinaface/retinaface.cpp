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
#include "decode.h"

#define DEVICE 0
#define BATCH_SIZE 1
#define USE_FP16

using namespace std;
using namespace nvinfer1;

static Logger gLogger;
static const char* INPUT_BLOB_NAME = "input";
static const char* OUTPUT_BLOB_NAME = "output";
static const int INPUT_W = 640;
static const int INPUT_H = 480;



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

    int len = weightMap[layerName + ".running_var"].count;

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

    IConvolutionLayer *conv1 = network->addConvolutionNd(input, outCh, DimsHW{1, 1}, weightMap[layerName + ".conv1.weight"], emptyWt);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1, 1}); // 差异点

    IScaleLayer *bn1 = addBN(network, weightMap, *conv1->getOutput(0), layerName + ".bn1", 1e-5);
    IActivationLayer *relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer *conv2 = network->addConvolutionNd(*relu1->getOutput(0), outCh, DimsHW{3, 3}, weightMap[layerName + ".conv2.weight"], emptyWt);
    assert(conv2);
    conv2->setStrideNd(DimsHW{stride, stride});
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer *bn2 = addBN(network, weightMap, *conv2->getOutput(0), layerName + ".bn2", 1e-5);
    IActivationLayer *relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer *conv3 = network->addConvolutionNd(*relu2->getOutput(0), outCh * 4, DimsHW{1, 1}, weightMap[layerName + ".conv3.weight"], emptyWt);
    assert(conv3);
    conv3->setStrideNd(DimsHW{1, 1});

    IScaleLayer *bn3 = addBN(network, weightMap, *conv3->getOutput(0), layerName + ".bn3", 1e-5);

    IElementWiseLayer *ew1;
    if (stride != 1 || inCh != outCh * 4) {
        // downsample
        IConvolutionLayer *conv4 = network->addConvolutionNd(input, outCh * 4, DimsHW{1, 1}, weightMap[layerName + ".downsample.0.weight"], emptyWt);
        assert(conv4);
        conv4->setStrideNd(DimsHW{stride, stride});

        IScaleLayer *bn4 = addBN(network, weightMap, *conv4->getOutput(0), layerName + ".downsample.1", 1e-5);

        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(*bn3->getOutput(0), input, ElementWiseOperation::kSUM);
    }

    IActivationLayer *relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);

    return relu3;
}

ILayer* conv_bn_relu(INetworkDefinition *network, map<string, Weights> &weightMap, ITensor &input, int outCh, int ksize, int stride, int padding, bool useRelu, string layerName) {
    Weights emptyWt{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer *conv = network->addConvolutionNd(input, outCh, DimsHW{ksize, ksize}, weightMap[layerName + ".weight"], emptyWt);
    assert(conv);
    conv->setStrideNd(DimsHW{stride, stride});
    conv->setPaddingNd(DimsHW{padding, padding});

    // .0 变为 .1操作
    layerName.replace(layerName.length()-1, layerName.length() - 1, "1");
    cout << layerName << endl;
    IScaleLayer *bn = addBN(network, weightMap, *conv->getOutput(0), layerName, 1e-5);

    if (!useRelu) return bn;

    IActivationLayer *relu = network->addActivation(*bn->getOutput(0), ActivationType::kRELU);
    assert(relu);

    return relu;
}

IActivationLayer* ssh(INetworkDefinition *network, map<string, Weights> &weightMap, ITensor &input, int outCh) {
    assert(outCh % 4 == 0 && "outChannels must % 4 == 0");
    ILayer *conv3X3 = conv_bn_relu(network, weightMap, input, outCh / 2, 3, 1, 1, false, "ssh1.conv3X3.0");
    ILayer *conv5X5_1 = conv_bn_relu(network, weightMap, input, outCh / 4, 3, 1, 1, true, "ssh1.conv5X5_1.0");
    ILayer *conv5X5_2 = conv_bn_relu(network, weightMap, *conv5X5_1->getOutput(0), outCh / 4, 3, 1, 1, false, "ssh1.conv5X5_2.0");
    ILayer *conv7X7_2 = conv_bn_relu(network, weightMap, *conv5X5_1->getOutput(0), outCh / 4, 3, 1, 1, true, "ssh1.conv7X7_2.0");
    ILayer *conv7X7_3 = conv_bn_relu(network, weightMap, *conv7X7_2->getOutput(0), outCh / 4, 3, 1, 1, false, "ssh1.conv7x7_3.0");

    ITensor *inputTensors[] = {conv3X3->getOutput(0), conv5X5_2->getOutput(0), conv7X7_3->getOutput(0)};
    IConcatenationLayer *cat = network->addConcatenation(inputTensors, 3);
    assert(cat);

    IActivationLayer *relu = network->addActivation(*cat->getOutput(0), ActivationType::kRELU);
    assert(relu);

    return relu;
}

IConvolutionLayer* bboxHead(INetworkDefinition *network, map<string, Weights> &weightMap, ITensor &input, int anchorNum, const string &layerName) {
    IConvolutionLayer *conv = network->addConvolutionNd(input, anchorNum * 4, DimsHW{1, 1}, weightMap[layerName + ".weight"], weightMap[layerName + ".bias"]);
    assert(conv);
    conv->setStrideNd(DimsHW{1, 1});

    return conv;
}

IConvolutionLayer* clsHead(INetworkDefinition *network, map<string, Weights> &weightMap, ITensor &input, int anchorNum, const string &layerName) {
    IConvolutionLayer *conv = network->addConvolutionNd(input, anchorNum * 2, DimsHW{1, 1}, weightMap[layerName + ".weight"], weightMap[layerName + ".bias"]);
    assert(conv);
    conv->setStrideNd(DimsHW{1, 1});

    return conv;
}

IConvolutionLayer* lmHead(INetworkDefinition *network, map<string, Weights> &weightMap, ITensor &input, int anchorNum, const string &layerName) {
    IConvolutionLayer *conv = network->addConvolutionNd(input, anchorNum * 10, DimsHW{1, 1}, weightMap[layerName + ".weight"], weightMap[layerName + ".bias"]);
    assert(conv);
    conv->setStrideNd(DimsHW{1, 1});

    return conv;
}


// just use tensorrt api to construct network
ICudaEngine* createEngine(int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dataType) {
    INetworkDefinition *network = builder->createNetworkV2(0U);

    // input
    ITensor *data = network->addInput(INPUT_BLOB_NAME, dataType, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    map<string, Weights> weightMap = loadWeight("retinaface.wts");
    Weights emptyWt{DataType::kFLOAT, nullptr, 0};

    // create resnet50 see resnet50
    IConvolutionLayer *conv1 = network->addConvolutionNd(*data, 64, DimsHW{7, 7}, weightMap["body.conv1.weight"], emptyWt);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});

    IScaleLayer *bn1 = addBN(network, weightMap, *conv1->getOutput(0), "body.bn1", 1e-5);

    // relu
    IActivationLayer *relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IPoolingLayer *pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});

    IActivationLayer *x;
    // blocks = [3, 4, 6, 3]
    x = bottleneck(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "body.layer1.0");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "body.layer1.1");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "body.layer1.2");

    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 128, 2, "body.layer2.0");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "body.layer2.1");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "body.layer2.2");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "body.layer2.3");
    IActivationLayer *layer2 = x;

    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 256, 2, "body.layer3.0");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.1");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.2");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.3");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.4");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.5");
    IActivationLayer *layer3 = x;

    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 512, 2, "body.layer4.0");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "body.layer4.1");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "body.layer4.2");
    IActivationLayer *layer4 = x;

    // FPN
    auto output1 = conv_bn_relu(network, weightMap, *layer2->getOutput(0), 256, 1, 1, 0, true, "fpn.output1.0");
    auto output2 = conv_bn_relu(network, weightMap, *layer3->getOutput(0), 256, 1, 1, 0, true, "fpn.output2.0");
    auto output3 = conv_bn_relu(network, weightMap, *layer4->getOutput(0), 256, 1, 1, 0, true, "fpn.output3.0");

    // up
    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; ++i) {
        deval[i] = 1.0;
    }

    Weights deconwts{DataType::kFLOAT, deval, 256 * 2 * 2};
    IDeconvolutionLayer *up3 = network->addDeconvolutionNd(*output3->getOutput(0), 256, DimsHW{2, 2}, deconwts, emptyWt);
    assert(up3);
    up3->setStrideNd(DimsHW{2, 2});
    up3->setNbGroups(256);
    weightMap["up3"] = deconwts;

    output2 = network->addElementWise(*output2->getOutput(0), *up3->getOutput(0), ElementWiseOperation::kSUM);
    assert(output2);
    // merge 2
    output2 = conv_bn_relu(network, weightMap, *output2->getOutput(0), 256, 3, 1, 1, true, "fpn.merge2.0");

    IDeconvolutionLayer *up2 = network->addDeconvolutionNd(*output2->getOutput(0), 256, DimsHW{2, 2}, deconwts, emptyWt);
    assert(up2);
    up2->setStrideNd(DimsHW{2, 2});
    up2->setNbGroups(256);

    output1 = network->addElementWise(*output1->getOutput(0), *up2->getOutput(0), ElementWiseOperation::kSUM);
    assert(output1);
    // merge 1
    output1 = conv_bn_relu(network,weightMap, *output1->getOutput(0), 256, 3, 1, 1, true, "fpn.merge1.0");

    IActivationLayer *ssh1 = ssh(network, weightMap, *output1->getOutput(0), 256);
    IActivationLayer *ssh2 = ssh(network, weightMap, *output2->getOutput(0), 256);
    IActivationLayer *ssh3 = ssh(network, weightMap, *output3->getOutput(0), 256);

    auto bboxHead1 = bboxHead(network, weightMap, *ssh1->getOutput(0), 2, "BboxHead.0.conv1x1");
    auto bboxHead2 = bboxHead(network, weightMap, *ssh2->getOutput(0), 2, "BboxHead.1.conv1x1");
    auto bboxHead3 = bboxHead(network, weightMap, *ssh3->getOutput(0), 2, "BboxHead.2.conv1x1");

    auto clsHead1 = clsHead(network, weightMap, *ssh1->getOutput(0), 2, "ClassHead.0.conv1x1");
    auto clsHead2 = clsHead(network, weightMap, *ssh2->getOutput(0), 2, "ClassHead.1.conv1x1");
    auto clsHead3 = clsHead(network, weightMap, *ssh3->getOutput(0), 2, "ClassHead.2.conv1x1");

    auto lmHead1 = lmHead(network, weightMap, *ssh1->getOutput(0), 2, "LandmarkHead.0.conv1x1");
    auto lmHead2 = lmHead(network, weightMap, *ssh2->getOutput(0), 2, "LandmarkHead.1.conv1x1");
    auto lmHead3 = lmHead(network, weightMap, *ssh3->getOutput(0), 2, "LandmarkHead.2.conv1x1");

    // decode bbox conf landmark
    ITensor *inputTensors1[] = {bboxHead1->getOutput(0), clsHead1->getOutput(0), lmHead1->getOutput(0)};
    auto cat1 = network->addConcatenation(inputTensors1, 3);
    assert(cat1);

    ITensor *inputTensors2[] = {bboxHead2->getOutput(0), clsHead2->getOutput(0), lmHead2->getOutput(0)};
    auto cat2 = network->addConcatenation(inputTensors2, 3);
    assert(cat2);

    ITensor *inputTensors3[] = {bboxHead3->getOutput(0), clsHead3->getOutput(0), lmHead3->getOutput(0)};
    auto cat3 = network->addConcatenation(inputTensors3, 3);
    assert(cat3);

    auto creator = getPluginRegistry()->getPluginCreator("Decode_TRT", "1");
    PluginFieldCollection pfc;
    IPluginV2 *pluginObj = creator->createPlugin("decode", &pfc);
    ITensor *inputTensors[] = {cat1->getOutput(0), cat2->getOutput(0), cat3->getOutput(0)};
    auto decodeLayer = network->addPluginV2(inputTensors, 3, *pluginObj);
    assert(decodeLayer);

    decodeLayer->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*decodeLayer->getOutput(0));

    // build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1<<30);

#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#endif

    cout << "Building engine, wait for a while" << endl;
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine && "bulid engine fail");
    cout << "Build engien successfully" << endl;

    network->destroy();
    // release memory
    for(auto& m : weightMap) {
        free((void *)(m.second.values));
        m.second.values = NULL;
    }

    return engine;
}


void APIToModel(int maxBatchSize, IHostMemory **modelStream) {
    // create builder
    IBuilder *builder = createInferBuilder(gLogger);
    // config
    IBuilderConfig *config = builder->createBuilderConfig();

    // create engine
    ICudaEngine *engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // serialize the engine
    (*modelStream) = engine->serialize();

    // free
    engine->destroy();
    builder->destroy();
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
        assert(modelStream != nullptr);
        ofstream p("retianface.engine", ios::binary);
        if (!p) {
            cerr << "engine file build fail" << endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else {
        cout << "Not finish coding" << endl;
        return 0;
    }
}
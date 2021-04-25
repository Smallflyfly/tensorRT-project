//
// Created by smallflyfly on 2021/4/22.
//

#include "argsParser.h"
#include "buffers.h"
#include "common.hpp"
#include "common/logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "yololayer.h"

#define CUDA_DEVICE 0
#define NET s
#define STR1(x) #x
#define STR2(x) STR1(x)

#define INPUT_BLOB_NAME "data"
#define BATCH_SIZE 1

#define USE_FP16

static sample::Logger gLogger;

static const int INPUT_W = Yolo::INPUT_W;
static const int INPUT_H = Yolo::INPUT_H;
const char* OUTPUT_BLOB_NAME = "prob";

using namespace std;



void ApiToModel(unsigned int batchSize, IHostMemory** modelStream);


ICudaEngine *createEngine_s(unsigned int batchSize, IBuilder *builder, IBuilderConfig *config, DataType type);

std::map<std::string, Weights> loadTrainedWeights(const string& file);

int main(int argc, char** argv) {
    cudaSetDevice(CUDA_DEVICE);
    char* trtModelStream{nullptr};
    size_t size{0};
    std::string engineName = STR2(NET);
    engineName = "yolov5" + engineName + ".engine";
    if (argc == 2 &&  strcmp(argv[1], "-s") == 0 ) {
        IHostMemory* modelStream{nullptr};
        ApiToModel(BATCH_SIZE, &modelStream);
    }

    return 0;

}

void ApiToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* builderConfig = builder->createBuilderConfig();
    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine_s(maxBatchSize, builder, builderConfig, DataType::kFLOAT);
}



ICudaEngine *createEngine_s(unsigned int batchSize, IBuilder *builder, IBuilderConfig *config, DataType type) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, type, Dims3(3,  INPUT_H, INPUT_W));
    assert(data);
    std::map<std::string, Weights> weightMap = loadTrainedWeights("weihts/yolov5.wts");
    Weights emptyWts{DataType::kFLOAT, nullptr, 0};

    //yolov5 backbone
    auto focus0 = focus(network, weightMap, *data, 3, 32, 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), 64, 3, 2, 1, "model.1");
    auto bottleneck_csp2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 64, 64, 1, true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_csp2->getOutput(0), 128, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 128, 128, 3, true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), 256, 3, 2, 1, "model.5");
    auto bottleneck_scp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 256, 256, 3, true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_scp6->getOutput(0), 512, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 512, 512, 5, 9, 13, "model.8");

    // yolo5 head
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), 256, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; ++i) {
        deval[i] = 1.0;
    }
    Weights deconvwt11{DataType::kFLOAT, deval, 256 * 2 * 2};
    IDeconvolutionLayer *deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 256, DimsHW{2, 2}, deconvwt11, emptyWts);
    deconv11->setStrideNd(DimsHW{2, 2});
    deconv11->setNbGroups(256);

    ITensor* inputTensors12[] = {deconv11->getOutput(0), bottleneck_scp6->getOutput(0)};
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 512, 256, 1, false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), 256, 1, 1, 1, "model.14");

    Weights deconv15wt{DataType::kFLOAT, deval, 128 * 2 * 2};
    IDeconvolutionLayer *deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 128, DimsHW{2, 2}, deconv15wt, emptyWts);
    deconv15->setStrideNd(DimsHW{2, 2});
    deconv15->setNbGroups(128);

    ITensor* inputTensors16[] = {deconv15->getOutput(0), bottleneck_csp4->getOutput(0)};
    auto cat16 = network->addConcatenation(inputTensors16, 2);
    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 256, 128, 1, false, 1, 0.5, "model.17");
    IConvolutionLayer *det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1},
                                                        weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), 128, 3, 2, 1, "model.18");
    ITensor *inputTensors19[] ={conv18->getOutput(0), conv14->getOutput(0)};
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = bottleneckCSP(network, weightMap, *cat19->getOutput(0), 256, 256, 3, false, 1, 0.5, "model.20");
    IConvolutionLayer *det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1},
                                                        weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);

    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), 256, 3, 2, 1, "model.21");
    ITensor *inputTensors22[] = {conv21->getOutput(0), conv10->getOutput(0)};
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = bottleneckCSP(network, weightMap, *cat22->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.23");
    IConvolutionLayer *det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1},
                                                        weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    const PluginFieldCollection *pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", pluginData);
    ITensor* inputTensor_yolo[] = {det2->getOutput(0), det1->getOutput(0), det0->getOutput(0)};
    auto yolo = network->addPluginV2(inputTensor_yolo, 3, *pluginObj);
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // build engine
    builder->setMaxBatchSize(batchSize);
    config->setMaxWorkspaceSize(32 * (1 << 20));
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    cout << "Building engine, wait for a while " << endl;
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    cout << "Build engine successfully!" << endl;

    // destroy network dont need network anymore
    network->destroy();

    // Release hots memory
    for (auto &mem : weightMap) {
        free((void*) (mem.second.values));
    }

    return engine;
}

std::map<std::string, Weights> loadTrainedWeights(const string& weightFile) {
    std::cout << "Loading weights: " << weightFile << endl;
    map<string, Weights> weightMap;
    ifstream input(weightFile);
    assert(input.is_open() && "Load weight file error!");

    // read number of weight blobs
    int32_t count;
    input >> count;
    assert(count>0 && "Invalid weight!");

    while (count --) {
        Weights wt{DataType::kFLOAT, nullptr, 0};

        u_int32_t size;
        string name;
        input >> name >> dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t *val = static_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t  x = 0, y = size;  x < y; ++ x) {
            input >> hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

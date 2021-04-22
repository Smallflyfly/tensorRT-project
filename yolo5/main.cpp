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

static sample::Logger gLogger;

static const int INPUT_W = Yolo::INPUT_W;
static const int INPUT_H = Yolo::INPUT_H;

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
    auto focus0 = focus(network, )

    return nullptr;
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

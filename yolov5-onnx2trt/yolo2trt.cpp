//
// Created by smallflyfly on 2021/6/28.
//

#include "NvInfer.h"
#include <iostream>
#include <fstream>
#include "logging.h"
#include <string>
#include "NvOnnxParser.h"


using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;

static const char *onnxName = "yolov5s.onnx";
static const char *trtName = "yolov5s.trt";

static Logger gLogger;

static const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

void yolo2trt(const string &onnx) {
    IBuilder *builder = createInferBuilder(gLogger);
    INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
    IParser *parser = createParser(*network, gLogger);
    parser->parseFromFile(onnx.c_str(), static_cast<int>(Logger::Severity::kWARNING));
    for (int i = 0; i < parser->getNbErrors(); ++i) {
        cerr <<"parser error: " << parser->getError(i)->desc() << endl;
    }
    cout << "parser onnx file successfully!" << endl;

    int maxBatchSize = 1;
    builder->setMaxBatchSize(maxBatchSize);
    IBuilderConfig *config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 30);
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);

    IHostMemory *trtStream = engine->serialize();
    ofstream p(trtName, ios::binary);
    if (!p) {
        cerr << "generate trt file error!" << endl;
        return;
    }
    p.write(reinterpret_cast<const char*>(trtStream->data()), trtStream->size());

    trtStream->destroy();
}

int main() {
    yolo2trt(onnxName);
    return 0;
}
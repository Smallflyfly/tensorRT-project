//
// Created by smallflyfly on 2021/7/15.
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

static const char *onnx = "extractor.onnx";
static const char *trt = "extractor.trt";

static Logger gLogger;

static const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

bool onnx2trt() {
    IBuilder *builder = createInferBuilder(gLogger);
    INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
    IParser *parser = createParser(*network, gLogger);
    parser->parseFromFile(onnx, static_cast<int>(Logger::Severity::kWARNING));
    for (int i = 0; i < parser->getNbErrors(); ++i) {
        cerr << "parser error : " << parser->getError(i) << endl;
    }
    cout << "parser onnx file successfully" << endl;

    int maxBatchSize = 1;
    builder->setMaxBatchSize(maxBatchSize);
    IBuilderConfig *config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1<<30);
    config->setFlag(BuilderFlag::kFP16);

    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);

    IHostMemory *trtStream = engine->serialize();
    assert(trtStream != nullptr);

    ofstream f(trt, ios::binary);
    if (!f) {
        cerr << "engine serialize fail" << endl;
        return false;
    }
    f.write(reinterpret_cast<const char*>(trtStream->data()), trtStream->size());

    trtStream->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();

    return true;
}

int main() {
    bool flag = onnx2trt();
    if (flag) {
        cout << "onnx convert to trt successfully!" << endl;
    } else {
        cout << "onnx convert to trt fail" << endl;
    }
    return 0;
}
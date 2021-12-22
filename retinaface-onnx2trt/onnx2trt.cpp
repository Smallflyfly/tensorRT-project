//
// Created by unicom on 2021/12/21.
//

//
// Created by smallflyfly on 2021/11/16.
//

#include "NvInfer.h"
#include <iostream>
#include <fstream>
#include "logging.h"
#include <string>
#include "NvOnnxParser.h"

using namespace std;
using namespace nvonnxparser;
using namespace nvinfer1;

static const char *onnxName = "retinaFace_1output.onnx";
static const char *trtName = "retinaFace_1output.trt";

static sample::Logger gLogger;

static const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

void onnx2trt(const char *onnx) {
    IBuilder *builder = createInferBuilder(gLogger);
    INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
    IParser *parser = createParser(*network, gLogger);
    parser->parseFromFile(onnx, static_cast<int>(sample::Logger::Severity::kWARNING));
    for (int i = 0; i < parser->getNbErrors(); ++i) {
        cerr << "parser error: " << parser->getError(i)->desc() << endl;
    }
    cout  << "parser onnx file successfully!" << endl;

    int maxBatchSize = 1;
    builder->setMaxBatchSize(maxBatchSize);
    IBuilderConfig *config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1<<30);
    config->setFlag(BuilderFlag::kFP16);

    IHostMemory *trtStream = builder->buildSerializedNetwork(*network, *config);

    ofstream p(trtName, ios::binary);
    if (!p) {
        cerr << "generate trt file error !" << endl;
        return;
    }
    p.write(reinterpret_cast<const char*>(trtStream->data()), trtStream->size());

    trtStream->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();

}

int main() {
    onnx2trt(onnxName);
    return 1;
}
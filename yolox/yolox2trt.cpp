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

static const char *onnxName = "yolox-helmet.onnx";
static const char *trtName = "yolox-helemt.trt";

static Logger gLogger;

static const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

void yolox2trt(const char *onnx) {
    IBuilder *builder = createInferBuilder(gLogger);
    INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
    IParser *parser = createParser(*network, gLogger);
    parser->parseFromFile(onnx, static_cast<int>(Logger::Severity::kWARNING));
    for (int i = 0; i < parser->getNbErrors(); ++i) {
        cerr << "parser error: " << parser->getError(i)->desc() << endl;
    }
    gLogger.log(Severity::kINFO, "parser onnx file successfully!");
}

int main() {
    yolox2trt(onnxName);

    return 1;
}
//
// Created by smallflyfly on 2021/6/28.
//

#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include "NvInfer.h"
#include "logging.h"
#include "cuda_runtime.h"
#include "yolov5.h"
#include <numeric>


using namespace std;
using namespace nvinfer1;

static Logger gLogger;

static const char *engineFile = "yolov5s.trt";
static const int BATCH_SIZE = 1;
static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const int OUTPUT_SIZE = 25200 * 85;
static const char *INPUT_BLOB_NAME = "images";
static const char *OUTPUT_BLOB_NAME = "output";

inline int64_t volume(const nvinfer1::Dims& d)
{
    return accumulate(d.d, d.d + d.nbDims, 1, multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

void doInference(IExecutionContext &context, float *input, float *output) {
    const ICudaEngine &engine = context.getEngine();
    assert(engine.getNbBindings() == 2);
    void *buffers[2];
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

//    for (int i = 0; i < engine.getNbBindings(); ++i) {
//        Dims dims = engine.getBindingDimensions(i);
//        DataType dataType = engine.getBindingDataType(i);
//        int64_t totalSize = volume(dims) * 1 * getElementSize(dataType);
//        buffers[i] = totalSize;
//
//    }

    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_W * INPUT_H * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float )));

    //cuda stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, BATCH_SIZE * 3 * INPUT_W * INPUT_H * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(BATCH_SIZE, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main() {
    ifstream file(engineFile, ios::binary);
    size_t size{0};
    char *trtStream{nullptr};
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtStream = new char[size];
        assert(trtStream != nullptr);
        file.read(trtStream, size);
        file.close();
    }
    // image file read
    float data[BATCH_SIZE * INPUT_W * INPUT_H];
    for (int i = 0; i < BATCH_SIZE * INPUT_W * INPUT_H; ++i) {
        data[i] = 1.0;
    }
    float output[BATCH_SIZE * OUTPUT_SIZE];

    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    ICudaEngine *engine = runtime->deserializeCudaEngine(trtStream, size);
    assert(engine != nullptr);

    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    doInference(*context, data, output);
}
//
// Created by smallflyfly on 2021/4/19.
//


#include "argsParser.h"
#include "buffers.h"
#include "common.h"
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

using namespace std;
using namespace sample;
using namespace cv;

class RetinaFace
{
    template<typename T>
    using SampleUniquePtr = unique_ptr<T, samplesCommon::InferDeleter>;

public:
    RetinaFace(samplesCommon::OnnxSampleParams  params) : mParams(std::move(params)), mEngine(nullptr) {
    }

    void build();

    void infer();

private:
    samplesCommon::OnnxSampleParams mParams;
    shared_ptr<nvinfer1::ICudaEngine> mEngine;

    nvinfer1::Dims mInputDims;
    nvinfer1::Dims mOutput1Dims;
    nvinfer1::Dims mOutput2Dims;
    nvinfer1::Dims mOutput3Dims;

    void constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
                          SampleUniquePtr<nvinfer1::IBuilderConfig>& config, SampleUniquePtr<nvonnxparser::IParser>& parser);

    void processInput(const samplesCommon::BufferManager& buffers);

    void verifyOutput(const samplesCommon::BufferManager& buffer);
};

void RetinaFace::constructNetwork(RetinaFace::SampleUniquePtr<IBuilder> &builder,
                                  RetinaFace::SampleUniquePtr<INetworkDefinition> &network,
                                  RetinaFace::SampleUniquePtr<IBuilderConfig> &config,
                                  RetinaFace::SampleUniquePtr<nvonnxparser::IParser> &parser) {
    bool parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed) {
        gLogError << "解析错误"<<endl;
    }
//    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(500_MiB);
    if (mParams.fp16) {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8) {
        config->setFlag(BuilderFlag::kINT8);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

}

void readImageFile(const string& imageFileName, const string& imageFilePath, u_int8_t* buffer,
                               int w, int h, int c) {
    const int imageMean[] = {104, 117, 123};
    Mat im = imread(imageFileName);
    Mat image;
    resize(im, image, Size(640, 640), INTER_NEAREST);
    imshow("image1", im);
    cvtColor(image, image, COLOR_BGR2RGB);
    for (int i=0; i<h; i++) {
        for (int j=0; j<w; j++) {
            image.at<Vec3d>(i, j)[0] -= imageMean[0];
            image.at<Vec3d>(i, j)[1] -= imageMean[1];
            image.at<Vec3d>(i, j)[2] -= imageMean[2];
        }
    }
    imshow("image2", image);
    waitKey();
    destroyAllWindows();

}

void RetinaFace::processInput(const samplesCommon::BufferManager &buffers) {
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];
    const int inputC = mInputDims.d[1];
    string imageFileName = "test.jpg";
    string imageFilePath = "./";
    vector<u_int8_t> fileData(inputH*inputW*inputC);
    readImageFile(imageFileName, imageFilePath, fileData.data(), inputW, inputH, inputC);

}

void RetinaFace::verifyOutput(const samplesCommon::BufferManager &buffer) {

}

void RetinaFace::build() {
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder) {
        gLogError << "builder 出错"<< endl;
    }

    const auto explicitBath = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBath));
    if (!network) {
        gLogError << "network出错" << endl;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        gLogError << "config出错" << endl;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser) {
        gLogError << "parser出错" << endl;
    }

    constructNetwork(builder, network, config, parser);

    mEngine = shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 4);

    assert(network->getNbOutputs() == 3);
    cout<< network->getNbOutputs() <<endl;
    mOutput1Dims = network->getOutput(0)->getDimensions();
    mOutput2Dims = network->getOutput(1)->getDimensions();
    mOutput3Dims = network->getOutput(2)->getDimensions();

}

void RetinaFace::infer() {
    samplesCommon::BufferManager buffers(mEngine);
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    assert(mParams.inputTensorNames.size() == 1);
    processInput(buffers);
}



samplesCommon::OnnxSampleParams initParams(samplesCommon::Args& args) {
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) {
        params.dataDirs.emplace_back("data/samples/");
    } else {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "FaceDetector.onnx";
    params.inputTensorNames.emplace_back("input");
    params.batchSize = 1;
    params.outputTensorNames.emplace_back("output1");
    params.outputTensorNames.emplace_back("output2");
    params.outputTensorNames.emplace_back("output3");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}


int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOk = parseArgs(args, argc, argv);
    if (!argsOk) {
        sample::gLogError << "Invalid arguments" << endl;
    }
    RetinaFace retinaFace(initParams(args));

    sample::gLogInfo << "Building and running a GPU inference egine for Onnx retinaface" <<endl;

    retinaFace.build();

    retinaFace.infer();

}

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

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"

#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) \
        { \
            cerr << "Cuda failure:" << ret << endl; \
            abort(); \
        } \
    } while(0)

using namespace std;
using namespace nvinfer1;
//using namespace sample;

//static sample::Logger gLogger;
static Logger gLogger;
static const char* INPUT_BLOB_NAME = "input";
static const char* OUTPUT_BLOB_NAME = "output";
static const int INPUT_H = 256;
static const int INPUT_W = 256;
static const int OUTPUT_SIZE = 2;


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
        int size;
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
//    if (layerName == "layer1.1.bn1") {
//        cout << layerName << endl;
//        cout << (float *)weightMap[layerName + ".weight"].values << endl;
//    }

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

    IConvolutionLayer *conv2 = network->addConvolutionNd(*relu1->getOutput(0), outCh, DimsHW{3, 3}, weightMap[layerName + "conv2.weight"], wtempty);
    assert(conv2);
    conv2->setStrideNd(DimsHW{stride, stride});
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer *bn2 = addBN2d(network, weightMap, *conv2->getOutput(0), layerName + "bn2", 1e-5);
    assert(bn2);

    IActivationLayer *relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer *conv3 = network->addConvolutionNd(*relu2->getOutput(0), outCh * 4, DimsHW{1, 1}, weightMap[layerName + "conv3.weight"], wtempty);
    assert(conv3);
    conv3->setStrideNd(DimsHW{1, 1});
    conv3->setPaddingNd(DimsHW{0, 0});

    IScaleLayer *bn3 = addBN2d(network, weightMap, *conv3->getOutput(0), layerName + "bn3", 1e-5);
    assert(bn3);

    IElementWiseLayer *ew1;

    if (stride != 1 || inCh != outCh * 4) {
        IConvolutionLayer *conv4 = network->addConvolutionNd(input, outCh * 4, DimsHW{1, 1}, weightMap[layerName + "downsample.0.weight"], wtempty);
        assert(conv4);
        conv4->setStrideNd(DimsHW{stride, stride});

        IScaleLayer *bn4 = addBN2d(network, weightMap, *conv4->getOutput(0), layerName + "downsample.1", 1e-5);
        assert(bn4);

        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
        assert(ew1);
    } else {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
        assert(ew1);
    }

    IActivationLayer *relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);

    return relu3;
}

// softmax layer
ILayer* reshapeSoftmax(INetworkDefinition *network, ITensor &input, int c) {
    IShuffleLayer *shuffleLayer = network->addShuffle(input);
    assert(shuffleLayer);
    shuffleLayer->setReshapeDimensions(Dims3(1, -1, c));

    Dims dim0 = shuffleLayer->getOutput(0)->getDimensions();

    cout << "softmax output dims " << dim0.d[0] << " " << dim0.d[1] << " " << dim0.d[2] << " " << dim0.d[3] << endl;
}

// create engine
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig* config, DataType dtype) {
//    const auto explictBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(0U);
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dtype, Dims3{3, INPUT_H, INPUT_W});
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

    // layers [3, 4, 6, 3]
    IActivationLayer *x;
    // layer1 output channel size
    x = bottleneck(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "layer1.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "layer1.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "layer1.2.");

    // layer2 output channel size
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 128, 2, "layer2.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.3.");

    // layer3
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 256, 2, "layer3.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.3.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.4.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.5.");

    // layer4
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 512, 2, "layer4.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "layer4.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "layer4.2.");

    // pool
    IPoolingLayer *pool4 = network->addPoolingNd(*x->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(pool4);
    pool4->setStrideNd(DimsHW{3, 3});

    // fc
    IFullyConnectedLayer *fc = network->addFullyConnected(*pool4->getOutput(0), 2, weightMap["fc.weight"], weightMap["fc.bias"]);
    assert(fc);

//    ISoftMaxLayer *prob = network->addSoftMax(*fc->getOutput(0));
//    assert(prob);

//    fc->getOutput(0)->setName(OUTPUT_BLOB_NAME);
//    cout << "set name out " << OUTPUT_BLOB_NAME << endl;
//    network->markOutput(*fc->getOutput(0));

//    prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
//    cout << "set name out " << OUTPUT_BLOB_NAME << endl;
//    network->markOutput(*prob->getOutput(0));
    ILayer *softMaxLay = reshapeSoftmax(network, *fc->getOutput(0), 2);

    // build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1<<20);
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine);
    cout << "engine build" << endl;

    // destroy network
    network->destroy();
    // Release host memeory
    for (auto& mem : weightMap) {
        free((void *) mem.second.values);
    }

    return engine;

}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);

    IBuilderConfig* config = builder->createBuilderConfig();

    //create engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // destroy everything
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext &context, float *input, float *output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // pointers to input and output device buffers to pass to engine
    // engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // according to input name and output name to bind buffers
    // indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // create steam
    cudaStream_t  stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device infer on the batch asynchronously and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // realease stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "arguments not right" << endl;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);
        ofstream p("resnet50.engine", ios::binary);
        if (!p) {
            cerr << "engine file error" << endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());

        modelStream->destroy();
        return 0;
    } else if (string(argv[1]) == "-d") {
        ifstream file("resnet50.engine", ios::binary);
        if (file.good()) {
            // 基地址为文件结束处，偏移地址为0，于是指针定位在文件结束处
            file.seekg(0, file.end);
            // tellg()函数不需要带参数，它返回当前定位指针的位置，也代表着输入流的大小。
            size = file.tellg();
            // //基地址为文件头，偏移量为0，于是定位在文件头
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        } else {
            cerr << "engine file error" << endl;
            return -1;
        }
    } else {
        return -1;
    }
    static float data[3 * INPUT_H * INPUT_W];
    for (float & i : data) {
        i = 1.0;
    }

    cv::Mat im = cv::imread("./test1.jpg");
    if (im.empty()) {
        cout << "image file is none!" << endl;
    }
    cv::resize(im, im, cv::Size(256, 256), cv::INTER_LINEAR);

//    cv::imshow("image1", im);

//    cvtColor(im, im, cv::COLOR_BGR2RGB);

//    cv::imshow("image2", im);

//    cv::waitKey();
//    cv::destroyAllWindows();

    cv::normalize(im, im, 0.0, 255.0, cv::NORM_MINMAX);
    unsigned vol = INPUT_H * INPUT_W * 3;
    auto* fileDataChar = (uchar *) malloc(1 * 3 * INPUT_H * INPUT_W * sizeof(uchar));
    fileDataChar = im.data;
    for (int i = 0; i < vol; ++i) {
        data[i] = (float)fileDataChar[i] * 1.0;
    }


    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);

    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    delete[] trtModelStream;



    // run inference
    static float prob[OUTPUT_SIZE];
    for (int i = 0; i < 100; ++i) {
        auto start = chrono::system_clock::now();
        doInference(*context, data, prob, 1);
        auto end = chrono::system_clock::now();
        cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    }

    // free engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // print output
    cout << "Output :" << endl;
    for (float i : prob) {
        cout << i << endl;
    }
}
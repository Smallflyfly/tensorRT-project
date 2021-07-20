//
// Created by smallflyfly on 2021/7/15.
//

#include "extractor.h"

using namespace nvinfer1;

ICudaEngine* readEngine(const string &trtFile) {
    string engineCached;
    ifstream file(trtFile, ios::binary);
    while (file.peek() != EOF) {
        stringstream buffer;
        buffer << file.rdbuf();
        engineCached.append(buffer.str());
    }
    file.close();
    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    ICudaEngine *engine = runtime->deserializeCudaEngine(engineCached.data(), engineCached.size(), nullptr);
    assert(engine != nullptr);
    return engine;
}

float *readImageData(const cv::Mat &im, float &scale, int &pw, int &ph) {
    int w = im.cols, h = im.rows;
    scale = EXTRACTOR_INPUT_W * 1.0 / w < EXTRACTOR_INPUT_H * 1.0 / h ? EXTRACTOR_INPUT_W * 1.0 / w : EXTRACTOR_INPUT_H * 1.0 / h;
    cv::Mat resizeIm;
    int w1 = w * scale;
    int h1 = h * scale;
    cv::resize(im, resizeIm, cv::Size(w1, h1));
    pw = (int)((EXTRACTOR_INPUT_W - w1) / 2), ph = (int)((EXTRACTOR_INPUT_H - h1) / 2);
    cv::Mat newIm = cv::Mat::zeros(cv::Size(EXTRACTOR_INPUT_W, EXTRACTOR_INPUT_H), CV_8UC3);
    float *data = (float *) malloc(EXTRACTOR_BATCH_SIZE * 3 * EXTRACTOR_INPUT_W * EXTRACTOR_INPUT_H * sizeof(float));
    resizeIm.copyTo(newIm(cv::Rect(pw, ph, resizeIm.cols, resizeIm.rows)));
//    cv::imshow("im", newIm);
//    cv::waitKey(0);
//    cv::destroyAllWindows();
    // - (104, 117, 123) n
    newIm.convertTo(newIm, CV_32FC3, 1.0/255);
    vector<cv::Mat> channels(3);
    vector<float> mean{0.485, 0.456, 0.406};
    vector<float> std{0.229, 0.224, 0.225};
    cv::split(newIm, channels);
    for (int i = 0; i < channels.size(); ++i) {
        // y = x * alpha + beta
        // normalize = (x - mean) / std   x / std - mean /std
        mean[i] = -1 * mean[i] / std[i];
        channels[i].convertTo(channels[i], CV_32FC1, 1.0 / std[i], mean[i]);
    }
    cv::merge(channels, newIm);
//    cv::imshow("im", newIm);
//    cv::waitKey(0);
//    cv::destroyAllWindows();
    int channelLength = EXTRACTOR_INPUT_H * EXTRACTOR_INPUT_W;
    vector<cv::Mat> splitIm = {
            cv::Mat(EXTRACTOR_INPUT_H, EXTRACTOR_INPUT_W, CV_32FC1, data + channelLength * 2),
            cv::Mat(EXTRACTOR_INPUT_H, EXTRACTOR_INPUT_W, CV_32FC1, data + channelLength * 1),
            cv::Mat(EXTRACTOR_INPUT_H, EXTRACTOR_INPUT_W, CV_32FC1, data + channelLength * 0)
    };

    cv::split(newIm, splitIm);
    return data;
}

float* doInferenceExtractor(IExecutionContext &context, float *input) {
    const ICudaEngine &engine = context.getEngine();
    assert(engine.getNbBindings() == 2);

    void *buffers[2];
    const int inputIndex = engine.getBindingIndex(EXTRACTOR_INPUT_NAME);
    const int outputIndex = engine.getBindingIndex(EXTRACTOR_OUTPUT_NAME);

    CHECK(cudaMalloc(&buffers[inputIndex], EXTRACTOR_BATCH_SIZE * EXTRACTOR_INPUT_H * EXTRACTOR_INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], EXTRACTOR_BATCH_SIZE * EXTRACTOR_OUTPUT_SIZE * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, EXTRACTOR_BATCH_SIZE * 3 * EXTRACTOR_INPUT_H * EXTRACTOR_INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(EXTRACTOR_BATCH_SIZE, buffers, stream, nullptr);
    float *output = (float *) malloc(EXTRACTOR_BATCH_SIZE * EXTRACTOR_OUTPUT_SIZE * sizeof(float));
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], EXTRACTOR_BATCH_SIZE * EXTRACTOR_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

    return output;
}
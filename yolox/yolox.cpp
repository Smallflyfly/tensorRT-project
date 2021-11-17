#include "yolox.h"
using namespace std;
using namespace cv;

Mat resizeImage(Mat im, float &scale, float &pw, float &ph) {
    int w = im.cols, h = im.rows;
    if (w > h) {
        scale = INPUT_W * 1.0 / w;
    } else {
        scale = INPUT_H * 1.0 / h;
    }
    int dw = w * scale;
    int dh = h * scale;
    pw = abs(INPUT_W - dw) / 2;
    ph = abs(INPUT_H - dh) / 2;
    Mat newIm(INPUT_H, INPUT_W, CV_8UC3, Scalar(114, 114, 114));
    resize(im, im, Size(), scale, scale);
    im.copyTo(newIm(Rect(pw, ph, im.cols, im.rows)));
    return newIm;
}

float *prepareImage(Mat image, float &scale, float &pw, float &ph) {
    Mat im = resizeImage(image, scale, pw, ph);
    imshow("im", im);
    waitKey(0);
    destroyAllWindows();
    return nullptr;
}

vector<Yolox::Detection> postProcess(float *output) {
    cout << "post process" << endl;
    vector<Yolox::Detection> result;
}


vector<Yolox::Detection> doInference(IExecutionContext &context, float *input) {
    const ICudaEngine &engine = context.getEngine();
    assert(engine.getNbBindings() == 2);

    void *buffers[2];
    const int inputIndex = engine.getBindingIndex(INPUT_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_NAME);

    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // cuda stream
    cudaStream_t cudaStream;
    CHECK(cudaStreamCreate(&cudaStream));

    // copy mem from host to device
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, BATCH_SIZE * 3 * INPUT_H * INPUT_W, cudaMemcpyHostToDevice, cudaStream));
    context.enqueue(BATCH_SIZE, buffers, cudaStream, nullptr);
    float *output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, cudaStream));
    cudaStreamSynchronize(cudaStream);
    cudaStreamDestroy(cudaStream);

    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

    vector<Yolox::Detection> detections = postProcess(output);
    return detections;
}


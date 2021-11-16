#include "yolox.h"

cv::Mat resizeImage(cv::Mat im, float &scale, float &pw, float &ph) {
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
    cv::Mat newIm(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(im, im, cv::Size(), scale, scale);
    im.copyTo(newIm(cv::Rect(pw, ph, im.cols, im.rows)));
    return newIm;
}

float *prepareImage(cv::Mat image, float &scale, float &pw, float &ph) {
    cv::Mat im = resizeImage(image, scale, pw, ph);
    cv::imshow("im", im);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return nullptr;
}

std::vector<Yolox::Detection> doInference(IExecutionContext &context, float *input) {
    const ICudaEngine &engine = context.getEngine();
    assert(engine.getNbBindings() == 2);

    void *buffer[2];
    const int inputIndex = engine.getBindingIndex(INPUT_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_NAME);

    return std::vector<Yolox::Detection>();
}


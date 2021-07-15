//
// Created by smallflyfly on 2021/7/12.
//

#include "retinaFace.h"

int main() {
    ICudaEngine *engine = readEngine();
    cout << "Load engine done!" << endl;
    float scale = 1.0;
    int pw = 0, ph = 0;
    for (int i = 0; i < 10; ++i) {
        cv::Mat im = cv::imread("test.jpg");
        float *data = readImageData(im, scale, pw, ph);
        IExecutionContext *context = engine->createExecutionContext();
        assert(context != nullptr);
        vector<RetinaFace::Detection> detections;
        vector<vector<float>> anchors = getAnchors();
        auto start = chrono::high_resolution_clock::now();
        detections = doInference(*context, data, anchors);
        auto end = chrono::high_resolution_clock::now();
        float t = chrono::duration<float, milli>(end - start).count();
        cout << "total cost time: " << t << endl;
        showResult(im, detections, scale, pw, ph);
        cv::imshow("im", im);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    return 0;

}
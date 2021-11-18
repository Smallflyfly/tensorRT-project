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
    vector<int> strides = {8, 16, 32};
    vector<vector<int>> grids;
    vector<Yolox::Detection> result;
    // generate grids
    for(auto &stride : strides) {
        int dx = INPUT_W / stride;
        int dy = INPUT_H / stride;
        for (int i = 0; i < dx; ++i) {
            for (int j = 0; j < dy; ++j) {
                grids.push_back({dx, dy, stride});
            }
        }
    }
    // generate detection > conf_threshold without nms
    for (int anchorIdx = 0; anchorIdx < grids.size(); anchorIdx++) {
        int dx = grids[anchorIdx][0];
        int dy = grids[anchorIdx][1];
        int stride = grids[anchorIdx][2];
        // (cx, cy, dw, dh) + conf + every class conf
        int basePos = anchorIdx * (5 + NUM_CLASSES);
        float pxCenter = (output[basePos + 0] + dx * 1.0) * stride;
        float pyCenter = (output[basePos + 1] + dy * 1.0) * stride;
        float pw = exp(output[basePos + 2]) * stride;
        float ph = exp(output[basePos + 3]) * stride;
        float xmin = pxCenter - 0.5 * pw;
        float ymin = pyCenter - 0.5 * ph;
        float objConf = output[basePos + 4];
        // cls conf
        for (int clsIdx = 0; clsIdx < NUM_CLASSES; clsIdx++) {
            float clsConf = output[basePos + 5 + clsIdx];
            float conf = objConf * clsConf;
            if (conf > CONF_THRESHOLD) {
                Yolox::Detection det{};
                det.x = xmin;
                det.y = ymin;
                det.w = pw;
                det.h = ph;
                det.classes = clsIdx;
                det.prob = conf;
                result.push_back(det);
            }
        }
        cout << "before nms detection size: " << result.size() << endl;
        nms(result);
        cout << "after nms detection size: " << result.size() << endl;
        return result;
    }
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

void nms(vector<Yolox::Detection> &dets) {
    sort(dets.begin(), dets.end(), [=](const Yolox::Detection &d1, const Yolox::Detection &d2) {
        return d1.prob > d2.prob;
    });
    for (int i = 0; i < dets.size(); ++i) {
        for (int j = i+1; j < dets.size(); ++j) {
            if (dets[i].classes == dets[j].classes) {
                float iou = iouCalculate(dets[i], dets[j]);
                if (iou > NMS_THRESHOLD) {
                    dets[j].prob = 0.0;
                }
            }
        }
    }
    dets.erase(remove_if(dets.begin(), dets.end(), [](const Yolox::Detection &detection){return detection.prob == 0.0;}), dets.end());
}

float iouCalculate(const Yolox::Detection &det1, const Yolox::Detection &det2) {
    float x11 = det1.x, y11 = det1.y, x12 = det1.x + det1.w, y12 = det1.y + det1.h;
    float x21 = det2.x, y21 = det2.y, x22 = det2.x + det2.w, y22 = det2.y + det2.h;
    float area1 = (x12 - x11) * (y12 - y11);
    float area2 = (x22 - x21) * (y22 - y21);

    float x1 = max(x11, x21);
    float x2 = min(x12, x22);
    float y1 = max(y11, y21);
    float y2 = min(y12, y22);

    if (x1 >= x2 || y1 >= y2) {
        return 0.0;
    } else {
        float overlap = (x2 - x1) * (y2 - y1);
        return (float) (overlap / (area1 + area2 - overlap + 1e-5));
    }
}

void fixDetection(vector<Yolox::Detection> &detections, float scale, float pw, float ph) {
    int n = detections.size();
    for (int i = 0; i < n; ++i) {
        float xmin = detections[i].x;
        float ymin = detections[i].y;
        float w = detections[i].w;
        float h = detections[i].h;
        float xmax = detections[i].x + w;
        float ymax = detections[i].y + h;
        xmin -= pw;
        ymin -= ph;
        xmax -= pw;
        ymax -= ph;
        xmin /= scale;
        ymin /= scale;
        xmax /= scale;
        ymax /= scale;
        w /= scale;
        h /= scale;
        detections[i].x = xmin;
        detections[i].y = ymin;
        detections[i].w = w;
        detections[i].h = h;
        detections[i].xmax = xmax;
        detections[i].ymax = ymax;
    }
}

void showResult(const vector<Yolox::Detection> &detections, Mat &image) {
    for (auto &detection : detections) {
        float xmin = detection.x;
        float ymin = detection.y;
        float w = detection.w;
        float h = detection.h;
        float prob = detection.prob;
        int cls = detection.classes;
        Scalar scalar(SHOW_COLOR[cls][0], SHOW_COLOR[cls][1], SHOW_COLOR[cls][2]);
        Rect rect(xmin, ymin, w, h);
        rectangle(image, rect, scalar * 255, 1);
    }
    imshow("helmet detection", image);
}


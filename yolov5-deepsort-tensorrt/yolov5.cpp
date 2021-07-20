//
// Created by smallflyfly on 2021/7/16.
//

#include "yolov5.h"


Yolo::Yolo(const string &configFile) {
    YAML::Node root = YAML::LoadFile(configFile);
    YAML::Node config = root["yolo"];
    strides = config["strides"].as<vector<int>>();
    numAnchors = config["num_anchors"].as<vector<int>>();
    anchors = config["anchors"].as<vector<vector<int>>>();
    for (int i = 0; i < strides.size(); ++i) {
        grids.push_back({numAnchors[i], int(inputH / strides[i]), int(inputW / strides[i])});
    }
    nameFile = config["labels_file"].as<string>();
    readNames();
    category = names.size();
    objThreshold = config["obj_threshold"].as<float>();
    nmsThreshold = config["nms_threshold"].as<float>();
}

Yolo::~Yolo() = default;

const vector<int> &Yolo::getStrides() const {
    return strides;
}

const vector<int> &Yolo::getNumAnchors() const {
    return numAnchors;
}

const vector<vector<int>> &Yolo::getAnchors() const {
    return anchors;
}

const vector<vector<int>> &Yolo::getGrids() const {
    return grids;
}

int Yolo::getInputW() const {
    return inputW;
}

int Yolo::getInputH() const {
    return inputH;
}

void Yolo::readNames() {
    ifstream file(nameFile);
    if (!file.is_open()) {
        cerr << "read names file error" << endl;
    }
    string strLine;
    int i = 0;
    while (getline(file, strLine)) {
        names.insert({i++, strLine});
    }
    file.close();
}

int Yolo::getCategory() const {
    return category;
}

float Yolo::getObjThreshold() const {
    return objThreshold;
}

float Yolo::getNmsThreshold() const {
    return nmsThreshold;
}

void Yolo::nms(vector<Detection> &dets) {
    sort(dets.begin(), dets.end(), [=](const Detection &d1, const Detection &d2){
        return d1.getProb() > d2.getProb();
    });

    for (int i = 0; i < dets.size(); ++i) {
        for (int j = i+1; j < dets.size(); ++j) {
            if (dets[i].getClasses() == dets[j].getClasses()) {
                float iou = iouCalculate(dets[i], dets[j]);
                if (iou > nmsThreshold) {
                    dets[j].setProb(0.0);
                }
            }
        }
    }
    dets.erase(remove_if(dets.begin(), dets.end(), [](const Detection &det){return det.getProb() == 0.0;}), dets.end());
}

float Yolo::iouCalculate(const Yolo::Detection &det1, const Yolo::Detection &det2) {
    float x11 = det1.getX(), y11 = det1.getY(), x12 = det1.getXmax(), y12 = det1.getYmax();
    float x21 = det2.getX(), y21 = det2.getY(), x22 = det2.getXmax(), y22 = det2.getYmax();
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

Mat resizeImage(Mat& im, float &scale, float &pw, float &ph) {
    int w = im.cols, h = im.rows;
    if (w > h) {
        scale = YOLO_INPUT_W * 1.0 / w;
    } else {
        scale = YOLO_INPUT_H * 1.0 / h;
    }
    int wNew = w * scale;
    int hNew = h * scale;
    pw = abs(YOLO_INPUT_W - wNew) / 2;
    ph = abs(YOLO_INPUT_H - hNew) / 2;
    Mat imNew = Mat::zeros(YOLO_INPUT_H, YOLO_INPUT_W, CV_8UC3);
    resize(im, im, Size(), scale, scale);
    im.copyTo(imNew(Rect(pw, ph, im.cols, im.rows)));
    return imNew;
}

float *prepareImage(Mat &frame, float &scale, float &pw, float &ph) {
    Mat im = resizeImage(frame, scale, pw, ph);
    im.convertTo(im, CV_32FC3, 1.0 / 255);
    float *data = (float *) malloc(YOLO_INPUT_W * YOLO_INPUT_H * 3 * sizeof(float));
    int index = 0;
    int channelLength = YOLO_INPUT_W * YOLO_INPUT_H;
    // BGR TO RGB
    vector<cv::Mat> splitImg = {
            cv::Mat(YOLO_INPUT_H, YOLO_INPUT_W, CV_32FC1, data + channelLength * (index + 2)),
            cv::Mat(YOLO_INPUT_H, YOLO_INPUT_W, CV_32FC1, data + channelLength * (index + 1)),
            cv::Mat(YOLO_INPUT_H, YOLO_INPUT_W, CV_32FC1, data + channelLength * (index + 0))
    };
    cv::split(im, splitImg);
    assert(data != nullptr);

    return data;
}

vector<Yolo::Detection> doInferenceYolo(IExecutionContext &context, float *input) {
    const ICudaEngine &engine = context.getEngine();
    cout << engine.getNbBindings() << endl;
    assert(engine.getNbBindings() == 2);
    void *buffers[2];
    const int inputIndex = engine.getBindingIndex(YOLO_INPUT_NAME);
    const int outputIndex = engine.getBindingIndex(YOLO_OUTPUT_NAME);

    CHECK(cudaMalloc(&buffers[inputIndex], YOLO_BATCH_SIZE * 3 * YOLO_INPUT_W * YOLO_INPUT_H * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], YOLO_BATCH_SIZE * YOLO_OUTPUT_SIZE * sizeof(float)));

    //cuda stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, YOLO_BATCH_SIZE * 3 * YOLO_INPUT_W * YOLO_INPUT_H * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(YOLO_BATCH_SIZE, buffers, stream, nullptr);
    float *output = (float*) malloc(YOLO_OUTPUT_SIZE * sizeof(float));
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], YOLO_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

    auto boxes = postProcess(output, YOLO_OUTPUT_SIZE);

    return boxes;
}

vector<Yolo::Detection> postProcess(float *output, const int &outputSize) {
    cout << "postProcess" << endl;
    vector<Yolo::Detection> result;
    Yolo yolo("config.yaml");
    const vector<vector<int>>& grids = yolo.getGrids();
    const vector<vector<int>>& anchors = yolo.getAnchors();
    int p = 0;
    int category = yolo.getCategory();
    for (int n = 0; n < grids.size(); ++n) {
        for (int c = 0; c < grids[n][0]; ++c) {
            vector<int> anchor = anchors[n * grids[n][0] + c];
            for (int h = 0; h < grids[n][1]; ++h) {
                for (int w = 0; w < grids[n][2]; ++w) {
                    float *row = output + p * (category + 5);
                    p++;
                    Yolo::Detection det{};
                    auto maxPos = max_element(row + 5, row + category + 5);
                    det.setProb(row[4] * row[maxPos - row]);
                    if (det.getProb() < yolo.getObjThreshold()) {
                        continue;
                    }
                    det.setClasses(maxPos - row - 5);
                    float px = (row[0] * 2 - 0.5 + w) / grids[n][2] * yolo.getInputW();
                    float py = (row[1] * 2 - 0.5 + h) / grids[n][1] * yolo.getInputH();
                    float pw = pow(row[2] * 2, 2) * anchor[0];
                    float ph = pow(row[3] * 2, 2) * anchor[1];
                    float xmin = px - pw / 2.0;
                    float ymin = py - ph / 2.0;
                    float xmax = px + pw / 2.0;
                    float ymax = py + ph / 2.0;
                    det.setX(xmin);
                    det.setY(ymin);
                    det.setW(pw);
                    det.setH(ph);
                    det.setXmax(xmax);
                    det.setYmax(ymax);
                    result.push_back(det);
                }
            }
        }
    }
    cout << "before nms result size: " << result.size() << endl;
    yolo.nms(result);
    cout << "after nms result size: " << result.size() << endl;
    return result;
}

void showResultYolo(const vector<Yolo::Detection> &dets, Mat &image) {
    for (int i = 0; i < dets.size(); ++i) {
        Yolo::Detection det = dets[i];
        float xmin = det.getX();
        float ymin = det.getY();
        float w = det.getW();
        float h = det.getH();
        float prob = det.getProb();
        Rect rec(xmin, ymin, w, h);
        rectangle(image, rec, Scalar(255, 255, 0), 2);
    }
    imshow("image", image);

    waitKey(0);
    destroyAllWindows();
}

vector<Yolo::Detection> fixDetections(vector<Yolo::Detection> &detections, float scale, float pw, float ph) {
    vector<Yolo::Detection> dets;
    for (int i = 0; i < detections.size(); ++i) {
        Yolo::Detection det = detections[i];
        float xmin = det.getX();
        float ymin = det.getY();
        float xmax = det.getXmax();
        float ymax = det.getYmax();
        float w = det.getW();
        float h = det.getH();
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
        det.setX(xmin);
        det.setY(ymin);
        det.setXmax(xmax);
        det.setYmax(ymax);
        det.setH(h);
        det.setW(w);
        dets.push_back(det);
    }

    return dets;
}
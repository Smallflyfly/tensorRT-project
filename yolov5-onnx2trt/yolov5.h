//
// Created by smallflyfly on 2021/6/28.
//

#ifndef YOLOV5_ONNX2TRT_YOLOV5_H
#define YOLOV5_ONNX2TRT_YOLOV5_H

#include "NvInfer.h"
#include <string>
#include "yaml-cpp/yaml.h"
#include <algorithm>
#include <vector>

#define max(a, b) a > b ? a : b
#define min(a, b) a < b ? a : b

#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            cerr << "Cuda failure: " << ret << endl; \
        } \
    } while(0)

using namespace std;

class Yolo {
public:
    class Detection {
    public:
        int getClasses() const {
            return classes;
        }

        void setClasses(int classes) {
            Detection::classes = classes;
        }

        float getX() const {
            return x;
        }

        void setX(float x) {
            Detection::x = x;
        }

        float getY() const {
            return y;
        }

        void setY(float y) {
            Detection::y = y;
        }

        float getW() const {
            return w;
        }

        void setW(float w) {
            Detection::w = w;
        }

        float getH() const {
            return h;
        }

        void setH(float h) {
            Detection::h = h;
        }

        float getProb() const {
            return prob;
        }

        void setProb(float prob) {
            Detection::prob = prob;
        }

        float getXmax() const {
            return xmax;
        }

        void setXmax(float xmax) {
            Detection::xmax = xmax;
        }

        float getYmax() const {
            return ymax;
        }

        void setYmax(float ymax) {
            Detection::ymax = ymax;
        }

        virtual ~Detection() {
        }

    private:
        int classes{};
        float x{};
        float y{};
        float w{};
        float h{};
        float prob{};
        float xmax{};
        float ymax{};
    };
    explicit Yolo(const string &config);

    ~Yolo();

    const vector<int> &getStrides() const;

    const vector<int> &getNumAnchors() const;

    const vector<vector<int>> &getAnchors() const;

    const vector<vector<int>> &getGrids() const;

    int getInputW() const;

    int getInputH() const;

    void readNames();

    int getCategory() const;

    float getObjThreshold() const;

    float getNmsThreshold() const;

    void nms(vector<Detection> &dets);

    float iouCalculate(const Detection &det1, const Detection &det2);

private:
    vector<int> strides;
    vector<int> numAnchors;
    vector<vector<int>> anchors;
    vector<vector<int>> grids;
    int inputW{640};
    int inputH{640};
    string nameFile;
    map<int, string> names;
    int category;
    float objThreshold;
    float nmsThreshold;
};

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
//    cout << dets[0].getProb() << endl;
//    cout << dets[1].getProb() << endl;
//    cout << dets[2].getProb() << endl;
//    cout << dets[3].getProb() << endl;
//    cout << dets[4].getProb() << endl;
    sort(dets.begin(), dets.end(), [=](const Detection &d1, const Detection &d2){
        return d1.getProb() > d2.getProb();
    });
//    cout << "=============================" << endl;
//    cout << dets[0].getProb() << endl;
//    cout << dets[1].getProb() << endl;
//    cout << dets[2].getProb() << endl;
//    cout << dets[3].getProb() << endl;
//    cout << dets[4].getProb() << endl;

    for (int i = 0; i < dets.size(); ++i) {
        for (int j = i+1; j < dets.size(); ++j) {
            if (dets[i].getClasses() == dets[j].getClasses()) {
                float iou = iouCalculate(dets[i], dets[j]);
                if (iou > nmsThreshold) {
                    dets[i].setProb(0.0);
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


#endif //YOLOV5_ONNX2TRT_YOLOV5_H

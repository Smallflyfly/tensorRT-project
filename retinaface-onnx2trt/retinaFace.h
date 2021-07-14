//
// Created by smallflyfly on 2021/7/13.
//

#ifndef RETINAFACE_ONNX2TRT_RETINAFACE_H
#define RETINAFACE_ONNX2TRT_RETINAFACE_H

#include <iostream>
#include <vector>

using namespace std;

#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            cerr << "Cuda failure: " << ret << endl; \
        } \
    } while(0)



class RetinaFace {
public:
    class Bbox{
    private:
        float xmin;
        float ymin;
        float xmax;
        float ymax;
    public:
        float getXmin() const;

        void setXmin(float xmin);

        float getYmin() const;

        void setYmin(float ymin);

        float getXmax() const;

        void setXmax(float xmax);

        float getYmax() const;

        void setYmax(float ymax);
    };
    class LandMark{
    private:
        vector<float> landmarks;
    public:
        const vector<float> &getLandmarks() const;

        void setLandmarks(const vector<float> &landmarks);
    };
    class Detection {
        float prob;
        Bbox bbox;
        LandMark landMark;
    public:
        float getProb() const;

        void setProb(float prob);

        const Bbox &getBbox() const;

        void setBbox(const Bbox &bbox);

        const LandMark &getLandMark() const;

        void setLandMark(const LandMark &landMark);
    };
};


#endif //RETINAFACE_ONNX2TRT_RETINAFACE_H

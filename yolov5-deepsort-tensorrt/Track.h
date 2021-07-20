//
// Created by smallflyfly on 2021/7/20.
//

#ifndef YOLOV5_ONNX2TRT_TRACK_H
#define YOLOV5_ONNX2TRT_TRACK_H


class Track {
private:
    float mean;
    float covariance;
    int trackId;
    int nInit;
    int maxAge;
    int timeSinceUpdate;

public:
    Track();
    Track();
};


#endif //YOLOV5_ONNX2TRT_TRACK_H

//
// Created by smallflyfly on 2021/6/28.
//

#ifndef YOLOV5_ONNX2TRT_YOLOV5_H
#define YOLOV5_ONNX2TRT_YOLOV5_H

#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            cerr << "Cuda failure: " << ret << endl; \
        } \
    } while(0)


#endif //YOLOV5_ONNX2TRT_YOLOV5_H

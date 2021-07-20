//
// Created by smallflyfly on 2021/6/28.
//

#include "yolov5.h"



using namespace std;
using namespace nvinfer1;
using namespace cv;

static Logger gLogger;

//inline int64_t volume(const nvinfer1::Dims& d)
//{
//    return accumulate(d.d, d.d + d.nbDims, 1, multiplies<int64_t>());
//}
//
//inline unsigned int getElementSize(nvinfer1::DataType t)
//{
//    switch (t)
//    {
//        case nvinfer1::DataType::kINT32: return 4;
//        case nvinfer1::DataType::kFLOAT: return 4;
//        case nvinfer1::DataType::kHALF: return 2;
//        case nvinfer1::DataType::kBOOL:
//        case nvinfer1::DataType::kINT8: return 1;
//    }
//    throw std::runtime_error("Invalid DataType.");
//}

int main(int argc, char **argv) {
    // argument ./yolov5_onnx2trt **.trt **.mp4
    if (argc != 3) {
        cerr << "argument error!" << endl;
        return -1;
    }
    String trtFile = argv[1];
    String videoFile = argv[2];
    size_t size{0};
    char *trtStream{nullptr};
    ifstream file(trtFile, ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtStream = new char[size];
        assert(trtStream != nullptr);
        file.read(trtStream, size);
        file.close();
    }

    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    ICudaEngine *engine = runtime->deserializeCudaEngine(trtStream, size);
    assert(engine != nullptr);

    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    delete[] trtStream;

    vector<Yolo::Detection> detections;

    //camera or video
    VideoCapture cap(videoFile);
    if (!cap.isOpened()) {
        cerr << "read camera error" << endl;
    }
    Mat frame, im;
    cap >> frame;
    int h = cap.get(CAP_PROP_FRAME_WIDTH);
    int w = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);
    cout << h << " " << w << " " << fps << endl;
    VideoWriter videoWriter;
    int coder = VideoWriter::fourcc('M', 'J', 'P', 'G');
//    int coder = VideoWriter::fourcc('m', 'p', '4', 'v');
    videoWriter.open("outPerson.avi", coder, fps, frame.size(), true);
    while (cap.isOpened()) {
        cap >> frame;
        if (frame.empty()) {
            cerr << "frame read error" << endl;
            break;
        }
        Mat image = frame.clone();
        cout << image.channels() << endl;
        // resize image to 640 * 640
        // fill size
        float scale = 1, pw = 0, ph = 0;
        float *data = prepareImage(frame, scale, pw, ph);
        detections = doInferenceYolo(*context, data);
        fixDetections(detections, scale, pw, ph);
        showResultYolo(detections, image);
        cout << image.rows << " " << image.cols << " " << image.channels() << endl;
        videoWriter.write(image);
        if (waitKey(1) == 'q') {
            break;
        }
    }

    videoWriter.release();
    cap.release();

    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;

}
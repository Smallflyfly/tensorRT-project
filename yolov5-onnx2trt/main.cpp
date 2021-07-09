//
// Created by smallflyfly on 2021/6/28.
//

#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include "NvInfer.h"
#include "logging.h"
#include "cuda_runtime.h"
#include "yolov5.h"
#include <numeric>
#include <vector>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc.hpp"


using namespace std;
using namespace nvinfer1;
using namespace cv;

static Logger gLogger;

static const char *ENGINE_FILE = "yolov5s.trt";
static const int BATCH_SIZE = 1;
static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const int64 OUTPUT_SIZE = 604800;
static const char *INPUT_BLOB_NAME = "images";
static const char *OUTPUT_BLOB_NAME = "output";

inline int64_t volume(const nvinfer1::Dims& d)
{
    return accumulate(d.d, d.d + d.nbDims, 1, multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
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

void showResult(const vector<Yolo::Detection> &dets, Mat &image,float scale, float pw, float ph) {
    for (int i = 0; i < dets.size(); ++i) {
        Yolo::Detection det = dets[i];
        float xmin = det.getX();
        float ymin = det.getY();
        float w = det.getW();
        float h = det.getH();
        float prob = det.getProb();
        xmin -= pw;
        ymin -= ph;
        xmin /= scale, ymin /= scale;
        w /= scale, h /= scale;
        Rect rec(xmin, ymin, w, h);
        rectangle(image, rec, Scalar(255, 255, 0), 2);
    }
    imshow("image", image);
}

vector<Yolo::Detection> doInference(IExecutionContext &context, float *input) {
    const ICudaEngine &engine = context.getEngine();
    cout << engine.getNbBindings() << endl;
    assert(engine.getNbBindings() == 2);
    void *buffers[2];
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_W * INPUT_H * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    //cuda stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, BATCH_SIZE * 3 * INPUT_W * INPUT_H * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(BATCH_SIZE, buffers, stream, nullptr);
    float *output = (float*) malloc(OUTPUT_SIZE * sizeof(float));
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

    auto boxes = postProcess(output, OUTPUT_SIZE);

    return boxes;
}

Mat resizeImage(Mat& im, float &scale, float &pw, float &ph) {
    int w = im.cols, h = im.rows;
    if (w > h) {
        scale = INPUT_W * 1.0 / w;
    } else {
        scale = INPUT_H * 1.0 / h;
    }
    int wNew = w * scale;
    int hNew = h * scale;
    pw = abs(INPUT_W - wNew) / 2;
    ph = abs(INPUT_H - hNew) / 2;
    Mat imNew = Mat::zeros(INPUT_H, INPUT_W, CV_8UC3);
    resize(im, im, Size(), scale, scale);
    im.copyTo(imNew(Rect(pw, ph, im.cols, im.rows)));
    return imNew;
}

float *prepareImage(Mat &frame, float &scale, float &pw, float &ph) {
    Mat im = resizeImage(frame, scale, pw, ph);
    im.convertTo(im, CV_32FC3, 1.0 / 255);
    float *data = (float *) malloc(INPUT_W * INPUT_H * 3 * sizeof(float));
    int index = 0;
    int channelLength = INPUT_W * INPUT_H;
    // BGR TO RGB
    vector<cv::Mat> splitImg = {
            cv::Mat(INPUT_H, INPUT_W, CV_32FC1, data + channelLength * (index + 2)),
            cv::Mat(INPUT_H, INPUT_W, CV_32FC1, data + channelLength * (index + 1)),
            cv::Mat(INPUT_H, INPUT_W, CV_32FC1, data + channelLength * (index + 0))
    };
    cv::split(im, splitImg);
    assert(data != nullptr);

    return data;
}

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
    videoWriter.open("out.avi", coder, fps, frame.size(), true);
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
        detections = doInference(*context, data);
        showResult(detections, image, scale, pw, ph);
        cout << image.rows << " " << image.cols << " " << image.channels() << endl;
        videoWriter.write(image);
        if (waitKey(1) == 'q') {
            break;
        }
    }
//    cv::Mat im = cv::imread("person.jpg");
//    Mat image = im.clone();
//    destroyAllWindows();
    videoWriter.release();
    cap.release();

    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;

}
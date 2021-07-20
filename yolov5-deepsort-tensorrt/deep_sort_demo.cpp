//
// Created by smallflyfly on 2021/7/15.
//

#include "extractor.h"
#include "yolov5.h"
#include "Deepsort.h"

int main() {
    // load yolo and extractor engine
    ICudaEngine *yoloEngine = readEngine("best.trt");
    cout << "Load yolo engine done!" << endl;

    IExecutionContext *yoloContext = yoloEngine->createExecutionContext();

    ICudaEngine *extractorEngine = readEngine("extractor.trt");
    cout << "Load extractor engine done!" << endl;

    IExecutionContext *extractorContext = extractorEngine->createExecutionContext();
    assert(extractorContext != nullptr);

    // read image and do inference
    float scale = 1.0;
    float pw = 0, ph = 0;
    cv::Mat im = cv::imread("person.jpg");
    Mat imOri = im.clone();
    float *yoloData = prepareImage(im, scale, pw, ph);
    cout << scale << " " << pw << " " << ph << endl;
    // yolo inference
    vector<Yolo::Detection> detections;
    detections = doInferenceYolo(*yoloContext, yoloData);
    detections = fixDetections(detections, scale, pw, ph);
//    showResultYolo(detections, imOri);

    NearestNeighborDistanceMetric metric(0.2, 100);
    DeepSort deepSort(0.2, 0.3, 0.5, 70, 3, 100);
    deepSort.update(detections, imOri, *extractorContext);


    for (int i = 0; i < detections.size(); ++i) {
        Yolo::Detection detection = detections[i];
        int xmin = detection.getX();
        int ymin = detection.getY();
        int w = detection.getW();
        int h = detection.getH();
        cout << xmin << " " << ymin << " " << w << " " << h <<endl;
        Mat imCrop;
        imCrop = imOri(Rect(xmin, ymin, w, h));
//        imshow("im", imCrop);
//        waitKey(0);
//        destroyAllWindows();



        //  extractor
        float escale = 1.0;
        int epw, eph;
        float *input = readImageData(imCrop, escale, epw, eph);
        float *output = (float *) malloc(EXTRACTOR_OUTPUT_SIZE * sizeof(float));
        cout << endl;
        output = doInferenceExtractor(*extractorContext, input);


    }



    return 0;
}

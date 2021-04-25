//
// Created by smallflyfly on 2021/4/22.
//

#ifndef YOLO5_YOLOLAYER_H
#define YOLO5_YOLOLAYER_H

#endif //YOLO5_YOLOLAYER_H

namespace Yolo {
    static const int INPUT_W = 640;
    static const int INPUT_H = 640;
    static const int CLASS_NUM = 1;
    static const int CHECK_COUNT = 3;

    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT];
    };
}

namespace nvinfer1
{
    class YoloLayerPlugin: public IPluginExt
    {
    public:
        explicit YoloLayerPlugin();
        YoloLayerPlugin(const void* data, size_t length);
        ~YoloLayerPlugin() override;

        int32_t getNbOutputs() const TRTNOEXCEPT override;

        Dims getOutputDimensions(int32_t index, const Dims *inputs, int32_t nbInputDims) TRTNOEXCEPT override;


    private:
        int mClassesCount;
        int mKernelCount;
        std::vector<Yolo::YoloKernel> mYoloKernel;
        int mThreadCount = 256;
        void** mAnchor;
        const char* mPluginNamespace;
    };
}
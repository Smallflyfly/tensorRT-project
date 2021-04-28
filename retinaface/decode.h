//
// Created by fangpf on 2021/4/28.
//

#ifndef RETINAFACE_DECODE_CUH
#define RETINAFACE_DECODE_CUH

#include "NvInfer.h"
#include <cstdio>




namespace decodeplugin
{
    struct alignas(float) Detection {
        float bbox[4];
        float class_confidence;
        float landmark[10];
    };
    static const int INPUT_H = 480;
    static const int INPUT_W = 640;
}

namespace nvinfer1
{
    class DecodePlugin: public IPluginV2IOExt
    {
    public:
        DecodePlugin();


    private:
        void forwardGpu(const float *const *inputs, float *output, cudaStream_t  stream, int batchSize = 1);
        int threadCount_ = 256;
        const char* mPluginNamespace;

    };


}



#endif //RETINAFACE_DECODE_CUH
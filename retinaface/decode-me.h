//
// Created by fangpf on 2021/4/28.
//

#ifndef RETINAFACE_DECODE_CUH
#define RETINAFACE_DECODE_CUH

#include "NvInfer.h"
#include <cstdio>
#include <string>
#include <vector>

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
        DecodePlugin(const void* data, size_t length);

        ~DecodePlugin();

        int getNbOutputs() const TRTNOEXCEPT override;

        Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) TRTNOEXCEPT override;

        int initialize() TRTNOEXCEPT override;

        void terminate() TRTNOEXCEPT override;

        size_t getWorkspaceSize(int maxBatchSize) const TRTNOEXCEPT override;

        int enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace,
                    cudaStream_t stream) TRTNOEXCEPT override;

        size_t getSerializationSize() const TRTNOEXCEPT override;

        void serialize(void *buffer) const TRTNOEXCEPT override;

        bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs,
                                       int nbOutputs) const TRTNOEXCEPT override;

        const char *getPluginType() const TRTNOEXCEPT override;

        const char *getPluginVersion() const TRTNOEXCEPT override;

        void destroy() TRTNOEXCEPT override;

        IPluginV2IOExt *clone() const TRTNOEXCEPT override;

        void setPluginNamespace(const char *pluginNamespace) TRTNOEXCEPT override;

        const char *getPluginNamespace() const TRTNOEXCEPT override;

        DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                   int nbInputs) const TRTNOEXCEPT override;

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted,
                                          int nbInputs) const TRTNOEXCEPT override;

        bool canBroadcastInputAcrossBatch(int inputIndex) const TRTNOEXCEPT override;

        void attachToContext(cudnnContext *context, cublasContext *cublasContext, IGpuAllocator *allocator) override;

        void configurePlugin(const PluginTensorDesc *in, int nbInput, const PluginTensorDesc *out,
                             int nbOutput) TRTNOEXCEPT override;

        void detachFromContext() override;


        int input_size_;

//    protected:
//        int getTensorRTVersion() const override;

    private:
        void forwardGpu(const float *const *inputs, float *output, cudaStream_t  stream, int batchSize = 1);
        int threadCount_ = 256;
        const char* mPluginNamespace;
    };

    class DecodePluginCreator : public IPluginCreator
    {
    public:
        DecodePluginCreator();

        virtual ~DecodePluginCreator();

        int getTensorRTVersion() const override;

        const char *getPluginName() const TRTNOEXCEPT override;

        const char *getPluginVersion() const TRTNOEXCEPT override;

        const PluginFieldCollection *getFieldNames() TRTNOEXCEPT override;

        IPluginV2IOExt *createPlugin(const char *name, const PluginFieldCollection *fc) TRTNOEXCEPT override;

        IPluginV2IOExt *
        deserializePlugin(const char *name, const void *serialData, size_t serialLength) TRTNOEXCEPT override;

        void setPluginNamespace(const char *pluginNamespace) TRTNOEXCEPT override;

        const char *getPluginNamespace() const TRTNOEXCEPT override;

    private:
        std::string mNamespace;
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
    };

    REGISTER_TENSORRT_PLUGIN(DecodePluginCreator);
};



#endif
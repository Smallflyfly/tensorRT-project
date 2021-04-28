//
// Created by fangpf on 2021/4/28.
//

#include "decode.h"

using namespace decodeplugin;

namespace nvinfer1
{
    DecodePlugin::DecodePlugin() {}
    DecodePlugin::~DecodePlugin() {
    }
    // create the plugin at runtime from a byte steam
    DecodePlugin::DecodePlugin(const void *data, size_t length) {
    }

    __device__ float Logist(float data) {
        return 1.0 / (1.0 + expf(-data));
    }

    __global__ void CalDetection(const float *input, float *output, int numElem, int step, int anchor, int outputElem) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= numElem) return;

        int h = decodeplugin::INPUT_H / step;
        int w = decodeplugin::INPUT_W / step;
        int totalGrid = h * w;
        int bnIdx = idx / totalGrid;
        idx = idx - bnIdx * totalGrid;
        int y = idx / w;
        int x = idx % w;
        const float *curInput = input + bnIdx * (4 + 2 + 10) * 2 * totalGrid;
        const float *bboxReg = &curInput[0];
        const float *clsReg = &curInput[2 * 4 * totalGrid];
        const float *lmReg = &curInput[2 * 4 * totalGrid + 2 * 2 + totalGrid];

        for (int k = 0; k < 2; k++) {
            float conf1 = clsReg[idx + k * totalGrid * 2];
            float conf2 = clsReg[idx + k * totalGrid * 2 + totalGrid];
            conf2 = expf(conf1) / (expf(conf1) + expf(conf2));
            if (conf2 <= 0.02) continue;

            float *resCount = output + bnIdx * outputElem;
            int count = atomicAdd(resCount, 1);
            char *data = (char *)resCount + sizeof(float) + count * sizeof(decodeplugin::Detection);
            decodeplugin::Detection* det = (decodeplugin::Detection*)(data);

            float prior[4];
            prior[0] = ((float)x + 0.5) / decodeplugin::INPUT_W;
            prior[1] = ((float)y+ 0.5) / decodeplugin::INPUT_H;
            prior[2] = (float)anchor * (k + 1) / decodeplugin::INPUT_W;
            prior[3] = (float)anchor * (k + 1) / decodeplugin::INPUT_H;

            // location
            det->bbox[0] = prior[0] + bboxReg[idx + k * totalGrid * 4] * 0.1 * prior[2];
            det->bbox[1] = prior[1] + bboxReg[idx + k * totalGrid * 4 + totalGrid] * 0.1 * prior[3];
            det->bbox[2] = prior[2] * expf(bboxReg[idx + k * totalGrid * 4 + totalGrid * 2] * 0.2);
            det->bbox[3] = prior[3] * expf(bboxReg[idx + k * totalGrid * 4 + totalGrid * 3] * 0.2);
            det->bbox[0] -= det->bbox[2] / 2;
            det->bbox[1] -= det->bbox[3] / 2;
            det->bbox[2] += det->bbox[0];
            det->bbox[3] += det->bbox[1];
            det->bbox[0] *= decodeplugin::INPUT_W;
            det->bbox[1] *= decodeplugin::INPUT_H;
            det->bbox[2] *= decodeplugin::INPUT_W;
            det->bbox[3] *= decodeplugin::INPUT_H;
            det->class_confidence = conf2;
            for (int i = 0; i < 10; i += 2) {
                det->landmark[i] = prior[0] + lmReg[idx + k * totalGrid * 10 + totalGrid] * 0.1  * prior[2];
                det->landmark[i+1] = prior[1] + lmReg[idx + k * totalGrid * 10 + totalGrid * (i + 1)] * 0.1 * prior[3];
                det->landmark[i] *= decodeplugin::INPUT_W;
                det->landmark[i+1] *= decodeplugin::INPUT_H;
            }
        }
    }

    void DecodePlugin::forwardGpu(const float *const *inputs, float *output, cudaStream_t stream, int batchSize) {
        int numElem = 0;
        int baseStep = 8;
        int baseAnchor = 16;
        int threadCount;

        int totalCount = 1;
        totalCount += decodeplugin::INPUT_H / 8 * decodeplugin::INPUT_W / 8 * 2 * sizeof(decodeplugin::Detection) / sizeof(float);
        totalCount += decodeplugin::INPUT_H / 16 * decodeplugin::INPUT_W / 16 * 2 * sizeof(decodeplugin::Detection) / sizeof(float);
        totalCount += decodeplugin::INPUT_H / 32 * decodeplugin::INPUT_W / 32 * 2 * sizeof(decodeplugin::Detection) / sizeof(float);
        for (int i = 0; i < batchSize; i++) {
            cudaMemset(output + i + totalCount, 0, sizeof(float));
        }

        for (int i = 0; i < 3; i++) {
            numElem = batchSize * decodeplugin::INPUT_H / baseStep * decodeplugin::INPUT_W / baseStep;
            threadCount = (numElem < threadCount_) ? numElem : threadCount_;
            CalDetection <<< (numElem + threadCount - 1) / threadCount, threadCount>>> (inputs[i], output, numElem, baseStep, baseAnchor, totalCount);
            baseStep *= 2;
            baseAnchor *= 4;
        }
    }

    const char *DecodePlugin::getPluginType() const {
        return "Decode_TRT";
    }

    const char *DecodePlugin::getPluginVersion() const {
        return "1";
    }

    int DecodePlugin::getNbOutputs() const {
        return 0;
    }

    Dims DecodePlugin::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) {
        // output the result to channel
        int totalCount = 0;
        totalCount += decodeplugin::INPUT_H / 8 * decodeplugin::INPUT_W / 8 * 2 * sizeof(decodeplugin::Detection) /
                sizeof(float);
        totalCount += decodeplugin::INPUT_H / 16 * decodeplugin::INPUT_W / 16 * 2 * sizeof(decodeplugin::Detection) /
                sizeof(float);
        totalCount += decodeplugin::INPUT_H / 32 * decodeplugin::INPUT_W / 32 * 2 * sizeof(decodeplugin::Detection) /
                sizeof(float);
        return Dims3(totalCount + 1, 1, 1);
    }

    int DecodePlugin::initialize() {
        return 0;
    }

    void DecodePlugin::terminate() {

    }

    size_t DecodePlugin::getWorkspaceSize(int maxBatchSize) const {
        return 0;
    }

    int DecodePlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace,
                                  cudaStream_t stream) {
        // GPU
        forwardGpu((const float *const *)inputs, (float *)outputs[0], stream, batchSize);
        return 0;
    }

    size_t DecodePlugin::getSerializationSize() const {
        return 0;
    }

    void DecodePlugin::serialize(void *buffer) const {

    }

    void DecodePlugin::destroy() {
        delete this;
    }

    void DecodePlugin::setPluginNamespace(const char *pluginNamespace) {
        mPluginNamespace = pluginNamespace;
    }

    const char *DecodePlugin::getPluginNamespace() const {
        return mPluginNamespace;
    }

    DataType
    DecodePlugin::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
        return DataType::kFLOAT;
    }

    bool DecodePlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted,
                                                    int nbInputs) const {
        return false;
    }

    bool DecodePlugin::canBroadcastInputAcrossBatch(int inputIndex) const {
        return false;
    }

    void DecodePlugin::attachToContext(cudnnContext *context, cublasContext *cublasContext, IGpuAllocator *allocator) {
    }

    void DecodePlugin::detachFromContext() {
    }

    // clone the plugin
    IPluginV2IOExt *DecodePlugin::clone() const {
        DecodePlugin *p = new DecodePlugin();
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    void DecodePlugin::configurePlugin(const PluginTensorDesc *in, int nbInput, const PluginTensorDesc *out,
                                       int nbOutput) {

    }

    bool DecodePlugin::supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs,
                                                 int nbOutputs) const {
        return false;
    }

    int DecodePlugin::getTensorRTVersion() const {
        return IPluginV2IOExt::getTensorRTVersion();
    }

    PluginFieldCollection DecodePluginCreator::mFC{};
    std::vector<PluginField> DecodePluginCreator::mPluginAttributes;

    DecodePluginCreator::DecodePluginCreator() {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    DecodePluginCreator::~DecodePluginCreator() {

    }



    int DecodePluginCreator::getTensorRTVersion() const {
        return IPluginCreator::getTensorRTVersion();
    }

    const char *DecodePluginCreator::getPluginName() const {
        return "Decode_TRT";
    }

    const char *DecodePluginCreator::getPluginVersion() const {
        return "1";
    }

    const PluginFieldCollection *DecodePluginCreator::getFieldNames() {
        return &mFC;
    }

    IPluginV2IOExt *DecodePluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) {
        DecodePlugin *obj = new DecodePlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt *DecodePluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) {
        // This object will be deleted when the network is destroyed, which will
        // call PReluPlugin::destroy()
        DecodePlugin *obj = new DecodePlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    void DecodePluginCreator::setPluginNamespace(const char *pluginNamespace) {
        mNamespace = pluginNamespace;
    }

    const char *DecodePluginCreator::getPluginNamespace() const {
        return mNamespace.c_str();
    }

}



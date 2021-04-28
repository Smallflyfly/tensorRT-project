//
// Created by fangpf on 2021/4/28.
//

#include "decode.h"

namespace nvinfer1
{
    DecodePlugin::DecodePlugin() {}

    __glob__ void CalDetection(const float *input, float *output, int numElem, int step, int anchor, int outputElem) {
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
            CalDetection << (numElem + threadCount - 1) / threadCount, threadCount>> ()

        }
    }
}



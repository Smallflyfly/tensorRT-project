//
// Created by smallflyfly on 2021/4/22.
//

#ifndef YOLO5_COMMON_HPP
#define YOLO5_COMMON_HPP

#endif //YOLO5_COMMON_HPP
#include "yololayer.h"

ILayer* focus(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int inCh,
              int outCh, int kSize, const std::string& layerName) {
    ISliceLayer* s1 = network->addSlice(input, Dims3{0, 0, 0}, Dims3{inCh, Yolo::INPUT_H / 2,
                                                                     Yolo::INPUT_W / 2}, Dims3{1, 2, 2});

}
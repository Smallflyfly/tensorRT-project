#include "yololayer.h"

using namespace Yolo;

namespace nvinfer1
{
    YoloLayerPlugin::

    YoloLayerPlugin::YoloLayerPlugin() {

    }

    YoloLayerPlugin::YoloLayerPlugin(const void *data, size_t length) {

    }

    YoloLayerPlugin::~YoloLayerPlugin() {

    }

    int32_t YoloLayerPlugin::getNbOutputs() const {
        return 1;
    }

    Dims YoloLayerPlugin::getOutputDimensions(int32_t index, const Dims *inputs, int32_t nbInputDims) {
        return Dims();
    }
}
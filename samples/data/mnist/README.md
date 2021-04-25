# Setting Up MNIST Samples

## Downloading PGMs for Inference

To download PGMs from the MNIST dataset, you can run the included `download_pgms.py` script:
```
python3 download_pgms.py
```

## Models

mnist.onnx: Opset 8, Retrieved from [ONNX Model Zoo](https://github.com/onnx/models/tree/master/vision/classification/mnist)

## Run ONNX model with trtexec

* FP32 precisons with fixed batch size 1
  * `./trtexec --explicitBatch --onnx=mnist.onnx --workspace=1024`
* Other precisions
  * Add `--fp16` for FP16 and `--int8` for INT8.
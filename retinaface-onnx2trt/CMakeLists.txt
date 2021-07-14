cmake_minimum_required(VERSION 3.19)
project(retinaface_onnx2trt)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(OpenCV)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
link_directories(/usr/local/cuda/lib)

include_directories(/home/TensorRT-7.2.2.3/samples/common)
include_directories(/home/TensorRT-7.2.2.3/include)
link_directories(/home/TensorRT-7.2.2.3/lib)
set(TENSORRT_LIB_PATH /home/TensorRT-7.2.2.3/lib)
file(GLOB LIBS ${TENSORRT_LIB_PATH}/*.so)

add_executable(retinaface_onnx2trt
        main.cpp retinaFace.cpp)

target_link_libraries(retinaface_onnx2trt ${LIBS} ${CUDA_LIBRARIES})
target_link_libraries(retinaface_onnx2trt ${OpenCV_LIBS} ${TENSORRT_LIB})
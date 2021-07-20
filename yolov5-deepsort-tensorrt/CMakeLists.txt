cmake_minimum_required(VERSION 3.17)
project(yolov5_onnx2trt)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

#add_executable(yolov5_onnx2trt
#        main.cpp)
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

set(YAML_INCLUDE ./yaml-cpp/include)
set(YAML_LIB_DIR ./yaml-cpp/libs)

#include_directories(${YAML_INCLUDE})
#link_libraries(${YAML_LIB_DIR})

add_executable(yolov5_onnx2trt
        deep_sort_demo.cpp yolov5.cpp extractor.cpp Deepsort.cpp NearestNeighborDistanceMetric.cpp Detection.cpp
        Tracker.cpp KalmanFilter.cpp Track.cpp Tracker.cpp)

target_link_libraries(yolov5_onnx2trt ${LIBS} ${CUDA_LIBRARIES})
target_link_libraries(yolov5_onnx2trt ${OpenCV_LIBS} ${TENSORRT_LIB} libyaml-cpp.a)

#include_directories(C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA/v10.2\\include)
#include_directories(D:\\TensorRT-7.0.0.11\\include)
#include_directories(D:\\TensorRT-7.0.0.11\\samples\\common\\windows)
#include_directories(D:\\TensorRT-7.0.0.11\\samples\\common)
#include_directories(D:\\TensorRT-7.0.0.11\\bin)
#include_directories(C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2\\lib\\x64)
#
#link_directories(C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2\\lib\\x64)
#link_directories(D:\\TensorRT-7.0.0.11\\lib)
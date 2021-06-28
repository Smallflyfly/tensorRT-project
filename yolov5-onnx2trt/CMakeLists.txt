cmake_minimum_required(VERSION 3.17)
project(yolov5_onnx2trt)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

add_executable(yolov5_onnx2trt
        main.cpp)
find_package(OpenCV)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(D:\\TensorRT-7.0.0.11\\lib)
set(TENSORRT_LIB_PATH D:\\TensorRT-7.0.0.11\\lib)
file(GLOB LIBS ${TENSORRT_LIB_PATH}\\*)

target_link_libraries(yolov5_onnx2trt ${LIBS} ${CUDA_LIBRARIES})
target_link_libraries(yolov5_onnx2trt ${OpenCV_LIBS} ${TENSORRT_LIB})

include_directories(C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA/v10.2\\include)
include_directories(D:\\TensorRT-7.0.0.11\\include)
include_directories(D:\\TensorRT-7.0.0.11\\samples\\common\\windows)
include_directories(D:\\TensorRT-7.0.0.11\\samples\\common)
include_directories(D:\\TensorRT-7.0.0.11\\bin)
include_directories(C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2\\lib\\x64)

link_directories(C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2\\lib\\x64)
link_directories(D:\\TensorRT-7.0.0.11\\lib)
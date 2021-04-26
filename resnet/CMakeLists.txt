cmake_minimum_required(VERSION 3.19)
project(resnet)

set(CMAKE_CXX_STANDARD 14)

add_executable(resnet
        resnet50.cpp)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
link_directories(/usr/local/cuda/lib)

include_directories(/home/TensorRT-7.2.2.3/samples/common)
include_directories(/home/TensorRT-7.2.2.3/include)
link_directories(/home/TensorRT-7.2.2.3/lib)
set(TENSORRT_LIB_PATH /home/TensorRT-7.2.2.3/lib)
file(GLOB LIBS ${TENSORRT_LIB_PATH}/*.so)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(resnet ${LIBS} ${CUDA_LIBRARIES})



#target_link_libraries(resnet ${CUDA_LIBRARIES})

#include_directories(C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA/v10.2\\include)
#include_directories(D:\\TensorRT-7.0.0.11\\include)
#include_directories(D:\\TensorRT-7.0.0.11\\samples\\common\\windows)
#include_directories(D:\\TensorRT-7.0.0.11\\samples\\common)
#include_directories(D:\\TensorRT-7.0.0.11\\bin)
#include_directories(C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2\\lib\\x64)
#
#link_directories(C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2\\lib\\x64)
#link_directories(D:\\TensorRT-7.0.0.11\\lib)


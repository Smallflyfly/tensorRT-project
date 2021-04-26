cmake_minimum_required(VERSION 3.19)
project(yolo5)

set(CMAKE_CXX_STANDARD 11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(SAMPLES_COMMON_SOURCES ./common/logger.cpp)

find_package(CUDA REQUIRED)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)

target_link_libraries(myplugins nvinfer cudart)


add_executable(yolo5 ${SAMPLES_COMMON_SOURCES} main.cpp)


find_package(OpenCV REQUIRED)

target_link_libraries(yolo5 ${OpenCV_LIBS} ${TENSORRT_LIB})

target_link_libraries(yolo5 nvinfer)
target_link_libraries(yolo5 cudart)
target_link_libraries(yolo5 myplugins)

include_directories(./common/)
include_directories(/home/TensorRT-7.2.2.3/samples/common)
include_directories(/home/TensorRT-7.2.2.3/include)
link_directories(/home/TensorRT-7.2.2.3/lib)
set(TENSORRT_LIB_PATH /home/TensorRT-7.2.2.3/lib)
file(GLOB LIBS ${TENSORRT_LIB_PATH}/*.so)
# add CUDA

message(“CUDA_LIBRARIES:${CUDA_LIBRARIES}”)
message(“CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}”)
include_directories(${CUDA_INCLUDE_DIRS})


target_link_libraries(yolo5 ${LIBS} ${CUDA_LIBRARIES})

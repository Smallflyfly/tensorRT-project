cmake_minimum_required(VERSION 3.19)
project(yolo5)

set(CMAKE_CXX_STANDARD 14)

set(SAMPLES_COMMON_SOURCES ./common/logger.cpp)

add_executable(yolo5 ${SAMPLES_COMMON_SOURCES} main.cpp)


find_package(OpenCV REQUIRED)

target_link_libraries(yolo5 ${OpenCV_LIBS} ${TENSORRT_LIB})
include_directories(./common/)
include_directories(/home/TensorRT-7.2.2.3/samples/common)
include_directories(/home/TensorRT-7.2.2.3/include)
set(TENSORRT_LIB_PATH /home/TensorRT-7.2.2.3/lib)
file(GLOB LIBS ${TENSORRT_LIB_PATH}/*.so)
# add CUDA
find_package(CUDA 11.0 REQUIRED)
message(“CUDA_LIBRARIES:${CUDA_LIBRARIES}”)
message(“CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}”)
include_directories(${CUDA_INCLUDE_DIRS})


target_link_libraries(yolo5 ${LIBS} ${CUDA_LIBRARIES})

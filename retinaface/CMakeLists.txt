cmake_minimum_required(VERSION 3.19)
project(retinaface)

set(CMAKE_CXX_STANDARD 14)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

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

cuda_add_library(decodeplugin SHARED decode.cu)
target_link_libraries(decodeplugin nvinfer cudart)

find_package(OpenCV REQUIRED)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(retinaface retinaface.cpp)

target_link_libraries(retinaface nvinfer)
target_link_libraries(retinaface cudart)
target_link_libraries(retinaface decodeplugin)
target_link_libraries(retinaface ${OpenCV_LIBRARIES})

target_link_libraries(retinaface ${LIBS} ${CUDA_LIBRARIES})

add_definitions(-O2 -pthread)
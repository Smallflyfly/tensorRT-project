# cmake needs this line
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 14)

# Define project name
project(opencv_example_project)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
#message(STATUS "OpenCV library status:")
#message(STATUS "    config: ${OpenCV_DIR}")
#message(STATUS "    version: ${OpenCV_VERSION}")
#message(STATUS "    libraries: ${OpenCV_LIBS}")
#message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# SAMPLES_COMMON_SOURCES
set(SAMPLES_COMMON_SOURCES ./common/logger.cpp)

#add_executable(${SAMPLES_COMMON_SOURCES})

# Declare the executable target built from your sources
#add_executable(opencv_example ${SAMPLES_COMMON_SOURCES} sampleOnnxMNIST.cpp)
add_executable(opencv_example ${SAMPLES_COMMON_SOURCES} retinaFaceOnnx.cpp)

# Link your application with OpenCV libraries
#set(TENSORRT_DIR /home/TensorRT-7.2.2.3)
#set(TENSORRT_LIB /home/TensorRT-7.2.2.3/lib)
#include_directories(${TENSORRT_DIR}/include)
#include_directories(${TENSORRT_DIR}/samples/common)
#link_directories(${TENSORRT_DIR}/lib)
#file(GLOB LIBS “${TENSORRT_LIB}/*.so”)
#
#
#find_package(CUDA 11.0 REQUIRED)
##find_package(CUDA REQUIRED)
#
##message(“CUDA_LIBRARIES:${CUDA_LIBRARIES}”)
##message(“CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}”)
#set(CUDA_INCLUDE_DIRS /home/cuda-11.0/include)
#include_directories(${CUDA_INCLUDE_DIRS})
#
##if(NOT CUDA_LIBRARY_DIRS)
##    set(CUDA_LIBRARY_DIRS /usr/local/cuda/lib64)
##endif()
#set(CUDA_LIBRARY_DIRS /home/cuda-11.0/lib64)
#link_directories(/home/cuda-11.0/lib64)
#include_directories(/home/cuda-11.0/include)
#
target_link_libraries(opencv_example ${LIBS} ${CUDA_LIBRARIES})
target_link_libraries(opencv_example ${OpenCV_LIBS} ${TENSORRT_LIB})
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
# 和文件连在一起
target_link_libraries(opencv_example  ${CUDA_LIBRARIES} ${LIBS})

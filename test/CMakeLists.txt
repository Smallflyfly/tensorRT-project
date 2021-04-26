cmake_minimum_required(VERSION 3.1)
project(test)

set(CMAKE_CXX_STANDARD 14)

add_executable(test main.cpp)

#set(OpenCV_DIR /home/smallflyfly/opencv-4.3.0)
#include_directories(${OpenCV_DIR}/lib/include)
#include_directories(${OpenCV_DIR}/build/include)

find_package(OpenCV REQUIRED)

target_link_libraries(test ${OpenCV_LIBS})
cmake_minimum_required(VERSION 3.19)
project(test)

set(CMAKE_CXX_STANDARD 14)

add_executable(test main.cpp)

find_package(OpenCV REQUIRED)

target_link_libraries(test ${OpenCV_LIBS})
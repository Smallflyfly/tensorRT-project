# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/smallflyfly/clion-2021.1/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/smallflyfly/clion-2021.1/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/smallflyfly/DATA_MANAGER/AI/TensorRT-project/yolo5

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/smallflyfly/DATA_MANAGER/AI/TensorRT-project/yolo5/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/yolo5.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/yolo5.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/yolo5.dir/flags.make

CMakeFiles/yolo5.dir/common/logger.cpp.o: CMakeFiles/yolo5.dir/flags.make
CMakeFiles/yolo5.dir/common/logger.cpp.o: ../common/logger.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/smallflyfly/DATA_MANAGER/AI/TensorRT-project/yolo5/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/yolo5.dir/common/logger.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolo5.dir/common/logger.cpp.o -c /media/smallflyfly/DATA_MANAGER/AI/TensorRT-project/yolo5/common/logger.cpp

CMakeFiles/yolo5.dir/common/logger.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolo5.dir/common/logger.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/smallflyfly/DATA_MANAGER/AI/TensorRT-project/yolo5/common/logger.cpp > CMakeFiles/yolo5.dir/common/logger.cpp.i

CMakeFiles/yolo5.dir/common/logger.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolo5.dir/common/logger.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/smallflyfly/DATA_MANAGER/AI/TensorRT-project/yolo5/common/logger.cpp -o CMakeFiles/yolo5.dir/common/logger.cpp.s

CMakeFiles/yolo5.dir/main.cpp.o: CMakeFiles/yolo5.dir/flags.make
CMakeFiles/yolo5.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/smallflyfly/DATA_MANAGER/AI/TensorRT-project/yolo5/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/yolo5.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolo5.dir/main.cpp.o -c /media/smallflyfly/DATA_MANAGER/AI/TensorRT-project/yolo5/main.cpp

CMakeFiles/yolo5.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolo5.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/smallflyfly/DATA_MANAGER/AI/TensorRT-project/yolo5/main.cpp > CMakeFiles/yolo5.dir/main.cpp.i

CMakeFiles/yolo5.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolo5.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/smallflyfly/DATA_MANAGER/AI/TensorRT-project/yolo5/main.cpp -o CMakeFiles/yolo5.dir/main.cpp.s

# Object files for target yolo5
yolo5_OBJECTS = \
"CMakeFiles/yolo5.dir/common/logger.cpp.o" \
"CMakeFiles/yolo5.dir/main.cpp.o"

# External object files for target yolo5
yolo5_EXTERNAL_OBJECTS =

yolo5: CMakeFiles/yolo5.dir/common/logger.cpp.o
yolo5: CMakeFiles/yolo5.dir/main.cpp.o
yolo5: CMakeFiles/yolo5.dir/build.make
yolo5: /usr/local/lib/libopencv_dnn.so.4.3.0
yolo5: /usr/local/lib/libopencv_gapi.so.4.3.0
yolo5: /usr/local/lib/libopencv_highgui.so.4.3.0
yolo5: /usr/local/lib/libopencv_ml.so.4.3.0
yolo5: /usr/local/lib/libopencv_objdetect.so.4.3.0
yolo5: /usr/local/lib/libopencv_photo.so.4.3.0
yolo5: /usr/local/lib/libopencv_stitching.so.4.3.0
yolo5: /usr/local/lib/libopencv_video.so.4.3.0
yolo5: /usr/local/lib/libopencv_videoio.so.4.3.0
yolo5: libmyplugins.so
yolo5: /home/TensorRT-7.2.2.3/lib/libmyelin.so
yolo5: /home/TensorRT-7.2.2.3/lib/libnvcaffe_parser.so
yolo5: /home/TensorRT-7.2.2.3/lib/libnvinfer.so
yolo5: /home/TensorRT-7.2.2.3/lib/libnvinfer_plugin.so
yolo5: /home/TensorRT-7.2.2.3/lib/libnvonnxparser.so
yolo5: /home/TensorRT-7.2.2.3/lib/libnvparsers.so
yolo5: /usr/local/cuda/lib64/libcudart_static.a
yolo5: /usr/lib/x86_64-linux-gnu/librt.so
yolo5: /usr/local/lib/libopencv_imgcodecs.so.4.3.0
yolo5: /usr/local/lib/libopencv_calib3d.so.4.3.0
yolo5: /usr/local/lib/libopencv_features2d.so.4.3.0
yolo5: /usr/local/lib/libopencv_flann.so.4.3.0
yolo5: /usr/local/lib/libopencv_imgproc.so.4.3.0
yolo5: /usr/local/lib/libopencv_core.so.4.3.0
yolo5: CMakeFiles/yolo5.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/smallflyfly/DATA_MANAGER/AI/TensorRT-project/yolo5/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable yolo5"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yolo5.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/yolo5.dir/build: yolo5

.PHONY : CMakeFiles/yolo5.dir/build

CMakeFiles/yolo5.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/yolo5.dir/cmake_clean.cmake
.PHONY : CMakeFiles/yolo5.dir/clean

CMakeFiles/yolo5.dir/depend:
	cd /media/smallflyfly/DATA_MANAGER/AI/TensorRT-project/yolo5/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/smallflyfly/DATA_MANAGER/AI/TensorRT-project/yolo5 /media/smallflyfly/DATA_MANAGER/AI/TensorRT-project/yolo5 /media/smallflyfly/DATA_MANAGER/AI/TensorRT-project/yolo5/cmake-build-debug /media/smallflyfly/DATA_MANAGER/AI/TensorRT-project/yolo5/cmake-build-debug /media/smallflyfly/DATA_MANAGER/AI/TensorRT-project/yolo5/cmake-build-debug/CMakeFiles/yolo5.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/yolo5.dir/depend


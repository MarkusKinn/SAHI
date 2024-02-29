cmake_minimum_required(VERSION 3.0)
project(SAHI)

# Set the OpenCV_DIR variable to the directory containing OpenCVConfig.cmake.
# This is usually inside the build directory of OpenCV or in 'share/OpenCV' of the installation directory.
set(OpenCV_DIR /usr/local/share/opencv4/opencv2)

# Find the OpenCV package.
find_package(OpenCV REQUIRED)

# Include directories for OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable.
add_executable(SAHI main.cpp
        SAHI.cpp
        SAHI.h
        CpuModel.cpp
        CpuModel.h
        Model.cpp
        Model.h
        yolov8.h)

# Link OpenCV libraries with the executable.
target_link_libraries(SAHI ${OpenCV_LIBS})

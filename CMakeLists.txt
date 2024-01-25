cmake_minimum_required(VERSION 3.0)
project(SAHI)

find_package(OpenCV REQUIRED) # Find OpenCV package

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(SAHI main.cpp ImageSlicer.h
        ImageSlicer.tpp
        ImageSlicer.tpp
        NMS.h
        temp.txt
        main.cpp)  # Assuming ImageSlicer.cpp is the name of the file
target_link_libraries(SAHI ${OpenCV_LIBS}) # Link OpenCV libraries

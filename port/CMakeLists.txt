cmake_minimum_required(VERSION 3.1)
project(port)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    src/common/microtime.cpp
    src/common/microtime.h
    src/gpu/cudaMain.cu
    src/main.cpp
    Makefile)

add_executable(port ${SOURCE_FILES} src/synfire/neuron.h)
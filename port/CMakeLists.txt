cmake_minimum_required(VERSION 3.1)
project(port)

set(CXX_FLAGS "-std=c++0x -malign-double -Wall")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_FLAGS}")

execute_process(COMMAND cygpath.exe -u $ENV{CUDA_INC_PATH}
                OUTPUT_VARIABLE CUDA_INC)

include_directories("${CUDA_INC}" SYSTEM)

set(SOURCE_FILES
    src/common/microtime.cpp
    src/common/microtime.h
    src/common/utility.h
    src/synfire/neuron.cpp
    src/synfire/neuron.h
    src/synfire/synapses.cpp
    src/synfire/synapses.h
    src/main.cpp
    src/common/random.h)

include_directories(.)
include_directories(src)
include_directories(src/common)
include_directories(src/synfire)


add_executable(port src/main.cpp ${SOURCE_FILES} src/synfire/synfire.h src/synfire/synfire.cpp src/common/random.cpp src/common/helpers.h src/common/helpers.cpp)
add_executable(gsynfire src/gpu/cudaMain.cu ${SOURCE_FILES} src/common/cuda_utils.h src/synfire/synfire.h src/synfire/synfire.cpp src/common/random.cpp src/gpu/CUSynfire.cpp src/gpu/CUSynfire.h src/common/helpers.h src/common/helpers.cpp)


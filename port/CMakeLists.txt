cmake_minimum_required(VERSION 3.1)
project(port)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CUDA_INC "/$ENV{CUDA_INC_PATH}/")

execute_process(COMMAND cygpath.exe -u $ENV{CUDA_INC_PATH}
                OUTPUT_VARIABLE CUDA_INC)

#MACRO(HEADER_DIRECTORIES return_list)
#    file(GLOB RECURSE expr "${CUDA_INC}/*.h")
#    SET(dir_list "")
#    FOREACH(file_path ${expr})
#        GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
#        SET(dir_list ${dir_list} ${dir_path})
#        message("adding ${dir_path}")
#    ENDFOREACH()
#    LIST(REMOVE_DUPLICATES dir_list)
#    SET(${return_list} ${dir_list})
#ENDMACRO()

#HEADER_DIRECTORIES(CUDA_DIRS)

message("Yo sup ${CUDA_INC}")
include_directories("${CUDA_INC}" SYSTEM)
#link_directories("${CNAME}")

get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("inc_dirs = ${inc_dirs}")

set(SOURCE_FILES
    src/common/microtime.cpp
    src/common/microtime.h
    src/common/utility.h
    src/synfire/neuron.cpp
    src/synfire/neuron.h
    src/synfire/synapses.cpp
    src/synfire/synapses.h
    src/main.cpp)

include_directories(.)
include_directories(src)
include_directories(src/common)
include_directories(src/synfire)


add_executable(port ${SOURCE_FILES})

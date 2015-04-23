MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
 CWD := $(notdir $(patsubst %/,%,$(dir $(MK_PATH))))

# Path Constants
SOURCES := ./src/

EXECUTABLE := synfireGrowth

CXX := g++44
CXXFLAGS := -std=c++0x
# CXXFLAGS := -W -Wall -Wimplicit -Wswitch -Wformat -Wchar-subscripts -Wparentheses -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type -Wno-unused-function -fno-strict-aliasing
CXXINCLUDES := -I./src/port/ -I./src/common/ -I./src/gpu/ -I./src/synfire/

NVCC := /usr/local/cuda/bin/nvcc
NVCCFLAGS := -O3 -arch sm_30
NVCCINCLUDES := -I. -I/usr/local/cuda/include  -I/usr/local/cuda/NVIDIA_GPU_Computing_SDK/shared/inc -I../src/port/
NVCCLIBS := -L/usr/local/cuda/lib64  -lGL -lcudart

SRC_PATH := src/
GPU_PATH := src/gpu/
SYN_PATH := src/synfire/
CMN_PATH := src/common/

default: $(EXECUTABLE)

# GPU MAKE TARGETS
syn_gpu: $(GPU_PATH)cudaMain.cu microtime.o CUSynfire.o neuron.o synapses.o microtime.o random.o
	$(NVCC) $(NVCCINCLUDES) $(CXXINCLUDES) $(NVCCFLAGS) $^ -o $@

CUSynfire.o: $(CMN_PATH)microtime.h $(GPU_PATH)CUSynfire.h $(GPU_PATH)CUSynfire.cpp
	$(NVCC) $(CXXINCLUDES) $(CXXINCLUDES) -c $(GPU_PATH)CUSynfire.cpp

# CPU MAKE TARGETS
synfireGrowth: $(SRC_PATH)main.cpp synfire.o microtime.o synfire.o neuron.o synapses.o microtime.o random.o
	$(CXX) $(CXXFLAGS) $(CXXINCLUDES) $^ -o $@

synfire.o: $(CMN_PATH)microtime.h $(SYN_PATH)synfire.h $(SYN_PATH)synfire.cpp
	$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c $(SYN_PATH)synfire.cpp

microtime.o: $(CMN_PATH)microtime.h $(CMN_PATH)microtime.cpp
	$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c $(CMN_PATH)microtime.cpp

synapses.o: $(SYN_PATH)synapses.h $(SYN_PATH)synapses.cpp
	$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c  $(SYN_PATH)synapses.cpp

neuron.o: $(SYN_PATH)neuron.h $(SYN_PATH)neuron.cpp
	$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c $(SYN_PATH)neuron.cpp

random.o: $(CMN_PATH)random.h $(CMN_PATH)random.cpp
	$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c $(CMN_PATH)random.cpp

clean:
	rm -rf *.o *~

realclean:
	rm -rf *.o *~ $(EXECUTABLE)

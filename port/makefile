MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CWD := $(notdir $(patsubst %/,%,$(dir $(MK_PATH))))

# Path Constants
HEADERS := ./headers/
SOURCES := ./src/

EXECUTABLE := synfire

CXX := g++

# CXXFLAGS := -W -Wall -Wimplicit -Wswitch -Wformat -Wchar-subscripts -Wparentheses -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type -Wno-unused-function -fno-strict-aliasing
# CXXINCLUDES := -I../src/port/ -I./

# NVCCFLAGS := -O3 -arch sm_30
# NVCCINCLUDES := -I. -I/usr/local/cuda/include  -I/usr/local/cuda/NVIDIA_GPU_Computing_SDK/shared/inc -I../src/port/
# NVCCLIBS := -L/usr/local/cuda/lib64  -lGL -lcudart

# SRC_PATH := ../src/port/

# default: $(EXECUTABLE)

# synfire: synfireGrowth.o synapses.o neuron.o microtime.o synfireGrowthCuda.o
# 	$(CXX) $(CXXFLAGS) $(CXXINCLUDES) $(NVCCLIBS) $(NVCCINCLUDES) $^ -o $@

# synfireGrowthCuda.o: synfireGrowthCuda.cu microtime.h $(SRC_PATH)synapses.h
# 	/usr/local/cuda/bin/nvcc $(NVCCFLAGS) $(NVCCINCLUDES) -c synfireGrowthCuda.cu

# synfireGrowth.o: $(SRC_PATH)neuron.h synfireGrowth.cpp
# 	 $(CXX) $(CXXINCLUDES) $(CXXFLAGS) -c synfireGrowth.cpp

# microtime.o: microtime.h microtime.cpp
# 	$(CXX) $(CXXINCLUDES) $(CXXFLAGS) -c microtime.cpp

# synapses.o: $(SRC_PATH)synapses.h $(SRC_PATH)synapses.cpp $(SRC_PATH)neuron.h
# 	$(CXX) $(CXXINCLUDES) $(CXXFLAGS) -c  $(SRC_PATH)synapses.cpp

# neuron.o: $(SRC_PATH)neuron.h $(SRC_PATH)neuron.cpp
# 	$(CXX) $(CXXINCLUDES) $(CXXFLAGS) -c $(SRC_PATH)neuron.cpp

# clean:
# 	rm -rf *.o *~

# realclean:
# 	rm -rf *.o *~ $(EXECUTABLE)

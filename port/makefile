# CXX := g++44
# CXX := g++
# CXXFLAGS := -std=c++0x
# CXXFLAGS := -W -Wall -Wimplicit -Wswitch -Wformat -Wchar-subscripts -Wparentheses -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type -Wno-unused-function -fno-strict-aliasing
# CXXINCLUDES := -I./src/port/ -I./src/common/ -I./src/gpu/ -I./src/synfire/

# NVCC := /usr/local/cuda/bin/nvcc
# NVCCFLAGS := -O3 -arch sm_30
# NVCCINCLUDES := -I. -I/usr/local/cuda/include  -I/usr/local/cuda/NVIDIA_GPU_Computing_SDK/shared/inc -I../src/port/
# NVCCLIBS := -L/usr/local/cuda/lib64  -lGL -lcudart

##################################################################
### PROJECT SETTINGS
# The name of the executable to be created
BIN_NAME := synfire

# Extension of source files used in the project
SRC_EXT = cpp

# Path to the source directory, relative to the makefile
SRC_PATH = .

# Destination directory, like a jail or mounted system
DESTDIR = .

# General compiler flags
# COMPILE_FLAGS = -std=c++11 -Wall -Wextra -g
CXXFLAGS = -std=c++0x

# Find all source files in the source directory, sorted by most
# recently modified
CPP_SOURCES := $(shell find . -path ./src/gpu -prune -o -name '*.cpp' -printf '%T@\t%p\n' | sort -k 1nr | cut -f2-)

# Add additional include paths
INCLUDES := $(shell find -name '*.h' -printf '-I%h\n' | sort -u)

# Set the object file names, with the source directory stripped
# from the path, and the build path prepended in its place
OBJECTS = $(CPP_SOURCES:$(SRC_PATH)/%.$(SRC_EXT)=%.o)

# Set the dependency files that will be used to add header dependencies
DEPS = $(OBJECTS:.o=.d)

#### HELPERS
# Function used to check variables. Use on the command line:
# make print-VARNAME
# Useful for debugging and adding features
print-%: ; @echo $*=$($*)

#### COMPILATION RULES
$(BIN_NAME): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

# Add dependency files, if they exist
-include $(DEPS)

%.o: $(SRC_PATH)/%.$(SRC_EXT)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MP -MMD -c $< -o $@

# Removes all build files
.PHONY: clean
clean:
	rm -r *.o *~
	# TEMP solution to cleaning up object files.
	find . -type f -name '*.o' -exec rm {} +
	find . -type f -name '*.d' -exec rm {} +

realclean: clean
	rm $(BIN_NAME)

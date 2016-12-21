PROGRAM_NAME := Solver

program_CXX_SRCS := $(wildcard *.cpp)
#program_CXX_SRCS += $(wildcard ../../*.cpp) 
program_CXX_OBJS := ${program_CXX_SRCS:.cpp=.o}

program_CU_SRCS := $(wildcard *.cu)
#program_CU_SRCS += $(wildcard ../../*.cu)
# program_CU_HEADERS := $(wildcard *.hpp)
#program_CU_HEADERS += $(wildcard ../../*.cuh)
program_CU_OBJS := ${program_CU_SRCS:.cu=.o}

OBJECTS := $(program_CU_OBJS) $(program_CXX_OBJS)

###################################################################################

CUDA_PATH := /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
CXX := g++5

CUDA_INCLUDE_DIR := -I $(CUDA_PATH)/samples/common/inc
# CPP_INCLUDE_DIR = 

#compile option
CXXFLAGS = -Wall -std=c++11 -fopenmp -O2 -pedantic
CUDAFLAGS = -arch=sm_35 -O2 -std=c++11 -lcusparse

#linker option
CUDALDFLAGS = -lcusparse -lgomp -O2 -arch=sm_35 -Xcompiler "-Wall -std=c++11 -fopenmp -O2"

####################################################################################
.PHONY: all clean debug

all:$(PROGRAM_NAME)

debug: CXXFLAGS = -ggdb3 -O0 -std=c++11 -Wall -pedantic -pg -fopenmp
debug: CUDAFLAGS = -g -G -arch=sm_35 -O0 -std=c++11 -lcusparse
debug: CUDALDFLAGS = -lcusparse -lgomp -O0 -g -G -arch=sm_35 -Xcompiler "-ggdb3  -O0 -std=c++11 -Wall -pedantic -pg -fopenmp"
debug: $(PROGRAM_NAME)

$(PROGRAM_NAME): $(OBJECTS)
	$(NVCC) $(CUDALDFLAGS) $(OBJECTS) -o $(PROGRAM_NAME)

%.o:%.cu
	$(NVCC) $(CUDAFLAGS) $(CUDA_INCLUDE_DIR) -c -o $@ $<

%.o:%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	$(RM) $(PROGRAM_NAME) $(OBJECTS) *~ 

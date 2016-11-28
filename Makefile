all:
	g++5 -c main.cpp -Wall -fopenmp -O3 -pedantic -std=c++11
	nvcc -c -I/usr/local/cuda/samples/common/inc cudaFunction.cu
	nvcc cudaFunction.o main.o -I/usr/local/cuda/samples/common/inc -Xcompiler "-Wall -fopenmp -O3 -pedantic -std=c++11"

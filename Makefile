all:
	g++5 -c main.cpp cg.cpp cr.cpp gcr.cpp bicg.cpp kskipcg.cpp kskipbicg.cpp gmres.cpp vpcg.cpp vpcr.cpp vpgcr.cpp vpgmres.cpp blas.cpp innerMethods.cpp outerMethods.cpp solver_collection.cpp -Wall -std=c++11 -fopenmp -O2
	nvcc -c -I /usr/local/cuda/samples/common/inc -arch=sm_35 cudaFunction.cu -O -use_fast_math -std=c++11 
	nvcc cudaFunction.o main.o cg.o cr.o gcr.o bicg.o kskipcg.o kskipbicg.o gmres.o vpcg.o vpcr.o vpgcr.o vpgmres.o blas.o innerMethods.o outerMethods.o solver_collection.o -Xcompiler "-Wall -std=c++11 -fopenmp -O2" -arch=sm_35 -O -use_fast_math

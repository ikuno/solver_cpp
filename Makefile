all:
	# g++5 -c main.cpp -Wall -fopenmp -O3 -pedantic -std=c++11
	# nvcc -c -I/usr/local/cuda/samples/common/inc cudaFunction.cu
	# nvcc cudaFunction.o main.o -I/usr/local/cuda/samples/common/inc -Xcompiler "-Wall -fopenmp -O3 -pedantic -std=c++11"
	g++5 main.cpp cg.cpp cr.cpp gcr.cpp bicg.cpp kskipcg.cpp kskipbicg.cpp gmres.cpp vpcg.cpp vpcr.cpp vpgcr.cpp vpgmres.cpp blas.cpp innerMethods.cpp outerMethods.cpp solver_collection.cpp -Wall -std=c++11 -fopenmp

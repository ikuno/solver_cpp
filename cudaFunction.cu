#include "cudaFunction.hpp"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>


__global__ void d_kernel_add(double *in1, double *in2, double *out, int size){
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if(id < size){
    out[id] = in1[id] + in2[id];
  }
}

cuda::cuda(){
}

cuda::cuda(int t, int b){
  this->SetCores(t, b);
}

void cuda::SetCores(int t, int b){
  this->ThreadsinBlock = t;
  this->BlocksinGrid = b;
}

void cuda::d_Free(double *ptr){
  checkCudaErrors(cudaFree(ptr));
}

void cuda::d_FreeHost(double *ptr){
  checkCudaErrors(cudaFreeHost(ptr));
}

void cuda::i_Free(int *ptr){
  checkCudaErrors(cudaFree(ptr));
}

void cuda::i_FreeHost(int *ptr){
  checkCudaErrors(cudaFreeHost(ptr));
}

void cuda::d_H2D(double *from, double *to, int size){
  checkCudaErrors(cudaMemcpy(to, from, sizeof(double)*size, cudaMemcpyHostToDevice));
}

void cuda::d_D2H(double *from, double *to, int size){
  checkCudaErrors(cudaMemcpy(to, from, sizeof(double)*size, cudaMemcpyDeviceToHost));
}

double* cuda::d_Malloc(int size){
  double *ptr = NULL;
  const int s = sizeof(double) * size;
  cudaMalloc((void**)&ptr, s);
  return ptr;
}

double* cuda::d_MallocHost(int size){
  double *ptr = NULL;
  const int s = sizeof(double) * size;
  cudaMallocHost((void**)&ptr, s);
  return ptr;
}

int* cuda::i_Malloc(int size){
  int *ptr = NULL;
  const int s = sizeof(int) * size;
  cudaMalloc((void**)&ptr, s);
  return ptr;
}

int* cuda::i_MallocHost(int size){
  int *ptr = NULL;
  const int s = sizeof(int) * size;
  cudaMallocHost((void**)&ptr, s);
  return ptr;
}


void cuda::d_add(double *in1, double *in2, double *out, int size){
  double *D_in1=NULL, *D_in2=NULL, *D_out=NULL;

  D_in1 = d_Malloc(size);
  D_in2 = d_Malloc(size);
  D_out = d_Malloc(size);

  d_H2D(in1, D_in1, size);
  d_H2D(in2, D_in2, size);

  cudaMemset(D_out, 1, sizeof(double)*size);

  d_kernel_add<<<this->BlocksinGrid, this->ThreadsinBlock>>>(D_in1, D_in2, D_out, size);

  d_D2H(D_out, out, size);

  for(int i=0; i<10; i++){
    std::cout << out[i] << std::endl;
  } 

  d_Free(D_in1);
  d_Free(D_in2);
  d_Free(D_out);
}

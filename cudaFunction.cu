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

__global__ void f_kernel_add(float *in1, float *in2, float *out, int size){
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if(id < size){
    out[id] = in1[id] + in2[id];
  }
}

__global__ void d_kernel_MtxVec_mult(int n, double *val, int *col, int *ptr, double *b, double *c){

  extern __shared__ double vals[];

  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_id = thread_id/32;
  int lane = thread_id & (32 - 1);

  int row = warp_id;
  if(row<n)
  {
    int row_start = ptr[row];
    int row_end = ptr[row+1];

    vals[threadIdx.x] = 0.0;

    for(int jj = row_start+lane; jj<row_end; jj+=32)
    { 
      vals[threadIdx.x]+=val[jj] * b[col[jj]];
    }

    if(lane <16)
      vals[threadIdx.x] += vals[threadIdx.x +16];
    if(lane<8)
      vals[threadIdx.x] += vals[threadIdx.x + 8];
    if(lane<4)
      vals[threadIdx.x] += vals[threadIdx.x + 4];
    if(lane<2)
      vals[threadIdx.x] += vals[threadIdx.x + 2];
    if(lane<1)
      vals[threadIdx.x] += vals[threadIdx.x + 1];

    if(lane == 0){
      c[row] += vals[threadIdx.x];
    }
  }
}

__global__ void f_kernel_MtxVec_mult(int n, float *val, int *col, int *ptr, float *b, float *c){

  extern __shared__ float vals[];

  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_id = thread_id/32;
  int lane = thread_id & (32 - 1);

  int row = warp_id;
  if(row<n)
  {
    int row_start = ptr[row];
    int row_end = ptr[row+1];

    vals[threadIdx.x] = 0.0;

    for(int jj = row_start+lane; jj<row_end; jj+=32)
    { 
      vals[threadIdx.x]+=val[jj] * b[col[jj]];
    }

    if(lane <16)
      vals[threadIdx.x] += vals[threadIdx.x +16];
    if(lane<8)
      vals[threadIdx.x] += vals[threadIdx.x + 8];
    if(lane<4)
      vals[threadIdx.x] += vals[threadIdx.x + 4];
    if(lane<2)
      vals[threadIdx.x] += vals[threadIdx.x + 2];
    if(lane<1)
      vals[threadIdx.x] += vals[threadIdx.x + 1];

    if(lane == 0){
      c[row] += vals[threadIdx.x];
    }
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

void cuda::f_Free(float *ptr){
  checkCudaErrors(cudaFree(ptr));
}

void cuda::f_FreeHost(float *ptr){
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

void cuda::f_H2D(float *from, float *to, int size){
  checkCudaErrors(cudaMemcpy(to, from, sizeof(float)*size, cudaMemcpyHostToDevice));
}

void cuda::f_D2H(float *from, float *to, int size){
  checkCudaErrors(cudaMemcpy(to, from, sizeof(float)*size, cudaMemcpyDeviceToHost));

void cuda::i_H2D(int *from, int *to, int size){
  checkCudaErrors(cudaMemcpy(to, from, sizeof(int)*size, cudaMemcpyHostToDevice));
}

void cuda::i_D2H(int *from, int *to, int size){
  checkCudaErrors(cudaMemcpy(to, from, sizeof(int)*size, cudaMemcpyDeviceToHost));
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

float* cuda::f_Malloc(int size){
  float *ptr = NULL;
  const int s = sizeof(float) * size;
  cudaMalloc((void**)&ptr, s);
  return ptr;
}

float* cuda::f_MallocHost(int size){
  float *ptr = NULL;
  const int s = sizeof(float) * size;
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

void cuda::f_add(float *in1, float *in2, float *out, int size){
  float *D_in1=NULL, *D_in2=NULL, *D_out=NULL;

  D_in1 = f_Malloc(size);
  D_in2 = f_Malloc(size);
  D_out = f_Malloc(size);

  f_H2D(in1, D_in1, size);
  f_H2D(in2, D_in2, size);

  cudaMemset(D_out, 1, sizeof(float)*size);

  f_kernel_add<<<this->BlocksinGrid, this->ThreadsinBlock>>>(D_in1, D_in2, D_out, size);

  f_D2H(D_out, out, size);

  for(int i=0; i<10; i++){
    std::cout << out[i] << std::endl;
  } 

  f_Free(D_in1);
  f_Free(D_in2);
  f_Free(D_out);
}

void d_MV(double *in, double *out, int size, double *val, int *col, int *ptr){
  double *D_in = NULL, *D_out = NULL;

  int ThreadPerBlock=128;
  int BlockPerGrid=(size-1)/(ThreadPerBlock/32)+1;

  D_in = d_Malloc(size);
  D_out = d_Malloc(size);

  d_H2D(in, D_in, size);

  d_kernel_MtxVec_mult<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock+16)>>>(size, val, col, ptr, D_in, D_out);

  d_D2H(D_out, out, size);

  d_Free(D_in);
  d_Free(D_out);
}

void f_MV(float *in, float *out, int size, float *val, int *col, int *ptr){
  float *D_in = NULL, *D_out = NULL;

  int ThreadPerBlock=128;
  int BlockPerGrid=(size-1)/(ThreadPerBlock/32)+1;

  D_in = d_Malloc(size);
  D_out = d_Malloc(size);

  f_H2D(in, D_in, size);

  f_kernel_MtxVec_mult<<<BlockPerGrid, ThreadPerBlock, sizeof(float)*(ThreadPerBlock+16)>>>(size, val, col, ptr, D_in, D_out);

  f_D2H(D_out, out, size);

  f_Free(D_in);
  f_Free(D_out);
}

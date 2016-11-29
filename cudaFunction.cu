#include "cudaFunction.hpp"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>

__global__ void kernel_dot (int N, double *a, double *b, double *c)
{
  /* __shared__ double cache[128]; */
  extern __shared__ double cache[];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;

  double temp = 0;
  while (tid < N) {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }

  // Set Cache
  cache[cacheIndex] = temp;
  // synchronize threads in the block
  __syncthreads ();

  // This code, threadsPerBlock should be power of 2
  int i = blockDim.x / 2;
  while (i != 0) {
    if (cacheIndex < i) {
      cache[cacheIndex] += cache[cacheIndex+i];
    }
    __syncthreads();
    i /= 2;
  }

  if (cacheIndex == 0) {
    c[blockIdx.x] = cache[0];
  }
}

__global__ void kernel_add(double *in1, double *in2, double *out, int size){
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if(id < size){
    out[id] = in1[id] + in2[id];
  }
}

__global__ void kernel_MtxVec_mult(int n, double *val, int *col, int *ptr, double *b, double *c){
  long row=blockDim.x * blockIdx.x + threadIdx.x;
  long int i;
  if(row<n){
    double tmp=0.0;
    long int row_start=ptr[row];
    long int row_end=ptr[row+1];
    for(i=row_start;i<row_end;i++){
      tmp+=val[i]*b[col[i]];
    }
    c[row]=tmp;
  }
}

__global__ void kernel_MtxVec_mult_2(int n, double *val, int *col, int *ptr, double *b, double *c){
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

cuda::cuda(){
}

cuda::cuda(int t, int b){
  this->SetCores(t, b);
}

void cuda::SetCores(int t, int b){
  this->ThreadsinBlock = t;
  this->BlocksinGrid = b;
}

void cuda::Free(double *ptr){
  checkCudaErrors(cudaFree(ptr));
}

void cuda::FreeHost(double *ptr){
  checkCudaErrors(cudaFreeHost(ptr));
}

void cuda::Free(int *ptr){
  checkCudaErrors(cudaFree(ptr));
}

void cuda::FreeHost(int *ptr){
  checkCudaErrors(cudaFreeHost(ptr));
}

void cuda::H2D(double *from, double *to, int size){
  checkCudaErrors(cudaMemcpy(to, from, sizeof(double)*size, cudaMemcpyHostToDevice));
}

void cuda::D2H(double *from, double *to, int size){
  checkCudaErrors(cudaMemcpy(to, from, sizeof(double)*size, cudaMemcpyDeviceToHost));
}

void cuda::H2D(int *from, int *to, int size){
  checkCudaErrors(cudaMemcpy(to, from, sizeof(int)*size, cudaMemcpyHostToDevice));
}

void cuda::D2H(int *from, int *to, int size){
  checkCudaErrors(cudaMemcpy(to, from, sizeof(int)*size, cudaMemcpyDeviceToHost));
}

double* cuda::d_Malloc(int size){
  double *ptr = NULL;
  const int s = sizeof(double) * size;
  checkCudaErrors(cudaMalloc((void**)&ptr, s));
  return ptr;
}

double* cuda::d_MallocHost(int size){
  double *ptr = NULL;
  const int s = sizeof(double) * size;
  checkCudaErrors(cudaMallocHost((void**)&ptr, s));
  return ptr;
}

int* cuda::i_Malloc(int size){
  int *ptr = NULL;
  const int s = sizeof(int) * size;
  checkCudaErrors(cudaMalloc((void**)&ptr, s));
  return ptr;
}

int* cuda::i_MallocHost(int size){
  int *ptr = NULL;
  const int s = sizeof(int) * size;
  checkCudaErrors(cudaMallocHost((void**)&ptr, s));
  return ptr;
}

void cuda::Memset(double *ptr, double val, int size){
  checkCudaErrors(cudaMemset(ptr, val, sizeof(double)*size));
}

void cuda::Memset(int *ptr, int val, int size){
  checkCudaErrors(cudaMemset(ptr, val, sizeof(int)*size));
}

void cuda::Reset(){
  checkCudaErrors(cudaDeviceReset());
}

void cuda::add(double *in1, double *in2, double *out, int size){
  double *D_in1=NULL, *D_in2=NULL, *D_out=NULL;

  D_in1 = d_Malloc(size);
  D_in2 = d_Malloc(size);
  D_out = d_Malloc(size);

  H2D(in1, D_in1, size);
  H2D(in2, D_in2, size);

  cudaMemset(D_out, 1, sizeof(double)*size);

  /* kernel_add<<<, this->ThreadsinBlock>>>(D_in1, D_in2, D_out, size); */

  D2H(D_out, out, size);

  for(int i=0; i<10; i++){
    std::cout << out[i] << std::endl;
  } 

  Free(D_in1);
  Free(D_in2);
  Free(D_out);
}

void cuda::MtxVec_mult(double *in, double *out, int size, double *val, int *col, int *ptr){
  double *D_in = NULL, *D_out = NULL;

  int ThreadPerBlock=128;
  int BlockPerGrid=(size-1)/(ThreadPerBlock/32)+1;

  D_in = d_Malloc(size);
  D_out = d_Malloc(size);

  H2D(in, D_in, size);
  Memset(D_out, 0, size);

  if(ThreadPerBlock*8 >= 49152){
    std::cout << "Request shared memory size is over max shared memory size in per block !!! Max = 49152 !!! Request = " << ThreadPerBlock*8 << std::endl;
  }
  /* kernel_MtxVec_mult<<<BlockPerGrid, ThreadPerBlock>>>(size, val, col, ptr, D_in, D_out); */
  ///
  kernel_MtxVec_mult_2<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock+16)>>>(size, val, col, ptr, D_in, D_out);
  checkCudaErrors( cudaPeekAtLastError() );

  D2H(D_out, out, size);

  Free(D_in);
  Free(D_out);
}

double cuda::dot(double *in1, double *in2, int size){
  double *D_in1=NULL, *D_in2=NULL;
  double *H_out=NULL, *D_out=NULL, sum=0.0;

  int ThreadPerBlock=128;
  int BlockPerGrid=ceil((double)size/(double)ThreadPerBlock);

  D_in1 = d_Malloc(size);
  D_in2 = d_Malloc(size);
  D_out = d_Malloc(BlockPerGrid);

  H_out = new double [BlockPerGrid];
  
  H2D(in1, D_in1, size);
  H2D(in2, D_in2, size);
 
  if(ThreadPerBlock*8 >= 49152){
    std::cout << "Request shared memory size is over max shared memory size in per block !!! Max = 49152 !!! Request = " << ThreadPerBlock*8 << std::endl;
  }

  kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock)>>>(size, D_in1, D_in2, D_out);
  checkCudaErrors( cudaPeekAtLastError() );

  D2H(D_out, H_out, BlockPerGrid);

#pragma omp parallel for schedule(static) reduction(+:sum)
  for(int i=0; i<BlockPerGrid; i++){
    sum += H_out[i];
  }
  

  delete[] H_out;
  Free(D_in1);
  Free(D_in2);
  Free(D_out);

  return sum;
}


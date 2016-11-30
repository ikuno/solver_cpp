#include "cudaFunction.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "cusparse.h"

#include "color.hpp"

__device__ __inline__ double shfl_xor(double value, int const lane)
{
  return __hiloint2double(__shfl_xor(__double2hiint(value), lane),
      __shfl_xor(__double2loint(value), lane)); 
}
__global__ void kernel_dot (const int N, const double *__restrict__ a, const double *__restrict__ b, double *c)
{
  extern __shared__ double cache[];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;

  double temp = 0;
  while (tid < N) {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }

  cache[cacheIndex] = temp;
  __syncthreads ();

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

__global__ void kernel_MtxVec_mult_2(int n, const double *val, const int *col, const int *ptr, const double *__restrict__ b, double *c){
  extern __shared__ volatile double vals[];

  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_id = thread_id/32;
  int lane = thread_id & (32 - 1);

  int row = warp_id;
  if(row<n)
  {
    int row_start = ptr[row];
    int row_end = ptr[row+1];

    double sum = 0.0;
    for(int jj = row_start+lane; jj<row_end; jj+=32)
    { 
      sum += val[jj] * b[col[jj]];
    }

    vals[threadIdx.x] = sum;
    vals[threadIdx.x] = sum = sum + vals[threadIdx.x + 16];
    vals[threadIdx.x] = sum = sum + vals[threadIdx.x + 8];
    vals[threadIdx.x] = sum = sum + vals[threadIdx.x + 4];
    vals[threadIdx.x] = sum = sum + vals[threadIdx.x + 2];
    sum = sum + vals[threadIdx.x+1];

    if(lane == 0){
      c[row] = sum;
    }
  }
}

__global__ void kernel_MtxVec_mult_3(int n, const double *val, const int *col, const int *ptr, const double *__restrict__ b, double *c){

  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_id = thread_id/32;
  int lane = thread_id & (32 - 1);

  int row = warp_id;
  if(row<n)
  {
    int row_start = ptr[row];
    int row_end = ptr[row+1];

    double sum = 0.0;
    for(int jj = row_start+lane; jj<row_end; jj+=32)
    { 
      sum += val[jj] * b[col[jj]];
    }

    sum += shfl_xor(sum, 16);
    sum += shfl_xor(sum, 8);
    sum += shfl_xor(sum, 4);
    sum += shfl_xor(sum, 2);
    sum += shfl_xor(sum, 1);


    if(lane == 0){
      c[row] = sum;
    }
  }
}

cuda::cuda(){
  time = new times();

  this->cu_d1 = NULL;
  this->cu_d2 = NULL;
  this->cu_d3 = NULL;
  this->cu_h1 = NULL;

  this->dot_copy_time = 0.0;
  this->dot_proc_time = 0.0;
  this->dot_malloc_time = 0.0;
  this->dot_reduce_time = 0.0;

  this->MV_copy_time = 0.0;
  this->MV_proc_time = 0.0;
  this->MV_malloc_time = 0.0;

  this->All_malloc_time = 0.0;


}

cuda::cuda(int size){
  this->size = size;
  time = new times();


  this->dot_copy_time = 0.0;
  this->dot_proc_time = 0.0;
  this->dot_malloc_time = 0.0;
  this->dot_reduce_time = 0.0;

  this->MV_copy_time = 0.0;
  this->MV_proc_time = 0.0;
  this->MV_malloc_time = 0.0;

  this->All_malloc_time = 0.0;

  this->time->start();
  int tmp = ceil((double)this->size/(double)128);
  this->cu_d1 = d_Malloc(this->size);
  this->cu_d2 = d_Malloc(this->size);
  this->cu_d3 = d_Malloc(tmp);
  this->cu_h1 = new double [tmp];
  this->time->end();
  this->All_malloc_time += this->time->getTime();
}

cuda::~cuda(){

  Free(cu_d1);
  Free(cu_d2);
  Free(cu_d3);
  delete[] cu_h1;

  delete this->time;
}


void cuda::Free(void* ptr){
  checkCudaErrors(cudaFree(ptr));
}

void cuda::FreeHost(void* ptr){
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

void cuda::MtxVec_mult(double *in, double *out, int size, double *val, int *col, int *ptr){
  double *D_in = NULL, *D_out = NULL;

  int ThreadPerBlock=128;
  int BlockPerGrid=(size-1)/(ThreadPerBlock/32)+1;

  this->time->start();
  D_in = d_Malloc(size);
  D_out = d_Malloc(size);
  Memset(D_out, 0, size);
  this->time->end();
  this->MV_malloc_time += this->time->getTime();

  this->time->start();
  H2D(in, D_in, size);
  this->time->end();
  this->MV_copy_time += this->time->getTime();

  if(ThreadPerBlock*8 >= 49152){
    std::cout << "Request shared memory size is over max shared memory size in per block !!! Max = 49152 !!! Request = " << ThreadPerBlock*8 << std::endl;
  }
  /* kernel_MtxVec_mult<<<BlockPerGrid, ThreadPerBlock>>>(size, val, col, ptr, D_in, D_out); */
  ///
  this->time->start();
  kernel_MtxVec_mult_2<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock+16)>>>(size, val, col, ptr, D_in, D_out);
  checkCudaErrors( cudaPeekAtLastError() );
  this->time->end();
  this->MV_proc_time += this->time->getTime();

  this->time->start();
  D2H(D_out, out, size);
  this->time->end();
  this->MV_copy_time += this->time->getTime();

  this->time->start();
  Free(D_in);
  Free(D_out);
  this->time->end();
  this->MV_malloc_time += this->time->getTime();
}

void cuda::MtxVec_mult(double *in, double *out, double *val, int *col, int *ptr){

  int ThreadPerBlock=128;
  int BlockPerGrid=(size-1)/(ThreadPerBlock/32)+1;

  this->time->start();
  Memset(this->cu_d2, 0, size);
  this->time->end();
  this->All_malloc_time += this->time->getTime();

  //d1 -> in
  //d2 -> out
  this->time->start();
  H2D(in, this->cu_d1, size);
  this->time->end();
  this->MV_copy_time += this->time->getTime();

  if(ThreadPerBlock*8 >= 49152){
    std::cout << "Request shared memory size is over max shared memory size in per block !!! Max = 49152 !!! Request = " << ThreadPerBlock*8 << std::endl;
  }
  /* kernel_MtxVec_mult<<<BlockPerGrid, ThreadPerBlock>>>(size, val, col, ptr, D_in, D_out); */
  /* kernel_MtxVec_mult_2<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock+16)>>>(this->size, val, col, ptr, cu_d1, cu_d2); */
  ///
  this->time->start();
  kernel_MtxVec_mult_3<<<BlockPerGrid, ThreadPerBlock>>>(this->size, val, col, ptr, cu_d1, cu_d2);
  checkCudaErrors( cudaPeekAtLastError() );
  this->time->end();
  this->MV_proc_time += this->time->getTime();

  this->time->start();
  D2H(cu_d2, out, size);
  this->time->end();
  this->MV_copy_time += this->time->getTime();
}

double cuda::dot(double *in1, double *in2, int size){
  double *D_in1=NULL, *D_in2=NULL;
  double *H_out=NULL, *D_out=NULL, sum=0.0;

  int ThreadPerBlock=128;
  int BlockPerGrid=ceil((double)size/(double)ThreadPerBlock);

  this->time->start();
  D_in1 = d_Malloc(size);
  D_in2 = d_Malloc(size);
  D_out = d_Malloc(BlockPerGrid);
  H_out = new double [BlockPerGrid];
  this->time->end();
  this->dot_malloc_time += this->time->getTime();

  
  this->time->start();
  H2D(in1, D_in1, size);
  H2D(in2, D_in2, size);
  this->time->end();
  this->dot_copy_time += this->time->getTime();
 
  if(ThreadPerBlock*8 >= 49152){
    std::cout << "Request shared memory size is over max shared memory size in per block !!! Max = 49152 !!! Request = " << ThreadPerBlock*8 << std::endl;
  }

  this->time->start();
  kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock)>>>(size, D_in1, D_in2, D_out);
  checkCudaErrors( cudaPeekAtLastError() );
  this->time->end();
  this->dot_proc_time += this->time->getTime();

  this->time->start();
  D2H(D_out, H_out, BlockPerGrid);
  this->time->end();
  this->dot_copy_time += this->time->getTime();

  this->time->start();
#pragma omp parallel for schedule(static) reduction(+:sum)
  for(int i=0; i<BlockPerGrid; i++){
    sum += H_out[i];
  }
  this->time->end();
  this->dot_reduce_time += this->time->getTime();
  

  this->time->start();
  delete[] H_out;
  Free(D_in1);
  Free(D_in2);
  Free(D_out);
  this->time->end();
  this->dot_malloc_time += this->time->getTime();

  return sum;
}

double cuda::dot(double *in1, double *in2){
  double sum=0.0;


  int ThreadPerBlock=128;
  int BlockPerGrid=ceil((double)size/(double)ThreadPerBlock);


  this->time->start();
  Memset(this->cu_d3, 0, BlockPerGrid);
  this->time->end();
  this->All_malloc_time += this->time->getTime();

  //d_1 -> in1
  //d_2 -> in2
  //d_3 -> out
  this->time->start();
  H2D(in1, this->cu_d1, size);
  H2D(in2, this->cu_d2, size);
  this->time->end();
  this->dot_copy_time += this->time->getTime();
 
  if(ThreadPerBlock*8 >= 49152){
    std::cout << "Request shared memory size is over max shared memory size in per block !!! Max = 49152 !!! Request = " << ThreadPerBlock*8 << std::endl;
  }

  this->time->start();
  kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock)>>>(this->size, cu_d1, cu_d2, cu_d3);
  checkCudaErrors( cudaPeekAtLastError() );
  this->time->end();
  this->dot_proc_time += this->time->getTime();

  //d_3 -> out
  //h_1 -> out(host)
  this->time->start();
  D2H(cu_d3, cu_h1, BlockPerGrid);
  this->time->end();
  this->dot_copy_time += this->time->getTime();

  this->time->start();
#pragma omp parallel for schedule(static) reduction(+:sum)
  for(int i=0; i<BlockPerGrid; i++){
    sum += cu_h1[i];
  }
  this->time->end();
  this->dot_reduce_time += this->time->getTime();

  return sum;
}

void cuda::CSR2CSC(double *dCSRval, int *dCSRcol, int *dCSRptr, double *CSCval, int *CSCrow, int *CSCptr, double *dCSCval, int *dCSCrow, int *dCSCptr, int N, int NNZ){
  cusparseHandle_t handle=0;
  cusparseCreate(&handle);

  std::cout << "Transpose Matrix in CUDA.........." << std::flush;
  cusparseStatus_t status = cusparseDcsr2csc(handle, N, N, NNZ, dCSRval, dCSRptr, dCSRcol, dCSCval, dCSCrow, dCSCptr, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
  std::cout << GREEN << "[○] Done" << RESET << std::endl;

  if(status != CUSPARSE_STATUS_SUCCESS){
    std::cout << "error in cusparse" << std::endl;
    exit(-1);
  }

  cudaMemcpy(CSCval, dCSCval, sizeof(double)*NNZ, cudaMemcpyDeviceToHost);
  cudaMemcpy(CSCrow, dCSCrow, sizeof(int)*NNZ, cudaMemcpyDeviceToHost);
  cudaMemcpy(CSCptr, dCSCptr, sizeof(int)*(N+1), cudaMemcpyDeviceToHost);

}

void cuda::CSR2CSC(double *CSRval, int *CSRcol, int *CSRptr, double *CSCval, int *CSCrow, int *CSCptr, int N, int NNZ){

  double *dCSRval;
  int *dCSRcol, *dCSRptr;
  double *dCSCval;
  int *dCSCrow, *dCSCptr;
  cusparseHandle_t handle=0;
  cusparseCreate(&handle);


  cudaMalloc((void**)&dCSRval, sizeof(double)*NNZ);
  cudaMalloc((void**)&dCSRcol, sizeof(int)*NNZ);
  cudaMalloc((void**)&dCSRptr, sizeof(int)*(N+1));

  cudaMalloc((void**)&dCSCval, sizeof(double)*NNZ);
  cudaMalloc((void**)&dCSCrow, sizeof(int)*NNZ);
  cudaMalloc((void**)&dCSCptr, sizeof(int)*(N+1));

  cudaMemcpy(dCSRval, CSRval, sizeof(double)*NNZ, cudaMemcpyHostToDevice);
  cudaMemcpy(dCSRcol, CSRcol, sizeof(int)*NNZ, cudaMemcpyHostToDevice );
  cudaMemcpy(dCSRptr, CSRptr, sizeof(int)*(N+1),  cudaMemcpyHostToDevice);

  memset(CSCval, 0, sizeof(double)*NNZ);
  memset(CSCrow, 0, sizeof(int)*NNZ);
  memset(CSCptr, 0, sizeof(int)*(N+1));

  cudaMemset(dCSCval, 0, sizeof(double)*NNZ);
  cudaMemset(dCSCrow, 0, sizeof(int)*NNZ);
  cudaMemset(dCSCptr, 0, sizeof(int)*(N+1));

  std::cout << "Transpose Matrix in CUDA.........."<< std::flush;
  cusparseStatus_t status = cusparseDcsr2csc(handle, N, N, NNZ, dCSRval, dCSRptr, dCSRcol, dCSCval, dCSCrow, dCSCptr, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
  std::cout << GREEN << "[○] Done" << RESET << std::endl;

  if(status != CUSPARSE_STATUS_SUCCESS){
    std::cout << "error in cusparse" << std::endl;
    exit(-1);
  }

  cudaMemcpy(CSCval, dCSCval, sizeof(double)*NNZ, cudaMemcpyDeviceToHost);
  cudaMemcpy(CSCrow, dCSCrow, sizeof(int)*NNZ, cudaMemcpyDeviceToHost);
  cudaMemcpy(CSCptr, dCSCptr, sizeof(int)*(N+1), cudaMemcpyDeviceToHost);

  cudaFree(dCSRval);
  cudaFree(dCSRcol);
  cudaFree(dCSRptr);

  cudaFree(dCSCval);
  cudaFree(dCSCrow);
  cudaFree(dCSCptr);

}

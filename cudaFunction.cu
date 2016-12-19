#include "cudaFunction.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "cusparse.h"
#include <cuda_profiler_api.h>

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

__global__ void kernel_dot (const int N, const double *__restrict__ a, const int aindex, const int asize, const double *__restrict__ b, double *c, const int cindex, const int csize)
{
  extern __shared__ double cache[];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;

  double temp = 0;
  while (tid < N) {
    temp += a[aindex * asize + tid] * b[tid];
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
    c[cindex * csize + blockIdx.x] = cache[0];
  }
}

/* kernel_MtxVec_mult_old_1<<<BlockPerGrid, ThreadPerBlock>>>(size, val, col, ptr, D_in, D_out); */
__global__ void kernel_MtxVec_mult_old_1(unsigned long int n, double *val, int *col, int *ptr, double *b, double *c){
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

  /* kernel_MtxVec_mult_old_2<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock+16)>>>(this->size, val, col, ptr, cu_d1, cu_d2); */
__global__ void kernel_MtxVec_mult_old_2(unsigned long int n, const double *val, const int *col, const int *ptr, const double *__restrict__ b, double *c){
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

/* kernel_MtxVec_mult<<<BlockPerGrid, ThreadPerBlock>>>(this->size, val, col, ptr, cu_d1, cu_d2); */
__global__ void kernel_MtxVec_mult(unsigned long int n, const double *val, const int *col, const int *ptr, const double *__restrict__ b, double *c){

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

__global__ void kernel_MtxVec_mult(unsigned long int n, const double *val, const int *col, const int *ptr, const double *__restrict__ b, double *c, const int cindex, const int csize){

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
      c[cindex * csize + row] = sum;
    }
  }
}

__global__ void kernel_MtxVec_mult(int n, const double *val, const int *col, const int *ptr, const double *__restrict__ b, const int bindex, const int bsize, double *c, const int cindex, const int csize){

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
      sum += val[jj] * b[bindex * bsize + col[jj]];
    }

    sum += shfl_xor(sum, 16);
    sum += shfl_xor(sum, 8);
    sum += shfl_xor(sum, 4);
    sum += shfl_xor(sum, 2);
    sum += shfl_xor(sum, 1);


    if(lane == 0){
      c[cindex * csize + row] = sum;
    }
  }
}

//----------------------------------------------------------------------------------------------

cuda::cuda(times *t){

  this->time = t;

  this->cu_d1 = NULL;
  this->cu_d2 = NULL;

  this->cu_d3 = NULL;
  this->cu_h1 = NULL;

  this->cu_d4 = NULL;
  this->cu_d5 = NULL;
  this->cu_d6 = NULL;
  this->cu_d7 = NULL;
  this->cu_d8 = NULL;
  
  this->cu_d9 = NULL;

  this->cu_h2 = NULL;
  this->cu_h3 = NULL;
  this->cu_h4 = NULL;

  this->cu_h5 = NULL;

  this->cu_d10 = NULL;
  this->cu_d11 = NULL;
  
  this->cu_h6 = NULL;
  this->cu_h7 = NULL;

  /* this->cu_h100 = NULL; */
  /* this->cu_h101 = NULL; */
  /* this->cu_h200 = NULL; */
  /* this->cu_h201 = NULL; */

}

cuda::cuda(times *t, unsigned long int size) : cuda::cuda(t){
  this->size = size;

  this->time->start();
  int tmp = ceil((double)this->size/(double)128);
  this->cu_d1 = d_Malloc(this->size);
  this->cu_d2 = d_Malloc(this->size);
  this->time->end();
  this->time->cu_dot_malloc_time += this->time->getTime();

  this->time->start();
  this->cu_d3 = d_Malloc(tmp);
  /* this->cu_h1 = new double [tmp]; */
  this->cu_h1 = d_MallocHost(tmp);
  this->time->end();
  this->time->cu_MV_malloc_time += this->time->getTime();

  this->cu_h8 = d_MallocHost(tmp * (1000+1));
  /* this->cu_d100 = d_Malloc(this->size); */
  /* this->cu_d101 = d_Malloc(tmp); */
  /*  */
  /*  */
  /* this->cu_d200 = d_Malloc(this->size * (1000+1)); */
  /* this->cu_d201 = d_Malloc(tmp * (1000+1)); */

}

cuda::cuda(times *t, unsigned long int size, int k) : cuda::cuda(t, size){
  this->k = k;
  int tmp = ceil((double)this->size/(double)128);

  this->time->start();
  this->cu_d4 = d_Malloc(this->size * (2*this->k + 1));
  this->cu_d5 = d_Malloc(this->size * (2*this->k + 2));
  this->time->end();
  this->time->cu_MV_malloc_time += this->time->getTime();

  this->time->start();
  this->cu_d6 = d_Malloc(tmp * (2*this->k));
  this->cu_d7 = d_Malloc(tmp * (2*this->k + 1));
  this->cu_d8 = d_Malloc(tmp * (2*this->k + 2));
 

  this->cu_h2 = d_MallocHost(tmp * (2*this->k));
  this->cu_h3 = d_MallocHost(tmp * (2*this->k+1));
  this->cu_h4 = d_MallocHost(tmp * (2*this->k+2));

  this->cu_d9 = d_Malloc(tmp * (2*this->k + 1));
  this->cu_h5 = d_MallocHost(tmp * (2*this->k+1));
  this->time->end();
  this->time->cu_dot_malloc_time += this->time->getTime();
}

cuda::cuda(times *t, unsigned long int size, double restart) : cuda::cuda(t, size){
  int r = static_cast<int>(restart);
  this->restart = r;
  int tmp = ceil((double)this->size/(double)128);
  this->cu_d10 = d_Malloc(r * size);
  this->cu_d11 = d_Malloc(tmp);
  this->cu_h6 = d_MallocHost(r * size);
  this->cu_h6 = d_MallocHost(tmp);
}

cuda::cuda(times *t, unsigned long int size, int k, double restart) : cuda::cuda(t, size, k){
  int r = static_cast<int>(restart);
  this->restart = r;
  int tmp = ceil((double)this->size/(double)128);
  this->cu_d10 = d_Malloc(r * size);
  this->cu_d11 = d_Malloc(tmp);
  this->cu_h6 = d_MallocHost(r * size);
  this->cu_h6 = d_MallocHost(tmp);
}

cuda::~cuda(){

  Free(cu_d1);
  Free(cu_d2);
  Free(cu_d3);
  Free(cu_d4);
  Free(cu_d5);
  Free(cu_d6);
  Free(cu_d7);
  Free(cu_d8);
  Free(cu_d9);
  Free(cu_d10);
  Free(cu_d11);
  FreeHost(cu_h1);
  FreeHost(cu_h2);
  FreeHost(cu_h3);
  FreeHost(cu_h4);
  FreeHost(cu_h5);
  FreeHost(cu_h6);
  FreeHost(cu_h7);

  /* Free(cu_d100); */
  /* Free(cu_d101); */
  /* Free(cu_d200); */
  /* Free(cu_d201); */
  FreeHost(cu_h8);

}

void cuda::Free(void* ptr){
  checkCudaErrors(cudaFree(ptr));
}

void cuda::FreeHost(void* ptr){
  checkCudaErrors(cudaFreeHost(ptr));
}

void cuda::H2D(double *from, double *to, unsigned long int size){
  checkCudaErrors(cudaMemcpy(to, from, sizeof(double)*size, cudaMemcpyHostToDevice));
}

void cuda::D2H(double *from, double *to, unsigned long int size){
  checkCudaErrors(cudaMemcpy(to, from, sizeof(double)*size, cudaMemcpyDeviceToHost));
}

void cuda::H2D(int *from, int *to, unsigned long int size){
  checkCudaErrors(cudaMemcpy(to, from, sizeof(int)*size, cudaMemcpyHostToDevice));
}

void cuda::D2H(int *from, int *to, unsigned long int size){
  checkCudaErrors(cudaMemcpy(to, from, sizeof(int)*size, cudaMemcpyDeviceToHost));
}

double* cuda::d_Malloc(unsigned long int size){
  double *ptr = NULL;
  unsigned long int s = sizeof(double) * size;
  checkCudaErrors(cudaMalloc((void**)&ptr, s));
  return ptr;
}

double* cuda::d_MallocHost(unsigned long int size){
  double *ptr = NULL;
  unsigned long int  s = sizeof(double) * size;
  checkCudaErrors(cudaMallocHost((void**)&ptr, s));
  /* checkCudaErrors(cudaHostAlloc((void**)&ptr, s, cudaHostAllocPortable)); */
  return ptr;
}


int* cuda::i_Malloc(unsigned long int size){
  int *ptr = NULL;
  unsigned long int s = sizeof(int) * size;
  checkCudaErrors(cudaMalloc((void**)&ptr, s));
  return ptr;
}

int* cuda::i_MallocHost(unsigned long int size){
  int *ptr = NULL;
  unsigned long int s = sizeof(int) * size;
  checkCudaErrors(cudaMallocHost((void**)&ptr, s));
  return ptr;
}

void cuda::Memset(double *ptr, double val, unsigned long int size){
  checkCudaErrors(cudaMemset(ptr, val, sizeof(double)*size));
}

void cuda::Memset(int *ptr, int val, unsigned long int size){
  checkCudaErrors(cudaMemset(ptr, val, sizeof(int)*size));
}

void cuda::Reset(){
  /* checkCudaErrors(cudaDeviceSynchronize()); */
  /* checkCudaErrors(cudaProfilerStop()); */
  checkCudaErrors(cudaDeviceReset());
}

void cuda::MtxVec_mult(double *in, double *out, unsigned long size, double *val, int *col, int *ptr){
  double *D_in = NULL, *D_out = NULL;

  int ThreadPerBlock=128;
  int BlockPerGrid=(size-1)/(ThreadPerBlock/32)+1;

  D_in = d_Malloc(size);
  D_out = d_Malloc(size);
  Memset(D_out, 0, size);

  this->time->start();
  H2D(in, D_in, size);
  this->time->end();
  this->time->cu_MV_copy_time += this->time->getTime();

  if(ThreadPerBlock*8 >= 49152){
    std::cout << "Request shared memory size is over max shared memory size in per block !!! Max = 49152 !!! Request = " << ThreadPerBlock*8 << std::endl;
  }
  
  this->time->start();
  kernel_MtxVec_mult<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock+16)>>>(size, val, col, ptr, D_in, D_out);
  checkCudaErrors( cudaPeekAtLastError() );
  this->time->end();
  this->time->cu_MV_proc_time += this->time->getTime();

  this->time->start();
  D2H(D_out, out, size);
  this->time->end();
  this->time->cu_MV_copy_time += this->time->getTime();

  Free(D_in);
  Free(D_out);
}

void cuda::MtxVec_mult(double *in, double *out, double *val, int *col, int *ptr){

  int ThreadPerBlock=128;
  int BlockPerGrid=(size-1)/(ThreadPerBlock/32)+1;


  Memset(this->cu_d2, 0, size);

  //d1 -> in
  //d2 -> out
  this->time->start();
  H2D(in, this->cu_d1, size);
  this->time->end();
  this->time->cu_MV_copy_time += this->time->getTime();


  if(ThreadPerBlock*8 >= 49152){
    std::cout << "Request shared memory size is over max shared memory size in per block !!! Max = 49152 !!! Request = " << ThreadPerBlock*8 << std::endl;
  }
  this->time->start();
  kernel_MtxVec_mult<<<BlockPerGrid, ThreadPerBlock>>>(this->size, val, col, ptr, cu_d1, cu_d2);
  checkCudaErrors( cudaPeekAtLastError() );
  this->time->end();
  this->time->cu_MV_proc_time += this->time->getTime();

  this->time->start();
  D2H(cu_d2, out, size);
  this->time->end();
  this->time->cu_MV_copy_time += this->time->getTime();
}

void cuda::MtxVec_mult(double *in, unsigned long int inindex, unsigned long int insize, double *out, unsigned long int outindex, unsigned long int outsize, double *val, int *col, int *ptr){
  int ThreadPerBlock=128;
  int BlockPerGrid=(size-1)/(ThreadPerBlock/32)+1;

  Memset(this->cu_d2, 0, size);

  //d1 -> in
  //d2 -> out
  this->time->start();
  H2D((double*)(in+(inindex*insize)), this->cu_d1, size);
  this->time->end();
  this->time->cu_MV_copy_time += this->time->getTime();

  if(ThreadPerBlock*8 >= 49152){
    std::cout << "Request shared memory size is over max shared memory size in per block !!! Max = 49152 !!! Request = " << ThreadPerBlock*8 << std::endl;
  }
  this->time->start();
  kernel_MtxVec_mult<<<BlockPerGrid, ThreadPerBlock>>>(this->size, val, col, ptr, cu_d1, cu_d2);
  checkCudaErrors( cudaPeekAtLastError() );
  this->time->end();
  this->time->cu_MV_proc_time += this->time->getTime();

  this->time->start();
  D2H(this->cu_d2, (double*)(out+(outindex*outsize)), size);
  this->time->end();
  this->time->cu_MV_copy_time += this->time->getTime();
}

void cuda::MtxVec_mult(double *in, unsigned long int inindex, unsigned long int insize, double *out, double *val, int *col, int *ptr){
  int ThreadPerBlock=128;
  int BlockPerGrid=(size-1)/(ThreadPerBlock/32)+1;

  Memset(this->cu_d2, 0, size);

  //d1 -> in
  //d2 -> out
  this->time->start();
  H2D((double*)(in+(inindex*insize)), this->cu_d1, size);
  this->time->end();
  this->time->cu_MV_copy_time += this->time->getTime();

  if(ThreadPerBlock*8 >= 49152){
    std::cout << "Request shared memory size is over max shared memory size in per block !!! Max = 49152 !!! Request = " << ThreadPerBlock*8 << std::endl;
  }
  this->time->start();
  kernel_MtxVec_mult<<<BlockPerGrid, ThreadPerBlock>>>(this->size, val, col, ptr, cu_d1, cu_d2);
  checkCudaErrors( cudaPeekAtLastError() );
  this->time->end();
  this->time->cu_MV_proc_time += this->time->getTime();

  this->time->start();
  D2H(this->cu_d2, out, size);
  this->time->end();
  this->time->cu_MV_copy_time += this->time->getTime();

}

double cuda::dot(double *in1, double *in2, unsigned long int size){
  double *D_in1=NULL, *D_in2=NULL;
  double *H_out=NULL, *D_out=NULL, sum=0.0;

  int ThreadPerBlock=128;
  int BlockPerGrid=ceil((double)size/(double)ThreadPerBlock);

  this->time->start();
  D_in1 = d_Malloc(size);
  D_in2 = d_Malloc(size);
  D_out = d_Malloc(BlockPerGrid);
  H_out = new double [BlockPerGrid];
  this->time->start();
  this->time->cu_dot_malloc_time += this->time->getTime();


  this->time->start();
  H2D(in1, D_in1, size);
  H2D(in2, D_in2, size);
  this->time->start();
  this->time->cu_dot_copy_time += this->time->getTime();


  if(ThreadPerBlock*8 >= 49152){
    std::cout << "Request shared memory size is over max shared memory size in per block !!! Max = 49152 !!! Request = " << ThreadPerBlock*8 << std::endl;
  }

  this->time->start();
  kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock)>>>(size, D_in1, D_in2, D_out);
  checkCudaErrors( cudaPeekAtLastError() );
  this->time->start();
  this->time->cu_dot_proc_time += this->time->getTime();

  this->time->start();
  D2H(D_out, H_out, BlockPerGrid);
  this->time->start();
  this->time->cu_dot_copy_time += this->time->getTime();

  this->time->start();
#pragma omp parallel for schedule(static) reduction(+:sum)
  for(int i=0; i<BlockPerGrid; i++){
    sum += H_out[i];
  }
  this->time->start();
  this->time->cu_dot_reduce_time += this->time->getTime();

  delete[] H_out;
  Free(D_in1);
  Free(D_in2);
  Free(D_out);

  return sum;
}

double cuda::dot(double *in1, double *in2){
  double sum=0.0;

  int ThreadPerBlock=128;
  int BlockPerGrid=ceil((double)size/(double)ThreadPerBlock);

  Memset(this->cu_d3, 0, BlockPerGrid);

  //d_1 -> in1
  //d_2 -> in2
  //d_3 -> out
  this->time->start();
  H2D(in1, this->cu_d1, size);
  H2D(in2, this->cu_d2, size);
  this->time->start();
  this->time->cu_dot_copy_time += this->time->getTime();


  if(ThreadPerBlock*8 >= 49152){
    std::cout << "Request shared memory size is over max shared memory size in per block !!! Max = 49152 !!! Request = " << ThreadPerBlock*8 << std::endl;
  }

  this->time->start();
  kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock)>>>(this->size, cu_d1, cu_d2, cu_d3);
  checkCudaErrors( cudaPeekAtLastError() );
  this->time->start();
  this->time->cu_dot_proc_time += this->time->getTime();


  //d_3 -> out
  //h_1 -> out(host)
  this->time->start();
  D2H(cu_d3, cu_h1, BlockPerGrid);
  this->time->start();
  this->time->cu_dot_copy_time += this->time->getTime();

  this->time->start();
#pragma omp parallel for schedule(static) reduction(+:sum)
  for(int i=0; i<BlockPerGrid; i++){
    sum += cu_h1[i];
  }
  this->time->start();
  this->time->cu_dot_reduce_time += this->time->getTime();

  return sum;
}


double cuda::dot(double *in1, unsigned long int in1index, unsigned long int in1size, double *in2, unsigned long int in2index, unsigned long int in2size){
  double sum=0.0;

  int ThreadPerBlock=128;
  int BlockPerGrid=ceil((double)size/(double)ThreadPerBlock);

  Memset(this->cu_d3, 0, BlockPerGrid);

  //d_1 -> in1
  //d_2 -> in2
  //d_3 -> out
  this->time->start();
  H2D((double*)(in1+(in1index*in1size)), this->cu_d1, size);
  H2D((double*)(in2+(in2index*in2size)), this->cu_d2, size);
  this->time->start();
  this->time->cu_dot_copy_time += this->time->getTime();

  if(ThreadPerBlock*8 >= 49152){
    std::cout << "Request shared memory size is over max shared memory size in per block !!! Max = 49152 !!! Request = " << ThreadPerBlock*8 << std::endl;
  }

  this->time->start();
  kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock)>>>(this->size, cu_d1, cu_d2, cu_d3);
  checkCudaErrors( cudaPeekAtLastError() );
  this->time->start();
  this->time->cu_dot_proc_time += this->time->getTime();


  //d_3 -> out
  //h_1 -> out(host)
  this->time->start();
  D2H(cu_d3, cu_h1, BlockPerGrid);
  this->time->start();
  this->time->cu_dot_copy_time += this->time->getTime();

  this->time->start();
#pragma omp parallel for schedule(static) reduction(+:sum)
  for(int i=0; i<BlockPerGrid; i++){
    sum += cu_h1[i];
  }
  this->time->start();
  this->time->cu_dot_reduce_time += this->time->getTime();

  return sum;

}

double cuda::dot(double *in1, double *in2, unsigned long int in2index, unsigned long int in2size){
  double sum=0.0;

  int ThreadPerBlock=128;
  int BlockPerGrid=ceil((double)size/(double)ThreadPerBlock);

  Memset(this->cu_d3, 0, BlockPerGrid);

  //d_1 -> in1
  //d_2 -> in2
  //d_3 -> out
  this->time->start();
  H2D(in1, this->cu_d1, size);
  H2D((double*)(in2+(in2index*in2size)), this->cu_d2, size);
  this->time->start();
  this->time->cu_dot_copy_time += this->time->getTime();


  if(ThreadPerBlock*8 >= 49152){
    std::cout << "Request shared memory size is over max shared memory size in per block !!! Max = 49152 !!! Request = " << ThreadPerBlock*8 << std::endl;
  }

  this->time->start();
  kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock)>>>(this->size, cu_d1, cu_d2, cu_d3);
  checkCudaErrors( cudaPeekAtLastError() );
  this->time->start();
  this->time->cu_dot_proc_time += this->time->getTime();


  //d_3 -> out
  //h_1 -> out(host)
  this->time->start();
  D2H(cu_d3, cu_h1, BlockPerGrid);
  this->time->start();
  this->time->cu_dot_copy_time += this->time->getTime();

  this->time->start();
#pragma omp parallel for schedule(static) reduction(+:sum)
  for(int i=0; i<BlockPerGrid; i++){
    sum += cu_h1[i];
  }
  this->time->start();
  this->time->cu_dot_reduce_time += this->time->getTime();

  return sum;
}

double cuda::dot(double *in1, unsigned long int in1index, unsigned long int in1size, double *in2){
  double sum=0.0;

  int ThreadPerBlock=128;
  int BlockPerGrid=ceil((double)size/(double)ThreadPerBlock);

  Memset(this->cu_d3, 0, BlockPerGrid);

  //d_1 -> in1
  //d_2 -> in2
  //d_3 -> out
  this->time->start();
  H2D((double*)(in1+(in1index*in1size)), this->cu_d2, size);
  H2D(in2, this->cu_d2, size);
  this->time->start();
  this->time->cu_dot_copy_time += this->time->getTime();


  if(ThreadPerBlock*8 >= 49152){
    std::cout << "Request shared memory size is over max shared memory size in per block !!! Max = 49152 !!! Request = " << ThreadPerBlock*8 << std::endl;
  }

  this->time->start();
  kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock)>>>(this->size, cu_d1, cu_d2, cu_d3);
  checkCudaErrors( cudaPeekAtLastError() );
  this->time->start();
  this->time->cu_dot_proc_time += this->time->getTime();


  //d_3 -> out
  //h_1 -> out(host)
  this->time->start();
  D2H(cu_d3, cu_h1, BlockPerGrid);
  this->time->start();
  this->time->cu_dot_copy_time += this->time->getTime();


  this->time->start();
#pragma omp parallel for schedule(static) reduction(+:sum)
  for(int i=0; i<BlockPerGrid; i++){
    sum += cu_h1[i];
  }
  this->time->start();
  this->time->cu_dot_reduce_time += this->time->getTime();


  return sum;
}

void cuda::CSR2CSC(double *dCSRval, int *dCSRcol, int *dCSRptr, double *CSCval, int *CSCrow, int *CSCptr, double *dCSCval, int *dCSCrow, int *dCSCptr, unsigned long int N, unsigned long int NNZ){
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

void cuda::CSR2CSC(double *CSRval, int *CSRcol, int *CSRptr, double *CSCval, int *CSCrow, int *CSCptr, unsigned long int N, unsigned long int NNZ){

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

void cuda::Kskip_cg_bicg_base(double *Ar, double *Ap, double *rvec, double *pvec, const int kskip, double *val, int *col, int *ptr){
  int ThreadPerBlock=128;
  int BlockPerGrid=(size-1)/(ThreadPerBlock/32)+1;

  Memset(this->cu_d4, 0, size*(2*k+1));
  Memset(this->cu_d5, 0, size*(2*k+2));

  //r -> d1
  //p -> d2
  this->time->start();
  H2D(rvec, this->cu_d1, size);
  H2D(pvec, this->cu_d2, size);
  this->time->end();
  this->time->cu_MV_copy_time += this->time->getTime();

  if(ThreadPerBlock*8 >= 49152){
    std::cout << "Request shared memory size is over max shared memory size in per block !!! Max = 49152 !!! Request = " << ThreadPerBlock*8 << std::endl;
  }

  // d1(in) --> d4(out)
  // d2(in) --> d5(out)

  this->time->start();
  kernel_MtxVec_mult<<<BlockPerGrid, ThreadPerBlock>>>(this->size, val, col, ptr, cu_d1, cu_d4, 0, this->size);
  checkCudaErrors( cudaPeekAtLastError() );
  
  kernel_MtxVec_mult<<<BlockPerGrid, ThreadPerBlock>>>(this->size, val, col, ptr, cu_d2, cu_d5, 0, this->size);
  checkCudaErrors( cudaPeekAtLastError() );
 
  for(int i=1; i<2*kskip+2; i++){
    if(i<2*kskip+1){
      kernel_MtxVec_mult<<<BlockPerGrid, ThreadPerBlock>>>(this->size, val, col, ptr, cu_d4, i-1, this->size, cu_d4, i, this->size);
      checkCudaErrors( cudaPeekAtLastError() );
    }
    kernel_MtxVec_mult<<<BlockPerGrid, ThreadPerBlock>>>(this->size, val, col, ptr, cu_d5, i-1, this->size, cu_d5, i, this->size);
    checkCudaErrors( cudaPeekAtLastError() );
  }
  this->time->end();
  this->time->cu_MV_proc_time += this->time->getTime();


  this->time->start();
  D2H(this->cu_d4, Ar, size*(2*kskip+1));
  D2H(this->cu_d5, Ap, size*(2*kskip+2));
  this->time->end();
  this->time->cu_MV_copy_time += this->time->getTime();

}

void cuda::Kskip_cg_bicg_base2(double *Ar, double *Ap, double *rvec, double *pvec, const int kskip, double *val, int *col, int *ptr){
  int ThreadPerBlock=128;
  int BlockPerGrid=(size-1)/(ThreadPerBlock/32)+1;

  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaMemsetAsync(cu_d4, 0, size*(2*kskip+1), stream1);

  cudaMemcpyAsync(this->cu_d1, rvec, sizeof(double)*size, cudaMemcpyHostToDevice, stream1);

  kernel_MtxVec_mult<<<BlockPerGrid, ThreadPerBlock, 0, stream1>>>(this->size, val, col, ptr, cu_d1, cu_d4, 0, this->size);

  cudaMemcpyAsync((double*)(Ar+(0*this->size)), (double*)(cu_d4+(0*this->size)), sizeof(double)*size, cudaMemcpyDeviceToHost, stream1);

  cudaMemsetAsync(cu_d5, 0, size*(2*kskip+2), stream2);

  cudaMemcpyAsync(this->cu_d2, pvec, sizeof(double)*size, cudaMemcpyHostToDevice, stream2);

  kernel_MtxVec_mult<<<BlockPerGrid, ThreadPerBlock, 0, stream2>>>(this->size, val, col, ptr, cu_d2, cu_d5, 0, this->size);

  cudaMemcpyAsync((double*)(Ap+(0*this->size)), (double*)(cu_d5+(0*this->size)), sizeof(double)*size, cudaMemcpyDeviceToHost, stream2);

  
  for(int i=1; i<2*kskip+2; i++){
    if(i<2*kskip+1){
      kernel_MtxVec_mult<<<BlockPerGrid, ThreadPerBlock, 0, stream1>>>(this->size, val, col, ptr, cu_d4, i-1, this->size, cu_d4, i, this->size);

      cudaMemcpyAsync((double*)(Ar+(i*this->size)), (double*)(cu_d4+(i*this->size)), sizeof(double)*size, cudaMemcpyDeviceToHost, stream1);
    }

    kernel_MtxVec_mult<<<BlockPerGrid, ThreadPerBlock, 0, stream2>>>(this->size, val, col, ptr, cu_d5, i-1, this->size, cu_d5, i, this->size);
    
    cudaMemcpyAsync((double*)(Ap+(i*this->size)), (double*)(cu_d5+(i*this->size)), sizeof(double)*size, cudaMemcpyDeviceToHost, stream2);
  }

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
}

void cuda::Kskip_cg_innerProduce(double *delta, double *eta, double *zeta, double *Ar, double *Ap, double *rvec, double *pvec, int kskip, double *val, int *col, int *ptr){

  int ThreadPerBlock=128;
  int BlockPerGrid=ceil((double)size/(double)ThreadPerBlock);

  Memset(this->cu_d6, 0, BlockPerGrid * (2*kskip));
  Memset(this->cu_d7, 0, BlockPerGrid * (2*kskip+1));
  Memset(this->cu_d8, 0, BlockPerGrid * (2*kskip+2));

  //d1 -> r
  //d2 -> p
  //d4 -> Ar
  //d5 -> Ap

  //d6 -> delta
  //d7 -> eta
  //d8 -> zeta

  if(ThreadPerBlock*8 >= 49152){
    std::cout << "Request shared memory size is over max shared memory size in per block !!! Max = 49152 !!! Request = " << ThreadPerBlock*8 << std::endl;
  }

  this->time->start();
  for(int i=0; i<2*kskip+2; i++){
    if(i<2*kskip){
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock)>>>(this->size, cu_d4, i, this->size, cu_d1, cu_d6, i, BlockPerGrid);
    }
    if(i<2*kskip+1){
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock)>>>(this->size, cu_d5, i, this->size, cu_d1, cu_d7, i, BlockPerGrid);
    }
    kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock)>>>(this->size, cu_d5, i, this->size, cu_d2, cu_d8, i, BlockPerGrid);
  }
  this->time->end();
  this->time->cu_dot_proc_time += this->time->getTime();


  //d6 -> delta -> h2
  //d7 -> eta -> h3
  //d8 -> zeta -> h4

  this->time->start();
  D2H(cu_d6, cu_h2, BlockPerGrid * (2*kskip));
  D2H(cu_d7, cu_h3, BlockPerGrid * (2*kskip+1));
  D2H(cu_d8, cu_h4, BlockPerGrid * (2*kskip+2));
  this->time->end();
  this->time->cu_dot_copy_time += this->time->getTime();

  this->time->start();
  double tmp1 = 0.0;
  double tmp2 = 0.0;
  double tmp3 = 0.0;
#pragma omp parallel for reduction(+:tmp1, tmp2, tmp3) schedule(static) firstprivate(delta, eta, zeta, cu_h2, cu_h3, cu_h4) lastprivate(delta, eta, zeta)
  for(int i=0; i<2*kskip+2; i++){
    tmp1 = 0.0;
    tmp2 = 0.0;
    tmp3 = 0.0;
    if(i<2*kskip){
      for(int j=0; j<BlockPerGrid; j++){
        tmp1 += cu_h2[i*BlockPerGrid+j];
      }
      delta[i] = tmp1;
    }
    if(i<2*kskip+1){
      for(int j=0; j<BlockPerGrid; j++){
        tmp2 += cu_h3[i*BlockPerGrid+j];
      }
      eta[i] = tmp2;
    }
    for(int j=0; j<BlockPerGrid; j++){
      tmp3 += cu_h4[i*BlockPerGrid+j];
    }
    zeta[i] = tmp3;
  }
  this->time->end();
  this->time->cu_dot_reduce_time += this->time->getTime();
}

void cuda::Kskip_cg_innerProduce2(double *delta, double *eta, double *zeta, double *Ar, double *Ap, double *rvec, double *pvec, int kskip, double *val, int *col, int *ptr){
  cudaStream_t stream1, stream2, stream3;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);

  int ThreadPerBlock=128;
  int BlockPerGrid=ceil((double)size/(double)ThreadPerBlock);

  cudaMemsetAsync(cu_d6, 0, sizeof(double)*BlockPerGrid*(2*kskip), stream1);
  cudaMemsetAsync(cu_d7, 0, sizeof(double)*BlockPerGrid*(2*kskip+1), stream2);
  cudaMemsetAsync(cu_d8, 0, sizeof(double)*BlockPerGrid*(2*kskip+2), stream3);

  //d1 -> r
  //d2 -> p
  //d4 -> Ar
  //d5 -> Ap

  //d6 -> delta
  //d7 -> eta
  //d8 -> zeta

double tmp1 = 0.0;
double tmp2 = 0.0;
double tmp3 = 0.0;

  for(int i=0; i<2*kskip+2; i++){
    tmp1 = 0.0;
    tmp2 = 0.0;
    tmp3 = 0.0;
    if(i<2*kskip){
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream1>>>(this->size, cu_d4, i, this->size, cu_d1, cu_d6, i, BlockPerGrid);
      cudaMemcpyAsync((double*)(cu_h2+(i*BlockPerGrid)), (double*)(cu_d6+(i*BlockPerGrid)), sizeof(double)*(BlockPerGrid), cudaMemcpyDeviceToHost, stream1);
      cudaStreamSynchronize(stream1);

      for(int j=0; j<BlockPerGrid; j++){
        tmp1 += cu_h2[i*BlockPerGrid+j];
      }
      delta[i] = tmp1;

    }
    if(i<2*kskip+1){
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream2>>>(this->size, cu_d5, i, this->size, cu_d1, cu_d7, i, BlockPerGrid);
      cudaMemcpyAsync((double*)(cu_h3+(i*BlockPerGrid)), (double*)(cu_d7+(i*BlockPerGrid)), sizeof(double)*(BlockPerGrid), cudaMemcpyDeviceToHost, stream2);
      cudaStreamSynchronize(stream2);

      for(int j=0; j<BlockPerGrid; j++){
        tmp2 += cu_h3[i*BlockPerGrid+j];
      }
      eta[i] = tmp2;

    }
    kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream3>>>(this->size, cu_d5, i, this->size, cu_d2, cu_d8, i, BlockPerGrid);
    cudaMemcpyAsync((double*)(cu_h4+(i*BlockPerGrid)), (double*)(cu_d8+(i*BlockPerGrid)), sizeof(double)*(BlockPerGrid), cudaMemcpyDeviceToHost, stream3);
    cudaStreamSynchronize(stream3);

    for(int j=0; j<BlockPerGrid; j++){
      tmp3 += cu_h4[i*BlockPerGrid+j];
    }
    zeta[i] = tmp3;

  }
}

void cuda::Kskip_cg_innerProduce3(double *delta, double *eta, double *zeta, double *Ar, double *Ap, double *rvec, double *pvec, int kskip, double *val, int *col, int *ptr){
  cudaStream_t stream1, stream2, stream3;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);

  int ThreadPerBlock=128;
  int BlockPerGrid=ceil((double)size/(double)ThreadPerBlock);

  cudaMemsetAsync(cu_d6, 0, sizeof(double)*BlockPerGrid*(2*kskip), stream1);
  cudaMemsetAsync(cu_d7, 0, sizeof(double)*BlockPerGrid*(2*kskip+1), stream2);
  cudaMemsetAsync(cu_d8, 0, sizeof(double)*BlockPerGrid*(2*kskip+2), stream3);

  //d1 -> r
  //d2 -> p
  //d4 -> Ar
  //d5 -> Ap

  //d6 -> delta
  //d7 -> eta
  //d8 -> zeta

  cudaDeviceSynchronize();
  
  double tmp1 = 0.0;
  double tmp2 = 0.0;
  double tmp3 = 0.0;

  for(int i=0; i<2*kskip+2; i++){
    tmp1 = 0.0;
    tmp2 = 0.0;
    tmp3 = 0.0;
    if(i<2*kskip){
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream1>>>(this->size, cu_d4, i, this->size, cu_d1, cu_d6, i, BlockPerGrid);
      if(i==2*kskip-1){
        cudaMemcpyAsync(cu_h2, cu_d6, sizeof(double)*(BlockPerGrid)*(2*kskip), cudaMemcpyDeviceToHost, stream1);
      }
    }
    if(i<2*kskip+1){
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream2>>>(this->size, cu_d5, i, this->size, cu_d1, cu_d7, i, BlockPerGrid);
      if(i==2*kskip){
        cudaMemcpyAsync(cu_h3, cu_d7, sizeof(double)*(BlockPerGrid)*(2*kskip+1), cudaMemcpyDeviceToHost, stream2);
      }
    }
    kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream3>>>(this->size, cu_d5, i, this->size, cu_d2, cu_d8, i, BlockPerGrid);
    if(i==2*kskip+1){
      cudaMemcpyAsync(cu_h4, cu_d8, sizeof(double)*(BlockPerGrid)*(2*kskip+2), cudaMemcpyDeviceToHost, stream3);
    }
    cudaDeviceSynchronize();
  }
#pragma omp parallel for reduction(+:tmp1, tmp2, tmp3) schedule(static) firstprivate(delta, eta, zeta, cu_h2, cu_h3, cu_h4) lastprivate(delta, eta, zeta)
  for(int i=0; i<2*kskip+2; i++){
    tmp1 = 0.0;
    tmp2 = 0.0;
    tmp3 = 0.0;
    if(i<2*kskip){
      for(int j=0; j<BlockPerGrid; j++){
        tmp1 += cu_h2[i*BlockPerGrid+j];
      }
      delta[i] = tmp1;
    }
    if(i<2*kskip+1){
      for(int j=0; j<BlockPerGrid; j++){
        tmp2 += cu_h3[i*BlockPerGrid+j];
      }
      eta[i] = tmp2;
    }
    for(int j=0; j<BlockPerGrid; j++){
      tmp3 += cu_h4[i*BlockPerGrid+j];
    }
    zeta[i] = tmp3;
  }
}

void cuda::Kskip_bicg_innerProduce(double *theta, double *eta, double *rho, double *phi, double *Ar, double *Ap, double *r_vec, double *p_vec, int kskip, double *val, int *col, int *ptr){

  int ThreadPerBlock=128;
  int BlockPerGrid=ceil((double)size/(double)ThreadPerBlock);

  Memset(this->cu_d6, 0, BlockPerGrid * (2*kskip));
  Memset(this->cu_d7, 0, BlockPerGrid * (2*kskip+1));
  Memset(this->cu_d9, 0, BlockPerGrid * (2*kskip+1));
  Memset(this->cu_d8, 0, BlockPerGrid * (2*kskip+2));


  this->time->start();
  H2D(r_vec, cu_d1, this->size);
  H2D(p_vec, cu_d2, this->size);
  this->time->end();
  this->time->cu_MV_copy_time += this->time->getTime();

  //d1 -> *r
  //d2 -> *p
  //d4 -> Ar
  //d5 -> Ap

  //d6 -> theta
  //d7 -> eta
  //d9 -> rho
  //d8 -> phi

  if(ThreadPerBlock*8 >= 49152){
    std::cout << "Request shared memory size is over max shared memory size in per block !!! Max = 49152 !!! Request = " << ThreadPerBlock*8 << std::endl;
  }

  this->time->start();
  for(int i=0; i<2*kskip+2; i++){
    if(i<2*kskip){
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock)>>>(this->size, cu_d4, i, this->size, cu_d1, cu_d6, i, BlockPerGrid);
    }
    if(i<2*kskip+1){
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock)>>>(this->size, cu_d5, i, this->size, cu_d1, cu_d7, i, BlockPerGrid);
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock)>>>(this->size, cu_d4, i, this->size, cu_d2, cu_d9, i, BlockPerGrid);
    }
    kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock)>>>(this->size, cu_d5, i, this->size, cu_d2, cu_d8, i, BlockPerGrid);
  }
  this->time->end();
  this->time->cu_dot_proc_time += this->time->getTime();


  //d6 -> theta -> h2
  //d7 -> eta -> h3
  //d9 -> rho -> h5
  //d8 -> phi -> h4

  this->time->start();
  D2H(cu_d6, cu_h2, BlockPerGrid * (2*kskip));
  D2H(cu_d7, cu_h3, BlockPerGrid * (2*kskip+1));
  D2H(cu_d9, cu_h5, BlockPerGrid * (2*kskip+1));
  D2H(cu_d8, cu_h4, BlockPerGrid * (2*kskip+2));
  this->time->end();
  this->time->cu_dot_copy_time += this->time->getTime();


  this->time->start();
  double tmp1 = 0.0;
  double tmp2 = 0.0;
  double tmp3 = 0.0;
  double tmp4 = 0.0;
#pragma omp parallel for reduction(+:tmp1, tmp2, tmp3, tmp4) schedule(static) firstprivate(theta, eta, rho, phi, cu_h2, cu_h3, cu_h4, cu_h5) lastprivate(theta, eta, rho, phi)
  for(int i=0; i<2*kskip+2; i++){
    tmp1 = 0.0;
    tmp2 = 0.0;
    tmp3 = 0.0;
    tmp4 = 0.0;
    if(i<2*kskip){
      for(int j=0; j<BlockPerGrid; j++){
        tmp1 += cu_h2[i*BlockPerGrid+j];
      }
      theta[i] = tmp1;
    }
    if(i<2*kskip+1){
      for(int j=0; j<BlockPerGrid; j++){
        tmp2 += cu_h3[i*BlockPerGrid+j];
        tmp3 += cu_h5[i*BlockPerGrid+j];
      }
      eta[i] = tmp2;
      rho[i] = tmp3;
    }
    for(int j=0; j<BlockPerGrid; j++){
      tmp4 += cu_h4[i*BlockPerGrid+j];
    }
    phi[i] = tmp4;
  }
  this->time->end();
  this->time->cu_dot_reduce_time += this->time->getTime();


}

void cuda::Kskip_bicg_innerProduce2(double *theta, double *eta, double *rho, double *phi, double *Ar, double *Ap, double *r_vec, double *p_vec, int kskip, double *val, int *col, int *ptr){
  //d1 -> *r
  //d2 -> *p
  //d4 -> Ar
  //d5 -> Ap

  //d6 -> theta
  //d7 -> eta
  //d9 -> rho
  //d8 -> phi

  //d6 -> theta -> h2
  //d7 -> eta -> h3
  //d9 -> rho -> h5
  //d8 -> phi -> h4

  cudaStream_t stream1, stream2, stream3, stream4;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);
  cudaStreamCreate(&stream4);

  int ThreadPerBlock=128;
  int BlockPerGrid=ceil((double)size/(double)ThreadPerBlock);

  cudaMemsetAsync(cu_d6, 0, sizeof(double)*BlockPerGrid*(2*kskip), stream1);
  cudaMemsetAsync(cu_d7, 0, sizeof(double)*BlockPerGrid*(2*kskip+1), stream2);
  cudaMemsetAsync(cu_d9, 0, sizeof(double)*BlockPerGrid*(2*kskip+1), stream3);
  cudaMemsetAsync(cu_d8, 0, sizeof(double)*BlockPerGrid*(2*kskip+2), stream4);


  cudaMemcpyAsync(cu_d1, r_vec, sizeof(double)*size, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(cu_d2, p_vec, sizeof(double)*size, cudaMemcpyHostToDevice, stream2);

  double tmp1 = 0.0;
  double tmp2 = 0.0;
  double tmp3 = 0.0;
  double tmp4 = 0.0;
  for(int i=0; i<2*kskip+2; i++){
    tmp1 = 0.0;
    tmp2 = 0.0;
    tmp3 = 0.0;
    tmp4 = 0.0;
    if(i<2*kskip){
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream1>>>(this->size, cu_d4, i, this->size, cu_d1, cu_d6, i, BlockPerGrid);
      cudaMemcpyAsync((double*)(cu_h2+(i*BlockPerGrid)), (double*)(cu_d6+(i*BlockPerGrid)), sizeof(double)*(BlockPerGrid), cudaMemcpyDeviceToHost, stream1);
      cudaStreamSynchronize(stream1);
#pragma omp parallel for reduction(+:tmp1) schedule(static)
      for(int j=0; j<BlockPerGrid; j++){
        tmp1 += cu_h2[i*BlockPerGrid+j];
      }
      theta[i] = tmp1;
    }
    if(i<2*kskip+1){
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream2>>>(this->size, cu_d5, i, this->size, cu_d1, cu_d7, i, BlockPerGrid);
      cudaMemcpyAsync((double*)(cu_h3+(i*BlockPerGrid)), (double*)(cu_d7+(i*BlockPerGrid)), sizeof(double)*(BlockPerGrid), cudaMemcpyDeviceToHost, stream2);
      cudaStreamSynchronize(stream2);
#pragma omp parallel for reduction(+:tmp2) schedule(static)
      for(int j=0; j<BlockPerGrid; j++){
        tmp2 += cu_h3[i*BlockPerGrid+j];
      }
      eta[i] = tmp2;
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream3>>>(this->size, cu_d4, i, this->size, cu_d2, cu_d9, i, BlockPerGrid);
      cudaMemcpyAsync((double*)(cu_h5+(i*BlockPerGrid)), (double*)(cu_d9+(i*BlockPerGrid)), sizeof(double)*(BlockPerGrid), cudaMemcpyDeviceToHost, stream3);
      cudaStreamSynchronize(stream3);
#pragma omp parallel for reduction(+:tmp3) schedule(static)
      for(int j=0; j<BlockPerGrid; j++){
        tmp3 += cu_h5[i*BlockPerGrid+j];
      }
      rho[i] = tmp3;
    }
    kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream4>>>(this->size, cu_d5, i, this->size, cu_d2, cu_d8, i, BlockPerGrid);
    cudaMemcpyAsync((double*)(cu_h4+(i*BlockPerGrid)), (double*)(cu_d8+(i*BlockPerGrid)), sizeof(double)*(BlockPerGrid), cudaMemcpyDeviceToHost, stream4);
    cudaStreamSynchronize(stream4);
#pragma omp parallel for reduction(+:tmp4) schedule(static)
    for(int j=0; j<BlockPerGrid; j++){
      tmp4 += cu_h4[i*BlockPerGrid+j];
    }
    phi[i] = tmp4;
  }

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaStreamDestroy(stream3);
  cudaStreamDestroy(stream4);

}

void cuda::Kskip_bicg_innerProduce3(double *theta, double *eta, double *rho, double *phi, double *Ar, double *Ap, double *r_vec, double *p_vec, int kskip, double *val, int *col, int *ptr){
  //d1 -> *r
  //d2 -> *p
  //d4 -> Ar
  //d5 -> Ap

  //d6 -> theta
  //d7 -> eta
  //d9 -> rho
  //d8 -> phi

  //d6 -> theta -> h2
  //d7 -> eta -> h3
  //d9 -> rho -> h5
  //d8 -> phi -> h4

  cudaStream_t stream1, stream2, stream3, stream4;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);
  cudaStreamCreate(&stream4);

  int ThreadPerBlock=128;
  int BlockPerGrid=ceil((double)size/(double)ThreadPerBlock);

  cudaMemsetAsync(cu_d6, 0, sizeof(double)*BlockPerGrid*(2*kskip), stream1);
  cudaMemsetAsync(cu_d7, 0, sizeof(double)*BlockPerGrid*(2*kskip+1), stream2);
  cudaMemsetAsync(cu_d9, 0, sizeof(double)*BlockPerGrid*(2*kskip+1), stream3);
  cudaMemsetAsync(cu_d8, 0, sizeof(double)*BlockPerGrid*(2*kskip+2), stream4);

  cudaMemcpyAsync(cu_d1, r_vec, sizeof(double)*size, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(cu_d2, p_vec, sizeof(double)*size, cudaMemcpyHostToDevice, stream2);

  cudaDeviceSynchronize();

  double tmp1 = 0.0;
  double tmp2 = 0.0;
  double tmp3 = 0.0;
  double tmp4 = 0.0;
  for(int i=0; i<2*kskip+2; i++){
    if(i<2*kskip){
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream1>>>(this->size, cu_d4, i, this->size, cu_d1, cu_d6, i, BlockPerGrid);
      if(i==2*kskip-1){
        cudaMemcpyAsync(cu_h2, cu_d6, sizeof(double)*(BlockPerGrid*(2*kskip)), cudaMemcpyDeviceToHost, stream1);
      }
    }
    if(i<2*kskip+1){
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream2>>>(this->size, cu_d5, i, this->size, cu_d1, cu_d7, i, BlockPerGrid);
      if(i==2*kskip){
        cudaMemcpyAsync(cu_h3, cu_d7, sizeof(double)*(BlockPerGrid*(2*kskip+1)), cudaMemcpyDeviceToHost, stream2);
      }
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream3>>>(this->size, cu_d4, i, this->size, cu_d2, cu_d9, i, BlockPerGrid);
      if(i==2*kskip){
        cudaMemcpyAsync(cu_h5, cu_d9, sizeof(double)*(BlockPerGrid*(2*kskip+1)), cudaMemcpyDeviceToHost, stream3);
      }
    }
    kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream4>>>(this->size, cu_d5, i, this->size, cu_d2, cu_d8, i, BlockPerGrid);
    if(i==2*kskip+1){
      cudaMemcpyAsync(cu_h4, cu_d8, sizeof(double)*(BlockPerGrid*(2*kskip+2)), cudaMemcpyDeviceToHost, stream4);
    }
    cudaDeviceSynchronize();
  }

#pragma omp parallel for reduction(+:tmp1, tmp2, tmp3, tmp4) schedule(static) firstprivate(theta, eta, rho, phi, cu_h2, cu_h3, cu_h4, cu_h5) lastprivate(theta, eta, rho, phi)
  for(int i=0; i<2*kskip+2; i++){
    tmp1 = 0.0;
    tmp2 = 0.0;
    tmp3 = 0.0;
    tmp4 = 0.0;
    if(i<2*kskip){
      for(int j=0; j<BlockPerGrid; j++){
        tmp1 += cu_h2[i*BlockPerGrid+j];
      }
      theta[i] = tmp1;
    }
    if(i<2*kskip+1){
      for(int j=0; j<BlockPerGrid; j++){
        tmp2 += cu_h3[i*BlockPerGrid+j];
        tmp3 += cu_h5[i*BlockPerGrid+j];
      }
      eta[i] = tmp2;
      rho[i] = tmp3;
    }
    for(int j=0; j<BlockPerGrid; j++){
      tmp4 += cu_h4[i*BlockPerGrid+j];
    }
    phi[i] = tmp4;
  }
  
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaStreamDestroy(stream3);
  cudaStreamDestroy(stream4);

}

void cuda::dot_gmres(double *wvec, double *vmtx, double *hmtx, int k, unsigned long int N){
  for(int i=0; i<=k; i++){
    hmtx[k+i*N] = dot(wvec, vmtx, i, N);
  }
}

void cuda::dot_gmres2(double *wvec, double *vmtx, double *hmtx, int k, unsigned long int N){
  int ThreadPerBlock=128;
  int BlockPerGrid=ceil((double)size/(double)ThreadPerBlock);


  cudaStream_t stream1, stream2, stream3;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);
  double tmp1=0;
  double tmp2=0;

  cudaMemcpyAsync(cu_d1, wvec, sizeof(double)*size, cudaMemcpyHostToDevice, stream1);
  cudaStreamSynchronize(stream1);

  if(k==0){
    //hmtx[0] = dot(wvec, vmtx);
    cudaMemsetAsync(cu_d3, 0, BlockPerGrid, stream1);
    cudaMemcpyAsync(cu_d2, (double*)(vmtx+(0*size)), sizeof(double)*size, cudaMemcpyHostToDevice, stream1);
    kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream1>>>(this->size, cu_d1, cu_d2, cu_d3);
    cudaMemcpyAsync(cu_h1, cu_d3, sizeof(double)*BlockPerGrid, cudaMemcpyDeviceToHost, stream1);
    cudaStreamSynchronize(stream1);
    for(int i=0; i<BlockPerGrid; i++){
      tmp1 += cu_h1[i];
    }
    hmtx[0] = tmp1;
  }else if((k+1)%2 == 0){
    //even
    /* for(int i=0; i<=k; i++){ */
    /*   hmtx[k+i*size] = dot(wvec, vmtx, i, size); */
    /* } */
    for(int i=0; i<=k; i+=2){
      tmp1 = 0;
      tmp2 = 0;
      cudaMemsetAsync(cu_d3, 0, BlockPerGrid, stream1);
      cudaMemcpyAsync(cu_d2, (double*)(vmtx+(i*size)), sizeof(double)*size, cudaMemcpyHostToDevice, stream1);
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream1>>>(this->size, cu_d1, cu_d2, cu_d3);
      cudaMemcpyAsync(cu_h1, cu_d3, sizeof(double)*BlockPerGrid, cudaMemcpyDeviceToHost, stream1);
      cudaStreamSynchronize(stream1);
      for(int j=0; j<BlockPerGrid; j++){
        tmp1 += cu_h1[j];
      }
      hmtx[k+i*size] = tmp1;
      cudaMemsetAsync(cu_d101, 0, BlockPerGrid, stream2);
      cudaMemcpyAsync(cu_d100, (double*)(vmtx+((i+1)*size)), sizeof(double)*size, cudaMemcpyHostToDevice, stream2);
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream2>>>(this->size, cu_d1, cu_d100, cu_d101);
      cudaMemcpyAsync(cu_h1, cu_d101, sizeof(double)*BlockPerGrid, cudaMemcpyDeviceToHost, stream2);
      cudaStreamSynchronize(stream2);
      for(int j=0; j<BlockPerGrid; j++){
        tmp2 += cu_h1[j];
      }
      hmtx[k+(i+1)*size] = tmp2;
    }

  }else{
    //odd
    /* for(int i=0; i<=k; i++){ */
    /*   hmtx[k+i*size] = dot(wvec, vmtx, i, size); */
    /* } */
    for(int i=0; i<=k-1; i+=2){
      tmp1 = 0;
      tmp2 = 0;
      cudaMemsetAsync(cu_d3, 0, BlockPerGrid, stream1);
      cudaMemcpyAsync(cu_d2, (double*)(vmtx+(i*size)), sizeof(double)*size, cudaMemcpyHostToDevice, stream1);
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream1>>>(this->size, cu_d1, cu_d2, cu_d3);
      cudaMemcpyAsync(cu_h1, cu_d3, sizeof(double)*BlockPerGrid, cudaMemcpyDeviceToHost, stream1);
      cudaStreamSynchronize(stream1);
      for(int j=0; j<BlockPerGrid; j++){
        tmp1 += cu_h1[j];
      }
      hmtx[k+i*size] = tmp1;
      cudaMemsetAsync(cu_d101, 0, BlockPerGrid, stream2);
      cudaMemcpyAsync(cu_d100, (double*)(vmtx+((i+1)*size)), sizeof(double)*size, cudaMemcpyHostToDevice, stream2);
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream2>>>(this->size, cu_d1, cu_d100, cu_d101);
      cudaMemcpyAsync(cu_h1, cu_d101, sizeof(double)*BlockPerGrid, cudaMemcpyDeviceToHost, stream2);
      cudaStreamSynchronize(stream2);
      for(int j=0; j<BlockPerGrid; j++){
        tmp2 += cu_h1[j];
      }
      hmtx[k+(i+1)*size] = tmp2;
    }

    tmp1 = 0;
    cudaMemsetAsync(cu_d3, 0, BlockPerGrid, stream1);
    cudaMemcpyAsync(cu_d2, (double*)(vmtx+(k*size)), sizeof(double)*size, cudaMemcpyHostToDevice, stream1);
    kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream1>>>(this->size, cu_d1, cu_d2, cu_d3);
    cudaMemcpyAsync(cu_h1, cu_d3, sizeof(double)*BlockPerGrid, cudaMemcpyDeviceToHost, stream1);
    cudaStreamSynchronize(stream1);
    for(int i=0; i<BlockPerGrid; i++){
      tmp1 += cu_h1[i];
    }
    hmtx[k+k*size] = tmp1;
  }

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaStreamDestroy(stream3);
}

void cuda::dot_gmres3(double *wvec, double *vmtx, double *hmtx, int k, unsigned long int N){
  int ThreadPerBlock=128;
  int BlockPerGrid=ceil((double)size/(double)ThreadPerBlock);


  
  cudaStream_t stream[32];
  for(int i=0; i<32; i++){
    cudaStreamCreate(&stream[i]);
  }

  cudaMemcpyAsync(cu_d1, wvec, sizeof(double)*size, cudaMemcpyHostToDevice, stream[0]);
  cudaMemsetAsync(cu_d201, 0, sizeof(double)*BlockPerGrid*(k+1), stream[1]);
  cudaStreamSynchronize(stream[0]);
  cudaStreamSynchronize(stream[1]);


  double tmp1=0.0;


  if(k==0){
    hmtx[0] = dot(wvec, vmtx);
  }else if((k+1)%2 == 0){
    //even
    /* for(int i=0; i<=k; i++){ */
    /*   hmtx[k+i*size] = dot(wvec, vmtx, i, size); */
    /* } */
    if(k+1 <= 32){
      for(int i=0; i<k+1; i++){
        cudaMemcpyAsync((double*)(cu_d200+(i*size)), (double*)(vmtx+(i*size)), sizeof(double)*size, cudaMemcpyHostToDevice, stream[i]);
        kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream[i]>>>(this->size, cu_d1, (double*)(cu_d200+(i*size)), (double*)(cu_d201+(i*BlockPerGrid)));
      }
      cudaDeviceSynchronize();
      cudaMemcpyAsync(cu_h8, cu_d201, sizeof(double)*BlockPerGrid*(k+1), cudaMemcpyDeviceToHost, stream[0]);
      cudaDeviceSynchronize();

#pragma omp parallel for reduction(+:tmp1) schedule(static) firstprivate(hmtx, eta, cu_h8) lastprivate(hmtx)
      for(int j=0; j<k+1; j++){
        tmp1 = 0.0;
        for(int q=0; q<BlockPerGrid; q++){
          tmp1 += cu_h8[j*BlockPerGrid+q];
        }
        hmtx[k+j*size] = tmp1;
      }
    }else if(k+1 > 32){
      int all = k+1;
      int offset = 0;
      while(all > 32){
        for(int i=0; i<32; i++){
          cudaMemcpyAsync((double*)(cu_d200+((i+offset)*size)), (double*)(vmtx+((i+offset)*size)), sizeof(double)*size, cudaMemcpyHostToDevice, stream[i]);
          kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream[i]>>>(this->size, cu_d1, (double*)(cu_d200+((i+offset)*size)), (double*)(cu_d201+((i+offset)*BlockPerGrid)));
        }
        cudaDeviceSynchronize();
        cudaMemcpyAsync(cu_h8, cu_d201, sizeof(double)*BlockPerGrid*(k+1), cudaMemcpyDeviceToHost, stream[0]);
        cudaDeviceSynchronize();

#pragma omp parallel for reduction(+:tmp1) schedule(static) firstprivate(hmtx, eta, cu_h8) lastprivate(hmtx)
        for(int j=offset; j<offset+32; j++){
          tmp1 = 0.0;
          for(int q=0; q<BlockPerGrid; q++){
            tmp1 += cu_h8[j*BlockPerGrid+q];
          }
          hmtx[k+j*size] = tmp1;
        }
        offset += 32;
        all -= 32;
      }
      for(int i=0; i<all; i++){
        cudaMemcpyAsync((double*)(cu_d200+((i+offset)*size)), (double*)(vmtx+((i+offset)*size)), sizeof(double)*size, cudaMemcpyHostToDevice, stream[i]);
        kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream[i]>>>(this->size, cu_d1, (double*)(cu_d200+((i+offset)*size)), (double*)(cu_d201+((i+offset)*BlockPerGrid)));
        cudaDeviceSynchronize();
        cudaMemcpyAsync(cu_h8, cu_d201, sizeof(double)*BlockPerGrid*(k+1), cudaMemcpyDeviceToHost, stream[0]);
        cudaDeviceSynchronize();

#pragma omp parallel for reduction(+:tmp1) schedule(static) firstprivate(hmtx, eta, cu_h8) lastprivate(hmtx)
        for(int j=offset; j<offset+all; j++){
          tmp1 = 0.0;
          for(int q=0; q<BlockPerGrid; q++){
            tmp1 += cu_h8[j*BlockPerGrid+q];
          }
          hmtx[k+j*size] = tmp1;
        }
      }
    }
  }else{
    //odd
    /* for(int i=0; i<=k; i++){ */
    /*   hmtx[k+i*size] = dot(wvec, vmtx, i, size); */
    /* } */
    if(k <= 32){
      for(int i=0; i<k; i++){
        cudaMemcpyAsync((double*)(cu_d200+(i*size)), (double*)(vmtx+(i*size)), sizeof(double)*size, cudaMemcpyHostToDevice, stream[i]);
        kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream[i]>>>(this->size, cu_d1, (double*)(cu_d200+(i*size)), (double*)(cu_d201+(i*BlockPerGrid)));
      }
      cudaDeviceSynchronize();
      cudaMemcpyAsync(cu_h8, cu_d201, sizeof(double)*BlockPerGrid*(k+1), cudaMemcpyDeviceToHost, stream[0]);
      cudaDeviceSynchronize();

#pragma omp parallel for reduction(+:tmp1) schedule(static) firstprivate(hmtx, eta, cu_h8) lastprivate(hmtx)
      for(int j=0; j<k; j++){
        tmp1 = 0.0;
        for(int q=0; q<BlockPerGrid; q++){
          tmp1 += cu_h8[j*BlockPerGrid+q];
        }
        hmtx[k+j*size] = tmp1;
      }
      cudaMemcpyAsync((double*)(cu_d200+(k*size)), (double*)(vmtx+(k*size)), sizeof(double)*size, cudaMemcpyHostToDevice, stream[0]);
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream[0]>>>(this->size, cu_d1, (double*)(cu_d200+(k*size)), (double*)(cu_d201+(k*BlockPerGrid)));
      cudaDeviceSynchronize();
      cudaMemcpyAsync(cu_h8, cu_d201, sizeof(double)*BlockPerGrid*(k+1), cudaMemcpyDeviceToHost, stream[0]);
      cudaDeviceSynchronize();
      tmp1 = 0.0;
      for(int q=0; q<BlockPerGrid; q++){
        tmp1 += cu_h8[k*BlockPerGrid+q]; 
      }
      hmtx[k+k*size] = tmp1;
    }else if(k>32){
      int all = k;
      int offset = 0;
      while(all > 32){
        for(int i=0; i<32; i++){
          cudaMemcpyAsync((double*)(cu_d200+((i+offset)*size)), (double*)(vmtx+((i+offset)*size)), sizeof(double)*size, cudaMemcpyHostToDevice, stream[i]);
          kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream[i]>>>(this->size, cu_d1, (double*)(cu_d200+((i+offset)*size)), (double*)(cu_d201+((i+offset)*BlockPerGrid)));
        }
        cudaDeviceSynchronize();
        cudaMemcpyAsync(cu_h8, cu_d201, sizeof(double)*BlockPerGrid*(k+1), cudaMemcpyDeviceToHost, stream[0]);
        cudaDeviceSynchronize();

#pragma omp parallel for reduction(+:tmp1) schedule(static) firstprivate(hmtx, eta, cu_h8) lastprivate(hmtx)
        for(int j=offset; j<offset+32; j++){
          tmp1 = 0.0;
          for(int q=0; q<BlockPerGrid; q++){
            tmp1 += cu_h8[j*BlockPerGrid+q];
          }
          hmtx[k+j*size] = tmp1;
        }
        offset += 32;
        all -= 32;
      }
      for(int i=0; i<all; i++){
        cudaMemcpyAsync((double*)(cu_d200+((i+offset)*size)), (double*)(vmtx+((i+offset)*size)), sizeof(double)*size, cudaMemcpyHostToDevice, stream[i]);
        kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream[i]>>>(this->size, cu_d1, (double*)(cu_d200+((i+offset)*size)), (double*)(cu_d201+((i+offset)*BlockPerGrid)));
        cudaDeviceSynchronize();
        cudaMemcpyAsync(cu_h8, cu_d201, sizeof(double)*BlockPerGrid*(k+1), cudaMemcpyDeviceToHost, stream[0]);
        cudaDeviceSynchronize();

#pragma omp parallel for reduction(+:tmp1) schedule(static) firstprivate(hmtx, eta, cu_h8) lastprivate(hmtx)
        for(int j=offset; j<offset+all; j++){
          tmp1 = 0.0;
          for(int q=0; q<BlockPerGrid; q++){
            tmp1 += cu_h8[j*BlockPerGrid+q];
          }
          hmtx[k+j*size] = tmp1;
        }
      }
      cudaMemcpyAsync((double*)(cu_d200+(k*size)), (double*)(vmtx+(k*size)), sizeof(double)*size, cudaMemcpyHostToDevice, stream[0]);
      kernel_dot<<<BlockPerGrid, ThreadPerBlock, sizeof(double)*(ThreadPerBlock), stream[0]>>>(this->size, cu_d1, (double*)(cu_d200+(k*size)), (double*)(cu_d201+(k*BlockPerGrid)));
      cudaDeviceSynchronize();
      cudaMemcpyAsync(cu_h8, cu_d201, sizeof(double)*BlockPerGrid*(k+1), cudaMemcpyDeviceToHost, stream[0]);
      cudaDeviceSynchronize();
      tmp1 = 0.0;
      for(int q=0; q<BlockPerGrid; q++){
        tmp1 += cu_h8[k*BlockPerGrid+q]; 
      }
      hmtx[k+k*size] = tmp1;
    }
  }

  for(int i=0; i<32; i++){
    cudaStreamDestroy(stream[i]);
  }
}

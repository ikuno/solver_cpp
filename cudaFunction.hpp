#ifndef CUDAFUNCTION_HPP_INCLUDED__
#define CUDAFUNCTION_HPP_INCLUDED__

#include <iostream>
#include "times.hpp"

class cuda { 
  private:
    double *cu_d1, *cu_d2, *cu_d3, *cu_h1;
    int size;

  public:
    times *time;
    double dot_copy_time;
    double dot_proc_time;
    double dot_malloc_time;
    double dot_reduce_time;

    double MV_copy_time;
    double MV_proc_time;
    double MV_malloc_time;

    double All_malloc_time;

    cuda();
    cuda(int size);
    ~cuda();

    //--------------------------------------
    double* d_Malloc(int size);
    double* d_MallocHost(int size);

    int* i_Malloc(int size);
    int* i_MallocHost(int size);

    void H2D(double* from, double* to, int size);
    void D2H(double* from, double* to, int size);

    void H2D(int* from, int* to, int size);
    void D2H(int* from, int* to, int size);

    void Free(void* ptr);

    void FreeHost(void* ptr);

    void Memset(double *ptr, double val, int size);

    void Memset(int *ptr, int val, int size);

    void Reset();

    //--------------------------------------
    void CSR2CSC(double *dCSRval, int *dCSRcol, int *dCSRptr, double *CSCval, int *CSCrow, int *CSCptr, double *dCSCval, int *dCSCrow, int *dCSCptr, int N, int NNZ);

    void CSR2CSC(double *CSRval, int *CSRcol, int *CSRptr, double *CSCval, int *CSCrow, int *CSCptr, int N, int NNZ);

    void MtxVec_mult(double *in, double *out, int size, double *val, int *col, int *ptr);

    void MtxVec_mult(double *in, double *out, double *val, int *col, int *ptr);

    void MtxVec_mult(double *in, int inindex, int insize, double *out, int outindex, int outsize, double *val, int *col, int *ptr);

    double dot(double *in1, double *in2, int size);

    double dot(double *in1, double *in2);

    double dot(double *in1, int in1index, int in1size, double *in2, int in2index, int in2size);

    double dot(double *in1, double *in2, int in2index, int in2size);

    double dot(double *in1, int in1index, int in1size, double *in2);
};
#endif //CUDAFUNCTION_HPP_INCLUDED__


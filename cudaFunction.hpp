#ifndef CUDAFUNCTION_HPP_INCLUDED__
#define CUDAFUNCTION_HPP_INCLUDED__

#include <iostream>
#include "times.hpp"

class cuda { 
  private:
    double *cu_d1, *cu_d2;//MV N, N
    double *cu_d3, *cu_h1;//dot blocks blocks
    double *cu_d4, *cu_d5, *cu_d6, *cu_d7, *cu_d8, *cu_h2, *cu_h3, *cu_h4;//kskipcg N*(2*k+1) N*(2*k+1) Blocks*(2*k) Blocks*(2*k+1) Blocks*(2*k+2) Blocks*(2*k) Blocks*(2*k+1) Blocks*(2*k+2)
    double *cu_d9, *cu_h5;//kskipbicg Blocks*(2*k+1) Blocks*(2*k+1)
    double *cu_d10, *cu_h6, *cu_d11, *cu_h7;//gmres vpgmres restart*N restart*N, blocks, blocks
    int size;
    int k;
    int restart;

  public:
    times *time;

    cuda(times *t);
    cuda(times *t, int size);
    cuda(times *t, int size, int k);
    cuda(times *t, int size, double restart);
    cuda(times *t, int size, int k, double restart);
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

    void MtxVec_mult(double *in, int inindex, int insize, double *out, double *val, int *col, int *ptr);

    double dot(double *in1, double *in2, int size);

    double dot(double *in1, double *in2);

    double dot(double *in1, int in1index, int in1size, double *in2, int in2index, int in2size);

    double dot(double *in1, double *in2, int in2index, int in2size);

    double dot(double *in1, int in1index, int in1size, double *in2);

    void Kskip_cg_bicg_base(double *Ar, double *Ap, double *rvec, double *pvec, const int kskip, double *val, int *col, int *ptr);

    void Kskip_cg_bicg_base2(double *Ar, double *Ap, double *rvec, double *pvec, const int kskip, double *val, int *col, int *ptr);

    void Kskip_cg_innerProduce(double *delta, double *eta, double *zeta, double *Ar, double *Ap, double *rvec, double *pvec, int kskip, double *val, int *col, int *ptr);

    void Kskip_cg_innerProduce2(double *delta, double *eta, double *zeta, double *Ar, double *Ap, double *rvec, double *pvec, int kskip, double *val, int *col, int *ptr);
    
    void Kskip_bicg_innerProduce(double *theta, double *eta, double *rho, double *phi, double *Ar, double *Ap, double *r_vec, double *p_vec, int kskip, double *val, int *col, int *ptr);

    void Kskip_bicg_innerProduce2(double *theta, double *eta, double *rho, double *phi, double *Ar, double *Ap, double *r_vec, double *p_vec, int kskip, double *val, int *col, int *ptr);

    void dot_gmres(double *wvec, double *vmtx, double *hmtx, int k, int N);
};
#endif //CUDAFUNCTION_HPP_INCLUDED__


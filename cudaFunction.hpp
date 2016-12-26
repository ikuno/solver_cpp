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
    double *cu_d100, *cu_d101; //test for gmres N, blocks
    double *cu_d200, *cu_d201, *cu_h8; // test  for gmres N*(r+1) B*(r+1) B*(r+1)


    double *cu_d1_1, *cu_d1_2, *cu_d2_1, *cu_d2_2;
    double *cu_d3_1, *cu_d3_2, *cu_h1_1, *cu_h1_2;

    unsigned long int size;
    unsigned long int size1, size2;
    int k;
    int restart;
    bool isMulti;

  public:
    times *time;

    cuda(times *t);
    cuda(times *t, unsigned long int size, unsigned long int size1 = 0, unsigned long int size2 = 0);
    cuda(times *t, unsigned long int size, int k, unsigned long int size1 = 0, unsigned long int size2 = 0);
    cuda(times *t, unsigned long int size, double restart, unsigned long int size1 = 0, unsigned long int size2 = 0);
    cuda(times *t, unsigned long int size, int k, double restart, unsigned long int size1 = 0, unsigned long int size2 = 0);
    ~cuda();

    //--------------------------------------
    // double* d_Malloc(unsigned long int size);
    double* d_Malloc(unsigned long int size, int DeviceNum = 0);
    // double* d_MallocHost(unsigned long int size);
    double* d_MallocHost(unsigned long int size, int DeviceNum = 0);

    // int* i_Malloc(unsigned long int size);
    int* i_Malloc(unsigned long int size, int DeviceNum = 0);
    // int* i_MallocHost(unsigned long int size);
    int* i_MallocHost(unsigned long int size, int DeviceNum = 0);

    // void H2D(double* from, double* to, unsigned long int size);
    // void D2H(double* from, double* to, unsigned long int size);
    //
    // void H2D(int* from, int* to, unsigned long int size);
    // void D2H(int* from, int* to, unsigned long int size);
    void H2D(double* from, double* to, unsigned long int size, bool timer = false, int DeviceNum = 0);
    void D2H(double* from, double* to, unsigned long int size, bool timer = false, int DeviceNum = 0);

    void H2D(int* from, int* to, unsigned long int size, bool timer = false, int DeviceNum = 0);
    void D2H(int* from, int* to, unsigned long int size, bool timer = false, int DeviceNum = 0);

    void Free(void* ptr);

    void FreeHost(void* ptr);

    // void Memset(double *ptr, double val, unsigned long int size);
    void Memset(double *ptr, double val, unsigned long int size, bool timer = false, int DeviceNum = 0);

    // void Memset(int *ptr, int val, unsigned long int size);
    void Memset(int *ptr, int val, unsigned long int size, bool timer = false, int DeviceNum = 0);

    void Reset(int DeviceNum);

    //--------------------------------------
    void CSR2CSC(double *dCSRval, int *dCSRcol, int *dCSRptr, double *CSCval, int *CSCrow, int *CSCptr, double *dCSCval, int *dCSCrow, int *dCSCptr, unsigned long int N, unsigned long int NNZ);

    void CSR2CSC(double *CSRval, int *CSRcol, int *CSRptr, double *CSCval, int *CSCrow, int *CSCptr, unsigned long int N, unsigned long int NNZ);

    void MtxVec_mult(double *in, double *out, unsigned long int size, double *val, int *col, int *ptr);

    void MtxVec_mult(double *in, double *out, double *val, int *col, int *ptr);

    void MtxVec_mult(double *in, unsigned long int inindex, unsigned long int insize, double *out, unsigned long int outindex, unsigned long int outsize, double *val, int *col, int *ptr);

    void MtxVec_mult(double *in, unsigned long int inindex, unsigned long int insize, double *out, double *val, int *col, int *ptr);

    double dot(double *in1, double *in2, unsigned long int size);

    double dot(double *in1, double *in2);

    double dot(double *in1, unsigned long int in1index, unsigned long int in1size, double *in2, unsigned long int in2index, unsigned long int in2size);

    double dot(double *in1, double *in2, unsigned long int in2index, unsigned long int in2size);

    double dot(double *in1, unsigned long int in1index, unsigned long int in1size, double *in2);

    void Kskip_cg_bicg_base(double *Ar, double *Ap, double *rvec, double *pvec, const int kskip, double *val, int *col, int *ptr);

    void Kskip_cg_bicg_base2(double *Ar, double *Ap, double *rvec, double *pvec, const int kskip, double *val, int *col, int *ptr);

    void Kskip_cg_innerProduce(double *delta, double *eta, double *zeta, double *Ar, double *Ap, double *rvec, double *pvec, int kskip, double *val, int *col, int *ptr);

    void Kskip_cg_innerProduce2(double *delta, double *eta, double *zeta, double *Ar, double *Ap, double *rvec, double *pvec, int kskip, double *val, int *col, int *ptr);

    void Kskip_cg_innerProduce3(double *delta, double *eta, double *zeta, double *Ar, double *Ap, double *rvec, double *pvec, int kskip, double *val, int *col, int *ptr);
    
    void Kskip_bicg_innerProduce(double *theta, double *eta, double *rho, double *phi, double *Ar, double *Ap, double *r_vec, double *p_vec, int kskip, double *val, int *col, int *ptr);

    void Kskip_bicg_innerProduce2(double *theta, double *eta, double *rho, double *phi, double *Ar, double *Ap, double *r_vec, double *p_vec, int kskip, double *val, int *col, int *ptr);

    void Kskip_bicg_innerProduce3(double *theta, double *eta, double *rho, double *phi, double *Ar, double *Ap, double *r_vec, double *p_vec, int kskip, double *val, int *col, int *ptr);

    void dot_gmres(double *wvec, double *vmtx, double *hmtx, int k, unsigned long int N);

    void dot_gmres2(double *wvec, double *vmtx, double *hmtx, int k, unsigned long int N);

    void dot_gmres3(double *wvec, double *vmtx, double *hmtx, int k, unsigned long int N);

    int GetDeviceNum();
    void ShowDevice();
    void MtxVec_mult_Multi(double *in, double *out, double *val1, int *col1, int *ptr1, double *val2, int *col2, int *ptr2);
    void SetSize_Multi(int size1, int size2);
    void EnableP2P();
};
#endif //CUDAFUNCTION_HPP_INCLUDED__


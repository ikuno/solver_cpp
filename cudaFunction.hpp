#ifndef CUDAFUNCTION_HPP_INCLUDED__
#define CUDAFUNCTION_HPP_INCLUDED__

#include <iostream>

class cuda { 
  private:
    public:
    int ThreadsinBlock;
    int BlocksinGrid;
    
    cuda();
    cuda(int t, int b);

    //--------------------------------------
    void SetCores(int t, int b);

    double* d_Malloc(int size);
    double* d_MallocHost(int size);

    int* i_Malloc(int size);
    int* i_MallocHost(int size);

    void H2D(double* from, double* to, int size);
    void D2H(double* from, double* to, int size);

    void H2D(int* from, int* to, int size);
    void D2H(int* from, int* to, int size);

    void Free(double* ptr);
    void FreeHost(double* ptr);

    void Free(int* ptr);
    void FreeHost(int* ptr);

    void Memset(double *ptr, double val, int size);

    void Memset(int *ptr, int val, int size);

    void Reset();

    //--------------------------------------
    void add(double* in1, double* in2, double* out, int size);

    void MtxVec_mult(double *in, double *out, int size, double *val, int *col, int *ptr);

    double dot(double *in1, double *in2, int size);
};
#endif //CUDAFUNCTION_HPP_INCLUDED__


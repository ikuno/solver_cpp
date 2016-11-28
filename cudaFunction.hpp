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

    float* f_Malloc(int size);
    float* f_MallocHost(int size);

    int* i_Malloc(int size);
    int* i_MallocHost(int size);

    void d_H2D(double* from, double* to, int size);
    void d_D2H(double* from, double* to, int size);

    void f_H2D(float* from, float* to, int size);
    void f_D2H(float* from, float* to, int size);

    void i_H2D(int* from, int* to, int size);
    void i_D2H(int* from, int* to, int size);

    void d_Free(double* ptr);
    void d_FreeHost(double* ptr);

    void f_Free(float* ptr);
    void f_FreeHost(float* ptr);

    void i_Free(int* ptr);
    void i_FreeHost(int* ptr);

    void d_add(double* in1, double* in2, double* out, int size);

    void f_add(float* in1, float* in2, float* out, int size);

    void d_MV(double *in, double *out, int size, double *val, int *col, int *ptr);

    void f_MV(float *in, float *out, int size, float *val, int *col, int *ptr);

    //--------------------------------------
    
};
#endif //CUDAFUNCTION_HPP_INCLUDED__


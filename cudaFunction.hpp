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


    void d_H2D(double* from, double* to, int size);
    void d_D2H(double* from, double* to, int size);

    void d_Free(double* ptr);
    void d_FreeHost(double* ptr);

    void i_Free(int* ptr);
    void i_FreeHost(int* ptr);

    void d_add(double* in1, double* in2, double* out, int size);

    //--------------------------------------
    
};
#endif //CUDAFUNCTION_HPP_INCLUDED__


#ifndef SOLVER_COLLECTION_HPP_INCLUDED__
#define SOLVER_COLLECTION_HPP_INCLUDED__

#include <iostream>
#include "cudaFunction.hpp"

enum SOLVERS_NAME {
    NONE,
    CG,
    BICG,
    CR,
    GCR,
    GMRES,

    KSKIPCG,
    KSKIPCR,//未実装
    KSKIPBICG,
    
    VPCG,
    VPBICG,//未実装
    VPCR,//BUG?
    VPGCR,
    VPGMRES
  };

class collection {
  private:
    cuda *cu;
  public:
    bool isVP;
    bool isCUDA;
    bool isVerbose;
    bool isInnerNow;
    bool isMixPrecision;
    bool isInnerKskip;
    bool isPinned;
    bool isMultiGPU;
    int OMPThread;
    int CUDADevice;

    std::string L1_Dir_Name;

    std::string CRS_Path;
    std::string CRS_Dir_Name;

    std::string MM_Path;
    std::string MM_Dir_Name;

    SOLVERS_NAME outerSolver;
    SOLVERS_NAME innerSolver;
    unsigned long int outerMaxLoop;
    unsigned long int innerMaxLoop;
    double outerEps;
    double innerEps;
    unsigned long int outerRestart;
    unsigned long int innerRestart;
    int outerKskip;
    int innerKskip;
    int outerFix;
    int innerFix;

    
//not init
    std::string CRS_Matrix_Name;
    std::string MM_Matrix_Name;
    std::string fullPath; 
    int inputSource; //1->CRS, 2->MM
    unsigned long int N; 
    unsigned long int NNZ;

    unsigned long int N1;
    unsigned long int N2;
    unsigned long int NNZ1;
    unsigned long int NNZ2;


    collection();
    ~collection();
    void readCMD(int argc, char* argv[]);
    void checkCMD();
    void showCMD();

    std::string enum2string(SOLVERS_NAME id);
    SOLVERS_NAME string2enum(std::string str);

    void checkCRSMatrix();
    void readMatrix();
    void CRSAlloc();
    void transposeMatrix();
    void transpose();
    void setOpenmpThread();

    void CudaCopy();
    void MultiGPUAlloc();

//pointer
    double* val;
    int* col;
    int* ptr;

    double* Tval;
    int* Tcol;
    int* Tptr;

    double* bvec;
    double* xvec;

    double* Cval;
    int* Ccol;
    int* Cptr;

    double* CTval;
    int* CTcol;
    int* CTptr;

    times *time;

//multi
    double* val1;
    int* col1;
    int* ptr1;

    double* val2;
    int* col2;
    int* ptr2;

    double* Cval1;
    int* Ccol1;
    int* Cptr1;

    double* Cval2;
    int* Ccol2;
    int* Cptr2;

};



#endif //SOLVER_COLLECTION_HPP_INCLUDED__


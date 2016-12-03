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
    int OMPThread;

    std::string L1_Dir_Name;

    std::string CRS_Path;
    std::string CRS_Dir_Name;

    std::string MM_Path;
    std::string MM_Dir_Name;

    SOLVERS_NAME outerSolver;
    SOLVERS_NAME innerSolver;
    long int outerMaxLoop;
    long int innerMaxLoop;
    double outerEps;
    double innerEps;
    long int outerRestart;
    long int innerRestart;
    int outerKskip;
    int innerKskip;
    int outerFix;
    int innerFix;

    
//not init
    std::string CRS_Matrix_Name;
    std::string MM_Matrix_Name;
    std::string fullPath; 
    long int N; 
    long int NNZ;
    int inputSource; //1->CRS, 2->MM


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

};



#endif //SOLVER_COLLECTION_HPP_INCLUDED__


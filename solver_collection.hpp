#ifndef SOLVER_COLLECTION_HPP_INCLUDED__
#define SOLVER_COLLECTION_HPP_INCLUDED__

#include <iostream>

enum SOLVERS_NAME {
    NONE,
    CG,
    BICG,
    CR,
    GCR,
    GMRES,

    KSKIPCG,
    KSKIPCR,
    KSKIPBICG,
    
    VPCG,
    VPBICG,
    VPCR,
    VPGCR,
    VPGMRES
  };

class collection {
  private:
    bool isVP;
    bool isCUDA;
    bool isOpenMP;
    bool isVerbose;
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

//pointer
    double* val;
    int *col;
    int *ptr;

  public:
    collection();
    void readCMD(int argc, char* argv[]);
    void checkCMD();
    void showCMD();

    std::string enum2string(SOLVERS_NAME id);
    SOLVERS_NAME string2enum(std::string str);

    void CRSMalloc();

};

#endif //SOLVER_COLLECTION_HPP_INCLUDED__


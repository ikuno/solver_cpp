#include <iostream>
#include "outerMethods.hpp"
#include "solver_collection.hpp"

//0->float 1->double
#define TYPE_SELECT 1

int main(int argc, char* argv[])
{
  if(TYPE_SELECT==0){
    collection<float> col;
    outerMethods<float> method(&col);
    col.readCMD(argc, argv);
    col.checkCMD();
    col.checkMatrix();
    col.CRSAlloc();
    col.readMatrix();
    col.transposeMatrix();
    col.setOpenmpThread();

    col.showCMD();
    method.outerSelect(col.outerSolver);
    if(col.isVerbose)
      col.showCMD();

  }else{
    collection<double> col;
    outerMethods<double> method(&col);
    col.readCMD(argc, argv);
    col.checkCMD();
    col.checkMatrix();
    col.CRSAlloc();
    col.readMatrix();
    col.transposeMatrix();
    col.setOpenmpThread();

    col.showCMD();
    method.outerSelect(col.outerSolver);
    if(col.isVerbose)
      col.showCMD();

  }

  return 0;
}

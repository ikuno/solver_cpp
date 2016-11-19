#include <iostream>
#include "solver_collection.hpp"
#include "methods.hpp"

//0->float 1->double
#define TYPE_SELECT 1

int main(int argc, char* argv[])
{
  if(TYPE_SELECT==0){
    collection<float> col;
    methods<float> method(&col);
    col.readCMD(argc, argv);
    col.checkCMD();
    col.checkMatrix();
    col.CRSAlloc();
    col.readMatrix();
    col.transposeMatrix();

    col.showCMD();
    method.outerSelect(col.outerSolver);
    if(col.isVerbose)
      col.showCMD();

  }else{
    collection<double> col;
    methods<double> method(&col);
    col.readCMD(argc, argv);
    col.checkCMD();
    col.checkMatrix();
    col.CRSAlloc();
    col.readMatrix();
    col.transposeMatrix();

    col.showCMD();
    method.outerSelect(col.outerSolver);
    if(col.isVerbose)
      col.showCMD();

  }

  return 0;
}

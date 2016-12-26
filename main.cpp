#include <iostream>
#include <cmath>
#include "outerMethods.hpp"
#include "solver_collection.hpp"

int main(int argc, char* argv[])
{
  collection col;
  outerMethods method(&col);

  col.readCMD(argc, argv);
  col.checkCMD();
  col.checkCRSMatrix();
  col.CRSAlloc();
  col.readMatrix();
  if(col.isMultiGPU){
    col.MultiGPUAlloc();
  }
  col.CudaCopy();
  col.transposeMatrix();
  col.setOpenmpThread();

  col.showCMD();
  method.outerSelect(col.outerSolver);

  if(col.isVerbose)
    col.showCMD();

  return 0;
}

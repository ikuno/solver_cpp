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
  col.CRSAlloc_Part1();
  col.readMatrix();
  col.CRSAlloc_Part2();
  col.CudaCopy_Part1();
  col.transposeMatrix();
  col.make2GPUMatrix();
  col.CudaCopy_Part2();
  col.setOpenmpThread();

  col.showCMD();
  method.outerSelect(col.outerSolver);

  if(col.isVerbose)
    col.showCMD();

  return 0;
}

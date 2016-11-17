#include <iostream>
#include "solver_collection.hpp"
#include "methods.hpp"

int main(int argc, char* argv[])
{
  collection col;
  methods method(&col);

  col.readCMD(argc, argv);
  col.checkCMD();
  col.checkMatrix();
  col.CRSAlloc();
  col.readMatrix();

  col.showCMD();
  method.outerSelect(col.outerSolver);
  col.showCMD();


  return 0;
}

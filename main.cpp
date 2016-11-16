#include <iostream>
#include "solver_collection.hpp"

int main(int argc, char* argv[])
{
  collection col;
  col.readCMD(argc, argv);
  col.checkCMD();
  col.showCMD();


  return 0;
}

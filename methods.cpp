#include "methods.hpp"

methods::methods(collection *coll){
  this->coll = coll;
}


void methods::outerSelect(SOLVERS_NAME solver){
  int result;
  if(solver == CG){
    cg solver(coll, coll->bvec, coll->xvec);
    result = solver.solve();
  }
  if(result == 0){
    std::cout << "converge" << std::endl;
  }else if(result == 2){
    std::cout << "not converge" << std::endl;
  }else{
    std::cerr << "error in methods" << std::endl;
  }
}

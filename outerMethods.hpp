#ifndef OUTERMETHODS_HPP_INCLUDED__
#define OUTERMETHODS_HPP_INCLUDED__

#include "solver_collection.hpp"
#include "cg.hpp"
#include "cr.hpp"
#include "gcr.hpp"
#include "bicg.hpp"
#include "gmres.hpp"
#include "kskipcg.hpp"
#include "kskipbicg.hpp"
#include "vpcg.hpp"
#include "vpcr.hpp"
#include "vpgcr.hpp"
#include "vpgmres.hpp"

template <typename T>
class outerMethods {
  private:
    collection<T> *coll;
  public:
    outerMethods(collection<T> *coll);
    void outerSelect(SOLVERS_NAME solver);
};

template <typename T>
outerMethods<T>::outerMethods(collection<T> *coll){
  this->coll = coll;
}

template <typename T>
void outerMethods<T>::outerSelect(SOLVERS_NAME solver){
  int result = 99;
  if(solver == CG){
    cg<T> outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }else if(solver == CR){
    cr<T> outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }else if(solver == GCR){
    gcr<T> outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }else if(solver == BICG){
    bicg<T> outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }else if(solver == GMRES){
    gmres<T> outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }else if(solver == KSKIPCG){
    kskipcg<T> outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }else if(solver == KSKIPCR){
    std::cout << RED << "Not implemented" << RESET << std::endl;
    result = 99;
  }else if(solver == KSKIPBICG){
    kskipBicg<T> outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }else if(solver == VPCG){
    vpcg<T> outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }else if(solver == VPBICG){
    std::cout << RED << "Not implemented" << RESET << std::endl;
    result = 99;
  }else if(solver == VPCR){
    vpcr<T> outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }else if(solver == VPGCR){
    vpgcr<T> outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }else if(solver == VPGMRES){
    vpgmres<T> outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }
  
  if(result == 0){
    std::cout << BOLDGREEN << "converge" << RESET << std::endl;
  }else if(result == 2){
    std::cout << BOLDRED << "not converge" << RESET << std::endl;
  }else{
    std::cerr << RED << "outer result -> "<< result << RESET << std::endl;
    std::cerr << RED << "[X]Some Error in methods" << RESET << std::endl;
  }
}
#endif //OUTERMETHODS_HPP_INCLUDED__


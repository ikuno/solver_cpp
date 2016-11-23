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
    cg<T> solver(coll, coll->bvec, coll->xvec, false);
    result = solver.solve();
  }else if(solver == CR){
    cr<T> solver(coll, coll->bvec, coll->xvec, false);
    result = solver.solve();
  }else if(solver == GCR){
    gcr<T> solver(coll, coll->bvec, coll->xvec, false);
    result = solver.solve();
  }else if(solver == BICG){
    bicg<T> solver(coll, coll->bvec, coll->xvec, false);
    result = solver.solve();
  }else if(solver == GMRES){
    gmres<T> solver(coll, coll->bvec, coll->xvec, false);
    result = solver.solve();
  }else if(solver == KSKIPCG){
    kskipcg<T> solver(coll, coll->bvec, coll->xvec, false);
    result = solver.solve();
  }else if(solver == KSKIPCR){
    std::cout << RED << "Not implemented" << RESET << std::endl;
    result = 99;
  }else if(solver == KSKIPBICG){
    kskipBicg<T> solver(coll, coll->bvec, coll->xvec, false);
    result = solver.solve();
  }else if(solver == VPCG){
    vpcg<T> solver(coll, coll->bvec, coll->xvec, false);
    result = solver.solve();
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


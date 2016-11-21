#ifndef METHODS_HPP_INCLUDED__
#define METHODS_HPP_INCLUDED__

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
class methods {
  private:
    collection<T> *coll;
  public:
    methods(collection<T> *coll);
    void outerSelect(SOLVERS_NAME solver);
    void innerSelect(SOLVERS_NAME solver, T *ibvec, T *ixvec);
};

template <typename T>
methods<T>::methods(collection<T> *coll){
  this->coll = coll;
}

template <typename T>
void methods<T>::outerSelect(SOLVERS_NAME solver){
  int result = 1;
  if(solver == CG){
    cg<T> solver(coll, coll->bvec, coll->xvec);
    result = solver.solve();
  }else if(solver == CR){
    cr<T> solver(coll, coll->bvec, coll->xvec);
    result = solver.solve();
  }else if(solver == GCR){
    gcr<T> solver(coll, coll->bvec, coll->xvec);
    result = solver.solve();
  }else if(solver == BICG){
    bicg<T> solver(coll, coll->bvec, coll->xvec);
    result = solver.solve();
  }else if(solver == GMRES){
    gmres<T> solver(coll, coll->bvec, coll->xvec);
    result = solver.solve();
  }else if(solver == KSKIPCG){
    kskipcg<T> solver(coll, coll->bvec, coll->xvec);
    result = solver.solve();
  }else if(solver == KSKIPCR){
    std::cout << RED << "Not implemented" << RESET << std::endl;
    result = 1;
  }else if(solver == KSKIPBICG){
    kskipBicg<T> solver(coll, coll->bvec, coll->xvec);
    result = solver.solve();
  }else if(solver == VPCG){
    // vpcg<T> solver(coll, coll->bvec, coll->xvec);
    // result = solver.solve();
  }
  
  if(result == 0){
    std::cout << BOLDGREEN << "converge" << RESET << std::endl;
  }else if(result == 2){
    std::cout << BOLDRED << "not converge" << RESET << std::endl;
  }else{
    std::cerr << RED << "[X]Some Error in methods" << RESET << std::endl;
  }
}

template <typename T>
void methods<T>::innerSelect(SOLVERS_NAME solver, T *ibvec, T *ixvec){
  int result = 1;
  if(solver == CG){
    cg<T> solver(coll, ibvec, ixvec);
    result = solver.solve();
  }else if(solver == CR){
    cr<T> solver(coll, ibvec, ixvec);
    result = solver.solve();
  }else if(solver == GCR){
    gcr<T> solver(coll, ibvec, ixvec);
    result = solver.solve();
  }else if(solver == BICG){
    bicg<T> solver(coll, ibvec, ixvec);
    result = solver.solve();
  }else if(solver == GMRES){
    gmres<T> solver(coll, ibvec, ixvec);
    result = solver.solve();
  }else if(solver == KSKIPCG){
    kskipcg<T> solver(coll, ibvec, ixvec);
    result = solver.solve();
  }else if(solver == KSKIPCR){
    std::cout << RED << "Not implemented" << RESET << std::endl;
    result = 1;
  }else if(solver == KSKIPBICG){
    kskipBicg<T> solver(coll, ibvec, ixvec);
    result = solver.solve();
  }
  
  if(result != 0 || result != 2){
    std::cerr << RED << "[X]Some Error in methods" << RESET << std::endl;
  }
}

#endif //METHODS_HPP_INCLUDED__

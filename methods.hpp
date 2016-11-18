#ifndef METHODS_HPP_INCLUDED__
#define METHODS_HPP_INCLUDED__

#include "solver_collection.hpp"
#include "cg.hpp"

template <typename T>
class methods {
  private:
    collection<T> *coll;
  public:
    methods(collection<T> *coll);
    void outerSelect(SOLVERS_NAME solver);
};

template <typename T>
methods<T>::methods(collection<T> *coll){
  this->coll = coll;
}

template <typename T>
void methods<T>::outerSelect(SOLVERS_NAME solver){
  int result;
  if(solver == CG){
    cg<T> solver(coll, coll->bvec, coll->xvec);
    result = solver.solve();
  }
  if(result == 0){
    std::cout << BOLDGREEN << "converge" << RESET << std::endl;
  }else if(result == 2){
    std::cout << BOLDRED << "not converge" << RESET << std::endl;
  }else{
    std::cerr << RED << "[X]Some Error in methods" << RESET << std::endl;
  }
}
#endif //METHODS_HPP_INCLUDED__

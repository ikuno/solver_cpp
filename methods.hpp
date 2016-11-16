#ifndef METHODS_HPP_INCLUDED__
#define METHODS_HPP_INCLUDED__

#include "solver_collection.hpp"

class methods {
  private:
    collection *col;
  public:
    methods(collection *col);
    void outerSelect(SOLVERS_NAME solver, collection *col);
};
#endif //METHODS_HPP_INCLUDED__

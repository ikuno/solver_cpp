#ifndef METHODS_HPP_INCLUDED__
#define METHODS_HPP_INCLUDED__

#include "solver_collection.hpp"
#include "cg.hpp"

class methods {
  private:
    collection *coll;
  public:
    methods(collection *coll);
    void outerSelect(SOLVERS_NAME solver);
};
#endif //METHODS_HPP_INCLUDED__

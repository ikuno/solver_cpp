#ifndef OUTERMETHODS_HPP_INCLUDED__
#define OUTERMETHODS_HPP_INCLUDED__

#include "solver_collection.hpp"

class outerMethods {
  private:
    collection *coll;
  public:
    outerMethods(collection *coll);
    void outerSelect(SOLVERS_NAME solver);
};

#endif //OUTERMETHODS_HPP_INCLUDED__


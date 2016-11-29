#ifndef INNERMETHODS_HPP_INCLUDED__
#define INNERMETHODS_HPP_INCLUDED__

#include "solver_collection.hpp"

class innerMethods {
  public:
    collection *Ccoll;
    innerMethods(collection *coll);
    void innerSelect(collection *hoge, SOLVERS_NAME solver, double *ibvec, double *ixvec);
};

#endif //INNERMETHODS_HPP_INCLUDED__


#ifndef INNERMETHODS_HPP_INCLUDED__
#define INNERMETHODS_HPP_INCLUDED__

#include "solver_collection.hpp"
#include "cudaFunction.hpp"
#include "blas.hpp"

class innerMethods {
  public:
    collection *Ccoll;
    innerMethods(collection *coll, cuda *cu, blas *bs);
    void innerSelect(collection *innercoll, SOLVERS_NAME solver, cuda *cu, blas *bs, double *ibvec, double *ixvec);
    void innerSelect(collection *innercoll, SOLVERS_NAME solver, cuda *cu, blas *bs, double *ibvec, double *ixvec, double **list);

};

#endif //INNERMETHODS_HPP_INCLUDED__


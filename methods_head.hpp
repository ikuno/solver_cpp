#ifndef METHODS_HEAD_HPP_INCLUDED__
#define METHODS_HEAD_HPP_INCLUDED__

#include "solver_collection.hpp"
#include "cg.hpp"
#include "cr.hpp"
#include "gcr.hpp"
#include "bicg.hpp"
#include "gmres.hpp"
#include "kskipcg.hpp"
#include "kskipbicg.hpp"
// #include "vpcg_head.hpp"

template <typename T>
class methods {
  private:
    collection<T> *coll;
  public:
    methods(collection<T> *coll);
    void outerSelect(SOLVERS_NAME solver);
    void innerSelect(SOLVERS_NAME solver, T *ibvec, T *ixvec);
};

#endif //METHODS_HEAD_HPP_INCLUDED__


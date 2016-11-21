#ifndef VPCG_HEAD_HPP_INCLUDED__
#define VPCG_HEAD_HPP_INCLUDED__

#include <iomanip>
#include <fstream>
#include "solver_collection.hpp"
// #include "methods.hpp"
#include "methods_head.hpp"
#include "blas.hpp"
#include "cg.hpp"

template <typename T>
class vpcg {
  private:
    collection<T> *coll;
    blas<T> *bs;
    methods<T> *innerMethod;
    
    long int loop;
    T *xvec, *bvec;
    T *rvec, *pvec, *mv, *x_0, dot, error;
    T *zvec;
    T alpha, beta, bnorm, rnorm;
    T rr, rr2;

    int maxloop;
    double eps;
    bool isVP, isVerbose, isCUDA, isInner;

    int exit_flag;
    T test_error;

    int N;

    std::ofstream f_his;
    std::ofstream f_x;

  public:
    vpcg(collection<T> *coll, T *bvec, T *xvec);
    ~vpcg();
    int solve();
};

#endif //VPCG_HEAD_HPP_INCLUDED__


#ifndef BICG_HPP_INCLUDED__
#define BICG_HPP_INCLUDED__

#include <fstream>
#include "solver_collection.hpp"
#include "times.hpp"
#include "blas.hpp"
#include "cudaFunction.hpp"

class bicg {
  private:
    collection *coll;
    blas *bs;
    cuda *cu;
    times time;

    long int loop;
    double *xvec, *bvec;
    double *rvec, *r_vec, *pvec, *p_vec, *mv, *x_0, dot, error;
    double alpha, beta, bnorm, rnorm;
    double rr, rr2;

    int maxloop;
    double eps;
    bool isVP, isVerbose, isCUDA, isInner;

    int exit_flag;
    double test_error;

    int N;

    std::ofstream f_his;
    std::ofstream f_x;

  public:

    bicg(collection *coll, double *bvec, double *xvec, bool inner = false, cuda *a_cu = NULL, blas *a_bs = NULL);
    ~bicg();
    int solve();
};


#endif //BICG_HPP_INCLUDED__


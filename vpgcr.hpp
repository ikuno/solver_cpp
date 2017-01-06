#ifndef VPGCR_HPP_INCLUDED__
#define VPGCR_HPP_INCLUDED__

#include <fstream>
#include "solver_collection.hpp"
#include "times.hpp"
#include "blas.hpp"
#include "innerMethods.hpp"
#include "cudaFunction.hpp"

class vpgcr {
  private:
    collection *coll;
    blas *bs;
    cuda *cu;
    innerMethods *in;
    times time;

    long int loop, iloop, kloop;
    double *xvec, *bvec;
    double *rvec, *zvec, *Av, *x_0, *qq, *beta_vec;
    double *qvec, *pvec;
    double dot_tmp, error;
    double alpha, beta, bnorm, rnorm;

    int maxloop;
    double eps;
    int restart;
    bool isVP, isVerbose, isCUDA, isInner, isPinned, isMultiGPU;

    int exit_flag;
    double test_error;
    bool out_flag;

    int N;

    std::ofstream f_his;
    std::ofstream f_x;

  public:
    vpgcr(collection *coll, double *bvec, double *xvec, bool inner = false);
    ~vpgcr();
    int solve();
};

#endif //VPGCR_HPP_INCLUDED__


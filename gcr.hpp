#ifndef gcr_HPP_INCLUDED__
#define gcr_HPP_INCLUDED__

#include <fstream>
#include "solver_collection.hpp"
#include "blas.hpp"
#include "times.hpp"

class gcr {
  private:
    collection *coll;
    blas *bs;
    cuda *cu;
    times time;

    long int loop, iloop, kloop;
    double *xvec, *bvec;
    double *rvec, *Av, *x_0, *qq, *beta_vec;
    double *qvec, *pvec;
    double dot, dot_tmp, error;
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
    std::ofstream f_in;

  public:
    // gcr(collection *coll, double *bvec, double *xvec, bool inner = false, cuda *a_cu = NULL, blas *a_bs = NULL);
    gcr(collection *coll, double *bvec, double *xvec, bool inner = false, cuda *a_cu = NULL, blas *a_bs = NULL, double **list = NULL);
    ~gcr();
    int solve();
};

#endif //gcr_HPP_INCLUDED__


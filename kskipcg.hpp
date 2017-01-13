#ifndef KSKIPCG_HPP_INCLUDED__
#define KSKIPCG_HPP_INCLUDED__

#include <fstream>
#include "solver_collection.hpp"
#include "blas.hpp"
#include "times.hpp"
#include "cudaFunction.hpp"

class kskipcg {
  private:
    collection *coll;
    blas *bs;
    cuda *cu;
    times time;

    long int nloop, iloop, jloop;
    double *xvec, *bvec;
    double *rvec, *pvec, *Av, *x_0, error;
    double *delta, *eta, *zeta;
    double *Ap, *Ar;
    double alpha, beta, gamma, bnorm, rnorm;

    int maxloop;
    double eps;
    bool isVP, isVerbose, isCUDA, isInner, isPinned, isMultiGPU;
    int kskip;
    int fix;

    int exit_flag;
    double test_error;

    int N;

    std::ofstream f_his;
    std::ofstream f_x;
    std::ofstream f_in;

  public:
    // kskipcg(collection *coll, double *bvec, double *xvec, bool inner = false, cuda *a_cu = NULL, blas *a_bs = NULL);
    kskipcg(collection *coll, double *bvec, double *xvec, bool inner = false, cuda *a_cu = NULL, blas *a_bs = NULL, double **list = NULL);
    ~kskipcg();
    int solve();
};


#endif //KSKIPCG_HPP_INCLUDED__


#ifndef KSKIPBICG_HPP_INCLUDED__
#define KSKIPBICG_HPP_INCLUDED__

#include <fstream>
#include "solver_collection.hpp"
#include "blas.hpp"
#include "times.hpp"
#include "cudaFunction.hpp"

class kskipBicg {
  private:
    collection *coll;
    blas *bs;
    cuda *cu;
    times time;

    long int nloop, iloop, jloop;
    double *xvec, *bvec;
    double *rvec, *r_vec, *pvec, *p_vec, *Av, *x_0, error;
    double *theta, *eta, *rho, *phi, bnorm, rnorm, alpha, beta, gamma;
    double *Ap, *Ar;

    int maxloop;
    double eps;
    bool isVP, isVerbose, isCUDA, isInner;
    int kskip;

    int exit_flag;
    double test_error;

    int N;

    std::ofstream f_his;
    std::ofstream f_x;

  public:
    kskipBicg(collection *coll, double *bvec, double *xvec, bool inner = false, cuda *a_cu = NULL, blas *a_bs = NULL);
    ~kskipBicg();
    int solve();
};


#endif //KSKIPBICG_HPP_INCLUDED__


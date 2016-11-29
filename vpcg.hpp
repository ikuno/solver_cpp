#ifndef VPCG_HPP_INCLUDED__
#define VPCG_HPP_INCLUDED__

#include <fstream>
#include "solver_collection.hpp"
#include "times.hpp"
#include "blas.hpp"
#include "innerMethods.hpp"

class vpcg {
  private:
    collection *coll;
    blas *bs;
    innerMethods *in;
    times time;

    long int loop;
    double *xvec, *bvec;
    double *rvec, *pvec, *mv, *x_0, dot, error;
    double *zvec;
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
    vpcg(collection *coll, double *bvec, double *xvec, bool inner);
    ~vpcg();
    int solve();
};


#endif //VPCG_HPP_INCLUDED__


#ifndef VPCR_HPP_INCLUDED__
#define VPCR_HPP_INCLUDED__

#include <fstream>
#include "solver_collection.hpp"
#include "times.hpp"
#include "blas.hpp"
#include "innerMethods.hpp"

class vpcr {
  private:
    collection *coll;
    blas *bs;
    innerMethods *in;
    times time;
    
    long int loop;
    double *xvec, *bvec;
    double *rvec, *pvec, *zvec, *Av, *Ap, *x_0, error;
    double alpha, beta, bnorm, rnorm;
    double zaz, zaz2;
    double tmp;

    int maxloop;
    double eps;
    bool isVP, isVerbose, isCUDA, isInner;

    int exit_flag;
    double test_error;

    int N;

    std::ofstream f_his;
    std::ofstream f_x;

  public:
    vpcr(collection *coll, double *bvec, double *xvec, bool inner);
    ~vpcr();
    int solve();
};


#endif //VPCR_HPP_INCLUDED__


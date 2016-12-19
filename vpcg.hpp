#ifndef VPCG_HPP_INCLUDED__
#define VPCG_HPP_INCLUDED__

#include <fstream>
#include "solver_collection.hpp"
#include "times.hpp"
#include "blas.hpp"
#include "innerMethods.hpp"
#include "cudaFunction.hpp"

class vpcg {
  private:
    collection *coll;
    blas *bs;
    cuda *cu;
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
    bool isVP, isVerbose, isCUDA, isInner, isPinned;

    int exit_flag;
    double test_error;

    int N;

    std::ofstream f_his;
    std::ofstream f_x;

  public:
    vpcg(collection *coll, double *bvec, double *xvec, bool inner = false);
    ~vpcg();
    int solve();
};


#endif //VPCG_HPP_INCLUDED__


#ifndef CR_HPP_INCLUDED__
#define CR_HPP_INCLUDED__

#include <fstream>
#include "solver_collection.hpp"
#include "blas.hpp"
#include "times.hpp"
#include "cudaFunction.hpp"
#include "times.hpp"

class cr {
  private:
    collection *coll;
    blas *bs;
    cuda *cu;
    times time;

    long int loop;
    double *xvec, *bvec;
    double *rvec, *pvec, *qvec, *svec, *x_0, dot, error;
    double alpha, beta, bnorm, rnorm;
    double rs, rs2;

    int maxloop;
    double eps;
    bool isVP, isVerbose, isCUDA, isInner;

    int exit_flag;
    double test_error;

    int N;

    std::ofstream f_his;
    std::ofstream f_x;

  public:
    cr(collection *coll, double *bvec, double *xvec, bool inner = false, cuda *a_cu = NULL, blas *a_bs = NULL);
    ~cr();
    int solve();
};


#endif //CR_HPP_INCLUDED__


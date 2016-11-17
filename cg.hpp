#ifndef CG_HPP_INCLUDED__
#define CG_HPP_INCLUDED__

#include <iomanip>
#include <fstream>
#include "solver_collection.hpp"
#include "blas.hpp"

class cg {
  private:
    collection *coll;
    blas *bs;
    
    long int loop;
    double *xvec, *bvec;
    double *rvec, *pvec, *mv, *x_0, dot, error;
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
    cg(collection *coll, double *bvec, double *xvec);
    ~cg();
    int solve();
};


#endif //CG_HPP_INCLUDED__


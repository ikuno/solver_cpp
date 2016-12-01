#ifndef GMRES_HPP_INCLUDED__
#define GMRES_HPP_INCLUDED__

#include <fstream>
#include "solver_collection.hpp"
#include "blas.hpp"
#include "times.hpp"
#include "cudaFunction.hpp"

class gmres {
  private:
    collection *coll;
    blas *bs;
    cuda *cu;

    times time;
    
    long int loop;
    double *xvec, *bvec;
    double *rvec, *axvec, *evec, *vvec, *vmtx, *hmtx, *yvec, *wvec, *avvec, *hvvec, *cvec, *svec, *x0vec, *tmpvec, *x_0;
    double wv_ip, error;
    double alpha, bnorm;
    double tmp, tmp2, rr2;
    long int count;

    int maxloop;
    double eps;
    bool isVP, isVerbose, isCUDA, isInner;
    int restart;

    int exit_flag, over_flag;
    double test_error;

    int N;

    std::ofstream f_his;
    std::ofstream f_x;

  public:
    gmres(collection *coll, double *bvec, double *xvec, bool inner);
    ~gmres();
    int solve();
};


#endif //GMRES_HPP_INCLUDED__


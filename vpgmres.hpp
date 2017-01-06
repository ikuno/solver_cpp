#ifndef VPGMRES_HPP_INCLUDED__
#define VPGMRES_HPP_INCLUDED__

#include <fstream>
#include "solver_collection.hpp"
#include "times.hpp"
#include "blas.hpp"
#include "innerMethods.hpp"
#include "cudaFunction.hpp"

class vpgmres {
  private:
    collection *coll;
    blas *bs;
    cuda *cu;
    innerMethods *in;
    times time;
    
    long int loop;
    double *xvec, *bvec;
    double *rvec, *axvec, *evec, *vvec, *vmtx, *hmtx, *yvec, *wvec, *avvec, *cvec, *svec, *x0vec, *tmpvec, *zmtx, *zvec, *x_0, *Av, *testvec, *testvec2;
    double wv_ip, error;
    double alpha, bnorm;
    double tmp, tmp2, rr2;
    long int count;

    int maxloop;
    double eps;
    bool isVP, isVerbose, isCUDA, isInner, isPinned, isMultiGPU;
    int restart;

    int exit_flag, over_flag;
    double test_error;

    int N;

    std::ofstream f_his;
    std::ofstream f_x;

  public:
    vpgmres(collection *coll, double *bvec, double *xvec, bool inner = false);
    ~vpgmres();
    int solve();
};

#endif //VPGMRES_HPP_INCLUDED__


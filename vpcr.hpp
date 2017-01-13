#ifndef VPCR_HPP_INCLUDED__
#define VPCR_HPP_INCLUDED__

#include <fstream>
#include "solver_collection.hpp"
#include "times.hpp"
#include "blas.hpp"
#include "innerMethods.hpp"
#include "cudaFunction.hpp"

class vpcr {
  private:
    collection *coll;
    blas *bs;
    cuda *cu;
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
    bool isVP, isVerbose, isCUDA, isInner, isPinned, isMultiGPU;

    int exit_flag;
    double test_error;

    int N;

    std::ofstream f_his;
    std::ofstream f_x;

    //
    double **list;
    //cg
    double *cg_rvec, *cg_pvec, *cg_mv, *cg_x_0;
    //cr
    double *cr_rvec, *cr_pvec, *cr_qvec, *cr_svec, *cr_x_0;
    //gcr
    double *gcr_rvec, *gcr_Av, *gcr_x_0, *gcr_qq, *gcr_qvec, *gcr_pvec, *gcr_beta_vec;
    //bicg
    double *bicg_rvec, *bicg_pvec, *bicg_r_vec, *bicg_p_vec, *bicg_mv, *bicg_x_0;
    //gmres
    double *gm_Av, *gm_rvec, *gm_evec, *gm_vvec, *gm_vmtx, *gm_hmtx, *gm_yvec, *gm_wvec, *gm_cvec, *gm_svec, *gm_x0vec, *gm_tmpvec, *gm_x_0, *gm_testvec, *gm_testvec2;
    //kskipcg
    double *kcg_rvec, *kcg_pvec, *kcg_Av, *kcg_x_0, *kcg_delta, *kcg_eta, *kcg_zeta, *kcg_Ar, *kcg_Ap;
    //kskipbicg
    double *kbicg_rvec, *kbicg_pvec, *kbicg_r_vec, *kbicg_p_vec, *kbicg_Av, *kbicg_x_0, *kbicg_theta, *kbicg_eta, *kbicg_rho, *kbicg_phi, *kbicg_Ar, *kbicg_Ap;


  public:
    vpcr(collection *coll, double *bvec, double *xvec, bool inner = false);
    ~vpcr();
    int solve();
};


#endif //VPCR_HPP_INCLUDED__


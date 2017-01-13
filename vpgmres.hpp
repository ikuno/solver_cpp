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
    vpgmres(collection *coll, double *bvec, double *xvec, bool inner = false);
    ~vpgmres();
    int solve();
};

#endif //VPGMRES_HPP_INCLUDED__


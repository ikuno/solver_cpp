#ifndef BLAS_HPP_INCLUDED__
#define BLAS_HPP_INCLUDED__

#include "solver_collection.hpp"
#include "cudaFunction.hpp"

class blas {
  private:
    collection *coll;
    bool mix;
  public:
    blas(collection *col);
    ~blas();

    double norm_1(double *v);

    double norm_2(double *v);

    void MtxVec_mult(double *in_vec, double *out_vec);

    void MtxVec_mult(double *in_vec, int xindex, int xsize, double *out_vec);

    void MtxVec_mult(double *Tval, int *Tcol, int *Tptr, double *in_vec, double *out_vec);

    void Vec_sub(double *x, double *y, double *out);

    void Vec_add(double *x, double *y, double *out);

    double dot(double *x, double *y);

    double dot(double *x, double *y, const long int size);

    double dot(double *x, double *y, int xindex, int xsize);

    void Scalar_ax(double a, double *x, double *out);

    void Scalar_ax(double a, double *x, int xindex, int xsize, double *out);

    void Scalar_axy(double a, double *x, double *y, double *out);

    void Scalar_x_div_a(double *x, double a, double *out);

    double Check_error(double *x_last, double *x_0);

    void Hye(double *h, double *y, double *e, const long int size);

    void Vec_copy(double *in, double *out);

    void Vec_copy(double *in, int xindex, int xsize, double *out);

    void Vec_copy(double *in, double *out, int xindex, int xsize);

    //-----------------------------------

    void Kskip_cg_base(double **Ar, double **Ap, double *rvec, double *pvec, const int kskip);

    void Kskip_cg_innerProduce(double *delta, double *eta, double *zeta, double **Ar, double **Ap, double *rvec, double *pvec, const int kskip);

    void Kskip_kskipBicg_base(double **Ar, double **Ap, double *rvec, double *pvec, const int kskip);

    void Kskip_kskipBicg_innerProduce(double *theta, double *eta, double *rho, double *phi, double **Ar, double **Ap, double *rvec, double *pvec, double *r_vec, double *p_vec, const int kskip);

    void Gmres_sp_1(int k, double *x, double *y, double *out);

};




#endif //BLAS_HPP_INCLUDED__


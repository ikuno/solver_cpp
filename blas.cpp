#include <fstream>
#include <ctime>
#include <cstring>
#include <cmath>
#include <typeinfo>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "blas.hpp"

blas::blas(collection *col, times *t){

  coll = col;
#ifdef _OPENMP
  omp_set_num_threads(this->coll->OMPThread);
#endif
  this->time = t;
}

blas::~blas(){
}

double blas::norm_1(double *v){
  double tmp = 0.0;
  for(unsigned long int i=0;i<this->coll->N;i++){
    tmp += fabs(v[i]);
  }
  return tmp;
}

double blas::norm_2(double *v){
  double tmp = 0.0;
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
#endif
  for(unsigned long int i=0;i<this->coll->N;i++){
    tmp += v[i] * v[i];
  }
  tmp = sqrt(tmp);
  return tmp;
}

void blas::MtxVec_mult(double *in_vec, double *out_vec){
  double tmp = 0.0;

  double *val=this->coll->val;
  int *ptr=this->coll->ptr;
  int *col=this->coll->col;

  unsigned long int N = this->coll->N;

  this->time->start();
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+:tmp) schedule(static) firstprivate(out_vec, val, in_vec) lastprivate(out_vec) num_threads(this->coll->OMPThread)
#endif
  for(unsigned long int i=0; i<N; i++){
    tmp = 0.0;
    for(int j=ptr[i]; j<ptr[i+1]; j++){
      tmp += val[j] * in_vec[col[j]];
    }
    out_vec[i] = tmp;
  }
  this->time->end();
  this->time->cpu_mv_time += this->time->getTime();
}

void blas::MtxVec_mult(double *in_vec, unsigned long int xindex, unsigned long int xsize, double *out_vec){
  double tmp = 0.0;
  double *val=this->coll->val;
  int *ptr=this->coll->ptr;
  int *col=this->coll->col;
  unsigned long int N = this->coll->N;

  this->time->start();
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+:tmp) schedule(static) firstprivate(out_vec, val, in_vec) lastprivate(out_vec) num_threads(this->coll->OMPThread)
#endif
  for(unsigned long int i=0; i<N; i++){
    tmp = 0.0;
    for(int j=ptr[i]; j<ptr[i+1]; j++){
      tmp += val[j] * in_vec[xindex*xsize+col[j]];
    }
    out_vec[i] = tmp;
  }
  this->time->end();
  this->time->cpu_mv_time += this->time->getTime();

}

void blas::MtxVec_mult(double *in_vec, unsigned long int xindex, unsigned long int xsize, double *out_vec, unsigned long int yindex, unsigned long int ysize){
  double tmp = 0.0;
  double *val=this->coll->val;
  int *ptr=this->coll->ptr;
  int *col=this->coll->col;
  unsigned long int N = this->coll->N;

  this->time->start();
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+:tmp) schedule(static) firstprivate(out_vec, val, in_vec) lastprivate(out_vec) num_threads(this->coll->OMPThread)
#endif
  for(unsigned long int i=0; i<N; i++){
    tmp = 0.0;
    for(int j=ptr[i]; j<ptr[i+1]; j++){
      tmp += val[j] * in_vec[xindex*xsize+col[j]];
    }
    out_vec[yindex*ysize+i] = tmp;
  }
  this->time->end();
  this->time->cpu_mv_time += this->time->getTime();
}

void blas::MtxVec_mult(double *Tval, int *Tcol, int *Tptr, double *in_vec, double *out_vec){
  double tmp = 0.0;
  unsigned long int N = this->coll->N;

  this->time->start();
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+:tmp) schedule(static) firstprivate(out_vec, Tval, in_vec) lastprivate(out_vec) num_threads(this->coll->OMPThread)
#endif
  for(unsigned long int i=0; i<N; i++){
    tmp = 0.0;
    for(int j=Tptr[i]; j<Tptr[i+1]; j++){
      tmp += Tval[j] * in_vec[Tcol[j]];
    }
    out_vec[i] = tmp;
  }
  this->time->end();
  this->time->cpu_mv_time += this->time->getTime();
}


void blas::Vec_sub(double *x, double *y, double *out){
  for(unsigned long int i=0; i<this->coll->N; i++){
    out[i] = x[i] - y[i];
  }
}

void blas::Vec_add(double *x, double *y, double *out){
  for(unsigned long int i=0; i<this->coll->N; i++){
    out[i] = x[i] + y[i];
  }
}
void blas::Vec_add(double *x, double *y, unsigned long int yindex, unsigned long int ysize, double *out, unsigned long int zindex, unsigned long int zsize){
  for(unsigned long int i=0; i<this->coll->N; i++){
    out[zindex*zsize+i] = x[i] + y[yindex*ysize+i];
  }
}

double blas::dot(double *x, double *y){
  double tmp = 0.0;
  unsigned long int N = this->coll->N;
  this->time->start();
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
#endif
  for(unsigned long int i=0; i<N; i++){
    tmp += x[i] * y[i];
  }
  this->time->end();
  this->time->cpu_dot_time += this->time->getTime();
  return tmp;
}

double blas::dot(double *x, double *y, unsigned long int size){
  double tmp = 0.0;
  this->time->start();
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
#endif
  for(unsigned long int i=0; i<size; i++){
    tmp += x[i] * y[i];
  }
  this->time->end();
  this->time->cpu_dot_time += this->time->getTime();

  return tmp;
}

double blas::dot(double *x, double *y, unsigned long int yindex, unsigned long int ysize){
  double tmp = 0.0;
  unsigned long int N = this->coll->N;
  this->time->start();
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
#endif
  for(unsigned long int i=0; i<N; i++){
    tmp += x[i] * y[yindex*ysize+i];
  }
  this->time->end();
  this->time->cpu_dot_time += this->time->getTime();

  return tmp;
}

double blas::dot(double *x, unsigned long int xindex, unsigned long int xsize, double *y, unsigned long int yindex, unsigned long int ysize){
  double tmp = 0.0;
  unsigned long int N = this->coll->N;
  this->time->start();
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
#endif
  for(unsigned long int i=0; i<N; i++){
    tmp += x[xindex*xsize+i] * y[yindex*ysize+i];
  }
  this->time->end();
  this->time->cpu_dot_time += this->time->getTime();

  return tmp;
}

void blas::Scalar_ax(double a, double *x, double *out){
  for(unsigned long int i=0; i<this->coll->N; i++){
    out[i] = a * x[i];
  }
}

void blas::Scalar_ax(double a, double *x, unsigned long int xindex, unsigned long int xsize, double *out){
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) firstprivate(out, a, x) lastprivate(out) num_threads(this->coll->OMPThread)
#endif
  for(unsigned long int i=0; i<this->coll->N; i++){
    out[i] = a * x[xindex*xsize+i];
  }
}

void blas::Scalar_axy(double a, double *x, double *y, double *out){
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) firstprivate(out) lastprivate(out) num_threads(this->coll->OMPThread)
#endif
  for(unsigned long int i=0; i<this->coll->N; i++){
    out[i] = (a * x[i]) + y[i];
  }
}

void blas::Scalar_axy(double a, double *x, unsigned long int xindex, unsigned long int xsize, double *y, double *out){
  double tmp = 0;
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) firstprivate(out, a, x, tmp, xindex, xsize) lastprivate(out) num_threads(this->coll->OMPThread)
#endif
  for(unsigned long int i=0; i<this->coll->N; i++){
    tmp = y[i];
    out[i] = (a * x[xindex*xsize+i]) + tmp;
  }
}

void blas::Scalar_axy(double a, double *x, unsigned long int xindex, unsigned long int xsize, double *y, unsigned long int yindex, unsigned long int ysize, double *out, unsigned long int zindex, unsigned long int zsize){
  double tmp = 0;
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) firstprivate(out, a, x, tmp, xindex, xsize, zindex, zsize) lastprivate(out) num_threads(this->coll->OMPThread)
#endif
  for(unsigned long int i=0; i<this->coll->N; i++){
    tmp = y[yindex*ysize+i];
    out[zindex*zsize+i] = (a * x[xindex*xsize+i]) + tmp;
  }
}

void blas::Scalar_x_div_a(double *x, double a, double *out){
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) firstprivate(out, a, x) lastprivate(out) num_threads(this->coll->OMPThread)
#endif
  for(unsigned long int i=0; i<this->coll->N; i++){
    out[i] = x[i] / a;
  }
}

double blas::Check_error(double *x_last, double *x_0){
  double tmp1 = 0.0;
  double tmp2 = 0.0;
  double tmp3 = 0.0;

  double *Ax, *Ax_0;

  Ax = new double[this->coll->N];
  Ax_0 = new double[this->coll->N];

  double *val=this->coll->val;
  int *ptr=this->coll->ptr;
  int *col=this->coll->col;
  unsigned long int N = this->coll->N;
  double *bvec=this->coll->bvec;

  for(unsigned long int i=0; i<N; i++){
    tmp1= 0.0;
    tmp2= 0.0;
    for(int j=ptr[i]; j<ptr[i+1]; j++){
      tmp1 += val[j] * x_last[col[j]];
      tmp2 += val[j] * x_0[col[j]];
    }
    Ax[i] = bvec[i] - tmp1;
    Ax_0[i] = bvec[i] - tmp2;
  }

  tmp1 = norm_2(Ax);
  tmp2 = norm_2(Ax_0);
  tmp3 = log10(tmp1/tmp2);

  delete[] Ax;
  delete[] Ax_0;

  return tmp3;
}

void blas::Hye(double *h, double *y, double *e, unsigned long int size){
  double tmp;
  std::memset(y, 0, sizeof(double)*size);

  for(int i=size-1; i>=0; i--){
    tmp = 0.0;
    for(unsigned long int j=i+1; j<size; j++){
      tmp += y[j] * h[i*(this->coll->N)+j];
    }
    y[i] = (e[i] - tmp) / h[i*(this->coll->N)+i];
  }
}

// void blas::Hye(double *h, double *y, double *e, unsigned long int size){
//   double tmp;
//   std::memset(y, 0, sizeof(double)*size);
//
//   unsigned long int i, j;
//
// #pragma omp parallel private(i, j) shared(y, e, h, tmp) num_threads(this->coll->OMPThread)
//   for(i=size-1; i>=0; i--){
// #pragma omp single
//     tmp = 0.0;
// #pragma omp for reduction(+:tmp) schedule(static)
//     for(j=i+1; j<size; j++){
//       tmp += y[j] * h[i*(this->coll->N)+j];
//     }
// #pragma omp single
//     y[i] = (e[i] - tmp) / h[i*(this->coll->N)+i];
//   }
// }

void blas::Vec_copy(double *in, double *out){
  for(unsigned long int i=0; i<this->coll->N; i++){
    out[i] = in[i];
  }
}

void blas::Vec_copy(double *in, unsigned long int xindex, unsigned long int xsize, double *out){
  for(unsigned long int i=0; i<this->coll->N; i++){
    out[i] = in[xindex*xsize+i];
  }
}

void blas::Vec_copy(double *in, double *out, unsigned long int xindex, unsigned long int xsize){
  for(unsigned long int i=0; i<this->coll->N; i++){
    out[xindex*xsize+i] = in[i];
  }
}

void blas::Kskip_cg_bicg_base(double *Ar, double *Ap, double *rvec, double *pvec, const int kskip){
  double tmp1 = 0.0;
  double tmp2 = 0.0;

  double *val=this->coll->val;
  int *ptr=this->coll->ptr;
  int *col=this->coll->col;
  unsigned long int N = this->coll->N;

  this->time->start();
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+:tmp1, tmp2) schedule(static) firstprivate(Ar, Ap, val, pvec, rvec) lastprivate(Ar, Ap) num_threads(this->coll->OMPThread)
#endif
  for(unsigned long int i=0; i<N; i++){
    tmp1 = 0.0;
    tmp2 = 0.0;
    for(int j=ptr[i]; j<ptr[i+1]; j++){
      tmp1 += val[j] * rvec[col[j]];
      tmp2 += val[j] * pvec[col[j]];
    }
    Ar[0*N+i] = tmp1;
    Ap[0*N+i] = tmp2;
  }
  for(int ii=1; ii<2*kskip+2; ii++){
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+:tmp1, tmp2) schedule(static) firstprivate(Ar, Ap, val) lastprivate(Ar, Ap) num_threads(this->coll->OMPThread)
#endif
    for(unsigned long int i=0; i<N; i++){
      tmp1 = 0.0;
      tmp2 = 0.0;
      for(int j=ptr[i]; j<ptr[i+1]; j++){
        if(ii<2*kskip+1){
          tmp1 += val[j] * Ar[(ii-1)*N+col[j]];
        }
        tmp2 += val[j] * Ap[(ii-1)*N+col[j]];
      }
      if(ii<2*kskip+1){
        Ar[(ii)*N+i] = tmp1;
      }
      Ap[(ii)*N+i] = tmp2;
    }
  }
  this->time->end();
  this->time->cpu_mv_time += this->time->getTime();


}

void blas::Kskip_cg_innerProduce(double *delta, double *eta, double *zeta, double *Ar, double *Ap, double *rvec, double *pvec, const int kskip){
  double tmp1=0.0;
  double tmp2=0.0;
  double tmp3=0.0;
  unsigned long int N = this->coll->N;

  this->time->start();
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+:tmp1, tmp2, tmp3) schedule(static) firstprivate(delta, eta, zeta, Ar, rvec, Ap, pvec) lastprivate(delta, eta, zeta) num_threads(this->coll->OMPThread)
#endif
  for(int i=0; i<2*kskip+2; i++){
    tmp1=0.0;
    tmp2=0.0;
    tmp3=0.0;
    for(unsigned long int j=0; j<N; j++){
      if(i<2*kskip){
        tmp1 += rvec[j] * Ar[i*N+j];
      }
      if(i<2*kskip+1){
        tmp2 += rvec[j] * Ap[i*N+j];
      }
      tmp3 += pvec[j] * Ap[i*N+j];
    }
    if(i<2*kskip){
      delta[i] = tmp1;
    }
    if(i<2*kskip+1){
      eta[i] = tmp2;
    }
    zeta[i] = tmp3;
  }
  this->time->end();
  this->time->cpu_dot_time += this->time->getTime();

}

void blas::Kskip_bicg_innerProduce(double *theta, double *eta, double *rho, double *phi, double *Ar, double *Ap, double *rvec, double *pvec, double *r_vec, double *p_vec, const int kskip){
  double tmp1=0.0;
  double tmp2=0.0;
  double tmp3=0.0;
  double tmp4=0.0;
  unsigned long int N = this->coll->N;

  this->time->start();
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+:tmp1, tmp2, tmp3, tmp4) schedule(static) firstprivate(theta, eta, rho, phi, Ar, rvec, Ap, pvec, r_vec, p_vec) lastprivate(theta, eta, rho, phi) num_threads(this->coll->OMPThread)
#endif
  for(int i=0; i<2*kskip+2; i++){
    tmp1=0.0;
    tmp2=0.0;
    tmp3=0.0;
    tmp4=0.0;
    for(unsigned long int j=0; j<N; j++){
      if(i<2*kskip){
        tmp1 += r_vec[j] * Ar[i*N+j];
      }
      if(i<2*kskip+1){
        tmp2 += r_vec[j] * Ap[i*N+j];
        tmp3 += p_vec[j] * Ar[i*N+j];
      }
      tmp4 += p_vec[j] * Ap[i*N+j];
    }
    if(i<2*kskip){
      theta[i] = tmp1;
    }
    if(i<2*kskip+1){
      eta[i] = tmp2;
      rho[i] = tmp3;
    }
    phi[i] = tmp4;
  }
  this->time->end();
  this->time->cpu_dot_time += this->time->getTime();

}

void blas::Gmres_sp_1(int k, double *x, double *y, double *out){
  unsigned long int N = this->coll->N;

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) firstprivate(out, x, y) lastprivate(out) num_threads(this->coll->OMPThread)
#endif
  for(unsigned long int j=0; j<N; j++){
    for(int i=0; i<=k; i++){
      out[j] -= x[i*N+k] * y[i*N+j];
    }
  }
}

void blas::MtxVec_mult_Multi(double *in_vec, double *out_vec){
  double tmp = 0.0;

  this->time->start();

  double *val1 = this->coll->val1;
  int *ptr1 = this->coll->ptr1;
  int *col1 = this->coll->col1;

  double *val2 = this->coll->val2;
  int *ptr2 = this->coll->ptr2;
  int *col2 = this->coll->col2;

  unsigned long int N1 = this->coll->N1;
  unsigned long int N2 = this->coll->N2;

  for(unsigned long int i=0; i<N1; i++){
    tmp = 0.0;
    for(int j=ptr1[i]; j<ptr1[i+1]; j++){
      tmp += val1[j] * in_vec[col1[j]];
    }
    out_vec[i] = tmp;
  }

  for(unsigned long int i=0; i<N2; i++){
    tmp = 0.0;
    for(int j=ptr2[i]; j<ptr2[i+1]; j++){
      tmp += val2[j] * in_vec[col2[j]];
    }
    out_vec[N1+i] = tmp;
  }

  this->time->end();
  this->time->cpu_mv_time += this->time->getTime();
}

void blas::Gcr_sp_1(int k, int N, double *Av, double *qvec, double *pvec, double *qq, double *beta_vec){
  double dot_tmp;

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static) firstprivate(Av, qvec, qq, beta_vec) lastprivate(beta_vec) num_threads(this->coll->OMPThread)
#endif
  for(int i=0; i<=k; i++){
    dot_tmp = dot(Av, qvec, i, N);
    beta_vec[i] = -(dot_tmp) / qq[i];
  }

  for(int i=0; i<=k; i++){
    // Scalar_axy(beta_vec[i], pvec, i, N, pvec, k+1, N, pvec, k+1, N);
    // Scalar_axy(beta_vec[i], qvec, i, N, qvec, k+1, N, qvec, k+1, N);
    Scalar_axy(beta_vec[i], (double*)(pvec+i*N), (double*)(pvec+(k+1)*N), (double*)(pvec+(k+1)*N));
    Scalar_axy(beta_vec[i], (double*)(qvec+i*N), (double*)(qvec+(k+1)*N), (double*)(qvec+(k+1)*N));
  }
}

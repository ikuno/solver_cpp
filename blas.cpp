#include <fstream>
#include <ctime>
#include <cstring>
#include <cmath>
#include <typeinfo>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "blas.hpp"

blas::blas(collection *col){
  time = new times();
  dot_proc_time = 0.0;
  MV_proc_time = 0.0;

  coll = col;
#ifdef _OPENMP
  omp_set_num_threads(this->coll->OMPThread);
#endif
}

blas::~blas(){
  delete time;
}

double blas::norm_1(double *v){
  double tmp = 0.0;
  for(long int i=0;i<this->coll->N;i++){
    tmp += fabs(v[i]);
  }
  return tmp;
}

double blas::norm_2(double *v){
  double tmp = 0.0;
  for(long int i=0;i<this->coll->N;i++){
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

  long int N = this->coll->N;

  this->time->start();
#pragma omp parallel for reduction(+:tmp) schedule(static) firstprivate(out_vec, val, in_vec) lastprivate(out_vec) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
    tmp = 0.0;
    for(long int j=ptr[i]; j<ptr[i+1]; j++){
      tmp += val[j] * in_vec[col[j]];
    }
    out_vec[i] = tmp;
  }
  this->time->end();
  this->MV_proc_time += this->time->getTime();
}

void blas::MtxVec_mult(double *in_vec, int xindex, int xsize, double *out_vec){
  double tmp = 0.0;
  double *val=this->coll->val;
  int *ptr=this->coll->ptr;
  int *col=this->coll->col;
  long int N = this->coll->N;

  this->time->start();
#pragma omp parallel for reduction(+:tmp) schedule(static) firstprivate(out_vec, val, in_vec) lastprivate(out_vec) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
    tmp = 0.0;
    for(long int j=ptr[i]; j<ptr[i+1]; j++){
      tmp += val[j] * in_vec[xindex*xsize+col[j]];
    }
    out_vec[i] = tmp;
  }
  this->time->end();
  this->MV_proc_time += this->time->getTime();
}

void blas::MtxVec_mult(double *in_vec, int xindex, int xsize, double *out_vec, int yindex, int ysize){
  double tmp = 0.0;
  double *val=this->coll->val;
  int *ptr=this->coll->ptr;
  int *col=this->coll->col;
  long int N = this->coll->N;

  this->time->start();
#pragma omp parallel for reduction(+:tmp) schedule(static) firstprivate(out_vec, val, in_vec) lastprivate(out_vec) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
    tmp = 0.0;
    for(long int j=ptr[i]; j<ptr[i+1]; j++){
      tmp += val[j] * in_vec[xindex*xsize+col[j]];
    }
    out_vec[yindex*ysize+i] = tmp;
  }
  this->time->end();
  this->MV_proc_time += this->time->getTime();
}

void blas::MtxVec_mult(double *Tval, int *Tcol, int *Tptr, double *in_vec, double *out_vec){
  double tmp = 0.0;
  long int N = this->coll->N;

  this->time->start();
#pragma omp parallel for reduction(+:tmp) schedule(static) firstprivate(out_vec, Tval, in_vec) lastprivate(out_vec) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
    tmp = 0.0;
    for(long int j=Tptr[i]; j<Tptr[i+1]; j++){
      tmp += Tval[j] * in_vec[Tcol[j]];
    }
    out_vec[i] = tmp;
  }
  this->time->end();
  this->MV_proc_time += this->time->getTime();
}


void blas::Vec_sub(double *x, double *y, double *out){
  for(long int i=0; i<this->coll->N; i++){
    out[i] = x[i] - y[i];
  }
}

void blas::Vec_add(double *x, double *y, double *out){
  for(long int i=0; i<this->coll->N; i++){
    out[i] = x[i] + y[i];
  }
}
void blas::Vec_add(double *x, double *y, int yindex, int ysize, double *out, int zindex, int zsize){
  for(long int i=0; i<this->coll->N; i++){
    out[zindex*zsize+i] = x[i] + y[yindex*ysize+i];
  }
}

double blas::dot(double *x, double *y){
  double tmp = 0.0;
  int N = this->coll->N;
  this->time->start();
#pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
    tmp += x[i] * y[i];
  }
  this->time->end();
  this->dot_proc_time += this->time->getTime();
  return tmp;
}

double blas::dot(double *x, double *y, const long int size){
  double tmp = 0.0;
  this->time->start();
#pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
  for(long int i=0; i<size; i++){
    tmp += x[i] * y[i];
  }
  this->time->end();
  this->dot_proc_time += this->time->getTime();
  return tmp;
}

double blas::dot(double *x, double *y, int yindex, int ysize){
  double tmp = 0.0;
  int N = this->coll->N;
  this->time->start();
#pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
    tmp += x[i] * y[yindex*ysize+i];
  }
  this->time->end();
  this->dot_proc_time += this->time->getTime();
  return tmp;
}

double blas::dot(double *x, int xindex, int xsize, double *y, int yindex, int ysize){
  double tmp = 0.0;
  int N = this->coll->N;
  this->time->start();
#pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
    tmp += x[xindex*xsize+i] * y[yindex*ysize+i];
  }
  this->time->end();
  this->dot_proc_time += this->time->getTime();
  return tmp;
}

void blas::Scalar_ax(double a, double *x, double *out){
  for(long int i=0; i<this->coll->N; i++){
    out[i] = a * x[i];
  }
}

void blas::Scalar_ax(double a, double *x, int xindex, int xsize, double *out){
#pragma omp parallel for schedule(static) firstprivate(out, a, x) lastprivate(out) num_threads(this->coll->OMPThread)
  for(long int i=0; i<this->coll->N; i++){
    out[i] = a * x[xindex*xsize+i];
  }
}

void blas::Scalar_axy(double a, double *x, double *y, double *out){
  double tmp;
  for(long int i=0; i<this->coll->N; i++){
    tmp = y[i];
    out[i] = (a * x[i]) + tmp;
  }
}

void blas::Scalar_axy(double a, double *x, int xindex, int xsize, double *y, double *out){
  double tmp;
  for(long int i=0; i<this->coll->N; i++){
    tmp = y[i];
    out[i] = (a * x[xindex*xsize+i]) + tmp;
  }
}

void blas::Scalar_axy(double a, double *x, int xindex, int xsize, double *y, int yindex, int ysize, double *out, int zindex, int zsize){
  double tmp;
  for(long int i=0; i<this->coll->N; i++){
    tmp = y[yindex*ysize+i];
    out[zindex*zsize+i] = (a * x[xindex*xsize+i]) + tmp;
  }
}

void blas::Scalar_x_div_a(double *x, double a, double *out){
  for(long int i=0; i<this->coll->N; i++){
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
  long int N = this->coll->N;
  double *bvec=this->coll->bvec;

  for(long int i=0; i<N; i++){
    tmp1= 0.0;
    tmp2= 0.0;
    for(long int j=ptr[i]; j<ptr[i+1]; j++){
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

void blas::Hye(double *h, double *y, double *e, const long int size){
  double tmp;
  std::memset(y, 0, sizeof(double)*size);
  for(long int i=size-1; i>=0; i--){
    tmp = 0.0;
    for(long int j=i+1; j<size; j++){
      tmp += y[j] * h[i*(this->coll->N)+j];
    }
    y[i] = (e[i] - tmp) / h[i*(this->coll->N)+i];
  }

}

void blas::Vec_copy(double *in, double *out){
  for(long int i=0; i<this->coll->N; i++){
    out[i] = in[i];
  }
}

void blas::Vec_copy(double *in, int xindex, int xsize, double *out){
  for(long int i=0; i<this->coll->N; i++){
    out[i] = in[xindex*xsize+i];
  }
}

void blas::Vec_copy(double *in, double *out, int xindex, int xsize){
  for(long int i=0; i<this->coll->N; i++){
    out[xindex*xsize+i] = in[i];
  }
}

void blas::Kskip_cg_base(double *Ar, double *Ap, double *rvec, double *pvec, const int kskip){
  double tmp1 = 0.0;
  double tmp2 = 0.0;

  double *val=this->coll->val;
  int *ptr=this->coll->ptr;
  int *col=this->coll->col;
  long int N = this->coll->N;

#pragma omp parallel for reduction(+:tmp1, tmp2) schedule(static) firstprivate(Ar, Ap, val, pvec, rvec) lastprivate(Ar, Ap) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
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
#pragma omp parallel for reduction(+:tmp1, tmp2) schedule(static) firstprivate(Ar, Ap, val) lastprivate(Ar, Ap) num_threads(this->coll->OMPThread)
    for(long int i=0; i<N; i++){
      tmp1 = 0.0;
      tmp2 = 0.0;
      for(int j=ptr[i]; j<ptr[i+1]; j++){
        if(ii<2*kskip){
          tmp1 += val[j] * Ar[(ii-1)*N+col[j]];
        }
        tmp2 += val[j] * Ap[(ii-1)*N+col[j]];
      }
      if(ii<2*kskip){
        Ar[(ii)*N+i] = tmp1;
      }
      Ap[(ii)*N+i] = tmp2;
    }
  }

}

void blas::Kskip_cg_innerProduce(double *delta, double *eta, double *zeta, double *Ar, double *Ap, double *rvec, double *pvec, const int kskip){
  double tmp1=0.0;
  double tmp2=0.0;
  double tmp3=0.0;
  long int N = this->coll->N;

#pragma omp parallel for reduction(+:tmp1, tmp2, tmp3) schedule(static) firstprivate(delta, eta, zeta, Ar, rvec, Ap, pvec) lastprivate(delta, eta, zeta) num_threads(this->coll->OMPThread)
  for(int i=0; i<2*kskip+2; i++){
    tmp1=0.0;
    tmp2=0.0;
    tmp3=0.0;
    for(long int j=0; j<N; j++){
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
}

void blas::Kskip_kskipBicg_base(double *Ar, double *Ap, double *rvec, double *pvec, const int kskip){
  double tmp1 = 0.0;
  double tmp2 = 0.0;
  double *val=this->coll->val;
  int *ptr=this->coll->ptr;
  int *col=this->coll->col;
  long int N = this->coll->N;

#pragma omp parallel for reduction(+:tmp1, tmp2) schedule(static) firstprivate(Ar, Ap, val, pvec, rvec) lastprivate(Ar, Ap) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
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
#pragma omp parallel for reduction(+:tmp1, tmp2) schedule(static) firstprivate(Ar, Ap, val) lastprivate(Ar, Ap) num_threads(this->coll->OMPThread)
    for(long int i=0; i<N; i++){
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

}

void blas::Kskip_kskipBicg_innerProduce(double *theta, double *eta, double *rho, double *phi, double *Ar, double *Ap, double *rvec, double *pvec, double *r_vec, double *p_vec, const int kskip){
  double tmp1=0.0;
  double tmp2=0.0;
  double tmp3=0.0;
  double tmp4=0.0;
  long int N = this->coll->N;

#pragma omp parallel for reduction(+:tmp1, tmp2, tmp3, tmp4) schedule(static) firstprivate(theta, eta, rho, phi, Ar, rvec, Ap, pvec, r_vec, p_vec) lastprivate(theta, eta, rho, phi) num_threads(this->coll->OMPThread)
  for(int i=0; i<2*kskip+2; i++){
    tmp1=0.0;
    tmp2=0.0;
    tmp3=0.0;
    tmp4=0.0;
    for(long int j=0; j<N; j++){
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

}

void blas::Gmres_sp_1(int k, double *x, double *y, double *out){
  int N = this->coll->N;

#pragma omp parallel for schedule(static) firstprivate(out, x, y) lastprivate(out) num_threads(this->coll->OMPThread)
  for(long int j=0; j<N; j++){
    for(int i=0; i<=k; i++){
      out[j] -= x[i*N+k] * y[i*N+j];
    }
  }
}

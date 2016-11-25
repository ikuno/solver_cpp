#ifndef BLAS_HPP_INCLUDED__
#define BLAS_HPP_INCLUDED__

#include <fstream>
#include <ctime>
#include <string>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "solver_collection.hpp"

template <typename T, typename F>
class blas {
  private:
    collection<T> *coll;
  public:
    blas(collection<T> *col);

    std::ofstream *output(std::string name);

    std::string get_date_time();

    F norm_1(T *v);

    F norm_2(T *v);

    void MtxVec_mult(T *in_vec, T *out_vec);
    
    void MtxVec_mult(T *in_vec, int xindex, int xsize, T *out_vec);

    void MtxVec_mult(T *Tval, int *Tcol, int *Tptr, T *in_vec, T *out_vec);

    void Vec_sub(T *x, T *y, T *out);

    void Vec_add(T *x, T *y, T *out);

    F dot(T *x, T *y);

    F dot(T *x, T *y, const long int size);

    F dot(T *x, T *y, int xindex, int xsize);

    void Scalar_ax(T a, T *x, T *out);
    
    void Scalar_ax(T a, T *x, int xindex, int xsize, T *out);

    void Scalar_axy(T a, T *x, T *y, T *out);

    void Scalar_x_div_a(T *x, T a, T *out);

    double Check_error(T *x_last, T *x_0);

    void Hye(T *h, T *y, T *e, const long int size);

    void Vec_copy(T *in, T *out);

    void Vec_copy(T *in, int xindex, int xsize, T *out);

    void Vec_copy(T *in, T *out, int xindex, int xsize);

//-----------------------------------

    void Kskip_cg_base(T **Ar, T **Ap, T *rvec, T *pvec, const int kskip);

    void Kskip_cg_innerProduce(T *delta, T *eta, T *zeta, T **Ar, T **Ap, T *rvec, T *pvec, const int kskip);

    void Kskip_kskipBicg_base(T **Ar, T **Ap, T *rvec, T *pvec, const int kskip);

    void Kskip_kskipBicg_innerProduce(T *theta, T *eta, T *rho, T *phi, T **Ar, T **Ap, T *rvec, T *pvec, T *r_vec, T *p_vec, const int kskip);

    void Gmres_sp_1(int k, T *x, T *y, T *out);

};

template <typename T>
blas<T, F>::blas(collection<T> *col){
  coll = col;
  omp_set_num_threads(this->coll->OMPThread);
}

template <typename T>
std::string blas<T, F>::get_date_time(){
  struct tm *date;
  time_t now;
  int month, day;
  int hour, minute, second;
  std::string date_time;

  time(&now);
  date = localtime(&now);

  month = date->tm_mon + 1;
  day = date->tm_mday;
  hour = date->tm_hour;
  minute = date->tm_min;
  second = date->tm_sec;

  date_time=std::to_string(month)+"-"+std::to_string(day)+"-"+std::to_string(hour)+"_"+std::to_string(minute)+"_"+std::to_string(second);

  return date_time;
}

template <typename T, typename F>
F blas<T, F>::norm_1(T *v){
  F tmp = 0.0;
  for(long int i=0;i<this->coll->N;i++){
    tmp += fabs(static_cast<F>(v[i]));
  }
  return tmp;
}

template <typename T, typename F>
F blas<T, F>::norm_2(T *v){
  F tmp = 0.0;
  for(long int i=0;i<this->coll->N;i++){
    tmp += static_cast<F>(v[i]) * static_cast<F>(v[i]);
  }
  return sqrt(tmp);
}

template <typename T, typename F>
void blas<T, F>::MtxVec_mult(T *in_vec, T *out_vec){
  F tmp = 0.0;
  T *val=this->coll->val;
  int *ptr=this->coll->ptr;
  int *col=this->coll->col;
  long int N = this->coll->N;
#pragma omp parallel for reduction(+:tmp) schedule(static) firstprivate(out_vec, val, in_vec) lastprivate(out_vec) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
    tmp = 0.0;
    for(long int j=ptr[i]; j<ptr[i+1]; j++){
      tmp += static_cast<F>(val[j]) * static_cast<F>(in_vec[col[j]]);
    }
    out_vec[i] = tmp;
  }
}

template <typename T, typename F>
void blas<T, F>::MtxVec_mult(T *in_vec, int xindex, int xsize, T *out_vec){
  F tmp = 0.0;
  T *val=this->coll->val;
  int *ptr=this->coll->ptr;
  int *col=this->coll->col;
  long int N = this->coll->N;
#pragma omp parallel for reduction(+:tmp) schedule(static) firstprivate(out_vec, val, in_vec) lastprivate(out_vec) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
    tmp = 0.0;
    for(long int j=ptr[i]; j<ptr[i+1]; j++){
      tmp += static_cast<F>(val[j]) * static_cast<F>(in_vec[xindex*xsize+col[j]]);
    }
    out_vec[i] = tmp;
  }
}

template <typename T, typename F>
void blas<T, F>::MtxVec_mult(T *Tval, int *Tcol, int *Tptr, T *in_vec, T *out_vec){
  F tmp = 0.0;
  long int N = this->coll->N;

#pragma omp parallel for reduction(+:tmp) schedule(static) firstprivate(out_vec, Tval, in_vec) lastprivate(out_vec) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
    tmp = 0.0;
    for(long int j=Tptr[i]; j<Tptr[i+1]; j++){
      tmp += static_cast<F>(Tval[j]) * static_cast<F>(in_vec[Tcol[j]]);
    }
    out_vec[i] = tmp;
  }
}


template <typename T, typename F>
void blas<T, F>::Vec_sub(T *x, T *y, T *out){
  for(long int i=0; i<this->coll->N; i++){
    out[i] = static_cast<F>(x[i]) - static_cast<F>(y[i]);
  }
}

template <typename T, typename F>
void blas<T, F>::Vec_add(T *x, T *y, T *out){
  for(long int i=0; i<this->coll->N; i++){
    out[i] = static_cast<F>(x[i]) - static_cast<F>(y[i]);
  }
}

template <typename T, typename F>
F blas<T, F>::dot(T *x, T *y){
  F tmp = 0.0;
  int N = this->coll->N;
// #pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
    tmp += static_cast<F>(x[i]) * static_cast<F>(y[i]);
  }
  return tmp;
}

template <typename T, typename F>
F blas<T, F>::dot(T *x, T *y, const long int size){
  F tmp = 0.0;
// #pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
  for(long int i=0; i<size; i++){
    tmp += static_cast<F>(x[i]) * static_cast<F>(y[i]);
  }
  return tmp;
}

template <typename T, typename F>
F blas<T, F>::dot(T *x, T *y, int xindex, int xsize){
  F tmp = 0.0;
  int N = this->coll->N;
// #pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
    tmp += static_cast<F>(x[i]) * static_cast<F>(y[xindex*xsize+i]);
  }
  return tmp;
}

template <typename T, typename F>
void blas<T, F>::Scalar_ax(T a, T *x, T *out){
  for(long int i=0; i<this->coll->N; i++){
    out[i] = static_cast<F>(a) * static_cast<F>(x[i]);
  }
}

template <typename T, typename F>
void blas<T, F>::Scalar_ax(T a, T *x, int xindex, int xsize, T *out){
#pragma omp parallel for schedule(static) firstprivate(out, a, x) lastprivate(out) num_threads(this->coll->OMPThread)
  for(long int i=0; i<this->coll->N; i++){
    out[i] = static_cast<F>(a) * static_cast<F>(x[xindex*xsize+i]);
  }
}

template <typename T, typename F>
void blas<T, F>::Scalar_axy(T a, T *x, T *y, T *out){
  F tmp;
  for(long int i=0; i<this->coll->N; i++){
    tmp = static_cast<F>(y[i]);
    out[i] = (static_cast<F>a * static_cast<F>(x[i])) + tmp;
  }
}

template <typename T, typename F>
void blas<T, F>::Scalar_x_div_a(T *x, T a, T *out){
  for(long int i=0; i<this->coll->N; i++){
    out[i] = static_cast<F>(x[i]) / static_cast<F>(a);
  }
}

template <typename T, typename F>
double blas<T, F>::Check_error(T *x_last, T *x_0){
  F tmp1 = 0.0;
  F tmp2 = 0.0;
  F tmp3 = 0.0;

  F *Ax, *Ax_0;

  Ax = new T[this->coll->N];
  Ax_0 = new T[this->coll->N];

  for(long int i=0; i<this->coll->N; i++){
    tmp1= 0.0;
    tmp2= 0.0;
    for(long int j=this->coll->ptr[i]; j<this->coll->ptr[i+1]; j++){
      tmp1 += static_cast<F>(this->coll->val[j]) * static_cast<F>(x_last[this->coll->col[j]]);
      tmp2 += static_cast<F>(this->coll->val[j]) * static_cast<F>(x_0[this->coll->col[j]]);
    }
    Ax[i] = static_cast<F>(this->coll->bvec[i]) - tmp1;
    Ax_0[i] = static_cast<F>(this->coll->bvec[i]) - tmp2;
  }

  tmp1 = norm_2(Ax);
  tmp2 = norm_2(Ax_0);
  tmp3 = log10(tmp1/tmp2);

  delete[] Ax;
  delete[] Ax_0;

  return tmp3;
}

template <typename T, typename F>
void blas<T, F>::Hye(T *h, T *y, T *e, const long int size){
  F tmp;
  // for(long int i=0; i<size; i++){
  //   y[i] = 0.0;
  // }
  std::memset(y, 0, sizeof(T)*size);
  for(long int i=size-1; i>=0; i--){
    tmp = 0.0;
    for(long int j=i+1; j<size; j++){
      tmp += static_cast<F>(y[j]) * static_cast<T>(h[i*(this->coll->N)+j]);
    }
    y[i] = (static_cast<F>(e[i]) - tmp) / static_cast<F>(h[i*(this->coll->N)+i]);
  }
}

template <typename T, typename F>
void blas<T, F>::Vec_copy(T *in, T *out){
  for(long int i=0; i<this->coll->N; i++){
    out[i] = static_cast<T>(in[i]);
  }
}

template <typename T, typename F>
void blas<T, F>::Vec_copy(T *in, int xindex, int xsize, T *out){
  for(long int i=0; i<this->coll->N; i++){
    out[i] = static_cast<F>(in[xindex*xsize+i]);
  }
}

template <typename T, typename F>
void blas<T, F>::Vec_copy(T *in, T *out, int xindex, int xsize){
  for(long int i=0; i<this->coll->N; i++){
    out[xindex*xsize+i] = static_cast<F>(in[i]);
  }
}

template <typename T, typename F>
void blas<T, F>::Kskip_cg_base(T **Ar, T **Ap, T *rvec, T *pvec, const int kskip){
  F tmp1 = 0.0;
  F tmp2 = 0.0;

  T *val=this->coll->val;
  int *ptr=this->coll->ptr;
  int *col=this->coll->col;
  long int N = this->coll->N;

#pragma omp parallel for reduction(+:tmp1, tmp2) schedule(static) firstprivate(Ar, Ap, val, pvec, rvec) lastprivate(Ar, Ap) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
    tmp1 = 0.0;
    tmp2 = 0.0;
    for(int j=ptr[i]; j<ptr[i+1]; j++){
      tmp1 += static_cast<F>(val[j]) * static_cast<F>(rvec[col[j]]);
      tmp2 += static_cast<F>(val[j]) * static_cast<F>(pvec[col[j]]);
    }
    Ar[0][i] = tmp1;
    Ap[0][i] = tmp2;
  }

  for(int ii=1; ii<2*kskip+2; ii++){
#pragma omp parallel for reduction(+:tmp1, tmp2) schedule(static) firstprivate(Ar, Ap, val) lastprivate(Ar, Ap) num_threads(this->coll->OMPThread)
    for(long int i=0; i<N; i++){
      tmp1 = 0.0;
      tmp2 = 0.0;
      for(int j=ptr[i]; j<ptr[i+1]; j++){
        if(ii<2*kskip){
          tmp1 += static_cast<F>(val[j]) * static_cast<F>(Ar[(ii-1)][col[j]]);
        }
        tmp2 += static_cast<F>(val[j]) * static_cast<F>(Ap[(ii-1)][col[j]]);
      }
      if(ii<2*kskip){
        Ar[(ii)][i] = tmp1;
      }
      Ap[(ii)][i] = tmp2;
    }
  }
}

template <typename T, typename F>
void blas<T, F>::Kskip_cg_innerProduce(T *delta, T *eta, T *zeta, T **Ar, T **Ap, T *rvec, T *pvec, const int kskip){
  F tmp1=0.0;
  F tmp2=0.0;
  F tmp3=0.0;
  long int N = this->coll->N;

#pragma omp parallel for reduction(+:tmp1, tmp2, tmp3) schedule(static) firstprivate(delta, eta, zeta, Ar, rvec, Ap, pvec) lastprivate(delta, eta, zeta) num_threads(this->coll->OMPThread)
  for(int i=0; i<2*kskip+2; i++){
    tmp1=0.0;
    tmp2=0.0;
    tmp3=0.0;
    for(long int j=0; j<N; j++){
      if(i<2*kskip){
        tmp1 += static_cast<F>(rvec[j]) * static_cast<F>(Ar[i][j]);
      }
      if(i<2*kskip+1){
        tmp2 += static_cast<F>(rvec[j]) * static_cast<F>(Ap[i][j]);
      }
      tmp3 += static_cast<F>(pvec[j]) * static_cast<F>(Ap[i][j]);
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

template <typename T, typename F>
void blas<T, F>::Kskip_kskipBicg_base(T **Ar, T **Ap, T *rvec, T *pvec, const int kskip){
  F tmp1 = 0.0;
  F tmp2 = 0.0;
  T *val=this->coll->val;
  int *ptr=this->coll->ptr;
  int *col=this->coll->col;
  long int N = this->coll->N;

#pragma omp parallel for reduction(+:tmp1, tmp2) schedule(static) firstprivate(Ar, Ap, val, pvec, rvec) lastprivate(Ar, Ap) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
    tmp1 = 0.0;
    tmp2 = 0.0;
    for(int j=ptr[i]; j<ptr[i+1]; j++){
      tmp1 += static_cast<F>(val[j]) * static_cast<F>(rvec[col[j]]);
      tmp2 += static_cast<F>(val[j]) * static_cast<F>(pvec[col[j]]);
    }
    Ar[0][i] = tmp1;
    Ap[0][i] = tmp2;
  }

  for(int ii=1; ii<2*kskip+2; ii++){
#pragma omp parallel for reduction(+:tmp1, tmp2) schedule(static) firstprivate(Ar, Ap, val) lastprivate(Ar, Ap) num_threads(this->coll->OMPThread)
    for(long int i=0; i<N; i++){
      tmp1 = 0.0;
      tmp2 = 0.0;
      for(int j=ptr[i]; j<ptr[i+1]; j++){
        if(ii<2*kskip+1){
          tmp1 += static_cast<F>(val[j]) * static_cast<F>(Ar[(ii-1)][col[j]]);
        }
        tmp2 += static_cast<F>(val[j]) * static_cast<F>(Ap[(ii-1)][col[j]]);
      }
      if(ii<2*kskip+1){
        Ar[(ii)][i] = tmp1;
      }
      Ap[(ii)][i] = tmp2;
    }
  }
}

template <typename T, typename F>
void blas<T, F>::Kskip_kskipBicg_innerProduce(T *theta, T *eta, T *rho, T *phi, T **Ar, T **Ap, T *rvec, T *pvec, T *r_vec, T *p_vec, const int kskip){
  F tmp1=0.0;
  F tmp2=0.0;
  F tmp3=0.0;
  F tmp4=0.0;
  long int N = this->coll->N;

#pragma omp parallel for reduction(+:tmp1, tmp2, tmp3, tmp4) schedule(static) firstprivate(theta, eta, rho, phi, Ar, rvec, Ap, pvec, r_vec, p_vec) lastprivate(theta, eta, rho, phi) num_threads(this->coll->OMPThread)
  for(int i=0; i<2*kskip+2; i++){
    tmp1=0.0;
    tmp2=0.0;
    tmp3=0.0;
    tmp4=0.0;
    for(long int j=0; j<N; j++){
      if(i<2*kskip){
        tmp1 += static_cast<F>(r_vec[j]) * static_cast<F>(Ar[i][j]);
      }
      if(i<2*kskip+1){
        tmp2 += static_cast<F>(r_vec[j]) * static_cast<F>(Ap[i][j]);
        tmp3 += static_cast<F>(p_vec[j]) * static_cast<F>(Ar[i][j]);
      }
      tmp4 += static_cast<F>(p_vec[j]) * static_cast<F>(Ap[i][j]);
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

template <typename T, typename F>
void blas<T, F>::Gmres_sp_1(int k, T *x, T *y, T *out){
  int N = this->coll->N;

#pragma omp parallel for schedule(static) firstprivate(out, x, y) lastprivate(out) num_threads(this->coll->OMPThread)
  for(long int j=0; j<N; j++){
    for(int i=0; i<=k; i++){
      out[j] -= static_cast<F>(x[i*N+k]) * static_cast<F>(y[i*N+j]);
    }
  }
}


#endif //BLAS_HPP_INCLUDED__


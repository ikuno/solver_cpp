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

template <typename T>
class blas {
  private:
    collection<T> *coll;
  public:
    blas(collection<T> *col);

    std::ofstream *output(std::string name);

    std::string get_date_time();

    T norm_1(T *v);

    T norm_2(T *v);

    void MtxVec_mult(T *in_vec, T *out_vec);
    
    void MtxVec_mult(T *in_vec, int xindex, int xsize, T *out_vec);

    void MtxVec_mult(T *Tval, int *Tcol, int *Tptr, T *in_vec, T *out_vec);

    void Vec_sub(T *x, T *y, T *out);

    void Vec_add(T *x, T *y, T *out);

    T dot(T *x, T *y);

    T dot(T *x, T *y, const long int size);

    T dot(T *x, T *y, int xindex, int xsize);

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
blas<T>::blas(collection<T> *col){
  coll = col;
  omp_set_num_threads(this->coll->OMPThread);
}

template <typename T>
std::string blas<T>::get_date_time(){
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

template <typename T>
T blas<T>::norm_1(T *v){
  T tmp = 0.0;
  for(long int i=0;i<this->coll->N;i++){
    tmp += fabs(v[i]);
  }
  return tmp;
}

template <typename T>
T blas<T>::norm_2(T *v){
  T tmp = 0.0;
  for(long int i=0;i<this->coll->N;i++){
    tmp += v[i] * v[i];
  }
  return sqrt(tmp);
}

template <typename T>
void blas<T>::MtxVec_mult(T *in_vec, T *out_vec){
  T tmp = 0.0;
  T *val=this->coll->val;
  int *ptr=this->coll->ptr;
  int *col=this->coll->col;
  long int N = this->coll->N;
#pragma omp parallel for reduction(+:tmp) schedule(static) firstprivate(out_vec, val, in_vec) lastprivate(out_vec) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
    tmp = 0.0;
    for(long int j=ptr[i]; j<ptr[i+1]; j++){
      tmp += val[j] * in_vec[col[j]];
    }
    out_vec[i] = tmp;
  }
}

template <typename T>
void blas<T>::MtxVec_mult(T *in_vec, int xindex, int xsize, T *out_vec){
  T tmp = 0.0;
  T *val=this->coll->val;
  int *ptr=this->coll->ptr;
  int *col=this->coll->col;
  long int N = this->coll->N;
#pragma omp parallel for reduction(+:tmp) schedule(static) firstprivate(out_vec, val, in_vec) lastprivate(out_vec) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
    tmp = 0.0;
    for(long int j=ptr[i]; j<ptr[i+1]; j++){
      tmp += val[j] * in_vec[xindex*xsize+col[j]];
    }
    out_vec[i] = tmp;
  }
}

template <typename T>
void blas<T>::MtxVec_mult(T *Tval, int *Tcol, int *Tptr, T *in_vec, T *out_vec){
  T tmp = 0.0;
  long int N = this->coll->N;

#pragma omp parallel for reduction(+:tmp) schedule(static) firstprivate(out_vec, Tval, in_vec) lastprivate(out_vec) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
    tmp = 0.0;
    for(long int j=Tptr[i]; j<Tptr[i+1]; j++){
      tmp += Tval[j] * in_vec[Tcol[j]];
    }
    out_vec[i] = tmp;
  }
}


template <typename T>
void blas<T>::Vec_sub(T *x, T *y, T *out){
  for(long int i=0; i<this->coll->N; i++){
    out[i] = x[i] - y[i];
  }
}

template <typename T>
void blas<T>::Vec_add(T *x, T *y, T *out){
  for(long int i=0; i<this->coll->N; i++){
    out[i] = x[i] + y[i];
  }
}

template <typename T>
T blas<T>::dot(T *x, T *y){
  T tmp = 0.0;
  int N = this->coll->N;
// #pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
    tmp += x[i] * y[i];
  }
  return tmp;
}

template <typename T>
T blas<T>::dot(T *x, T *y, const long int size){
  T tmp = 0.0;
// #pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
  for(long int i=0; i<size; i++){
    tmp += x[i] * y[i];
  }
  return tmp;
}

template <typename T>
T blas<T>::dot(T *x, T *y, int xindex, int xsize){
  T tmp = 0.0;
  int N = this->coll->N;
// #pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
  for(long int i=0; i<N; i++){
    tmp += x[i] * y[xindex*xsize+i];
  }
  return tmp;
}

template <typename T>
void blas<T>::Scalar_ax(T a, T *x, T *out){
  for(long int i=0; i<this->coll->N; i++){
    out[i] = a * x[i];
  }
}

template <typename T>
void blas<T>::Scalar_ax(T a, T *x, int xindex, int xsize, T *out){
#pragma omp parallel for schedule(static) firstprivate(out, a, x) lastprivate(out) num_threads(this->coll->OMPThread)
  for(long int i=0; i<this->coll->N; i++){
    out[i] = a * x[xindex*xsize+i];
  }
}

template <typename T>
void blas<T>::Scalar_axy(T a, T *x, T *y, T *out){
  T tmp;
  for(long int i=0; i<this->coll->N; i++){
    tmp = y[i];
    out[i] = (a * x[i]) + tmp;
  }
}

template <typename T>
void blas<T>::Scalar_x_div_a(T *x, T a, T *out){
  for(long int i=0; i<this->coll->N; i++){
    out[i] = x[i] / a;
  }
}

template <typename T>
double blas<T>::Check_error(T *x_last, T *x_0){
  T tmp1 = 0.0;
  T tmp2 = 0.0;
  double tmp3 = 0.0;

  T *Ax, *Ax_0;

  Ax = new T[this->coll->N];
  Ax_0 = new T[this->coll->N];

  for(long int i=0; i<this->coll->N; i++){
    tmp1= 0.0;
    tmp2= 0.0;
    for(long int j=this->coll->ptr[i]; j<this->coll->ptr[i+1]; j++){
      tmp1 += this->coll->val[j] * x_last[this->coll->col[j]];
      tmp2 += this->coll->val[j] * x_0[this->coll->col[j]];
    }
    Ax[i] = this->coll->bvec[i] - tmp1;
    Ax_0[i] = this->coll->bvec[i] - tmp2;
  }

  tmp1 = norm_2(Ax);
  tmp2 = norm_2(Ax_0);
  tmp3 = log10(tmp1/tmp2);

  delete[] Ax;
  delete[] Ax_0;

  return tmp3;
}

template <typename T>
void blas<T>::Hye(T *h, T *y, T *e, const long int size){
  T tmp;
  for(long int i=0; i<size; i++){
    y[i] = 0.0;
  }
  for(long int i=size-1; i>=0; i--){
    tmp = 0.0;
    for(long int j=i+1; j<size; j++){
      tmp += y[j] * h[i*(this->coll->N)+j];
    }
    y[i] = (e[i] - tmp) / h[i*(this->coll->N)+i];
  }
}

template <typename T>
void blas<T>::Vec_copy(T *in, T *out){
  for(long int i=0; i<this->coll->N; i++){
    out[i] = in[i];
  }
}

template <typename T>
void blas<T>::Vec_copy(T *in, int xindex, int xsize, T *out){
  for(long int i=0; i<this->coll->N; i++){
    out[i] = in[xindex*xsize+i];
  }
}

template <typename T>
void blas<T>::Vec_copy(T *in, T *out, int xindex, int xsize){
  for(long int i=0; i<this->coll->N; i++){
    out[xindex*xsize+i] = in[i];
  }
}

template <typename T>
void blas<T>::Kskip_cg_base(T **Ar, T **Ap, T *rvec, T *pvec, const int kskip){
  T tmp1 = 0.0;
  T tmp2 = 0.0;

  T *val=this->coll->val;
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
          tmp1 += val[j] * Ar[(ii-1)][col[j]];
        }
        tmp2 += val[j] * Ap[(ii-1)][col[j]];
      }
      if(ii<2*kskip){
        Ar[(ii)][i] = tmp1;
      }
      Ap[(ii)][i] = tmp2;
    }
  }
}

template <typename T>
void blas<T>::Kskip_cg_innerProduce(T *delta, T *eta, T *zeta, T **Ar, T **Ap, T *rvec, T *pvec, const int kskip){
  T tmp1=0.0;
  T tmp2=0.0;
  T tmp3=0.0;
  long int N = this->coll->N;

#pragma omp parallel for reduction(+:tmp1, tmp2, tmp3) schedule(static) firstprivate(delta, eta, zeta, Ar, rvec, Ap, pvec) lastprivate(delta, eta, zeta) num_threads(this->coll->OMPThread)
  for(int i=0; i<2*kskip+2; i++){
    tmp1=0.0;
    tmp2=0.0;
    tmp3=0.0;
    for(long int j=0; j<N; j++){
      if(i<2*kskip){
        tmp1 += rvec[j] * Ar[i][j];
      }
      if(i<2*kskip+1){
        tmp2 += rvec[j] * Ap[i][j];
      }
      tmp3 += pvec[j] * Ap[i][j];
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

template <typename T>
void blas<T>::Kskip_kskipBicg_base(T **Ar, T **Ap, T *rvec, T *pvec, const int kskip){
  T tmp1 = 0.0;
  T tmp2 = 0.0;
  T *val=this->coll->val;
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
          tmp1 += val[j] * Ar[(ii-1)][col[j]];
        }
        tmp2 += val[j] * Ap[(ii-1)][col[j]];
      }
      if(ii<2*kskip+1){
        Ar[(ii)][i] = tmp1;
      }
      Ap[(ii)][i] = tmp2;
    }
  }
}

template <typename T>
void blas<T>::Kskip_kskipBicg_innerProduce(T *theta, T *eta, T *rho, T *phi, T **Ar, T **Ap, T *rvec, T *pvec, T *r_vec, T *p_vec, const int kskip){
  T tmp1=0.0;
  T tmp2=0.0;
  T tmp3=0.0;
  T tmp4=0.0;
  long int N = this->coll->N;

#pragma omp parallel for reduction(+:tmp1, tmp2, tmp3, tmp4) schedule(static) firstprivate(theta, eta, rho, phi, Ar, rvec, Ap, pvec, r_vec, p_vec) lastprivate(theta, eta, rho, phi) num_threads(this->coll->OMPThread)
  for(int i=0; i<2*kskip+2; i++){
    tmp1=0.0;
    tmp2=0.0;
    tmp3=0.0;
    tmp4=0.0;
    for(long int j=0; j<N; j++){
      if(i<2*kskip){
        tmp1 += r_vec[j] * Ar[i][j];
      }
      if(i<2*kskip+1){
        tmp2 += r_vec[j] * Ap[i][j];
        tmp3 += p_vec[j] * Ar[i][j];
      }
      tmp4 += p_vec[j] * Ap[i][j];
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

template <typename T>
void blas<T>::Gmres_sp_1(int k, T *x, T *y, T *out){
  int N = this->coll->N;

#pragma omp parallel for schedule(static) firstprivate(out, x, y) lastprivate(out) num_threads(this->coll->OMPThread)
  for(long int j=0; j<N; j++){
    for(int i=0; i<=k; i++){
      out[j] -= x[i*N+k] * y[i*N+j];
    }
  }
}


#endif //BLAS_HPP_INCLUDED__


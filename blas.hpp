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
#include "cudaFunction.hpp"

template <typename T>
class blas {
  private:
    collection<T> *coll;
    cuda *cu;
    bool mix;
  public:
    blas(collection<T> *col);
    ~blas();

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
  mix = col->isInnerNow;
  omp_set_num_threads(this->coll->OMPThread);
  cu = new cuda();
}

template <typename T>
blas<T>::~blas(){
  delete cu;
}

template <typename T>
T blas<T>::norm_1(T *v){
  T tmp = 0.0;
  if(mix){
    for(long int i=0;i<this->coll->N;i++){
      tmp += fabs(static_cast<float>(v[i]));
    }
  }else{
    for(long int i=0;i<this->coll->N;i++){
      tmp += fabs(v[i]);
    }
  }
  return tmp;
}

template <typename T>
T blas<T>::norm_2(T *v){
  T tmp = 0.0;
  if(mix){
    for(long int i=0;i<this->coll->N;i++){
      tmp += static_cast<float>(v[i]) * static_cast<float>(v[i]);
    }
    tmp = sqrt(static_cast<float>(tmp));
  }else{
    for(long int i=0;i<this->coll->N;i++){
      tmp += v[i] * v[i];
    }
    tmp = sqrt(tmp);
  }
  return tmp;
}

template <typename T>
void blas<T>::MtxVec_mult(T *in_vec, T *out_vec){
  T tmp = 0.0;

  T *val=this->coll->val;
  int *ptr=this->coll->ptr;
  int *col=this->coll->col;

  double *Cval=this->coll->Cval;
  int *Cptr=this->coll->Cptr;
  int *Ccol=this->coll->Ccol;

  long int N = this->coll->N;
  if(this->coll->isCUDA){
    cu->d_MV(in_vec, out_vec, N, Cval, Ccol, Cptr);
  }else{
    if(mix){
#pragma omp parallel for reduction(+:tmp) schedule(static) firstprivate(out_vec, val, in_vec) lastprivate(out_vec) num_threads(this->coll->OMPThread)
      for(long int i=0; i<N; i++){
        tmp = 0.0;
        for(long int j=ptr[i]; j<ptr[i+1]; j++){
          tmp += static_cast<float>(val[j]) * static_cast<float>(in_vec[col[j]]);
        }
        out_vec[i] = static_cast<float>(tmp);
      }
    }else{
#pragma omp parallel for reduction(+:tmp) schedule(static) firstprivate(out_vec, val, in_vec) lastprivate(out_vec) num_threads(this->coll->OMPThread)
      for(long int i=0; i<N; i++){
        tmp = 0.0;
        for(long int j=ptr[i]; j<ptr[i+1]; j++){
          tmp += val[j] * in_vec[col[j]];
        }
        out_vec[i] = tmp;
      }
    }
  }
}

template <typename T>
void blas<T>::MtxVec_mult(T *in_vec, int xindex, int xsize, T *out_vec){
  T tmp = 0.0;
  T *val=this->coll->val;
  int *ptr=this->coll->ptr;
  int *col=this->coll->col;
  long int N = this->coll->N;

  if(mix){
#pragma omp parallel for reduction(+:tmp) schedule(static) firstprivate(out_vec, val, in_vec) lastprivate(out_vec) num_threads(this->coll->OMPThread)
    for(long int i=0; i<N; i++){
      tmp = 0.0;
      for(long int j=ptr[i]; j<ptr[i+1]; j++){
        tmp += static_cast<float>(val[j]) * static_cast<float>(in_vec[xindex*xsize+col[j]]);
      }
      out_vec[i] = static_cast<float>(tmp);
    }
  }else{
#pragma omp parallel for reduction(+:tmp) schedule(static) firstprivate(out_vec, val, in_vec) lastprivate(out_vec) num_threads(this->coll->OMPThread)
    for(long int i=0; i<N; i++){
      tmp = 0.0;
      for(long int j=ptr[i]; j<ptr[i+1]; j++){
        tmp += val[j] * in_vec[xindex*xsize+col[j]];
      }
      out_vec[i] = tmp;
    }
  }
}

template <typename T>
void blas<T>::MtxVec_mult(T *Tval, int *Tcol, int *Tptr, T *in_vec, T *out_vec){
  T tmp = 0.0;
  long int N = this->coll->N;
  
  if(mix){
#pragma omp parallel for reduction(+:tmp) schedule(static) firstprivate(out_vec, Tval, in_vec) lastprivate(out_vec) num_threads(this->coll->OMPThread)
    for(long int i=0; i<N; i++){
      tmp = 0.0;
      for(long int j=Tptr[i]; j<Tptr[i+1]; j++){
        tmp += static_cast<float>(Tval[j]) * static_cast<float>(in_vec[Tcol[j]]);
      }
      out_vec[i] = static_cast<float>(tmp);
    }
  }else{
#pragma omp parallel for reduction(+:tmp) schedule(static) firstprivate(out_vec, Tval, in_vec) lastprivate(out_vec) num_threads(this->coll->OMPThread)
    for(long int i=0; i<N; i++){
      tmp = 0.0;
      for(long int j=Tptr[i]; j<Tptr[i+1]; j++){
        tmp += Tval[j] * in_vec[Tcol[j]];
      }
      out_vec[i] = tmp;
    }
  }
}


template <typename T>
void blas<T>::Vec_sub(T *x, T *y, T *out){
  if(mix){
    for(long int i=0; i<this->coll->N; i++){
      out[i] = static_cast<float>(x[i]) - static_cast<float>(y[i]);
    }
  }else{
    for(long int i=0; i<this->coll->N; i++){
      out[i] = x[i] - y[i];
    }
  }
}

template <typename T>
void blas<T>::Vec_add(T *x, T *y, T *out){
  if(mix){
    for(long int i=0; i<this->coll->N; i++){
      out[i] = static_cast<float>(x[i]) + static_cast<float>(y[i]);
    }
  }else{
    for(long int i=0; i<this->coll->N; i++){
      out[i] = x[i] + y[i];
    }
  }
}

template <typename T>
T blas<T>::dot(T *x, T *y){
  T tmp = 0.0;
  int N = this->coll->N;
  if(mix){
#pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
    for(long int i=0; i<N; i++){
      tmp += static_cast<float>(x[i]) * static_cast<float>(y[i]);
    }
  }else{
#pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
    for(long int i=0; i<N; i++){
      tmp += x[i] * y[i];
    }
  }
  return tmp;
}

template <typename T>
T blas<T>::dot(T *x, T *y, const long int size){
  T tmp = 0.0;
  if(mix){
#pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
    for(long int i=0; i<size; i++){
      tmp += static_cast<float>(x[i]) * static_cast<float>(y[i]);
    }
  }else{
#pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
    for(long int i=0; i<size; i++){
      tmp += x[i] * y[i];
    }
  }
  return tmp;
}

template <typename T>
T blas<T>::dot(T *x, T *y, int xindex, int xsize){
  T tmp = 0.0;
  int N = this->coll->N;
  if(mix){
#pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
    for(long int i=0; i<N; i++){
      tmp += static_cast<float>(x[i]) * static_cast<float>(y[xindex*xsize+i]);
    }
  }else{
#pragma omp parallel for schedule(static) reduction(+:tmp) num_threads(this->coll->OMPThread)
    for(long int i=0; i<N; i++){
      tmp += x[i] * y[xindex*xsize+i];
    }
  }
  return tmp;
}

template <typename T>
void blas<T>::Scalar_ax(T a, T *x, T *out){
  if(mix){
    for(long int i=0; i<this->coll->N; i++){
      out[i] = static_cast<float>(a) * static_cast<float>(x[i]);
    }
  }else{
    for(long int i=0; i<this->coll->N; i++){
      out[i] = a * x[i];
    }
  }
}

template <typename T>
void blas<T>::Scalar_ax(T a, T *x, int xindex, int xsize, T *out){
  if(mix){
#pragma omp parallel for schedule(static) firstprivate(out, a, x) lastprivate(out) num_threads(this->coll->OMPThread)
    for(long int i=0; i<this->coll->N; i++){
      out[i] = static_cast<float>(a) * static_cast<float>(x[xindex*xsize+i]);
    }
  }else{
#pragma omp parallel for schedule(static) firstprivate(out, a, x) lastprivate(out) num_threads(this->coll->OMPThread)
    for(long int i=0; i<this->coll->N; i++){
      out[i] = a * x[xindex*xsize+i];
    }
  }
}

template <typename T>
void blas<T>::Scalar_axy(T a, T *x, T *y, T *out){
  T tmp;
  if(mix){
    for(long int i=0; i<this->coll->N; i++){
      tmp = static_cast<float>(y[i]);
      out[i] = (static_cast<float>(a) * static_cast<float>(x[i])) + static_cast<float>(tmp);
    }
  }else{
    for(long int i=0; i<this->coll->N; i++){
      tmp = y[i];
      out[i] = (a * x[i]) + tmp;
    }
  }
}

template <typename T>
void blas<T>::Scalar_x_div_a(T *x, T a, T *out){
  if(mix){
    for(long int i=0; i<this->coll->N; i++){
      out[i] = static_cast<float>(x[i]) / static_cast<float>(a);
    }
  }else{
    for(long int i=0; i<this->coll->N; i++){
      out[i] = x[i] / a;
    }
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

  T *val=this->coll->val;
  int *ptr=this->coll->ptr;
  int *col=this->coll->col;
  long int N = this->coll->N;
  T *bvec=this->coll->bvec;

  if(mix){
    for(long int i=0; i<N; i++){
      tmp1= 0.0;
      tmp2= 0.0;
      for(long int j=ptr[i]; j<ptr[i+1]; j++){
        tmp1 += static_cast<float>(val[j]) * static_cast<float>(x_last[col[j]]);
        tmp2 += static_cast<float>(val[j]) * static_cast<float>(x_0[col[j]]);
      }
      Ax[i] = static_cast<float>(bvec[i]) - static_cast<float>(tmp1);
      Ax_0[i] = static_cast<float>(bvec[i]) - static_cast<float>(tmp2);
    }
    tmp1 = norm_2(Ax);
    tmp2 = norm_2(Ax_0);
    tmp3 = log10(static_cast<float>(tmp1)/static_cast<float>(tmp2));
  }else{
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
  }

  delete[] Ax;
  delete[] Ax_0;

  return tmp3;
}

template <typename T>
void blas<T>::Hye(T *h, T *y, T *e, const long int size){
  T tmp;
  // for(long int i=0; i<size; i++){
  //   y[i] = 0.0;
  // }
  std::memset(y, 0, sizeof(T)*size);
  if(mix){
    for(long int i=size-1; i>=0; i--){
      tmp = 0.0;
      for(long int j=i+1; j<size; j++){
        tmp += static_cast<float>(y[j]) * static_cast<float>(h[i*(this->coll->N)+j]);
      }
      y[i] = (static_cast<float>(e[i]) - static_cast<float>(tmp)) / static_cast<float>(h[i*(this->coll->N)+i]);
    }
  }else{
    for(long int i=size-1; i>=0; i--){
      tmp = 0.0;
      for(long int j=i+1; j<size; j++){
        tmp += y[j] * h[i*(this->coll->N)+j];
      }
      y[i] = (e[i] - tmp) / h[i*(this->coll->N)+i];
    }
  }

}

template <typename T>
void blas<T>::Vec_copy(T *in, T *out){
  if(mix){
    for(long int i=0; i<this->coll->N; i++){
      out[i] = static_cast<float>(in[i]);
    }
  }else{
    for(long int i=0; i<this->coll->N; i++){
      out[i] = in[i];
    }
  }
}

template <typename T>
void blas<T>::Vec_copy(T *in, int xindex, int xsize, T *out){
  if(mix){
    for(long int i=0; i<this->coll->N; i++){
      out[i] = static_cast<float>(in[xindex*xsize+i]);
    }
  }else{
    for(long int i=0; i<this->coll->N; i++){
      out[i] = in[xindex*xsize+i];
    }
  }
}

template <typename T>
void blas<T>::Vec_copy(T *in, T *out, int xindex, int xsize){
  if(mix){
    for(long int i=0; i<this->coll->N; i++){
      out[xindex*xsize+i] = static_cast<float>(in[i]);
    }
  }else{
    for(long int i=0; i<this->coll->N; i++){
      out[xindex*xsize+i] = in[i];
    }
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

  if(mix){
#pragma omp parallel for reduction(+:tmp1, tmp2) schedule(static) firstprivate(Ar, Ap, val, pvec, rvec) lastprivate(Ar, Ap) num_threads(this->coll->OMPThread)
    for(long int i=0; i<N; i++){
      tmp1 = 0.0;
      tmp2 = 0.0;
      for(int j=ptr[i]; j<ptr[i+1]; j++){
        tmp1 += static_cast<float>(val[j]) * static_cast<float>(rvec[col[j]]);
        tmp2 += static_cast<float>(val[j]) * static_cast<float>(pvec[col[j]]);

      }
      Ar[0][i] = static_cast<float>(tmp1);
      Ap[0][i] = static_cast<float>(tmp2);

    }
    for(int ii=1; ii<2*kskip+2; ii++){
#pragma omp parallel for reduction(+:tmp1, tmp2) schedule(static) firstprivate(Ar, Ap, val) lastprivate(Ar, Ap) num_threads(this->coll->OMPThread)
      for(long int i=0; i<N; i++){
        tmp1 = 0.0;
        tmp2 = 0.0;
        for(int j=ptr[i]; j<ptr[i+1]; j++){
          if(ii<2*kskip){
            tmp1 += static_cast<float>(val[j]) * static_cast<float>(Ar[(ii-1)][col[j]]);
          }
          tmp2 += static_cast<float>(val[j]) * (Ap[(ii-1)][col[j]]);
        }
        if(ii<2*kskip){
          Ar[(ii)][i] = static_cast<float>(tmp1);
        }
        Ap[(ii)][i] = tmp2;
      }
    }

  }else{
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

}

template <typename T>
void blas<T>::Kskip_cg_innerProduce(T *delta, T *eta, T *zeta, T **Ar, T **Ap, T *rvec, T *pvec, const int kskip){
  T tmp1=0.0;
  T tmp2=0.0;
  T tmp3=0.0;
  long int N = this->coll->N;

  if(mix){
#pragma omp parallel for reduction(+:tmp1, tmp2, tmp3) schedule(static) firstprivate(delta, eta, zeta, Ar, rvec, Ap, pvec) lastprivate(delta, eta, zeta) num_threads(this->coll->OMPThread)
    for(int i=0; i<2*kskip+2; i++){
      tmp1=0.0;
      tmp2=0.0;
      tmp3=0.0;
      for(long int j=0; j<N; j++){
        if(i<2*kskip){
          tmp1 += static_cast<float>(rvec[j]) * static_cast<float>(Ar[i][j]);
        }
        if(i<2*kskip+1){
          tmp2 += static_cast<float>(rvec[j]) * Ap[i][j];
        }
        tmp3 += static_cast<float>(pvec[j]) * Ap[i][j];
      }
      if(i<2*kskip){
        delta[i] = static_cast<float>(tmp1);
      }
      if(i<2*kskip+1){
        eta[i] = tmp2;
      }
      zeta[i] = tmp3;
    }

  }else{
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
}

template <typename T>
void blas<T>::Kskip_kskipBicg_base(T **Ar, T **Ap, T *rvec, T *pvec, const int kskip){
  T tmp1 = 0.0;
  T tmp2 = 0.0;
  T *val=this->coll->val;
  int *ptr=this->coll->ptr;
  int *col=this->coll->col;
  long int N = this->coll->N;

  if(mix){
#pragma omp parallel for reduction(+:tmp1, tmp2) schedule(static) firstprivate(Ar, Ap, val, pvec, rvec) lastprivate(Ar, Ap) num_threads(this->coll->OMPThread)
    for(long int i=0; i<N; i++){
      tmp1 = 0.0;
      tmp2 = 0.0;
      for(int j=ptr[i]; j<ptr[i+1]; j++){
        tmp1 += static_cast<float>(val[j]) * static_cast<float>(rvec[col[j]]);
        tmp2 += static_cast<float>(val[j]) * static_cast<float>(pvec[col[j]]);
      }
      Ar[0][i] = static_cast<float>(tmp1);
      Ap[0][i] = static_cast<float>(tmp2);
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

  }else{
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
}

template <typename T>
void blas<T>::Kskip_kskipBicg_innerProduce(T *theta, T *eta, T *rho, T *phi, T **Ar, T **Ap, T *rvec, T *pvec, T *r_vec, T *p_vec, const int kskip){
  T tmp1=0.0;
  T tmp2=0.0;
  T tmp3=0.0;
  T tmp4=0.0;
  long int N = this->coll->N;

  if(mix){
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

  }else{
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
}

template <typename T>
void blas<T>::Gmres_sp_1(int k, T *x, T *y, T *out){
  int N = this->coll->N;

  if(mix){
#pragma omp parallel for schedule(static) firstprivate(out, x, y) lastprivate(out) num_threads(this->coll->OMPThread)
    for(long int j=0; j<N; j++){
      for(int i=0; i<=k; i++){
        out[j] -= static_cast<float>(x[i*N+k]) * static_cast<float>(y[i*N+j]);
      }
    }
  }else{
#pragma omp parallel for schedule(static) firstprivate(out, x, y) lastprivate(out) num_threads(this->coll->OMPThread)
    for(long int j=0; j<N; j++){
      for(int i=0; i<=k; i++){
        out[j] -= x[i*N+k] * y[i*N+j];
      }
    }
  }
}


#endif //BLAS_HPP_INCLUDED__


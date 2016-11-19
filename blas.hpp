#ifndef BLAS_HPP_INCLUDED__
#define BLAS_HPP_INCLUDED__

#include <fstream>
#include <ctime>
#include <string>
#include "blas.hpp"
#include "solver_collection.hpp"
#include "cmath"

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

    void MtxVec_mult(T *Tval, int *Tcol, int *Tptr, T *in_vec, T *out_vec);

    void Vec_sub(T *x, T *y, T *out);

    void Vec_add(T *x, T *y, T *out);

    T dot(T *x, T *y);

    void Scalar_ax(T a, T *x, T *out);

    void Scalar_axy(T a, T *x, T *y, T *out);

    double Check_error(T *x_last, T *x_0);

    void Hye(T *h, T *y, T *e, const long int size);

    void Vec_copy(T *in, T *out);
};

template <typename T>
blas<T>::blas(collection<T> *col){
  coll = col;
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
  for(long int i=0; i<this->coll->N; i++){
    tmp = 0.0;
    for(long int j=this->coll->ptr[i]; j<this->coll->ptr[i+1]; j++){
      tmp += this->coll->val[j] * in_vec[this->coll->col[j]];
    }
    out_vec[i] = tmp;
  }
}

template <typename T>
void blas<T>::MtxVec_mult(T *Tval, int *Tcol, int *Tptr, T *in_vec, T *out_vec){
  T tmp = 0.0;
  for(long int i=0; i<this->coll->N; i++){
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
  for(long int i=0; i<this->coll->N; i++){
    tmp += x[i] * y[i];
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
void blas<T>::Scalar_axy(T a, T *x, T *y, T *out){
  T tmp;
  for(long int i=0; i<this->coll->N; i++){
    tmp = y[i];
    out[i] = (a * x[i]) + tmp;
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
  for(long int i=0; i<this->coll->N; i++){
    y[i] = 0.0;
  }
  for(long int i=this->coll->N-1; i>=0; i--){
    tmp = 0.0;
    for(long int j=i+1; j<this->coll->N; j++){
      tmp += y[i] * h[i*size+j];
    }
    y[i] = (e[i] - tmp) / h[i*size+i];
  }
}

template <typename T>
void blas<T>::Vec_copy(T *in, T *out){
  for(long int i=0; i<this->coll->N; i++){
    out[i] = in[i];
  }
}
#endif //BLAS_HPP_INCLUDED__


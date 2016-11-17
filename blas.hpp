#include "solver_collection.hpp"
#include "cmath"

class blas {
  collection *coll;
  public:

  blas(collection *col){
    coll = col;
  }

std::ofstream *output(std::string name);
std::string get_date_time();

  template <typename T1>
    T1 norm_1(T1 *v, const long int N){
      T1 tmp = 0.0;
      for(long int i=0;i<N;i++){
        tmp += fabs(v[i]);
      }
      return tmp;
    }

  template <typename T1>
    T1 norm_2(T1 *v, const long int N){
      T1 tmp = 0.0;
      for(long int i=0;i<N;i++){
        tmp += v[i] * v[i];
      }
      return sqrt(tmp);
    }

  template <typename T1>
    void MtxVec_mult(T1 *in_vec, T1 *out_vec, const long int N){
      T1 tmp = 0.0;
      for(long int i=0; i<N; i++){
        tmp = 0.0;
        for(long int j=this->coll->ptr[i]; j<this->coll->ptr[i+1]; j++){
          tmp += this->coll->val[j] * in_vec[this->coll->col[j]];
        }
        out_vec[i] = tmp;
      }
    }

  template <typename T1>
    void Vec_sub(T1 *x, T1 *y, T1 *out, const long int N){
      for(long int i=0; i<N; i++){
        out[i] = x[i] - y[i];
      }
    }

  template <typename T1>
    void Vec_add(T1 *x, T1 *y, T1 *out, const long int N){
      for(long int i=0; i<N; i++){
        out[i] = x[i] + y[i];
      }
    }

  template <typename T1>
    T1 dot(T1 *x, T1 *y, const long int N){
      T1 tmp = 0.0;
      for(long int i=0; i<N; i++){
        tmp += x[i] * y[i];
      }
      return tmp;
    }

  template <typename T1>
    void Scalar_ax(T1 a, T1 *x, T1 *out, const long int N){
      for(long int i=0; i<N; i++){
        out[i] = a * x[i];
      }
    }

  template <typename T1>
    void Scalar_axy(T1 a, T1 *x, T1 *y, T1 *out, const long int N){
      T1 tmp;
      for(long int i=0; i<N; i++){
        tmp = y[i];
        out[i] = (a * x[i]) + tmp;
      }
    }

  template <typename T1>
    double Check_error(T1 *x_last, T1 *x_0, const long int N){
      T1 tmp1 = 0.0;
      T1 tmp2 = 0.0;
      double tmp3 = 0.0;

      T1 *Ax, *Ax_0;

      Ax = new T1[N];
      Ax_0 = new T1[N];

      for(long int i=0; i<N; i++){
        tmp1= 0.0;
        tmp2= 0.0;
        for(long int j=this->coll->ptr[i]; j<this->coll->ptr[i+1]; j++){
          tmp1 += this->coll->val[j] * x_last[this->coll->col[j]];
          tmp2 += this->coll->val[j] * x_0[this->coll->col[j]];
        }
        Ax[i] = this->coll->bvec[i] - tmp1;
        Ax_0[i] = this->coll->bvec[i] - tmp2;
      }
      tmp1 = norm_2<T1>(Ax, N);
      tmp2 = norm_2<T1>(Ax_0, N);
      tmp3 = log10(tmp1/tmp2);

      delete[] Ax;
      delete[] Ax_0;

      return tmp3;
    }

  template <typename T1>
    void Hye(T1 *h, T1 *y, T1 *e, const long int N, const long int size){
      T1 tmp;
      for(long int i=0; i<N; i++){
        y[i] = 0.0;
      }
      for(long int i=N-1; i>=0; i--){
        tmp = 0.0;
        for(long int j=i+1; j<N; j++){
          tmp += y[i] * h[i*size+j];
        }
        y[i] = (e[i] - tmp) / h[i*size+i];
      }
    }

  template <typename T1>
    void Vec_copy(T1 *in, T1 *out, const long int N){
      for(long int i=0; i<N; i++){
        out[i] = in[i];
      }
    }




};


#ifndef gcr_HPP_INCLUDED__
#define gcr_HPP_INCLUDED__

#include <iomanip>
#include <fstream>
#include "solver_collection.hpp"
#include "blas.hpp"

template <typename T>
class gcr {
  private:
    collection<T> *coll;
    blas<T> *bs;

    long int loop, iloop, kloop;
    T *xvec, *bvec;
    T *rvec, *Av, *x_0, *qq;
    T **qvec, **pvec;
    T dot, dot_tmp, error;
    T alpha, beta, bnorm, rnorm;

    int maxloop;
    double eps;
    int restart;
    bool isVP, isVerbose, isCUDA, isInner;

    int exit_flag;
    T test_error;
    bool out_flag;

    int N;

    std::ofstream f_his;
    std::ofstream f_x;

  public:
    gcr(collection<T> *coll, T *bvec, T *xvec);
    ~gcr();
    int solve();
};

template <typename T>
gcr<T>::gcr(collection<T> *coll, T *bvec, T *xvec){
  this->coll = coll;
  bs = new blas<T>(this->coll);

  if(isVP && this->coll->isInner ){
    maxloop = this->coll->innerMaxLoop;
    eps = this->coll->innerEps;
    restart = this->coll->innerRestart;
  }else{
    maxloop = this->coll->outerMaxLoop;
    eps = this->coll->outerEps;
    restart = this->coll->outerRestart;
  }

  loop = 1;

  N = this->coll->N;

  this->xvec = xvec;
  this->bvec = bvec;

  rvec = new T [N];
  Av = new T [N];
  x_0 = new T [N];
  qq = new T [restart];
  qvec = new T* [restart];
  pvec = new T* [restart];
  for(long int i=0; i<restart; i++){
    qvec[i] = new T [N];
    pvec[i] = new T [N];
  }

  exit_flag = 2;
  isVP = this->coll->isVP;
  isVerbose = this->coll->isVerbose;
  isCUDA = this->coll->isCUDA;
  isInner = this->coll->isInner;

  for(long int i=0; i<N; i++){
    rvec[i] = 0.0;
    Av[i] = 0.0;
    xvec[i] = 0.0;
  }
  for(long int i=0; i<restart; i++){
    qq[i] = 0.0;
  }
  for(int i=0; i<restart; i++){
    for(long int j=0; j<N; j++){
      qvec[i][j] = 0.0;
      pvec[i][j] = 0.0;
    }
  }

  f_his.open("./output/GCR_his.txt");
  if(!f_his.is_open()){
    std::cerr << "File open error" << std::endl;
    exit(-1);
  }

  f_x.open("./output/GCR_xvec.txt");
  if(!f_x.is_open()){
    std::cerr << "File open error" << std::endl;
    exit(-1);
  }

  out_flag = false;

}

template <typename T>
gcr<T>::~gcr(){
  delete this->bs;
  delete[] rvec;
  delete[] Av;
  delete[] qq;
  delete[] x_0;
  for(int i=0; i<restart; i++){
    delete[] qvec[i];
    delete[] pvec[i];
  }
  delete[] qvec;
  delete[] pvec;
  f_his.close();
  f_x.close();
}

template <typename T>
int gcr<T>::solve(){

  //x_0 = x
  bs->Vec_copy(xvec, x_0, N);

  //b 2norm
  bnorm = bs->norm_2(bvec, N);

  while(loop<maxloop){
    //Ax
    if(isCUDA){

    }else{
      bs->MtxVec_mult(xvec, Av, N);
    }

    //r=b-Ax
    bs->Vec_sub(bvec, Av, rvec, N);

    //p=r
    bs->Vec_copy(rvec, pvec[0], N);

    //Ap
    if(isCUDA){

    }else{
      bs->MtxVec_mult(pvec[0], qvec[0], N);
    }

    for(kloop=0; kloop<restart; kloop++){
      rnorm = bs->norm_2(rvec, N);
      error = rnorm / bnorm;
      if(!isInner){
        if(isVerbose){
          std::cout << loop << " " << std::scientific << std::setprecision(12) << std::uppercase << error << std::endl;
        }
        f_his << loop << " " << std::scientific << std::setprecision(12) << std::uppercase << error << std::endl;
      }
      if(error <= eps){
        exit_flag = 0;
        out_flag = true;
        break;
      }else if(loop >= maxloop){
        exit_flag = 2;
        out_flag = true;
        break;
      }
      loop++;

      //(q, q)
      if(isCUDA){

      }else{
        dot_tmp = bs->dot(qvec[kloop], qvec[kloop], N);
      }
      qq[kloop] = dot_tmp;

      //alpha = (r, q)/(q, q)
      if(isCUDA){

      }else{
        dot_tmp = bs->dot(rvec, qvec[kloop], N);
      }
      alpha = dot_tmp / qq[kloop];

      //x = alpha * pvec[k] + xvec
      bs->Scalar_axy(alpha, pvec[kloop], xvec, xvec, N);
      if(kloop == restart-1){
        break;
      }

      //r = -alpha * qvec[k] + rvec
      bs->Scalar_axy(-alpha, qvec[kloop], rvec, rvec, N);

      //Ar
      if(isCUDA){

      }else{
        bs->MtxVec_mult(rvec, Av, N);
      }

      //init p[k+1] q[k+1]
      for(long int i=0; i<N; i++){
        pvec[kloop+1][i] = 0.0;
        qvec[kloop+1][i] = 0.0;
      }

      for(iloop=0; iloop<=kloop; iloop++){
        //beta = -(Av, qvec) / (q, q)
        if(isCUDA){

        }else{
          dot_tmp = bs->dot(Av, qvec[iloop], N);
        }
        beta = -(dot_tmp) / qq[iloop];

        //pvec[k+1] = beta * pvec[i] + pvec[k+1]
        bs->Scalar_axy(beta, pvec[iloop], pvec[kloop+1], pvec[kloop+1], N);
        //qvec[k+1] = beta * qvec[i] + qvec[k+1]
        bs->Scalar_axy(beta, qvec[iloop], qvec[kloop+1], qvec[kloop+1], N);
      }
      //p[k+1] = r + p[k+1]
      bs->Vec_add(rvec, pvec[kloop+1], pvec[kloop+1], N);
      //q[k+1] = Av + q[k+1]
      bs->Vec_add(Av, qvec[kloop+1], qvec[kloop+1], N);
    }
    if(out_flag){
      break;
    }
  }
  if(!isInner){
    test_error = bs->Check_error(xvec, x_0, N);
    std::cout << "|b-ax|2/|b|2 = " << std::fixed << std::setprecision(1) << test_error << std::endl;
    std::cout << "loop = " << loop << std::endl;

    for(long int i=0; i<N; i++){
      f_x << i << " " << std::scientific << std::setprecision(12) << std::uppercase << xvec[i] << std::endl;
    }
  }else{

  }

  return exit_flag;
}
#endif //gcr_HPP_INCLUDED__


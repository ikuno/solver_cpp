#ifndef gcr_HPP_INCLUDED__
#define gcr_HPP_INCLUDED__

#include <iomanip>
#include <fstream>
#include "solver_collection.hpp"
#include "blas.hpp"
#include "times.hpp"

template <typename T>
class gcr {
  private:
    collection<T> *coll;
    blas<T> *bs;
    times time;

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
    gcr(collection<T> *coll, T *bvec, T *xvec, bool inner);
    ~gcr();
    int solve();
};

template <typename T>
gcr<T>::gcr(collection<T> *coll, T *bvec, T *xvec, bool inner){
  this->coll = coll;
  bs = new blas<T>(this->coll);

  exit_flag = 2;
  isVP = this->coll->isVP;
  isVerbose = this->coll->isVerbose;
  isCUDA = this->coll->isCUDA;
  isInner = inner;


  if(isVP && isInner ){
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

  
  std::memset(rvec, 0, sizeof(T)*N);
  std::memset(Av, 0, sizeof(T)*N);
  std::memset(xvec, 0, sizeof(T)*N);
  std::memset(qq, 0, sizeof(T)*restart);

  for(int i=0; i<restart; i++){
    std::memset(qvec[i], 0, sizeof(T)*N);
    std::memset(pvec[i], 0, sizeof(T)*N);
  }

  if(!isInner){
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

  time.start();

  //x_0 = x
  bs->Vec_copy(xvec, x_0);

  //b 2norm
  bnorm = bs->norm_2(bvec);

  while(loop<maxloop){
    //Ax
    if(isCUDA){

    }else{
      bs->MtxVec_mult(xvec, Av);
    }

    //r=b-Ax
    bs->Vec_sub(bvec, Av, rvec);

    //p=r
    bs->Vec_copy(rvec, pvec[0]);

    //Ap
    if(isCUDA){

    }else{
      bs->MtxVec_mult(pvec[0], qvec[0]);
    }

    for(kloop=0; kloop<restart; kloop++){
      rnorm = bs->norm_2(rvec);
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
        dot_tmp = bs->dot(qvec[kloop], qvec[kloop]);
      }
      qq[kloop] = dot_tmp;

      //alpha = (r, q)/(q, q)
      if(isCUDA){

      }else{
        dot_tmp = bs->dot(rvec, qvec[kloop]);
      }
      alpha = dot_tmp / qq[kloop];

      //x = alpha * pvec[k] + xvec
      bs->Scalar_axy(alpha, pvec[kloop], xvec, xvec);
      if(kloop == restart-1){
        break;
      }

      //r = -alpha * qvec[k] + rvec
      bs->Scalar_axy(-alpha, qvec[kloop], rvec, rvec);

      //Ar
      if(isCUDA){

      }else{
        bs->MtxVec_mult(rvec, Av);
      }

      //init p[k+1] q[k+1]
      std::memset(pvec[kloop+1], 0, sizeof(T)*N);
      std::memset(qvec[kloop+1], 0, sizeof(T)*N);

      for(iloop=0; iloop<=kloop; iloop++){
        //beta = -(Av, qvec) / (q, q)
        if(isCUDA){

        }else{
          dot_tmp = bs->dot(Av, qvec[iloop]);
        }
        beta = -(dot_tmp) / qq[iloop];

        //pvec[k+1] = beta * pvec[i] + pvec[k+1]
        bs->Scalar_axy(beta, pvec[iloop], pvec[kloop+1], pvec[kloop+1]);
        //qvec[k+1] = beta * qvec[i] + qvec[k+1]
        bs->Scalar_axy(beta, qvec[iloop], qvec[kloop+1], qvec[kloop+1]);
      }
      //p[k+1] = r + p[k+1]
      bs->Vec_add(rvec, pvec[kloop+1], pvec[kloop+1]);
      //q[k+1] = Av + q[k+1]
      bs->Vec_add(Av, qvec[kloop+1], qvec[kloop+1]);
    }
    if(out_flag){
      break;
    }
  }

  time.end();

  if(!isInner){
    test_error = bs->Check_error(xvec, x_0);
    std::cout << "|b-ax|2/|b|2 = " << std::fixed << std::setprecision(1) << test_error << std::endl;
    std::cout << "loop = " << loop << std::endl;
    std::cout << "time = " << std::setprecision(6) << time.getTime() << std::endl;

    for(long int i=0; i<N; i++){
      f_x << i << " " << std::scientific << std::setprecision(12) << std::uppercase << xvec[i] << std::endl;
    }
  }else{
    if(isVerbose){
      if(exit_flag==0){
        std::cout << GREEN << "\t" <<  loop << " = " << std::scientific << std::setprecision(12) << std::uppercase << error << RESET << std::endl;
      }else if(exit_flag==2){
        std::cout << RED << "\t" << loop << " = " << std::scientific << std::setprecision(12) << std::uppercase << error << RESET << std::endl;
      }else{
        std::cout << RED << " ERROR " << loop << RESET << std::endl;
      }
    }
  }

  return exit_flag;
}
#endif //gcr_HPP_INCLUDED__


#ifndef CG_HPP_INCLUDED__
#define CG_HPP_INCLUDED__

#include <iomanip>
#include <fstream>
#include "solver_collection.hpp"
#include "blas.hpp"
#include "times.hpp"

template <typename T>
class cg {
  private:
    collection<T> *coll;
    blas<T, T> *bs;
    blas<T, float> *bs_f;
    times time;
    
    long int loop;
    T *xvec, *bvec;
    T *rvec, *pvec, *mv, *x_0, dot, error;
    T alpha, beta, bnorm, rnorm;
    T rr, rr2;

    int maxloop;
    double eps;
    bool isVP, isVerbose, isCUDA, isInner;

    int exit_flag;
    T test_error;

    int N;

    std::ofstream f_his;
    std::ofstream f_x;

  public:
    cg(collection<T> *coll, T *bvec, T *xvec, bool inner);
    ~cg();
    int solve();
};

template <typename T>
cg<T>::cg(collection<T> *coll, T *bvec, T *xvec, bool inner){
  this->coll = coll;
  bs = new blas<T>(this->coll);
  bs_f = new blas<T, float>(this->coll);
  
  N = this->coll->N;
  rvec = new T [N];
  pvec = new T [N];
  mv = new T [N];
  x_0 = new T [N];

  this->xvec = xvec;
  this->bvec = bvec;

  exit_flag = 2;
  isVP = this->coll->isVP;
  isVerbose = this->coll->isVerbose;
  isCUDA = this->coll->isCUDA;
  isInner = inner;

  if(isVP && isInner ){
    maxloop = this->coll->innerMaxLoop;
    eps = this->coll->innerEps;
  }else{
    maxloop = this->coll->outerMaxLoop;
    eps = this->coll->outerEps;
  }

  std::memset(rvec, 0, sizeof(T)*N);
  std::memset(pvec, 0, sizeof(T)*N);
  std::memset(mv, 0, sizeof(T)*N);
  std::memset(xvec, 0, sizeof(T)*N);

  if(!isInner){
    f_his.open("./output/CG_his.txt");
    if(!f_his.is_open()){
      std::cerr << "File open error" << std::endl;
      exit(-1);
    }

    f_x.open("./output/CG_xvec.txt");
    if(!f_x.is_open()){
      std::cerr << "File open error" << std::endl;
      exit(-1);
    }
  }

}

template <typename T>
cg<T>::~cg(){
  delete this->bs;
  delete this->bs_f;
  delete[] rvec;
  delete[] pvec;
  delete[] mv;
  delete[] x_0;
  f_his.close();
  f_x.close();
}

template <typename T>
int cg<T>::solve(){

  time.start();

  //x_0 = x
  if(isInner){
    bs_f->Vec_copy(xvec, x_0);
  }else{
    bs->Vec_copy(xvec, x_0);
  }

  //b 2norm
  if(isInner){
    bnorm = bs_f->norm_2(bvec);
  }else{
    bnorm = bs->norm_2(bvec);
  }

  //mv = Ax
  if(isCUDA){

  }else{
    if(isInner){
      bs_f->MtxVec_mult(xvec, mv);
    }else{
      bs->MtxVec_mult(xvec, mv);
    }
  }

  //r = b - Ax
  if(isInner){
    bs_f->Vec_sub(bvec, mv, rvec);
  }else{
    bs->Vec_sub(bvec, mv, rvec);
  }


  //p = r
  if(isInner){
    bs_f->Vec_copy(rvec, pvec);
  }else{
    bs->Vec_copy(rvec, pvec);
  }

  //r dot
  if(isCUDA){

  }else{
    if(isInner){
      rr = bs_f->dot(rvec, rvec);
    }else{
      rr = bs->dot(rvec, rvec);
    }
  }

  for(loop=1; loop<=maxloop; loop++){
    if(isInner){
      rnorm = bs_f->norm_2(rvec);
    }else{
      rnorm = bs->norm_2(rvec);
    }
    error = rnorm/bnorm;
    if(!isInner){
      if(isVerbose){
        std::cout << loop << " " << std::scientific << std::setprecision(12) << std::uppercase << error << std::endl;
      }
      f_his << loop << " " << std::scientific << std::setprecision(12) << std::uppercase << error << std::endl;
    }


    if(error <= eps){
      exit_flag = 0;
      break;
    }

    //mv = Ap
    if(isCUDA){

    }else{
      if(isInner){
        bs_f->MtxVec_mult(pvec, mv);
      }else{
        bs->MtxVec_mult(pvec, mv);
      }
    }

    //alpha = (r,r) / (p,ap)
    if(isCUDA){
    }else{
      if(isInner){
        dot = bs_f->dot(pvec, mv);
      }else{
        dot = bs->dot(pvec, mv);
      }
    }
    alpha = rr / dot;

    //x = alpha * pvec + x
    if(isInner){
      bs->Scalar_axy(alpha, pvec, xvec, xvec);
    }else{
      bs_f->Scalar_axy(alpha, pvec, xvec, xvec);
    }

    //r = -alpha * AP(mv) + r
    if(isInner){
      bs_f->Scalar_axy(-alpha, mv, rvec, rvec);
    }else{
      bs->Scalar_axy(-alpha, mv, rvec, rvec);
    }

    //rr2 dot
    if(isCUDA){

    }else{
      if(isInner){
        rr2 = bs_f->dot(rvec, rvec);
      }else{
        rr2 = bs->dot(rvec, rvec);
      }
    }

    beta = rr2/rr;

    rr = rr2;

    //p = beta * p + r
    if(isInner){
      bs_f->Scalar_axy(beta, pvec, rvec, pvec);
    }else{
      bs->Scalar_axy(beta, pvec, rvec, pvec);
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
#endif //CG_HPP_INCLUDED__


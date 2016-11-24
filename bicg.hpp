#ifndef BICG_HPP_INCLUDED__
#define BICG_HPP_INCLUDED__

#include <iomanip>
#include <fstream>
#include "solver_collection.hpp"
#include "blas.hpp"
#include "times.hpp"

template <typename T>
class bicg {
  private:
    collection<T> *coll;
    blas<T> *bs;
    times time;
    
    long int loop;
    T *xvec, *bvec;
    T *rvec, *r_vec, *pvec, *p_vec, *mv, *x_0, dot, error;
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
    bicg(collection<T> *coll, T *bvec, T *xvec, bool inner);
    ~bicg();
    int solve();
};

template <typename T>
bicg<T>::bicg(collection<T> *coll, T *bvec, T *xvec, bool inner){
  this->coll = coll;
  bs = new blas<T>(this->coll);

  N = this->coll->N;
  rvec = new T [N];
  pvec = new T [N];
  r_vec = new T [N];
  p_vec = new T [N];
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
  std::memset(r_vec, 0, sizeof(T)*N);
  std::memset(p_vec, 0, sizeof(T)*N);
  std::memset(mv, 0, sizeof(T)*N);
  std::memset(xvec, 0, sizeof(T)*N);
  
  f_his.open("./output/BICG_his.txt");
  if(!f_his.is_open()){
    std::cerr << "File open error" << std::endl;
    exit(-1);
  }

  f_x.open("./output/BICG_xvec.txt");
  if(!f_x.is_open()){
    std::cerr << "File open error" << std::endl;
    exit(-1);
  }

}

template <typename T>
bicg<T>::~bicg(){
  delete this->bs;
  delete[] rvec;
  delete[] pvec;
  delete[] r_vec;
  delete[] p_vec;
  delete[] mv;
  delete[] x_0;
  f_his.close();
  f_x.close();
}

template <typename T>
int bicg<T>::solve(){

  time.start();

  //x_0 = x
  bs->Vec_copy(xvec, x_0);

  //b 2norm
  bnorm = bs->norm_2(bvec);

  //mv = Ax
  if(isCUDA){

  }else{
    bs->MtxVec_mult(xvec, mv);
  }

  //r = b - Ax
  bs->Vec_sub(bvec, mv, rvec);

  //r* = r
  bs->Vec_copy(rvec, r_vec);

  //p = r
  bs->Vec_copy(rvec, pvec);

  //p* = *r
  bs->Vec_copy(r_vec, p_vec);

  //r * r*
  if(isCUDA){

  }else{
    rr = bs->dot(r_vec, rvec);
  }

  for(loop=1; loop<=maxloop; loop++){
    rnorm = bs->norm_2(rvec);
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
      bs->MtxVec_mult(pvec, mv);
    }

    //alpha = (r*,r) / (p*,ap)
    if(isCUDA){
    }else{
      dot = bs->dot(p_vec, mv);
    }
    alpha = rr / dot;

    //x = alpha * pvec + x
    bs->Scalar_axy(alpha, pvec, xvec, xvec);

    //r = -alpha * AP(mv) + r
    bs->Scalar_axy(-alpha, mv, rvec, rvec);

    //mv = A(T)p*
    if(isCUDA){

    }else{
      bs->MtxVec_mult(this->coll->Tval, this->coll->Tcol, this->coll->Tptr, p_vec, mv);
    }

    //r* = r* - alpha * A(T)p*
    bs->Scalar_axy(-alpha, mv, r_vec, r_vec);

    //r * r*
    if(isCUDA){

    }else{
      rr2 = bs->dot(r_vec, rvec);
    }

    beta = rr2/rr;

    rr = rr2;

    //p = beta * p + r
    bs->Scalar_axy(beta, pvec, rvec, pvec);
    //p* = beta * p* + r*
    bs->Scalar_axy(beta, p_vec, r_vec, p_vec);

  }

  time.end();

  if(!isInner){
    test_error = bs->Check_error(xvec, x_0);
    std::cout << "|b-ax|2/|b|2 = " << std::fixed << std::setprecision(1) << test_error << std::endl;
    std::cout << "loop = " << loop << std::endl;
    std::cout << "time = " << std::setprecision(6) << time.getTime() << std::endl;
    for(long int i=0; i<N; i++){
      f_x << i << " " << std::setprecision(12) << std::uppercase << xvec[i] << std::endl;
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
#endif //BICG_HPP_INCLUDED__


#ifndef CG_HPP_INCLUDED__
#define CG_HPP_INCLUDED__

#include <iomanip>
#include <fstream>
#include "solver_collection.hpp"
#include "blas.hpp"

template <typename T>
class cg {
  private:
    collection<T> *coll;
    blas<T> *bs;
    
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
    cg(collection<T> *coll, T *bvec, T *xvec);
    ~cg();
    int solve();
};

template <typename T>
cg<T>::cg(collection<T> *coll, T *bvec, T *xvec){
  this->coll = coll;
  bs = new blas<T>(this->coll);

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
  isInner = this->coll->isInner;

  if(isVP && this->coll->isInner ){
    maxloop = this->coll->innerMaxLoop;
    eps = this->coll->innerEps;
  }else{
    maxloop = this->coll->outerMaxLoop;
    eps = this->coll->outerEps;
  }

  for(long int i=0; i<N; i++){
    rvec[i] = 0.0;
    pvec[i] = 0.0;
    mv[i] = 0.0;
    xvec[i] = 0.0;
  }
  
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

template <typename T>
cg<T>::~cg(){
  delete this->bs;
  delete[] rvec;
  delete[] pvec;
  delete[] mv;
  delete[] x_0;
  f_his.close();
  f_x.close();
}

template <typename T>
int cg<T>::solve(){
  //x_0 = x
  bs->Vec_copy(xvec, x_0, N);

  //b 2norm
  bnorm = bs->norm_2(bvec, N);

  //mv = Ax
  if(isCUDA){

  }else{
    bs->MtxVec_mult(xvec, mv, N);
  }

  //r = b - Ax
  bs->Vec_sub(bvec, mv, rvec, N);


  //p = r
  bs->Vec_copy(rvec, pvec, N);

  //r dot
  if(isCUDA){

  }else{
    rr = bs->dot(rvec, rvec, N);
  }

  for(loop=1; loop<=maxloop; loop++){
    rnorm = bs->norm_2(rvec, N);
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
      bs->MtxVec_mult(pvec, mv, N);
    }

    //alpha = (r,r) / (p,ap)
    if(isCUDA){
    }else{
      dot = bs->dot(rvec, mv, N);
    }
    alpha = rr / dot;

    //x = alpha * pvec + x
    bs->Scalar_axy(alpha, pvec, xvec, xvec, N);

    //r = -alpha * AP(mv) + r
    bs->Scalar_axy(-alpha, mv, rvec, rvec, N);

    //rr2 dot
    if(isCUDA){

    }else{
      rr2 = bs->dot(rvec, rvec, N);
    }

    beta = rr2/rr;

    rr = rr2;

    //p = beta * p + r
    bs->Scalar_axy(beta, pvec, rvec, pvec, N);
  }

  if(!isInner){
    test_error = bs->Check_error(xvec, x_0, N);
    std::cout << "|b-ax|2/|b|2 = " << std::fixed << std::setprecision(1) << test_error << std::endl;
    std::cout << "loop = " << loop-1 << std::endl;

    for(long int i=0; i<N; i++){
      f_x << i << " " << std::scientific << std::setprecision(12) << std::uppercase << xvec[i] << std::endl;
    }
  }else{

  }

  return exit_flag;
}
#endif //CG_HPP_INCLUDED__


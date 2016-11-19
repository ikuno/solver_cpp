#ifndef CR_HPP_INCLUDED__
#define CR_HPP_INCLUDED__

#include <iomanip>
#include <fstream>
#include "solver_collection.hpp"
#include "blas.hpp"

template <typename T>
class cr {
  private:
    collection<T> *coll;
    blas<T> *bs;
    
    long int loop;
    T *xvec, *bvec;
    T *rvec, *pvec, *qvec, *svec, *x_0, dot, error;
    T alpha, beta, bnorm, rnorm;
    T rs, rs2;

    int maxloop;
    double eps;
    bool isVP, isVerbose, isCUDA, isInner;

    int exit_flag;
    T test_error;

    int N;

    std::ofstream f_his;
    std::ofstream f_x;

  public:
    cr(collection<T> *coll, T *bvec, T *xvec);
    ~cr();
    int solve();
};

template <typename T>
cr<T>::cr(collection<T> *coll, T *bvec, T *xvec){
  this->coll = coll;
  bs = new blas<T>(this->coll);

  N = this->coll->N;
  rvec = new T [N];
  pvec = new T [N];
  qvec = new T [N];
  svec = new T [N];
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

  std::memset(rvec, 0, sizeof(T)*N);
  std::memset(pvec, 0, sizeof(T)*N);
  std::memset(qvec, 0, sizeof(T)*N);
  std::memset(svec, 0, sizeof(T)*N);
  std::memset(xvec, 0, sizeof(T)*N);
  
  f_his.open("./output/CR_his.txt");
  if(!f_his.is_open()){
    std::cerr << "File open error" << std::endl;
    exit(-1);
  }

  f_x.open("./output/CR_xvec.txt");
  if(!f_x.is_open()){
    std::cerr << "File open error" << std::endl;
    exit(-1);
  }

}

template <typename T>
cr<T>::~cr(){
  delete this->bs;
  delete[] rvec;
  delete[] pvec;
  delete[] qvec;
  delete[] svec;
  delete[] x_0;
  f_his.close();
  f_x.close();
}

template <typename T>
int cr<T>::solve(){
  //x_0 = x
  bs->Vec_copy(xvec, x_0, N);

  //b 2norm
  bnorm = bs->norm_2(bvec, N);

  //qvec = Ax
  if(isCUDA){

  }else{
    bs->MtxVec_mult(xvec, qvec, N);
  }

  //r = b - Ax(qvec)
  bs->Vec_sub(bvec, qvec, rvec, N);


  //p = r
  bs->Vec_copy(rvec, pvec, N);

  //qvec = Ap
  if(isCUDA){

  }else{
    bs->MtxVec_mult(pvec, qvec, N);
  }

  //s = q
  bs->Vec_copy(qvec, svec, N);

  //(r, s)
  if(isCUDA){

  }else{
    rs = bs->dot(rvec, svec, N);
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

    //alpha = (r,s) / (q,q)
    if(isCUDA){
    }else{
      dot = bs->dot(qvec, qvec, N);
    }
    alpha = rs / dot;

    //x = alpha * pvec + x
    bs->Scalar_axy(alpha, pvec, xvec, xvec, N);

    //r = -alpha * qvec + r
    bs->Scalar_axy(-alpha, qvec, rvec, rvec, N);

    //s=Ar
    if(isCUDA){

    }else{
      bs->MtxVec_mult(rvec, svec, N);
    }

    //r2=(r, s)
    if(isCUDA){

    }else{
      rs2 = bs->dot(rvec, svec, N);
    }

    //beta=(r_new, s_new)/(r, s)
    beta = rs2/rs;

    rs = rs2;

    //p = beta * p + r
    bs->Scalar_axy(beta, pvec, rvec, pvec, N);

    //q = beta * q + s
    bs->Scalar_axy(beta, qvec, svec, qvec, N);
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
#endif //CR_HPP_INCLUDED__


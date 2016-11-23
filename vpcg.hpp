#ifndef VPCG_HPP_INCLUDED__
#define VPCG_HPP_INCLUDED__

#include <iomanip>
#include <fstream>
#include "solver_collection.hpp"
#include "blas.hpp"
#include "innerMethods.hpp"

template <typename T>
class vpcg {
  private:
    collection<T> *coll;
    blas<T> *bs;
    innerMethods<T> *in;

    long int loop;
    T *xvec, *bvec;
    T *rvec, *pvec, *mv, *x_0, dot, error;
    T *zvec;
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
    vpcg(collection<T> *coll, T *bvec, T *xvec, bool inner);
    ~vpcg();
    int solve();
};

template <typename T>
vpcg<T>::vpcg(collection<T> *coll, T *bvec, T *xvec, bool inner){
  this->coll = coll;
  bs = new blas<T>(this->coll);
  in = new innerMethods<T>(this->coll);

  N = this->coll->N;
  rvec = new T [N];
  pvec = new T [N];
  zvec = new T [N];
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
  std::memset(zvec, 0, sizeof(T)*N);

  f_his.open("./output/VPCG_his.txt");
  if(!f_his.is_open()){
    std::cerr << "File open error" << std::endl;
    exit(-1);
  }

  f_x.open("./output/VPCG_xvec.txt");
  if(!f_x.is_open()){
    std::cerr << "File open error" << std::endl;
    exit(-1);
  }

}

template <typename T>
vpcg<T>::~vpcg(){
  delete this->bs;
  delete this->in;
  delete[] rvec;
  delete[] pvec;
  delete[] zvec;
  delete[] mv;
  delete[] x_0;
  f_his.close();
  f_x.close();
}

template <typename T>
int vpcg<T>::solve(){

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

  // inner->innerSelect(this->coll->innerSolver, rvec, zvec);
  in->innerSelect(this->coll, this->coll->innerSolver, rvec, zvec);

  //p = z
  bs->Vec_copy(zvec, pvec);

  //r,z dot
  if(isCUDA){

  }else{
    rr = bs->dot(rvec, zvec);
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

    //alpha = (r,r) / (p,ap)
    if(isCUDA){
    }else{
      dot = bs->dot(pvec, mv);
    }
    alpha = rr / dot;

    //x = alpha * pvec + x
    bs->Scalar_axy(alpha, pvec, xvec, xvec);

    //r = -alpha * AP(mv) + r
    bs->Scalar_axy(-alpha, mv, rvec, rvec);

    std::memset(zvec, 0, sizeof(T)*N);

    // inner->innerSelect(this->coll->innerSolver, rvec, zvec);
    in->innerSelect(this->coll, this->coll->innerSolver, rvec, zvec);

    //z, r  dot
    if(isCUDA){

    }else{
      rr2 = bs->dot(rvec, zvec);
    }

    beta = rr2/rr;

    rr = rr2;

    //p = beta * p + z
    bs->Scalar_axy(beta, pvec, zvec, pvec);

  }

  if(!isInner){
    test_error = bs->Check_error(xvec, x_0);
    std::cout << "|b-ax|2/|b|2 = " << std::fixed << std::setprecision(1) << test_error << std::endl;
    std::cout << "loop = " << loop << std::endl;

    for(long int i=0; i<N; i++){
      f_x << i << " " << std::scientific << std::setprecision(12) << std::uppercase << xvec[i] << std::endl;
    }
  }else{

  }

  return exit_flag;
}
#endif //VPCG_HPP_INCLUDED__


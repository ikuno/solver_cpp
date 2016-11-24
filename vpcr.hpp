#ifndef VPCR_HPP_INCLUDED__
#define VPCR_HPP_INCLUDED__

#include <iomanip>
#include <fstream>
#include "solver_collection.hpp"
#include "blas.hpp"
#include "innerMethods.hpp"
#include "times.hpp"

template <typename T>
class vpcr {
  private:
    collection<T> *coll;
    blas<T> *bs;
    innerMethods<T> *in;
    times time;
    
    long int loop;
    T *xvec, *bvec;
    T *rvec, *pvec, *zvec, *Av, *Ap, *x_0, error;
    T alpha, beta, bnorm, rnorm;
    T zaz, zaz2;
    T tmp;

    int maxloop;
    double eps;
    bool isVP, isVerbose, isCUDA, isInner;

    int exit_flag;
    T test_error;

    int N;

    std::ofstream f_his;
    std::ofstream f_x;

  public:
    vpcr(collection<T> *coll, T *bvec, T *xvec, bool inner);
    ~vpcr();
    int solve();
};

template <typename T>
vpcr<T>::vpcr(collection<T> *coll, T *bvec, T *xvec, bool inner){
  this->coll = coll;
  bs = new blas<T>(this->coll);
  in = new innerMethods<T>(this->coll);

  N = this->coll->N;
  rvec = new T [N];
  pvec = new T [N];
  zvec = new T [N];
  Av = new T [N];
  Ap = new T [N];
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
  std::memset(zvec, 0, sizeof(T)*N);
  std::memset(Av, 0, sizeof(T)*N);
  std::memset(Ap, 0, sizeof(T)*N);
  std::memset(xvec, 0, sizeof(T)*N);

  if(!isInner){
    f_his.open("./output/VPCR_his.txt");
    if(!f_his.is_open()){
      std::cerr << "File open error" << std::endl;
      exit(-1);
    }

    f_x.open("./output/VPCR_xvec.txt");
    if(!f_x.is_open()){
      std::cerr << "File open error" << std::endl;
      exit(-1);
    }
  }

}

template <typename T>
vpcr<T>::~vpcr(){
  delete this->bs;
  delete this->in;
  delete[] rvec;
  delete[] pvec;
  delete[] zvec;
  delete[] Av;
  delete[] Ap;
  delete[] x_0;
  f_his.close();
  f_x.close();
}

template <typename T>
int vpcr<T>::solve(){

  time.start();

  //x_0 = x
  bs->Vec_copy(xvec, x_0);

  //b 2norm
  bnorm = bs->norm_2(bvec);

  //Ax
  if(isCUDA){

  }else{
    bs->MtxVec_mult(xvec, Av);
  }

  //r = b - Ax
  bs->Vec_sub(bvec, Av, rvec);

  //Az=r
  in->innerSelect(this->coll, this->coll->innerSolver, rvec, zvec);

  //p = z
  bs->Vec_copy(zvec, pvec);

  //Az(Av)
  if(isCUDA){

  }else{
    bs->MtxVec_mult(zvec, Av);
  }

  //Ap = Az(Av)
  bs->Vec_copy(Av, Ap);

  //(z, Az(Av))
  if(isCUDA){

  }else{
    zaz = bs->dot(zvec, Av);
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

    //alpha = (z,Az) / (Ap,Ap)
    if(isCUDA){
    }else{
      tmp = bs->dot(Ap, Ap);
    }
    alpha = zaz / tmp;

    //x = alpha * pvec + x
    bs->Scalar_axy(alpha, pvec, xvec, xvec);

    //r = -alpha * Ap + r
    bs->Scalar_axy(-alpha, Ap, rvec, rvec);

    std::memset(zvec, 0, sizeof(T)*N);

    //Az = r
    in->innerSelect(this->coll, this->coll->innerSolver, rvec, zvec);

    //Az
    if(isCUDA){

    }else{
      bs->MtxVec_mult(zvec, Av);
    }

    //(z, Az)
    if(isCUDA){

    }else{
      zaz2 = bs->dot(zvec, Av);
    }

    beta = zaz2/zaz;

    zaz = zaz2;

    //p = beta * p + r
    bs->Scalar_axy(beta, pvec, zvec, pvec);

    //Ap = beta * Ap + Az
    bs->Scalar_axy(beta, Ap, Av, Ap);
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
  }

  return exit_flag;
}
#endif //VPCR_HPP_INCLUDED__


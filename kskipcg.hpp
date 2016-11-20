#include <iomanip>
#include <fstream>
#include "solver_collection.hpp"
#include "blas.hpp"

template <typename T>
class kskipcg {
  private:
    collection<T> *coll;
    blas<T> *bs;
    
    long int nloop, iloop, jloop;
    T *xvec, *bvec;
    T *rvec, *pvec, *Av, *x_0, error;
    T *delta, *eta, *zeta;
    T **Ap, **Ar;
    T alpha, beta, gamma, bnorm, rnorm;

    int maxloop;
    double eps;
    bool isVP, isVerbose, isCUDA, isInner;
    int kskip;
    int fix;

    int exit_flag;
    T test_error;

    int N;

    std::ofstream f_his;
    std::ofstream f_x;

  public:
    kskipcg(collection<T> *coll, T *bvec, T *xvec);
    ~kskipcg();
    int solve();
};

template <typename T>
kskipcg<T>::kskipcg(collection<T> *coll, T *bvec, T *xvec){
  this->coll = coll;
  bs = new blas<T>(this->coll);

  isVP = this->coll->isVP;
  isVerbose = this->coll->isVerbose;
  isCUDA = this->coll->isCUDA;
  isInner = this->coll->isInner;

  if(isVP && this->coll->isInner ){
    maxloop = this->coll->innerMaxLoop;
    eps = this->coll->innerEps;
    kskip = this->coll->innerKskip;
    fix = this->coll->innerFix;
  }else{
    maxloop = this->coll->outerMaxLoop;
    eps = this->coll->outerEps;
    kskip = this->coll->outerKskip;
    fix = this->coll->outerFix;
  }

  N = this->coll->N;
  rvec = new T [N];
  pvec = new T [N];
  Av = new T [N];
  x_0 = new T [N];

  delta = new T [2*kskip];
  eta = new T [2*kskip+1];
  zeta = new T [2*kskip+2];

  Ar = new T* [(2*kskip+1)];
  Ap = new T* [(2*kskip+2)];
  for(int i=0; i<2*kskip+1; i++){
    Ar[i] = new T[N];
  }
  for(int i=0; i<2*kskip+2; i++){
    Ap[i] = new T[N];
  }
  this->xvec = xvec;
  this->bvec = bvec;

  exit_flag = 2;

  std::memset(rvec, 0, sizeof(T)*N);
  std::memset(pvec, 0, sizeof(T)*N);
  std::memset(Av, 0, sizeof(T)*N);
  std::memset(xvec, 0, sizeof(T)*N);
  std::memset(delta, 0, sizeof(T)*(2*kskip));
  std::memset(eta, 0, sizeof(T)*(2*kskip+1));
  std::memset(zeta, 0, sizeof(T)*(2*kskip+2));

  for(int i=0; i<2*kskip+1; i++){
    std::memset(Ar[i], 0, sizeof(T)*N);
  }
  for(int i=0; i<2*kskip+2; i++){
    std::memset(Ap[i], 0, sizeof(T)*N);
  }
  // std::memset(Ar, 0, sizeof(T)*(N*2*kskip+1));
  // std::memset(Ap, 0, sizeof(T)*(N*2*kskip+2));

  // for(int i=0; i<2*kskip+1; i++){
  //   for(long int j=0; j<N; j++){
  //     Ar[i][j] = 0.0;
  //   }
  // }
  //
  // for(int i=0; i<2*kskip+2; i++){
  //   for(long int j=0; j<N; j++){
  //     Ap[i][j] = 0.0;
  //   }
  // }

  f_his.open("./output/KSKIPCG_his.txt");
  if(!f_his.is_open()){
    std::cerr << "File open error" << std::endl;
    exit(-1);
  }

  f_x.open("./output/KSKIPCG_xvec.txt");
  if(!f_x.is_open()){
    std::cerr << "File open error" << std::endl;
    exit(-1);
  }

}

template <typename T>
kskipcg<T>::~kskipcg(){
  delete this->bs;
  delete[] rvec;
  delete[] pvec;
  delete[] Av;
  delete[] x_0;
  delete[] delta;
  delete[] eta;
  delete[] zeta;
  for(int i=0; i<2*kskip+1; i++){
    delete[] Ar[i];
  }
  for(int i=0; i<2*kskip+2; i++){
    delete[] Ap[i];
  }
  delete[] Ar;
  delete[] Ap;
  f_his.close();
  f_x.close();
}

template <typename T>
int kskipcg<T>::solve(){
  //x_0 = x
  bs->Vec_copy(xvec, x_0);

  //Ax
  if(isCUDA){

  }else{
    bs->MtxVec_mult(xvec, Av);
  }

  //r = b - Ax
  bs->Vec_sub(bvec, Av, rvec);

  //p = r
  bs->Vec_copy(rvec, pvec);

  //b 2norm
  bnorm = bs->norm_2(bvec);

  for(nloop=0; nloop<maxloop; nloop+=(kskip+1)){
    rnorm = bs->norm_2(rvec);
    error = rnorm/bnorm;
    if(!isInner){
      if(isVerbose){
        std::cout << nloop+1 << " " << std::scientific << std::setprecision(12) << std::uppercase << error << std::endl;
      }
      f_his << nloop+1 << " " << std::scientific << std::setprecision(12) << std::uppercase << error << std::endl;
    }

    if(error <= eps){
      exit_flag = 0;
      break;
    }

    //Ar-> Ar^2k
    //Ap-> Ap^2k+2
    if(isCUDA){

    }else{
      bs->Kskip_base(Ar, Ap, rvec, pvec, kskip);
    }

    //gamma=(r, r)
    if(isCUDA){

    }else{
      gamma = bs->dot(rvec, rvec);
    }

    //delta=(r,Ar)
    //eta=(r,Ap)
    //zeta=(p,Ap)
    if(isCUDA){

    }else{
      bs->Kskip_innerProduce(delta, eta, zeta, Ar, Ap, rvec, pvec, kskip);
    }

    for(iloop=nloop; iloop<=nloop+kskip; iloop++){
      //alpha = gamma/zeta_1
      alpha = gamma / zeta[0];

      //beta = (alpha * zeta_2 / zeta_1) - 1
      beta = alpha * zeta[1] / zeta[0] - 1.0;

      //fix
      if(fix == 1){
        gamma = beta * gamma;
      }else if(fix == 2){
        T tmp0 = gamma - alpha * eta[0];
        T tmp1 = eta[0] - alpha * zeta[1];
        gamma = tmp0 - alpha * tmp1;
      }

      //update delta eta zeta
      for(jloop=0; jloop<2*kskip-2*(iloop-nloop); jloop++){
        delta[jloop] = delta[jloop] - 2*alpha*eta[jloop+1] + alpha*alpha*eta[jloop+2];
        T eta_old = eta[jloop];
        eta[jloop] = delta[jloop] + beta*zeta[jloop+1] - alpha*beta*zeta[jloop+1];
        zeta[jloop] = eta[jloop+1] + beta*eta_old + beta*beta*zeta[jloop] - alpha*beta*zeta[jloop+1];
      }

      //Ap
      if(isCUDA){

      }else{
        bs->MtxVec_mult(pvec, Av);
      }

      //x=x+alpha*p
      bs->Scalar_axy(alpha, pvec, xvec, xvec);

      //r=r-alpha*Ap
      bs->Scalar_axy(-alpha, Av, rvec, rvec);

      //p=r+beta*p
      bs->Scalar_axy(beta, pvec, rvec, pvec);
    }
  }
  if(!isInner){
    test_error = bs->Check_error(xvec, x_0);
    std::cout << "|b-ax|2/|b|2 = " << std::fixed << std::setprecision(1) << test_error << std::endl;
    std::cout << "loop = " << nloop-kskip+1 << std::endl;

    for(long int i=0; i<N; i++){
      f_x << i << " " << std::scientific << std::setprecision(12) << std::uppercase << xvec[i] << std::endl;
    }
  }else{

  }

  return exit_flag;
}

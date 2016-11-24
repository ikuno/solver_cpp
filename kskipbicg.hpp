#ifndef KSKIPBICG_HPP_INCLUDED__
#define KSKIPBICG_HPP_INCLUDED__

#include <iomanip>
#include <fstream>
#include "solver_collection.hpp"
#include "blas.hpp"
#include "times.hpp"

template <typename T>
class kskipBicg {
  private:
    collection<T> *coll;
    blas<T> *bs;
    times time;
    
    long int nloop, iloop, jloop;
    T *xvec, *bvec;
    T *rvec, *r_vec, *pvec, *p_vec, *Av, *x_0, error;
    T *theta, *eta, *rho, *phi, bnorm, rnorm, alpha, beta, gamma;
    T **Ap, **Ar;

    int maxloop;
    double eps;
    bool isVP, isVerbose, isCUDA, isInner;
    int kskip;

    int exit_flag;
    T test_error;

    int N;

    std::ofstream f_his;
    std::ofstream f_x;

  public:
    kskipBicg(collection<T> *coll, T *bvec, T *xvec, bool inner);
    ~kskipBicg();
    int solve();
};

template <typename T>
kskipBicg<T>::kskipBicg(collection<T> *coll, T *bvec, T *xvec, bool inner){
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
    kskip = this->coll->innerKskip;
  }else{
    maxloop = this->coll->outerMaxLoop;
    eps = this->coll->outerEps;
    kskip = this->coll->outerKskip;
  }

  N = this->coll->N;
  rvec = new T [N];
  pvec = new T [N];
  r_vec = new T [N];
  p_vec = new T [N];
  Av = new T [N];
  x_0 = new T [N];
  theta = new T [2*kskip];
  eta = new T [2*kskip+1];
  rho = new T [2*kskip+1];
  phi = new T [2*kskip+2];

  Ar = new T* [2*kskip+1];
  Ap = new T* [2*kskip+2];
  for(int i=0; i<2*kskip+1; i++){
    Ar[i] = new T [N];
  }
  for(int i=0; i<2*kskip+2; i++){
    Ap[i] = new T [N];
  }

  this->xvec = xvec;
  this->bvec = bvec;

  std::memset(rvec, 0, sizeof(T)*N);
  std::memset(pvec, 0, sizeof(T)*N);
  std::memset(r_vec, 0, sizeof(T)*N);
  std::memset(p_vec, 0, sizeof(T)*N);
  std::memset(Av, 0, sizeof(T)*N);
  std::memset(xvec, 0, sizeof(T)*N);

  std::memset(theta, 0, sizeof(T)*(2*kskip));
  std::memset(eta, 0, sizeof(T)*(2*kskip+1));
  std::memset(rho, 0, sizeof(T)*(2*kskip+1));
  std::memset(phi, 0, sizeof(T)*(2*kskip+2));

  for(int i=0; i<2*kskip+1; i++){
    std::memset(Ar[i], 0, sizeof(T)*N);
  }
  for(int i=0; i<2*kskip+2; i++){
    std::memset(Ap[i], 0, sizeof(T)*N);
  }

  if(!isInner){
    f_his.open("./output/KSKIPBICG_his.txt");
    if(!f_his.is_open()){
      std::cerr << "File open error" << std::endl;
      exit(-1);
    }

    f_x.open("./output/KSKIPBICG_xvec.txt");
    if(!f_x.is_open()){
      std::cerr << "File open error" << std::endl;
      exit(-1);
    }
  }

}

template <typename T>
kskipBicg<T>::~kskipBicg(){
  delete this->bs;
  delete[] rvec;
  delete[] pvec;
  delete[] r_vec;
  delete[] p_vec;
  delete[] Av;
  delete[] x_0;
  delete[] theta;
  delete[] eta;
  delete[] rho;
  delete[] phi;
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
int kskipBicg<T>::solve(){

  time.start();

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

  //r* = r
  bs->Vec_copy(rvec, r_vec);

  //p* = *r
  bs->Vec_copy(r_vec, p_vec);

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

    //Ar-> Ar^2k+1
    //Ap-> Ap^2k+2
    if(isCUDA){

    }else{
      bs->Kskip_kskipBicg_base(Ar, Ap, rvec, pvec, kskip);
    }

    //gamma=(r*,r)
    if(isCUDA){

    }else{
      gamma = bs->dot(r_vec, rvec);
    }

    /* theta (2*i_kskip); */
    /* eta = (2*i_kskip+1); */
    /* rho = (2*i_kskip+1); */
    /* phi = (2*i_kskip+2); */
    //theta = (r*, Ar)
    //eta = (r*, Ap)
    //rho = (p*, Ar)
    //phi = (p*, Ap)
    if(isCUDA){

    }else{
      bs->Kskip_kskipBicg_innerProduce(theta, eta, rho, phi, Ar, Ap, rvec, pvec, r_vec, p_vec, kskip);
    }

    for(iloop=nloop; iloop<=nloop+kskip; iloop++){
      //alpha = gamma/phi_1
      alpha=gamma/phi[0];
      //beta = 1 - (eta_1 + rho_1 - alpha*phi_2)/phi_1
      beta=1.0 - (eta[0] + rho[0] - alpha*phi[1])/phi[0];
      //gamma = gamma - alpha*eta_1 - alpha*rho_1 + alpha^2*phi_2
      /* gamma = gamma - alpha*eta[0] - alpha*rho[0] + alpha*alpha*phi[1]; */
      gamma = gamma*beta;

      //update theta eta rho phi
      for(jloop=0; jloop<2*kskip-2*(iloop-nloop); jloop++){
        theta[jloop] = theta[jloop] - alpha * eta[jloop+1] - alpha * rho[jloop+1] + alpha * alpha * phi[jloop+2];
        T tmp = theta[jloop] - alpha * beta * phi[jloop+1];
        eta[jloop] = tmp + beta * eta[jloop];
        rho[jloop] = tmp + beta * rho[jloop];
        phi[jloop] = eta[jloop] + rho[jloop] - theta[jloop] + beta * beta * phi[jloop];
      }

      //Ap
      if(isCUDA){

      }else{
        bs->MtxVec_mult(pvec, Av);
      }

      //x=alpha*p+x
      bs->Scalar_axy(alpha, pvec, xvec, xvec);

      //r=-alpha*Ap+r
      bs->Scalar_axy(-alpha, Av, rvec, rvec);

      //A^Tp*
      if(isCUDA){

      }else{
        bs->MtxVec_mult(this->coll->Tval, this->coll->Tcol, this->coll->Tptr, p_vec, Av);
      }

      //r*=r*-alpha*A^Tp*
      bs->Scalar_axy(-alpha, Av, r_vec, r_vec);
      //p=r+beta*p
      bs->Scalar_axy(beta, pvec, rvec, pvec);
      //p*=r*+beta*p*
      bs->Scalar_axy(beta, p_vec, r_vec, p_vec);
    }
  }

  time.end();

  if(!isInner){
    test_error = bs->Check_error(xvec, x_0);
    std::cout << "|b-ax|2/|b|2 = " << std::fixed << std::setprecision(1) << test_error << std::endl;
    std::cout << "loop = " << nloop+1 << std::endl;
    std::cout << "time = " << std::setprecision(6) << time.getTime() << std::endl;

    for(long int i=0; i<N; i++){
      f_x << i << " " << std::scientific << std::setprecision(12) << std::uppercase << xvec[i] << std::endl;
    }
  }else{
    if(isVerbose){
      if(exit_flag==0){
        std::cout << GREEN << "\t" <<  nloop+1 << " = " << std::scientific << std::setprecision(12) << std::uppercase << error << RESET << std::endl;
      }else if(exit_flag==2){
        std::cout << RED << "\t" << nloop+1 << " = " << std::scientific << std::setprecision(12) << std::uppercase << error << RESET << std::endl;
      }else{
        std::cout << RED << " ERROR " << nloop+1 << RESET << std::endl;
      }
    }
  }

  return exit_flag;
}
#endif //KSKIPBICG_HPP_INCLUDED__


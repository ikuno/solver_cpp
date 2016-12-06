#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include "color.hpp"
#include "kskipcg.hpp"

kskipcg::kskipcg(collection *coll, double *bvec, double *xvec, bool inner, cuda *a_cu, blas *a_bs){
  this->coll = coll;
  isInner = inner;

  isVP = this->coll->isVP;
  isVerbose = this->coll->isVerbose;
  isCUDA = this->coll->isCUDA;

  if(isVP && isInner ){
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

  if(isInner){
    this->bs = a_bs;
    this->cu = a_cu;
  }else{
    bs = new blas(this->coll, this->coll->time);
    cu = new cuda(this->coll->time, this->coll->N, this->kskip);
  }

  N = this->coll->N;

  if(isCUDA){
    rvec = cu->d_MallocHost(N);
    pvec = cu->d_MallocHost(N);
    Av = cu->d_MallocHost(N);
    x_0 = cu->d_MallocHost(N);
    delta = cu->d_MallocHost(2*kskip);
    eta = cu->d_MallocHost(2*kskip+1);
    zeta = cu->d_MallocHost(2*kskip+2);
    Ar = cu->d_MallocHost((2*kskip+1)*N);
    Ap = cu->d_MallocHost((2*kskip+2)*N);
  }else{
    rvec = new double [N];
    pvec = new double [N];
    Av = new double [N];
    x_0 = new double [N];
    delta = new double [2*kskip];
    eta = new double [2*kskip+1];
    zeta = new double [2*kskip+2];
    Ar = new double [(2*kskip+1)*N];
    Ap = new double [(2*kskip+2)*N];
  }

  this->xvec = xvec;
  this->bvec = bvec;

  exit_flag = 2;

  std::memset(rvec, 0, sizeof(double)*N);
  std::memset(pvec, 0, sizeof(double)*N);
  std::memset(Av, 0, sizeof(double)*N);
  std::memset(xvec, 0, sizeof(double)*N);
  std::memset(delta, 0, sizeof(double)*(2*kskip));
  std::memset(eta, 0, sizeof(double)*(2*kskip+1));
  std::memset(zeta, 0, sizeof(double)*(2*kskip+2));
  std::memset(Ar, 0, sizeof(double)*(2*kskip+1)*N);
  std::memset(Ap, 0, sizeof(double)*(2*kskip+2)*N);

  if(!isInner){
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

}

kskipcg::~kskipcg(){
  if(isCUDA){
    cu->FreeHost(rvec);
    cu->FreeHost(pvec);
    cu->FreeHost(Av);
    cu->FreeHost(x_0);
    cu->FreeHost(delta);
    cu->FreeHost(eta);
    cu->FreeHost(zeta);
    cu->FreeHost(Ar);
    cu->FreeHost(Ap);
  }else{
    delete[] rvec;
    delete[] pvec;
    delete[] Av;
    delete[] x_0;
    delete[] delta;
    delete[] eta;
    delete[] zeta;
    delete[] Ar;
    delete[] Ap;
  }

  if(!isInner){
    delete this->bs;
    delete cu;
  }
  f_his.close();
  f_x.close();
}

int kskipcg::solve(){

  time.start();
  //x_0 = x
  bs->Vec_copy(xvec, x_0);

  //Ax
  if(isCUDA){
    cu->MtxVec_mult(xvec, Av, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
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
      // bs->Kskip_cg_base(Ar, Ap, rvec, pvec, kskip);
      // cu->Kskip_cg_bicg_base(Ar, Ap, rvec, pvec, kskip, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
      cu->Kskip_cg_bicg_base2(Ar, Ap, rvec, pvec, kskip, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
    }else{
      bs->Kskip_cg_bicg_base(Ar, Ap, rvec, pvec, kskip);
    }

    //gamma=(r, r)
    if(isCUDA){
      // gamma = cu->dot(rvec, rvec);
      gamma = bs->dot(rvec, rvec);
    }else{
      gamma = bs->dot(rvec, rvec);
    }

    //delta=(r,Ar)
    //eta=(r,Ap)
    //zeta=(p,Ap)
    if(isCUDA){
      // bs->Kskip_cg_innerProduce(delta, eta, zeta, Ar, Ap, rvec, pvec, kskip);
      // cu->Kskip_cg_innerProduce(delta, eta, zeta, Ar, Ap, rvec, pvec, kskip, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
      cu->Kskip_cg_innerProduce2(delta, eta, zeta, Ar, Ap, rvec, pvec, kskip, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
    }else{
      bs->Kskip_cg_innerProduce(delta, eta, zeta, Ar, Ap, rvec, pvec, kskip);
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
        double tmp0 = gamma - alpha * eta[0];
        double tmp1 = eta[0] - alpha * zeta[1];
        gamma = tmp0 - alpha * tmp1;
      }

      //update delta eta zeta
      for(jloop=0; jloop<2*kskip-2*(iloop-nloop); jloop++){
        delta[jloop] = delta[jloop] - 2*alpha*eta[jloop+1] + alpha*alpha*eta[jloop+2];
        double eta_old = eta[jloop];
        eta[jloop] = delta[jloop] + beta*zeta[jloop+1] - alpha*beta*zeta[jloop+1];
        zeta[jloop] = eta[jloop+1] + beta*eta_old + beta*beta*zeta[jloop] - alpha*beta*zeta[jloop+1];
      }

      //Ap
      if(isCUDA){
        cu->MtxVec_mult(pvec, Av, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
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
  time.end();

  if(!isInner){
    test_error = bs->Check_error(xvec, x_0);
    std::cout << "|b-ax|2/|b|2 = " << std::fixed << std::setprecision(1) << test_error << std::endl;
    std::cout << "loop = " << nloop-kskip+1 << std::endl;
    std::cout << "time = " << std::setprecision(6) << time.getTime() << std::endl;

    if(this->coll->isCUDA){
      this->coll->time->showTimeOnGPU(time.getTime(), this->coll->time->showTimeOnCPU(time.getTime(), true));
    }else{
      this->coll->time->showTimeOnCPU(time.getTime());
    }

    for(long int i=0; i<N; i++){
      f_x << i << " " << std::scientific << std::setprecision(12) << std::uppercase << xvec[i] << std::endl;
    }
  }else{
    if(isVerbose){
      if(exit_flag==0){
        std::cout << GREEN << "\t" <<  nloop-kskip+1 << " = " << std::scientific << std::setprecision(12) << std::uppercase << error << RESET << std::endl;
      }else if(exit_flag==2){
        std::cout << RED << "\t" << nloop-kskip+1 << " = " << std::scientific << std::setprecision(12) << std::uppercase << error << RESET << std::endl;
      }else{
        std::cout << RED << " ERROR " << nloop-kskip+1 << RESET << std::endl;
      }
    }
  }

  return exit_flag;
}

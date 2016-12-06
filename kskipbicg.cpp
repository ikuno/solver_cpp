#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include "color.hpp"
#include "kskipbicg.hpp"

kskipBicg::kskipBicg(collection *coll, double *bvec, double *xvec, bool inner, cuda *a_cu, blas *a_bs){
  this->coll = coll;
  isInner = inner;

  exit_flag = 2;
  isVP = this->coll->isVP;
  isVerbose = this->coll->isVerbose;
  isCUDA = this->coll->isCUDA;

  if(isVP && isInner ){
    maxloop = this->coll->innerMaxLoop;
    eps = this->coll->innerEps;
    kskip = this->coll->innerKskip;
  }else{
    maxloop = this->coll->outerMaxLoop;
    eps = this->coll->outerEps;
    kskip = this->coll->outerKskip;
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
    r_vec = cu->d_MallocHost(N);
    p_vec = cu->d_MallocHost(N);
    Av = cu->d_MallocHost(N);
    x_0 = cu->d_MallocHost(N);
    theta = cu->d_MallocHost(2*kskip);
    eta = cu->d_MallocHost(2*kskip+1);
    rho = cu->d_MallocHost(2*kskip+1);
    phi = cu->d_MallocHost(2*kskip+2);
    Ar = cu->d_MallocHost((2*kskip+1)*N);
    Ap = cu->d_MallocHost((2*kskip+2)*N);
  }else{
    rvec = new double [N];
    pvec = new double [N];
    r_vec = new double [N];
    p_vec = new double [N];
    Av = new double [N];
    x_0 = new double [N];
    theta = new double [2*kskip];
    eta = new double [2*kskip+1];
    rho = new double [2*kskip+1];
    phi = new double [2*kskip+2];

    Ar = new double [(2*kskip+1)*N];
    Ap = new double [(2*kskip+2)*N];
  }

  this->xvec = xvec;
  this->bvec = bvec;

  std::memset(rvec, 0, sizeof(double)*N);
  std::memset(pvec, 0, sizeof(double)*N);
  std::memset(r_vec, 0, sizeof(double)*N);
  std::memset(p_vec, 0, sizeof(double)*N);
  std::memset(Av, 0, sizeof(double)*N);
  std::memset(xvec, 0, sizeof(double)*N);

  std::memset(theta, 0, sizeof(double)*(2*kskip));
  std::memset(eta, 0, sizeof(double)*(2*kskip+1));
  std::memset(rho, 0, sizeof(double)*(2*kskip+1));
  std::memset(phi, 0, sizeof(double)*(2*kskip+2));

  std::memset(Ar, 0, sizeof(double)*(2*kskip+1)*N);
  std::memset(Ap, 0, sizeof(double)*(2*kskip+2)*N);

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

kskipBicg::~kskipBicg(){
  if(isCUDA){
    cu->FreeHost(rvec);
    cu->FreeHost(pvec);
    cu->FreeHost(r_vec);
    cu->FreeHost(p_vec);
    cu->FreeHost(Av);
    cu->FreeHost(x_0);
    cu->FreeHost(theta);
    cu->FreeHost(eta);
    cu->FreeHost(rho);
    cu->FreeHost(phi);
    cu->FreeHost(Ar);
    cu->FreeHost(Ap);
  }else{
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

int kskipBicg::solve(){

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
      // cu->Kskip_cg_bicg_base(Ar, Ap, rvec, pvec, kskip, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
      cu->Kskip_cg_bicg_base2(Ar, Ap, rvec, pvec, kskip, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
    }else{
      bs->Kskip_cg_bicg_base(Ar, Ap, rvec, pvec, kskip);
    }

    //gamma=(r*,r)
    if(isCUDA){
      // gamma = cu->dot(r_vec, rvec);
      gamma = bs->dot(r_vec, rvec);
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
      // bs->Kskip_bicg_innerProduce(theta, eta, rho, phi, Ar, Ap, rvec, pvec, r_vec, p_vec, kskip);
      // cu->Kskip_bicg_innerProduce(theta, eta, rho, phi, Ar, Ap, r_vec, p_vec, kskip, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
      cu->Kskip_bicg_innerProduce2(theta, eta, rho, phi, Ar, Ap, r_vec, p_vec, kskip, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
    }else{
      bs->Kskip_bicg_innerProduce(theta, eta, rho, phi, Ar, Ap, rvec, pvec, r_vec, p_vec, kskip);
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
        double tmp = theta[jloop] - alpha * beta * phi[jloop+1];
        eta[jloop] = tmp + beta * eta[jloop];
        rho[jloop] = tmp + beta * rho[jloop];
        phi[jloop] = eta[jloop] + rho[jloop] - theta[jloop] + beta * beta * phi[jloop];
      }

      //Ap
      if(isCUDA){
        cu->MtxVec_mult(pvec, Av, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
      }else{
        bs->MtxVec_mult(pvec, Av);
      }

      //x=alpha*p+x
      bs->Scalar_axy(alpha, pvec, xvec, xvec);

      //r=-alpha*Ap+r
      bs->Scalar_axy(-alpha, Av, rvec, rvec);

      //A^Tp*
      if(isCUDA){
        cu->MtxVec_mult(p_vec, Av, this->coll->CTval, this->coll->CTcol, this->coll->CTptr);
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

#include <iomanip>
#include <cstring>
#include <cstdlib>
#include "color.hpp"
#include "vpcr.hpp"

vpcr::vpcr(collection *coll, double *bvec, double *xvec, bool inner){
  this->coll = coll;
  bs = new blas(this->coll, this->coll->time);
  in = new innerMethods(this->coll, cu, bs);
  isMultiGPU = this->coll->isMultiGPU;

  if(this->coll->isInnerKskip){
    if(isMultiGPU){
      cu = new cuda(this->coll->time, this->coll->N, this->coll->innerKskip, this->coll->N1, this->coll->N2);
    }else{
      cu = new cuda(this->coll->time, this->coll->N, this->coll->innerKskip);
    }
  }else{
    if(isMultiGPU){
      cu = new cuda(this->coll->time, this->coll->N, this->coll->N1, this->coll->N2);
    }else{
      cu = new cuda(this->coll->time, this->coll->N);
    }
  }


  exit_flag = 2;
  isVP = this->coll->isVP;
  isVerbose = this->coll->isVerbose;
  isCUDA = this->coll->isCUDA;
  isInner = inner;
  isPinned = this->coll->isPinned;

  N = this->coll->N;
  if(isCUDA){
    if(isPinned){
      rvec = cu->d_MallocHost(N);
      pvec = new double [N];
      zvec = cu->d_MallocHost(N);
      Av = cu->d_MallocHost(N);
      Ap = cu->d_MallocHost(N);
      x_0 = new double [N];
    }else{
      rvec = new double [N];
      pvec = new double [N];
      zvec = new double [N];
      Av = new double [N];
      Ap = new double [N];
      x_0 = new double [N];
    }
  }else{
    rvec = new double [N];
    pvec = new double [N];
    zvec = new double [N];
    Av = new double [N];
    Ap = new double [N];
    x_0 = new double [N];
  }

  this->xvec = xvec;
  this->bvec = bvec;

  
  if(isVP && isInner ){
    maxloop = this->coll->innerMaxLoop;
    eps = this->coll->innerEps;
  }else{
    maxloop = this->coll->outerMaxLoop;
    eps = this->coll->outerEps;
  }

  std::memset(rvec, 0, sizeof(double)*N);
  std::memset(pvec, 0, sizeof(double)*N);
  std::memset(zvec, 0, sizeof(double)*N);
  std::memset(Av, 0, sizeof(double)*N);
  std::memset(Ap, 0, sizeof(double)*N);
  std::memset(xvec, 0, sizeof(double)*N);

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

  if(isVP){
    std::string name = this->coll->enum2string(this->coll->innerSolver);
    name = "./output/" + name + "_inner.txt";
    std::ifstream in(name.c_str());
    if(in.good()){
      std::cout << "Delete inner solver's loop file" << std::endl;
      std::remove(name.c_str());
    }else{
      std::cout << "Has no inner solver's loop file yet" << std::endl;
    }
  }


}

vpcr::~vpcr(){
  if(isCUDA){
    if(isPinned){
      cu->FreeHost(rvec);
      delete[] pvec;
      cu->FreeHost(zvec);
      cu->FreeHost(Av);
      cu->FreeHost(Ap);
      delete[] x_0;
    }else{
      delete[] rvec;
      delete[] pvec;
      delete[] zvec;
      delete[] Av;
      delete[] Ap;
      delete[] x_0;
    }
  }else{
    delete[] rvec;
    delete[] pvec;
    delete[] zvec;
    delete[] Av;
    delete[] Ap;
    delete[] x_0;
  }

  delete this->bs;
  delete this->in;
  delete this->cu;

  f_his.close();
  f_x.close();
}

int vpcr::solve(){

  time.start();

  //x_0 = x
  bs->Vec_copy(xvec, x_0);

  //b 2norm
  bnorm = bs->norm_2(bvec);

  //Ax
  if(isCUDA){
    cu->MtxVec_mult(xvec, Av, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
  }else{
    bs->MtxVec_mult(xvec, Av);
  }

  //r = b - Ax
  bs->Vec_sub(bvec, Av, rvec);

  //Az=r
  in->innerSelect(this->coll, this->coll->innerSolver, cu, bs, rvec, zvec);

  //p = z
  bs->Vec_copy(zvec, pvec);

  //Az(Av)
  if(isCUDA){
    cu->MtxVec_mult(zvec, Av, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
  }else{
    bs->MtxVec_mult(zvec, Av);
  }

  //Ap = Az(Av)
  bs->Vec_copy(Av, Ap);

  //(z, Az(Av))
  if(isCUDA){
    // zaz = cu->dot(zvec, Av);
    zaz = bs->dot(zvec, Av);
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
      // tmp = cu->dot(Ap, Ap);
      tmp = bs->dot(Ap, Ap);
    }else{
      tmp = bs->dot(Ap, Ap);
    }
    alpha = zaz / tmp;

    //x = alpha * pvec + x
    bs->Scalar_axy(alpha, pvec, xvec, xvec);

    //r = -alpha * Ap + r
    bs->Scalar_axy(-alpha, Ap, rvec, rvec);

    std::memset(zvec, 0, sizeof(double)*N);

    //Az = r
    in->innerSelect(this->coll, this->coll->innerSolver, cu, bs, rvec, zvec);

    //Az
    if(isCUDA){
      cu->MtxVec_mult(zvec, Av, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
    }else{
      bs->MtxVec_mult(zvec, Av);
    }

    //(z, Az)
    if(isCUDA){
      // zaz2 = cu->dot(zvec, Av);
      zaz2 = bs->dot(zvec, Av);
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

    if(this->coll->isCUDA){
      this->coll->time->showTimeOnGPU(time.getTime(), this->coll->time->showTimeOnCPU(time.getTime(), true));
    }else{
      this->coll->time->showTimeOnCPU(time.getTime());
    }

    for(long int i=0; i<N; i++){
      f_x << i << " " << std::scientific << std::setprecision(12) << std::uppercase << xvec[i] << std::endl;
    }
  }else{
  }

  return exit_flag;
}

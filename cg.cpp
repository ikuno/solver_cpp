#include <iomanip>
#include <cstring>
#include <cstdlib>
#include "color.hpp"
#include "cg.hpp"

cg::cg(collection *coll, double *bvec, double *xvec, bool inner, cuda *cu, blas *bs){
  this->coll = coll;
  isInner = inner;
  if(isInner){
    this->bs = bs;
    this->cu = cu;
  }else{
    bs = new blas(this->coll, this->coll->time);
    cu = new cuda(this->coll->time, this->coll->N);
  }

  exit_flag = 2;
  isVP = this->coll->isVP;
  isVerbose = this->coll->isVerbose;
  isCUDA = this->coll->isCUDA;

  N = this->coll->N;
  if(isCUDA){
    rvec = cu->d_MallocHost(N);
    pvec = cu->d_MallocHost(N);
    mv = cu->d_MallocHost(N);
    x_0 = cu->d_MallocHost(N);
  }else{
    rvec = new double [N];
    pvec = new double [N];
    mv = new double [N];
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
  std::memset(mv, 0, sizeof(double)*N);
  std::memset(xvec, 0, sizeof(double)*N);

  if(!isInner){
    f_his.open("./output/CG_his.txt");
    if(!f_his.is_open()){
      std::cerr << "File open error" << std::endl;
      std::exit(-1);
    }

    f_x.open("./output/CG_xvec.txt");
    if(!f_x.is_open()){
      std::cerr << "File open error" << std::endl;
      std::exit(-1);
    }
  }

}

cg::~cg(){

  if(isCUDA){
    cu->FreeHost(rvec);
    cu->FreeHost(pvec);
    cu->FreeHost(mv);
    cu->FreeHost(x_0);
  }else{
    delete[] rvec;
    delete[] pvec;
    delete[] mv;
    delete[] x_0;
  }

  if(!isInner){
    delete this->bs;
    delete this->cu;
  }
  f_his.close();
  f_x.close();
}

int cg::solve(){

  time.start();

  //x_0 = x
  bs->Vec_copy(xvec, x_0);

  //b 2norm
  bnorm = bs->norm_2(bvec);

  //mv = Ax
  if(isCUDA){
    cu->MtxVec_mult(xvec, mv, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
  }else{
    bs->MtxVec_mult(xvec, mv);
  }

  //r = b - Ax
  bs->Vec_sub(bvec, mv, rvec);


  //p = r
  bs->Vec_copy(rvec, pvec);

  //r dot
  if(isCUDA){
    // rr = cu->dot(rvec, rvec, this->coll->N);
    rr = bs->dot(rvec, rvec);
  }else{
    rr = bs->dot(rvec, rvec);
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
      cu->MtxVec_mult(pvec, mv, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
    }else{
      bs->MtxVec_mult(pvec, mv);
    }

    //alpha = (r,r) / (p,ap)
    if(isCUDA){
      // dot = cu->dot(pvec, mv);
      dot = bs->dot(pvec, mv);
    }else{
      dot = bs->dot(pvec, mv);
    }
    alpha = rr / dot;

    //x = alpha * pvec + x
    bs->Scalar_axy(alpha, pvec, xvec, xvec);

    //r = -alpha * AP(mv) + r
    bs->Scalar_axy(-alpha, mv, rvec, rvec);

    //rr2 dot
    if(isCUDA){
      // rr2 = cu->dot(rvec, rvec);
      rr2 = bs->dot(rvec, rvec);
    }else{
      rr2 = bs->dot(rvec, rvec);
    }

    beta = rr2/rr;

    rr = rr2;

    //p = beta * p + r
    bs->Scalar_axy(beta, pvec, rvec, pvec);
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

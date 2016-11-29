#include <iomanip>
#include <cstring>
#include <cstdlib>
#include "color.hpp"
#include "vpcg.hpp"

vpcg::vpcg(collection *coll, double *bvec, double *xvec, bool inner){
  this->coll = coll;
  bs = new blas(this->coll);
  in = new innerMethods(this->coll);

  N = this->coll->N;
  rvec = new double [N];
  pvec = new double [N];
  zvec = new double [N];
  mv = new double [N];
  x_0 = new double [N];

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

  std::memset(rvec, 0, sizeof(double)*N);
  std::memset(pvec, 0, sizeof(double)*N);
  std::memset(mv, 0, sizeof(double)*N);
  std::memset(xvec, 0, sizeof(double)*N);
  std::memset(zvec, 0, sizeof(double)*N);

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

vpcg::~vpcg(){
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

int vpcg::solve(){

  time.start();

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

    std::memset(zvec, 0, sizeof(double)*N);

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

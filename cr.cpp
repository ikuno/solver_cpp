#include <iomanip>
#include <cstring>
#include <cstdlib>
#include "color.hpp"
#include "cr.hpp"

cr::cr(collection *coll, double *bvec, double *xvec, bool inner){
  this->coll = coll;
  bs = new blas(this->coll);

  N = this->coll->N;
  rvec = new double [N];
  pvec = new double [N];
  qvec = new double [N];
  svec = new double [N];
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
  std::memset(qvec, 0, sizeof(double)*N);
  std::memset(svec, 0, sizeof(double)*N);
  std::memset(xvec, 0, sizeof(double)*N);

  if(!isInner){
    f_his.open("./output/CR_his.txt");
    if(!f_his.is_open()){
      std::cerr << "File open error" << std::endl;
      std::exit(-1);
    }

    f_x.open("./output/CR_xvec.txt");
    if(!f_x.is_open()){
      std::cerr << "File open error" << std::endl;
      std::exit(-1);
    }
  }

}

cr::~cr(){
  delete this->bs;
  delete[] rvec;
  delete[] pvec;
  delete[] qvec;
  delete[] svec;
  delete[] x_0;
  f_his.close();
  f_x.close();
}

int cr::solve(){

  time.start();

  //x_0 = x
  bs->Vec_copy(xvec, x_0);

  //b 2norm
  bnorm = bs->norm_2(bvec);

  //qvec = Ax
  if(isCUDA){

  }else{
    bs->MtxVec_mult(xvec, qvec);
  }

  //r = b - Ax(qvec)
  bs->Vec_sub(bvec, qvec, rvec);


  //p = r
  bs->Vec_copy(rvec, pvec);

  //qvec = Ap
  if(isCUDA){

  }else{
    bs->MtxVec_mult(pvec, qvec);
  }

  //s = q
  bs->Vec_copy(qvec, svec);

  //(r, s)
  if(isCUDA){

  }else{
    rs = bs->dot(rvec, svec);
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

    //alpha = (r,s) / (q,q)
    if(isCUDA){
    }else{
      dot = bs->dot(qvec, qvec);
    }
    alpha = rs / dot;

    //x = alpha * pvec + x
    bs->Scalar_axy(alpha, pvec, xvec, xvec);

    //r = -alpha * qvec + r
    bs->Scalar_axy(-alpha, qvec, rvec, rvec);

    //s=Ar
    if(isCUDA){

    }else{
      bs->MtxVec_mult(rvec, svec);
    }

    //r2=(r, s)
    if(isCUDA){

    }else{
      rs2 = bs->dot(rvec, svec);
    }

    //beta=(r_new, s_new)/(r, s)
    beta = rs2/rs;

    rs = rs2;

    //p = beta * p + r
    bs->Scalar_axy(beta, pvec, rvec, pvec);

    //q = beta * q + s
    bs->Scalar_axy(beta, qvec, svec, qvec);
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

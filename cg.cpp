#include "cg.hpp"

cg::cg(collection *coll){
  this->coll = coll;
  bs = new blas(this->coll);

  N = this->coll->N;
  rvec = new double [N];
  pvec = new double [N];
  mv = new double [N];
  x_0 = new double [N];

  xvec = this->coll->xvec;
  bvec = this->coll->bvec;

  exit_flag = 2;
  isVP = this->coll->isVP;
  isVerbose = this->coll->isVerbose;
  isCUDA = this->coll->isCUDA;
  isInner = this->coll->isInner;

  if(isVP && this->coll->isInner ){
    maxloop = this->coll->innerMaxLoop;
    eps = this->coll->innerEps;
  }else{
    maxloop = this->coll->outerMaxLoop;
    eps = this->coll->outerEps;
  }

  for(long int i=0; i<N; i++){
    rvec[i] = 0.0;
    pvec[i] = 0.0;
    mv[i] = 0.0;
    xvec[i] = 0.0;
  }
}

cg::~cg(){
  delete this->bs;
  delete[] rvec;
  delete[] pvec;
  delete[] mv;
  delete[] x_0;
}

int cg::solve(){
  //x_0 = x
  bs->Vec_copy<double>(xvec, x_0, N);

  //b 2norm
  bnorm = bs->norm_2<double>(bvec, N);


  //mv = Ax
  if(isCUDA){

  }else{
    bs->MtxVec_mult<double>(xvec, mv, N);
  }

  //r = b - Ax
  bs->Vec_sub<double>(bvec, mv, rvec, N);


  //p = r
  bs->Vec_copy<double>(rvec, pvec, N);

  //r dot
  if(isCUDA){

  }else{
    rr = bs->dot<double>(rvec, rvec, N);
  }

  for(loop=0; loop<maxloop; loop++){
    rnorm = bs->norm_2<double>(rvec, N);
    error = rnorm/bnorm;
    if(!isInner){
      if(isVerbose){
        std::cout << loop+1 << " " << std::scientific << std::setprecision(12) << std::uppercase << error << std::endl;
      }
    }
    if(error <= eps){
      exit_flag = 0;
      break;
    }

    //mv = Ap
    if(isCUDA){

    }else{
      bs->MtxVec_mult<double>(pvec, mv, N);
    }

    //alpha = (r,r) / (p,ap)
    if(isCUDA){
    }else{
      dot = bs->dot<double>(rvec, mv, N);
    }
    alpha = rr / dot;

    //x = alpha * pvec + x
    bs->Scalar_axy<double>(alpha, pvec, xvec, xvec, N);

    //r = -alpha * AP(mv) + r
    bs->Scalar_axy<double>(-alpha, mv, rvec, rvec, N);

    //rr2 dot
    if(isCUDA){

    }else{
      rr2 = bs->dot<double>(rvec, rvec, N);
    }

    beta = rr2/rr;

    rr = rr2;

    //p = beta * p + r
    bs->Scalar_axy<double>(beta, pvec, rvec, pvec, N);
  }

  if(!isInner){
    test_error = bs->Check_error(xvec, x_0, N);
    std::cout << "|b-ax|2/|b|2 = " << std::fixed << std::setprecision(1) << test_error << std::endl;
    std::cout << "loop = " << loop+1 << std::endl;
  }else{

  }

  return exit_flag;
}

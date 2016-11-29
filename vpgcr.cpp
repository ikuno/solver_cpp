#include <iomanip>
#include <cstring>
#include <cstdlib>
#include "color.hpp"
#include "vpgcr.hpp"

vpgcr::vpgcr(collection *coll, double *bvec, double *xvec, bool inner){
  this->coll = coll;
  bs = new blas(this->coll);
  in = new innerMethods(this->coll);

  exit_flag = 2;
  isVP = this->coll->isVP;
  isVerbose = this->coll->isVerbose;
  isCUDA = this->coll->isCUDA;
  isInner = inner;


  if(isVP && isInner ){
    maxloop = this->coll->innerMaxLoop;
    eps = this->coll->innerEps;
    restart = this->coll->innerRestart;
  }else{
    maxloop = this->coll->outerMaxLoop;
    eps = this->coll->outerEps;
    restart = this->coll->outerRestart;
  }

  loop = 1;

  N = this->coll->N;

  this->xvec = xvec;
  this->bvec = bvec;

  rvec = new double [N];
  zvec = new double [N];
  Av = new double [N];
  x_0 = new double [N];
  qq = new double [restart];
  qvec = new double* [restart];
  pvec = new double* [restart];
  for(long int i=0; i<restart; i++){
    qvec[i] = new double [N];
    pvec[i] = new double [N];
  }

  
  std::memset(rvec, 0, sizeof(double)*N);
  std::memset(Av, 0, sizeof(double)*N);
  std::memset(xvec, 0, sizeof(double)*N);
  std::memset(zvec, 0, sizeof(double)*N);

  for(long int i=0; i<restart; i++){
    qq[i] = 0.0;
  }
  std::memset(qq, 0, sizeof(double)*restart);

  for(int i=0; i<restart; i++){
    std::memset(qvec[i], 0, sizeof(double)*N);
    std::memset(pvec[i], 0, sizeof(double)*N);
  }

  f_his.open("./output/VPGCR_his.txt");
  if(!f_his.is_open()){
    std::cerr << "File open error" << std::endl;
    exit(-1);
  }

  f_x.open("./output/VPGCR_xvec.txt");
  if(!f_x.is_open()){
    std::cerr << "File open error" << std::endl;
    exit(-1);
  }

  out_flag = false;

}

vpgcr::~vpgcr(){
  delete this->bs;
  delete this->in;
  delete[] rvec;
  delete[] Av;
  delete[] qq;
  delete[] x_0;
  for(int i=0; i<restart; i++){
    delete[] qvec[i];
    delete[] pvec[i];
  }
  delete[] qvec;
  delete[] pvec;
  delete[] zvec;
  f_his.close();
  f_x.close();
}

int vpgcr::solve(){

  time.start();

  //x_0 = x
  bs->Vec_copy(xvec, x_0);

  //b 2norm
  bnorm = bs->norm_2(bvec);

  while(loop<maxloop){
    //Ax
    if(isCUDA){

    }else{
      bs->MtxVec_mult(xvec, Av);
    }

    //r=b-Ax
    bs->Vec_sub(bvec, Av, rvec);

    std::memset(pvec[0], 0, sizeof(double)*N);

    //Ap p = r
    in->innerSelect(this->coll, this->coll->innerSolver, rvec, pvec[0]);

    //q[0*ndata+x]=A*p[0*ndata+x]
    if(isCUDA){

    }else{
      bs->MtxVec_mult(pvec[0], qvec[0]);
    }

    for(kloop=0; kloop<restart; kloop++){
      rnorm = bs->norm_2(rvec);
      error = rnorm / bnorm;
      if(!isInner){
        if(isVerbose){
          std::cout << loop << " " << std::scientific << std::setprecision(12) << std::uppercase << error << std::endl;
        }
        f_his << loop << " " << std::scientific << std::setprecision(12) << std::uppercase << error << std::endl;
      }
      if(error <= eps){
        exit_flag = 0;
        out_flag = true;
        break;
      }else if(loop >= maxloop){
        exit_flag = 2;
        out_flag = true;
        break;
      }
      loop++;

      //(q, q)
      if(isCUDA){

      }else{
        dot_tmp = bs->dot(qvec[kloop], qvec[kloop]);
      }
      qq[kloop] = dot_tmp;

      //alpha = (r, q)/(q, q)
      if(isCUDA){

      }else{
        dot_tmp = bs->dot(rvec, qvec[kloop]);
      }
      alpha = dot_tmp / qq[kloop];

      //x = alpha * pvec[k] + xvec
      bs->Scalar_axy(alpha, pvec[kloop], xvec, xvec);

      if(kloop == restart-1){
        break;
      }

      //r = -alpha * qvec[k] + rvec
      bs->Scalar_axy(-alpha, qvec[kloop], rvec, rvec);

      std::memset(zvec, 0, sizeof(double)*N);

      //Az = r
      in->innerSelect(this->coll, this->coll->innerSolver, rvec, zvec);

      //Az = r
      if(isCUDA){

      }else{
        bs->MtxVec_mult(zvec, Av);
      }

      //init p[k+1] q[k+1]
      std::memset(pvec[kloop+1], 0, sizeof(double)*N);
      std::memset(qvec[kloop+1], 0, sizeof(double)*N);

      for(iloop=0; iloop<=kloop; iloop++){
        //beta = -(Av, qvec) / (q, q)
        if(isCUDA){

        }else{
          dot_tmp = bs->dot(Av, qvec[iloop]);
        }
        beta = -(dot_tmp) / qq[iloop];

        //pvec[k+1] = beta * pvec[i] + pvec[k+1]
        bs->Scalar_axy(beta, pvec[iloop], pvec[kloop+1], pvec[kloop+1]);
        //qvec[k+1] = beta * qvec[i] + qvec[k+1]
        bs->Scalar_axy(beta, qvec[iloop], qvec[kloop+1], qvec[kloop+1]);
      }

      //p[k+1] = r + p[k+1]
      bs->Vec_add(zvec, pvec[kloop+1], pvec[kloop+1]);

      //q[k+1] = Av + q[k+1]
      bs->Vec_add(Av, qvec[kloop+1], qvec[kloop+1]);
    }
    if(out_flag){
      break;
    }
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

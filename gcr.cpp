#include <iomanip>
#include <cstring>
#include <cstdlib>
#include "color.hpp"
#include "gcr.hpp"

gcr::gcr(collection *coll, double *bvec, double *xvec, bool inner){
  this->coll = coll;
  bs = new blas(this->coll);
  cu = new cuda(this->coll->N);

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

  if(isCUDA){
    rvec = cu->d_MallocHost(N);
    Av = cu->d_MallocHost(N);
    x_0 = cu->d_MallocHost(N);
    qq = cu->d_MallocHost(restart);
    qvec = cu->d_MallocHost(restart * N);
    pvec = cu->d_MallocHost(restart * N);
  }else{
    rvec = new double [N];
    Av = new double [N];
    x_0 = new double [N];
    qq = new double [restart];
    qvec = new double [restart * N];
    pvec = new double [restart * N];
  }

  
  std::memset(rvec, 0, sizeof(double)*N);
  std::memset(Av, 0, sizeof(double)*N);
  std::memset(xvec, 0, sizeof(double)*N);
  std::memset(qq, 0, sizeof(double)*restart);

  std::memset(qvec, 0, sizeof(double)*(restart * N));
  std::memset(pvec, 0, sizeof(double)*(restart * N));

  if(!isInner){
    f_his.open("./output/GCR_his.txt");
    if(!f_his.is_open()){
      std::cerr << "File open error" << std::endl;
      exit(-1);
    }

    f_x.open("./output/GCR_xvec.txt");
    if(!f_x.is_open()){
      std::cerr << "File open error" << std::endl;
      exit(-1);
    }
  }
  out_flag = false;
}

gcr::~gcr(){
  delete this->bs;
  if(isCUDA){
    cu->FreeHost(rvec);
    cu->FreeHost(Av);
    cu->FreeHost(qq);
    cu->FreeHost(x_0);
    cu->FreeHost(qvec);
    cu->FreeHost(pvec);
  }else{
    delete[] rvec;
    delete[] Av;
    delete[] qq;
    delete[] x_0;
    delete[] qvec;
    delete[] pvec;
  }
  delete cu;
  f_his.close();
  f_x.close();
}

int gcr::solve(){

  time.start();

  //x_0 = x
  bs->Vec_copy(xvec, x_0);

  //b 2norm
  bnorm = bs->norm_2(bvec);

  while(loop<maxloop){
    //Ax
    if(isCUDA){
      cu->MtxVec_mult(xvec, Av, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
    }else{
      bs->MtxVec_mult(xvec, Av);
    }

    //r=b-Ax
    bs->Vec_sub(bvec, Av, rvec);

    //p=r
    bs->Vec_copy(rvec, pvec, 0, N);

    //Ap
    if(isCUDA){
      cu->MtxVec_mult(pvec, 0, N, qvec, 0, N, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
    }else{
      bs->MtxVec_mult(pvec, 0, N, qvec, 0, N);
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
        // dot_tmp = cu->dot(qvec, kloop, N, qvec, kloop, N);
        dot_tmp = bs->dot(qvec, kloop, N, qvec, kloop, N);
      }else{
        dot_tmp = bs->dot(qvec, kloop, N, qvec, kloop, N);
      }
      qq[kloop] = dot_tmp;

      //alpha = (r, q)/(q, q)
      if(isCUDA){
        // dot_tmp = cu->dot(rvec, qvec, kloop, N);
        dot_tmp = bs->dot(rvec, qvec, kloop, N);
      }else{
        dot_tmp = bs->dot(rvec, qvec, kloop, N);
      }
      alpha = dot_tmp / qq[kloop];

      //x = alpha * pvec[k] + xvec
      bs->Scalar_axy(alpha, pvec, kloop, N, xvec, xvec);
      if(kloop == restart-1){
        break;
      }

      //r = -alpha * qvec[k] + rvec
      bs->Scalar_axy(-alpha, qvec, kloop, N, rvec, rvec);

      //Ar
      if(isCUDA){
        cu->MtxVec_mult(rvec, Av, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
      }else{
        bs->MtxVec_mult(rvec, Av);
      }

      //init p[k+1] q[k+1]
      for(int i=0; i<N; i++){
        pvec[(kloop+1)*N+i] = 0;
        qvec[(kloop+1)*N+i] = 0;
      }


      for(iloop=0; iloop<=kloop; iloop++){
        //beta = -(Av, qvec) / (q, q)
        if(isCUDA){
          // dot_tmp = cu->dot(Av, qvec, iloop, N);
          dot_tmp = bs->dot(Av, qvec, iloop, N);
        }else{
          dot_tmp = bs->dot(Av, qvec, iloop, N);
        }
        beta = -(dot_tmp) / qq[iloop];

        //pvec[k+1] = beta * pvec[i] + pvec[k+1]
        bs->Scalar_axy(beta, pvec, iloop, N, pvec, kloop+1, N, pvec, kloop+1, N);
        //qvec[k+1] = beta * qvec[i] + qvec[k+1]
        bs->Scalar_axy(beta, qvec, iloop, N, qvec, kloop+1, N, qvec, kloop+1, N);
      }
      //p[k+1] = r + p[k+1]
      bs->Vec_add(rvec, pvec, kloop+1, N, pvec, kloop+1, N);
      //q[k+1] = Av + q[k+1]
      bs->Vec_add(Av, qvec, kloop+1, N, qvec, kloop+1, N);
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

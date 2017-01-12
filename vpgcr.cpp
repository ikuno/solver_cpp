#include <iomanip>
#include <cstring>
#include <cstdlib>
#include "color.hpp"
#include "vpgcr.hpp"

vpgcr::vpgcr(collection *coll, double *bvec, double *xvec, bool inner){
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
    if(isPinned){
      rvec = new double [N];
      zvec = cu->d_MallocHost(N);
      Av = cu->d_MallocHost(N);
      x_0 = new double [N];
      qq = new double [restart];
      qvec = cu->d_MallocHost(restart * N);
      pvec = cu->d_MallocHost(restart * N);

      beta_vec = cu->d_MallocHost(restart);
    }else{
      rvec = new double [N];
      zvec = new double [N];
      Av = new double [N];
      x_0 = new double [N];
      qq = new double [restart];
      qvec = new double [restart * N];
      pvec = new double [restart * N];

      beta_vec = new double [restart];
    }
  }else{
    rvec = new double [N];
    zvec = new double [N];
    Av = new double [N];
    x_0 = new double [N];
    qq = new double [restart];
    qvec = new double [restart * N];
    pvec = new double [restart * N];

    beta_vec = new double [restart];
  }

  
  std::memset(rvec, 0, sizeof(double)*N);
  std::memset(Av, 0, sizeof(double)*N);
  std::memset(xvec, 0, sizeof(double)*N);
  std::memset(zvec, 0, sizeof(double)*N);

  std::memset(qq, 0, sizeof(double)*restart);

  std::memset(qvec, 0, sizeof(double)*restart*N);
  std::memset(pvec, 0, sizeof(double)*restart*N);

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

  if(isVP){
    std::string name = this->coll->enum2string(this->coll->innerSolver);
    name = "./output/" + name + "_inner.txt";
    std::ifstream in(name.c_str());
    if(in.good()){
      std::cout << "Delete inner solver's old file" << std::endl;
      std::remove(name.c_str());
    }else{
      std::cout << "Has no inner solver's file yet" << std::endl;
    }
  }

}

vpgcr::~vpgcr(){
  if(isCUDA){
    if(isPinned){
      delete[] rvec;
      cu->FreeHost(Av);
      delete[] qq;
      delete[] x_0;
      cu->FreeHost(qvec);
      cu->FreeHost(pvec);
      cu->FreeHost(zvec);

      cu->FreeHost(beta_vec);
    }else{
      delete[] rvec;
      delete[] Av;
      delete[] qq;
      delete[] x_0;
      delete[] qvec;
      delete[] pvec;
      delete[] zvec;

      delete[] beta_vec;
    }
  }else{
    delete[] rvec;
    delete[] Av;
    delete[] qq;
    delete[] x_0;
    delete[] qvec;
    delete[] pvec;
    delete[] zvec;

    delete[] beta_vec;
  }
  delete this->bs;
  delete this->in;
  delete this->cu;

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
      if(isMultiGPU){
      // bs->MtxVec_mult(xvec, Av);
        cu->MtxVec_mult_Multi(xvec, Av, this->coll->Cval1, this->coll->Ccol1, this->coll->Cptr1, this->coll->Cval2, this->coll->Ccol2, this->coll->Cptr2);
      }else{
        cu->MtxVec_mult(xvec, Av, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
      }
    }else{
      bs->MtxVec_mult(xvec, Av);
    }
    //r=b-Ax
    bs->Vec_sub(bvec, Av, rvec);

    // std::memset(pvec[0], 0, sizeof(double)*N);
    for(int i=0; i<N; i++){
      pvec[0*N+i] = 0.0;
    }

    //Ap p = r
    in->innerSelect(this->coll, this->coll->innerSolver, cu, bs, rvec, Av);

    //copy  Av -> pvec[0]
    bs->Vec_copy(Av, pvec, 0, N);

    //Ap
    if(isCUDA){
      if(isMultiGPU){
      // bs->MtxVec_mult((double*)(pvec+(0*N)), (double*)(qvec+(0*N)));
        cu->MtxVec_mult_Multi((double*)(pvec+(0*N)), (double*)(qvec+(0*N)), this->coll->Cval1, this->coll->Ccol1, this->coll->Cptr1, this->coll->Cval2, this->coll->Ccol2, this->coll->Cptr2); 
      }else{
        cu->MtxVec_mult((double*)(pvec+(0*N)), (double*)(qvec+(0*N)), this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
      }
    }else{
      bs->MtxVec_mult((double*)(pvec+(0*N)), (double*)(qvec+(0*N)));
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
        dot_tmp = bs->dot((double*)(qvec+(kloop*N)), (double*)(qvec+(kloop*N)));
      }else{
        dot_tmp = bs->dot((double*)(qvec+(kloop*N)), (double*)(qvec+(kloop*N)));
      }
      qq[kloop] = dot_tmp;

      //alpha = (r, q)/(q, q)
      if(isCUDA){
        // dot_tmp = cu->dot(rvec, qvec, kloop, N);
        dot_tmp = bs->dot(rvec, (double*)(qvec+(kloop*N)));
        
      }else{
        dot_tmp = bs->dot(rvec, qvec, kloop, N);
      }
      alpha = dot_tmp / qq[kloop];

      //x = alpha * pvec[k] + xvec
      bs->Scalar_axy(alpha, (double*)(pvec+(kloop*N)), xvec, xvec);

      if(kloop == restart-1){
        break;
      }

      //r = -alpha * qvec[k] + rvec
      bs->Scalar_axy(-alpha, (double*)(qvec+(kloop*N)), rvec, rvec);

      std::memset(zvec, 0, sizeof(double)*N);

      //Az = r
      in->innerSelect(this->coll, this->coll->innerSolver, cu, bs, rvec, zvec);

      //Az = r
      if(isCUDA){
        if(isMultiGPU){
          cu->MtxVec_mult_Multi(zvec, Av, this->coll->Cval1, this->coll->Ccol1, this->coll->Cptr1, this->coll->Cval2, this->coll->Ccol2, this->coll->Cptr2);
        // bs->MtxVec_mult(zvec, Av);
        }else{
          cu->MtxVec_mult(zvec, Av, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
        }
      }else{
        bs->MtxVec_mult(zvec, Av);
      }

      //init p[k+1] q[k+1]
      for(int i=0; i<N; i++){
        pvec[(kloop+1)*N+i] = 0.0;
        qvec[(kloop+1)*N+i] = 0.0;
      }

      // for(iloop=0; iloop<=kloop; iloop++){
      //   //beta = -(Av, qvec) / (q, q)
      //   if(isCUDA){
      //     // dot_tmp = cu->dot(Av, qvec, iloop, N);
      //     dot_tmp = bs->dot(Av, (double*)(qvec+(iloop*N)));
      //   }else{
      //     dot_tmp = bs->dot(Av, (double*)(qvec+(iloop*N)));
      //   }
      //   beta = -(dot_tmp) / qq[iloop];
      //
      //   //pvec[k+1] = beta * pvec[i] + pvec[k+1]
      //   bs->Scalar_axy(beta, (double*)(pvec+(iloop*N)), (double*)(pvec+((kloop+1)*N)), (double*)(pvec+((kloop+1)*N)));
      //   bs->Scalar_axy(beta, (double*)(qvec+(iloop*N)), (double*)(qvec+((kloop+1)*N)), (double*)(qvec+((kloop+1)*N)));
      // }

      if(isCUDA){
        bs->Gcr_sp_1(kloop, N, Av, qvec, pvec, qq, beta_vec);
      }else{
        bs->Gcr_sp_1(kloop, N, Av, qvec, pvec, qq, beta_vec);
      }

      //p[k+1] = r + p[k+1]
      bs->Vec_add(zvec, (double*)(pvec+((kloop+1)*N)), (double*)(pvec+((kloop+1)*N)));

      //q[k+1] = Av + q[k+1]
      bs->Vec_add(Av, (double*)(qvec+((kloop+1)*N)), (double*)(qvec+((kloop+1)*N)));
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

    if(this->coll->isCUDA){
      this->coll->time->showTimeOnGPU(time.getTime(), this->coll->time->showTimeOnCPU(time.getTime(), true));
    }else{
      this->coll->time->showTimeOnCPU(time.getTime());
    }

    for(long int i=0; i<N; i++){
      f_x << i << " " << std::scientific << std::setprecision(12) << std::uppercase << xvec[i] << std::endl;
    }
  }else{
    if(exit_flag==0){
      std::cout << GREEN << "\t" <<  loop << " = " << std::scientific << std::setprecision(12) << std::uppercase << error << RESET << std::endl;
    }else if(exit_flag==2){
      std::cout << RED << "\t" << loop << " = " << std::scientific << std::setprecision(12) << std::uppercase << error << RESET << std::endl;
    }else{
      std::cout << RED << " ERROR " << loop << RESET << std::endl;
    }
  }

  return exit_flag;
}

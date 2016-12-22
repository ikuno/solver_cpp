#include <iomanip>
#include <cstring>
#include <cstdlib>
#include "color.hpp"
#include "cr.hpp"

cr::cr(collection *coll, double *bvec, double *xvec, bool inner, cuda *a_cu, blas *a_bs){
  this->coll = coll;
  isInner = inner;
  isMultiGPU = this->coll->isMultiGPU;
  if(isInner){
    this->bs = a_bs;
    this->cu = a_cu;
  }else{
    bs = new blas(this->coll, this->coll->time);
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
  isPinned = this->coll->isPinned;

  N = this->coll->N;
  if(isCUDA){
    if(isPinned){
    rvec = cu->d_MallocHost(N);
    pvec = cu->d_MallocHost(N);
    qvec = cu->d_MallocHost(N);
    svec = cu->d_MallocHost(N);
    x_0 = new double [N];
    }else{
      rvec = new double [N];
      pvec = new double [N];
      qvec = new double [N];
      svec = new double [N];
      x_0 = new double [N];
    }
  }else{
    rvec = new double [N];
    pvec = new double [N];
    qvec = new double [N];
    svec = new double [N];
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
  }else{
    f_in.open("./output/CR_inner.txt", std::ofstream::out | std::ofstream::app);
    if(!f_in.is_open()){
      std::cerr << "File open error inner" << std::endl;
      std::exit(-1);
    }
  }


}

cr::~cr(){
  if(isCUDA){
    if(isPinned){
      cu->FreeHost(rvec);
      cu->FreeHost(pvec);
      cu->FreeHost(qvec);
      cu->FreeHost(svec);
      delete[] x_0;
    }else{
      delete[] rvec;
      delete[] pvec;
      delete[] qvec;
      delete[] svec;
      delete[] x_0;
    }
  }else{
    delete[] rvec;
    delete[] pvec;
    delete[] qvec;
    delete[] svec;
    delete[] x_0;
  }

  if(!isInner){
    delete this->bs;
    delete cu;
  }
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
    if(isMultiGPU){
      cu->MtxVec_mult_Multi(xvec, qvec, this->coll->Cval1, this->coll->Ccol1, this->coll->Cptr1, this->coll->Cval2, this->coll->Ccol2, this->coll->Cptr2);
    }else{
      cu->MtxVec_mult(xvec, qvec, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
    }
  }else{
    bs->MtxVec_mult(xvec, qvec);
  }

  //r = b - Ax(qvec)
  bs->Vec_sub(bvec, qvec, rvec);


  //p = r
  bs->Vec_copy(rvec, pvec);

  //qvec = Ap
  if(isCUDA){
    if(isMultiGPU){
      cu->MtxVec_mult_Multi(pvec, qvec, this->coll->Cval1, this->coll->Ccol1, this->coll->Cptr1, this->coll->Cval2, this->coll->Ccol2, this->coll->Cptr2);
    }else{
      cu->MtxVec_mult(pvec, qvec, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
    }
  }else{
    bs->MtxVec_mult(pvec, qvec);
  }

  //s = q
  bs->Vec_copy(qvec, svec);

  //(r, s)
  if(isCUDA){
    // rs = cu->dot(rvec, svec);
    rs = bs->dot(rvec, svec);
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
      // dot = cu->dot(qvec, qvec);
      dot = bs->dot(qvec, qvec);
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
      if(isMultiGPU){
        cu->MtxVec_mult_Multi(rvec, svec, this->coll->Cval1, this->coll->Ccol1, this->coll->Cptr1, this->coll->Cval2, this->coll->Ccol2, this->coll->Cptr2);
      }else{
        cu->MtxVec_mult(rvec, svec, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
      }
    }else{
      bs->MtxVec_mult(rvec, svec);
    }

    //r2=(r, s)
    if(isCUDA){
      // rs2 = cu->dot(rvec, svec);
      rs2 = bs->dot(rvec, svec);
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
    f_in << loop << std::endl;
  }

  return exit_flag;
}

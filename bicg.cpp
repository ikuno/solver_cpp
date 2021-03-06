#include <iomanip>
#include <cstring>
#include <cstdlib>
#include "color.hpp"
#include "bicg.hpp"

// bicg::bicg(collection *coll, double *bvec, double *xvec, bool inner, cuda *a_cu, blas *a_bs){
bicg::bicg(collection *coll, double *bvec, double *xvec, bool inner, cuda *a_cu, blas *a_bs, double **list){
  this->coll = coll;
  this->coll->time->start();
  isInner = inner;
  isMultiGPU = this->coll->isMultiGPU;
  if(isInner){
    this->bs = a_bs;
    this->cu = a_cu;
  }else{
    bs = new blas(this->coll, this->coll->time);
    if(this->coll->isInnerKskip){
      cu = new cuda(this->coll->time, this->coll->N, this->coll->innerKskip);
    }else{
      if(isMultiGPU){
        cu = new cuda(this->coll->time, this->coll->N, this->coll->N1, this->coll->N2);
      }else{
        cu = new cuda(this->coll->time, this->coll->N);
      }
    }
  }

  exit_flag = 2;
  isVP = this->coll->isVP;
  isVerbose = this->coll->isVerbose;
  isCUDA = this->coll->isCUDA;
  isPinned = this->coll->isPinned;

  N = this->coll->N;

  if(isInner){

    rvec = list[0];
    pvec = list[1];
    r_vec = list[2];
    p_vec = list[3];
    mv = list[4];
    x_0 = list[5];

  }else{

    if(isCUDA){
      if(isPinned){
        rvec = new double [N];
        pvec = cu->d_MallocHost(N);
        r_vec = new double [N];
        p_vec = cu->d_MallocHost(N);
        mv = cu->d_MallocHost(N);
        x_0 = new double [N];
      }else{
        rvec = new double [N];
        pvec = new double [N];
        r_vec = new double [N];
        p_vec = new double [N];
        mv = new double [N];
        x_0 = new double [N];
      }
    }else{
      rvec = new double [N];
      pvec = new double [N];
      r_vec = new double [N];
      p_vec = new double [N];
      mv = new double [N];
      x_0 = new double [N];
    }

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

  // std::memset(rvec, 0, sizeof(double)*N);
  // std::memset(pvec, 0, sizeof(double)*N);
  // std::memset(r_vec, 0, sizeof(double)*N);
  // std::memset(p_vec, 0, sizeof(double)*N);
  // std::memset(mv, 0, sizeof(double)*N);
  std::memset(xvec, 0, sizeof(double)*N);
  
  if(!isInner){
    f_his.open("./output/BICG_his.txt");
    if(!f_his.is_open()){
      std::cerr << "File open error" << std::endl;
      exit(-1);
    }

    f_x.open("./output/BICG_xvec.txt");
    if(!f_x.is_open()){
      std::cerr << "File open error" << std::endl;
      exit(-1);
    }
  }else{
    // f_in.open("./output/BICG_inner.txt", std::ofstream::out | std::ofstream::app);
    // if(!f_in.is_open()){
    //   std::cerr << "File open error inner" << std::endl;
    //   std::exit(-1);
    // }
  }

  this->coll->time->end();
  this->coll->time->cons_time += this->coll->time->getTime();

}

bicg::~bicg(){
  this->coll->time->start();
  if(isInner){

  }else{
    if(isCUDA){
      if(isPinned){
        delete[] rvec;
        cu->FreeHost(pvec);
        delete[] r_vec;
        cu->FreeHost(p_vec);
        cu->FreeHost(mv);
        delete[] x_0;
      }else{
        delete[] rvec;
        delete[] pvec;
        delete[] r_vec;
        delete[] p_vec;
        delete[] mv;
        delete[] x_0;
      }
    }else{
      delete[] rvec;
      delete[] pvec;
      delete[] r_vec;
      delete[] p_vec;
      delete[] mv;
      delete[] x_0;
    }
  }

  if(!isInner){
    delete this->bs;
    delete cu;
    f_his.close();
    f_x.close();
  }

  this->coll->time->end();
  this->coll->time->dis_time += this->coll->time->getTime();

}

int bicg::solve(){

  time.start();

  //x_0 = x
  bs->Vec_copy(xvec, x_0);

  //b 2norm
  bnorm = bs->norm_2(bvec);

  //mv = Ax
  if(isCUDA){
    if(isMultiGPU){
      cu->MtxVec_mult_Multi(xvec, mv, this->coll->Cval1, this->coll->Ccol1, this->coll->Cptr1, this->coll->Cval2, this->coll->Ccol2, this->coll->Cptr2);
    }else{
      cu->MtxVec_mult(xvec, mv, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
    }
  }else{
    bs->MtxVec_mult(xvec, mv);
  }

  //r = b - Ax
  bs->Vec_sub(bvec, mv, rvec);

  //r* = r
  bs->Vec_copy(rvec, r_vec);

  //p = r
  bs->Vec_copy(rvec, pvec);

  //p* = *r
  bs->Vec_copy(r_vec, p_vec);

  //r * r*
  if(isCUDA){
    // rr = cu->dot(r_vec, rvec);
    rr = bs->dot(r_vec, rvec);
  }else{
    rr = bs->dot(r_vec, rvec);
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
      if(isMultiGPU){
        cu->MtxVec_mult_Multi(pvec, mv, this->coll->Cval1, this->coll->Ccol1, this->coll->Cptr1, this->coll->Cval2, this->coll->Ccol2, this->coll->Cptr2);
      }else{
        cu->MtxVec_mult(pvec, mv, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
      }
    }else{
      bs->MtxVec_mult(pvec, mv);
    }

    //alpha = (r*,r) / (p*,ap)
    if(isCUDA){
      // dot = cu->dot(p_vec, mv);
      dot = bs->dot(p_vec, mv);
    }else{
      dot = bs->dot(p_vec, mv);
    }
    alpha = rr / dot;

    //x = alpha * pvec + x
    bs->Scalar_axy(alpha, pvec, xvec, xvec);

    //r = -alpha * AP(mv) + r
    bs->Scalar_axy(-alpha, mv, rvec, rvec);

    //mv = A(T)p*
    if(isCUDA){
      if(isMultiGPU){
        cu->MtxVec_mult_Multi(p_vec, mv, this->coll->CTval1, this->coll->CTcol1, this->coll->CTptr1, this->coll->CTval2, this->coll->CTcol2, this->coll->CTptr2);
      }else{
        cu->MtxVec_mult(p_vec, mv, this->coll->CTval, this->coll->CTcol, this->coll->CTptr);
      }
    }else{
      bs->MtxVec_mult(this->coll->Tval, this->coll->Tcol, this->coll->Tptr, p_vec, mv);
    }

    //r* = r* - alpha * A(T)p*
    bs->Scalar_axy(-alpha, mv, r_vec, r_vec);

    //r * r*
    if(isCUDA){
      // rr2 = cu->dot(r_vec, rvec);
      rr2 = bs->dot(r_vec, rvec);
    }else{
      rr2 = bs->dot(r_vec, rvec);
    }

    beta = rr2/rr;

    rr = rr2;

    //p = beta * p + r
    bs->Scalar_axy(beta, pvec, rvec, pvec);
    //p* = beta * p* + r*
    bs->Scalar_axy(beta, p_vec, r_vec, p_vec);

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
    // f_in << loop << std::endl;
  }

  return exit_flag;
}

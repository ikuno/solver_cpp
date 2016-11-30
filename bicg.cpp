#include <iomanip>
#include <cstring>
#include <cstdlib>
#include "color.hpp"
#include "bicg.hpp"

bicg::bicg(collection *coll, double *bvec, double *xvec, bool inner){
  this->coll = coll;
  bs = new blas(this->coll);
  cu = new cuda(this->coll->N);

  exit_flag = 2;
  isVP = this->coll->isVP;
  isVerbose = this->coll->isVerbose;
  isCUDA = this->coll->isCUDA;
  isInner = inner;

  N = this->coll->N;
  if(isCUDA){
    rvec = cu->d_MallocHost(N);
    pvec = cu->d_MallocHost(N);
    r_vec = cu->d_MallocHost(N);
    p_vec = cu->d_MallocHost(N);
    mv = cu->d_MallocHost(N);
    x_0 = cu->d_MallocHost(N);
  }else{
    rvec = new double [N];
    pvec = new double [N];
    r_vec = new double [N];
    p_vec = new double [N];
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
  std::memset(r_vec, 0, sizeof(double)*N);
  std::memset(p_vec, 0, sizeof(double)*N);
  std::memset(mv, 0, sizeof(double)*N);
  std::memset(xvec, 0, sizeof(double)*N);
  
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

}

bicg::~bicg(){
  delete this->bs;
  if(isCUDA){
    cu->FreeHost(rvec);
    cu->FreeHost(pvec);
    cu->FreeHost(r_vec);
    cu->FreeHost(p_vec);
    cu->FreeHost(mv);
    cu->FreeHost(x_0);
  }else{
    delete[] rvec;
    delete[] pvec;
    delete[] r_vec;
    delete[] p_vec;
    delete[] mv;
    delete[] x_0;
  }

  delete cu;
  f_his.close();
  f_x.close();
}

int bicg::solve(){

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

  //r* = r
  bs->Vec_copy(rvec, r_vec);

  //p = r
  bs->Vec_copy(rvec, pvec);

  //p* = *r
  bs->Vec_copy(r_vec, p_vec);

  //r * r*
  if(isCUDA){
    rr = cu->dot(r_vec, rvec);
    // rr = bs->dot(r_vec, rvec);
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
      cu->MtxVec_mult(pvec, mv, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
    }else{
      bs->MtxVec_mult(pvec, mv);
    }

    //alpha = (r*,r) / (p*,ap)
    if(isCUDA){
      dot = cu->dot(p_vec, mv);
      // dot = bs->dot(p_vec, mv);
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
      cu->MtxVec_mult(p_vec, mv, this->coll->CTval, this->coll->CTcol, this->coll->CTptr);
    }else{
      bs->MtxVec_mult(this->coll->Tval, this->coll->Tcol, this->coll->Tptr, p_vec, mv);
    }

    //r* = r* - alpha * A(T)p*
    bs->Scalar_axy(-alpha, mv, r_vec, r_vec);

    //r * r*
    if(isCUDA){
      rr2 = cu->dot(r_vec, rvec);
      // rr2 = bs->dot(r_vec, rvec);
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
      double dot_t = cu->dot_copy_time + cu->dot_proc_time + cu->dot_reduce_time;
      double mv_t = cu->MV_copy_time + cu->MV_proc_time;
      double malloc_t = cu->All_malloc_time;
      double all_t = dot_t + mv_t + malloc_t;
      std::cout << "\tdot copy time   = " << std::setprecision(6) << cu->dot_copy_time << ", " << std::setprecision(2) << cu->dot_copy_time/time.getTime()*100 << "%" << std::endl;
      std::cout << "\tdot proc time   = " << std::setprecision(6) << cu->dot_proc_time <<  ", " << std::setprecision(2) << cu->dot_proc_time/time.getTime()*100 << "%" << std::endl;
      std::cout << "\tdot reduce time = " << std::setprecision(6) << cu->dot_reduce_time <<  ", " << std::setprecision(2) << cu->dot_reduce_time/time.getTime()*100 << "%" << std::endl;
      std::cout << "\t                = " << std::setprecision(6) << dot_t <<  ", " << std::setprecision(2) << dot_t/time.getTime()*100 << "%" << std::endl;
      std::cout << "\tMV copy time    = " << std::setprecision(6) << cu->MV_copy_time <<  ", " << std::setprecision(2) << cu->MV_copy_time/time.getTime()*100 << "%" << std::endl;
      std::cout << "\tMV proc time    = " << std::setprecision(6) << cu->MV_proc_time <<  ", " << std::setprecision(2) << cu->MV_proc_time/time.getTime()*100 << "%" << std::endl;
      std::cout << "\t                = " << std::setprecision(6) << mv_t <<  ", " << std::setprecision(2) << mv_t/time.getTime()*100 << "%" << std::endl;
      std::cout << "\tAll malloc time = " << std::setprecision(6) << cu->All_malloc_time <<  ", " << std::setprecision(2) << cu->All_malloc_time/time.getTime()*100 << "%" << std::endl;
      std::cout << "\tother time      = " << std::setprecision(6) << time.getTime()-all_t <<  ", " << std::setprecision(2) << (time.getTime()-all_t)/time.getTime()*100 << "%" << std::endl;
    }else{
      double all_t = bs->dot_proc_time + bs->MV_proc_time;
      std::cout << "\tdot proc time   = " << std::setprecision(6) << bs->dot_proc_time << ", " << std::setprecision(2) << bs->dot_proc_time/time.getTime()*100 << "%" << std::endl;
      std::cout << "\tMV proc time    = " << std::setprecision(6) << bs->MV_proc_time <<  ", " << std::setprecision(2) << bs->MV_proc_time/time.getTime()*100 << "%" << std::endl;
      std::cout << "\tother time      = " << std::setprecision(6) << time.getTime()-all_t <<  ", " << std::setprecision(2) << (time.getTime()-all_t)/time.getTime()*100 << "%" << std::endl;
    }
    for(long int i=0; i<N; i++){
      f_x << i << " " << std::setprecision(12) << std::uppercase << xvec[i] << std::endl;
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
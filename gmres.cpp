#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include "color.hpp"
#include "gmres.hpp"

gmres::gmres(collection *coll, double *bvec, double *xvec, bool inner, cuda *cu, blas *bs){
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
  over_flag = 0;
  isVP = this->coll->isVP;
  isVerbose = this->coll->isVerbose;
  isCUDA = this->coll->isCUDA;

  if(isVP && isInner ){
    maxloop = this->coll->innerMaxLoop;
    eps = this->coll->innerEps;
    restart = this->coll->innerRestart;
  }else{
    maxloop = this->coll->outerMaxLoop;
    eps = this->coll->outerEps;
    restart = this->coll->outerRestart;
  }

  N = this->coll->N;
  if(isCUDA){
    rvec = cu->d_MallocHost(N);
    axvec = cu->d_MallocHost(N);
    evec = cu->d_MallocHost(restart);
    vvec = cu->d_MallocHost(N);
    vmtx = cu->d_MallocHost(N*(restart+1));
    hmtx = cu->d_MallocHost(N*(restart+1));
    yvec = cu->d_MallocHost(restart);
    wvec = cu->d_MallocHost(N);
    avvec = cu->d_MallocHost(N);
    hvvec = cu->d_MallocHost(restart*(restart+1));
    cvec = cu->d_MallocHost(restart);
    svec = cu->d_MallocHost(restart);
    x0vec = cu->d_MallocHost(N);
    tmpvec = cu->d_MallocHost(N);
    x_0 = cu->d_MallocHost(N);
  }else{
    rvec = new double [N];
    axvec = new double [N];
    evec = new double [restart];
    vvec = new double [N];
    vmtx = new double [N*(restart+1)];
    hmtx = new double [N*(restart+1)];
    yvec = new double [restart];
    wvec = new double [N];
    avvec = new double [N];
    hvvec = new double [restart*(restart+1)];
    cvec = new double [restart];
    svec = new double [restart];
    x0vec = new double [N];
    tmpvec = new double [N];
    x_0 = new double [N];
  }

  this->xvec = xvec;
  this->bvec = bvec;

  std::memset(rvec, 0, sizeof(double)*N);
  std::memset(axvec, 0, sizeof(double)*N);
  std::memset(evec, 0, sizeof(double)*restart);
  std::memset(vvec, 0, sizeof(double)*N);
  std::memset(vmtx, 0, sizeof(double)*(N*restart+1));
  std::memset(hmtx, 0, sizeof(double)*(N*restart+1));
  std::memset(yvec, 0, sizeof(double)*restart);
  std::memset(wvec, 0, sizeof(double)*N);
  std::memset(avvec, 0, sizeof(double)*N);
  std::memset(hvvec, 0, sizeof(double)*(restart*(restart+1)));
  std::memset(cvec, 0, sizeof(double)*restart);
  std::memset(svec, 0, sizeof(double)*restart);
  std::memset(x0vec, 0, sizeof(double)*N);
  std::memset(tmpvec, 0, sizeof(double)*N);
  std::memset(xvec, 0, sizeof(double)*N);


  if(!isInner){
    f_his.open("./output/GMRES_his.txt");
    if(!f_his.is_open()){
      std::cerr << "File open error" << std::endl;
      exit(-1);
    }

    f_x.open("./output/GMRES_xvec.txt");
    if(!f_x.is_open()){
      std::cerr << "File open error" << std::endl;
      exit(-1);
    }
  }

}

gmres::~gmres(){
  if(isCUDA){
    cu->FreeHost(rvec);
    cu->FreeHost(axvec);
    cu->FreeHost(evec);
    cu->FreeHost(vvec);
    cu->FreeHost(vmtx);
    cu->FreeHost(hmtx);
    cu->FreeHost(yvec);
    cu->FreeHost(wvec);
    cu->FreeHost(avvec);
    cu->FreeHost(hvvec);
    cu->FreeHost(cvec);
    cu->FreeHost(svec);
    cu->FreeHost(x0vec);
    cu->FreeHost(tmpvec);
    cu->FreeHost(x_0);
  }else{
    delete[] rvec;
    delete[] axvec;
    delete[] evec;
    delete[] vvec;
    delete[] vmtx;
    delete[] hmtx;
    delete[] yvec;
    delete[] wvec;
    delete[] avvec;
    delete[] hvvec;
    delete[] cvec;
    delete[] svec;
    delete[] x0vec;
    delete[] tmpvec;
    delete[] x_0;
  }

  if(!isInner){
    delete this->bs;
    delete cu;
  }
  f_his.close();
  f_x.close();
}

int gmres::solve(){

  time.start();

  //b 2norm
  bnorm = bs->norm_2(bvec);

  //x_0 = x
  bs->Vec_copy(xvec, x_0);

  for(count=0; count<maxloop;)
  {
    //Ax0
    if(isCUDA){
      cu->MtxVec_mult(xvec, axvec, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
    }else{
      bs->MtxVec_mult(xvec, axvec);
    }

    //r0=b-Ax0
    bs->Vec_sub(bvec, axvec, rvec);

    //2norm rvec
    tmp = bs->norm_2(rvec);

    //
    bs->Scalar_x_div_a(rvec, tmp, vvec);

    bs->Vec_copy(vvec, vmtx, 0, N);

    std::memset(evec, 0, sizeof(double)*restart);

    evec[0] = tmp;

    for(int k=0; k<restart-1; k++){
      error = fabs(evec[k]) / bnorm;
      if(!isInner){
        if(isVerbose){
          std::cout << count+1 << " " << std::scientific << std::setprecision(12) << std::uppercase << error << std::endl;
        }
        f_his << count+1 << " " << std::scientific << std::setprecision(12) << std::uppercase << error << std::endl;
      }

      if(count+1 >= maxloop){
        bs->Hye(hmtx, yvec, evec, k);

        std::memset(tmpvec, 0, sizeof(double)*N);

        for(int i=0; i<k; i++)
        {
          tmp = yvec[i];
          // for(long int j=0; j<N; j++){
          //   tmpvec[j] += tmp * vmtx[i*N+j];
          // }
          bs->Scalar_ax(tmp, vmtx, i, N, tmpvec);
        }

        bs->Vec_add(x0vec, tmpvec, xvec);

        over_flag = 1;
        break;
      }

      if(error <= eps){
        bs->Hye(hmtx, yvec, evec, k);

        std::memset(tmpvec, 0, sizeof(double)*N);

        for(int i=0; i<k; i++)
        {
          tmp = yvec[i];
          // for(long int j=0; j<N; j++){
          //   tmpvec[j] += tmp * vmtx[i*N+j];
          // }
          bs->Scalar_ax(tmp, vmtx, i, N, tmpvec);
        }

        bs->Vec_add(x0vec, tmpvec, xvec);

        exit_flag = 0;
        break;
      }

      //Av & w
      if(isCUDA){
        cu->MtxVec_mult(vmtx, k, N, avvec, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
      }else{
        bs->MtxVec_mult(vmtx, k, N, avvec);
      }

      bs->Vec_copy(avvec, wvec);

      //h_i_k & w update
      for(int i=0; i<=k; i++){
        if(isCUDA){
          // wv_ip = cu->dot(wvec, vmtx, i, N);
          wv_ip = bs->dot(wvec, vmtx, i, N);
        }else{
          wv_ip = bs->dot(wvec, vmtx, i, N);
        }
        hmtx[i*N+k] = wv_ip;
      }

      bs->Gmres_sp_1(k, hmtx, vmtx, wvec);

      //h_k+1 update
      tmp = bs->norm_2(wvec);
      hmtx[(k+1)*N+k] = tmp;

      //v update
      bs->Scalar_x_div_a(wvec, tmp, vvec);
      bs->Vec_copy(vvec, vmtx, k+1, N);

      //h update
      for(int i=0; i<=(k-1); i++){
        tmp = hmtx[i*N+k];
        tmp2 = hmtx[(i+1)*N+k];
        hmtx[i*N+k] = cvec[i] * tmp - svec[i] * tmp2;
        hmtx[(i+1)*N+k] = svec[i] * tmp + cvec[i] * tmp2;
      }

      //alpha = root(h_kk * h_kk + h_k+1_k * h_k+1_k)
      alpha = sqrt(hmtx[k*N+k] * hmtx[k*N+k] + hmtx[(k+1)*N+k] * hmtx[(k+1)*N+k]);

      cvec[k] = hmtx[k*N+k] / alpha;
      svec[k] = -hmtx[(k+1)*N+k] / alpha;
      evec[k+1] = svec[k] * evec[k];
      evec[k] = cvec[k] * evec[k];
      hmtx[k*N+k] = cvec[k] * hmtx[k*N+k] - svec[k] * hmtx[(k+1)*N+k];
      hmtx[(k+1)*N+k] = 0.0;

      count++;

    }

    if(exit_flag==0 || over_flag==1){
      break;
    }

    bs->Hye(hmtx, yvec, evec, restart-1);

    std::memset(tmpvec, 0, sizeof(double)*N);

    for(int i=0; i<restart; i++){
      for(long int j=0; j<N; j++){
        tmpvec[j] += yvec[i] * vmtx[i*N+j];
      }
    }

    bs->Vec_add(x0vec, tmpvec, xvec);

    bs->Vec_copy(xvec, x0vec);

  }
  time.end();

  if(!isInner){
    test_error = bs->Check_error(xvec, x_0);
    std::cout << "|b-ax|2/|b|2 = " << std::fixed << std::setprecision(1) << test_error << std::endl;
    std::cout << "loop = " << count+1 << std::endl;
    std::cout << "time = " << std::setprecision(6) << time.getTime() << std::endl;


    for(long int i=0; i<N; i++){
      f_x << i << " " << std::scientific << std::setprecision(12) << std::uppercase << xvec[i] << std::endl;
    }
  }else{
    if(isVerbose){
      if(exit_flag==0){
        std::cout << GREEN << "\t" <<  count+1 << " = " << std::scientific << std::setprecision(12) << std::uppercase << error << RESET << std::endl;
      }else if(exit_flag==2){
        std::cout << RED << "\t" << count+1 << " = " << std::scientific << std::setprecision(12) << std::uppercase << error << RESET << std::endl;
      }else{
        std::cout << RED << " ERROR " << loop << RESET << std::endl;
      }
    }
  }

  return exit_flag;
}

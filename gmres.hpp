#ifndef GMRES_HPP_INCLUDED__
#define GMRES_HPP_INCLUDED__

#include <iomanip>
#include <fstream>
#include "solver_collection.hpp"
#include "blas.hpp"
#include "times.hpp"

template <typename T>
class gmres {
  private:
    collection<T> *coll;
    blas<T> *bs;
    times time;
    
    long int loop;
    T *xvec, *bvec;
    T *rvec, *axvec, *evec, *vvec, *vmtx, *hmtx, *yvec, *wvec, *avvec, *hvvec, *cvec, *svec, *x0vec, *tmpvec, *x_0;
    T wv_ip, error;
    T alpha, bnorm;
    T tmp, tmp2, rr2;
    long int count;

    int maxloop;
    double eps;
    bool isVP, isVerbose, isCUDA, isInner;
    int restart;

    int exit_flag, over_flag;
    T test_error;

    int N;

    std::ofstream f_his;
    std::ofstream f_x;

  public:
    gmres(collection<T> *coll, T *bvec, T *xvec, bool inner);
    ~gmres();
    int solve();
};

template <typename T>
gmres<T>::gmres(collection<T> *coll, T *bvec, T *xvec, bool inner){
  this->coll = coll;
  bs = new blas<T>(this->coll);

  exit_flag = 2;
  over_flag = 0;
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

  N = this->coll->N;
  rvec = new T [N];
  axvec = new T [N];
  evec = new T [restart];
  vvec = new T [N];
  vmtx = new T [N*(restart+1)];
  hmtx = new T [N*(restart+1)];
  yvec = new T [restart];
  wvec = new T [N];
  avvec = new T [N];
  hvvec = new T [restart*(restart+1)];
  cvec = new T [restart];
  svec = new T [restart];
  x0vec = new T [N];
  tmpvec = new T [N];
  x_0 = new T [N];

  this->xvec = xvec;
  this->bvec = bvec;

  std::memset(rvec, 0, sizeof(T)*N);
  std::memset(axvec, 0, sizeof(T)*N);
  std::memset(evec, 0, sizeof(T)*restart);
  std::memset(vvec, 0, sizeof(T)*N);
  std::memset(vmtx, 0, sizeof(T)*(N*restart+1));
  std::memset(hmtx, 0, sizeof(T)*(N*restart+1));
  std::memset(yvec, 0, sizeof(T)*restart);
  std::memset(wvec, 0, sizeof(T)*N);
  std::memset(avvec, 0, sizeof(T)*N);
  std::memset(hvvec, 0, sizeof(T)*(restart*(restart+1)));
  std::memset(cvec, 0, sizeof(T)*restart);
  std::memset(svec, 0, sizeof(T)*restart);
  std::memset(x0vec, 0, sizeof(T)*N);
  std::memset(tmpvec, 0, sizeof(T)*N);
  std::memset(xvec, 0, sizeof(T)*N);


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

template <typename T>
gmres<T>::~gmres(){
  delete this->bs;
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
  f_his.close();
  f_x.close();
}

template <typename T>
int gmres<T>::solve(){

  time.start();

  //b 2norm
  bnorm = bs->norm_2(bvec);

  //x_0 = x
  bs->Vec_copy(xvec, x_0);

  for(count=0; count<maxloop;)
  {
    //Ax0
    if(isCUDA){

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

    std::memset(evec, 0, sizeof(T)*restart);

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

        std::memset(tmpvec, 0, sizeof(T)*N);

        for(int i=0; i<k; i++)
        {
          for(long int j=0; j<N; j++){
            tmpvec[j] += yvec[i] * vmtx[i*N+j];
          }
        }

        bs->Vec_add(x0vec, tmpvec, xvec);

        over_flag = 1;
        break;
      }

      if(error <= eps){
        bs->Hye(hmtx, yvec, evec, k);

        std::memset(tmpvec, 0, sizeof(T)*N);

        for(int i=0; i<k; i++)
        {
          for(long int j=0; j<N; j++){
            tmpvec[j] += yvec[i] * vmtx[i*N+j];
          }
        }

        bs->Vec_add(x0vec, tmpvec, xvec);

        exit_flag = 0;
        break;
      }

      //Av & w
      for(long int i=0; i<N; i++){
        tmp = 0.0;
        for(int j=this->coll->ptr[i]; j<this->coll->ptr[i+1]; j++){
          tmp += this->coll->val[j] * vmtx[k*N+(this->coll->col[j])];
        }
        avvec[i] = tmp;
        wvec[i] = avvec[i];
      }

      //h_i_k & w update
      for(int i=0; i<=k; i++){
        wv_ip = 0.0;
        for(long int j=0; j<N; j++){
          wv_ip+=wvec[j] * vmtx[i*N+j];
        }
        hmtx[i*N+k] = wv_ip;
      }

      for(int i=0; i<=k; i++){
        for(long int j=0; j<N; j++){
          wvec[j] -= hmtx[i*N+k] * vmtx[i*N+j];
        }
      }

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

    std::memset(tmpvec, 0, sizeof(T)*N);

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
#endif //GMRES_HPP_INCLUDED__


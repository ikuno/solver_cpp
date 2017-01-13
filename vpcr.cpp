#include <iomanip>
#include <cstring>
#include <cstdlib>
#include "color.hpp"
#include "vpcr.hpp"

vpcr::vpcr(collection *coll, double *bvec, double *xvec, bool inner){
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

  N = this->coll->N;
  if(isCUDA){
    if(isPinned){
      rvec = cu->d_MallocHost(N);
      pvec = new double [N];
      zvec = cu->d_MallocHost(N);
      Av = cu->d_MallocHost(N);
      Ap = cu->d_MallocHost(N);
      x_0 = new double [N];
    }else{
      rvec = new double [N];
      pvec = new double [N];
      zvec = new double [N];
      Av = new double [N];
      Ap = new double [N];
      x_0 = new double [N];
    }
  }else{
    rvec = new double [N];
    pvec = new double [N];
    zvec = new double [N];
    Av = new double [N];
    Ap = new double [N];
    x_0 = new double [N];
  }

  if(coll->innerSolver == CG){
    list = new double*[4];

    if(isCUDA){
      if(isPinned){
        cg_rvec = new double[N];
        cg_pvec = cu->d_MallocHost(N);
        cg_mv = cu->d_MallocHost(N);
        cg_x_0 = new double[N];
      }else{
        cg_rvec = new double[N];
        cg_pvec = new double[N];
        cg_mv = new double[N];
        cg_x_0 = new double[N];
      }
    }else{
      cg_rvec = new double[N];
      cg_pvec = new double[N];
      cg_mv = new double[N];
      cg_x_0 = new double[N];
    }

    this->list[0] = cg_rvec;
    this->list[1] = cg_pvec;
    this->list[2] = cg_mv;
    this->list[3] = cg_x_0;
  }else if(coll->innerSolver == CR){
    list = new double*[5];
    if(isCUDA){
      if(isPinned){
        cr_rvec = cu->d_MallocHost(N);
        cr_pvec = cu->d_MallocHost(N);
        cr_qvec = cu->d_MallocHost(N);
        cr_svec = cu->d_MallocHost(N);
        cr_x_0 = new double [N];
      }else{
        cr_rvec = new double [N];
        cr_pvec = new double [N];
        cr_qvec = new double [N];
        cr_svec = new double [N];
        cr_x_0 = new double [N];
      }
    }else{
      cr_rvec = new double [N];
      cr_pvec = new double [N];
      cr_qvec = new double [N];
      cr_svec = new double [N];
      cr_x_0 = new double [N];
    }
    this->list[0] = cr_rvec;
    this->list[1] = cr_pvec;
    this->list[2] = cr_qvec;
    this->list[3] = cr_svec;
    this->list[4] = cr_x_0;
  }else if(coll->innerSolver == GCR){
    list = new double*[7];
    if(isCUDA){
      if(isPinned){
        gcr_rvec = cu->d_MallocHost(N);
        gcr_Av = cu->d_MallocHost(N);
        gcr_x_0 = new double [N];
        gcr_qq = new double [coll->innerRestart];
        gcr_qvec = cu->d_MallocHost(coll->innerRestart * N);
        gcr_pvec = cu->d_MallocHost(coll->innerRestart * N);
        gcr_beta_vec = cu->d_MallocHost(coll->innerRestart);
      }else{
        gcr_rvec = new double [N];
        gcr_Av = new double [N];
        gcr_x_0 = new double [N];
        gcr_qq = new double [coll->innerRestart];
        gcr_qvec = new double [coll->innerRestart * N];
        gcr_pvec = new double [coll->innerRestart * N];
        gcr_beta_vec = new double [coll->innerRestart];
      }
    }else{
      gcr_rvec = new double [N];
      gcr_Av = new double [N];
      gcr_x_0 = new double [N];
      gcr_qq = new double [coll->innerRestart];
      gcr_qvec = new double [coll->innerRestart * N];
      gcr_pvec = new double [coll->innerRestart * N];
      gcr_beta_vec = new double [coll->innerRestart];
    }
    this->list[0] = gcr_rvec;
    this->list[1] = gcr_Av;
    this->list[2] = gcr_x_0;
    this->list[3] = gcr_qq;
    this->list[4] = gcr_qvec;
    this->list[5] = gcr_pvec;
    this->list[6] = gcr_beta_vec;
  }else if(coll->innerSolver == BICG){
    list = new double*[6];
    if(isCUDA){
      if(isPinned){
        bicg_rvec = new double [N];
        bicg_pvec = cu->d_MallocHost(N);
        bicg_r_vec = new double [N];
        bicg_p_vec = cu->d_MallocHost(N);
        bicg_mv = cu->d_MallocHost(N);
        bicg_x_0 = new double [N];
      }else{
        bicg_rvec = new double [N];
        bicg_pvec = new double [N];
        bicg_r_vec = new double [N];
        bicg_p_vec = new double [N];
        bicg_mv = new double [N];
        bicg_x_0 = new double [N];
      }
    }else{
      bicg_rvec = new double [N];
      bicg_pvec = new double [N];
      bicg_r_vec = new double [N];
      bicg_p_vec = new double [N];
      bicg_mv = new double [N];
      bicg_x_0 = new double [N];
    }
    this->list[0] = bicg_rvec;
    this->list[1] = bicg_pvec;
    this->list[2] = bicg_r_vec;
    this->list[3] = bicg_p_vec;
    this->list[4] = bicg_mv;
    this->list[5] = bicg_x_0;
  }else if(coll->innerSolver == GMRES){
    list = new double*[15];
    if(isCUDA){
      if(isPinned){
        gm_Av = cu->d_MallocHost(N);
        gm_rvec = new double [N];
        gm_evec = new double [coll->innerRestart];
        gm_vvec = new double [N];
        gm_vmtx = new double [N*(coll->innerRestart+1)];
        gm_hmtx = new double [N*(coll->innerRestart+1)];
        gm_yvec = new double [coll->innerRestart];
        gm_wvec = cu->d_MallocHost(N);
        gm_cvec = new double [coll->innerRestart];
        gm_svec = new double [coll->innerRestart];
        gm_x0vec = new double [N];
        gm_tmpvec = new double [N];
        gm_x_0 = new double [N];
        gm_testvec = new double [N];
        gm_testvec2 = cu->d_MallocHost(N);
      }else{
        gm_Av = new double [N];
        gm_rvec = new double [N];
        gm_evec = new double [coll->innerRestart];
        gm_vvec = new double [N];
        gm_vmtx = new double [N*(coll->innerRestart+1)];
        gm_hmtx = new double [N*(coll->innerRestart+1)];
        gm_yvec = new double [coll->innerRestart];
        gm_wvec = new double [N];
        gm_cvec = new double [coll->innerRestart];
        gm_svec = new double [coll->innerRestart];
        gm_x0vec = new double [N];
        gm_tmpvec = new double [N];
        gm_x_0 = new double [N];
        gm_testvec = new double [N];
        gm_testvec2 = new double [N];
      }
    }else{
      gm_Av = new double [N];
      gm_rvec = new double [N];
      gm_evec = new double [coll->innerRestart];
      gm_vvec = new double [N];
      gm_vmtx = new double [N*(coll->innerRestart+1)];
      gm_hmtx = new double [N*(coll->innerRestart+1)];
      gm_yvec = new double [coll->innerRestart];
      gm_wvec = new double [N];
      gm_cvec = new double [coll->innerRestart];
      gm_svec = new double [coll->innerRestart];
      gm_x0vec = new double [N];
      gm_tmpvec = new double [N];
      gm_x_0 = new double [N];
      gm_testvec = new double [N];
      gm_testvec2 = new double [N];
    }
    this->list[0] = gm_Av;
    this->list[1] = gm_rvec;
    this->list[2] = gm_evec;
    this->list[3] = gm_vvec;
    this->list[4] = gm_vmtx;
    this->list[5] = gm_hmtx;
    this->list[6] = gm_yvec;
    this->list[7] = gm_wvec;
    this->list[8] = gm_cvec;
    this->list[9] = gm_svec;
    this->list[10] = gm_x0vec;
    this->list[11] = gm_tmpvec;
    this->list[12] = gm_x_0;
    this->list[13] = gm_testvec;
    this->list[14] = gm_testvec2;
  }else if(coll->innerSolver == KSKIPCG){
    list = new double*[9];
    if(isCUDA){
      if(isPinned){
        kcg_rvec = cu->d_MallocHost(N);
        kcg_pvec = cu->d_MallocHost(N);
        kcg_Av = cu->d_MallocHost(N);
        kcg_x_0 = new double [N];
        kcg_delta = new double [2*coll->innerKskip];
        kcg_eta = new double [2*coll->innerKskip+1];
        kcg_zeta = new double [2*coll->innerKskip+2];
        kcg_Ar = cu->d_MallocHost((2*coll->innerKskip+1)*N);
        kcg_Ap = cu->d_MallocHost((2*coll->innerKskip+2)*N);
      }else{
        kcg_rvec = new double [N];
        kcg_pvec = new double [N];
        kcg_Av = new double [N];
        kcg_x_0 = new double [N];
        kcg_delta = new double [2*coll->innerKskip];
        kcg_eta = new double [2*coll->innerKskip+1];
        kcg_zeta = new double [2*coll->innerKskip+2];
        kcg_Ar = new double [(2*coll->innerKskip+1)*N];
        kcg_Ap = new double [(2*coll->innerKskip+2)*N];
      }
    }else{
        kcg_rvec = new double [N];
        kcg_pvec = new double [N];
        kcg_Av = new double [N];
        kcg_x_0 = new double [N];
        kcg_delta = new double [2*coll->innerKskip];
        kcg_eta = new double [2*coll->innerKskip+1];
        kcg_zeta = new double [2*coll->innerKskip+2];
        kcg_Ar = new double [(2*coll->innerKskip+1)*N];
        kcg_Ap = new double [(2*coll->innerKskip+2)*N];
    }
    this->list[0] = kcg_rvec;
    this->list[1] = kcg_pvec;
    this->list[2] = kcg_Av;
    this->list[3] = kcg_x_0;
    this->list[4] = kcg_delta;
    this->list[5] = kcg_eta;
    this->list[6] = kcg_zeta;
    this->list[7] = kcg_Ar;
    this->list[8] = kcg_Ap;
  }else if(coll->innerSolver == KSKIPBICG){
    list = new double*[12];
    if(isCUDA){
      if(isPinned){
        kbicg_rvec = cu->d_MallocHost(N);
        kbicg_pvec = cu->d_MallocHost(N);
        kbicg_r_vec = cu->d_MallocHost(N);
        kbicg_p_vec = cu->d_MallocHost(N);
        kbicg_Av = cu->d_MallocHost(N);
        kbicg_x_0 = new double [N];
        kbicg_theta = new double [2*coll->innerKskip];
        kbicg_eta = new double [2*coll->innerKskip+1];
        kbicg_rho = new double [2*coll->innerKskip+1];
        kbicg_phi = new double [2*coll->innerKskip+2];
        kbicg_Ar = cu->d_MallocHost((2*coll->innerKskip+1)*N);
        kbicg_Ap = cu->d_MallocHost((2*coll->innerKskip+2)*N);
      }else{
        kbicg_rvec = new double [N];
        kbicg_pvec = new double [N];
        kbicg_r_vec = new double [N];
        kbicg_p_vec = new double [N];
        kbicg_Av = new double [N];
        kbicg_x_0 = new double [N];
        kbicg_theta = new double [2*coll->innerKskip];
        kbicg_eta = new double [2*coll->innerKskip+1];
        kbicg_rho = new double [2*coll->innerKskip+1];
        kbicg_phi = new double [2*coll->innerKskip+2];
        kbicg_Ar = new double [(2*coll->innerKskip+1)*N];
        kbicg_Ap = new double [(2*coll->innerKskip+2)*N];
      }
    }else{
      kbicg_rvec = new double [N];
      kbicg_pvec = new double [N];
      kbicg_r_vec = new double [N];
      kbicg_p_vec = new double [N];
      kbicg_Av = new double [N];
      kbicg_x_0 = new double [N];
      kbicg_theta = new double [2*coll->innerKskip];
      kbicg_eta = new double [2*coll->innerKskip+1];
      kbicg_rho = new double [2*coll->innerKskip+1];
      kbicg_phi = new double [2*coll->innerKskip+2];
      kbicg_Ar = new double [(2*coll->innerKskip+1)*N];
      kbicg_Ap = new double [(2*coll->innerKskip+2)*N];
    }
    this->list[0] = kbicg_rvec;
    this->list[1] = kbicg_pvec;
    this->list[2] = kbicg_r_vec;
    this->list[3] = kbicg_p_vec;
    this->list[4] = kbicg_Av;
    this->list[5] = kbicg_x_0;
    this->list[6] = kbicg_theta;
    this->list[7] = kbicg_eta;
    this->list[8] = kbicg_rho;
    this->list[9] = kbicg_phi;
    this->list[10] = kbicg_Ar;
    this->list[11] = kbicg_Ap;
  }else{
    std::cout << "ERROR InnerSolver" << std::endl;
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
  std::memset(zvec, 0, sizeof(double)*N);
  std::memset(Av, 0, sizeof(double)*N);
  std::memset(Ap, 0, sizeof(double)*N);
  std::memset(xvec, 0, sizeof(double)*N);

  if(!isInner){
    f_his.open("./output/VPCR_his.txt");
    if(!f_his.is_open()){
      std::cerr << "File open error" << std::endl;
      exit(-1);
    }

    f_x.open("./output/VPCR_xvec.txt");
    if(!f_x.is_open()){
      std::cerr << "File open error" << std::endl;
      exit(-1);
    }
  }

  if(isVP){
    std::string name = this->coll->enum2string(this->coll->innerSolver);
    name = "./output/" + name + "_inner.txt";
    std::ifstream in(name.c_str());
    if(in.good()){
      std::cout << "Delete inner solver's loop file" << std::endl;
      std::remove(name.c_str());
    }else{
      std::cout << "Has no inner solver's loop file yet" << std::endl;
    }
  }


}

vpcr::~vpcr(){
  if(isCUDA){
    if(isPinned){
      cu->FreeHost(rvec);
      delete[] pvec;
      cu->FreeHost(zvec);
      cu->FreeHost(Av);
      cu->FreeHost(Ap);
      delete[] x_0;
    }else{
      delete[] rvec;
      delete[] pvec;
      delete[] zvec;
      delete[] Av;
      delete[] Ap;
      delete[] x_0;
    }
  }else{
    delete[] rvec;
    delete[] pvec;
    delete[] zvec;
    delete[] Av;
    delete[] Ap;
    delete[] x_0;
  }

  if(coll->innerSolver == CG){
    if(isCUDA){
      if(isPinned){
        delete[] cg_rvec;
        cu->FreeHost(cg_pvec);
        cu->FreeHost(cg_mv);
        delete[] cg_x_0;
      }else{
        delete[] cg_rvec;
        delete[] cg_pvec;
        delete[] cg_mv;
        delete[] cg_x_0;
      }
    }else{
        delete[] cg_rvec;
        delete[] cg_pvec;
        delete[] cg_mv;
        delete[] cg_x_0;
    }
    delete[] list;
  }else if(coll->innerSolver == CR){
    if(isCUDA){
      if(isPinned){
        cu->FreeHost(cr_pvec);
        cu->FreeHost(cr_qvec);
        cu->FreeHost(cr_svec);
        delete[] cr_x_0;
      }else{
        delete[] cr_pvec;
        delete[] cr_qvec;
        delete[] cr_svec;
        delete[] cr_x_0;
      }
    }else{
      delete[] cr_pvec;
      delete[] cr_qvec;
      delete[] cr_svec;
      delete[] cr_x_0;  
    }
    delete[] list;
  }else if(coll->innerSolver == GCR){
    if(isCUDA){
      if(isPinned){
        cu->FreeHost(gcr_rvec);
        cu->FreeHost(gcr_Av);
        delete[] gcr_x_0;
        delete[] gcr_qq;
        cu->FreeHost(gcr_qvec);
        cu->FreeHost(gcr_pvec);
        cu->FreeHost(gcr_beta_vec);
      }else{
        delete[] gcr_rvec;
        delete[] gcr_Av;
        delete[] gcr_x_0;
        delete[] gcr_qq;
        delete[] gcr_qvec;
        delete[] gcr_pvec;
        delete[] gcr_beta_vec;
      }
    }else{
      delete[] gcr_rvec;
      delete[] gcr_Av;
      delete[] gcr_x_0;
      delete[] gcr_qq;
      delete[] gcr_qvec;
      delete[] gcr_pvec;
      delete[] gcr_beta_vec;
    }
    delete[] list;
  }else if(coll->innerSolver == BICG){
    if(isCUDA){
      if(isPinned){
        delete[] bicg_rvec;
        cu->FreeHost(bicg_pvec);
        delete[] bicg_r_vec;
        cu->FreeHost(bicg_p_vec);
        cu->FreeHost(bicg_mv);
        delete[] bicg_x_0;
      }else{
        delete[] bicg_rvec;
        delete[] bicg_pvec;
        delete[] bicg_r_vec;
        delete[] bicg_p_vec;
        delete[] bicg_mv;
        delete[] bicg_x_0;
      }
    }else{
      delete[] bicg_rvec;
      delete[] bicg_pvec;
      delete[] bicg_r_vec;
      delete[] bicg_p_vec;
      delete[] bicg_mv;
      delete[] bicg_x_0;
    }
    delete[] list;
  }else if(coll->innerSolver == GMRES){
    if(isCUDA){
      if(isPinned){
        cu->FreeHost(gm_Av);
        delete[] gm_rvec;
        delete[] gm_evec;
        delete[] gm_vvec;
        delete[] gm_vmtx;
        delete[] gm_hmtx;
        delete[] gm_yvec;
        cu->FreeHost(gm_wvec);
        delete[] gm_cvec;
        delete[] gm_svec;
        delete[] gm_x0vec;
        delete[] gm_tmpvec;
        delete[] gm_x_0;
        delete[] gm_testvec;
        cu->FreeHost(gm_testvec2);
      }else{
        delete[] gm_Av;
        delete[] gm_rvec;
        delete[] gm_evec;
        delete[] gm_vvec;
        delete[] gm_vmtx;
        delete[] gm_hmtx;
        delete[] gm_yvec;
        delete[] gm_wvec;
        delete[] gm_cvec;
        delete[] gm_svec;
        delete[] gm_x0vec;
        delete[] gm_tmpvec;
        delete[] gm_x_0;
        delete[] gm_testvec;
        delete[] gm_testvec2;
      }
    }else{
      delete[] gm_Av;
      delete[] gm_rvec;
      delete[] gm_evec;
      delete[] gm_vvec;
      delete[] gm_vmtx;
      delete[] gm_hmtx;
      delete[] gm_yvec;
      delete[] gm_wvec;
      delete[] gm_cvec;
      delete[] gm_svec;
      delete[] gm_x0vec;
      delete[] gm_tmpvec;
      delete[] gm_x_0;
      delete[] gm_testvec;
      delete[] gm_testvec2;
    }
    delete[] list;
  }else if(coll->innerSolver == KSKIPCG){
    if(isCUDA){
      if(isPinned){
        cu->FreeHost(kcg_rvec);
        cu->FreeHost(kcg_pvec);
        cu->FreeHost(kcg_Av);
        delete[] kcg_x_0;
        delete[] kcg_delta;
        delete[] kcg_eta;
        delete[] kcg_zeta;
        cu->FreeHost(kcg_Ar);
        cu->FreeHost(kcg_Ap);
      }else{
        delete[] kcg_rvec;
        delete[] kcg_pvec;
        delete[] kcg_Av;
        delete[] kcg_x_0;
        delete[] kcg_delta;
        delete[] kcg_eta;
        delete[] kcg_zeta;
        delete[] kcg_Ar;
        delete[] kcg_Ap;
      }
    }else{
      delete[] kcg_rvec;
      delete[] kcg_pvec;
      delete[] kcg_Av;
      delete[] kcg_x_0;
      delete[] kcg_delta;
      delete[] kcg_eta;
      delete[] kcg_zeta;
      delete[] kcg_Ar;
      delete[] kcg_Ap;
    }
    delete[] list;
  }else if(coll->innerSolver == KSKIPBICG){
    if(isCUDA){
      if(isPinned){
        cu->FreeHost(kbicg_rvec);
        cu->FreeHost(kbicg_pvec);
        cu->FreeHost(kbicg_r_vec);
        cu->FreeHost(kbicg_p_vec);
        cu->FreeHost(kbicg_Av);
        delete[] kbicg_x_0;
        delete[] kbicg_theta;
        delete[] kbicg_eta;
        delete[] kbicg_rho;
        delete[] kbicg_phi;
        cu->FreeHost(kbicg_Ar);
        cu->FreeHost(kbicg_Ap);
      }else{
        delete[] kbicg_rvec;
        delete[] kbicg_pvec;
        delete[] kbicg_r_vec;
        delete[] kbicg_p_vec;
        delete[] kbicg_Av;
        delete[] kbicg_x_0;
        delete[] kbicg_theta;
        delete[] kbicg_eta;
        delete[] kbicg_rho;
        delete[] kbicg_phi;
        delete[] kbicg_Ar;
        delete[] kbicg_Ap;
      }
    }else{
      delete[] kbicg_rvec;
      delete[] kbicg_pvec;
      delete[] kbicg_r_vec;
      delete[] kbicg_p_vec;
      delete[] kbicg_Av;
      delete[] kbicg_x_0;
      delete[] kbicg_theta;
      delete[] kbicg_eta;
      delete[] kbicg_rho;
      delete[] kbicg_phi;
      delete[] kbicg_Ar;
      delete[] kbicg_Ap;
    }
    delete[] list;
  }else{
    std::cout << "ERROR InnerSolver" << std::endl;
  }


  delete this->bs;
  delete this->in;
  delete this->cu;

  f_his.close();
  f_x.close();
}

int vpcr::solve(){

  time.start();

  //x_0 = x
  bs->Vec_copy(xvec, x_0);

  //b 2norm
  bnorm = bs->norm_2(bvec);

  //Ax
  if(isCUDA){
    cu->MtxVec_mult(xvec, Av, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
  }else{
    bs->MtxVec_mult(xvec, Av);
  }

  //r = b - Ax
  bs->Vec_sub(bvec, Av, rvec);

  //Az=r
  // in->innerSelect(this->coll, this->coll->innerSolver, cu, bs, rvec, zvec);
  in->innerSelect(this->coll, this->coll->innerSolver, cu, bs, rvec, zvec, this->list);

  //p = z
  bs->Vec_copy(zvec, pvec);

  //Az(Av)
  if(isCUDA){
    cu->MtxVec_mult(zvec, Av, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
  }else{
    bs->MtxVec_mult(zvec, Av);
  }

  //Ap = Az(Av)
  bs->Vec_copy(Av, Ap);

  //(z, Az(Av))
  if(isCUDA){
    // zaz = cu->dot(zvec, Av);
    zaz = bs->dot(zvec, Av);
  }else{
    zaz = bs->dot(zvec, Av);
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

    //alpha = (z,Az) / (Ap,Ap)
    if(isCUDA){
      // tmp = cu->dot(Ap, Ap);
      tmp = bs->dot(Ap, Ap);
    }else{
      tmp = bs->dot(Ap, Ap);
    }
    alpha = zaz / tmp;

    //x = alpha * pvec + x
    bs->Scalar_axy(alpha, pvec, xvec, xvec);

    //r = -alpha * Ap + r
    bs->Scalar_axy(-alpha, Ap, rvec, rvec);

    std::memset(zvec, 0, sizeof(double)*N);

    //Az = r
    // in->innerSelect(this->coll, this->coll->innerSolver, cu, bs, rvec, zvec);
    in->innerSelect(this->coll, this->coll->innerSolver, cu, bs, rvec, zvec, this->list);

    //Az
    if(isCUDA){
      cu->MtxVec_mult(zvec, Av, this->coll->Cval, this->coll->Ccol, this->coll->Cptr);
    }else{
      bs->MtxVec_mult(zvec, Av);
    }

    //(z, Az)
    if(isCUDA){
      // zaz2 = cu->dot(zvec, Av);
      zaz2 = bs->dot(zvec, Av);
    }else{
      zaz2 = bs->dot(zvec, Av);
    }

    beta = zaz2/zaz;

    zaz = zaz2;

    //p = beta * p + r
    bs->Scalar_axy(beta, pvec, zvec, pvec);

    //Ap = beta * Ap + Az
    bs->Scalar_axy(beta, Ap, Av, Ap);
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
  }

  return exit_flag;
}

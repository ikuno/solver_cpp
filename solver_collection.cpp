#include "solver_collection.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <string>
#include <typeinfo>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "cmdline.h"
#include "color.hpp"

collection::collection() {
  time = new times(); 
  cu = new cuda(time);

  isVP = false;
  isCUDA = false;
  isVerbose = false;
  isInnerNow = false;
  isInnerKskip = false;
  isPinned = false;
  OMPThread = 8;
  CUDADevice = cu->GetDeviceNum();

  outerSolver = NONE;
  innerSolver = NONE;

  outerMaxLoop = 10000;
  innerMaxLoop = 50;
  outerEps = 1e-8;
  innerEps = 1e-1;
  outerRestart = 1000;
  innerRestart = 1000;
  outerKskip = 2;
  innerKskip = 2;
  outerFix = 2;
  innerFix = 2;

  L1_Dir_Name = "Matrix";

  CRS_Dir_Name = "CRS";
  CRS_Path = "../"; 

  MM_Dir_Name = "MM";
  MM_Path = "../";

  N = 0;
  NNZ = 0;

  val = NULL;
  col = NULL;
  ptr = NULL;
  bvec = NULL;
  xvec = NULL;

  Tval = NULL;
  Tcol = NULL;
  Tptr = NULL;

  Cval = NULL;
  Ccol = NULL;
  Cptr = NULL;
  
  CTval = NULL;
  CTcol = NULL;
  CTptr = NULL;

  N1 = 0;
  N2 = 0;
  NNZ1 = 0;
  NNZ2 = 0;

  val1 = NULL;
  col1 = NULL;
  ptr1 = NULL;

  val2 = NULL;
  col2 = NULL;
  ptr2 = NULL;

  CTval1 = NULL;
  CTcol1 = NULL;
  CTptr1 = NULL;

  CTval2 = NULL;
  CTcol2 = NULL;
  CTptr2 = NULL;

  if(isMultiGPU){
    cu->Reset(0);
    cu->Reset(1);
  }else{
    cu->Reset(0);
  }

}

collection::~collection() {
  delete[] bvec;
  if(!isPinned){
    delete[] xvec;
  }else{
    cu->FreeHost(xvec);
  }

  if(!isPinned){
    delete[] val;
    delete[] col;
    delete[] ptr;
  }else{
    cu->FreeHost(val);
    cu->FreeHost(col);
    cu->FreeHost(ptr);
  }

  if(this->outerSolver == BICG || this->outerSolver == KSKIPBICG || this->outerSolver == VPBICG || this->innerSolver == BICG || this->innerSolver == KSKIPBICG || this->innerSolver == VPBICG){
    if(!isPinned){
      delete[] Tval;
      delete[] Tcol;
      delete[] Tptr;
    }else{
      cu->FreeHost(Tval);
      cu->FreeHost(Tcol);
      cu->FreeHost(Tptr);
    }
  }

  if(isMultiGPU){
    if(!isPinned){
      delete[] val1;
      delete[] col1;
      delete[] ptr1;
      delete[] val2;
      delete[] col2;
      delete[] ptr2;
    }else{
      cu->FreeHost(val1);
      cu->FreeHost(col1);
      cu->FreeHost(ptr1);
      cu->FreeHost(val2);
      cu->FreeHost(col2);
      cu->FreeHost(ptr2);
    }
    if(this->outerSolver == BICG || this->outerSolver == KSKIPBICG || this->outerSolver == VPBICG || this->innerSolver == BICG || this->innerSolver == KSKIPBICG || this->innerSolver == VPBICG){
      if(!isPinned){
        delete[] Tval1;
        delete[] Tcol1;
        delete[] Tptr1;
        delete[] Tval2;
        delete[] Tcol2;
        delete[] Tptr2;
      }else{
        cu->FreeHost(Tval1);
        cu->FreeHost(Tcol1);
        cu->FreeHost(Tptr1);
        cu->FreeHost(Tval2);
        cu->FreeHost(Tcol2);
        cu->FreeHost(Tptr2);
      }
    }
  }

  if(isCUDA){
    if(!isMultiGPU){
      cu->Free(Cval);
      cu->Free(Ccol);
      cu->Free(Cptr);
      if(this->outerSolver == BICG || this->outerSolver == KSKIPBICG || this->outerSolver == VPBICG || this->innerSolver == BICG || this->innerSolver == KSKIPBICG || this->innerSolver == VPBICG){
        cu->Free(CTval);
        cu->Free(CTcol);
        cu->Free(CTptr);
      }
    }else{
      cu->Free(Cval1);
      cu->Free(Ccol1);
      cu->Free(Cptr1);
      cu->Free(Cval2);
      cu->Free(Ccol2);
      cu->Free(Cptr2);
      if(this->outerSolver == BICG || this->outerSolver == KSKIPBICG || this->outerSolver == VPBICG || this->innerSolver == BICG || this->innerSolver == KSKIPBICG || this->innerSolver == VPBICG){
        cu->Free(CTval1);
        cu->Free(CTcol1);
        cu->Free(CTptr1);
        cu->Free(CTval2);
        cu->Free(CTcol2);
        cu->Free(CTptr2);
      }
    }
  }

  if(isMultiGPU){
    cu->Reset(0);
    cu->Reset(1);
  }else{
    cu->Reset(0);
  }
  delete cu;
  delete time;
}

std::string collection::enum2string(SOLVERS_NAME id){
  if(id == 0){
    return "NONE";
  }else if(id == 1){
    return "CG";
  }else if(id == 2){
    return "BICG";
  }else if(id == 3){
    return "CR";
  }else if(id == 4){
    return "GCR";
  }else if(id == 5){
    return "GMRES";
  }else if(id == 6){
    return "KSKIPCG";
  }else if(id == 7){
    return "KSKIPCR";
  }else if(id == 8){
    return "KSKIPBICG";
  }else if(id == 9){
    return "VPCG";
  }else if(id == 10){
    return "VPBICG";
  }else if(id == 11){
    return "VPCR";
  }else if(id == 12){
    return "VPGCR";
  }else if(id == 13){
    return "VPGMRES";
  }
  return "NONE";
}

SOLVERS_NAME collection::string2enum(std::string str){
  if(str == "cg" || str == "CG"){
    return CG;
  }else if(str == "bicg" || str == "BICG"){
    return BICG;
  }else if(str == "cr" || str == "CR"){
    return CR;
  }else if(str == "gcr" || str == "GCR"){
    return GCR;
  }else if(str == "gmres" || str == "GMRES"){
    return GMRES;
  }else if(str == "kskipcg" || str == "KSKIPCG"){
    return KSKIPCG;
  }else if(str == "kskipcr" || str == "KSKIPCR"){
    return KSKIPCR;
  }else if(str == "kskipbicg" || str == "KSKIPBICG"){
    return KSKIPBICG;
  }else if(str == "vpcg" || str == "VPCG"){
    return VPCG;
  }else if(str == "vpbicg" || str == "VPBICG"){
    return VPBICG;
  }else if(str == "vpcr" || str == "VPCR"){
    return VPCR;
  }else if(str == "vpgcr" || str == "VPGCR"){
    return VPGCR;
  }else if(str == "vpgmres" || str == "VPGMRES"){
    return VPGMRES;
  }
  return NONE;
}

void collection::readCMD(int argc, char* argv[]){
  cmdline::parser cmd;


  //1.long name
  //2.short name
  //3.description
  //4.mandatory
  //5.default value
  cmd.add<std::string>("L1_Dir_Name", 'N', "Default Matrix files Name", false, this->L1_Dir_Name);

  cmd.add<std::string>("CRS_Path", 'p', "CRS format files path", false, this->CRS_Path);
  cmd.add<std::string>("CRS_Dir_Name", 'd', "Default CRS format files Name", false, this->CRS_Dir_Name);
  cmd.add<std::string>("CRS_Matrix_Name", 'm', "CRS Matrix Name", false, "NONE");

  cmd.add<std::string>("MM_Path", 'P', "MatrixMarket files path", false, this->MM_Path);
  cmd.add<std::string>("MM_Dir_Name", 'D', "Default MM format files Name", false, this->MM_Dir_Name);
  cmd.add<std::string>("MM_Matrix_Name", 'M', "MM Matrix Name", false, "NONE");

  cmd.add<int>("Openmp_Thread", 't', "Threads use in OpenMP", false, this->OMPThread);

  cmd.add<std::string>("OuterSolver", 'S', "method use in outersolver", true, "");
  cmd.add<int>("OuterMaxLoop", 'L', "max loops in outersolver", false, this->outerMaxLoop);
  cmd.add<double>("OuterEps", 'E', "eps in outersolver", false, this->outerEps);
  cmd.add<int>("OuterRestart", 'R', "Restart number in outersolver", false, this->outerRestart);
  cmd.add<int>("OuterKskip", 'K', "kskip number in outersolver", false, this->outerKskip);
  cmd.add<int>("OuterFix", 'F', "fix bug in outersolver", false, this->outerFix);
  
  cmd.add<std::string>("InnerSolver", 's', "method use in innersolver", false, "NO");
  cmd.add<int>("InnerMaxLoop", 'l', "max loops in innersolver", false, this->innerMaxLoop);
  cmd.add<double>("InnerEps", 'e', "eps in innersolver", false, this->innerEps);
  cmd.add<int>("InnerRestart", 'r', "Restart number in innersolver", false, this->innerRestart);
  cmd.add<int>("InnerKskip", 'k', "kskip number in innersolver", false, this->innerKskip);
  cmd.add<int>("InnerFix", 'f', "fix bug in innersolver", false, this->innerFix);

  cmd.add("verbose", 'v', "verbose mode will printout all detel ");
  cmd.add("cuda", 'c', "cuda");
  cmd.add("pinned", 'x', "use pinned memory");
  cmd.add("multiGPU", 'g', "use multiGPU in CUDA");

  cmd.parse_check(argc, argv);

  std::cout << "Reading CommandLine Option .........."<< std::flush;

  this->L1_Dir_Name = cmd.get<std::string>("L1_Dir_Name");

  this->CRS_Path = cmd.get<std::string>("CRS_Path");
  this->CRS_Dir_Name = cmd.get<std::string>("CRS_Dir_Name");
  this->CRS_Matrix_Name = cmd.get<std::string>("CRS_Matrix_Name");

  this->MM_Path = cmd.get<std::string>("MM_Path");
  this->MM_Dir_Name = cmd.get<std::string>("MM_Dir_Name");
  this->MM_Matrix_Name = cmd.get<std::string>("MM_Matrix_Name");

  this->OMPThread = cmd.get<int>("Openmp_Thread");

  this->outerSolver = string2enum(cmd.get<std::string>("OuterSolver"));
  this->outerMaxLoop = cmd.get<int>("OuterMaxLoop");
  this->outerEps = cmd.get<double>("OuterEps");
  this->outerRestart = cmd.get<int>("OuterRestart");
  this->outerKskip = cmd.get<int>("OuterKskip");
  this->outerFix = cmd.get<int>("OuterFix");

  this->innerSolver = string2enum(cmd.get<std::string>("InnerSolver"));
  this->innerMaxLoop = cmd.get<int>("InnerMaxLoop");
  this->innerEps = cmd.get<double>("InnerEps");
  this->innerRestart = cmd.get<int>("InnerRestart");
  this->innerKskip = cmd.get<int>("InnerKskip");
  this->innerFix = cmd.get<int>("InnerFix");

  if(this->MM_Matrix_Name == "NONE" && this->CRS_Matrix_Name == "NONE"){
    std::cerr << RED << "[X]Must set one input source" << RESET << std::endl;
    exit(-1);
  }

  if(this->MM_Matrix_Name != "NONE" && this->CRS_Matrix_Name != "NONE"){
    std::cerr << RED << "[X]Only can set one input source" << RESET << std::endl;
    exit(-1);
  }

  if(this->MM_Matrix_Name != "NONE"){
    this->inputSource = 2;
  }else if(this->CRS_Matrix_Name != "NONE"){
    this->inputSource = 1;
  }

  if(this->outerSolver == NONE){
    std::cerr << RED << "[X]OuterSolverName can not found in List" << RESET << std::endl;
    exit(-1);
  }

  if(this->innerSolver != NONE){
    isVP = true;
  }

  if(this->innerSolver == KSKIPCG || this->innerSolver == KSKIPBICG){
    isInnerKskip = true;
  }

  if(cmd.exist("verbose")) this->isVerbose=true;
  if(cmd.exist("cuda")) this->isCUDA=true;
  if(cmd.exist("pinned")) this->isPinned=true;
  if(cmd.exist("multiGPU")) {
    this->isMultiGPU=true;
  }
  if(this->isMultiGPU && this->CUDADevice<=1){
    std::cout << "Number of CUDA Device is less than 2, Can not Enable MultiGPU Mode" << std::endl;
    exit(-1);
  }

  if(this->isMultiGPU){
    cu->EnableP2P();
  }


  this->setOpenmpThread();

  std::cout << GREEN << "[○] Done" << RESET << std::endl;
}

void collection::checkCMD(){
  DIR *dp;
  dirent* entry;

  std::string fullDir;
  std::string getDir;

  bool hasFiles = false;
  bool hasCRSFiles[3]={false, false, false};
  bool hasMMFiles = false;

  std::cout << "Checking Input Matrix Type .........."<< std::flush;

  // CRS input
  if(this->inputSource == 1){

    std::cout << GREEN << "[○] CRS" << RESET << std::endl;
    dp = opendir(this->CRS_Path.c_str());
    if(dp == NULL) exit(1);

    std::cout << "Checking 1L Dir .........."<< std::flush;

    do{
      entry = readdir(dp);
      if(entry!=NULL){
        if(this->L1_Dir_Name == entry->d_name){
          std::cout << GREEN << "[○] " << this->L1_Dir_Name << RESET << std::endl;
          getDir = entry->d_name;
          hasFiles = true;
        }
      }
    }while(entry!=NULL);

    if(!hasFiles){
      std::cerr << RED << "[X]Can't found 1L Dir -> " << this->L1_Dir_Name << RESET << std::endl;
      exit(-1);
    }

    hasFiles = false;

    fullDir += this->CRS_Path;
    fullDir += "/";
    fullDir += getDir;
    dp = opendir(fullDir.c_str());

    std::cout << "Checking CRS Dir .........."<< std::flush;

    do{
      entry = readdir(dp);
      if(entry!=NULL){
        if(this->CRS_Dir_Name == entry->d_name){
          std::cout << GREEN << "[○] " << this->CRS_Dir_Name << RESET << std::endl;
          getDir = entry->d_name;
          hasFiles = true;
        }
      }
    }while(entry!=NULL);

    if(!hasFiles){
      std::cerr << RED << "[X]Error no CRS Dir -> " << this->CRS_Dir_Name << RESET << std::endl;
      exit(-1);
    }

    fullDir += "/";
    fullDir += getDir;

    hasFiles = false;
    dp = opendir(fullDir.c_str());

    std::cout << "Checking CRS files .........."<< std::flush;

    do{
      entry = readdir(dp);
      if(entry!=NULL){
        if(this->CRS_Matrix_Name == entry->d_name){
          std::cout << GREEN << "[○] " << this->CRS_Matrix_Name << RESET << std::endl;
          getDir = entry->d_name;
          hasFiles = true;
        }
      }
    }while(entry!=NULL);

    if(!hasFiles){
      std::cout << RED << "[○]Error no CRS files Dir -> " << this->CRS_Matrix_Name << RESET << std::endl;
      exit(-1);
    }

    fullDir += "/";
    fullDir += getDir;
    this->fullPath = fullDir;

    dp = opendir(fullDir.c_str());

    std::cout << "Checking CRS files name .........."<< std::flush;

    do{
      entry = readdir(dp);
      if(entry!=NULL){
        if(std::string("ColVal.txt") == entry->d_name){
          hasCRSFiles[0]=true;
        }else if(std::string("Ptr.txt") == entry->d_name){
          hasCRSFiles[1]=true;
        }else if(std::string("bx.txt") == entry->d_name){
          hasCRSFiles[2]=true;
        }
      }
    }while(entry!=NULL);

    if(!hasCRSFiles[0] || !hasCRSFiles[1] || !hasCRSFiles[2]){
      std::cerr << RED << "[X] Error in CRS files " << RESET << std::endl;
      exit(-1);
    }else{
      std::cout << GREEN << "[○] Get CRS files " << RESET << std::endl;
    }

    // MM input
  }else if(this->inputSource == 2){

    std::cout << "\tLoading MM path .........."<< std::flush;

    dp = opendir(this->MM_Path.c_str());
    if(dp == NULL) exit(1);

    do{
      entry = readdir(dp);
      if(entry!=NULL){
        if(this->L1_Dir_Name == entry->d_name){
          std::cout << GREEN << "[○] " << this->L1_Dir_Name << RESET << std::endl;
          getDir = entry->d_name;
          hasFiles = true;
        }
      }
    }while(entry!=NULL);

    if(!hasFiles){
      std::cerr << RED << "[X] Error no 1L Dir -> " << this->L1_Dir_Name << RESET << std::endl;
      exit(-1);
    }

    hasFiles = false;

    fullDir += this->MM_Path;
    fullDir += "/";
    fullDir += getDir;
    dp = opendir(fullDir.c_str());

    std::cout << "\tLoading MM Dir .........."<< std::flush;

    do{
      entry = readdir(dp);
      if(entry!=NULL){
        if(this->MM_Dir_Name == entry->d_name){
          std::cout << GREEN << "[○] " << this->MM_Dir_Name << RESET << std::endl;
          getDir = entry->d_name;
          hasFiles = true;
        }
      }
    }while(entry!=NULL);

    if(!hasFiles){
      std::cerr << RED << "[X] Error no MM Dir -> " << this->MM_Dir_Name << RESET << std::endl;
      exit(-1);
    }

    fullDir += "/";
    fullDir += getDir;

    hasFiles = false;
    dp = opendir(fullDir.c_str());

    std::cout << "\tLoading MM files .........."<< std::flush;

    do{
      entry = readdir(dp);
      if(entry!=NULL){
        if(this->MM_Matrix_Name == entry->d_name){
          std::cout << GREEN << "[○] " << this->MM_Matrix_Name << RESET << std::endl;
          getDir = entry->d_name;
          hasFiles = true;
        }
      }
    }while(entry!=NULL);

    if(!hasFiles){
      std::cerr << RED << "[X] Error no MM files Dir -> " << this->MM_Matrix_Name << RESET << std::endl;
      exit(-1);
    }

    fullDir += "/";
    fullDir += getDir;
    this->fullPath = fullDir;
    dp = opendir(fullDir.c_str());

    std::cout << "\tLoading mtx file .........."<< std::flush;

    do{
      entry = readdir(dp);
      if(entry!=NULL){
        if((this->MM_Matrix_Name)+".mtx" == entry->d_name){
          hasMMFiles = true;
        }
      }
    }while(entry!=NULL);
    

    if(!hasMMFiles){
      std::cerr << RED << "[X] Error no mm file" << RESET << std::endl;
      exit(-1);
    }else{
      std::cout << GREEN << "[○] Get mm file" << RESET << std::endl;
    }

  }
}

void collection::showCMD(){
  std::cout <<"=========================================================" << std::endl;
  if(this->inputSource == 1){
    std::cout << "InputSource : CRS" << std::endl;
  }else if(this->inputSource == 2){
    std::cout << "InputSource : MM" << std::endl;
  }
  std::cout << "FullPath : " << this->fullPath << std::endl;
  std::cout << "N : " << this->N << std::endl;
  std::cout << "NNZ : " << this->NNZ << std::endl;

  if(this->isMultiGPU){
    std::cout << "N1 : " << this->N1 << std::endl;
    std::cout << "N2 : " << this->N2 << std::endl;
    std::cout << "NNZ1 : " << this->NNZ1 << std::endl;
    std::cout << "NNZ2 : " << this->NNZ2 << std::endl;
  }

  std::cout << "-------------" << std::endl;
  
  std::cout << "OpenMP thread : " << this->OMPThread << std::endl;
  std::cout << "GPU Num : " << this->CUDADevice << std::endl;

  std::cout << "-------------" << std::endl;

  std::cout << "VP : " << (this->isVP ? "On":"Off") << std::endl;
  std::cout << "CUDA : " << (this->isCUDA ? "On":"Off") << std::endl;
  std::cout << "Pinned : " << (this->isPinned ? "On":"Off") << std::endl;
  std::cout << "MultiGPU : " << (this->isMultiGPU ? "On":"Off") << std::endl;
  

  std::cout << "-------------" << std::endl;

  std::cout << "OuterSolver : " << enum2string(this->outerSolver) << std::endl;
  std::cout << "OuterMaxLoop : " << this->outerMaxLoop << std::endl;
  std::cout << "OuterEps : " << this->outerEps << std::endl;
  std::cout << "OuterRestart : " << this->outerRestart << std::endl;
  std::cout << "OuterKskip : " << this->outerKskip << std::endl;
  std::cout << "OuterFix : " << this->outerFix << std::endl;

  std::cout << "-------------" << std::endl;

  std::cout << "InnerSolver : " << enum2string(this->innerSolver) << std::endl;
  std::cout << "InnerMaxLoop : " << this->innerMaxLoop << std::endl;
  std::cout << "InnerEps : " << this->innerEps << std::endl;
  std::cout << "InnerRestart : " << this->innerRestart << std::endl;
  std::cout << "InnerKskip : " << this->innerKskip << std::endl;
  std::cout << "InnerFix : " << this->innerFix << std::endl;
  std::cout << "=========================================================" << std::endl;
}

void collection::checkCRSMatrix(){
  std::cout << "Checking CRS type Matrix .........."<< std::flush;
  //CRS input
  if(this->inputSource == 1){
    std::string valcol_path, ptr_path, bx_path;
    valcol_path = this->fullPath+"/ColVal.txt";
    ptr_path = this->fullPath+"/Ptr.txt";
    bx_path = this->fullPath+"/bx.txt";

    unsigned long int r_N[3];
    unsigned long int r_M[3];
    unsigned long int r_NNZ[3];

    std::ifstream valcol_file(valcol_path.c_str());
    if (valcol_file.fail())
    {
      std::cerr << "[X] Read valcol fail" << std::endl;
      exit(-1);
    }
    valcol_file >> r_N[0];
    valcol_file >> r_M[0];
    valcol_file >> r_NNZ[0];
    valcol_file.close();

    std::ifstream ptr_file(ptr_path.c_str());
    if (ptr_file.fail())
    {
      std::cerr << "[X] Read ptr fail" << std::endl;
      exit(-1);
    }
    ptr_file >> r_N[1];
    ptr_file >> r_M[1];
    ptr_file >> r_NNZ[1];
    ptr_file.close();

    std::ifstream bx_file(bx_path.c_str());
    if (bx_file.fail())
    {
      std::cerr << "[X] Read bx fail" << std::endl;
      exit(-1);
    }
    bx_file >> r_N[2];
    bx_file >> r_M[2];
    bx_file >> r_NNZ[2];
    bx_file.close();

    if(r_N[0] != r_N[1] || r_N[1] != r_N[2] || r_N[2] != r_N[1]){
      std::cerr << "[X] Three N in files is not same" << std::endl;
      exit(-1);
    }

    if(r_M[0] != r_M[1] || r_M[1] != r_M[2] || r_M[2] != r_M[1]){
      std::cerr << "[X] Three M in files is not same" << std::endl;
      exit(-1);
    }
    
    if(r_NNZ[0] != r_NNZ[1] || r_NNZ[1] != r_NNZ[2] || r_NNZ[2] != r_NNZ[1]){
      std::cerr << "[X] Three NNZ in files is not same" << std::endl;
      exit(-1);
    }

    if(r_N[0] != r_M[0]){
      std::cerr << "[X] N != M" << std::endl;
      exit(-1);
    }
    this->N = r_N[0];
    this->NNZ = r_NNZ[0];
    std::cout << GREEN << "[○] Done" << RESET << std::endl;
  }
}

void collection::readMatrix(){
  if(this->inputSource == 1){
    std::cout << "Loading CRS type Matrix .........."<< std::flush;
    std::string valcol_path, ptr_path, bx_path;
    valcol_path = this->fullPath+"/ColVal.txt";
    ptr_path = this->fullPath+"/Ptr.txt";
    bx_path = this->fullPath+"/bx.txt";


    // unsigned long int dammy;
    // unsigned long int counter[3]={0,0,0};
    int skip1, skip2, skip3;

    // std::ifstream valcol_file(valcol_path.c_str());
    // if (valcol_file.fail())
    // {
    //   std::cerr << "[X] Read valcol fail" << std::endl;
    //   exit(-1);
    // }
    // valcol_file >> dammy;
    // valcol_file >> dammy;
    // valcol_file >> dammy;
    //
    // while(!valcol_file.eof()){
    //   valcol_file >> this->col[counter[0]];
    //   valcol_file >> this->val[counter[0]];
    //   counter[0]++;
    //   std::cout << counter[0] << std::endl;
    // }
    // valcol_file.close();
    std::FILE* valcol_f = std::fopen(valcol_path.c_str(), "r");
    setvbuf(valcol_f,NULL,_IOFBF,512*1024);
    fscanf(valcol_f, "%d %d %d\n", &skip1, &skip2, &skip3);
    for(unsigned long int z=0; z<this->NNZ; z++){
      int fuga1;
      double fuga2;
      fscanf(valcol_f, "%d %le\n", &fuga1, &fuga2);
      this->col[z] = fuga1;
      this->val[z] = fuga2;
    }
    std::fclose(valcol_f);

    std::cout << "\n\tvalcol .........."<< GREEN << "[○] Done" << RESET << std::endl;

    // std::ifstream ptr_file(ptr_path.c_str());
    // if (ptr_file.fail())
    // {
    //   std::cerr << "[X] Read ptr fail" << std::endl;
    //   exit(-1);
    // }
    // ptr_file >> dammy;
    // ptr_file >> dammy;
    // ptr_file >> dammy;
    // while(!ptr_file.eof()){
    //   ptr_file >> this->ptr[counter[1]];
    //   counter[1]++;
    // }
    // ptr_file.close();
    std::FILE* ptr_f = std::fopen(ptr_path.c_str(), "r");
    setvbuf(ptr_f,NULL,_IOFBF,512*1024);
    fscanf(ptr_f, "%d %d %d\n", &skip1, &skip2, &skip3);
    for(unsigned long int z=0; z<this->N+1; z++){
      int fuga1;
      fscanf(ptr_f, "%d\n", &fuga1);
      this->ptr[z] = fuga1;
    }
    std::fclose(ptr_f);

    std::cout << "\tptr .........."<< GREEN << "[○] Done" << RESET << std::endl;

    // std::ifstream bx_file(bx_path.c_str());
    // if (bx_file.fail())
    // {
    //   std::cerr << "[X] Read bx fail" << std::endl;
    //   exit(-1);
    // }
    // bx_file >> dammy;
    // bx_file >> dammy;
    // bx_file >> dammy;
    // while(!bx_file.eof()){
    //   bx_file >> this->bvec[counter[2]];
    //   bx_file >> this->xvec[counter[2]];
    //   counter[2]++;
    // }
    // bx_file.close();
    std::FILE* bx_f = std::fopen(bx_path.c_str(), "r");
    setvbuf(bx_f,NULL,_IOFBF,512*1024);
    fscanf(bx_f, "%d %d %d\n", &skip1, &skip2, &skip3);
    for(unsigned long int z=0; z<this->N; z++){
      double fuga1;
      double fuga2;
      fscanf(bx_f, "%le %le\n", &fuga1, &fuga2);
      this->bvec[z] = fuga1;
      this->xvec[z] = fuga2;
    }
    std::fclose(bx_f);

    std::cout << "\tbx .........."<< GREEN << "[○] Done" << RESET << std::endl;

    std::cout << "        ................." << GREEN << "[○] Done" << RESET << std::endl;
  }

}

void collection::transpose(){
  unsigned long int col_counter = 0;

  std::memset(this->Tptr, -1, sizeof(int)*(this->N+1));

  std::cout << "Transpose Matrix in CPU .........."<< std::flush;

  for(unsigned long int i=0; i<N; i++){
    for(unsigned long int j=0; j<N; j++){
      for(int k=this->ptr[j]; k<this->ptr[j+1]; k++){
        if(this->col[k] == (int)i){
          if(this->Tptr[i] == -1){
            this->Tptr[i] = col_counter;
          }
          this->Tcol[col_counter] = j;
          this->Tval[col_counter] = this->val[k];
          col_counter++;
          continue;
        }
      }
    }
  }

  this->Tptr[N] = this->NNZ;
  std::cout << GREEN << "[○] Done" << RESET << std::endl;
}

void collection::transposeMatrix(){
  if(this->outerSolver == BICG || this->outerSolver == KSKIPBICG || this->outerSolver == VPBICG || this->innerSolver == BICG || this->innerSolver == KSKIPBICG || this->innerSolver == VPBICG){
    if(this->isCUDA){
      std::cout << "[GPU] Transpose Matrix in CUDA Use CSR2CSC.......... Cval -> CTval Tval" << std::flush;
      cu->CSR2CSC(this->Cval, this->Ccol, this->Cptr, this->Tval, this->Tcol, this->Tptr, this->CTval, this->CTcol, this->CTptr, this->N, this->NNZ);
    }else{
      std::cout << "[CPU] Transpose Matrix in CUDA Use CSR2CSC.......... val -> Tval"<< std::flush;
      cu->CSR2CSC(this->val, this->col, this->ptr, this->Tval, this->Tcol, this->Tptr, this->N, this->NNZ);
    }
    std::cout << GREEN << "[○] Done" << RESET << std::endl;
  }
}

void collection::setOpenmpThread(){
#ifdef _OPENMP
  omp_set_num_threads(this->OMPThread);
#endif
  std::string name = "OMP_NUM_THREADS";
  // std::string num = std::to_string(this->OMPThread);
  std::ostringstream stm;
  stm << this->OMPThread;
  setenv(name.c_str(), stm.str().c_str(), 1);
}

void collection::make2GPUMatrix(){
  
  if(isMultiGPU){
    std::cout << "[GPU] Make MultiGPU CPU side Matrix.......... val -> val1 + val2" << std::flush;
    for(unsigned long int i=0; i<this->N; i++){
      if(i<this->N1){
        this->ptr1[i] = this->ptr[i];
      }else{
        this->ptr2[i-N1] = this->ptr[i] - this->NNZ1;
      }
    }
    this->ptr1[this->N1] = this->NNZ1;
    this->ptr2[this->N2] = this->NNZ2;

    for(unsigned long int i=0; i<this->NNZ; i++){
      if(i<this->NNZ1){
        this->col1[i] = this->col[i];
        this->val1[i] = this->val[i];
      }else{
        this->col2[i-this->NNZ1] = this->col[i];
        this->val2[i-this->NNZ1] = this->val[i];
      }
    }
    std::cout << GREEN << "[○] Done" << RESET << std::endl;

    if(this->outerSolver == BICG || this->outerSolver == KSKIPBICG || this->outerSolver == VPBICG || this->innerSolver == BICG || this->innerSolver == KSKIPBICG || this->innerSolver == VPBICG){
      std::cout << "[GPU] Make MultiGPU CPU side Transpose Matrix .......... Tval -> Tval1 + Tval2" << std::flush;
      for(unsigned long int i=0; i<this->N; i++){
        if(i<this->N1){
          this->Tptr1[i] = this->Tptr[i];
        }else{
          this->Tptr2[i-N1] = this->Tptr[i] - this->NNZ1;
        }
      }
      this->Tptr1[this->N1] = this->NNZ1;
      this->Tptr2[this->N2] = this->NNZ2;

      for(unsigned long int i=0; i<this->NNZ; i++){
        if(i<this->NNZ1){
          this->Tcol1[i] = this->Tcol[i];
          this->Tval1[i] = this->Tval[i];
        }else{
          this->Tcol2[i-this->NNZ1] = this->Tcol[i];
          this->Tval2[i-this->NNZ1] = this->Tval[i];
        }
      }
      std::cout << GREEN << "[○] Done" << RESET << std::endl;
    }
  }
}

void collection::CudaCopy_Part1(){
  if(isCUDA){
    // if(!isMultiGPU){
      std::cout << "[GPU] Copy Matrix to CUDA.......... val -> Cval" << std::flush;
      cu->H2D(this->val, this->Cval, this->NNZ);
      cu->H2D(this->col, this->Ccol, this->NNZ);
      cu->H2D(this->ptr, this->Cptr, this->N+1);
      std::cout << GREEN << "[○] Done" << RESET << std::endl;
    // }
  }
}
void collection::CudaCopy_Part2(){
  if(isCUDA){
    if(!isMultiGPU){
      // if(this->outerSolver == BICG || this->outerSolver == KSKIPBICG || this->outerSolver == VPBICG || this->innerSolver == BICG || this->innerSolver == KSKIPBICG || this->innerSolver == VPBICG){
      //   std::cout << "[GPU] Copy Transpose Matrix to CUDA.......... Tval -> CTval"<< std::flush;
      //   cu->H2D(this->Tval, this->CTval, this->NNZ);
      //   cu->H2D(this->Tcol, this->CTcol, this->NNZ);
      //   cu->H2D(this->Tptr, this->CTptr, this->N+1);
      //   std::cout << GREEN << "[○] Done" << RESET << std::endl;
      // }
    }else{
      std::cout << "[GPU] Copy Matrix to CUDA [MultiGPU Part1].......... val1 -> Cval1" << std::flush;
      cu->H2D(this->val1, this->Cval1, this->NNZ1);
      cu->H2D(this->col1, this->Ccol1, this->NNZ1);
      cu->H2D(this->ptr1, this->Cptr1, this->N1+1);
      std::cout << GREEN << "[○] Done" << RESET << std::endl;

      std::cout << "[GPU] Copy Matrix to CUDA [MultiGPU Part2].......... val2 -> Cval2" << std::flush;
      cu->H2D(this->val2, this->Cval2, this->NNZ2);
      cu->H2D(this->col2, this->Ccol2, this->NNZ2);
      cu->H2D(this->ptr2, this->Cptr2, this->N2+1);
      std::cout << GREEN << "[○] Done" << RESET << std::endl;

      if(this->outerSolver == BICG || this->outerSolver == KSKIPBICG || this->outerSolver == VPBICG || this->innerSolver == BICG || this->innerSolver == KSKIPBICG || this->innerSolver == VPBICG){
        std::cout << "[GPU] Copy Transpose Matrix to CUDA [MultiGPU Part1].......... Tval1 -> CTval1"<< std::flush;
        cu->H2D(this->Tval1, this->CTval1, this->NNZ1);
        cu->H2D(this->Tcol1, this->CTcol1, this->NNZ1);
        cu->H2D(this->Tptr1, this->CTptr1, this->N1+1);
        std::cout << GREEN << "[○] Done" << RESET << std::endl;
        std::cout << "[GPU] Copy Transpose Matrix to CUDA [MultiGPU Part2].......... Tval2 -> CTval2"<< std::flush;
        cu->H2D(this->Tval2, this->CTval2, this->NNZ2);
        cu->H2D(this->Tcol2, this->CTcol2, this->NNZ2);
        cu->H2D(this->Tptr2, this->CTptr2, this->N2+1);
        std::cout << GREEN << "[○] Done" << RESET << std::endl;
      }
    }
  }
}

void collection::CRSAlloc_Part1(){


  // CPU side
  if(!isPinned){
    std::cout << "[CPU] Allocing Matrix .......... val col ptr"<< std::flush;
    this->val = new double [this->NNZ];
    this->col = new int [this->NNZ];
    this->ptr = new int [this->N+1];

    this->bvec = new double [this->N];
    this->xvec = new double [this->N];
  }else{
    std::cout << "[CPU] Allocing Matrix [Pinned Memory].......... val col ptr"<< std::flush;
    this->val = cu->d_MallocHost(this->NNZ);
    this->col = cu->i_MallocHost(this->NNZ);
    this->ptr = cu->i_MallocHost(this->N+1);

    this->bvec = new double [this->N];
    this->xvec = cu->d_MallocHost(this->N);
  }
  std::cout << GREEN << "[○] Done" << RESET << std::endl;
  //BI
  if(this->outerSolver == BICG || this->outerSolver == KSKIPBICG || this->outerSolver == VPBICG || this->innerSolver == BICG || this->innerSolver == KSKIPBICG || this->innerSolver == VPBICG){
    if(!isPinned){
      std::cout << "[CPU] Allocing Transpose Matrix .......... Tval Tcol Tptr"<< std::flush;
      this->Tval = new double [this->NNZ];
      this->Tcol = new int [this->NNZ];
      this->Tptr = new int [this->N+1];
    }else{
      std::cout << "[CPU] Allocing Transpose Matrix [Pinned Memory].......... Tval Tcol Tptr"<< std::flush;
      this->Tval = cu->d_MallocHost(this->NNZ);
      this->Tcol = cu->i_MallocHost(this->NNZ);
      this->Tptr = cu->i_MallocHost(this->N+1);
    }
    std::cout << GREEN << "[○] Done" << RESET << std::endl;
  }
}

void collection::CRSAlloc_Part2(){
if(this->N%2 == 0){
    this->N1 = this->N/2;
    this->N2 = this->N1;
  }else{
    this->N1 = std::ceil(this->N / 2.0);
    this->N2 = this->N - this->N1;
  }

  this->NNZ1 = this->ptr[this->N1];
  this->NNZ2 = this->NNZ - this->ptr[this->N1];

  // 2CPU
  if(isMultiGPU){
    if(!isPinned){
      std::cout << "[CPU] Allocing  Matrix [MultiCPU Part1].......... val1 col1 ptr1"<< std::flush;
      this->val1 = new double [NNZ1];
      this->col1 = new int [NNZ1];
      this->ptr1 = new int [N1+1];
      std::cout << GREEN << "[○] Done" << RESET << std::endl;
      std::cout << "[CPU] Allocing  Matrix [MultiCPU Part2].......... val2 col2 ptr2"<< std::flush;
      this->val2 = new double [NNZ2];
      this->col2 = new int [NNZ2];
      this->ptr2 = new int [N2+1];
      std::cout << GREEN << "[○] Done" << RESET << std::endl;
    }else{
      std::cout << "[CPU] Allocing  Matrix [MultiCPU Part1] [Pinned Memory].......... val1 col1 ptr1"<< std::flush;
      this->val1 = cu->d_MallocHost(this->NNZ1);
      this->col1 = cu->i_MallocHost(this->NNZ1);
      this->ptr1 = cu->i_MallocHost((this->N1)+1);
      std::cout << GREEN << "[○] Done" << RESET << std::endl;
      std::cout << "[CPU] Allocing  Matrix [MultiCPU Part2] [Pinned Memory].......... val2 col2 ptr2"<< std::flush;
      this->val2 = cu->d_MallocHost(this->NNZ2);
      this->col2 = cu->i_MallocHost(this->NNZ2);
      this->ptr2 = cu->i_MallocHost((this->N2)+1);
      std::cout << GREEN << "[○] Done" << RESET << std::endl;
    }
    if(this->outerSolver == BICG || this->outerSolver == KSKIPBICG || this->outerSolver == VPBICG || this->innerSolver == BICG || this->innerSolver == KSKIPBICG || this->innerSolver == VPBICG){
      if(!isPinned){
        std::cout << "[CPU] Allocing Transpose Matrix [MultiCPU Part1] .......... Tval1 Tcol1 Tptr1"<< std::flush;
        this->Tval1 = new double [this->NNZ1];
        this->Tcol1 = new int [this->NNZ1];
        this->Tptr1 = new int [(this->N1)+1];
        std::cout << GREEN << "[○] Done" << RESET << std::endl;
        std::cout << "[CPU] Allocing Transpose Matrix [MultiCPU Part2] .......... Tval2 Tcol2 Tptr2"<< std::flush;
        this->Tval2 = new double [this->NNZ2];
        this->Tcol2 = new int [this->NNZ2];
        this->Tptr2 = new int [(this->N2)+1];
        std::cout << GREEN << "[○] Done" << RESET << std::endl;
      }else{
        std::cout << "[CPU] Allocing Transpose Matrix [Pinned Memory] [MultiCPU Part1] .......... Tval1 Tcol1 Tptr1"<< std::flush;
        this->Tval1 = cu->d_MallocHost(this->NNZ1);
        this->Tcol1 = cu->i_MallocHost(this->NNZ1);
        this->Tptr1 = cu->i_MallocHost((this->N1)+1);
        std::cout << GREEN << "[○] Done" << RESET << std::endl;
        std::cout << "[CPU] Allocing Transpose Matrix [Pinned Memory] [MultiCPU Part2] .......... Tval2 Tcol2 Tptr2"<< std::flush;
        this->Tval2 = cu->d_MallocHost(this->NNZ2);
        this->Tcol2 = cu->i_MallocHost(this->NNZ2);
        this->Tptr2 = cu->i_MallocHost((this->N2)+1);
        std::cout << GREEN << "[○] Done" << RESET << std::endl;
      }
    }
  }

  //GPU side
  if(isCUDA){
    //1GPU
      std::cout << "[GPU] Allocing Matrix .......... Cval Ccol Cptr"<< std::flush;
      this->Cval = cu->d_Malloc(this->NNZ);
      this->Ccol = cu->i_Malloc(this->NNZ);
      this->Cptr = cu->i_Malloc(this->N+1);
      std::cout << GREEN << "[○] Done" << RESET << std::endl;
      if(this->outerSolver == BICG || this->outerSolver == KSKIPBICG || this->outerSolver == VPBICG || this->innerSolver == BICG || this->innerSolver == KSKIPBICG || this->innerSolver == VPBICG){
        std::cout << "[GPU] Allocing Transpose Matrix .......... CTval CTcol CTptr"<< std::flush;
        this->CTval = cu->d_Malloc(this->NNZ);
        this->CTcol = cu->i_Malloc(this->NNZ);
        this->CTptr = cu->i_Malloc(this->N+1);
        std::cout << GREEN << "[○] Done" << RESET << std::endl;
      }
    if(isMultiGPU){
      std::cout << "[GPU] Allocing Matrix [MultiCPU Part1].......... Cval1 Ccol1 Cptr1"<< std::flush;
      this->Cval1 = cu->d_Malloc(this->NNZ1, 0);
      this->Ccol1 = cu->i_Malloc(this->NNZ1, 0);
      this->Cptr1 = cu->i_Malloc((this->N1)+1, 0);
      std::cout << GREEN << "[○] Done" << RESET << std::endl;
      std::cout << "[GPU] Allocing Matrix [MultiCPU Part2].......... Cval2 Ccol2 Cptr2"<< std::flush;
      this->Cval2 = cu->d_Malloc(this->NNZ2, 1);
      this->Ccol2 = cu->i_Malloc(this->NNZ2, 1);
      this->Cptr2 = cu->i_Malloc((this->N2)+1, 1);
      std::cout << GREEN << "[○] Done" << RESET << std::endl;
      if(this->outerSolver == BICG || this->outerSolver == KSKIPBICG || this->outerSolver == VPBICG || this->innerSolver == BICG || this->innerSolver == KSKIPBICG || this->innerSolver == VPBICG){
        std::cout << "[GPU] Allocing Transpose Matrix [MultiCPU Part1].......... CTval1 CTcol1 CTptr1"<< std::flush;
        this->CTval1 = cu->d_Malloc(this->NNZ1, 0);
        this->CTcol1 = cu->i_Malloc(this->NNZ1, 0);
        this->CTptr1 = cu->i_Malloc((this->N1)+1, 0);
        std::cout << GREEN << "[○] Done" << RESET << std::endl;
        std::cout << "[GPU] Allocing Transpose Matrix [MultiCPU Part2].......... CTval2 CTcol2 CTptr2"<< std::flush;
        this->CTval2 = cu->d_Malloc(this->NNZ2, 1);
        this->CTcol2 = cu->i_Malloc(this->NNZ2, 1);
        this->CTptr2 = cu->i_Malloc((this->N2)+1, 1);
        std::cout << GREEN << "[○] Done" << RESET << std::endl;
      }
    }
  }
}

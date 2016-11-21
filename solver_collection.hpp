#ifndef SOLVER_COLLECTION_HPP_INCLUDED__
#define SOLVER_COLLECTION_HPP_INCLUDED__

#include <iostream>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <string>

// #include "solver_collection.hpp"
#include "cmdline.h"
#include "color.hpp"

enum SOLVERS_NAME {
    NONE,
    CG,
    BICG,
    CR,
    GCR,
    GMRES,

    KSKIPCG,
    KSKIPCR,
    KSKIPBICG,
    
    VPCG,
    VPBICG,
    VPCR,
    VPGCR,
    VPGMRES
  };

template <typename T>
class collection {
  public:
    bool isVP;
    bool isCUDA;
    bool isOpenMP;
    bool isVerbose;
    bool isInner;
    int OMPThread;

    std::string L1_Dir_Name;

    std::string CRS_Path;
    std::string CRS_Dir_Name;

    std::string MM_Path;
    std::string MM_Dir_Name;

    SOLVERS_NAME outerSolver;
    SOLVERS_NAME innerSolver;
    long int outerMaxLoop;
    long int innerMaxLoop;
    double outerEps;
    double innerEps;
    long int outerRestart;
    long int innerRestart;
    int outerKskip;
    int innerKskip;
    int outerFix;
    int innerFix;

    
//not init
    std::string CRS_Matrix_Name;
    std::string MM_Matrix_Name;
    std::string fullPath; 
    long int N; 
    long int NNZ;
    int inputSource; //1->CRS, 2->MM


    collection();
    ~collection();
    void readCMD(int argc, char* argv[]);
    void checkCMD();
    void showCMD();

    std::string enum2string(SOLVERS_NAME id);
    SOLVERS_NAME string2enum(std::string str);

    void checkMatrix();
    void readMatrix();
    void CRSAlloc();
    void transposeMatrix();
    void transpose();

//pointer
    T* val;
    int* col;
    int* ptr;

    T* Tval;
    int* Tcol;
    int* Tptr;

    T* bvec;
    T* xvec;


};

template <typename T>
collection<T>::collection() {
  isVP = false;
  isCUDA = false;
  isOpenMP = true;
  isVerbose = false;
  isInner = false;
  OMPThread = 8;

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
}

template <typename T>
collection<T>::~collection() {
  delete[] val;
  delete[] col;
  delete[] ptr;
  delete[] bvec;
  delete[] xvec;
  if(this->outerSolver == BICG || this->outerSolver == KSKIPBICG || this->outerSolver == VPBICG || this->innerSolver == BICG || this->innerSolver == KSKIPBICG || this->innerSolver == VPBICG){
    delete[] Tval;
    delete[] Tcol;
    delete[] Tptr;
  }
}

template <typename T>
std::string collection<T>::enum2string(SOLVERS_NAME id){
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

template <typename T>
SOLVERS_NAME collection<T>::string2enum(std::string str){
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

template <typename T>
void collection<T>::readCMD(int argc, char* argv[]){
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
  
  cmd.add<std::string>("InnerSolver", 's', "method use in innersolver", false, "");
  cmd.add<int>("InnerMaxLoop", 'l', "max loops in innersolver", false, this->innerMaxLoop);
  cmd.add<double>("InnerEps", 'e', "eps in innersolver", false, this->innerEps);
  cmd.add<int>("InnerRestart", 'r', "Restart number in innersolver", false, this->innerRestart);
  cmd.add<int>("InnerKskip", 'k', "kskip number in innersolver", false, this->innerKskip);
  cmd.add<int>("InnerFix", 'f', "fix bug in innersolver", false, this->innerFix);

  cmd.add("verbose", 'v', "verbose mode will printout all detel ");

  cmd.parse_check(argc, argv);

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

  if(cmd.exist("verbose")) this->isVerbose=true;

}

template <typename T>
void collection<T>::checkCMD(){
  DIR *dp;
  dirent* entry;

  std::string fullDir;
  std::string getDir;

  bool hasFiles = false;
  bool hasCRSFiles[3]={false, false, false};
  bool hasMMFiles = false;

  // CRS input
  if(this->inputSource == 1){

    std::cout << GREEN << "[○]Input Matrix Type => CRS" << RESET << std::endl;
    dp = opendir(this->CRS_Path.c_str());
    if(dp == NULL) exit(1);

    do{
      entry = readdir(dp);
      if(entry!=NULL){
        if(this->L1_Dir_Name == entry->d_name){
          std::cout << GREEN << "[○]Get 1L Dir -> " << this->L1_Dir_Name << RESET << std::endl;
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
    do{
      entry = readdir(dp);
      if(entry!=NULL){
        if(this->CRS_Dir_Name == entry->d_name){
          std::cout << GREEN << "[○]Get CRS Dir -> " << this->CRS_Dir_Name << RESET << std::endl;
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
    do{
      entry = readdir(dp);
      if(entry!=NULL){
        if(this->CRS_Matrix_Name == entry->d_name){
          std::cout << GREEN << "[○]Get CRS files Dir -> " << this->CRS_Matrix_Name << RESET << std::endl;
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
      std::cerr << RED << "[X]Error in CRS files " << RESET << std::endl;
      exit(-1);
    }else{
      std::cout << GREEN << "[○]Get CRS files " << RESET << std::endl;
    }

    // MM input
  }else if(this->inputSource == 2){

    std::cout << GREEN << "[○]Input Matrix Type => CRS" << RESET << std::endl;

    dp = opendir(this->MM_Path.c_str());
    if(dp == NULL) exit(1);

    do{
      entry = readdir(dp);
      if(entry!=NULL){
        if(this->L1_Dir_Name == entry->d_name){
          std::cout << GREEN << "[○]Get 1L Dir -> " << this->L1_Dir_Name << RESET << std::endl;
          getDir = entry->d_name;
          hasFiles = true;
        }
      }
    }while(entry!=NULL);

    if(!hasFiles){
      std::cerr << RED << "[X]Error no 1L Dir -> " << this->L1_Dir_Name << RESET << std::endl;
      exit(-1);
    }

    hasFiles = false;

    fullDir += this->MM_Path;
    fullDir += "/";
    fullDir += getDir;
    dp = opendir(fullDir.c_str());

    do{
      entry = readdir(dp);
      if(entry!=NULL){
        if(this->MM_Dir_Name == entry->d_name){
          std::cout << GREEN << "[○]Get MM Dir -> " << this->MM_Dir_Name << RESET << std::endl;
          getDir = entry->d_name;
          hasFiles = true;
        }
      }
    }while(entry!=NULL);

    if(!hasFiles){
      std::cerr << RED << "[X]Error no MM Dir -> " << this->MM_Dir_Name << RESET << std::endl;
      exit(-1);
    }

    fullDir += "/";
    fullDir += getDir;

    hasFiles = false;
    dp = opendir(fullDir.c_str());

    do{
      entry = readdir(dp);
      if(entry!=NULL){
        if(this->MM_Matrix_Name == entry->d_name){
          std::cout << GREEN << "[○]Get MM files Dir -> " << this->MM_Matrix_Name << RESET << std::endl;
          getDir = entry->d_name;
          hasFiles = true;
        }
      }
    }while(entry!=NULL);

    if(!hasFiles){
      std::cerr << RED << "[X]Error no MM files Dir -> " << this->MM_Matrix_Name << RESET << std::endl;
      exit(-1);
    }

    fullDir += "/";
    fullDir += getDir;
    this->fullPath = fullDir;
    dp = opendir(fullDir.c_str());

    do{
      entry = readdir(dp);
      if(entry!=NULL){
        if((this->MM_Matrix_Name)+".mtx" == entry->d_name){
          std::cout << GREEN << "[○]Get mtx matrix -> " << this->MM_Matrix_Name << RESET << std::endl;
          hasMMFiles = true;
        }
      }
    }while(entry!=NULL);

    if(!hasMMFiles){
      std::cerr << RED << "[X]Error no mm file" << RESET << std::endl;
      exit(-1);
    }else{
      std::cout << GREEN << "[○]Get mm file" << RESET << std::endl;
    }

    std::cout << GREEN << "[○] PreProcess Done" << RESET << std::endl;
  }
}

template <typename T>
void collection<T>::showCMD(){
  std::cout <<"=========================================================" << std::endl;
  if(this->inputSource == 1){
    std::cout << "InputSource : CRS" << std::endl;
  }else if(this->inputSource == 2){
    std::cout << "InputSource : MM" << std::endl;
  }
  std::cout << "FullPath : " << this->fullPath << std::endl;
  std::cout << "N : " << this->N << std::endl;
  std::cout << "NNZ : " << this->NNZ << std::endl;

  std::cout << "-------------" << std::endl;

  std::cout << "VP : " << this->isVP << std::endl;
  std::cout << "CUDA : " << this->isCUDA << std::endl;
  std::cout << "OpenMP : " << this->isOpenMP << std::endl;
  std::cout << "OpenMP thread : " << this->OMPThread << std::endl;

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

template <typename T>
void collection<T>::checkMatrix(){
  //CRS input
  if(this->inputSource == 1){
    std::string valcol_path, ptr_path, bx_path;
    valcol_path = this->fullPath+"/ColVal.txt";
    ptr_path = this->fullPath+"/Ptr.txt";
    bx_path = this->fullPath+"/bx.txt";

    long int r_N[3];
    long int r_M[3];
    long int r_NNZ[3];

    std::ifstream valcol_file(valcol_path);
    if (valcol_file.fail())
    {
      std::cerr << "read valcol fail" << std::endl;
      exit(-1);
    }
    valcol_file >> r_N[0];
    valcol_file >> r_M[0];
    valcol_file >> r_NNZ[0];
    valcol_file.close();

    std::ifstream ptr_file(ptr_path);
    if (ptr_file.fail())
    {
      std::cerr << "read ptr fail" << std::endl;
      exit(-1);
    }
    ptr_file >> r_N[1];
    ptr_file >> r_M[1];
    ptr_file >> r_NNZ[1];
    ptr_file.close();

    std::ifstream bx_file(bx_path);
    if (bx_file.fail())
    {
      std::cerr << "read bx fail" << std::endl;
      exit(-1);
    }
    bx_file >> r_N[2];
    bx_file >> r_M[2];
    bx_file >> r_NNZ[2];
    bx_file.close();

    if(r_N[0] != r_N[1] || r_N[1] != r_N[2] || r_N[2] != r_N[1]){
      std::cerr << "three N in files is not same" << std::endl;
      exit(-1);
    }

    if(r_M[0] != r_M[1] || r_M[1] != r_M[2] || r_M[2] != r_M[1]){
      std::cerr << "three M in files is not same" << std::endl;
      exit(-1);
    }
    
    if(r_NNZ[0] != r_NNZ[1] || r_NNZ[1] != r_NNZ[2] || r_NNZ[2] != r_NNZ[1]){
      std::cerr << "three NNZ in files is not same" << std::endl;
      exit(-1);
    }

    if(r_N[0] != r_M[0]){
      std::cerr << "N != M" << std::endl;
      exit(-1);
    }
    this->N = r_N[0];
    this->NNZ = r_NNZ[0];
    std::cout << GREEN << "[○]Get N NNZ" << RESET << std::endl;
  }
}

template <typename T>
void collection<T>::readMatrix(){
  if(this->inputSource == 1){
    std::string valcol_path, ptr_path, bx_path;
    valcol_path = this->fullPath+"/ColVal.txt";
    ptr_path = this->fullPath+"/Ptr.txt";
    bx_path = this->fullPath+"/bx.txt";

    long int dammy;
    long int counter[3]={0,0,0};

    std::ifstream valcol_file(valcol_path);
    if (valcol_file.fail())
    {
      std::cerr << "read valcol fail" << std::endl;
      exit(-1);
    }
    valcol_file >> dammy;
    valcol_file >> dammy;
    valcol_file >> dammy;
    while(!valcol_file.eof()){
      valcol_file >> this->col[counter[0]];
      valcol_file >> this->val[counter[0]];
      counter[0]++;
    }
    valcol_file.close();

    std::ifstream ptr_file(ptr_path);
    if (ptr_file.fail())
    {
      std::cerr << "read ptr fail" << std::endl;
      exit(-1);
    }
    ptr_file >> dammy;
    ptr_file >> dammy;
    ptr_file >> dammy;
    while(!ptr_file.eof()){
      ptr_file >> this->ptr[counter[1]];
      counter[1]++;
    }
    ptr_file.close();

    std::ifstream bx_file(bx_path);
    if (bx_file.fail())
    {
      std::cerr << "read bx fail" << std::endl;
      exit(-1);
    }
    bx_file >> dammy;
    bx_file >> dammy;
    bx_file >> dammy;
    while(!bx_file.eof()){
      bx_file >> this->bvec[counter[2]];
      bx_file >> this->xvec[counter[2]];
      counter[2]++;
    }
    bx_file.close();

    std::cout << GREEN << "[○]Get Matrix"<< RESET << std::endl;
  }

}

template <typename T>
void collection<T>::CRSAlloc(){
  this->val = new T [this->NNZ];
  this->col = new int [this->NNZ];
  this->ptr = new int [this->N+1];
  this->bvec = new T [this->N];
  this->xvec = new T [this->N];

  if(this->outerSolver == BICG || this->outerSolver == KSKIPBICG || this->outerSolver == VPBICG || this->innerSolver == BICG || this->innerSolver == KSKIPBICG || this->innerSolver == VPBICG){
    this->Tval = new T [this->NNZ];
    this->Tcol = new int [this->NNZ];
    this->Tptr = new int [this->N+1];
  }
}

template <typename T>
void collection<T>::transpose(){
  long int col_counter = 0;
  std::memset(Tptr, -1, sizeof(int)*this->N+1);

  for(long int i=0; i<N; i++){
    for(long int j=0; j<N; j++){
      for(long int k=this->ptr[j]; k<this->ptr[j+1]; k++){
        if(this->col[k] == i){
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
}

template <typename T>
void collection<T>::transposeMatrix(){
  if(this->outerSolver == BICG || this->outerSolver == KSKIPBICG || this->outerSolver == VPBICG || this->innerSolver == BICG || this->innerSolver == KSKIPBICG || this->innerSolver == VPBICG){
    this->transpose();
  }
}

#endif //SOLVER_COLLECTION_HPP_INCLUDED__


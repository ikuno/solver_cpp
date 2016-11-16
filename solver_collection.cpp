#include <cstdlib>
#include <dirent.h>

#include "solver_collection.hpp"
#include "cmdline.h"

collection::collection() {
  isVP = false;
  isCUDA = false;
  isOpenMP = true;
  isVerbose = false;
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
  outerFix = 1;
  innerFix = 1;

  L1_Dir_Name = "Matrix";

  CRS_Dir_Name = "CRS";
  CRS_Path = "../"; 

  MM_Dir_Name = "MM";
  MM_Path = "../";
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
  
  cmd.add<std::string>("InnerSolver", 's', "method use in innersolver", false, "");
  cmd.add<int>("InnerMaxLoop", 'l', "max loops in innersolver", false, this->innerMaxLoop);
  cmd.add<double>("InnerEps", 'e', "eps in innersolver", false, this->innerEps);
  cmd.add<int>("InnerRestart", 'r', "Restart number in innersolver", false, this->innerRestart);
  cmd.add<int>("InnerKskip", 'k', "kskip number in innersolver", false, this->innerKskip);
  cmd.add<int>("InnerFix", 'f', "fix bug in innersolver", false, this->innerFix);

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
    std::cout << "[!]Must set one input source" << std::endl;
    exit(-1);
  }

  if(this->MM_Matrix_Name != "NONE" && this->CRS_Matrix_Name != "NONE"){
    std::cout << "[!]Only can set one input source" << std::endl;
    exit(-1);
  }

  if(this->MM_Matrix_Name != "NONE"){
    this->inputSource = 2;
  }else if(this->CRS_Matrix_Name != "NONE"){
    this->inputSource = 1;
  }

}

void collection::checkCMD(){
  DIR *dp;
  dirent* entry;

  std::string fullDir;
  std::string getDir;

  bool hasFiles = false;
  bool hasCRSFiles[3]={false, false, false};
  bool hasMMFiles = false;

  // CRS input
  if(this->inputSource == 1){

    std::cout << "Input => CRS" << std::endl;
    dp = opendir(this->CRS_Path.c_str());
    if(dp == NULL) exit(1);

    do{
      entry = readdir(dp);
      if(entry!=NULL){
        if(this->L1_Dir_Name == entry->d_name){
          std::cout << "Get 1L Dir -> " << this->L1_Dir_Name << std::endl;
          getDir = entry->d_name;
          hasFiles = true;
        }
      }
    }while(entry!=NULL);

    if(!hasFiles){
      std::cout << "Error no 1L Dir -> " << this->L1_Dir_Name << std::endl;
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
          std::cout << "Get CRS Dir -> " << this->CRS_Dir_Name << std::endl;
          getDir = entry->d_name;
          hasFiles = true;
        }
      }
    }while(entry!=NULL);

    if(!hasFiles){
      std::cout << "Error no CRS Dir -> " << this->CRS_Dir_Name << std::endl;
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
          std::cout << "Get CRS files Dir " << this->CRS_Matrix_Name << std::endl;
          getDir = entry->d_name;
          hasFiles = true;
        }
      }
    }while(entry!=NULL);

    if(!hasFiles){
      std::cout << "Error no CRS files Dir" << this->CRS_Matrix_Name << std::endl;
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
      std::cout << "Error in CRS files " << std::endl;
      exit(-1);
    }else{
      std::cout << "Get CRS files " << std::endl;
    }

    std::cout << "All done" << std::endl;

    // MM input
  }else if(this->inputSource == 2){

    std::cout << "Input => CRS" << std::endl;

    dp = opendir(this->MM_Path.c_str());
    if(dp == NULL) exit(1);

    do{
      entry = readdir(dp);
      if(entry!=NULL){
        if(this->L1_Dir_Name == entry->d_name){
          std::cout << "Get 1L Dir -> " << this->L1_Dir_Name << std::endl;
          getDir = entry->d_name;
          hasFiles = true;
        }
      }
    }while(entry!=NULL);

    if(!hasFiles){
      std::cout << "Error no 1L Dir -> " << this->L1_Dir_Name << std::endl;
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
          std::cout << "Get MM Dir -> " << this->MM_Dir_Name << std::endl;
          getDir = entry->d_name;
          hasFiles = true;
        }
      }
    }while(entry!=NULL);

    if(!hasFiles){
      std::cout << "Error no MM Dir -> " << this->MM_Dir_Name << std::endl;
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
          std::cout << "Get MM files Dir -> " << this->MM_Matrix_Name << std::endl;
          getDir = entry->d_name;
          hasFiles = true;
        }
      }
    }while(entry!=NULL);

    if(!hasFiles){
      std::cout << "Error no MM files Dir -> " << this->MM_Matrix_Name << std::endl;
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
          std::cout << "Get mtx matrix" << std::endl;
          hasMMFiles = true;
        }
      }
    }while(entry!=NULL);

    if(!hasMMFiles){
      std::cout << "Error no mm file" << std::endl;
      exit(-1);
    }else{
      std::cout << "Get mm file" << std::endl;
    }

    std::cout << "ALL done" << std::endl;
  }
}

void collection::showCMD(){
  std::cout << "=========================================================" << std::endl;
  if(this->inputSource == 1){
    std::cout << "InputSource : CRS" << std::endl;
  }else if(this->inputSource == 2){
    std::cout << "InputSource : MM" << std::endl;
  }
  std::cout << "FullPath : " << this->fullPath << std::endl;

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

void collection::CRSMalloc(){

}

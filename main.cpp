#include <iostream>
#include <cmath>
#include "outerMethods.hpp"
#include "solver_collection.hpp"

int main(int argc, char* argv[])
{
  collection col;
  outerMethods method(&col);

  col.readCMD(argc, argv);
  col.checkCMD();
  col.checkCRSMatrix();
  col.CRSAlloc();
  col.readMatrix();
  if(col.isMultiGPU){
    col.MultiGPUAlloc();
  }
  col.CudaCopy();
  col.transposeMatrix();
  col.setOpenmpThread();

  // double *vec = new double [col.N];
  // double *out = new double [col.N];
  // double tmp;
  //
  // for(unsigned long int i=0; i<col.N; i++){
  //   vec[i] = 2.0;
  //   out[i] = 0.0;
  // }
  //
  // unsigned long int N1,N2;
  //
  // if(col.N%2 == 0){
  //   N1 = col.N/2;
  //   N2 = N1;
  // }else{
  //   N1 = std::ceil(col.N / 2.0);
  //   N2 = col.N - N1;
  // }
  //
  // std::cout << "N=" << col.N << ", N1=" << N1 << ", N2=" << N2 << std::endl;
  //
  // int *col1, *ptr1;
  // int *col2, *ptr2;
  // double *val1, *val2;
  //
  // unsigned int NNZ1, NNZ2;
  // NNZ1 = col.ptr[N1];
  // NNZ2 = col.NNZ - col.ptr[N1];
  //
  // std::cout << "NNZ1=" << NNZ1 << ", NNZ2=" << NNZ2 << std::endl;
  //
  //
  // val1 = new double [NNZ1];
  // col1 = new int [NNZ1];
  // ptr1 = new int [N1+1];
  //
  // val2 = new double [NNZ2];
  // col2 = new int [NNZ2];
  // ptr2 = new int [N2+1];
  //
  // for(unsigned long int i=0; i<col.N; i++){
  //   if(i<N1){
  //     ptr1[i] = col.ptr[i];
  //   }else{
  //     ptr2[i-N1] = col.ptr[i] - NNZ1;
  //   }
  // }
  // ptr1[N1] = NNZ1;
  // ptr2[N2] = NNZ2;
  //
  //
  // for(unsigned long int i=0; i<col.NNZ; i++){
  //   if(i<NNZ1){
  //     col1[i] = col.col[i];
  //     val1[i] = col.val[i];
  //   }else{
  //     col2[i-NNZ1] = col.col[i];
  //     val2[i-NNZ1] = col.val[i];
  //   }
  // }

  //
  // for(unsigned long int i=0; i<N1; i++){
  //   tmp = 0.0;
  //   for(int j=ptr1[i]; j<ptr1[i+1]; j++){
  //     tmp += val1[j] * vec[col1[j]];
  //   }
  //   out[i] = tmp;
  // }
  //
  //
  // for(unsigned long int i=0; i<N2; i++){
  //   tmp = 0.0;
  //   for(int j=ptr2[i]; j<ptr2[i+1]; j++){
  //     tmp += val2[j] * vec[col2[j]];
  //   }
  //   out[N1 + i] = tmp;
  // }
  //
//
//
  // for(unsigned long int i=0; i<col.N; i++){
  //   tmp = 0.0;
  //   for(int j=col.ptr[i]; j<col.ptr[i+1]; j++){
  //     tmp += col.val[j] * vec[col.col[j]];
  //   }
  //   out[i] = tmp;
  // }
  //
  //
  //
  // std::cout << "-----------------------------" << std::endl;
  // for(unsigned long int i=0; i<col.N; i++){
  //   std::cout << out[i] << std::endl;
  // }
  //
  // delete vec;
  // delete out;
  //
  // delete col1;
  // delete ptr1;
  // delete val1;
  //
  // delete col2;
  // delete ptr2;
  // delete val2;
  //

  col.showCMD();
  method.outerSelect(col.outerSolver);

  if(col.isVerbose)
    col.showCMD();

  return 0;
}

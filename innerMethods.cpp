#include "innerMethods.hpp"
#include "color.hpp"

#include "cg.hpp"
#include "cr.hpp"
#include "gcr.hpp"
#include "bicg.hpp"
#include "gmres.hpp"
#include "kskipcg.hpp"
#include "kskipbicg.hpp"

innerMethods::innerMethods(collection *coll){
  this->Ccoll = coll;
}

void innerMethods::innerSelect(collection *innercoll, SOLVERS_NAME solver, double *ibvec, double *ixvec){
  int result = 1;
  if(solver == CG){
    cg innerSolver(innercoll, ibvec, ixvec, true);
    result = innerSolver.solve();
  }
  
  else if(solver == CR){
    cr innerSolver(innercoll, ibvec, ixvec, true);
    result = innerSolver.solve();
  }
  
  else if(solver == GCR){
    
    gcr innerSolver(innercoll, ibvec, ixvec, true);
    result = innerSolver.solve();
   
  }
  
  else if(solver == BICG){
    
    bicg innerSolver(innercoll, ibvec, ixvec, true);
    result = innerSolver.solve();
    
  }
  
  else if(solver == GMRES){
    
    gmres innerSolver(innercoll, ibvec, ixvec, true);
    result = innerSolver.solve();
    
  }
  
  else if(solver == KSKIPCG){
    kskipcg innerSolver(innercoll, ibvec, ixvec, true);
    result = innerSolver.solve();
    
  }
  
  else if(solver == KSKIPCR){
    std::cout << RED << "Not implemented" << RESET << std::endl;
    result = 1;
  }
  
  else if(solver == KSKIPBICG){
    
    kskipBicg innerSolver(innercoll, ibvec, ixvec, true);
    result = innerSolver.solve();
    
  }

  if(result != 2 && result !=0 ){
    std::cerr << RED << "inner result -> "<< result << RESET << std::endl;
    std::cerr << RED << "[X]Some Error in methods" << RESET << std::endl;
  }
}

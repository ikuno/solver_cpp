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
    cg *innerSolver = new cg(innercoll, ibvec, ixvec, true);
    result = innerSolver->solve();
    delete innerSolver;
  }
  
  else if(solver == CR){
    cr *innerSolver = new cr(innercoll, ibvec, ixvec, true);
    result = innerSolver->solve();
    delete innerSolver;
  }
  
  else if(solver == GCR){
    gcr *innerSolver = new gcr(innercoll, ibvec, ixvec, true);
    result = innerSolver->solve();
    delete innerSolver;
  }
  
  else if(solver == BICG){
    bicg *innerSolver = new bicg(innercoll, ibvec, ixvec, true);
    result = innerSolver->solve();
    delete innerSolver;
  }
  
  else if(solver == GMRES){
    gmres *innerSolver = new gmres(innercoll, ibvec, ixvec, true);
    result = innerSolver->solve();
    delete innerSolver;
  }
  
  else if(solver == KSKIPCG){
    kskipcg *innerSolver = new kskipcg(innercoll, ibvec, ixvec, true);
    result = innerSolver->solve();
    delete innerSolver;
  }
  
  else if(solver == KSKIPCR){
    std::cout << RED << "Not implemented" << RESET << std::endl;
    result = 1;
  }
  
  else if(solver == KSKIPBICG){
    kskipBicg *innerSolver = new kskipBicg(innercoll, ibvec, ixvec, true);
    result = innerSolver->solve();
    delete innerSolver;
  }

  if(result != 2 && result !=0 ){
    std::cerr << RED << "inner result -> "<< result << RESET << std::endl;
    std::cerr << RED << "[X]Some Error in methods" << RESET << std::endl;
  }
}

#include "outerMethods.hpp"
#include "color.hpp"

#include "cg.hpp"
#include "cr.hpp"
#include "gcr.hpp"
#include "bicg.hpp"
#include "gmres.hpp"
#include "kskipcg.hpp"
#include "kskipbicg.hpp"
#include "vpcg.hpp"
#include "vpcr.hpp"
#include "vpgcr.hpp"
#include "vpgmres.hpp"

outerMethods::outerMethods(collection *coll){
  this->coll = coll;
}

void outerMethods::outerSelect(SOLVERS_NAME solver){
  int result = 99;
  if(solver == CG){
    cg outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }else if(solver == CR){
    cr outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }else if(solver == GCR){
    gcr outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }else if(solver == BICG){
    bicg outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }else if(solver == GMRES){
    gmres outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }else if(solver == KSKIPCG){
    kskipcg outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }else if(solver == KSKIPCR){
    std::cout << RED << "Not implemented" << RESET << std::endl;
    result = 99;
  }else if(solver == KSKIPBICG){
    kskipBicg outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }else if(solver == VPCG){
    vpcg outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }else if(solver == VPBICG){
    std::cout << RED << "Not implemented" << RESET << std::endl;
    result = 99;
  }else if(solver == VPCR){
    vpcr outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }else if(solver == VPGCR){
    vpgcr outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }else if(solver == VPGMRES){
    vpgmres outerSolver(coll, coll->bvec, coll->xvec, false);
    result = outerSolver.solve();
  }
  
  if(result == 0){
    std::cout << BOLDGREEN << "converge" << RESET << std::endl;
  }else if(result == 2){
    std::cout << BOLDRED << "not converge" << RESET << std::endl;
  }else{
    std::cerr << RED << "outer result -> "<< result << RESET << std::endl;
    std::cerr << RED << "[X]Some Error in methods" << RESET << std::endl;
  }
}

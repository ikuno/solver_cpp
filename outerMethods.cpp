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
    cg *outerSolver = new cg(coll, coll->bvec, coll->xvec, false);
    result = outerSolver->solve();
    delete outerSolver;
  }else if(solver == CR){
    cr *outerSolver = new cr(coll, coll->bvec, coll->xvec, false);
    result = outerSolver->solve();
    delete outerSolver;
  }else if(solver == GCR){
    gcr *outerSolver = new gcr(coll, coll->bvec, coll->xvec, false);
    result = outerSolver->solve();
    delete outerSolver;
  }else if(solver == BICG){
    bicg *outerSolver = new bicg(coll, coll->bvec, coll->xvec, false);
    result = outerSolver->solve();
    delete outerSolver;
  }else if(solver == GMRES){
    gmres *outerSolver = new gmres(coll, coll->bvec, coll->xvec, false);
    result = outerSolver->solve();
    delete outerSolver;
  }else if(solver == KSKIPCG){
    kskipcg *outerSolver = new kskipcg(coll, coll->bvec, coll->xvec, false);
    result = outerSolver->solve();
    delete outerSolver;
  }else if(solver == KSKIPCR){
    std::cout << RED << "Not implemented" << RESET << std::endl;
    result = 99;
  }else if(solver == KSKIPBICG){
    kskipBicg *outerSolver = new kskipBicg(coll, coll->bvec, coll->xvec, false);
    result = outerSolver->solve();
    delete outerSolver;
  }else if(solver == VPCG){
    vpcg *outerSolver = new vpcg(coll, coll->bvec, coll->xvec, false);
    result = outerSolver->solve();
    delete outerSolver;
  }else if(solver == VPBICG){
    std::cout << RED << "Not implemented" << RESET << std::endl;
    result = 99;
  }else if(solver == VPCR){
    vpcr *outerSolver = new vpcr(coll, coll->bvec, coll->xvec, false);
    result = outerSolver->solve();
    delete outerSolver;
  }else if(solver == VPGCR){
    vpgcr *outerSolver = new vpgcr(coll, coll->bvec, coll->xvec, false);
    result = outerSolver->solve();
    delete outerSolver;
  }else if(solver == VPGMRES){
    vpgmres *outerSolver = new vpgmres(coll, coll->bvec, coll->xvec, false);
    result = outerSolver->solve();
    delete outerSolver;
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

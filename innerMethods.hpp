#ifndef INNERMETHODS_HPP_INCLUDED__
#define INNERMETHODS_HPP_INCLUDED__

#include "solver_collection.hpp"
#include "cg.hpp"
#include "cr.hpp"
#include "gcr.hpp"
#include "bicg.hpp"
#include "gmres.hpp"
#include "kskipcg.hpp"
#include "kskipbicg.hpp"

template <typename T>
class innerMethods {
  public:
    collection<T> *Ccoll;
    innerMethods(collection<T> *coll);
    void innerSelect(collection<T> *hoge, SOLVERS_NAME solver, T *ibvec, T *ixvec);
};

template <typename T>
innerMethods<T>::innerMethods(collection<T> *coll){
  this->Ccoll = coll;

}

template <typename T>
void innerMethods<T>::innerSelect(collection<T> *innercoll, SOLVERS_NAME solver, T *ibvec, T *ixvec){
  int result = 1;
  if(solver == CG){
    if(innercoll->isMixPrecision){
      innercoll->isInnerNow = true;
    }
    cg<T> innerSolver(innercoll, ibvec, ixvec, true);
    result = innerSolver.solve();
    if(innercoll->isMixPrecision){
      innercoll->isInnerNow = false;
    }
  }
  
  else if(solver == CR){
    if(innercoll->isMixPrecision){
      innercoll->isInnerNow = true;
    }

    cr<T> innerSolver(innercoll, ibvec, ixvec, true);
    result = innerSolver.solve();
    if(innercoll->isMixPrecision){
      innercoll->isInnerNow = false;
    }
  }
  
  else if(solver == GCR){
    if(innercoll->isMixPrecision){
      innercoll->isInnerNow = true;
    }
    gcr<T> innerSolver(innercoll, ibvec, ixvec, true);
    result = innerSolver.solve();
    if(innercoll->isMixPrecision){
      innercoll->isInnerNow = false;
    }
  }
  
  else if(solver == BICG){
    if(innercoll->isMixPrecision){
      innercoll->isInnerNow = true;
    }
    bicg<T> innerSolver(innercoll, ibvec, ixvec, true);
    result = innerSolver.solve();
    if(innercoll->isMixPrecision){
      innercoll->isInnerNow = false;
    }
  }
  
  else if(solver == GMRES){
    if(innercoll->isMixPrecision){
      innercoll->isInnerNow = true;
    }
    gmres<T> innerSolver(innercoll, ibvec, ixvec, true);
    result = innerSolver.solve();
    if(innercoll->isMixPrecision){
      innercoll->isInnerNow = false;
    }
  }
  
  else if(solver == KSKIPCG){
    if(innercoll->isMixPrecision){
      innercoll->isInnerNow = true;
    }
    kskipcg<T> innerSolver(innercoll, ibvec, ixvec, true);
    result = innerSolver.solve();
    if(innercoll->isMixPrecision){
      innercoll->isInnerNow = false;
    }
  }
  
  else if(solver == KSKIPCR){
    std::cout << RED << "Not implemented" << RESET << std::endl;
    result = 1;
  }
  
  else if(solver == KSKIPBICG){
    if(innercoll->isMixPrecision){
      innercoll->isInnerNow = true;
    }
    kskipBicg<T> innerSolver(innercoll, ibvec, ixvec, true);
    result = innerSolver.solve();
    if(innercoll->isMixPrecision){
      innercoll->isInnerNow = false;
    }
  }

  if(result != 2 && result !=0 ){
    std::cerr << RED << "inner result -> "<< result << RESET << std::endl;
    std::cerr << RED << "[X]Some Error in methods" << RESET << std::endl;
  }
}

#
#endif //INNERMETHODS_HPP_INCLUDED__


#ifndef TIMES_HPP_INCLUDED__
#define TIMES_HPP_INCLUDED__

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include "color.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <sys/time.h>

class times {
  public:
    double cu_dot_copy_time;
    double cu_dot_proc_time;
    double cu_dot_malloc_time;
    double cu_dot_reduce_time;

    double cu_MV_copy_time;
    double cu_MV_proc_time;
    double cu_MV_malloc_time;

    double cpu_dot_proc_time;
    double cpu_MV_proc_time;
    double cpu_all_malloc_time;

    std::chrono::system_clock::time_point c_start, c_end;
    double o_start, o_end;
    double e_start, e_end;

    times();
    ~times();

    std::string get_date_time();

  
    void start(){
      c_start = std::chrono::system_clock::now();
#ifdef _OPENMP
      o_start = omp_get_wtime();
#endif
      e_start = getEtime();
    }

    void end(){
      c_end = std::chrono::system_clock::now();
#ifdef _OPENMP
      o_end = omp_get_wtime();
#endif
      e_end = getEtime();
    }

    double getTime(){
      double tmp = std::chrono::duration_cast<std::chrono::nanoseconds>(c_end-c_start).count();
      tmp = tmp*1e-9;
      return tmp;
    }

    double getEtime(){
      struct timeval tv;
      gettimeofday(&tv, NULL);
      return tv.tv_sec + (double)tv.tv_usec*1e-6;
    }

    double getTime_o(){
      return o_end-o_start;
    }

    double getTime_e(){
      return e_end-e_start;
    }

    double showTimeOnCPU(double total, bool hasGPU = false);

    void showTimeOnGPU(double total, double timeCPU);
    

};

#endif //TIMES_HPP_INCLUDED__


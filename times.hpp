#ifndef TIMES_HPP_INCLUDED__
#define TIMES_HPP_INCLUDED__

#include <chrono>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <sys/time.h>

class times {
  public:
    std::chrono::system_clock::time_point c_start, c_end;
    double o_start, o_end;
    double e_start, e_end;

    std::string get_date_time(){
      struct tm *date;
      time_t now;
      int month, day;
      int hour, minute, second;
      std::string date_time;

      time(&now);
      date = localtime(&now);

      month = date->tm_mon + 1;
      day = date->tm_mday;
      hour = date->tm_hour;
      minute = date->tm_min;
      second = date->tm_sec;

      date_time=std::to_string(month)+"-"+std::to_string(day)+"-"+std::to_string(hour)+"_"+std::to_string(minute)+"_"+std::to_string(second);

      return date_time;
    }

    double getEtime(){
      struct timeval tv;
      gettimeofday(&tv, NULL);
      return tv.tv_sec + (double)tv.tv_usec*1e-6;
    }
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
    double getTime_o(){
      return o_end-o_start;
    }
    double getTime_e(){
      return e_end-e_start;
    }

};

#endif //TIMES_HPP_INCLUDED__


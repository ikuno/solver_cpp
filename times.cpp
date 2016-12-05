#include "times.hpp"

times::times(){
  cu_dot_copy_time=0;
  cu_dot_proc_time=0;
  cu_dot_malloc_time=0;
  cu_dot_reduce_time=0;

  cu_MV_copy_time=0;
  cu_MV_proc_time=0;
  cu_MV_malloc_time=0;

  cpu_dot_proc_time=0;
  cpu_MV_proc_time=0;
  cpu_all_malloc_time=0;
}

times::~times(){

}
std::string times::get_date_time(){
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

double times::showTimeOnCPU(double total, bool hasGPU){
  double all = cpu_dot_proc_time + cpu_MV_proc_time + cpu_all_malloc_time;
  std::cout << CYAN << "Execution time on CPU" << RESET << std::endl;
  std::cout << "\tDot process time = " << std::setprecision(6) << cpu_dot_proc_time << ", " << std::setprecision(2) << cpu_dot_proc_time/total*100 << "%" << std::endl;
  std::cout << "\tMV process time  = " << std::setprecision(6) << cpu_MV_proc_time <<  ", " << std::setprecision(2) << cpu_MV_proc_time/total*100 << "%" << std::endl;
  if(!hasGPU){
    std::cout << "\tother time       = " << std::setprecision(6) << total-all <<  ", " << std::setprecision(2) << (total-all)/total*100 << "%" << std::endl;
    return (-1.0);
  }else{
    return all;
  }
}

void times::showTimeOnGPU(double total, double timeCPU){
  double dot_t = cu_dot_copy_time + cu_dot_proc_time + cu_dot_malloc_time + cu_dot_reduce_time;
  double mv_t = cu_MV_copy_time + cu_MV_proc_time + cu_MV_malloc_time;
  double all = dot_t + mv_t;
  std::cout << CYAN << "Execution time on GPU" << RESET << std::endl;
  std::cout << "\tDot malloc time  = " << std::setprecision(6) << cu_dot_malloc_time << ", " << std::setprecision(2) << cu_dot_malloc_time/total*100 << "%" << std::endl;
  std::cout << "\tDot copy time    = " << std::setprecision(6) << cu_dot_copy_time << ", " << std::setprecision(2) << cu_dot_copy_time/total*100 << "%" << std::endl;
  std::cout << "\tDot process time = " << std::setprecision(6) << cu_dot_proc_time << ", " << std::setprecision(2) << cu_dot_proc_time/total*100 << "%" << std::endl;
  std::cout << "\tDot reduce time  = " << std::setprecision(6) << cu_dot_reduce_time << ", " << std::setprecision(2) << cu_dot_reduce_time/total*100 << "%" << std::endl;
  std::cout << "\tMV malloc time   = " << std::setprecision(6) << cu_MV_malloc_time << ", " << std::setprecision(2) << cu_MV_malloc_time/total*100 << "%" << std::endl;
  std::cout << "\tMV copy time     = " << std::setprecision(6) << cu_MV_copy_time << ", " << std::setprecision(2) << cu_MV_copy_time/total*100 << "%" << std::endl;
  std::cout << "\tMV process time  = " << std::setprecision(6) << cu_MV_proc_time << ", " << std::setprecision(2) << cu_MV_proc_time/total*100 << "%" << std::endl;
  std::cout << "\tother time       = " << std::setprecision(6) << total-all-timeCPU <<  ", " << std::setprecision(2) << (total-all-timeCPU)/total*100 << "%" << std::endl;
}

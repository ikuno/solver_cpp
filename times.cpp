#include "times.hpp"

times::times(){
  mv_time = 0.0;
  dot_time = 0.0;
  memset_time = 0.0;
  h2d_time = 0.0;
  d2h_time = 0.0;

  cpu_mv_time = 0.0;
  cpu_dot_time = 0.0;

  // reg_time = 0.0;
  // unreg_time = 0.0;
  cp_time = 0.0;
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
  double all = cpu_mv_time + cpu_dot_time;
  std::cout << CYAN << "Execution time on CPU" << RESET << std::endl;
  std::cout << "\tDot process time = " << std::setprecision(6) << cpu_dot_time << ", " << std::setprecision(2) << cpu_dot_time/total*100 << "%" << std::endl;
  std::cout << "\tMV process time  = " << std::setprecision(6) << cpu_mv_time <<  ", " << std::setprecision(2) << cpu_mv_time/total*100 << "%" << std::endl;
  if(!hasGPU){
    std::cout << "\tother time     = " << std::setprecision(6) << total-all <<  ", " << std::setprecision(2) << (total-all)/total*100 << "%" << std::endl;
    return (-1.0);
  }else{
    return all;
  }
}

void times::showTimeOnGPU(double total, double timeCPU){
  double all = mv_time + dot_time;
  // double inall = h2d_time + d2h_time + memset_time;
  std::cout << CYAN << "Execution time on GPU" << RESET << std::endl;
  std::cout << "\tDot malloc time  = " << std::setprecision(6) << dot_time << ", " << std::setprecision(2) << dot_time/total*100 << "%" << std::endl;
  std::cout << "\tMV malloc time   = " << std::setprecision(6) << mv_time << ", " << std::setprecision(2) << mv_time/total*100 << "%" << std::endl;
  std::cout << "\t  H2D time       = " << std::setprecision(6) << h2d_time << ", " << std::setprecision(2) << h2d_time/total*100 << "%" << std::endl;
  std::cout << "\t  D2H time       = " << std::setprecision(6) << d2h_time << ", " << std::setprecision(2) << d2h_time/total*100 << "%" << std::endl;
  std::cout << "\t  Memset time    = " << std::setprecision(6) << memset_time << ", " << std::setprecision(2) << memset_time/total*100 << "%" << std::endl;
  std::cout << "\tother time       = " << std::setprecision(6) << total-all-timeCPU <<  ", " << std::setprecision(2) << (total-all-timeCPU)/total*100 << "%" << std::endl;

  // std::cout << "\t  Register time    = " << std::setprecision(6) << reg_time << ", " << std::setprecision(2) << reg_time/total*100 << "%" << std::endl;
  // std::cout << "\t  Unregister time    = " << std::setprecision(6) << unreg_time << ", " << std::setprecision(2) << unreg_time/total*100 << "%" << std::endl;
  std::cout << "\t  Copy time    = " << std::setprecision(6) << cp_time << ", " << std::setprecision(2) << cp_time/total*100 << "%" << std::endl;
}

#include "PID.h"

//using namespace std;

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
  this -> Kp = Kp;
  this -> Ki = Ki;
  this -> Kd = Kd;
  this -> p_error = 0;
  this -> i_error = 0;
  this -> d_error = 0;
}

void PID::UpdateError(double cte) {
  //Proportional Error Calculation
  p_error = cte;

  //Integral Error Calculation
  i_error += cte;

  //Derivative Error Calculation
  d_error = cte - p_error;
}

double PID::TotalError() {
  // Total Error Calculation
  return -(p_error * Kp) - (i_error * Ki) - (d_error * Kd);
}

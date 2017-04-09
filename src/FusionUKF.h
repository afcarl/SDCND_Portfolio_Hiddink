#ifndef FusionUKF_H
#define FusionUKF_H

#include "KalmanFilter.h"
#include "tools.h"
#include <iostream>

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

class FusionUKF {
public:

  VectorXd x_;
  MatrixXd P_;

  KalmanFilter kf_;

  FusionUKF();
  virtual ~FusionUKF();

  void ProcessMeasurement(MeasurementPackage meas_package);

private:

  bool is_initialized_;

  // Previous measurement
  long previous_timestamp_;
  MeasurementPackage previous_measurement_;

  bool use_laser_;
  bool use_radar_;

};

#endif /* FusionUKF_H */

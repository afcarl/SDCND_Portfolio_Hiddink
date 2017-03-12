#ifndef FusionEKF_H_
#define FusionEKF_H_

#include "measurement_package.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "kalman_filter.h"
#include "tools.h"
#include "Eigen/Dense"

class FusionEKF {
public:

  // Constructor
  FusionEKF();

  // Destructor
  virtual ~FusionEKF();

  // Run the Kalman Filter flow through here
  void ProcessMeasurement(const MeasurementPackage &measurement_pack);

  // Object used to perform Update and Prediction calculations
  KalmanFilter ekf_;

private:

  // Check for first measurement
  bool is_initialized_;

  // Previous measurement timestamp
  long previous_timestamp_;

  // Object used to perform Jacobian and RSME calculations
  Tools tools;

  MatrixXd R_laser_;
  MatrixXd R_radar_;
  MatrixXd H_laser_;
  MatrixXd Hj_;
};

#endif /* FusionEKF_H_ */

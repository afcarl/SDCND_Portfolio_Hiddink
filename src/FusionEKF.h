#ifndef FusionEKF_H_
#define FusionEKF_H_

#include "measurement_package.h"
#include <vector>
#include <string>
#include <fstream>
#include "kalman_filter.h"
#include "tools.h"

class FusionEKF {
public:

  FusionEKF();
  virtual ~FusionEKF();

  void ProcessMeasurement(const MeasurementPackage &measurement_pack);

  KalmanFilter ekf_;

private:

  bool is_initialized_;
  long previous_timestamp_;

  Tools tools;

  MatrixXd R_laser_;
  MatrixXd R_radar_;
  MatrixXd H_laser_;
  MatrixXd Hj_;
};

#endif /* FusionEKF_H_ */

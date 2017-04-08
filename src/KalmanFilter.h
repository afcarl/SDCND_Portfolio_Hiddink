#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include "FusionUKF.h"
#include "Tools.h"
#include <iostream>
#include "Eigen/Dense"
#include <vector>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class KalmanFilter {
public:

  KalmanFilter();

  virtual ~KalmanFilter();

  void Prediction(double delta_t);

  void UpdateLidar(MeasurementPackage meas_package);

  void UpdateRadar(MeasurementPackage meas_package);

private:
  int n_aug_;
  int n_sigma_;

  VectorXd x_;
  MatrixXd P_;

  MatrixXd Xsig_pred_;
  MatrixXd Xsig_aug_;

  Tools tools_;

};

#endif /* KALMAN_FILTER_H_ */

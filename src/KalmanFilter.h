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

  int n_z_;

  KalmanFilter();

  virtual ~KalmanFilter();

  void Init();

  void Prediction(double delta_t);

  void UpdateLidar(MeasurementPackage meas_package);

  void UpdateRadar(MeasurementPackage meas_package);

};

#endif /* KALMAN_FILTER_H_ */

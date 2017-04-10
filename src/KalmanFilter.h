#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include "tools.h"
#include "measurement_package.h"
#include "ground_truth_package.h"

#include <iostream>
#include "Eigen/Dense"
#include <vector>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class KalmanFilter {
public:

  Tools tools_;
  MeasurementPackage meas_package;

  VectorXd weights_;

  int n_x_;
  int n_aug_;
  int n_sigma_;
  int n_z_;

  VectorXd x_;
  MatrixXd P_;

  MatrixXd Xsig_pred_;
  MatrixXd Xsig_aug_;

  double NIS_lidar_;
  double NIS_radar_;

  KalmanFilter();

  virtual ~KalmanFilter();

  void Init();

  void Prediction(double delta_t);

  void UpdateLidar(MeasurementPackage meas_package, VectorXd &x,
                   MatrixXd &P, const MatrixXd &Xsig_pred,
                   const VectorXd &z_pred, const MatrixXd &S,
                   const MatrixXd &Zsig);

  void UpdateRadar(MeasurementPackage meas_package, VectorXd &x,
                   MatrixXd &P, const MatrixXd &Xsig_pred,
                   const VectorXd &z_pred, const MatrixXd &S,
                   const MatrixXd &Zsig);

};

#endif /* KALMAN_FILTER_H_ */

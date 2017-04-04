#include "FusionUKF.h"
#include "tools.h"
#include <iostream>

using namespace std;

FusionUKF::FusionUKF() {

  is_initialized_ = false;

  use_laser_ = true;
  use_radar_ = true;

  x_ = VectorXd(5);
  P_ = MatrixXd(5, 5);

  std_a_ = 0.2;
  std_yawdd_ = 0.2;

  std_laspx_ = 0.15;
  std_laspy_ = 0.15;

  std_radr_ = 0.3;
  std_radphi_ = 0.03;
  std_radrd_ = 0.3;

  Xsig_pred_ = MatrixXd(11, 5);
  time_us_ = 0;

  VectorXd weights_ = VectorXd(0);

  n_x_ = 5;
  n_aug_ = 7;
  n_z_ = 3;

  lambda_ = 3 - n_x;

  NIS_radar_ = 0;
  NIS_laser_ = 0;

}

FusionUKF::~FusionUKF() {}

void FusionUKF::ProcessMeasurement(MeasurementPackage meas_package) {

  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_ == false) {
    return;
  }
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_ == false) {
    return;
  }

  /*************************
   *   Initialization
   ************************/
  if (!is_initialized_) {
    previous_timestamp_ = meas_package.timestamp_;

    // Initialize state vector x
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      x << rho * cos(phi), rho * sin(phi), 0, 0, 0;
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    } else {
      cout << "Invalid sensor type." << endl;
    }

    // Initialize state covariance matrix P
    MatrixXd P = P_;
    P << 0.0043, -0.0013, 0.0030, -0.0022, -0.0020,
        -0.0013,  0.0077, 0.0011,  0.0071,  0.0060,
         0.0030,  0.0011, 0.0054,  0.0007,  0.0008,
        -0.0022,  0.0071, 0.0007,  0.0098,  0.0100,
        -0.0020,  0.0060, 0.0008,  0.0100,  0.0123;

    is_initialized_ = true;
    return;
  }

  /*************************
   *   Prediction
   ************************/
   double dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;

   while (dt > 0.1) {
     KalmanFilter::Prediction(0.05);
     dt -= 0.05;
   }

   KalmanFilter::Prediction(dt);

   previous_timestamp_ = meas_package.timestamp_;
   previous_measurement_ = meas_package;

   /*************************
    *   Update
    ************************/
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR and use_radar_) {
      KalmanFilter::UpdateRadar(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER and use_laser_) {
      KalmanFilter::UpdateLidar(meas_package);
    }
}

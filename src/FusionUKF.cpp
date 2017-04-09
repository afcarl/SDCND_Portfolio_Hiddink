#include "FusionUKF.h"

using namespace std;

FusionUKF::FusionUKF() {

  is_initialized_ = false;

  previous_timestamp_ = 0;

  use_laser_ = true;
  use_radar_ = true;

  x_ = VectorXd(5);
  P_ = MatrixXd(5, 5);

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
    VectorXd x = x_;
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
     kf_.Prediction(0.05);
     dt -= 0.05;
   }

   kf_.Prediction(dt);

   previous_timestamp_ = meas_package.timestamp_;
   previous_measurement_ = meas_package;

   /*************************
    *   Update
    ************************/
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR and use_radar_) {
      kf_.UpdateRadar(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER and use_laser_) {
      kf_.UpdateLidar(meas_package);
    }
}

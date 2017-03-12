#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

// Constructor
FusionEKF::FusionEKF() {

  // Initialize variables
  is_initialized_ = false;
  previous_timestamp_ = 0;

  // Initialize matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  // Finish initialization
  ekf_.x_ = VectorXd(4);
  ekf_.x_ << 1, 1, 1, 1;

  // State covariance matrix P
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;

  // Measurement covariance
  ekf_.R_ = MatrixXd(2, 2);
  ekf_.R_ << 0.0225, 0, 0, 0.0225;

  // Measurement matrix
  ekf_.H_ = MatrixXd(2, 4);
  ekf_.H_ << 1, 0, 0, 0,
             0, 1, 0, 0;

  // Initial transition matrix F
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;

  noise_ax = 5;
  noise_ay = 5;

}

// Destructor
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {

    // First measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {

      // Convert from polar to cartesian coordinates
      float rho = measurement_pack.raw_measurements_(0);
      float phi = measurement_pack.raw_measurements_(1);
      float p1_meas = rho * cos(phi);
      float ps_meas = rho * sin(phi);

      // Initialize state
      ekf_.x_ << p1_meas, ps_meas, 0, 0;
      previous_timestamp_ = measurement_pack.timestamp_;

    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {

      // Initialize state
  		ekf_.x_ << measurement_pack.raw_measurements_[0],
      measurement_pack.raw_measurements_[1], 0, 0;
      previous_timestamp_ = measurement_pack.timestamp_;

    }

    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  cout << dt << endl;

  // Update the state transition matrix, F, according to new elapsed time
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // Update the process noise covariance matrix, Q
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << pow(dt, 4) * noise_ax / 4, 0, pow(dt, 3) * noise_ax / 2 , 0,
             0, pow(dt, 4) * noise_ay / 4, 0, pow(dt, 3) * noise_ay / 2,
             pow(dt, 3) * noise_ax / 2, 0, pow(dt, 2) * noise_ax / 1, 0,
             0, pow(dt, 3) * noise_ay / 2, 0, pow(dt, 2) * noise_ay / 1;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  // Use the sensor type to perform the update step.
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {

    // Update Radar state and covariance matrices
    ekf_.Update(measurement_pack.raw_measurements_);

  } else {

    // // Update Laser state and covariance matrices
    ekf_.Update(measurement_pack.raw_measurements_);
    
  }

  // Print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}

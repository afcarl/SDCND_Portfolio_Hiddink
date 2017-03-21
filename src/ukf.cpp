#include <iostream>
#include "ukf.h"

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  ///* predicted sigma points matrix
  Xsig_pred_ = MatrixXd(11, 5);

  ///* time when the state is true, in us
  time_us_ = 0;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* State dimension
  n_x_ = 5;

  ///* Augmented state dimension
  n_aug_ = 7;

  ///* Sigma point spreading parameter
  lambda_ = 3 - n_x;

  ///* the current NIS for radar
  double NIS_radar_;

  ///* the current NIS for laser
  double NIS_laser_;

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  MatrixXd Xsig_pred = Xsig_pred_;
  ukf.GenerateSigmaPoints(&Xsig_pred);

  std:cout << "Xsig = " << std::endl << Xsig << std::endl;

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}

/**
 * GenerateSigmaPoints outputs a matrix of sigma points based on the state and state covariance matrices
 * @param Xsig_out The matrix of sigma points
 */
void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) {

  // Set state dimension
  int n_x = n_x_;

  // Define spreading parameter
  lambda = lambda_;

  // Set state matrix
  VectorXd x = x_;

  // Set state covariance matrix
  MatrixXd P = P_;

  // Create square root of state covariance matrix
  MatrixXd A = P.llt().matrixL();

  // Calculate sigma points
  Xsig_pred.col(0) = x;

  for (int i = 0; i < n_x; i++) {
    Xsig_pred.col(i + 1) = x + sqrt(lambda + n_x) * A.col(i);
    Xsig_pred.col(i + 1 + n_x) = x - sqrt(lambda + n_x) * A.col(i);
  }

  // Print result for debugging
  std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

  // Write result to function parameter
  *Xsig_out = Xsig_pred;

}

/**
 * AugmentedSigmaPoints outputs a matrix of sigma points based on augmented state and augmented state covariance matrices
 * @param Xsig_out The matrix of sigma points
 */
void AugmentedSigmaPoints(MatrixXd* Xsig_out) {

  // Set state dimension
  int n_x = n_x_; // 5

  // Set augmented dimension
  int n_aug = n_aug_; // 7

  // Set process noise standard deviation longitudinal acceleration in m/s^2
  double std_a = std_a_; // 0.2

  // Set process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd = std_yawdd_; // 0.2

  // Set augmented spreading parameter
  lambda_ = 3 - n_aug;
  double lambda = lambda_; // 3 - n_aug

  // Set state matrix
  VectorXd x = x_;

  // Set state covariance matrix
  MatrixXd P = P_;

  // Create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  // Create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  // Create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);

  // Create augmented mean state
  x_aug.head(5) = x;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // Create augmented state covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P;
  P_aug(5, 5) = std_a * std_a;
  P_aug(6, 6) = std_yawdd * std_yawdd;

  // Create square root matrix from augmented state covariance matrix
  MatrixXd L = P_aug.llt().matrixL();

  // Calculate augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug; i++) {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda + n_aug) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug) = x_aug - sqrt(lambda + n_aug) * L.col(i);
  }

  // Print result for debugging
  std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;

  // Write result to function parameter
  *Xsig_out = Xsig_aug;

}

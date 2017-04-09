#include "KalmanFilter.h"
#include "FusionUKF.h"

KalmanFilter::KalmanFilter() {


}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init() {
  // Initialize variables
  int n_z_ = meas_package.raw_measurements_.rows();

  int n_x_ = 5;
  int n_aug_ = 7;

  int n_sigma_ = 2 * n_aug_ + 1;

}

/**
  * Prediction Predicts sigma points, the state, and the state covariance
  * matrix
  * @param delta_t Time between k and k+1 in s
  */
void KalmanFilter::Prediction(double delta_t) {

  MatrixXd Xsig_aug(n_aug_, n_sigma_);

  tools_.GenerateAugmentedSigmaPoints(x_, P_, Xsig_aug);
  tools_.SigmaPointPrediction(delta_t, Xsig_aug, Xsig_pred_);
  tools_.PredictMeanAndCovariance(Xsig_pred_, x_, P_);

}

/**
  * Updates the state and the state covariance matrix using a laser measurement
  * @param meas_package The measurement at k+1
  */
void KalmanFilter::UpdateLidar(MeasurementPackage meas_package) {

  VectorXd z_pred(n_z_);
  MatrixXd S(n_z_, n_z_);
  MatrixXd Zsig(n_z_, n_sigma_)

  KalmanFilter::Prediction()

}

/**
  * Updates the state and the state covariance matrix using a radar measurement
  * @param meas_package The measurement at k+1
  */
void KalmanFilter::UpdateRadar(MeasurementPackage meas_package) {

  // Define spreading parameter
  double lambda = 3 - n_aug_;

  // Set vector for weights
  weights_ = VectorXd(2*n_aug_+1);
  double weight_0 = lambda / (lambda + n_aug_);
  weights_(0) = weight_0;
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double weight = 0.5 / (n_aug_ + lambda);
    weights_(i) = weight;
  }

  MatrixXd Xsig_pred = Xsig_pred_;
  Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Vector for predicted state mean
  VectorXd x = x_;
  x = VectorXd(n_x_);

  // Matrix for predicted state variance
  MatrixXd P = P_;
  P = MatrixXd(n_x_, n_x_) ;

  // Matrix with sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

  // Vector for mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);

  // Matrix for predicted measurement covariance
  MatrixXd S = MatrixXd(n_z_, n_z_);

  // Vector for incoming radar measurement
  VectorXd z = VectorXd(n_z_);

  // Matrix for cross-correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_);

  // Calculate cross-correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) { // for all 2n + 1 sigma points

    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // Angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    // State difference
    VectorXd x_diff = Xsig_pred.col(i) - x;
    // Angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    Tc = Tc + weights(i) * x_diff * z_diff.transpose();

  }

  // Calculate Kalman gain, K
  MatrixXd K = Tc * S.inverse();

  // Update state mean (x) and state covariance (P) matrices
  x += K * z_diff;
  P -= K * S * K.transpose();

  // Print result
  std::cout << "Updated state x: " << std::endl << x << std::endl;
  std:cout << "Updated state covariance P: " << std::endl << P << std::endl;

  // Write result to function parameter
  *x_out = x;
  *P_out = P;
}

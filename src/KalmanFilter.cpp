#include "KalmanFilter.h"
#include "FusionUKF.h"

KalmanFilter::KalmanFilter() {

  n_x_ = 5;
  n_aug_ = 7;

  n_sigma_ = 2 * n_aug_ + 1;
  n_z_ = meas_package.raw_measurements_.rows();

}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init() {}

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
void UpdateLidar(MeasurementPackage meas_package, VectorXd &x,
                 MatrixXd &P, const MatrixXd &Xsig_pred,
                 const VectorXd &z_pred, const MatrixXd &S,
                 const MatrixXd &Zsig) {

  int n_z = meas_package.raw_measurements_.rows();

  VectorXd z_pred(n_z);
  MatrixXd S(n_z, n_z);
  MatrixXd Zsig(n_z, n_sigma_);

  /**************
   * PREDICTION
   *************/

  // Transform sigma points into measurement space
  for (int i = 0; i < n_sigma_; i++) {
    double p_x = Xsig_pred(0, i);
    double p_y = Xsig_pred(1, i);

    Zsig(0, i) = p_x;
    Zsig(1, i) = p_y;
  }

  // Mean predicted measurement
  z_pred.fill(0.0);
  for (int i = 0; i < n_sigma_; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // Measurement covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < n_sigma_; i++) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << tools_.std_laspx_ * tools_.std_laspx_, 0, 0, tools_.std_laspy_ * tools_.std_laspy_;

  S = S + R;

  /**************
   * UPDATE
   *************/

   n_z = z_pred.rows();
   n_x = x.rows();

   // Create matrix for cross correlation Tc
   MatrixXd Tc = MatrixXd(n_x, n_z);

   // Calculate cross correlation matrix
   Tc.fill(0.0);
   for (int i = 0; i < n_sigma_; i++) {

     // Residual
     VectorXd z_diff = Zsig.col(i) - z_pred;
     z_diff(1) = normalizeRadiansPiToMinusPi(z_diff(1));

     // State difference
     VectorXd x_diff = Xsig_pred.col(i) - x;
     x_diff(3) = normalizeRadiansPiToMinusPi(x_diff(3));

     Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
   }

   // Kalman gain K
   MatrixXd K = Tc * S.inverse();

   // Residual
   VectorXd z_diff = z - z_pred;

   z_diff(1) = normalizeRadiansPiToMinusPi(z_diff(1));

   // Update state mean and covariance matrix
   x = x + K * z_diff;
   P = P - K * S * K.transpose();

  // Normalized Innovation Squared (NIS) Measurement Gate/Threshold
  NIS_lidar_ = ((meas_package.raw_measurements_ - z_pred).transpose()) * S.inverse() * (meas_package.raw_measurements_ - z_pred);

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

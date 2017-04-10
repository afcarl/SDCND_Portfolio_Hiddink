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
   * PREDICT
   *************/

  // Set measurement dimension for lidar
  int n_z = z_pred.rows();

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

void UpdateRadar(MeasurementPackage meas_package, VectorXd &x,
                 MatrixXd &P, const MatrixXd &Xsig_pred,
                 const VectorXd &z_pred, const MatrixXd &S,
                 const MatrixXd &Zsig) {

  int n_z = meas_package.raw_measurements_.rows();

  VectorXd z_pred(n_z);
  MatrixXd S(n_z, n_z);
  // Create matrix for sigma points in measurement space
  MatrixXd Zsig(n_z, n_sigma_);

  /**************
   * PREDICT
   *************/

   // Set measurement dimension, radar needs r, phi, ro_dot
   int n_z = z_pred.rows();

   // Transform sigma points into measurement space
   for (int i = 0; i < n_sigma_; i++) {

     // Extract values for better readability
     double px = Xsig_pred(0, i);
     double py = Xsig_pred(1, i);
     double v = Xsig_pred(2, i);
     double yaw = Xsig_pred(3, i);

     double v1 = cos(yaw) * v;
     double v2 = sin(yaw) * v;

     // Measurement model
     Zsig(0, i) = sqrt((px * px) + (py * py)); // rho
     Zsig(1, i) = atan2(py, px); // phi
     if (Zsig(0, i) != 0) {
       Zsig(2, i) = (((px * v1) + (py * v2)) / (Zsig(0, i))); // rho_dot
     } else {
       Zsig(2, i) = 0; // rho_dot
     }
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
     z_diff(1) = normalizeRadiansPiToMinusPi(z_diff(1));
     S = S + weights_(i) * z_diff * z_diff.transpose();
   }

   // Add measurement noise covariance matrix to the measurement covariance matrix
   MatrixXd R = MatrixXd(n_z, n_z);
   R << tools_.std_radr_ * tools_.std_radr_, 0, 0,
        0, tools_.std_radphi_ * tools_.std_radphi_, 0,
        0, 0, tools_.std_radrd_ * tools_.std_radrd_;

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

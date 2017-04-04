#include <iostream>
#include "tools.h"

class Tools {

  Tools() {

    n_aug_ = 7;
    lambda_ = 3 - n_aug_;
    n_sigma_ = 2 * n_aug_ + 1;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 1.9;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.6;

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

  }

  virtual ~Tools() {}

  VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth) {

  }

  MatrixXd GenerateSigmaPoints() {

  }

  void GenerateAugmentedSigmaPoints(const VectorXd &x, const MatrixXd &P, MatrixXd &Xsig_aug) {

    // Set state dimension
    int n_x = x.rows();

    // Create augmented mean vector
    VectorXd x_aug(KalmanFilter.n_aug_);

    // Create augmented state covariance
    MatrixXd P_aug(KalmanFilter.n_aug_, KalmanFilter.n_aug_);

    // Create augmented mean state - adding zero for the extra noise values
    x_aug << x, 0, 0;

    // Create augmented covariance matrix - add process noise values
    P_aug.fill(0.0);
    P_aug.topLeftCorner(n_x, n_x) = P;
    P_aug(n_aug_-2, n_aug_-2) = std_a_ * std_a_;
    P_aug(n_aug_-1, n_aug_-1) = std_yawdd_ * std_yawdd_;

    // MatrixXd L = P_aug.llt().matrixL(); <--- CODE FROM LECTURE NOT STABLE
    // Take matrix square root:
    // 1. Compute Cholesky decomposition of P_aug
    Eigen::LLT<MatrixXd> lltOfPaug(P_aug);
    if (lltOfPaug.info() == Eigen::NumericalIssue) {
    // If decomposition fails, there are numerical issues
    std::cout << "LLT failed!" << std::endl;
    throw std::range_error("LLT failed");
    }
    // 2. Get the lower triangle of the matrix
    MatrixXd L = lltOfPaug.matrixL();

    // Create augmented sigma points
    Xsig_aug.col(0) = x_aug;
    for (int i = 0; i < n_aug_; i++) {
      Xsig_aug.col(i+1) = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
      Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
    }
  }

  void SigmaPointPrediction(double delta_t, const MatrixXd &Xsig_aug, MatrixXd &Xsig_pred) {

    for (int i = 0; i < n_sigma_; i++) {

      // Extract values for better readability
      double p_x = Xsig_aug(0, i);
      double p_y = Xsig_aug(1, i);
      double v = Xsig_aug(2, i);
      double yaw = Xsig_aug(3, i);
      double yawd = Xsig_aug(4, i);
      double nu_a = Xsig_aug(5, i);
      double nu_yawdd = Xsig_aug(6, i);

      // Predicted state values
      double px_p, py_p;

      // Avoid division by zero
      if (fabs(yawd) > 0.001) {
        px_p = p_x + v / yawd * (sin(yawd * delta_t) - sin(yaw));
        py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
      } else {
        px_p = p_x + v * delta_t * cos(yaw);
        py_p = p_y + v * delta_t * sin(yaw);
      }

      double v_p = v;
      double yaw_p = yaw + yawd * delta_t;
      double yawd_p = yawd;

      // Add noise values
      px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
      py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
      v_p = v_p + nu_a * delta_t;

      yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
      yawd_p = yawd_p + nu_yawdd * delta_t;

      // Write predicted sigma point into right column
      Xsig_pred(0, i) = px_p;
      Xsig_pred(1, i) = py_p;
      Xsig_pred(2, i) = v_p;
      Xsig_pred(3, i) = yaw_p;
      Xsig_pred(4, i) = yawd_p;
    }
  }

}

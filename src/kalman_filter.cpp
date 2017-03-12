#include "kalman_filter.h"

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &Q_in, MatrixXd &H_in, MatrixXd &R_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  Q_ = Q_in;
  H_ = H_in;
  R_ = R_in;
}

void KalmanFilter::Predict() {

  // Predict the state
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;

}

void KalmanFilter::Update(const VectorXd &z) {

  // Update the state using Kalman Filter equations
  VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd S = H_ * P_ * H_.transpose() + R_;
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * S.inverse();

  // New estimate
  x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

  // Pre-compute a set of terms to avoid repeated calculations
  float px = x_[0];
  float py = x_[1];
  float vx = x_[2];
  float vy = x_[3];
  float c1 = px * px + py * py;
  float c2 = px * vx;
  float c3 = py * vy;

  float range = sqrt(c1)
  float bearing = atan(py / px);
  float rangeRate = (c2 + c3) / range;

  VectorXd z_pred(3);
  z_pred << range, bearing, rangeRate;

  // Update state by implementing EKF equations
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd PHt = P_ * H_.transpose();
  MatrixXd K = PHt * S.inverse();

  // New estimate
  x_ = x_ + (K * y);
  long x_size = x_size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

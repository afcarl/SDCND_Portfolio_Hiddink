#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Tools {
public:

  Tools();

  virtual ~Tools();

  VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

  void GenerateAugmentedSigmaPoints(const VectorXd &x, const MatrixXd &P, MatrixXd &Xsig_aug);

  void SigmaPointPrediction(double delta_t, const MatrixXd &Xsig_aug, MatrixXd &Xsig_pred);

  void PredictMeanAndCovariance(const MatrixXd &Xsig_pred, VectorXd &x, MatrixXd &P);

  int n_x_;
  int n_aug_;
  double lambda_;
  int n_sigma_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_;

};

#endif /* TOOLS_H_ */

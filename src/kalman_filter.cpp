#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
static const int PX_VECTOR_INDEX  = 0;
static const int PY_VECTOR_INDEX  = 1;
static const int VX_VECTOR_INDEX  = 2;
static const int VY_VECTOR_INDEX  = 3;
static const int PHI_VECTOR_INDEX = 1;
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  VectorXd z_pred = H_ * x_;
  VectorXd y      = z - z_pred;

  DoUpdate(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
  float px = x_(PX_VECTOR_INDEX);
  float py = x_(PY_VECTOR_INDEX);
  float vx = x_(VX_VECTOR_INDEX);
  float vy = x_(VY_VECTOR_INDEX);

  double rho     = sqrt(px * px + py * py);
  double phi     = atan2(py, px);
  double rho_dot = (px * vx + py * vy) / std::max(rho, 0.00001);

  VectorXd z_pred(3);

  z_pred << rho, phi, rho_dot;

  VectorXd y = z - z_pred;

  while (y(PHI_VECTOR_INDEX) < -M_PI || y(PHI_VECTOR_INDEX) > M_PI)
  {
    if (y(PHI_VECTOR_INDEX) < -M_PI)
      y(PHI_VECTOR_INDEX) += 2 * M_PI;
    else
      y(PHI_VECTOR_INDEX) -= 2 * M_PI;
  }

  DoUpdate(y);

}

void KalmanFilter::DoUpdate(const Eigen::VectorXd& y)
{
  MatrixXd Ht  = H_.transpose();
  MatrixXd S   = H_ * P_ * Ht + R_;
  MatrixXd PHt = P_ * Ht;
  MatrixXd K   = PHt * S.inverse();

  //new estimate
  x_ = x_ + (K * y);
  P_ = (MatrixXd::Identity(x_.size(), x_.size()) - K * H_) * P_;
}

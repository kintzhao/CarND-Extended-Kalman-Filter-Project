#include <iostream>
#include "tools.h"


static const int JACOBIAN_ROWS_NUMBER    = 3;
static const int JACOBIAN_COLUMNS_NUMBER = 4;
static const int PX_INDEX                = 0;
static const int PY_INDEX                = 1;
static const int VX_INDEX                = 2;
static const int VY_INDEX                = 3;
static const int RMSE_VECTOR_SIZE        = 4;

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
  const bool hasCorrectSize = estimations.size() > 0 && estimations.size() == ground_truth.size();
  VectorXd   rmse(RMSE_VECTOR_SIZE);

  rmse << 0,0,0,0;

  if(hasCorrectSize == false)
  {
    //TODO: Invalid estimation or ground_truth data.
    return rmse;
  }

  for(unsigned int i = 0; i < estimations.size(); ++i)
  {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse    += residual;
  }

  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
  MatrixXd Hj(JACOBIAN_ROWS_NUMBER,JACOBIAN_COLUMNS_NUMBER);

  const float px = x_state(PX_INDEX);
  const float py = x_state(PY_INDEX);
  const float vx = x_state(VX_INDEX);
  const float vy = x_state(VY_INDEX);
  const float c1 = px * px + py * py;
  const float c2 = sqrt(c1);
  const float c3 = (c1 * c2);

  if(fabs(c1) < 0.0001)
  {
    // TODO: Division by zero error.
    return Hj;
  }

  Hj <<                     (px / c2),                     (py / c2),       0,       0,
                           -(py / c1),                     (px / c1),       0,       0,
        py * (vx * py - vy * px) / c3, px * (px * vy - py * vx) / c3, px / c2, py / c2;

  return Hj;
}

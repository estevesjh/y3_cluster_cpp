#ifndef Y3_CLUSTER_B_SEL_CUH
#define Y3_CLUSTER_B_SEL_CUH

// Selection bias b_sel(theta | lob, ltr, zob) formulas for GPU.
//
// Costanzi-2026 formulas:
//   Delta_RND = P1 + b_eff * I2
//   delta_prj = (lob - ltr) / Delta_RND - 1
//   b_infty = b_eff * (1 + 0.13 * delta_prj)
//   b_zero = [(lob - ltr) - P1 - b_infty * I1] / J
//   b_sel(theta) = b_zero + (b_infty - b_zero) * sigmoid(theta)
//
// Sigmoid:
//   sigma(theta) = 1 / (1 + exp(-k * (theta - theta0)))
//   k = 2.5 / theta_lob,  theta0 = theta_lob / 2
//   theta_lob = R_lambda(lob) * (1 + zob) / chi_o

#include <cmath>

namespace y3_cuda {

  // R_lambda(lob) = (lob/100)^0.2 in cMpc/h
  __host__ __device__ inline double
  R_lambda(double lob)
  {
    return pow(lob / 100.0, 0.2);
  }

  // theta_lob = R_lambda(lob) * (1 + zob) / chi_o
  __host__ __device__ inline double
  theta_lob(double lob, double zob, double chi_o)
  {
    return R_lambda(lob) * (1.0 + zob) / chi_o;
  }

  // Costanzi-2026 sigmoid
  __host__ __device__ inline double
  b_sel_sigmoid(double theta, double theta_lob_val)
  {
    double const k = 2.5 / theta_lob_val;
    double const theta0 = 0.5 * theta_lob_val;
    return 1.0 / (1.0 + exp(-k * (theta - theta0)));
  }

  // Compute b_zero and b_infty (theta-independent asymptotes)
  __host__ __device__ inline void
  b_sel_asymptotes(double lob, double ltr,
                   double P1, double I1, double J, double b_eff,
                   double& b_zero, double& b_infty)
  {
    double const I2 = I1 + J;
    double const Delta_RND = P1 + b_eff * I2;

    if (fabs(Delta_RND) < 1e-30) {
      b_zero = b_eff;
      b_infty = b_eff;
      return;
    }

    double const delta_prj = (lob - ltr) / Delta_RND - 1.0;
    b_infty = b_eff * (1.0 + 0.13 * delta_prj);

    if (fabs(J) > 1e-12 * (fabs(I1) + fabs(I2))) {
      b_zero = ((lob - ltr) - P1 - b_infty * I1) / J;
    } else {
      b_zero = b_infty;
    }
  }

  // b_sel(theta) from asymptotes
  __host__ __device__ inline double
  b_sel_from_asymptotes(double theta, double b_zero, double b_infty, double theta_lob_val)
  {
    double const sig = b_sel_sigmoid(theta, theta_lob_val);
    return b_zero + (b_infty - b_zero) * sig;
  }

  // Full b_sel(theta | lob, ltr, zob)
  __host__ __device__ inline double
  b_sel(double theta, double lob, double ltr, double zob, double chi_o,
        double P1, double I1, double J, double b_eff)
  {
    double b_zero, b_infty;
    b_sel_asymptotes(lob, ltr, P1, I1, J, b_eff, b_zero, b_infty);
    double const th_lob = theta_lob(lob, zob, chi_o);
    return b_sel_from_asymptotes(theta, b_zero, b_infty, th_lob);
  }

}  // namespace y3_cuda

#endif

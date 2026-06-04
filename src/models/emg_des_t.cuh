#ifndef Y3_CLUSTER_CPP_EMG_DES_T_CUH
#define Y3_CLUSTER_CPP_EMG_DES_T_CUH

// EMG (Exponentially Modified Gaussian) kernel for P(lambda_ob | lambda_tr, z)
// This implements the Costanzi observation scatter model.
//
// The EMG PDF describes how observed richness lambda_ob is scattered from
// the true richness lambda_tr due to projection effects and measurement noise.
//
// Parameters (mu, sigma, tau, fprj) are functions of (lambda_tr, z) via
// the plob_ltr_params splines.
//
// References:
//   - Costanzi et al. 2019: https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.4779C
//   - src/modules/sel_function/sel_function.py (Python reference implementation)

#include "cosmosis/datablock/datablock.hh"
#include "common/cuda/Interp1D.cuh"
#include "utils/make_interp_1d.cuh"

#include <cmath>
#include <vector>

namespace y3_cuda {

// Constants
namespace emg_constants {
  __host__ __device__ constexpr double SQRT2 = 1.4142135623730951;
  __host__ __device__ constexpr double SQRT2PI = 2.5066282746310002;
  __host__ __device__ constexpr double SQRTPI = 1.7724538509055159;
  __host__ __device__ constexpr double SQRT2_INV = 0.7071067811865475;
  __host__ __device__ constexpr double SQRT_2_OVER_PI = 0.7978845608028654;
}

// erfcx(x) = exp(x^2) * erfc(x) - numerically stable complementary error function
// For large |x|, erfc(x) underflows but erfcx(x) remains finite.
__host__ __device__ inline double
erfcx_impl(double x)
{
  // For small |x|, direct computation is stable
  // For large |x|, use asymptotic expansion
  double ax = fabs(x);

  if (ax < 4.0) {
    // Direct computation: erfcx(x) = exp(x^2) * erfc(x)
    // This is stable for |x| < ~4
    return exp(x * x) * erfc(x);
  } else {
    // Asymptotic expansion for large x:
    // erfcx(x) ~ (1/sqrt(pi)) * (1/x) * (1 - 1/(2x^2) + 3/(4x^4) - ...)
    // For x < 0: erfcx(x) = 2*exp(x^2) - erfcx(-x)
    double x2 = ax * ax;
    double result = (1.0 / emg_constants::SQRTPI) / ax *
                    (1.0 - 0.5 / x2 + 0.75 / (x2 * x2));

    if (x < 0.0) {
      // erfcx(-|x|) = 2*exp(x^2) - erfcx(|x|)
      result = 2.0 * exp(x2) - result;
    }
    return result;
  }
}

// Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
__host__ __device__ inline double
phi_cdf(double x)
{
  return 0.5 * (1.0 + erf(x * emg_constants::SQRT2_INV));
}

// Standard normal PDF: phi(x) = exp(-x^2/2) / sqrt(2*pi)
__host__ __device__ inline double
phi_pdf(double x)
{
  return exp(-0.5 * x * x) / emg_constants::SQRT2PI;
}


class EMG_DES_t {
private:
  // Spline coefficients for EMG parameters as functions of z
  // mu(ltr, z) = a_mu(z) + b_mu(z) * ltr
  // sigma(ltr, z) = b_sig(z) * ltr^a_sig(z)
  // tau(ltr, z) = b_tau(z) / ltr^a_tau(z)
  // fprj(ltr, z) = min(1, b_fprj(z) / (1 + exp(-ltr))^a_fprj(z))

  quad::Interp1D a_mu_;
  quad::Interp1D b_mu_;
  quad::Interp1D a_sig_;
  quad::Interp1D b_sig_;
  quad::Interp1D a_tau_;
  quad::Interp1D b_tau_;
  quad::Interp1D a_fprj_;
  quad::Interp1D b_fprj_;

public:
  size_t get_device_mem_footprint()
  {
    size_t size = 0;
    size += a_mu_.get_device_mem_footprint();
    size += b_mu_.get_device_mem_footprint();
    size += a_sig_.get_device_mem_footprint();
    size += b_sig_.get_device_mem_footprint();
    size += a_tau_.get_device_mem_footprint();
    size += b_tau_.get_device_mem_footprint();
    size += a_fprj_.get_device_mem_footprint();
    size += b_fprj_.get_device_mem_footprint();
    return size;
  }

  // Constructor from datablock - reads plob_ltr_params section
  explicit EMG_DES_t(cosmosis::DataBlock& sample)
    : a_mu_(make_Interp1D(sample, "plob_ltr_params", "z", "a_mu"))
    , b_mu_(make_Interp1D(sample, "plob_ltr_params", "z", "b_mu"))
    , a_sig_(make_Interp1D(sample, "plob_ltr_params", "z", "a_sig"))
    , b_sig_(make_Interp1D(sample, "plob_ltr_params", "z", "b_sig"))
    , a_tau_(make_Interp1D(sample, "plob_ltr_params", "z", "a_tau"))
    , b_tau_(make_Interp1D(sample, "plob_ltr_params", "z", "b_tau"))
    , a_fprj_(make_Interp1D(sample, "plob_ltr_params", "z", "a_fprj"))
    , b_fprj_(make_Interp1D(sample, "plob_ltr_params", "z", "b_fprj"))
  {}

  // Compute EMG parameters at (lambda_tr, z)
  __host__ __device__ void
  get_params(double ltr, double z,
             double& mu, double& sigma, double& tau, double& fprj) const
  {
    // Clamp lambda_tr to avoid numerical issues
    double ltr_safe = fmax(ltr, 0.5);

    // Evaluate spline coefficients at z
    double a_mu_z = a_mu_.clamp(z);
    double b_mu_z = b_mu_.clamp(z);
    double a_sig_z = a_sig_.clamp(z);
    double b_sig_z = b_sig_.clamp(z);
    double a_tau_z = a_tau_.clamp(z);
    double b_tau_z = b_tau_.clamp(z);
    double a_fprj_z = a_fprj_.clamp(z);
    double b_fprj_z = b_fprj_.clamp(z);

    // Compute EMG parameters
    mu = a_mu_z + b_mu_z * ltr_safe;
    sigma = b_sig_z * pow(ltr_safe, a_sig_z);
    tau = b_tau_z / pow(ltr_safe, a_tau_z);

    // fprj with sigmoid-like denominator
    double denom = pow(1.0 + exp(-ltr_safe), a_fprj_z);
    fprj = fmin(1.0, b_fprj_z / fmax(denom, 1e-10));

    // Ensure positive values
    sigma = fmax(sigma, 1e-6);
    tau = fmax(tau, 1e-6);
    fprj = fmax(0.0, fmin(1.0, fprj));
  }

  // EMG CDF: F(lambda_ob | lambda_tr, z)
  // Matches sel_function.py::_cdf_lob
  __host__ __device__ double
  cdf(double lob, double ltr, double z) const
  {
    // Handle infinite bounds: CDF(+inf) = 1, CDF(-inf) = 0
    if (!isfinite(lob)) {
      return lob > 0.0 ? 1.0 : 0.0;
    }

    double mu, sigma, tau, fprj;
    get_params(ltr, z, mu, sigma, tau, fprj);

    double z_std = (lob - mu) / sigma;
    double u = (tau * sigma - z_std) * emg_constants::SQRT2_INV;

    // Compute tail term
    double exp_mz2 = exp(-0.5 * z_std * z_std);
    double abs_u = fabs(u);
    double tail_base = 0.5 * erfcx_impl(abs_u) * exp_mz2;

    double tail;
    if (u < 0.0) {
      // u < 0 branch
      double A = -tau * (lob - mu) + 0.5 * tau * tau * sigma * sigma;
      A = fmax(-700.0, fmin(700.0, A));  // Clip for numerical stability
      tail = exp(A) - tail_base;
    } else {
      tail = tail_base;
    }

    // CDF = Phi(z_std) - fprj * tail
    double result = phi_cdf(z_std) - fprj * tail;
    return fmax(0.0, fmin(1.0, result));
  }

  // EMG PDF: P(lambda_ob | lambda_tr, z)
  // Computed as derivative of CDF with respect to lambda_ob
  __host__ __device__ double
  operator()(double lob, double ltr, double z) const
  {
    double mu, sigma, tau, fprj;
    get_params(ltr, z, mu, sigma, tau, fprj);

    double z_std = (lob - mu) / sigma;
    double u = (tau * sigma - z_std) * emg_constants::SQRT2_INV;

    // Gaussian PDF term: (1/sigma) * phi(z_std)
    double gauss_pdf = phi_pdf(z_std) / sigma;

    // Compute d(tail)/d(lob) for the correction term
    double exp_mz2 = exp(-0.5 * z_std * z_std);
    double abs_u = fabs(u);
    double erfcx_u = erfcx_impl(abs_u);

    // d(erfcx)/dx = 2x * erfcx(x) - 2/sqrt(pi)
    // du/d(lob) = -1/(sigma * sqrt(2))
    // dz_std/d(lob) = 1/sigma

    double dtail_dlob;
    if (u >= 0.0) {
      // tail = 0.5 * erfcx(u) * exp(-z^2/2)
      // d(tail)/dz = 0.5 * exp(-z^2/2) * [erfcx'(u) * du/dz - z * erfcx(u)]
      // where erfcx'(u) = 2u * erfcx(u) - 2/sqrt(pi)
      // and du/dz = -1/sqrt(2)

      double erfcx_prime_u = 2.0 * u * erfcx_u - 2.0 / emg_constants::SQRTPI;
      double du_dz = -emg_constants::SQRT2_INV;
      double dtail_dz = 0.5 * exp_mz2 * (erfcx_prime_u * du_dz - z_std * erfcx_u);
      dtail_dlob = dtail_dz / sigma;
    } else {
      // tail = exp(A) - 0.5 * erfcx(|u|) * exp(-z^2/2)
      // where A = -tau*(lob - mu) + 0.5*(tau*sigma)^2
      // dA/d(lob) = -tau

      double A = -tau * (lob - mu) + 0.5 * tau * tau * sigma * sigma;
      A = fmax(-700.0, fmin(700.0, A));
      double exp_A = exp(A);

      // d(exp(A))/d(lob) = exp(A) * dA/d(lob) = -tau * exp(A)
      double dexpA_dlob = -tau * exp_A;

      // For the erfcx term with u < 0, we use |u| = -u
      // d(|u|)/d(lob) = -d(u)/d(lob) = 1/(sigma * sqrt(2))
      double erfcx_prime_abs_u = 2.0 * abs_u * erfcx_u - 2.0 / emg_constants::SQRTPI;
      double dabs_u_dz = emg_constants::SQRT2_INV;  // d|u|/dz when u < 0
      double dtail_base_dz = 0.5 * exp_mz2 * (erfcx_prime_abs_u * dabs_u_dz - z_std * erfcx_u);

      dtail_dlob = dexpA_dlob - dtail_base_dz / sigma;
    }

    // PDF = gauss_pdf - fprj * d(tail)/d(lob)
    double pdf = gauss_pdf - fprj * dtail_dlob;

    // Ensure non-negative
    return fmax(0.0, pdf);
  }

  // Convenience: PDF with explicit parameter output for debugging
  __host__ __device__ double
  pdf_with_params(double lob, double ltr, double z,
                  double& mu_out, double& sigma_out,
                  double& tau_out, double& fprj_out) const
  {
    get_params(ltr, z, mu_out, sigma_out, tau_out, fprj_out);
    return (*this)(lob, ltr, z);
  }
};

} // namespace y3_cuda

#endif

#ifndef Y3_CLUSTER_CPP_EMG_DES_T_CUH
#define Y3_CLUSTER_CPP_EMG_DES_T_CUH

#include "cosmosis/datablock/datablock.hh"
#include "common/cuda/Interp1D.cuh"
#include "utils/make_interp_1d.cuh"
#include "utils/primitives.cuh"

#include <cmath>  // exp, erfc, erf, pow, isfinite
#include <vector>

namespace y3_cuda {

namespace emg_constants {
  __host__ __device__ constexpr double SQRT2 = 1.4142135623730951;
  __host__ __device__ constexpr double SQRT2_INV = 0.7071067811865475;
}

class EMG_DES_t {
private:
  // Interpolated DES richness-kernel calibration parameters.
  quad::Interp1D a_mu_;
  quad::Interp1D b_mu_;
  quad::Interp1D a_sig_;
  quad::Interp1D b_sig_;
  quad::Interp1D a_tau_;
  quad::Interp1D b_tau_;
  quad::Interp1D a_fprj_;
  quad::Interp1D b_fprj_;

  struct Params {
    double mu;
    double sigma;
    double tau;
    double fprj;
  };

  // Standard Gaussian CDF Phi(x). Used in Eq. 16, Eq. 18, Eq. 29, Eq. 30.
  __host__ __device__ static double Phi(double x)
  {
    return 0.5 * (1.0 + erf(x * emg_constants::SQRT2_INV));
  }

  __host__ __device__ static double clamp01(double x)
  {
    return fmax(0.0, fmin(1.0, x));
  }

  __host__ __device__ Params params(double ltr, double z) const
  {
    double const l = fmax(ltr, 0.5);

    // Delta_mu(lambda_tr,z): Gaussian mean bias. Eq. 12.
    double const delta_mu = a_mu_.clamp(z) + b_mu_.clamp(z) * l;

    Params p;

    // mu = lambda_tr + Delta_mu. Eq. 12.
    p.mu = l + delta_mu;

    // sigma(lambda_tr,z), tau(lambda_tr,z), f_prj(lambda_tr,z). Eq. 11.
    p.sigma = b_sig_.clamp(z) * pow(l, a_sig_.clamp(z));
    p.tau   = b_tau_.clamp(z) / pow(l, a_tau_.clamp(z));

    double const denom = pow(1.0 + exp(-l), a_fprj_.clamp(z));
    p.fprj = b_fprj_.clamp(z) / fmax(denom, 1e-10);

    // Numerical safety.
    p.sigma = fmax(p.sigma, 1e-6);
    p.tau   = fmax(p.tau, 1e-6);
    p.fprj  = clamp01(p.fprj);

    return p;
  }

  // EMG CDF: F_EMG(x; mu, sigma, tau). Eq. 18.
  __host__ __device__ static double emg_cdf(double lob, Params p)
  {
    double const x = (lob - p.mu) / p.sigma;

    double A = -p.tau * (lob - p.mu)
             + 0.5 * p.tau * p.tau * p.sigma * p.sigma;
    A = fmax(-700.0, fmin(700.0, A));

    return Phi(x) - exp(A) * Phi(x - p.tau * p.sigma);
  }

public:
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

  __host__ __device__ void
  get_params(double ltr, double z,
             double& mu, double& sigma, double& tau, double& fprj) const
  {
    Params const p = params(ltr, z);
    mu    = p.mu;
    sigma = p.sigma;
    tau   = p.tau;
    fprj  = p.fprj;
  }

  // CDF used for richness-bin probability:
  // Ki = cdf(lambda_max) - cdf(lambda_min). Eq. 17, Eq. 29, Eq. 30.
  __host__ __device__ double
  cdf(double lob, double ltr, double z) const
  {
    if (!isfinite(lob)) {
      return lob > 0.0 ? 1.0 : 0.0;
    }

    Params const p = params(ltr, z);

    // Gaussian CDF piece. Eq. 16.
    double const F_gauss = Phi((lob - p.mu) / p.sigma);

    // EMG CDF piece. Eq. 18.
    double const F_emg = emg_cdf(lob, p);

    // Mixture CDF: (1 - f_prj) F_G + f_prj F_EMG. Eq. 13 and Eq. 29.
    return clamp01((1.0 - p.fprj) * F_gauss + p.fprj * F_emg);
  }

  // PDF P(lambda_ob | lambda_tr, z). Eq. 11.
  __host__ __device__ double
  operator()(double lob, double ltr, double z) const
  {
    Params const p = params(ltr, z);

    // Gaussian PDF term. Eq. 11.
    double const G = y3_cuda::gaussian(lob, p.mu, p.sigma);

    // EMG PDF term. Eq. 11.
    double const A = 0.5 * p.tau *
                     (2.0 * p.mu + p.tau * p.sigma * p.sigma - 2.0 * lob);
    double const u = (p.mu + p.tau * p.sigma * p.sigma - lob)
                   / (emg_constants::SQRT2 * p.sigma);
    double const EMG = 0.5 * p.tau * exp(A) * erfc(u);

    // Full Gaussian + EMG mixture. Eq. 11 and Eq. 13.
    return (1.0 - p.fprj) * G + p.fprj * EMG;
  }

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

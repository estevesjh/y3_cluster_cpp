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
  __host__ __device__ constexpr double SQRT2_INV = 0.7071067811865475;
  __host__ __device__ constexpr double SQRT2     = 1.4142135623730951;
}

class EMG_DES_t {
private:
  // DES calibration parameters for P(lambda_ob | lambda_tr, z).
  quad::Interp1D a_mu_;
  quad::Interp1D b_mu_;
  quad::Interp1D a_sig_;
  quad::Interp1D b_sig_;
  quad::Interp1D a_tau_;
  quad::Interp1D b_tau_;
  quad::Interp1D a_fprj_;
  quad::Interp1D b_fprj_;

  struct Params {
    double delta_mu;
    double mu;
    double sigma;
    double tau;
    double fprj;
  };

  // Phi(x): standard Gaussian CDF.
  __host__ __device__ static double Phi(double x)
  {
    return 0.5 * (1.0 + erf(x * emg_constants::SQRT2_INV));
  }

  __host__ __device__ static double clamp01(double x)
  {
    return fmax(0.0, fmin(1.0, x));
  }

  __host__ __device__ Params params(double lambda_tr, double z) const
  {
    double const l = fmax(lambda_tr, 0.5);

    Params p;

    // Delta_mu(lambda_tr,z). Eq. 12.
    p.delta_mu = a_mu_.clamp(z) + b_mu_.clamp(z) * l;

    // mu = lambda_tr + Delta_mu. Eq. 12.
    p.mu = l + p.delta_mu;

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

  // One richness-bin edge of Eq. 30.
  // Ki(lambda_tr,z) = Ki_edge(lambda_max) - Ki_edge(lambda_min).
  __host__ __device__ static double Ki_edge(double lambda_ob, Params p)
  {
    if (!isfinite(lambda_ob)) {
      return lambda_ob > 0.0 ? 1.0 : 0.0;
    }

    // x = (lambda_ob - lambda_tr - Delta_mu) / sigma. Eq. 30.
    double const x = (lambda_ob - p.mu) / p.sigma;

    // A = -tau(lambda_ob - lambda_tr - Delta_mu) + 0.5 tau^2 sigma^2. Eq. 30.
    double A = -p.tau * (lambda_ob - p.mu)
             + 0.5 * p.tau * p.tau * p.sigma * p.sigma;
    A = fmax(-700.0, fmin(700.0, A));

    // Edge form of Eq. 30.
    return Phi(x) - p.fprj * exp(A) * Phi(x - p.tau * p.sigma);
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
  get_params(double lambda_tr, double z,
             double& mu, double& sigma, double& tau, double& fprj) const
  {
    Params const p = params(lambda_tr, z);
    mu    = p.mu;
    sigma = p.sigma;
    tau   = p.tau;
    fprj  = p.fprj;
  }

  // K_j(z): redshift-bin probability. Eq. 3.
  __host__ __device__ static double
  Kj_photoz(double z_true, double z_min, double z_max, double sigma_z)
  {
    double const sig = fmax(sigma_z, 1e-12);
    double const hi = Phi((z_max - z_true) / sig);  // Eq. 3 upper edge.
    double const lo = Phi((z_min - z_true) / sig);  // Eq. 3 lower edge.
    return clamp01(hi - lo);                        // Eq. 3 bin probability.
  }

  // K_i(lambda_tr,z): richness-bin probability. Eq. 30.
  __host__ __device__ double
  Ki_richness(double lambda_tr,
              double z,
              double lambda_min,
              double lambda_max) const
  {
    Params const p = params(lambda_tr, z);

    double const hi = Ki_edge(lambda_max, p);  // Eq. 30 at lambda_max.
    double const lo = Ki_edge(lambda_min, p);  // Eq. 30 at lambda_min.

    return clamp01(hi - lo);                  // Eq. 30 evaluated on Delta lambda_i.
  }

  // K_ij(lambda_tr,z): total richness-redshift bin probability. Eq. 3.
  __host__ __device__ double
  Ktot_ij(double lambda_tr,
          double z_true,
          double lambda_min,
          double lambda_max,
          double z_min,
          double z_max,
          double sigma_z) const
  {
    double const Ki = Ki_richness(lambda_tr, z_true, lambda_min, lambda_max); // Eq. 30.
    double const Kj = Kj_photoz(z_true, z_min, z_max, sigma_z);              // Eq. 3.
    return Ki * Kj;                                                          // Eq. 3.
  }

  // Backward-compatible name: this is one edge of Eq. 30, not the full bin Ki.
  __host__ __device__ double
  cdf(double lambda_ob, double lambda_tr, double z) const
  {
    return clamp01(Ki_edge(lambda_ob, params(lambda_tr, z)));
  }

  // PDF P(lambda_ob | lambda_tr,z). Eq. 11.
  __host__ __device__ double
  operator()(double lambda_ob, double lambda_tr, double z) const
  {
    Params const p = params(lambda_tr, z);

    // Gaussian PDF term. Eq. 11.
    double const G = y3_cuda::gaussian(lambda_ob, p.mu, p.sigma);

    // EMG PDF term. Eq. 11.
    double const A = 0.5 * p.tau *
                     (2.0 * p.mu + p.tau * p.sigma * p.sigma - 2.0 * lambda_ob);
    double const u = (p.mu + p.tau * p.sigma * p.sigma - lambda_ob)
                   / (emg_constants::SQRT2 * p.sigma);
    double const EMG = 0.5 * p.tau * exp(A) * erfc(u);

    // Mixture PDF. Eq. 11.
    return (1.0 - p.fprj) * G + p.fprj * EMG;
  }

  __host__ __device__ double
  pdf_with_params(double lambda_ob, double lambda_tr, double z,
                  double& mu_out, double& sigma_out,
                  double& tau_out, double& fprj_out) const
  {
    get_params(lambda_tr, z, mu_out, sigma_out, tau_out, fprj_out);
    return (*this)(lambda_ob, lambda_tr, z);
  }
};

} // namespace y3_cuda

#endif

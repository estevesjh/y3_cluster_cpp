#ifndef Y3_CLUSTER_MOR_SHIFTED_POISSON_T_CUH
#define Y3_CLUSTER_MOR_SHIFTED_POISSON_T_CUH

// Costanzi-2026 shifted-Poisson MOR for GPU.
//
// Matches the CPU implementation in p_operator_t.hh (lines 177-235).
// This is the continuous shifted-Poisson form, which is used in p_operator_gpu_t.cuh instead of the Costanzi-2019
// skewed-Gaussian used by mor_des_log_t.cuh
//
// Formula:
//   l_sat(M, z) = ((M - Mmin) / (M1 - Mmin))^alpha * ((1+z)/(1+z_pivot))^epsilon
//   mi = (l_sat * sigma_intr)^2
//   lam = l_sat + mi
//   x = ltr + mi
//   pdf(ltr | M, z) = exp(-lam + (x - 1) * log(lam) - lgamma(x))
//

#include "cosmosis/datablock/datablock.hh"

#include <cmath>

namespace y3_cuda {

  class MOR_SHIFTED_POISSON_t {
  private:
    double log10_Mmin_;
    double log10_M1_;
    double alpha_;
    double sigma_intr_;
    double epsilon_;
    double z_pivot_;

    // Derived quantities (computed once)
    double Mmin_;
    double M1_;
    double dM1_;  // M1 - Mmin

  public:
    __host__ __device__
    MOR_SHIFTED_POISSON_t()
      : log10_Mmin_(13.0)
      , log10_M1_(14.0)
      , alpha_(1.0)
      , sigma_intr_(0.2)
      , epsilon_(0.0)
      , z_pivot_(0.5)
      , Mmin_(1e13)
      , M1_(1e14)
      , dM1_(M1_ - Mmin_)
    {}

    MOR_SHIFTED_POISSON_t(double log10_Mmin,
                          double log10_M1,
                          double alpha,
                          double sigma_intr,
                          double epsilon,
                          double z_pivot)
      : log10_Mmin_(log10_Mmin)
      , log10_M1_(log10_M1)
      , alpha_(alpha)
      , sigma_intr_(sigma_intr)
      , epsilon_(epsilon)
      , z_pivot_(z_pivot)
      , Mmin_(pow(10.0, log10_Mmin))
      , M1_(pow(10.0, log10_M1))
      , dM1_(M1_ - Mmin_)
    {}

    // Construct from datablock - reads from cluster_mor section
    // (same as CPU MOR_HOD_t)
    explicit MOR_SHIFTED_POISSON_t(cosmosis::DataBlock& sample)
    {
      // Try cluster_mor first (CPU convention), fall back to cluster_abundance
      if (sample.has_val("cluster_mor", "log10_Mmin")) {
        log10_Mmin_ = sample.view<double>("cluster_mor", "log10_Mmin");
        log10_M1_   = sample.view<double>("cluster_mor", "log10_M1");
        alpha_      = sample.view<double>("cluster_mor", "alpha");
        sigma_intr_ = sample.view<double>("cluster_mor", "sigma_lambda");
        epsilon_    = sample.view<double>("cluster_mor", "epsilon");
        z_pivot_    = sample.view<double>("cluster_mor", "z_pivot");
      } else {
        // Fallback to cluster_abundance with ratio encoding
        double const mor_logMmin = sample.view<double>("cluster_abundance", "mor_logMmin");
        double const mor_logRatio = sample.view<double>("cluster_abundance", "mor_logRatio");
        log10_Mmin_ = mor_logMmin;
        log10_M1_   = mor_logMmin + mor_logRatio;  // log10(M1) = log10(Mmin) + log10(M1/Mmin)
        alpha_      = sample.view<double>("cluster_abundance", "mor_alpha");
        sigma_intr_ = sample.view<double>("cluster_abundance", "mor_sigma");
        epsilon_    = sample.view<double>("cluster_abundance", "mor_epsilon");
        z_pivot_    = sample.view<double>("cluster_abundance", "z_mor_pivot");
      }

      Mmin_ = pow(10.0, log10_Mmin_);
      M1_   = pow(10.0, log10_M1_);
      dM1_  = M1_ - Mmin_;
    }

    size_t get_device_mem_footprint() const { return 0; }

    // Accessors for parameters (used by pre-caching in p_operator)
    __host__ __device__ double log10_Mmin() const { return log10_Mmin_; }
    __host__ __device__ double log10_M1() const { return log10_M1_; }
    __host__ __device__ double alpha() const { return alpha_; }
    __host__ __device__ double sigma_lambda() const { return sigma_intr_; }
    __host__ __device__ double epsilon() const { return epsilon_; }
    __host__ __device__ double z_pivot() const { return z_pivot_; }

    // Evaluate P(ltr | M, z) using shifted-Poisson formula
    __host__ __device__ double
    operator()(double ltr, double lnM, double z) const
    {
      double const M = exp(lnM);

      // l_sat(M, z) = ((M - Mmin) / (M1 - Mmin))^alpha * ((1+z)/(1+z_pivot))^epsilon
      double const dM = M - Mmin_;
      if (dM <= 0.0 || dM1_ <= 0.0) {
        // Below minimum mass - return 0
        return 0.0;
      }

      double const zfac = pow((1.0 + z) / (1.0 + z_pivot_), epsilon_);
      double const l_sat = pow(dM / dM1_, alpha_) * zfac;

      // Shifted-Poisson parameters
      double const mi = (l_sat * sigma_intr_) * (l_sat * sigma_intr_);
      double const lam = fmax(l_sat + mi, 1.0e-300);
      double const x = ltr + mi;

      // Require ltr >= 0 and x > 0 for lgamma
      if (ltr < 0.0 || x <= 0.0) {
        return 0.0;
      }

      // pdf = exp(-lam + (x - 1) * log(lam) - lgamma(x))
      double const log_lam = log(lam);
      double const log_pdf = -lam + (x - 1.0) * log_lam - lgamma(x);

      return exp(log_pdf);
    }
  };

}  // namespace y3_cuda

#endif

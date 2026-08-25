#ifndef Y3_CLUSTER_MOR_HOD_T_CUH
#define Y3_CLUSTER_MOR_HOD_T_CUH

// Device mirror of y3_cluster::MOR_HOD_t (src/models/mor_hod_t.hh) —
// the Costanzi-2019 shifted-Poisson HOD P(lambda_tr | M, z) with the
// central-galaxy count shift. This is the MOR convention of the CPU
// explicit-3d backends and the Python reference (kernels/mor_hod.py);
// the explicit-3d GPU backends must use THIS class so CPU<->GPU
// cross-backend pins compare the same physics.
//
// The Costanzi-2026 continuous form WITHOUT the central shift
// (x = ltr + delta) lives in mor_shifted_poisson_t.cuh and is
// reserved for the P[X]/b_sel operators, whose CPU twin
// (p_operator_t.hh) uses that form too. The two are related exactly by
// MOR_SP(ltr) = MOR_HOD(ltr + 1) above Mmin — pinned in
// test/num_counts_3d_gpu.test.cu.
//
// Formula (identical to the host class, kernels/mor_hod.py:91-105):
//   lambda_central = 1 for M >= Mmin, else 0
//   mu_sat(M, z)   = ((M - Mmin)/(M1 - Mmin))^alpha
//                    * ((1+z)/(1+z_pivot))^epsilon
//   delta          = (sigma_lambda * mu_sat)^2
//   nu             = mu_sat + delta
//   x              = ltr - lambda_central + delta
//   pdf(ltr)       = exp(-nu + (x - 1) ln(nu) - lgamma(x))    x > 0
// with the mu_sat -> 0 branch collapsing to a narrow Gaussian at
// lambda_central, exactly as the host class.

#include "cosmosis/datablock/datablock.hh"

#include <cmath>

namespace y3_cuda {

  class MOR_HOD_t {
  private:
    double log10_Mmin_;
    double log10_M1_;
    double alpha_;
    double sigma_lambda_;
    double epsilon_;
    double z_pivot_;

    // Derived (computed once, host side)
    double Mmin_;
    double M1_;
    double dM1_; // M1 - Mmin

    static constexpr double POISSON_TOL = 1e-8;      // MOR_HOD_t twin
    static constexpr double FALLBACK_SIGMA = 1.0e-3; // MOR_HOD_t twin
    static constexpr double SQRT_2PI = 2.5066282746310002;

  public:
    __host__ __device__
    MOR_HOD_t()
      : log10_Mmin_(13.0)
      , log10_M1_(14.0)
      , alpha_(1.0)
      , sigma_lambda_(0.2)
      , epsilon_(0.0)
      , z_pivot_(0.45)
      , Mmin_(1e13)
      , M1_(1e14)
      , dM1_(M1_ - Mmin_)
    {}

    // Same parameter order as the host y3_cluster::MOR_HOD_t:
    // (log10_Mmin, log10_M1, alpha, epsilon, sigma_lambda, z_pivot).
    MOR_HOD_t(double log10_Mmin,
              double log10_M1,
              double alpha,
              double epsilon,
              double sigma_lambda,
              double z_pivot = 0.45)
      : log10_Mmin_(log10_Mmin)
      , log10_M1_(log10_M1)
      , alpha_(alpha)
      , sigma_lambda_(sigma_lambda)
      , epsilon_(epsilon)
      , z_pivot_(z_pivot)
      , Mmin_(pow(10.0, log10_Mmin))
      , M1_(pow(10.0, log10_M1))
      , dM1_(M1_ - Mmin_)
    {}

    // Same cluster_mor input contract as the host MOR_HOD_t: strict
    // log10_Mmin; log10_ratio (= log10(M1/Mmin)) wins over log10_M1
    // when both are present; z_pivot optional with the shared 0.45
    // default.
    explicit MOR_HOD_t(cosmosis::DataBlock& sample)
    {
      log10_Mmin_ = sample.view<double>("cluster_mor", "log10_Mmin");
      log10_M1_ =
        sample.has_val("cluster_mor", "log10_ratio")
          ? log10_Mmin_ + sample.view<double>("cluster_mor", "log10_ratio")
          : sample.view<double>("cluster_mor", "log10_M1");
      alpha_         = sample.view<double>("cluster_mor", "alpha");
      sigma_lambda_  = sample.view<double>("cluster_mor", "sigma_lambda");
      epsilon_       = sample.view<double>("cluster_mor", "epsilon");
      z_pivot_       = sample.has_val("cluster_mor", "z_pivot")
                         ? sample.view<double>("cluster_mor", "z_pivot")
                         : 0.45; // MOR_HOD_t::Z_PIVOT_DEFAULT

      Mmin_ = pow(10.0, log10_Mmin_);
      M1_   = pow(10.0, log10_M1_);
      dM1_  = M1_ - Mmin_;
    }

    size_t get_device_mem_footprint() const { return 0; }

    __host__ __device__ double log10_Mmin() const { return log10_Mmin_; }
    __host__ __device__ double log10_M1() const { return log10_M1_; }
    __host__ __device__ double alpha() const { return alpha_; }
    __host__ __device__ double sigma_lambda() const { return sigma_lambda_; }
    __host__ __device__ double epsilon() const { return epsilon_; }
    __host__ __device__ double z_pivot() const { return z_pivot_; }

    // P(lt | M, z) — verbatim the host MOR_HOD_t::operator() algebra.
    __host__ __device__ double
    operator()(double lt, double lnM, double zt) const
    {
      if (lt < 0.0)
        return 0.0;

      double const M = exp(lnM);
      double const lcentral = (M >= Mmin_) ? 1.0 : 0.0;
      if (lt < lcentral)
        return 0.0; // lambda_sat >= 0

      // mu_sat(M, z)
      double mu = 0.0;
      double const dM = M - Mmin_;
      if (dM > 0.0 && dM1_ > 0.0) {
        double const zfac = pow((1.0 + zt) / (1.0 + z_pivot_), epsilon_);
        mu = pow(dM / dM1_, alpha_) * zfac;
      }

      // mu_sat -> 0 branch: collapse to narrow Gaussian at lcentral.
      if (mu <= POISSON_TOL) {
        double const dx = (lt - lcentral) / FALLBACK_SIGMA;
        return exp(-0.5 * dx * dx) / (SQRT_2PI * FALLBACK_SIGMA);
      }

      double const delta = (sigma_lambda_ * mu) * (sigma_lambda_ * mu);
      double const nu = mu + delta;
      double const x = lt - lcentral + delta;
      if (x <= 0.0)
        return 0.0;

      double const log_P = -nu + (x - 1.0) * log(nu) - lgamma(x);
      return exp(log_P);
    }
  };

} // namespace y3_cuda

#endif

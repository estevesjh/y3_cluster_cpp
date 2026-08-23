// Device ports of the full_ltmz selection kernels — des_y3 scoped.
//
// The CPU backend composes the immutable host models MOR_HOD_t,
// PlobLtrEMG_t and RichnessKernel_t; those have no .cuh counterparts,
// and the approved plan forbids editing them, so the CUDA backend
// carries these verbatim __host__ __device__ ports instead:
//
//   MorHodDevice      <-> src/models/mor_hod_t.hh (shifted-Poisson HOD)
//   PlobEmgDevice     <-> src/models/plob_ltr_emg_t.hh (EMG parameters;
//                          clamped linear interpolation over the 15
//                          z-nodes, exactly Interp1D::clamp semantics)
//   richness_kernel   <-> src/models/richness_kernel_t.hh (K_i via the
//                          erfcx-stable EMG CDF)
//   zkernel           <-> richness_zkernel (Gaussian photo-z K_j)
//
// The structs are trivially copyable (fixed-size arrays, no pointers)
// so the integrand object can be memcpy'd to the device by PAGANI.
// Backend equivalence against the CPU module is part of the full_ltmz
// validation (see ../python/validate_vs_production.py and README.md),
// which is what keeps these ports honest.
#ifndef Y3_CLUSTER_CPP_DES_Y3_FULL_LTMZ_DEVICE_KERNELS_CUH
#define Y3_CLUSTER_CPP_DES_Y3_FULL_LTMZ_DEVICE_KERNELS_CUH

#include "cosmosis/datablock/datablock.hh"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace y3_cuda_des_y3 {

  constexpr double SQRT2 = 1.41421356237309504880;

  __host__ __device__ inline double
  norm_cdf(double x)
  {
    return 0.5 * (1.0 + erf(x / SQRT2));
  }

  // erfcx(t) = exp(t^2) erfc(t), stable at large t — verbatim
  // rk_detail::erfcx.
  __host__ __device__ inline double
  erfcx_stable(double t)
  {
    constexpr double ERFCX_CUTOFF = 25.0;
    if (t < ERFCX_CUTOFF) { return exp(t * t) * erfc(t); }
    double const inv = 1.0 / t;
    double const inv2 = inv * inv;
    return inv / sqrt(M_PI) *
           (1.0 - 0.5 * inv2 * (1.0 - 1.5 * inv2 * (1.0 - 2.5 * inv2)));
  }

  // EMG CDF — verbatim rk_detail::F_EMG.
  __host__ __device__ inline double
  f_emg(double x, double mu, double sigma, double tau)
  {
    double const z = (x - mu) / sigma;
    double const u = (tau * sigma - z) / SQRT2;
    bool const neg = u < 0.0;
    double const abs_u = neg ? -u : u;
    double const exp_mz2 = exp(-0.5 * z * z);
    double const tail_base = 0.5 * erfcx_stable(abs_u) * exp_mz2;
    double tail;
    if (neg) {
      double const A = -tau * (x - mu) + 0.5 * tau * tau * sigma * sigma;
      tail = exp(A) - tail_base;
    } else {
      tail = tail_base;
    }
    return fmin(fmax(norm_cdf(z) - tail, 0.0), 1.0);
  }

  // Bin-integrated K_i given the EMG parameters at (ltr, z) — verbatim
  // RichnessKernel_t::operator().
  __host__ __device__ inline double
  richness_kernel(double lob_min, double lob_max, double mu, double sigma,
                  double tau, double fprj)
  {
    double const f = fmin(1.0, fprj);
    double const gauss_piece =
      norm_cdf((lob_max - mu) / sigma) - norm_cdf((lob_min - mu) / sigma);
    double const emg_piece =
      f_emg(lob_max, mu, sigma, tau) - f_emg(lob_min, mu, sigma, tau);
    return (1.0 - f) * gauss_piece + f * emg_piece;
  }

  // Gaussian photo-z kernel K_j — verbatim richness_zkernel.
  __host__ __device__ inline double
  zkernel(double zt, double zob_min, double zob_max, double sigma_z)
  {
    return norm_cdf((zob_max - zt) / sigma_z) -
           norm_cdf((zob_min - zt) / sigma_z);
  }

  // EMG parameter provider: the 8 coefficient tables over the (15)
  // z-nodes of the plob_ltr_params section, held in fixed-size arrays
  // with clamped linear interpolation (Interp1D::clamp semantics).
  struct PlobEmgDevice {
    static constexpr int MAX_NODES = 32;

    int n{0};
    double z[MAX_NODES];
    double a_mu[MAX_NODES], b_mu[MAX_NODES];
    double a_sig[MAX_NODES], b_sig[MAX_NODES];
    double a_tau[MAX_NODES], b_tau[MAX_NODES];
    double a_fprj[MAX_NODES], b_fprj[MAX_NODES];

    __host__ __device__ double
    lin(double const* y, double zq) const
    {
      if (zq <= z[0]) return y[0];
      if (zq >= z[n - 1]) return y[n - 1];
      int i = 0;
      while (zq > z[i + 1]) ++i;
      double const t = (zq - z[i]) / (z[i + 1] - z[i]);
      return y[i] + t * (y[i + 1] - y[i]);
    }

    __host__ __device__ double
    mu(double ltr, double zq) const
    {
      return lin(a_mu, zq) + lin(b_mu, zq) * ltr;
    }
    __host__ __device__ double
    sigma(double ltr, double zq) const
    {
      return lin(b_sig, zq) * pow(ltr, lin(a_sig, zq));
    }
    __host__ __device__ double
    tau(double ltr, double zq) const
    {
      return lin(b_tau, zq) / pow(ltr, lin(a_tau, zq));
    }
    __host__ __device__ double
    fprj(double ltr, double zq) const
    {
      return fmin(1.0,
                  lin(b_fprj, zq) / pow(1.0 + exp(-ltr), lin(a_fprj, zq)));
    }

    // Host-side fill from the plob_ltr_params datablock section.
    static PlobEmgDevice
    from_datablock(cosmosis::DataBlock& sample)
    {
      PlobEmgDevice out;
      auto read = [&sample](char const* key) {
        return sample.view<std::vector<double>>("plob_ltr_params", key);
      };
      auto const zs = read("z");
      if (zs.size() > MAX_NODES)
        throw std::runtime_error(
          "PlobEmgDevice: plob_ltr_params/z exceeds MAX_NODES");
      out.n = static_cast<int>(zs.size());
      auto fill = [&out](double* dst, std::vector<double> const& src) {
        for (std::size_t i = 0; i != src.size(); ++i) dst[i] = src[i];
      };
      fill(out.z, zs);
      fill(out.a_mu, read("a_mu"));
      fill(out.b_mu, read("b_mu"));
      fill(out.a_sig, read("a_sig"));
      fill(out.b_sig, read("b_sig"));
      fill(out.a_tau, read("a_tau"));
      fill(out.b_tau, read("b_tau"));
      fill(out.a_fprj, read("a_fprj"));
      fill(out.b_fprj, read("b_fprj"));
      return out;
    }
  };

  // Shifted-Poisson HOD P(lt | M, z) — verbatim MOR_HOD_t, including
  // the mu_sat -> 0 narrow-Gaussian fallback and both support
  // boundaries. lgamma is available in device code.
  struct MorHodDevice {
    static constexpr double POISSON_TOL = 1e-8;
    static constexpr double FALLBACK_SIGMA = 1.0e-3;
    static constexpr double Z_PIVOT_DEFAULT = 0.45;

    double log10_Mmin{0.0}, log10_M1{0.0};
    double alpha{1.0}, epsilon{0.0}, sigma_lambda{0.0};
    double z_pivot{Z_PIVOT_DEFAULT};

    __host__ __device__ double
    operator()(double lt, double lnM, double zt) const
    {
      if (lt < 0.0) return 0.0;
      double const M = exp(lnM);
      double const Mmin = pow(10.0, log10_Mmin);
      double const lcentral = (M >= Mmin) ? 1.0 : 0.0;
      if (lt < lcentral) return 0.0;

      double const M1 = pow(10.0, log10_M1);
      double const dM1 = M1 - Mmin;
      double mu = 0.0;
      double const dM = fmax(M - Mmin, 0.0);
      if (dM1 > 0.0 && dM > 0.0) {
        mu = pow(dM / dM1, alpha) *
             pow((1.0 + zt) / (1.0 + z_pivot), epsilon);
      }
      if (mu <= POISSON_TOL) {
        double const dx = (lt - lcentral) / FALLBACK_SIGMA;
        return exp(-0.5 * dx * dx) /
               (sqrt(2.0 * M_PI) * FALLBACK_SIGMA);
      }
      double const delta = (sigma_lambda * mu) * (sigma_lambda * mu);
      double const nu = mu + delta;
      double const x = lt - lcentral + delta;
      if (x <= 0.0) return 0.0;
      double const log_P = -nu + (x - 1.0) * log(nu) - lgamma(x);
      return exp(log_P);
    }

    // Host-side fill from cluster_mor, with the same log10_ratio /
    // log10_M1 precedence and optional z_pivot as MOR_HOD_t.
    static MorHodDevice
    from_datablock(cosmosis::DataBlock& sample)
    {
      MorHodDevice out;
      out.log10_Mmin = sample.view<double>("cluster_mor", "log10_Mmin");
      if (sample.has_val("cluster_mor", "log10_ratio")) {
        out.log10_M1 =
          out.log10_Mmin + sample.view<double>("cluster_mor", "log10_ratio");
      } else {
        out.log10_M1 = sample.view<double>("cluster_mor", "log10_M1");
      }
      out.alpha = sample.view<double>("cluster_mor", "alpha");
      out.epsilon = sample.view<double>("cluster_mor", "epsilon");
      out.sigma_lambda = sample.view<double>("cluster_mor", "sigma_lambda");
      if (sample.has_val("cluster_mor", "z_pivot")) {
        out.z_pivot = sample.view<double>("cluster_mor", "z_pivot");
      }
      return out;
    }
  };

} // namespace y3_cuda_des_y3

#endif

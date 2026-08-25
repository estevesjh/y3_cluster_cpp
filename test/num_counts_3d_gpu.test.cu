#include "catch2/catch.hpp"

// This is the code we're actually testing: the device models the
// explicit-3d GPU backends compose — y3_cuda::MOR_HOD_t (the device
// mirror of the host HOD MOR), y3_cuda::EMG_DES_t's closed-form
// primitives, the b_sel-operator MOR y3_cuda::MOR_SAT_ONLY_t —
// plus the zkernel_sj observed-redshift kernel carried by
// num_counts_3d_gpu_t.cuh. NumCounts3dGpu
// itself (the CosmoSIS integrand) composes these with the pre-existing
// y3_cuda::HMF_t/DV_DO_DZ_t/OMEGA_Z_DES device models (already covered
// by their own tests) and needs a live cosmosis::DataBlock to
// construct, so this test isolates the genuinely new pieces instead.
#include "models/emg_des_t.cuh"
#include "models/mor_hod_t.cuh"
#include "models/mor_sat_only_t.cuh"
#include "pipelines/des_y3/number_counts/cuda/3d/num_counts_3d_gpu_t.cuh"

// Host twins: rk_detail's phi/erfcx closed forms (richness_kernel_t.hh)
// for the EMG primitives and richness_zkernel for S_j; MOR_HOD_t for
// the shifted-Poisson identity below. Comparing against them directly
// is a stronger, more independent check than hand-picking reference
// numbers, and needs no DataBlock or dump.
#include "models/mor_hod_t.hh"
#include "models/richness_kernel_t.hh"

#include <cmath>

using y3_cluster::MOR_HOD_t;
namespace rk = y3_cluster::rk_detail;

namespace {
  constexpr double PORT_TOL = 1.0e-13;
}

TEST_CASE("emg_des_t device phi_cdf/erfcx_impl match the host rk_detail twins")
{
  for (double x : {-3.5, -1.0, -0.1, 0.0, 0.37, 1.2, 4.0}) {
    CHECK(y3_cuda::phi_cdf(x) == Approx(rk::phi(x)).epsilon(PORT_TOL));
  }
  // Cover both erfcx_impl branches: direct (|x| < 4) and the asymptotic
  // series (|x| >= 4).
  for (double t : {0.0, 0.5, 3.0, 3.9, 4.0, 10.0, 30.0, 100.0}) {
    CHECK(y3_cuda::erfcx_impl(t) == Approx(rk::erfcx(t)).epsilon(PORT_TOL));
  }
}

TEST_CASE("zkernel_sj matches the host richness_zkernel closed form")
{
  for (auto const& p : {std::array<double, 4>{0.30, 0.25, 0.45, 0.02},
                        std::array<double, 4>{0.42, 0.25, 0.45, 0.02},
                        std::array<double, 4>{0.55, 0.25, 0.45, 0.02},
                        std::array<double, 4>{0.10, 0.25, 0.45, 0.08}}) {
    double const zt = p[0], zmin = p[1], zmax = p[2], sig = p[3];
    CHECK(y3_cuda_des_y3::zkernel_sj(zt, zmin, zmax, sig) ==
          Approx(y3_cluster::richness_zkernel(zt, zmin, zmax, sig))
            .epsilon(PORT_TOL));
  }
}

TEST_CASE("MOR_SAT_ONLY_t is MOR_HOD_t shifted by the central count")
{
  // The Costanzi-2026 P-operator form (x = ltr + delta) relates to
  // MOR_HOD_t's central-shifted form (x = ltr - lambda_cen + delta,
  // lambda_cen = 1 above Mmin) exactly by
  //
  //   MOR_SAT_ONLY(ltr, lnM, z) = MOR_HOD(ltr + 1, lnM, z)   for M >= Mmin
  //
  // away from the mu_sat -> 0 fallback branch. This pins the
  // b_sel-operator device model's formula; the explicit-3d GPU
  // backends themselves use y3_cuda::MOR_HOD_t (parity test below),
  // so no cross-backend offset survives in the pipeline.
  double const log10_Mmin = 13.8, log10_M1 = 14.5, alpha = 1.1,
               sigma_lambda = 0.35, epsilon = -0.2, z_pivot = 0.45;
  MOR_HOD_t const mor_hod(log10_Mmin, log10_M1, alpha, epsilon,
                          sigma_lambda, z_pivot);
  y3_cuda::MOR_SAT_ONLY_t const mor_sat(
    log10_Mmin, log10_M1, alpha, sigma_lambda, epsilon, z_pivot);

  double const lnM = std::log(std::pow(10.0, 14.2));
  double const lnM_high = std::log(std::pow(10.0, 14.8));
  for (auto const& p : {std::array<double, 3>{5.0, lnM, 0.3},
                        std::array<double, 3>{12.0, lnM, 0.5},
                        std::array<double, 3>{40.0, lnM_high, 0.4},
                        std::array<double, 3>{80.0, lnM_high, 0.6}}) {
    double const lt = p[0], m = p[1], zt = p[2];
    CHECK(mor_sat(lt, m, zt) ==
          Approx(mor_hod(lt + 1.0, m, zt)).epsilon(PORT_TOL));
  }

  // Below Mmin both conventions vanish.
  double const lnM_below = std::log(std::pow(10.0, 13.5));
  CHECK(mor_sat(5.0, lnM_below, 0.4) == 0.0);
}

TEST_CASE("y3_cuda::MOR_HOD_t matches the host MOR_HOD_t exactly")
{
  // The device HOD MOR is the model the explicit-3d GPU backends
  // (NumCounts3dGpu, Shear1h3dGpu) actually compose; it must be
  // bit-level the same algebra as the host class so CPU<->GPU
  // cross-backend pins compare identical physics.
  double const log10_Mmin = 11.4, log10_M1 = 12.7, alpha = 0.86,
               sigma_lambda = 0.18, epsilon = 0.0, z_pivot = 0.45;
  MOR_HOD_t const host(log10_Mmin, log10_M1, alpha, epsilon,
                       sigma_lambda, z_pivot);
  y3_cuda::MOR_HOD_t const dev(log10_Mmin, log10_M1, alpha, epsilon,
                               sigma_lambda, z_pivot);

  // Span the DES Y3 integration range: masses below/at/above Mmin,
  // richness from the lambda_central cutoff into the bins, both
  // redshift edges. Includes the mu_sat -> 0 Gaussian fallback branch
  // (M barely above Mmin) and the lt < lambda_central hard zero.
  for (double log10_M : {11.0, 11.4000001, 11.8, 13.0, 14.5, 15.5}) {
    double const lnM = std::log(std::pow(10.0, log10_M));
    for (double lt : {0.0, 0.5, 0.9999, 1.0, 2.0, 20.0, 60.0, 199.0}) {
      for (double zt : {0.05, 0.45, 0.80}) {
        double const h = host(lt, lnM, zt);
        double const d = dev(lt, lnM, zt);
        if (h == 0.0) {
          CHECK(d == 0.0);
        } else {
          CHECK(d == Approx(h).epsilon(PORT_TOL));
        }
      }
    }
  }
}

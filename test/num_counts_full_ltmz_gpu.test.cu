#include "catch2/catch.hpp"

// This is the code we're actually testing: the gpu_prj_costanzi2026
// device models (Arwa Qadi, upstream PR #3) the full_ltmz GPU backends
// now compose — y3_cuda::MOR_SHIFTED_POISSON_t and y3_cuda::EMG_DES_t's
// closed-form primitives — plus the zkernel_sj observed-redshift kernel
// carried by num_counts_full_ltmz_gpu_t.cuh. NumCountsFullLtmzGpu
// itself (the CosmoSIS integrand) composes these with the pre-existing
// y3_cuda::HMF_t/DV_DO_DZ_t/OMEGA_Z_DES device models (already covered
// by their own tests) and needs a live cosmosis::DataBlock to
// construct, so this test isolates the genuinely new pieces instead.
#include "models/emg_des_t.cuh"
#include "models/mor_shifted_poisson_t.cuh"
#include "pipelines/des_y3/number_counts/full_ltmz/cuda/num_counts_full_ltmz_gpu_t.cuh"

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

TEST_CASE("MOR_SHIFTED_POISSON_t is MOR_HOD_t shifted by the central count")
{
  // The Costanzi-2026 P-operator form (x = ltr + delta) relates to
  // MOR_HOD_t's central-shifted form (x = ltr - lambda_cen + delta,
  // lambda_cen = 1 above Mmin) exactly by
  //
  //   MOR_SP(ltr, lnM, z) = MOR_HOD(ltr + 1, lnM, z)   for M >= Mmin
  //
  // away from the mu_sat -> 0 fallback branch. This pins both the
  // device model's formula and the documented convention offset between
  // the CPU and GPU full_ltmz backends.
  double const log10_Mmin = 13.8, log10_M1 = 14.5, alpha = 1.1,
               sigma_lambda = 0.35, epsilon = -0.2, z_pivot = 0.45;
  MOR_HOD_t const mor_hod(log10_Mmin, log10_M1, alpha, epsilon,
                          sigma_lambda, z_pivot);
  y3_cuda::MOR_SHIFTED_POISSON_t const mor_sp(
    log10_Mmin, log10_M1, alpha, sigma_lambda, epsilon, z_pivot);

  double const lnM = std::log(std::pow(10.0, 14.2));
  double const lnM_high = std::log(std::pow(10.0, 14.8));
  for (auto const& p : {std::array<double, 3>{5.0, lnM, 0.3},
                        std::array<double, 3>{12.0, lnM, 0.5},
                        std::array<double, 3>{40.0, lnM_high, 0.4},
                        std::array<double, 3>{80.0, lnM_high, 0.6}}) {
    double const lt = p[0], m = p[1], zt = p[2];
    CHECK(mor_sp(lt, m, zt) ==
          Approx(mor_hod(lt + 1.0, m, zt)).epsilon(PORT_TOL));
  }

  // Below Mmin both conventions vanish.
  double const lnM_below = std::log(std::pow(10.0, 13.5));
  CHECK(mor_sp(5.0, lnM_below, 0.4) == 0.0);
}

#include "catch2/catch.hpp"

// This is the code we're actually testing: the des_y3-scoped __host__
// __device__ ports in full_ltmz_device_kernels.cuh. NumCountsFullLtmzGpu
// itself (the CosmoSIS integrand) composes these with the pre-existing
// y3_cuda::HMF_t/DV_DO_DZ_t/OMEGA_Z_DES device models (already covered by
// their own tests) and needs a live cosmosis::DataBlock to construct, so
// this test isolates the genuinely new pieces instead.
#include "pipelines/des_y3/observables/number_counts/full_ltmz/cuda/full_ltmz_device_kernels.cuh"

// Host twins these are meant to be verbatim ports of (see the .cuh file's
// header comment): MOR_HOD_t (mor_hod_t.hh) and the rk_detail free
// functions + richness_zkernel (richness_kernel_t.hh). Comparing against
// them directly is a stronger, more independent check than hand-picking
// reference numbers, and needs no DataBlock or dump.
#include "models/mor_hod_t.hh"
#include "models/richness_kernel_t.hh"

#include <cmath>

using y3_cluster::MOR_HOD_t;
namespace rk = y3_cluster::rk_detail;

namespace {
  constexpr double PORT_TOL = 1.0e-13;
}

TEST_CASE("full_ltmz device norm_cdf/erfcx_stable match the host rk_detail twins")
{
  for (double x : {-3.5, -1.0, -0.1, 0.0, 0.37, 1.2, 4.0}) {
    CHECK(y3_cuda_des_y3::norm_cdf(x) == Approx(rk::phi(x)).epsilon(PORT_TOL));
  }
  // Cover both erfcx_stable branches: direct (t < 25) and the asymptotic
  // series (t >= 25).
  for (double t : {0.0, 0.5, 3.0, 10.0, 24.9, 25.0, 30.0, 100.0}) {
    CHECK(y3_cuda_des_y3::erfcx_stable(t) == Approx(rk::erfcx(t)).epsilon(PORT_TOL));
  }
}

TEST_CASE("full_ltmz device f_emg/richness_kernel/zkernel match the host closed forms")
{
  // Grid spanning both F_EMG branches (u >= 0 and u < 0, i.e. x on either
  // side of mu + tau*sigma^2).
  double const mus[] = {20.0, 45.0};
  double const sigmas[] = {3.0, 8.0};
  double const taus[] = {0.05, 0.20};
  double const xs[] = {5.0, 22.0, 40.0, 70.0};

  for (double mu : mus)
    for (double sigma : sigmas)
      for (double tau : taus)
        for (double x : xs) {
          CHECK(y3_cuda_des_y3::f_emg(x, mu, sigma, tau) ==
                Approx(rk::F_EMG(x, mu, sigma, tau)).epsilon(PORT_TOL));
        }

  // richness_kernel(lob_min, lob_max, mu, sigma, tau, fprj) is exactly
  // RichnessKernel_t's (1-fprj)*gauss_piece + fprj*emg_piece composition,
  // built from the same two primitives checked above.
  double const lob_min = 20.0, lob_max = 30.0;
  for (double mu : mus)
    for (double sigma : sigmas)
      for (double tau : taus)
        for (double fprj : {0.0, 0.3, 0.8, 1.0, 1.4}) {
          double const f = std::min(1.0, fprj);
          double const expected =
            (1.0 - f) * (rk::phi((lob_max - mu) / sigma) -
                        rk::phi((lob_min - mu) / sigma)) +
            f * (rk::F_EMG(lob_max, mu, sigma, tau) -
                rk::F_EMG(lob_min, mu, sigma, tau));
          double const got = y3_cuda_des_y3::richness_kernel(
            lob_min, lob_max, mu, sigma, tau, fprj);
          CHECK(got == Approx(expected).epsilon(PORT_TOL));
          // Physical bound: a bin-integrated probability.
          CHECK(got >= -1e-12);
          CHECK(got <= 1.0 + 1e-12);
        }

  for (double zt : {0.10, 0.25, 0.40, 0.55}) {
    double const got = y3_cuda_des_y3::zkernel(zt, 0.20, 0.35, 0.03);
    double const expected = y3_cluster::richness_zkernel(zt, 0.20, 0.35, 0.03);
    CHECK(got == Approx(expected).epsilon(PORT_TOL));
    CHECK(got >= -1e-12);
    CHECK(got <= 1.0 + 1e-12);
  }
}

TEST_CASE("full_ltmz device MorHodDevice matches the host MOR_HOD_t exactly")
{
  // Covers the mu_sat -> 0 narrow-Gaussian fallback (small alpha/dM) and
  // the normal shifted-Poisson branch, plus both z_pivot handling paths.
  y3_cuda_des_y3::MorHodDevice dev;
  dev.log10_Mmin = 13.5;
  dev.log10_M1 = 14.3;
  dev.alpha = 1.1;
  dev.epsilon = -0.5;
  dev.sigma_lambda = 0.25;
  dev.z_pivot = 0.45;

  MOR_HOD_t host(dev.log10_Mmin, dev.log10_M1, dev.alpha, dev.epsilon,
                dev.sigma_lambda, dev.z_pivot);

  double const lnMs[] = {std::log(1.0e13), std::log(3.16e13),
                         std::log(1.0e14), std::log(1.0e15)};
  double const zts[] = {0.10, 0.45, 0.80};
  double const lts[] = {0.0, 0.5, 1.0, 5.0, 20.0};

  for (double lnM : lnMs)
    for (double zt : zts)
      for (double lt : lts) {
        CHECK(dev(lt, lnM, zt) == Approx(host(lt, lnM, zt)).epsilon(PORT_TOL));
      }

  // lt < 0 is unphysical support -> 0 on both sides.
  CHECK(dev(-1.0, lnMs[2], zts[0]) == 0.0);
  CHECK(host(-1.0, lnMs[2], zts[0]) == 0.0);
}

TEST_CASE("full_ltmz device PlobEmgDevice interpolates and composes correctly")
{
  y3_cuda_des_y3::PlobEmgDevice p;
  p.n = 4;
  double const z[] = {0.10, 0.30, 0.50, 0.80};
  double const a_mu[] = {1.0, 1.2, 1.5, 2.0};
  double const b_mu[] = {0.10, 0.12, 0.15, 0.20};
  double const a_sig[] = {0.80, 0.82, 0.85, 0.90};
  double const b_sig[] = {2.0, 2.2, 2.5, 3.0};
  double const a_tau[] = {0.30, 0.32, 0.35, 0.40};
  double const b_tau[] = {0.05, 0.06, 0.07, 0.08};
  double const a_fprj[] = {1.0, 1.0, 1.1, 1.2};
  double const b_fprj[] = {0.9, 0.9, 0.85, 0.80};
  for (int i = 0; i != 4; ++i) {
    p.z[i] = z[i];
    p.a_mu[i] = a_mu[i]; p.b_mu[i] = b_mu[i];
    p.a_sig[i] = a_sig[i]; p.b_sig[i] = b_sig[i];
    p.a_tau[i] = a_tau[i]; p.b_tau[i] = b_tau[i];
    p.a_fprj[i] = a_fprj[i]; p.b_fprj[i] = b_fprj[i];
  }

  // lin(): exact node hit, hand-computed midpoint interpolation, and
  // clamping below/above the node range (Interp1D::clamp semantics).
  CHECK(p.lin(a_mu, 0.30) == Approx(1.2).epsilon(1e-14));
  double const t = (0.20 - 0.10) / (0.30 - 0.10);
  CHECK(p.lin(a_mu, 0.20) == Approx(a_mu[0] + t * (a_mu[1] - a_mu[0])).epsilon(1e-14));
  CHECK(p.lin(a_mu, -5.0) == Approx(a_mu[0]).epsilon(1e-14));
  CHECK(p.lin(a_mu, 5.0) == Approx(a_mu[3]).epsilon(1e-14));

  // mu/sigma/tau/fprj must compose the interpolated coefficients exactly
  // per the closed forms documented in full_ltmz_device_kernels.cuh and
  // mirrored in plob_ltr_emg_t.hh (PlobLtrEMG_t).
  for (double zq : {0.15, 0.30, 0.62}) {
    for (double ltr : {5.0, 20.0, 60.0}) {
      CHECK(p.mu(ltr, zq) ==
            Approx(p.lin(a_mu, zq) + p.lin(b_mu, zq) * ltr).epsilon(1e-13));
      CHECK(p.sigma(ltr, zq) ==
            Approx(p.lin(b_sig, zq) * std::pow(ltr, p.lin(a_sig, zq)))
              .epsilon(1e-13));
      CHECK(p.tau(ltr, zq) ==
            Approx(p.lin(b_tau, zq) / std::pow(ltr, p.lin(a_tau, zq)))
              .epsilon(1e-13));
      double const expected_fprj = std::min(
        1.0, p.lin(b_fprj, zq) / std::pow(1.0 + std::exp(-ltr),
                                          p.lin(a_fprj, zq)));
      CHECK(p.fprj(ltr, zq) == Approx(expected_fprj).epsilon(1e-13));
    }
  }
}

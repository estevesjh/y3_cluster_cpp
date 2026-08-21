// Unit tests for the shared richness/photo-z kernel headers
// (src/models/richness_kernel_t.hh, src/models/plob_ltr_emg_t.hh) used by
// NumCountsFullLtmz and Shear1hFullLtmz -- the "kernels inside the main
// modules" review flagged as untested: the existing composition tests
// (numcounts_full_ltmz_test, shear1h_full_ltmz_test) only spot-check a few
// golden values of the fully-composed operator(), never the kernel's own
// mathematical properties in isolation.
//
// Notation follows docs/source/science/index.md ("Selection functions"),
// not the earlier K_i/K_j shorthand: the observed-richness kernel is
// script-S, S_i(lambda_tr, z); the observed-redshift kernel is script-S,
// S_j(z); the richness selection function (after marginalising over the
// mass-richness relation) is plain S_i(M, z). Code identifiers
// (RichnessKernel_t, richness_zkernel) are unchanged -- CLAUDE.md's
// pipeline-language migration note says the paper notation lands in docs
// first, code identifiers later.
//
// PlobLtrEMG_t::operator()(lob, ltr, z) is the raw analytical
// P(lambda_ob | lambda_tr, z) kernel itself (the Gaussian-plus-EMG-tail
// density, science/index.md "Observed richness: the projection kernel") --
// its header comment says "for debugging / tests", but nothing before this
// file ever called it. RichnessKernel_t::operator() is the bin-integrated
// S_i(lambda_tr, z) = int_{lob_min}^{lob_max} P(lob | ltr, z) d(lob), built
// from a SEPARATE code path (CDF differencing via rk_detail::phi/F_EMG,
// not the density formula). This file cross-checks the two paths against
// each other and against calculus (density = d(CDF)/d(lob)), which a
// shared bug in the density formula alone, or the CDF formula alone,
// cannot pass by accident.
#include "catch2/catch.hpp"

#include "models/plob_ltr_emg_t.hh"
#include "models/richness_kernel_t.hh"

#include "cosmosis/datablock/datablock.hh"

#include <cmath>
#include <vector>

using y3_cluster::PlobLtrEMG_t;
using y3_cluster::RichnessKernel_t;
using y3_cluster::richness_zkernel;

namespace {
  constexpr double REL_TOL = 1.0e-3;

  cosmosis::DataBlock
  make_plob_config()
  {
    cosmosis::DataBlock cfg;
    std::vector<double> const z{0.10, 0.45, 0.80};
    auto const flat = [](double v) { return std::vector<double>{v, v, v}; };
    cfg.put_val("plob_ltr_params", "z", z);
    cfg.put_val("plob_ltr_params", "a_mu", flat(0.2));
    cfg.put_val("plob_ltr_params", "b_mu", flat(1.05));
    cfg.put_val("plob_ltr_params", "a_sig", flat(0.55));
    cfg.put_val("plob_ltr_params", "b_sig", flat(0.9));
    cfg.put_val("plob_ltr_params", "a_tau", flat(0.30));
    cfg.put_val("plob_ltr_params", "b_tau", flat(0.02));
    cfg.put_val("plob_ltr_params", "a_fprj", flat(1.2));
    cfg.put_val("plob_ltr_params", "b_fprj", flat(0.65));
    return cfg;
  }

  // CDF pieces exactly as RichnessKernel_t builds them internally (same
  // rk_detail free functions), exposed here so this file can integrate/
  // differentiate them independently of RichnessKernel_t's bin-difference
  // wiring.
  double
  cdf_lob(double lob, PlobLtrEMG_t const& plob, double ltr, double z)
  {
    double const mu = plob.mu(ltr, z);
    double const sigma = plob.sigma(ltr, z);
    double const tau = plob.tau(ltr, z);
    double const fprj = std::min(1.0, plob.fprj(ltr, z));
    double const gauss = y3_cluster::rk_detail::phi((lob - mu) / sigma);
    double const emg = y3_cluster::rk_detail::F_EMG(lob, mu, sigma, tau);
    return (1.0 - fprj) * gauss + fprj * emg;
  }

  // Composite Simpson's rule on a uniform grid -- no dependency beyond
  // <cmath>, adequate for a smooth EMG density over a finite bin.
  double
  simpson(std::vector<double> const& y, double h)
  {
    std::size_t const n = y.size() - 1; // even
    double s = y.front() + y.back();
    for (std::size_t i = 1; i != n; ++i) s += y[i] * (i % 2 == 0 ? 2.0 : 4.0);
    return s * h / 3.0;
  }
}

TEST_CASE("PlobLtrEMG_t::operator() (the analytical P(lob|ltr,z) kernel) "
          "equals the finite-difference derivative of the CDF pieces "
          "RichnessKernel_t actually uses")
{
  auto cfg = make_plob_config();
  PlobLtrEMG_t const plob(cfg);

  for (auto const& point : {std::pair{8.0, 0.30}, std::pair{25.0, 0.55},
                            std::pair{60.0, 0.70}}) {
    double const ltr = point.first, z = point.second;
    double const mu = plob.mu(ltr, z);
    double const sigma = plob.sigma(ltr, z);

    for (double lob : {mu - 2.0 * sigma, mu, mu + 1.5 * sigma,
                       mu + 6.0 * sigma}) {
      double const h = 1.0e-4 * std::max(1.0, sigma);
      double const dcdf = (cdf_lob(lob + h, plob, ltr, z) -
                           cdf_lob(lob - h, plob, ltr, z)) / (2.0 * h);
      double const density = plob(lob, ltr, z);
      CHECK(dcdf == Approx(density).epsilon(1e-4).margin(1e-8));
    }
  }
}

TEST_CASE("RichnessKernel_t's bin-integrated S_i(ltr,z) equals a direct "
          "Simpson's-rule integral of PlobLtrEMG_t's density over the bin")
{
  auto cfg = make_plob_config();
  PlobLtrEMG_t const plob(cfg);

  struct Bin { double lo, hi; };
  for (Bin const bin : {Bin{20.0, 30.0}, Bin{30.0, 45.0}, Bin{45.0, 60.0},
                        Bin{60.0, 200.0}}) {
    RichnessKernel_t const s_i(bin.lo, bin.hi);
    for (auto const& point : {std::pair{22.0, 0.30}, std::pair{40.0, 0.50},
                              std::pair{90.0, 0.65}}) {
      double const ltr = point.first, z = point.second;
      double const from_cdf_diff = s_i(ltr, z, plob);

      constexpr int N = 4000; // even, for Simpson's rule
      double const h = (bin.hi - bin.lo) / N;
      std::vector<double> y(N + 1);
      for (int k = 0; k <= N; ++k)
        y[k] = plob(bin.lo + k * h, ltr, z);
      double const from_density_integral = simpson(y, h);

      CHECK(from_cdf_diff == Approx(from_density_integral).epsilon(REL_TOL));
    }
  }
}

TEST_CASE("The observed-richness kernel's CDF saturates to exactly 0/1 far "
          "outside its support (total probability is conserved)")
{
  // science/index.md: "the kernel is *not* normalised to unity over
  // lambda_ob for f_prj > 0 only because Delta_prj >= 0; total probability
  // is conserved by construction through the decomposition" -- i.e. the
  // CDF -> 1 as lob -> +infinity and -> 0 as lob -> -infinity exactly,
  // regardless of f_prj.
  auto cfg = make_plob_config();
  PlobLtrEMG_t const plob(cfg);
  double const ltr = 25.0, z = 0.5;
  double const mu = plob.mu(ltr, z);
  double const sigma = plob.sigma(ltr, z);
  double const tau = plob.tau(ltr, z);

  // The Gaussian piece decays on the sigma scale, but the EMG piece's
  // one-sided exponential tail decays on the SEPARATE scale 1/tau (tau can
  // be small here, e.g. ~0.0085 at this (ltr, z) -- a decay length of
  // ~118, far larger than a handful of sigma ~ 6). Both must be cleared
  // for the CDF to actually reach 1 to high precision.
  double const far = std::max(60.0 * sigma, 60.0 / tau);
  CHECK(cdf_lob(mu + far, plob, ltr, z) == Approx(1.0).margin(1e-9));
  CHECK(cdf_lob(mu - far, plob, ltr, z) == Approx(0.0).margin(1e-9));

  // A full-support bin must therefore reproduce S_i -> 1.
  RichnessKernel_t const s_i_full(mu - far, mu + far);
  CHECK(s_i_full(ltr, z, plob) == Approx(1.0).margin(1e-9));
}

TEST_CASE("The observed-redshift kernel S_j(z) is the plain Gaussian-CDF "
          "difference and saturates the same way")
{
  double const sigma_z = 0.03;
  // The real DES Y3 bin edges are only ~2.5 sigma_z wide by design (a
  // narrow photo-z window), so S_j there is genuinely ~0.988, not ~1 --
  // that is physics, not a bug. The saturation claim needs a
  // deliberately wide synthetic bin instead.
  CHECK(richness_zkernel(0.425, 0.35, 0.50, sigma_z) ==
        Approx(0.9875806693).epsilon(REL_TOL));
  CHECK(richness_zkernel(0.425 + 60.0 * sigma_z, 0.35, 0.50, sigma_z) ==
        Approx(0.0).margin(1e-9));
  CHECK(richness_zkernel(0.425, 0.425 - 60.0 * sigma_z,
                         0.425 + 60.0 * sigma_z, sigma_z) ==
        Approx(1.0).margin(1e-9));
}

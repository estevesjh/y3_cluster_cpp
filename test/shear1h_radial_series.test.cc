// Unit tests for the radial_series C++ backend table + series core
// (src/pipelines/des_y3/shear_1h2h/cpp/0d).
//
// Golden values were computed 2026-08-12 from the committed
// data/radial_series text tables with an independent Python bilinear /
// linear interpolator (see the python/ directory of the backend), so
// they pin both the table contents and the GSL interpolation semantics.
// The amplitude cross-check drives the composite A0(y) * u_mis through
// the production NFW_DSIGMA_MIS reader, so the fixed conventions of the
// two paths cannot drift apart silently.
//
// Requires Y3_CLUSTER_CPP_DIR to point at the source tree (data/).
#include "catch2/catch.hpp"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include "pipelines/des_y3/shear_1h2h/cpp/0d/shear1h_radial_series_t.hh"
#include "pipelines/shared/sel_gl_weights.hh"

#include <array>
#include <cmath>
#include <vector>

using y3_cluster::des_y3::A0_of_y;
using y3_cluster::des_y3::RadialSeriesTable;
using y3_cluster::des_y3::y_of_lnM;

namespace {
  constexpr double SERIES_REL_TOL = 1.0e-3;

  double
  direct_profile(RadialSeriesTable const& tab, double R, double r_mis,
                 double y, double f_mis, double rho_ref)
  {
    return A0_of_y(y, rho_ref) *
           tab.u_mix(0, std::log(R) - y, std::log(r_mis) - y, f_mis);
  }

  template<std::size_t N>
  double
  direct_population(RadialSeriesTable const& tab, double R, double r_mis,
                    double norm, double ybar,
                    std::array<double, N> const& delta_y,
                    std::array<double, N> const& probability, double f_mis,
                    double rho_ref)
  {
    double out = 0.0;
    for (std::size_t i = 0; i != N; ++i)
      out += probability[i] * direct_profile(
                                tab, R, r_mis, ybar + delta_y[i], f_mis,
                                rho_ref);
    return norm * out;
  }

  template<std::size_t N>
  double
  central_moment(std::array<double, N> const& delta_y,
                 std::array<double, N> const& probability, int order)
  {
    double out = 0.0;
    for (std::size_t i = 0; i != N; ++i)
      out += probability[i] * std::pow(delta_y[i], order);
    return out;
  }
}

TEST_CASE("radial_series NFW scale and amplitude factorisation is exact")
{
  // r_s is proportional to M^(1/3), while A0 is proportional to r_s.
  // These identities are the separation that makes the U_ell tables
  // independent of the MCMC sample.
  double const lnM = 33.2;
  // The identities hold for any rho_ref; use the fiducial rho_m0.
  double const rho_ref = 0.3096 * 2.77533742639e+11;
  for (double delta_y : {-0.4, -0.05, 0.1, 0.7}) {
    double const y0 = y_of_lnM(lnM, rho_ref);
    double const y1 = y_of_lnM(lnM + 3.0 * delta_y, rho_ref);
    CHECK(y1 == Approx(y0 + delta_y).epsilon(1e-12));
    CHECK(A0_of_y(y1, rho_ref) ==
          Approx(A0_of_y(y0, rho_ref) * std::exp(delta_y)).epsilon(1e-12));
  }
}

TEST_CASE("radial_series table interpolation matches golden values")
{
  RadialSeriesTable const tab;

  struct Golden {
    double lnx, lnxm, u0, u2, u3, c0;
  };
  // Bilinear (mis) / linear (cen) on the committed text tables.
  Golden const golden[] = {
    {0.1234, -0.987, 1.352591378821e-01, 1.588740771068e-01,
     5.981286639335e-02, 2.604112758336e-01},
    {1.5, 0.25, 4.439326085706e-02, 9.551480617341e-02,
     5.754488821577e-02, 7.476007404807e-02},
    {-2.0, -1.5, 2.710753422364e-02, 1.790178350799e-02,
     5.566988339746e-03, 4.771010214995e-01},
  };
  for (auto const& g : golden) {
    CHECK(tab.u_mis(0, g.lnx, g.lnxm) == Approx(g.u0).epsilon(1e-9));
    CHECK(tab.u_mis(2, g.lnx, g.lnxm) == Approx(g.u2).epsilon(1e-9));
    CHECK(tab.u_mis(3, g.lnx, g.lnxm) == Approx(g.u3).epsilon(1e-9));
    CHECK(tab.u_cen(0, g.lnx) == Approx(g.c0).epsilon(1e-9));
  }
}

TEST_CASE("radial_series series assembly matches golden values")
{
  RadialSeriesTable const tab;

  // Synthetic but representative moments; goldens from the Python
  // bilinear reference with identical inputs.
  double const R = 0.83625;
  double const r_mis = 0.17 * std::pow(52.5 / 100.0, 0.2);
  double const norm = 141.4;
  double const ybar = std::log(0.30);
  double const mu2 = 0.021;
  double const mu3 = -0.0035;
  double const f_mis = 0.22;
  // UNIFIED rho_m convention: rho_ref = Omega_m rho_crit,0 drives BOTH
  // boundary and amplitude (regenerated pins, 2026-08-24).
  double const rho_ref = 0.3096 * 2.77533742639e+11;

  CHECK(tab.series(R, r_mis, norm, ybar, mu2, mu3, f_mis, rho_ref, 2) ==
        Approx(4.783606100876e+03).epsilon(1e-9));
  CHECK(tab.series(R, r_mis, norm, ybar, mu2, mu3, f_mis, rho_ref, 3) ==
        Approx(4.768781628135e+03).epsilon(1e-9));
}

TEST_CASE("radial_series mixture decomposition has exact endpoints")
{
  RadialSeriesTable const tab;
  double const lnx = 0.371;
  double const lnxm = -0.824;

  for (int ell : {0, 1, 2, 3}) {
    double const cen = tab.u_cen(ell, lnx);
    double const mis = tab.u_mis(ell, lnx, lnxm);
    CHECK(tab.u_mix(ell, lnx, lnxm, 0.0) == Approx(cen).epsilon(1e-13));
    CHECK(tab.u_mix(ell, lnx, lnxm, 1.0) == Approx(mis).epsilon(1e-13));

    // The target-cluster mixture must remain affine in f_mis (both
    // components share rho_ref inside A0 -- unified rho_m convention).
    double const f_mis = 0.37;
    double const expected = (1.0 - f_mis) * cen + f_mis * mis;
    CHECK(tab.u_mix(ell, lnx, lnxm, f_mis) ==
          Approx(expected).epsilon(1e-13));
  }
}

TEST_CASE("radial_series second-order decomposition matches a direct symmetric population")
{
  RadialSeriesTable const tab;
  double const r_mis = 0.15;
  double const norm = 137.0;
  double const ybar = std::log(0.30);
  double const f_mis = 0.22;
  double const rho_ref = 0.3096 * 2.77533742639e+11;

  // A narrow symmetric population has mu_3 = 0, isolating U_2.  The direct
  // side evaluates U_0 separately at every mass; the series side evaluates
  // U_0 and U_2 only at the effective scale radius.
  std::array<double, 3> const dy = {-0.04, 0.0, 0.04};
  std::array<double, 3> const p = {0.25, 0.50, 0.25};
  double const mu2 = central_moment(dy, p, 2);
  double const mu3 = central_moment(dy, p, 3);
  CHECK(mu3 == Approx(0.0).margin(1e-15));

  for (double R : {0.20, 0.84, 3.0, 10.0}) {
    double const direct = direct_population(
      tab, R, r_mis, norm, ybar, dy, p, f_mis, rho_ref);
    double const series = tab.series(R, r_mis, norm, ybar, mu2, mu3,
                                     f_mis, rho_ref, 2);
    CHECK(series == Approx(direct).epsilon(SERIES_REL_TOL));
  }
}

TEST_CASE("radial_series third-order decomposition matches a direct skewed population")
{
  RadialSeriesTable const tab;
  double const r_mis = 0.15;
  double const norm = 137.0;
  double const ybar = std::log(0.30);
  double const f_mis = 0.22;
  double const rho_ref = 0.3096;

  // The weighted displacement is exactly zero, but mu_3 is nonzero.  This
  // exercises U_3 independently of the symmetric-population test above.
  std::array<double, 2> const dy = {-0.03, 0.06};
  std::array<double, 2> const p = {2.0 / 3.0, 1.0 / 3.0};
  CHECK(central_moment(dy, p, 1) == Approx(0.0).margin(1e-15));
  double const mu2 = central_moment(dy, p, 2);
  double const mu3 = central_moment(dy, p, 3);
  REQUIRE(std::abs(mu3) > 0.0);

  for (double R : {0.20, 0.84, 3.0, 10.0}) {
    double const direct = direct_population(
      tab, R, r_mis, norm, ybar, dy, p, f_mis, rho_ref);
    double const series = tab.series(R, r_mis, norm, ybar, mu2, mu3,
                                     f_mis, rho_ref, 3);
    CHECK(series == Approx(direct).epsilon(SERIES_REL_TOL));
  }
}

TEST_CASE("radial_series amplitude matches the production NFW_DSIGMA_MIS")
{
  RadialSeriesTable const tab;
  y3_cluster::NFW_DSIGMA_MIS dsigma_mis(4.0, 2.77533742639e+11,
                                        y3_cluster::GAMMA);
  // UNIFIED rho_m convention: reader and family share one rho_ref.
  double const rho_ref = 0.3096 * 2.77533742639e+11;
  dsigma_mis.set_rho_ref(rho_ref);

  // A0(y; rho_ref) * U0(ln x, ln x_mis) must reproduce the production
  // reader (which interpolates the original data/nfw_off_center table)
  // at the level the generator measured for U0 fidelity (<= ~5e-4 over
  // the physical window).
  struct Point {
    double R, r_mis, lnM;
  };
  Point const pts[] = {{0.5, 0.14, 32.0}, {1.5, 0.15, 33.5},
                       {3.0, 0.16, 34.5}};
  for (auto const& p : pts) {
    double const y = y_of_lnM(p.lnM, rho_ref);
    double const mine =
      A0_of_y(y, rho_ref) *
      tab.u_mis(0, std::log(p.R) - y, std::log(p.r_mis) - y);
    double const prod = dsigma_mis(p.R, p.r_mis, p.lnM);
    CHECK(mine == Approx(prod).epsilon(5e-4));
  }
}

TEST_CASE("radial_series miscentred table recovers the centred limit")
{
  RadialSeriesTable const tab;
  // For x >> x_mis the gamma-averaged profile reduces to the centred
  // one; at the table's smallest x_mis (~0.01) the suppression at
  // x >= 1 is a few 1e-4.
  for (double lnx : {0.0, 1.0, 2.0}) {
    double const ratio = tab.u_mis(0, lnx, -4.625) / tab.u_cen(0, lnx);
    CHECK(ratio == Approx(1.0).epsilon(2e-3));
  }
}

namespace {
  // Minimal synthetic DataBlock exercising the full Shear1hRadialSeries
  // class (issue #22: this backend used to silently ignore
  // one_halo_physical_density -- its Python twin, shear1h_radial_series.py,
  // already implemented it). Mirrors shear1h_gl.test.cc's make_*_block
  // pattern (same SelGlWeights core, same required sections), minus
  // haloModel/dSigma_nfw (Shear1hGl-only) plus this module's own options
  // (ell_max, lob_centers).
  double constexpr RS_ZT_LO = 0.10;
  double constexpr RS_ZT_HI = 0.50;
  double constexpr RS_LNM_LO = 30.0;
  double constexpr RS_LNM_HI = 36.0;
  int constexpr RS_N_Z = 6;
  int constexpr RS_N_LNM = 8;

  cosmosis::DataBlock
  rs_make_cfg_block(std::vector<double> const& lob_centers)
  {
    cosmosis::DataBlock cfg;
    cfg.put_val("Shear1hRadialSeries", "n_lnm", RS_N_LNM);
    cfg.put_val("Shear1hRadialSeries", "n_z", RS_N_Z);
    cfg.put_val("Shear1hRadialSeries", "zt_low", RS_ZT_LO);
    cfg.put_val("Shear1hRadialSeries", "zt_high", RS_ZT_HI);
    cfg.put_val("Shear1hRadialSeries", "lnm_low", RS_LNM_LO);
    cfg.put_val("Shear1hRadialSeries", "lnm_high", RS_LNM_HI);
    cfg.put_val("Shear1hRadialSeries", "lob_centers", lob_centers);
    return cfg;
  }

  cosmosis::DataBlock
  rs_make_sample_block(double f_mis, double tau_mis)
  {
    cosmosis::DataBlock s;
    s.put_val("cosmological_parameters", "omega_m", 0.3);
    s.put_val("cosmological_parameters", "omega_lambda", 0.7);
    s.put_val("cosmological_parameters", "omega_k", 0.0);
    s.put_val("cosmological_parameters", "omega_nu", 0.0);
    s.put_val("cosmological_parameters", "h0", 0.7);

    s.put_val("cluster_abundance", "hmf_s", 0.0);
    s.put_val("cluster_abundance", "hmf_q", 1.0);
    s.put_val("mass_function", "m_h", std::vector<double>{1.0e13, 1.0e15});
    s.put_val("mass_function", "z", std::vector<double>{0.0, 1.0});
    s.put_val("mass_function", "dndlnmh",
              cosmosis::ndarray<double>(std::vector<double>(4, 7.0),
                                        {2, 2}));

    s.put_val("distances", "z", std::vector<double>{0.0, 1.0});
    s.put_val("distances", "d_a", std::vector<double>{1500.0, 1500.0});

    s.put_val("average_sigma_crit_inv", "zlense", std::vector<double>{0.0, 1.0});
    s.put_val("average_sigma_crit_inv", "sci_average",
              std::vector<double>{3.0e-4, 3.0e-4});

    s.put_val("sel_function", "lnM", std::vector<double>{RS_LNM_LO, RS_LNM_HI});
    s.put_val("sel_function", "z", std::vector<double>{RS_ZT_LO, RS_ZT_HI});
    std::vector<double> s_stack;
    for (double c : {2.0, 3.5})
      for (int i = 0; i != 4; ++i) s_stack.push_back(c);
    s.put_val("sel_function", "S_stack",
              cosmosis::ndarray<double>(s_stack, {2, 2, 2})); // (bin,z,lnm)

    s.put_val("haloModel", "rho_m_ref", 0.3096 * 2.77533742639e+11);

    s.put_val("miscentering", "f_mis", f_mis);
    s.put_val("miscentering", "tau_mis", tau_mis);
    return s;
  }
}

TEST_CASE("Shear1hRadialSeries one_halo_physical_density matches an independently rebuilt (1+z)^2/R(1+z) reference")
{
  std::vector<double> const lob_centers{25.0, 130.0};
  auto cfg = rs_make_cfg_block(lob_centers);
  auto s = rs_make_sample_block(/*f_mis=*/0.22, /*tau_mis=*/0.17);

  y3_cluster::des_y3::Shear1hRadialSeries mod(cfg);
  s.put_val("haloModel", "one_halo_physical_density", 0);
  mod.set_sample(s);

  double const rho_ref = s.view<double>("haloModel", "rho_m_ref");

  for (int b : {0, 1}) {
    for (double R : {0.5, 2.0, 8.0}) {
      double const off = mod.evaluate({static_cast<double>(b), R})[0];

      s.replace_val("haloModel", "one_halo_physical_density", 1);
      mod.set_sample(s);
      double const on = mod.evaluate({static_cast<double>(b), R})[0];
      s.replace_val("haloModel", "one_halo_physical_density", 0);
      mod.set_sample(s);

      CHECK(std::isfinite(on));
      CHECK(on > 0.0);
      CHECK(on != Approx(off));

      // Independent reference: rebuild the SAME z-resolved weights with
      // z_amp_power=2 (the amplitude half) via a standalone SelGlWeights,
      // and query the table at R*(1+z_eff) (the radius half) -- the two
      // halves of the identity documented in
      // docs/source/observables/shear_halo.md, computed here without
      // going through Shear1hRadialSeries::evaluate() at all.
      y3_pipelines::SelGlWeights ref(cfg, "Shear1hRadialSeries");
      ref.build_weights(s, /*include_sci=*/true, /*z_amp_power=*/2.0);
      auto const& xs = ref.lnm_x();
      auto const& ws = ref.lnm_w();
      auto const& Wb = ref.weights(b);
      std::vector<double> y(xs.size());
      for (std::size_t k = 0; k != xs.size(); ++k)
        y[k] = y3_cluster::des_y3::y_of_lnM(xs[k], rho_ref);
      double n0 = 0.0, n1 = 0.0;
      for (std::size_t k = 0; k != xs.size(); ++k) {
        n0 += ws[k] * Wb[k];
        n1 += ws[k] * Wb[k] * y[k];
      }
      double const ybar = n1 / n0;
      double m2 = 0.0;
      for (std::size_t k = 0; k != xs.size(); ++k) {
        double const d = y[k] - ybar;
        m2 += ws[k] * Wb[k] * d * d;
      }
      double const mu2 = m2 / n0;
      double const q = 1.0 + ref.z_eff(b);
      double const r_mis = 0.17 * y3_pipelines::R_lambda(
                                    lob_centers[b % lob_centers.size()]);

      RadialSeriesTable const tab;
      double const expected = tab.series(R * q, r_mis * q, n0, ybar, mu2,
                                         0.0, 0.22, rho_ref, 2);
      CHECK(on == Approx(expected).epsilon(1e-9));
    }
  }
}

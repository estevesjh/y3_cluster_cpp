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

#include "pipelines/des_y3/shear_1h2h/cpp/0d/shear1h_radial_series_t.hh"

#include <array>
#include <cmath>

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

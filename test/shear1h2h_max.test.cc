// Unit tests for the shear1h2h_max C++ backend (the traditional 1h+2h
// "max model" shear -- src/pipelines/des_y3/observables/shear_1h2h/
// fast_mass/cpp/Shear1h2hMax.cc).
//
// Unlike shear1h_radial_series (RadialSeriesTable) or Shear1hFastMass
// (SelGLCore in models/n_operator_sel_gl_t.hh), Shear1h2hMax.cc has NO
// header-only core to instantiate directly: its class is defined
// entirely inside the module .cc, private to that translation unit, and
// its constructor needs a full CosmoSIS DataBlock wired up with
// HMF_t/DV_DO_DZ_t/OMEGA_Z_DES/SelFunction_t plus real haloModel tables
// (dSigma_nfw, bias, dSigma_hh) -- i.e. effectively the whole
// real_pipeline_extract.ini with compute_lensing_2h = T. That is out of
// scope for a unit test, and per CLAUDE.md's dSigma_hh caveat
// (docs/dsigma_hh_debug_flag.md: 60% NaN by construction, degenerate z
// axis, dummy exclusion parameters), any number derived that way would
// be pinning a known-buggy table rather than testing physics.
//
// This file instead exercises, with the SAME public primitives the
// module composes (Interp2D + NFW_DSIGMA_MIS) and a byte-for-byte mirror
// of the (lnM, z) accumulation in Shear1h2hMax::evaluate():
//
//   * the one-halo mixture (1-f_mis)*dSigma_nfw + f_mis*dSigma_mis --
//     textually identical to Shear1hFastMass.cc's mixture -- is
//     well-defined, finite, non-negative, and has exact f_mis endpoints;
//   * the sanitize-NaN-to-zero-before-Interp2D technique the module uses
//     for haloModel/dSigma_hh actually prevents NaN propagation, by
//     contrasting the SAME Interp2D constructor overload
//     (make_sanitized_hh's) with and without sanitizing a table with a
//     realistic NaN fraction;
//   * the max(1h, b*2h) composition: forcing the (sanitized) two-halo
//     term to zero recovers the pure one-halo stack exactly (the
//     documented max(1h, 0) = 1h identity), and the composition
//     genuinely picks up the two-halo branch when it is larger, so both
//     sides of the max() are exercised.
//
// None of these numbers are golden physics values pinned against
// dSigma_hh; the synthetic tables below play the same role as
// interp_2d.test.cc's `fxy` -- a stand-in shape, not a fiducial
// prediction. Most checks are exact algebraic identities, so they use a
// tight epsilon rather than the project's default 1e-3 (there is no
// approximate empirical reference here to size 1e-3 against).
//
// Requires Y3_CLUSTER_CPP_DIR to point at the source tree (NFW_DSIGMA_MIS
// reads data/nfw_off_center/*).
#include "catch2/catch.hpp"

#include "models/nfw_dsigma_mis.hh"
#include "utils/interp_2d.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

using y3_cluster::GAMMA;
using y3_cluster::Interp2D;
using y3_cluster::NFW_DSIGMA_MIS;

namespace {
  // Synthetic dSigma_nfw(R, lnM) table -- not real halo-model physics (no
  // fiducial dump in this tree has haloModel/dSigma_nfw from a
  // compute_lensing_2h = T run), just a smooth, positive, monotonic
  // stand-in for "the production centred profile", same role as
  // interp_2d.test.cc's fxy synthetic function.
  double
  dcen_fn(double R, double lnM)
  {
    return 5.0 * std::exp(0.5 * (lnM - 33.0)) / (0.2 + R);
  }

  Interp2D
  make_dcen_table()
  {
    std::vector<double> const R_nodes = {0.3, 1.0, 3.0};
    std::vector<double> const M_nodes = {31.0, 33.0, 35.0};
    std::vector<double> zs(R_nodes.size() * M_nodes.size());
    // Column-major, x (R) fastest -- the layout Interp2D's vector ctor
    // wants (same convention Shear1h2hMax.cc documents for dSigma_hh).
    for (std::size_t j = 0; j != M_nodes.size(); ++j)
      for (std::size_t i = 0; i != R_nodes.size(); ++i)
        zs[j * R_nodes.size() + i] = dcen_fn(R_nodes[i], M_nodes[j]);
    return Interp2D(R_nodes, M_nodes, zs);
  }

  // Synthetic haloModel/dSigma_hh(R, z) table with the SAME defect shape
  // documented in docs/dsigma_hh_debug_flag.md: NaN at low R (here the
  // entire R = 0.5 row -- 4/9 of this small table, the same "undefined
  // at low R" pattern as the real table's ~60%). Built with the identical
  // Interp2D(xs, ys, vector<double>) overload
  // Shear1h2hMax::make_sanitized_hh uses -- that overload does NO
  // finiteness check (only the Point3D-vector overload does; see
  // src/utils/interp_2d.hh), which is exactly why the module must
  // sanitize before constructing it.
  std::vector<double> const R_HH = {0.5, 1.0, 2.0};
  std::vector<double> const Z_HH = {0.2, 0.4, 0.6};
  double const NAN_D = std::numeric_limits<double>::quiet_NaN();

  // [R][z], for readability; flattened below into the column-major
  // (x = R fastest) layout the Interp2D vector ctor wants.
  double const RAW_HH[3][3] = {
    {NAN_D, NAN_D, NAN_D}, // R = 0.5 -- undefined at low R, like the real table
    {NAN_D, 0.8, 1.0},     // R = 1.0
    {1.2, 1.5, 2.0},       // R = 2.0
  };

  std::vector<double>
  flatten_column_major(double const tbl[3][3])
  {
    std::vector<double> zs(9);
    for (std::size_t j = 0; j != 3; ++j)
      for (std::size_t i = 0; i != 3; ++i)
        zs[j * 3 + i] = tbl[i][j];
    return zs;
  }

  Interp2D
  make_raw_hh()
  {
    return Interp2D(R_HH, Z_HH, flatten_column_major(RAW_HH));
  }

  // Shear1h2hMax::make_sanitized_hh's own recipe: replace every
  // non-finite table entry with 0 BEFORE building the Interp2D, which is
  // exact for a max model since max(1h, 0) = 1h where the 2h term is
  // undefined.
  Interp2D
  make_sanitized_hh()
  {
    auto zs = flatten_column_major(RAW_HH);
    for (auto& v : zs)
      if (!std::isfinite(v)) v = 0.0;
    return Interp2D(R_HH, Z_HH, zs);
  }
}

TEST_CASE("shear1h2h_max one-halo mixture has exact f_mis endpoints and stays bounded")
{
  Interp2D const dcen_tab = make_dcen_table();
  NFW_DSIGMA_MIS dsigma_mis(4.0, 2.77533742639e+11, GAMMA);
  double const omega_m = 0.3096;
  dsigma_mis.set_rho_mult(omega_m);

  double const R = 1.5, lnM = 32.0, r_mis = 0.15;
  double const d_cen = dcen_tab.clamp(R, lnM);
  double const d_mis = dsigma_mis(R, r_mis, lnM);
  REQUIRE(d_cen > 0.0);
  REQUIRE(d_mis > 0.0);

  // Same expression as Shear1h2hMax::evaluate()'s `one[k]` (and, textually,
  // Shear1hFastMass.cc's mixture term): a convex combination of the
  // centred and miscentred profiles.
  auto mixture = [&](double f_mis) {
    return (1.0 - f_mis) * d_cen + f_mis * d_mis;
  };

  CHECK(mixture(0.0) == Approx(d_cen).epsilon(1e-12));
  CHECK(mixture(1.0) == Approx(d_mis).epsilon(1e-12));

  for (double f_mis : {0.1, 0.37, 0.6, 0.9}) {
    double const expected = (1.0 - f_mis) * d_cen + f_mis * d_mis;
    CHECK(mixture(f_mis) == Approx(expected).epsilon(1e-12));
    // Convex combination: must stay within [min, max] of the two terms.
    CHECK(mixture(f_mis) <= std::max(d_cen, d_mis) + 1e-9);
    CHECK(mixture(f_mis) >= std::min(d_cen, d_mis) - 1e-9);
  }
}

TEST_CASE("shear1h2h_max one-halo mixture is finite and non-negative across the sample window")
{
  Interp2D const dcen_tab = make_dcen_table();
  NFW_DSIGMA_MIS dsigma_mis(4.0, 2.77533742639e+11, GAMMA);
  dsigma_mis.set_rho_mult(0.3096);

  // The 1-halo piece is the part of the max model that must be
  // well-defined and NaN-free on its own -- unlike the 2-halo term, it
  // does not depend on the buggy dSigma_hh table at all.
  for (double R : {0.3, 0.8, 1.5, 3.0})
    for (double lnM : {31.0, 32.5, 34.0, 35.0})
      for (double f_mis : {0.0, 0.25, 0.6, 1.0}) {
        double const d_cen = dcen_tab.clamp(R, lnM);
        double const d_mis = dsigma_mis(R, 0.15, lnM);
        double const one = (1.0 - f_mis) * d_cen + f_mis * d_mis;
        CHECK(std::isfinite(one));
        CHECK(one >= 0.0);
      }
}

TEST_CASE("shear1h2h_max sanitizes dSigma_hh NaNs to zero before interpolation")
{
  Interp2D const raw = make_raw_hh();
  Interp2D const clean = make_sanitized_hh();

  // The raw table's NaN corner is returned exactly at the grid node...
  CHECK(std::isnan(raw.clamp(0.5, 0.2)));
  // ...and poisons any cell touching it -- exactly the failure mode the
  // module's comment warns about ("keeps NaN out of the interpolator's
  // stencil").
  CHECK(std::isnan(raw.clamp(0.75, 0.3)));

  // Sanitizing first (Shear1h2hMax::make_sanitized_hh's recipe) replaces
  // that corner with exactly 0...
  CHECK(clean.clamp(0.5, 0.2) == Approx(0.0).margin(1e-12));
  // ...and every interpolated value nearby, and across the whole domain
  // (including the fully-NaN low-R row), is finite -- the values below
  // are the bilinear averages of the sanitized corners, e.g.
  // (0.75, 0.3) sits at the midpoint of the {(0.5,0.2), (1.0,0.2),
  // (0.5,0.4), (1.0,0.4)} cell = mean(0, 0, 0, 0.8) = 0.2.
  CHECK(clean.clamp(0.75, 0.3) == Approx(0.2).epsilon(1e-9));
  CHECK(clean.clamp(1.5, 0.5) == Approx(1.325).epsilon(1e-9));

  for (double R : {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0})
    for (double z : {0.2, 0.3, 0.4, 0.5, 0.6})
      CHECK(std::isfinite(clean.clamp(R, z)));
  // clamp() also protects out-of-range queries (the module always uses
  // ::clamp, never raw ::eval, for exactly this reason).
  CHECK(std::isfinite(clean.clamp(0.1, 0.05)));
}

TEST_CASE("shear1h2h_max phi_max composition: zero two-halo term recovers the pure one-halo stack")
{
  // one[k]: the one-halo term at 2 mass nodes, f_mis = 0 (the exact
  // centred limit certified above) to keep this hand-checkable.
  Interp2D const dcen_tab = make_dcen_table();
  double const R = 1.5;
  std::array<double, 2> const lnm_x = {31.5, 34.5};
  std::array<double, 2> const one = {dcen_tab.clamp(R, lnm_x[0]),
                                     dcen_tab.clamp(R, lnm_x[1])};
  REQUIRE(one[0] >= 0.0);
  REQUIRE(one[1] >= 0.0);

  // Arbitrary positive quadrature weights and bias values: this
  // composition only cares that they multiply into the sum, not their
  // physical origin (in the real module w2d comes from HMF*S_ij*zfac and
  // bias_kq from the haloModel bias table).
  std::array<std::array<double, 2>, 2> const w2d = {{{0.10, 0.05},
                                                      {0.20, 0.15}}};
  std::array<std::array<double, 2>, 2> const bias_kq = {{{0.9, 1.6},
                                                          {1.0, 0.2}}};

  // Byte-for-byte mirror of the (lnM, z) double sum in
  // Shear1h2hMax::evaluate():
  //   acc += w2[k*Nz+q] * std::max(one_k, brow[q]*two[q]);
  auto phi_max_sum = [&](std::array<double, 2> const& two) {
    double acc = 0.0;
    for (std::size_t k = 0; k != 2; ++k)
      for (std::size_t q = 0; q != 2; ++q)
        acc += w2d[k][q] * std::max(one[k], bias_kq[k][q] * two[q]);
    return acc;
  };

  // two[q] = 0 for every z node is exactly what a fully-NaN (hence fully
  // sanitized) dSigma_hh table produces -- the CLAUDE.md-documented
  // "provisional until fixed" scenario for the 2-halo term.
  std::array<double, 2> const two_zero = {0.0, 0.0};
  double const acc = phi_max_sum(two_zero);

  double const row_sum0 = w2d[0][0] + w2d[0][1];
  double const row_sum1 = w2d[1][0] + w2d[1][1];
  // max(one_k, 0) = one_k since one_k >= 0 (checked above), so the whole
  // double sum collapses to the plain 1-halo stack.
  double const expected_1h_only = one[0] * row_sum0 + one[1] * row_sum1;
  CHECK(acc == Approx(expected_1h_only).epsilon(1e-12));
}

TEST_CASE("shear1h2h_max phi_max composition picks the larger of one-halo and biased two-halo terms")
{
  Interp2D const dcen_tab = make_dcen_table();
  Interp2D const clean_hh = make_sanitized_hh();

  double const R = 1.5;
  std::array<double, 2> const lnm_x = {31.5, 34.5};
  std::array<double, 2> const z_x = {0.3, 0.5};

  std::array<double, 2> const one = {dcen_tab.clamp(R, lnm_x[0]),
                                     dcen_tab.clamp(R, lnm_x[1])};
  std::array<double, 2> const two = {clean_hh.clamp(R, z_x[0]),
                                     clean_hh.clamp(R, z_x[1])};

  std::array<std::array<double, 2>, 2> const w2d = {{{0.10, 0.05},
                                                      {0.20, 0.15}}};
  std::array<std::array<double, 2>, 2> const bias_kq = {{{0.9, 1.6},
                                                          {1.0, 0.2}}};

  // Confirm the scenario genuinely exercises both branches of the max():
  // (k=0, q=1) is where the biased two-halo term wins; everywhere else
  // the one-halo term wins. If this drifts (e.g. someone edits the
  // synthetic tables above), the test should fail here, loudly, rather
  // than silently degenerating into "the max model always returns 1h".
  REQUIRE(bias_kq[0][1] * two[1] > one[0]);
  REQUIRE(bias_kq[0][0] * two[0] < one[0]);
  REQUIRE(bias_kq[1][0] * two[0] < one[1]);
  REQUIRE(bias_kq[1][1] * two[1] < one[1]);

  double acc = 0.0;
  for (std::size_t k = 0; k != 2; ++k)
    for (std::size_t q = 0; q != 2; ++q)
      acc += w2d[k][q] * std::max(one[k], bias_kq[k][q] * two[q]);

  double const expected = w2d[0][0] * one[0] +
                          w2d[0][1] * (bias_kq[0][1] * two[1]) +
                          w2d[1][0] * one[1] +
                          w2d[1][1] * one[1];
  CHECK(acc == Approx(expected).epsilon(1e-12));
  CHECK(std::isfinite(acc));
}

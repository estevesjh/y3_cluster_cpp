// Unit tests for the des_y3 Phase 2 fast_mass projection-shear C++ driver
// (src/pipelines/des_y3/shear_projection/fast_mass/cpp/
// ShearPrjFastMass.cc).
//
// ShearPrjFastMass.cc has no fast_mass-specific header of its own -- it is
// a thin CosmoSISScalarEvaluatorModule wrapper that instantiates the
// shared, immutable production core y3_cluster::sp_detail::ShearPrjCore
// (systematics/shear_prj/cpp/sigma_prj_t.hh) directly and publishes both
// dsigma_prj and
// shear_prj from one core pass.  So "the underlying model class" this
// test exercises is that shared core itself, plus the free geometry
// functions and the NFW miscentering profile it is built from.
//
// ShearPrjCore::set_sample() (and therefore evaluate()/dsigma_prj()/
// shear_prj()) needs a full real-pipeline sample: HMF, halo bias,
// xi_nl, distances, average_sigma_crit_inv, and the b_sel_marginalised
// tensor -- i.e. essentially all of docs/figs/real_pipeline_extract.ini
// (see validate_vs_production.py, which replays a dump of exactly that
// rather than faking one).  Reproducing that by hand in a unit test
// would mostly test whether this test got the DataBlock keys right, not
// the physics, so this file instead pins the two DataBlock-free
// building blocks that determine ShearPrjCore's numerics --
// sp_detail::build_theta_grid (the per-slice theta quadrature) and the
// NFW_DSIGMA_MIS "single"-kernel profile it dot-products against -- plus
// a structural check that the core's constructor parses its own wall
// grid the way the ctor promises.
//
// Golden values for build_theta_grid / theta_excl_at_z were computed by
// running this repo's own Python fast_mass port
// (../python/shear_prj_fast_mass.py), which the module's docstring
// documents as a line-for-line port of these exact two functions,
// exercised with the same inputs -- a cross-language identity check in
// the same spirit as shear1h_radial_series.test.cc's golden values.
//
// NFW_DSIGMA_MIS convention note (c=4 vs c=5): ShearPrjCore constructs
// its miscentering kernel as
//   dsigma_mis_.emplace(4.0, 2.77533742639e+11, SINGLE);
//   dsigma_mis_->set_rho_mult(omega_m);
// i.e. concentration c=4, rho_s built from rho_crit * rho_mult (rho_mult
// set to Omega_m -> rho_mean), single-offset kernel.  Per CLAUDE.md,
// this is NOT yet apples-to-apples with the legacy Python reference
// (richness_selection.nfw.NFWMiscentered, c=5).  This test therefore
// does NOT compare against that legacy reference.  Instead its golden
// values come from shared/lensing_profiles.py::
// NfwDsigmaMisProduction(kernel="single"), the repo's already-existing
// c=4/rho_crit-convention-exact replica of this same C++ class (same
// data/nfw_off_center table files, same clamped-bilinear scheme) used
// elsewhere in this repo's validation scripts -- i.e. the C++-family
// convention, not the c=5 one.
//
// Requires Y3_CLUSTER_CPP_DIR to point at the source tree (data/).
#include "catch2/catch.hpp"

#include "cosmosis/datablock/datablock.hh"
#include "models/nfw_dsigma_mis.hh"
#include "pipelines/systematics/shear_prj/cpp/sigma_prj_t.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

using y3_cluster::sp_detail::R_lambda;
using y3_cluster::sp_detail::ShearPrjCore;
using y3_cluster::sp_detail::build_theta_grid;
using y3_cluster::sp_detail::default_lob_centers;
using y3_cluster::sp_detail::theta_excl_at_z;

TEST_CASE("sigma_prj R_lambda and default_lob_centers pin the DES-Y3 richness convention")
{
  CHECK(R_lambda(100.0) == Approx(1.0).epsilon(1e-12));
  CHECK(R_lambda(25.0) == Approx(std::pow(0.25, 0.2)).epsilon(1e-12));

  auto const& centres = default_lob_centers();
  REQUIRE(centres.size() == 4);
  CHECK(centres[0] == Approx(25.0));
  CHECK(centres[1] == Approx(37.5));
  CHECK(centres[2] == Approx(52.5));
  CHECK(centres[3] == Approx(130.0));
}

TEST_CASE("sigma_prj theta grid (build_theta_grid) is bounded, monotone, and matches the Python fast_mass port")
{
  // Representative (lob, zob) slice: three radii sharing one theta grid,
  // as ShearPrjCore::set_sample builds per (lambda_bin, zob) slice.
  double const lobc = 37.5;
  double const zob = 0.42;
  double const chi_o = 1085.0;               // Mpc/h (h0-scaled, see CLAUDE.md)
  double const d_a_o = chi_o / (1.0 + zob);
  std::vector<double> const r_vec{0.4117, 1.0257, 4.0265};
  // Production always derives R_excl from the same richness-bin centre
  // used for theta_lam, so here R_excl and theta_lam share a numerator.
  double const r_excl = R_lambda(lobc) * (1.0 + zob);
  std::size_t const n_per_seg = 6;
  double const r_max_cmpch = 35.0;

  auto const tg = build_theta_grid(lobc, zob, r_vec, chi_o, d_a_o, r_excl,
                                   n_per_seg, r_max_cmpch, {});

  // Bounds, computed independently from the same closed-form breakpoint
  // recipe documented in sp_detail::build_theta_grid, so this check does
  // not depend on the golden numbers below.
  double const theta_lam = R_lambda(lobc) * (1.0 + zob) / chi_o;
  double const theta_excl_o = r_excl / chi_o;
  std::vector<double> theta_r{r_vec[0] / d_a_o, r_vec[1] / d_a_o,
                              r_vec[2] / d_a_o};
  double const theta_r_min =
    *std::min_element(theta_r.begin(), theta_r.end());
  double const theta_r_max =
    *std::max_element(theta_r.begin(), theta_r.end());
  double const theta_max =
    std::max(r_max_cmpch / d_a_o, 3.0 * theta_r_max);
  double const lower =
    std::max(1.0e-8, 0.1 * std::min({theta_excl_o, theta_r_min, theta_lam}));

  REQUIRE(tg.theta.size() == tg.weight.size());
  REQUIRE(tg.theta.size() == 36);  // 6 dedup'd log-segments x n_per_seg
  CHECK(tg.theta.front() > lower);
  CHECK(tg.theta.back() < theta_max);
  for (std::size_t i = 1; i != tg.theta.size(); ++i)
    CHECK(tg.theta[i] > tg.theta[i - 1]);

  // dtheta = theta d(ln theta) Jacobian folded into weight: summing the
  // weights over all segments is a fixed-GL quadrature of
  // int_lower^theta_max dtheta = theta_max - lower.  exp(u) is analytic,
  // so this converges far tighter than the requested 1e-3 default even
  // at n_per_seg = 6; checking it near that convergence limit pins the
  // Jacobian, not just the node placement.
  double const sum_w =
    std::accumulate(tg.weight.begin(), tg.weight.end(), 0.0);
  CHECK(sum_w == Approx(theta_max - lower).epsilon(1e-6));

  // Golden values from ../python/shear_prj_fast_mass.py::build_theta_grid
  // with these same inputs (see file header).
  CHECK(tg.theta.front() == Approx(5.8237767207010004e-05).epsilon(1e-3));
  CHECK(tg.theta[tg.theta.size() / 2] ==
        Approx(0.0013639379714246158).epsilon(1e-3));
  CHECK(tg.theta.back() == Approx(0.04258104827263785).epsilon(1e-3));
  CHECK(sum_w == Approx(0.0457525701381797).epsilon(1e-3));
}

TEST_CASE("sigma_prj LoS-slab exclusion angle (theta_excl_at_z) matches the Python fast_mass port")
{
  double const chi_o = 1085.0;
  double const lobc = 37.5;
  double const zob = 0.42;
  double const r_excl = R_lambda(lobc) * (1.0 + zob);

  // Far outside the exclusion ring (|chi_z - chi_o| >= ~2 R_excl): the
  // law-of-cosines argument clips to >= 1, so the exclusion angle is
  // exactly zero -- this holds for any chi_o >> R_excl, not just this
  // particular pair of numbers.
  CHECK(theta_excl_at_z(chi_o - 2.0 * r_excl, chi_o, r_excl) ==
        Approx(0.0).margin(1e-12));
  CHECK(theta_excl_at_z(chi_o + 2.0 * r_excl, chi_o, r_excl) ==
        Approx(0.0).margin(1e-12));

  // Inside the ring: golden values from
  // ../python/shear_prj_fast_mass.py::theta_excl_at_z.
  CHECK(theta_excl_at_z(chi_o - 0.5 * r_excl, chi_o, r_excl) ==
        Approx(0.0009317777252103878).epsilon(1e-3));
  CHECK(theta_excl_at_z(chi_o, chi_o, r_excl) ==
        Approx(0.0010756348895233233).epsilon(1e-3));
  CHECK(theta_excl_at_z(chi_o + 0.5 * r_excl, chi_o, r_excl) ==
        Approx(0.0009312767333775487).epsilon(1e-3));
}

TEST_CASE("ShearPrjCore's DSigma_mis convention (c=4, single kernel) matches the shared Python replica")
{
  // Exactly ShearPrjCore's ctor line (systematics/shear_prj/cpp/sigma_prj_t.hh):
  //   dsigma_mis_.emplace(4.0, 2.77533742639e+11, SINGLE);
  //   dsigma_mis_->set_rho_mult(omega_m);
  y3_cluster::NFW_DSIGMA_MIS dsigma_mis(4.0, 2.77533742639e+11,
                                       y3_cluster::SINGLE);
  double const omega_m = 0.3096;
  dsigma_mis.set_rho_mult(omega_m);

  struct Point {
    double R, r_mis, lnM, expected;
  };
  // Golden values from shared/lensing_profiles.py::
  // NfwDsigmaMisProduction(kernel="single")(R, r_mis, lnM, rho_mult=omega_m)
  // -- the c=4/rho_crit-convention replica of this same class, not the
  // legacy c=5 richness_selection reference (see file header).
  Point const pts[] = {
    {0.4117, 0.30, 32.0, 9.699110243638033},
    {1.0257, 0.75, 33.5, 11.003453477283488},
    {4.0265, 1.50, 34.5, 8.772326127385625},
  };
  for (auto const& p : pts) {
    CHECK(dsigma_mis(p.R, p.r_mis, p.lnM) == Approx(p.expected).epsilon(1e-3));
  }

  // rho_mult enters as a pure multiplicative amplitude (see operator()'s
  // "norm" term) -- an exact algebraic identity, independent of the
  // interpolation table, so it gets a much tighter tolerance than the
  // golden-value pins above.
  y3_cluster::NFW_DSIGMA_MIS dsigma_unit(4.0, 2.77533742639e+11,
                                        y3_cluster::SINGLE);
  double const raw = dsigma_unit(pts[0].R, pts[0].r_mis, pts[0].lnM);
  CHECK(dsigma_mis(pts[0].R, pts[0].r_mis, pts[0].lnM) ==
        Approx(omega_m * raw).epsilon(1e-9));
}

TEST_CASE("ShearPrjCore construction parses its own wall grid and (lambda_bin, zob) slicing")
{
  // The config contract ShearPrjFastMass.cc relies on: ShearPrjCore's
  // ctor (systematics/shear_prj/cpp/sigma_prj_t.hh) reads the zipped wall
  // (lambda_bin/zo_low/zo_high/radii), the zt/lnm integration bounds,
  // and the GL/theta-grid knobs, then groups the wall into
  // (lambda_bin, zob) slices.  This constructs the actual core class
  // ShearPrjFastMass.cc instantiates; it stops short of set_sample()
  // (see file header) since that needs a full pipeline sample.
  cosmosis::DataBlock cfg;
  char const* section = "ShearPrjFastMassTest";

  cfg.put_val(section, "lambda_bin", std::vector<int>{0, 0, 1});
  cfg.put_val(section, "zo_low", std::vector<double>{0.20, 0.20, 0.35});
  cfg.put_val(section, "zo_high", std::vector<double>{0.35, 0.35, 0.50});
  cfg.put_val(section, "radii", std::vector<double>{0.4117, 1.0257, 2.0});
  cfg.put_val(section, "zt_low", 0.10);
  cfg.put_val(section, "zt_high", 0.75);
  cfg.put_val(section, "lnm_low", 29.9336);
  cfg.put_val(section, "lnm_high", 35.6814);
  cfg.put_val(section, "n_lnm", 8);
  cfg.put_val(section, "n_per_seg", 6);
  cfg.put_val(section, "n_zring", 6);
  cfg.put_val(section, "n_zouter", 6);
  cfg.put_val(section, "R_max_cMpch", 35.0);

  auto const grid = ShearPrjCore::make_grid_points(cfg, section);
  REQUIRE(grid.size() == 3);
  CHECK(grid.points[0][0] == Approx(0.0));
  CHECK(grid.points[0][1] == Approx(0.20));
  CHECK(grid.points[0][2] == Approx(0.35));
  CHECK(grid.points[0][3] == Approx(0.4117));
  CHECK(grid.points[1][3] == Approx(1.0257));
  CHECK(grid.points[2][0] == Approx(1.0));
  CHECK(grid.points[2][1] == Approx(0.35));
  CHECK(grid.points[2][3] == Approx(2.0));

  // Must not throw: the ctor also reads the single-kernel NFW
  // off-center tables (data/nfw_off_center/*single*) once here, the
  // same construction cost ShearPrjFastMassCpp pays.
  ShearPrjCore core(cfg, section);
  (void)core;
}

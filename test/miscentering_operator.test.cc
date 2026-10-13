// Unit tests for src/pipelines/systematics/miscentering/Miscentering.hh.
//
// Truth: gamma-miscentred DeltaSigma of an analytic NFW computed in Hankel
// space (test/make_miscentering_nfw_truth_pins.py) — no Sigma -> DeltaSigma
// aperture integral, so it cannot share a low-R bias with the code under test.
//
//   1. NfwWrightBrainerd: DeltaSigma = barSigma - Sigma closes.
//   2. MiscenteringOperator, analytic-Sigma callback (D1)+(D2) vs the pins.
//   3. Weight-row form (D4) == callback form on the same log-r interpolant.
//   4. Table from r_core = 1e-3 + uniform-disk core slot vs the pins.
//   5. MiscenteringCuhre (full adaptive integral, spline input) vs the pins.
//   6. Contract: parameter validation, DataBlock keys, grid coverage, (E4).
#include "catch2/catch.hpp"

#include "cosmosis/datablock/datablock.hh"
#include "pipelines/systematics/miscentering/Miscentering.hh"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

using y3_cluster::LogRadialGrid;
using y3_cluster::MiscenteringCuhre;
using y3_cluster::MiscenteringOperator;
using y3_cluster::NfwWrightBrainerd;

namespace {

  // Fiducial of make_miscentering_nfw_truth_pins.py.
  double const LNM = std::log(2.0e14);
  double constexpr CONC = 4.0;
  double constexpr RHO_M = 0.3 * 2.77533742639e11;
  double const SIGMA = 0.17 * std::pow(52.5 / 100.0, 0.2);

  // {R [h^-1 Mpc], DeltaSigma_mis [M_sun h / pc^2]}
  std::vector<std::pair<double, double>> const PINS{
    {0.03, 1.00838276768405},  {0.1, 8.09866721243061},
    {0.2, 20.8234031961866},   {0.5848, 40.4102464151427},
    {1.0, 33.548752085129},    {2.44521, 13.4915155646354},
    {5.0, 5.04917392631643}};

  // Measured callback-form error at 32/32/32 nodes (rel): 3e-4 at R = 0.03,
  // 3.5e-6 at 0.1, <= 3.4e-7 for R >= 0.2 (the production wall).
  double
  callback_tolerance(double R)
  {
    return R < 0.05 ? 1.0e-3 : (R < 0.15 ? 1.0e-5 : 1.0e-6);
  }

  NfwWrightBrainerd const NFW(LNM, CONC, RHO_M);

  double
  rel(double a, double b)
  {
    return std::abs(a / b - 1.0);
  }

}  // namespace

TEST_CASE("NfwWrightBrainerd: DeltaSigma closes with the aperture of Sigma")
{
  // barSigma(<R) = int_0^1 4 t^3 Sigma(R t^2) dt, 200-node GL.
  std::vector<double> t, w;
  y3_pipelines::gl_nodes(0.0, 1.0, 200, t, w);
  for (double R : {0.01, 0.2, NFW.r_s(), 1.0, 10.0}) {
    double bar = 0.0;
    for (std::size_t j = 0; j != t.size(); ++j)
      bar += 4.0 * w[j] * t[j] * t[j] * t[j] * NFW.surface_density(R * t[j] * t[j]);
    CHECK(rel(bar - NFW.surface_density(R), NFW.excess_surface_density(R)) < 1.0e-9);
  }
  // Series branch at x = 1 joins the closed form.
  double const rs = NFW.r_s();
  CHECK(rel(NFW.surface_density(rs * (1.0 + 0.999e-3)),
            NFW.surface_density(rs * (1.0 + 1.001e-3))) < 1.0e-5);
}

TEST_CASE("MiscenteringOperator: analytic-Sigma callback vs Hankel truth")
{
  MiscenteringOperator const op(0.22, 0.17);
  auto sig = [](double r) { return NFW.surface_density(r); };
  for (auto const& [R, truth] : PINS) {
    INFO("R = " << R);
    CHECK(rel(op.excess_surface_density(R, SIGMA, sig), truth) <
          callback_tolerance(R));
  }
}

TEST_CASE("MiscenteringOperator: weight row == callback on the same interpolant")
{
  MiscenteringOperator const op(0.22, 0.17);
  LogRadialGrid const grid(1.0e-4, 2.0 * 5.0 + 20.0 * SIGMA, 512);
  std::vector<double> tab(grid.size());
  for (std::size_t m = 0; m != grid.size(); ++m)
    tab[m] = NFW.surface_density(grid.r(m));
  double const ln_lo = std::log(grid.r_lo());
  auto interp = [&](double r) {  // linear in ln r, clamped
    double const u = (std::log(r) - ln_lo) / grid.dln();
    if (!(u > 0.0)) return tab.front();
    if (u >= static_cast<double>(grid.size() - 1)) return tab.back();
    auto const i = static_cast<std::size_t>(u);
    double const f = u - static_cast<double>(i);
    return (1.0 - f) * tab[i] + f * tab[i + 1];
  };
  std::vector<double> row;
  for (auto const& [R, truth] : PINS) {
    op.excess_surface_density_row(R, SIGMA, grid, row);
    double dot = 0.0;
    for (std::size_t m = 0; m != row.size(); ++m) dot += row[m] * tab[m];
    INFO("R = " << R);
    CHECK(rel(dot, op.excess_surface_density(R, SIGMA, interp)) < 1.0e-11);
  }
}

TEST_CASE("MiscenteringOperator: table from 1e-3 + uniform-disk core vs Hankel truth")
{
  // The haloModel contract: Sigma tabulated from r_core = 1e-3 and the core
  // given only by its projected mass M_2d(<r_core) (haloModel/m2d_core).
  // Measured (Python twin): 1.4e-5 at N_r = 1024 on the wall, the same as
  // the exact NFW interior; the error falls ~4x per doubling of N_r.
  MiscenteringOperator const op(0.22, 0.17);
  double const r_core = 1.0e-3;
  double const sigma_core =
    NFW.surface_density(r_core) + NFW.excess_surface_density(r_core);  // barSigma(<r_core)
  for (std::size_t n_r : {512u, 1024u}) {
    LogRadialGrid const grid(r_core, 2.0 * 5.0 + 20.0 * SIGMA, n_r, /*core=*/true);
    REQUIRE(grid.row_size() == n_r + 1);
    std::vector<double> tab(grid.row_size());
    for (std::size_t m = 0; m != grid.size(); ++m)
      tab[m] = NFW.surface_density(grid.r(m));
    tab.back() = sigma_core;
    std::vector<double> row;
    double worst = 0.0;
    for (auto const& [R, truth] : PINS) {
      if (R < 0.15) continue;  // production wall starts at 0.2
      op.excess_surface_density_row(R, SIGMA, grid, row);
      double dot = 0.0;
      for (std::size_t m = 0; m != row.size(); ++m) dot += row[m] * tab[m];
      worst = std::max(worst, rel(dot, truth));
    }
    INFO("N_r = " << n_r << ", max rel = " << worst);
    CHECK(worst < (n_r == 512 ? 1.0e-4 : 4.0e-5));
  }
}

TEST_CASE("MiscenteringCuhre: callable profile == spline profile on the same data")
{
  // The callable constructor (used by Shear1h2hMax mis_method = cuhre) and
  // the cluster-toolkit spline constructor integrate the same way.
  auto sig = [](double r) { return NFW.surface_density(r); };
  MiscenteringCuhre const callable(sig, 1.0e-6);
  auto const res = callable.excess_surface_density(1.0, SIGMA);
  CHECK(res.converged);
  CHECK(rel(res.value, 33.548752085129) < 1.0e-6);
}

TEST_CASE("MiscenteringCuhre: full adaptive integral vs Hankel truth")
{
  // cluster-toolkit-style input: cubic spline in ln r on [1e-3, 100], the
  // analytic NFW core below, 0 above.
  std::vector<double> r_tab, s_tab;
  for (int i = 0; i != 600; ++i) {
    double const r = 1.0e-3 * std::pow(1.0e5, i / 599.0);
    r_tab.push_back(r);
    s_tab.push_back(NFW.surface_density(r));
  }
  MiscenteringCuhre const ref(
    r_tab, s_tab, [](double r) { return NFW.surface_density(r); }, 1.0e-6);
  for (auto const& [R, truth] : PINS) {
    if (R != 0.2 && R != 1.0 && R != 5.0) continue;
    auto const res = ref.excess_surface_density(R, SIGMA);
    INFO("R = " << R << ", value = " << res.value << ", err = " << res.error
                << ", neval = " << res.neval);
    CHECK(res.converged);
    CHECK(rel(res.value, truth) < 1.0e-5);
  }
}

TEST_CASE("MiscenteringOperator: contract")
{
  CHECK_THROWS_AS(MiscenteringOperator(-0.1, 0.17), std::invalid_argument);
  CHECK_THROWS_AS(MiscenteringOperator(1.1, 0.17), std::invalid_argument);
  CHECK_THROWS_AS(MiscenteringOperator(0.2, 0.0), std::invalid_argument);
  CHECK_THROWS_AS(MiscenteringOperator(0.2, 0.17, 0), std::invalid_argument);

  MiscenteringOperator const op(0.25, 0.17);
  CHECK(op.apply(8.0, 4.0) == Approx(7.0));
  CHECK(op.offset_scale(100.0) == Approx(0.17));
  CHECK(op.max_ring_radius(5.0, 0.1) == Approx(12.0));

  // Grid must reach 2 R + x_max sigma.
  LogRadialGrid const short_grid(1.0e-4, 5.0, 64);
  std::vector<double> row;
  CHECK_THROWS_AS(op.excess_surface_density_row(5.0, 0.15, short_grid, row),
                  std::out_of_range);

  cosmosis::DataBlock s;
  CHECK_THROWS(MiscenteringOperator::from_datablock(s));
  s.put_val("miscentering", "f_mis", 0.22);
  CHECK_THROWS(MiscenteringOperator::from_datablock(s));  // tau_mis missing
  s.put_val("miscentering", "tau_mis", 0.17);
  auto const from_db = MiscenteringOperator::from_datablock(s);
  CHECK(from_db.f_mis() == 0.22);
  CHECK(from_db.tau_mis() == 0.17);
}

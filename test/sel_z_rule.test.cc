// SelZRule / SelZBins / SelFunctionZ_t (models/sel_z_rule.hh): the
// composite-GL true-z rule of the fixed-GL selection evaluators.
//
//   1. Panel layout: steps at zob +- L sigma_z sit mid-panel, the seam is a
//      gap, the rule covers exactly the union of the bins' supports.
//   2. Accuracy against closed form. With f = 1 the z integral of the
//      photo-z factor has an exact antiderivative,
//        int Phi((e - z)/s) dz = -s G((e - z)/s),  G(x) = x Phi(x) + phi(x),
//      so int_{zt_lo}^{zt_hi} S_j(z) 1[z outside seam] dz is exact. The
//      composite rule must hit it for sigma_z = 0.001, 0.01, 0.03 and the
//      one-halo DES Buzzard binning; the old single 64-node panel must not
//      (it misses the sigma_z = 0.001 bins by percent).
//   3. The global fallback (z_rule = global, or a block without the
//      factorised keys) is bit-identical to the old gl_nodes rule.
//   4. SelFunctionZ_t = S_rich x S_j x seam, exactly zero in the seam.
#include "catch2/catch.hpp"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include "models/sel_z_rule.hh"
#include "pipelines/shared/sel_gl_weights.hh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using y3_cluster::SelZBins;
using y3_cluster::SelZRule;

namespace {

  constexpr double ZT_LO = 0.05, ZT_HI = 0.80;

  double
  Phi(double x)
  {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
  }
  double
  G(double x)
  {
    return x * Phi(x) + std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
  }
  // int_{a}^{c} Phi((e - z)/s) dz
  double
  int_phi(double e, double s, double a, double c)
  {
    return s * (G((e - a) / s) - G((e - c) / s));
  }
  // Exact int of S_j * 1[outside seam] over [ZT_LO, ZT_HI], constant sigma.
  double
  exact(SelZBins const& zb, int b)
  {
    auto seg = [&](double a, double c) {
      double const s = zb.sigma_z[b];
      return int_phi(zb.zob_max[b], s, a, c) - int_phi(zb.zob_min[b], s, a, c);
    };
    if (!zb.has_excl())
      return seg(ZT_LO, ZT_HI);
    return seg(ZT_LO, zb.excl_lo) + seg(zb.excl_hi, ZT_HI);
  }

  SelZBins
  buzzard_bins(double sigma)
  {
    SelZBins zb;
    zb.zob_min = {0.20, 0.37, 0.50};
    zb.zob_max = {0.33, 0.50, 0.65};
    zb.sigma_z.assign(3, sigma);
    zb.excl_lo = 0.33;
    zb.excl_hi = 0.37;
    return zb;
  }

  SelZBins
  contiguous_bins(double sigma)
  {
    SelZBins zb;
    zb.zob_min = {0.20, 0.35, 0.50};
    zb.zob_max = {0.35, 0.50, 0.65};
    zb.sigma_z.assign(3, sigma);
    return zb;
  }

  // Composite rule applied to bin b's S_j: n nodes per panel.
  double
  composite(SelZBins const& zb, int b, std::size_t n, double L = 5.0)
  {
    auto const& [u, den] = y3_cluster::sel_z_detail::gl_unit(n);
    double acc = 0.0;
    for (auto const& [a, c] : zb.panels(ZT_LO, ZT_HI, L))
      for (std::size_t i = 0; i != u.size(); ++i) {
        double const z = 0.5 * (a + c) + 0.5 * (c - a) * u[i];
        acc += 2.0 * (0.5 * (c - a)) / den[i] * zb(b, z);
      }
    return acc;
  }

  double
  global(SelZBins const& zb, int b, std::size_t n)
  {
    std::vector<double> xs, ws;
    y3_pipelines::gl_nodes(ZT_LO, ZT_HI, n, xs, ws);
    double acc = 0.0;
    for (std::size_t i = 0; i != n; ++i)
      acc += ws[i] * zb(b, xs[i]);
    return acc;
  }

  cosmosis::DataBlock
  config(std::string const& rule = "")
  {
    cosmosis::DataBlock cfg;
    cfg.put_val("T", "zt_low", ZT_LO);
    cfg.put_val("T", "zt_high", ZT_HI);
    if (!rule.empty())
      cfg.put_val("T", "z_rule", rule);
    return cfg;
  }

  // sel_function outputs for two bins on a 2 x 2 (z, lnM) table; the
  // factorised keys only when `factored`.
  cosmosis::DataBlock
  sample(bool factored)
  {
    cosmosis::DataBlock s;
    s.put_val("sel_function", "lnM", std::vector<double>{30.0, 36.0});
    s.put_val("sel_function", "z", std::vector<double>{ZT_LO, ZT_HI});
    s.put_val("sel_function", "S_stack",
              cosmosis::ndarray<double>(std::vector<double>(8, 0.5),
                                        {2, 2, 2}));
    s.put_val("sel_function", "zob_min", std::vector<double>{0.20, 0.37});
    s.put_val("sel_function", "zob_max", std::vector<double>{0.33, 0.50});
    if (factored) {
      // S_rich linear in z: 1 at ZT_LO, 2 at ZT_HI (both bins, all lnM).
      std::vector<double> rich{1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0};
      s.put_val("sel_function", "S_rich_stack",
                cosmosis::ndarray<double>(rich, {2, 2, 2}));
      s.put_val("sel_function", "sigma_z", std::vector<double>{1e-3, 1e-3});
      s.put_val("sel_function", "sigma_z_one_plus_z", 0);
      s.put_val("sel_function", "zt_excl_low", 0.33);
      s.put_val("sel_function", "zt_excl_high", 0.37);
    }
    return s;
  }

} // namespace

TEST_CASE("SelZBins panels: steps mid-panel, seam a gap, union support")
{
  auto const zb = buzzard_bins(1e-3);
  auto const p = zb.panels(ZT_LO, ZT_HI, 5.0);
  REQUIRE(!p.empty());
  CHECK(p.front().first == Approx(0.195).margin(1e-12));
  CHECK(p.back().second == Approx(0.655).margin(1e-12));
  for (std::size_t i = 0; i != p.size(); ++i) {
    CHECK(p[i].second > p[i].first);
    // nothing inside the seam
    CHECK(!(p[i].first >= 0.33 - 1e-12 && p[i].second <= 0.37 + 1e-12));
    if (i + 1 < p.size()) {
      bool const gap = p[i].second == Approx(0.33) &&
                       p[i + 1].first == Approx(0.37);
      CHECK((gap || p[i].second == p[i + 1].first));
    }
  }
  // every step's +-5 sigma window is a panel edge pair
  std::vector<double> edges;
  for (auto const& [a, c] : p) {
    edges.push_back(a);
    edges.push_back(c);
  }
  for (double e : {0.195, 0.205, 0.325, 0.375, 0.495, 0.505, 0.645, 0.655,
                   0.504})
    CHECK(std::any_of(edges.begin(), edges.end(),
                      [e](double x) { return std::abs(x - e) < 1e-12; }));
  // 0.7 (OMEGA_Z_DES) lies outside the support: not a break.
  CHECK(std::none_of(edges.begin(), edges.end(),
                     [](double x) { return std::abs(x - 0.7) < 1e-12; }));
}

TEST_CASE("Composite z rule integrates S_j x seam to closed form")
{
  for (double sigma : {1e-3, 1e-2, 3e-2})
    for (auto const& zb : {buzzard_bins(sigma), contiguous_bins(sigma)})
      for (int b = 0; b != 3; ++b) {
        double const ref = exact(zb, b);
        INFO("sigma_z " << sigma << " bin " << b << " seam "
                        << zb.has_excl());
        CHECK(composite(zb, b, 8) / ref - 1.0 == Approx(0.0).margin(2e-5));
        // 16/panel: down to the support truncation at +-5 sigma_z
        // (~ sigma_z phi(5) / 25 / bin width, 4e-9 at sigma_z = 0.01).
        CHECK(composite(zb, b, 16) / ref - 1.0 == Approx(0.0).margin(1e-8));
      }
}

TEST_CASE("The old single 64-node panel misses the sigma_z = 0.001 steps")
{
  // Documents the defect the composite rule fixes (percent level).
  auto const zb = buzzard_bins(1e-3);
  double worst = 0.0;
  for (int b = 0; b != 3; ++b)
    worst = std::max(worst, std::abs(global(zb, b, 64) / exact(zb, b) - 1.0));
  CHECK(worst > 1e-2);
}

TEST_CASE("SelZRule: global fallback is the old rule bit for bit")
{
  std::vector<double> xs, ws;
  y3_pipelines::gl_nodes(ZT_LO, ZT_HI, 64, xs, ws);

  auto cfg = config();
  SelZRule rule(cfg, "T");
  auto s_old = sample(/*factored=*/false);
  rule.update(s_old);
  CHECK(!rule.factored());
  REQUIRE(rule.size() == 64u);
  for (std::size_t i = 0; i != 64; ++i) {
    CHECK(rule.xs()[i] == xs[i]);
    CHECK(rule.ws()[i] == ws[i]);
  }

  auto cfg_g = config("global");
  SelZRule forced(cfg_g, "T");
  auto s_new = sample(/*factored=*/true);
  forced.update(s_new);
  CHECK(!forced.factored());
  CHECK(forced.xs() == xs);
  CHECK(forced.ws() == ws);

  auto cfg_bad = config("adaptive");
  CHECK_THROWS_AS(SelZRule(cfg_bad, "T"), std::invalid_argument);
}

TEST_CASE("SelZRule: composite nodes and the factorised S_ij")
{
  auto cfg = config();
  cfg.put_val("T", "n_z_panel", 6);
  cfg.put_val("T", "sel_interp", std::string("bilinear"));  // 2 x 2 table
  SelZRule rule(cfg, "T");
  auto s = sample(/*factored=*/true);
  CHECK(rule.update(s));
  CHECK(rule.factored());
  CHECK(rule.size() == 6u * rule.panels().size());
  CHECK(!rule.update(s));   // same layout: no rebuild

  double wsum = 0.0;
  for (double w : rule.ws())
    wsum += w;
  double len = 0.0;
  for (auto const& [a, c] : rule.panels())
    len += c - a;
  CHECK(wsum == Approx(len).epsilon(1e-13));
  for (double z : rule.xs())
    CHECK(!(z > 0.33 && z < 0.37));

  auto const sel = rule.sel(s, 0);
  for (double z : {0.25, 0.3301, 0.35, 0.369})
    for (double lnM : {31.0, 35.0}) {
      double const rich = 1.0 + (z - ZT_LO) / (ZT_HI - ZT_LO);
      double const sj = (z > 0.33 && z < 0.37)
                          ? 0.0
                          : Phi((0.33 - z) / 1e-3) - Phi((0.20 - z) / 1e-3);
      CHECK(sel(lnM, z) == Approx(rich * sj).epsilon(1e-12).margin(1e-300));
    }
  CHECK(sel(36.5, 0.3) == 0.0);   // outside the lnM table
}

// Unit tests for the REAL traditional 1h+2h max-model header
// (src/pipelines/des_y3/shear_1h2h/cpp/0d/shear1h2h_max_t.hh, module
// Shear1h2hMax.cc).
//
// Why a second file next to test/shear1h2h_max.test.cc: that test was
// written when Shear1h2hMax lived entirely inside the module .cc and was
// private to that translation unit, so it pins a byte-for-byte MIRROR of
// the (lnM, z) accumulation rather than the class.  The class is now
// public in the header; this file drives it directly.
//
// Pinned here, on a self-contained synthetic DataBlock:
//
//   1. Shear1h2hMax::evaluate() vs an independently assembled
//      z-RESOLVED reference of the piecewise max model
//        R <  r_trans : DeltaSigma_1h + Delta_gap^1 (R / r_trans)^2
//        R >= r_trans : b DeltaSigma_hh + C_0 / R^2.
//   2. Both sides of r_trans are genuinely exercised: the inner side is
//      independent of the 2-halo table, the outer side linear in it.
//   3. An all-NaN dSigma_hh is sanitized to zero (the module's own
//      make_sanitized_table path, reached through set_sample).
//   4. Bsel (Sunayama23): non-local correction, and the flat-B limit.
//   5. The DataBlock contract: every required sample key, and every
//      required [Shear1h2hMax] ini option, fails LOUDLY when absent --
//      in particular r_trans / c0_trans / delta_gap_1 (no max(1h, 2h)
//      fallback) and, with bsel on, Sigma_bar_nfw / Sigma_bar_hh /
//      delta_gap_0.
//   6. Guard clauses: out-of-range bin_index, empty lob_centers.
#include "catch2/catch.hpp"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include "models/dv_do_dz_t.hh"
#include "models/hmf_t.hh"
#include "models/omega_z_des.hh"
#include "models/sel_function_t.hh"
#include "pipelines/shared/lensing_helpers.hh"
#include "pipelines/shared/sel_gl_weights.hh"
#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"
#include "utils/make_interp_2d.hh"

#include "pipelines/des_y3/shear_1h2h/cpp/0d/shear1h2h_max_t.hh"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  double constexpr ZT_LO = 0.10;
  double constexpr ZT_HI = 0.50;
  double constexpr LNM_LO = 30.0;
  double constexpr LNM_HI = 36.0;
  int constexpr N_Z = 6;
  int constexpr N_LNM = 8;
  double constexpr RHO_M_REF = 0.3096 * 2.77533742639e+11;
  // Piecewise terms, constant in (lnM, z) so the composition is
  // hand-checkable.  R_TRANS splits R_QUERY: 0.5, 1.0 inside; 2.0 outside.
  double constexpr R_TRANS = 1.5;
  double constexpr C0_TRANS = 2.0e12;  // C_0 / R^2 * 1e-12 = 2 / R^2
  double constexpr GAP_1 = 0.4;
  double constexpr GAP_0 = 0.1;
  std::vector<double> const LOB_CENTERS{30.0, 90.0};
  std::vector<double> const R_QUERY{0.5, 1.0, 2.0};
  // GL tolerance: default (64/32) vs doubled nodes, measured on the real
  // des_y3_cpp0d_fast_apriori.ini block at <= 6e-5 (bicubic xi_nl).
  double constexpr PRECISION_GL = 1.0e-4;

  cosmosis::DataBlock
  make_cfg_block(std::vector<double> const& lob_centers = LOB_CENTERS)
  {
    cosmosis::DataBlock cfg;
    char const* label = Shear1h2hMax::module_label();
    cfg.put_val(label, "n_lnm", N_LNM);
    cfg.put_val(label, "n_z", N_Z);
    cfg.put_val(label, "zt_low", ZT_LO);
    cfg.put_val(label, "zt_high", ZT_HI);
    cfg.put_val(label, "lnm_low", LNM_LO);
    cfg.put_val(label, "lnm_high", LNM_HI);
    cfg.put_val(label, "lob_centers", lob_centers);
    return cfg;
  }

  cosmosis::DataBlock
  make_bsel_cfg_block()
  {
    auto cfg = make_cfg_block();
    cfg.put_val(Shear1h2hMax::module_label(), "bsel", 1);
    cfg.put_val(Shear1h2hMax::module_label(), "bsel_model",
                std::string("Sunayama23"));
    return cfg;
  }

  struct SampleOpts {
    // `hh_amp` scales the whole two-halo table; `hh_nan` replaces it with
    // an all-NaN table (the historical defect shape).
    double hh_amp = 1.0;
    bool hh_nan = false;
    // Sigma_bar_nfw / Sigma_bar_hh / delta_gap_0 (bsel).
    bool sigma_bar = false;
    // xi_nl section + concentration/rho_m_ref (profile_method = gauss_legendre).
    bool xi = false;
  };

  cosmosis::DataBlock
  make_gl_cfg_block(bool with_wall)
  {
    auto cfg = make_cfg_block();
    char const* label = Shear1h2hMax::module_label();
    cfg.put_val(label, "profile_method", std::string("gauss_legendre"));
    if (with_wall)
      cfg.put_val(label, "r_perp", R_QUERY);
    return cfg;
  }

  struct DbKey { char const* section; char const* key; };

  cosmosis::ndarray<double>
  const_table(double v)
  {
    return cosmosis::ndarray<double>(std::vector<double>(4, v), {2, 2});
  }

  double
  dsig_1h(double r)
  {
    return 5.0 - 0.3 * r;
  }

  // Two knots per axis, all tables either constant or linear along the one
  // axis the formula varies, so every interpolator is exact in range --
  // the same construction as shear1h_gl.test.cc.  `skip` omits exactly
  // one key, for the DataBlock contract test.
  cosmosis::DataBlock
  make_sample_block(SampleOpts const& o = {},
                    DbKey const& skip = DbKey{"none", "none"})
  {
    cosmosis::DataBlock s;
    auto keep = [&](char const* sec, char const* key) {
      return !(std::string(skip.section) == sec &&
               std::string(skip.key) == key);
    };

    if (keep("cosmological_parameters", "omega_m"))
      s.put_val("cosmological_parameters", "omega_m", 0.3);
    s.put_val("cosmological_parameters", "omega_lambda", 0.7);
    s.put_val("cosmological_parameters", "omega_k", 0.0);
    if (keep("cosmological_parameters", "omega_nu"))
      s.put_val("cosmological_parameters", "omega_nu", 0.0);
    if (keep("cosmological_parameters", "h0"))
      s.put_val("cosmological_parameters", "h0", 0.7);

    if (keep("cluster_abundance", "hmf_s"))
      s.put_val("cluster_abundance", "hmf_s", 0.0);
    if (keep("cluster_abundance", "hmf_q"))
      s.put_val("cluster_abundance", "hmf_q", 1.0);
    if (keep("mass_function", "m_h"))
      s.put_val("mass_function", "m_h", std::vector<double>{1.0e13, 1.0e15});
    if (keep("mass_function", "z"))
      s.put_val("mass_function", "z", std::vector<double>{0.0, 1.0});
    if (keep("mass_function", "dndlnmh"))
      s.put_val("mass_function", "dndlnmh", const_table(7.0));

    if (keep("distances", "z"))
      s.put_val("distances", "z", std::vector<double>{0.0, 1.0});
    if (keep("distances", "d_a"))
      s.put_val("distances", "d_a", std::vector<double>{1500.0, 1500.0});

    if (keep("average_sigma_crit_inv", "zlense"))
      s.put_val("average_sigma_crit_inv", "zlense",
                std::vector<double>{0.0, 1.0});
    if (keep("average_sigma_crit_inv", "sci_average"))
      s.put_val("average_sigma_crit_inv", "sci_average",
                std::vector<double>{3.0e-4, 3.0e-4});

    if (keep("sel_function", "lnM"))
      s.put_val("sel_function", "lnM", std::vector<double>{LNM_LO, LNM_HI});
    if (keep("sel_function", "z"))
      s.put_val("sel_function", "z", std::vector<double>{ZT_LO, ZT_HI});
    if (keep("sel_function", "S_stack")) {
      std::vector<double> s_stack;
      for (double c : {2.0, 3.5})
        for (int i = 0; i != 4; ++i) s_stack.push_back(c);
      s.put_val("sel_function", "S_stack",
                cosmosis::ndarray<double>(s_stack, {2, 2, 2}));
    }

    // haloModel: r_sigma x lnM for dSigma_nfw, lnM x z for bias and the
    // piecewise terms, r_sigma x z for dSigma_hh.
    if (keep("haloModel", "r_sigma"))
      s.put_val("haloModel", "r_sigma", std::vector<double>{0.05, 10.0});
    if (keep("haloModel", "lnM"))
      s.put_val("haloModel", "lnM", std::vector<double>{28.0, 37.0});
    if (keep("haloModel", "z"))
      s.put_val("haloModel", "z", std::vector<double>{0.0, 1.0});
    if (keep("haloModel", "dSigma_nfw"))
      s.put_val("haloModel", "dSigma_nfw",
                cosmosis::ndarray<double>(
                  std::vector<double>{dsig_1h(0.05), dsig_1h(10.0),
                                      dsig_1h(0.05), dsig_1h(10.0)},
                  {2, 2}));
    if (keep("haloModel", "bias"))
      s.put_val("haloModel", "bias", const_table(2.0));
    if (keep("haloModel", "dSigma_hh")) {
      double const nan_d = std::numeric_limits<double>::quiet_NaN();
      s.put_val("haloModel", "dSigma_hh",
                const_table(o.hh_nan ? nan_d : o.hh_amp));
    }
    if (keep("haloModel", "r_trans"))
      s.put_val("haloModel", "r_trans", const_table(R_TRANS));
    if (keep("haloModel", "c0_trans"))
      s.put_val("haloModel", "c0_trans", const_table(C0_TRANS));
    if (keep("haloModel", "delta_gap_1"))
      s.put_val("haloModel", "delta_gap_1", const_table(GAP_1));

    if (o.xi) {
      // gauss_legendre inputs: concentration + rho_m_ref for xi_nfw, and
      // xi_nl(r, z) as an (n_z, n_r) table on the halo_model grid
      // r = logspace(-3, 3, 128); 4 z rows (bicubic needs >= 4 per axis).
      std::vector<double> r_xi, xi_vals;
      std::vector<double> const z_xi{0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0};
      for (int i = 0; i != 128; ++i)
        r_xi.push_back(std::pow(10.0, -3.0 + 6.0 * i / 127.0));
      for (double zz : z_xi)
        for (double r : r_xi)
          xi_vals.push_back(o.hh_amp * std::pow(r / 5.0, -1.8) / (1.0 + zz));
      s.put_val("xi_nl", "r", r_xi);
      s.put_val("xi_nl", "z", z_xi);
      s.put_val("xi_nl", "xi_nl",
                cosmosis::ndarray<double>(xi_vals, {4, 128}));
      s.put_val("haloModel", "rho_m_ref", RHO_M_REF);
      s.put_val("haloModel", "concentration", std::vector<double>{4.0, 4.0});
    }

    if (o.sigma_bar) {
      if (keep("haloModel", "Sigma_bar_nfw"))
        s.put_val("haloModel", "Sigma_bar_nfw", const_table(8.0));
      if (keep("haloModel", "Sigma_bar_hh"))
        s.put_val("haloModel", "Sigma_bar_hh", const_table(2.0));
      if (keep("haloModel", "delta_gap_0"))
        s.put_val("haloModel", "delta_gap_0", const_table(GAP_0));
      s.put_val("haloModel", "rho_m_ref", RHO_M_REF);
      s.put_val("haloModel", "concentration",
                std::vector<double>{4.0, 4.0});
    }
    return s;
  }

  void
  put_sunayama23(cosmosis::DataBlock& s, double pi0, double c)
  {
    s.put_val("bsel_profile_sunayama23", "pi0", std::vector<double>{pi0, pi0});
    s.put_val("bsel_profile_sunayama23", "r0", std::vector<double>{3.5, 3.5});
    s.put_val("bsel_profile_sunayama23", "c", c);
    s.put_val("bsel_profile_sunayama23", "lambda_bin",
              std::vector<double>{0.0, 1.0});
  }

  // Independent z-RESOLVED reference: recombines the same leaf models
  // Shear1h2hMax composes, but written as an explicit (lnM, z) double sum
  // rather than by reusing any of the module's own caches.
  struct ReferenceMax {
    y3_cluster::HMF_t hmf;
    y3_cluster::DV_DO_DZ_t dv;
    y3_cluster::OMEGA_Z_DES omega;
    y3_cluster::Interp1D sci;
    std::vector<double> lnm_x, lnm_w, z_x, z_w;
    double hh_amp;

    ReferenceMax(cosmosis::DataBlock& s, double hh)
      : hmf(s)
      , dv(s)
      , omega(s)
      , sci(y3_pipelines::load_sigma_crit_inv(s))
      , hh_amp(hh)
    {
      y3_pipelines::gl_nodes(LNM_LO, LNM_HI, N_LNM, lnm_x, lnm_w);
      y3_pipelines::gl_nodes(ZT_LO, ZT_HI, N_Z, z_x, z_w);
    }

    static double
    piecewise(double R, double hh)
    {
      if (R >= R_TRANS)
        return 2.0 * hh + C0_TRANS / (R * R) * 1.0e-12;
      return dsig_1h(R) + GAP_1 * (R / R_TRANS) * (R / R_TRANS);
    }

    double
    acc(cosmosis::DataBlock& s, int bin, double R) const
    {
      y3_cluster::SelFunction_t const sel(s, bin);
      double total = 0.0;
      for (std::size_t k = 0; k != lnm_x.size(); ++k)
        for (std::size_t q = 0; q != z_x.size(); ++q) {
          double const lnM = lnm_x[k];
          double const z = z_x[q];
          double const w = lnm_w[k] * z_w[q] * dv(z) * omega(z) *
                           sci.clamp(z) * hmf(lnM, z) * sel(lnM, z);
          total += w * piecewise(R, hh_amp);
        }
      return total;
    }
  };

} // namespace

TEST_CASE("Shear1h2hMax matches an independently assembled piecewise reference")
{
  auto cfg = make_cfg_block();
  auto s = make_sample_block(SampleOpts{/*hh_amp=*/1.5});

  Shear1h2hMax mod(cfg);
  mod.set_sample(s);
  ReferenceMax const ref(s, 1.5);

  for (int b : {0, 1})
    for (double R : R_QUERY) {
      auto const out = mod.evaluate({static_cast<double>(b), R});
      INFO("bin " << b << " R " << R);
      CHECK(out[0] == Approx(ref.acc(s, b, R)).epsilon(1e-9));
      CHECK(std::isfinite(out[0]));
      CHECK(out[0] > 0.0);
    }
}

TEST_CASE("Shear1h2hMax exercises both sides of r_trans")
{
  // Inside r_trans the 2-halo table must not matter; outside, the stack
  // is affine in it with the C_0 / R^2 term as the offset.
  auto cfg = make_cfg_block();

  Shear1h2hMax zero(cfg), one(cfg), two(cfg);
  auto s_zero = make_sample_block(SampleOpts{/*hh_amp=*/0.0});
  auto s_one = make_sample_block(SampleOpts{/*hh_amp=*/1.0});
  auto s_two = make_sample_block(SampleOpts{/*hh_amp=*/2.0});
  zero.set_sample(s_zero);
  one.set_sample(s_one);
  two.set_sample(s_two);

  for (int b : {0, 1})
    for (double R : R_QUERY) {
      auto const pt = Shear1h2hMax::grid_point_t{static_cast<double>(b), R};
      double const v0 = zero.evaluate(pt)[0];
      double const v1 = one.evaluate(pt)[0];
      double const v2 = two.evaluate(pt)[0];
      INFO("bin " << b << " R " << R);
      if (R < R_TRANS) {
        CHECK(v1 == Approx(v0).epsilon(1e-12));
      } else {
        CHECK(v1 > v0);
        CHECK(v2 - v1 == Approx(v1 - v0).epsilon(1e-9));
      }
    }
}

TEST_CASE("Shear1h2hMax sanitizes an all-NaN dSigma_hh to zero")
{
  auto cfg = make_cfg_block();

  Shear1h2hMax nan_mod(cfg);
  auto s_nan = make_sample_block(SampleOpts{1.0, /*hh_nan=*/true});
  nan_mod.set_sample(s_nan);

  Shear1h2hMax zero_mod(cfg);
  auto s_zero = make_sample_block(SampleOpts{/*hh_amp=*/0.0});
  zero_mod.set_sample(s_zero);

  for (int b : {0, 1})
    for (double R : R_QUERY) {
      auto const pt = Shear1h2hMax::grid_point_t{static_cast<double>(b), R};
      double const v = nan_mod.evaluate(pt)[0];
      CHECK(std::isfinite(v));
      CHECK(v == Approx(zero_mod.evaluate(pt)[0]).epsilon(1e-12));
    }
}

TEST_CASE("Shear1h2hMax applies the Sunayama23 Bsel operator with non-local correction")
{
  auto cfg_selected = make_bsel_cfg_block();
  auto cfg_baseline = make_cfg_block();
  auto sample = make_sample_block(SampleOpts{1.5, false, /*sigma_bar=*/true});
  put_sunayama23(sample, 0.45, -0.1);

  Shear1h2hMax selected(cfg_selected);
  Shear1h2hMax baseline(cfg_baseline);
  selected.set_sample(sample);
  baseline.set_sample(sample);

  y3_cluster::BselSunayama23 const bsel(
    std::vector<double>{0.45, 0.45}, std::vector<double>{3.5, 3.5}, -0.1,
    std::vector<double>{0.0, 1.0});
  for (int bin : {0, 1})
    for (double R : R_QUERY) {
      auto const pt = Shear1h2hMax::grid_point_t{
        static_cast<double>(bin), R};
      double const sel_val = selected.evaluate(pt)[0];
      double const base_val = baseline.evaluate(pt)[0];
      // DeltaSigma_sel = B DeltaSigma - M_sel / (pi R^2); dB/dR > 0 and
      // barSigma > 0 make M_sel > 0, so sel_val < B(R) * base_val.
      INFO("bin " << bin << " R " << R);
      CHECK(std::isfinite(sel_val));
      CHECK(sel_val > 0.0);
      CHECK(sel_val < base_val * bsel(R, bin));
    }

  // Flat limit (B = 1, dB/dR = 0): DeltaSigma_sel = DeltaSigma.
  auto sample_flat = make_sample_block(SampleOpts{1.5, false, true});
  put_sunayama23(sample_flat, 0.0, 0.0);
  selected.set_sample(sample_flat);
  baseline.set_sample(sample_flat);
  for (int bin : {0, 1})
    for (double R : R_QUERY) {
      auto const pt = Shear1h2hMax::grid_point_t{
        static_cast<double>(bin), R};
      CHECK(selected.evaluate(pt)[0] ==
            Approx(baseline.evaluate(pt)[0]).epsilon(1e-12));
    }
}

TEST_CASE("Shear1h2hMax piecewise Bsel: tabulated r_perp wall == per-node apply()")
{
  // With r_perp set, set_sample tabulates B, dB/dR and barSigma at the
  // apply() nodes of every (bin, wall R); without it, evaluate() calls
  // BselDeltaSigmaOperator::apply per (lnM, z) node.  Same nodes, same
  // operation order: bit-identical.  R = 5 > r0 = 3.5 exercises the
  // break-radius panel split.
  std::vector<double> const wall{0.5, 1.0, 2.0, 5.0};
  auto cfg_wall = make_bsel_cfg_block();
  cfg_wall.put_val(Shear1h2hMax::module_label(), "r_perp", wall);
  auto cfg_nodes = make_bsel_cfg_block();
  auto sample = make_sample_block(SampleOpts{1.5, false, /*sigma_bar=*/true});
  put_sunayama23(sample, 0.45, -0.1);

  Shear1h2hMax tabulated(cfg_wall);
  Shear1h2hMax per_node(cfg_nodes);
  tabulated.set_sample(sample);
  per_node.set_sample(sample);
  for (int bin : {0, 1})
    for (double R : {0.5, 1.0, 2.0, 5.0, 3.0 /* off-wall: fallback */}) {
      auto const pt = Shear1h2hMax::grid_point_t{static_cast<double>(bin), R};
      INFO("bin " << bin << " R " << R);
      CHECK(tabulated.evaluate(pt)[0] == per_node.evaluate(pt)[0]);
    }
}

TEST_CASE("Shear1h2hMax ini-option contract: [Shear1h2hMax] required vs optional")
{
  // Cross-checked against cosmosis-models/real_pipeline_extract_max2h.ini:
  //
  //   [Shear1h2hMax]
  //   file      = .../des_y3_shear1h_0d_cpp/Shear1h2hMax.so
  //   bin_index = 0 ... 11    REQUIRED (wall, slow axis)
  //   r_perp    = 0.2 ... 5.0 REQUIRED (wall, fast axis)
  //   zt_low / zt_high / lnm_low / lnm_high   REQUIRED
  //   n_lnm = 96 / n_z = 64                   optional (those defaults)
  //   lob_centers = 25 37.5 52.5 130          optional (DES-Y3 default)
  //   bsel = F / bsel_model                   optional; bsel_model
  //                                           required when bsel = T
  char const* label = Shear1h2hMax::module_label();
  for (char const* missing : {"zt_low", "zt_high", "lnm_low", "lnm_high"}) {
    cosmosis::DataBlock cfg;
    if (std::string(missing) != "zt_low") cfg.put_val(label, "zt_low", ZT_LO);
    if (std::string(missing) != "zt_high")
      cfg.put_val(label, "zt_high", ZT_HI);
    if (std::string(missing) != "lnm_low")
      cfg.put_val(label, "lnm_low", LNM_LO);
    if (std::string(missing) != "lnm_high")
      cfg.put_val(label, "lnm_high", LNM_HI);
    INFO("missing " << missing);
    CHECK_THROWS(Shear1h2hMax{cfg});
  }

  cosmosis::DataBlock cfg_default;
  cfg_default.put_val(label, "zt_low", ZT_LO);
  cfg_default.put_val(label, "zt_high", ZT_HI);
  cfg_default.put_val(label, "lnm_low", LNM_LO);
  cfg_default.put_val(label, "lnm_high", LNM_HI);
  cfg_default.put_val(label, "n_lnm", N_LNM);
  cfg_default.put_val(label, "n_z", N_Z);
  REQUIRE_NOTHROW(Shear1h2hMax{cfg_default});
  CHECK(y3_pipelines::read_lob_centers(cfg_default, label) ==
        y3_pipelines::default_lob_centers());

  // bsel = T without bsel_model, or with an unknown tag, is an error.
  cosmosis::DataBlock cfg_bsel_no_model = cfg_default;
  cfg_bsel_no_model.put_val(label, "bsel", 1);
  CHECK_THROWS(Shear1h2hMax{cfg_bsel_no_model});
  cosmosis::DataBlock cfg_bsel_bad = cfg_bsel_no_model;
  cfg_bsel_bad.put_val(label, "bsel_model", std::string("bad"));
  CHECK_THROWS_AS(Shear1h2hMax{cfg_bsel_bad}, std::invalid_argument);

  // An explicitly EMPTY lob_centers is a configuration error, not a
  // silent fall-back to the default.
  cosmosis::DataBlock cfg_empty = cfg_default;
  cfg_empty.put_val(label, "lob_centers", std::vector<double>{});
  CHECK_THROWS_AS(Shear1h2hMax{cfg_empty}, std::runtime_error);

  // The wall itself: bin_index x r_perp, bin slow / R fast.
  CHECK_THROWS(Shear1h2hMax::make_grid_points(cfg_default));
  cosmosis::DataBlock cfg_wall = cfg_default;
  cfg_wall.put_val(label, "bin_index", std::vector<double>{0.0, 1.0});
  cfg_wall.put_val(label, "r_perp", std::vector<double>{0.5, 1.0, 2.0});
  auto const grid = Shear1h2hMax::make_grid_points(cfg_wall);
  REQUIRE(grid.size() == 6);
  CHECK(grid.points[0][0] == Approx(0.0));
  CHECK(grid.points[0][1] == Approx(0.5));
  CHECK(grid.points[1][1] == Approx(1.0)); // R is the fast axis
  CHECK(grid.points[3][0] == Approx(1.0)); // bin is the slow axis

  CHECK(std::string(Shear1h2hMax::output_sections()[0]) == "shear1h2h_max");
}

TEST_CASE("Shear1h2hMax DataBlock contract: every required sample key fails loudly")
{
  // average_sigma_crit_inv is required (the shear weight folds
  // Sigma_crit^-1 in), and so are the piecewise terms: there is no
  // max(1h, 2h) fallback when r_trans / c0_trans / delta_gap_1 are absent.
  std::vector<DbKey> const required{
    {"cosmological_parameters", "omega_m"},
    {"cosmological_parameters", "omega_nu"},
    {"cosmological_parameters", "h0"},
    {"cluster_abundance", "hmf_s"},
    {"cluster_abundance", "hmf_q"},
    {"mass_function", "m_h"},
    {"mass_function", "z"},
    {"mass_function", "dndlnmh"},
    {"distances", "z"},
    {"distances", "d_a"},
    {"average_sigma_crit_inv", "zlense"},
    {"average_sigma_crit_inv", "sci_average"},
    {"sel_function", "lnM"},
    {"sel_function", "z"},
    {"sel_function", "S_stack"},
    {"haloModel", "r_sigma"},
    {"haloModel", "lnM"},
    {"haloModel", "z"},
    {"haloModel", "dSigma_nfw"},
    {"haloModel", "bias"},
    {"haloModel", "dSigma_hh"},
    {"haloModel", "r_trans"},
    {"haloModel", "c0_trans"},
    {"haloModel", "delta_gap_1"},
  };
  auto cfg = make_cfg_block();
  for (auto const& k : required) {
    auto s = make_sample_block(SampleOpts{}, k);
    Shear1h2hMax mod(cfg);
    INFO("omitted " << k.section << "/" << k.key);
    CHECK_THROWS(mod.set_sample(s));
  }

  auto s_full = make_sample_block();
  Shear1h2hMax mod(cfg);
  REQUIRE_NOTHROW(mod.set_sample(s_full));
  CHECK(mod.evaluate({0.0, 1.0})[0] > 0.0);

  // With bsel on, barSigma(<r) needs the Sigma_bar tables and delta_gap_0.
  auto cfg_bsel = make_bsel_cfg_block();
  for (char const* key : {"Sigma_bar_nfw", "Sigma_bar_hh", "delta_gap_0"}) {
    auto s = make_sample_block(SampleOpts{1.0, false, true},
                               DbKey{"haloModel", key});
    put_sunayama23(s, 0.45, -0.1);
    Shear1h2hMax sel(cfg_bsel);
    INFO("bsel, omitted haloModel/" << key);
    CHECK_THROWS(sel.set_sample(s));
  }
}

TEST_CASE("Shear1h2hMax guard clause: bin_index out of range")
{
  auto cfg = make_cfg_block();
  auto s = make_sample_block();
  Shear1h2hMax mod(cfg);
  mod.set_sample(s);

  CHECK_THROWS_AS(mod.evaluate({2.0, 1.0}), std::out_of_range);
  CHECK_THROWS_AS(mod.evaluate({-1.0, 1.0}), std::out_of_range);
}

TEST_CASE("OperatorMax1h2h_gl reproduces the analytic NFW Sigma and DeltaSigma")
{
  // b = 0 and xi_nfw = rho_NFW / rho_m (no -1): the GL projection must
  // return the closed-form NFW profile (Wright & Brainerd 2000).
  double const lnM = std::log(1.0e14), c = 4.0, rho_m = RHO_M_REF;
  double const PI = 3.14159265358979323846;
  double const r200 = std::cbrt(3.0 * std::exp(lnM) / (800.0 * PI * rho_m));
  double const rs = r200 / c;
  double const rho_s = (200.0 / 3.0) * c * c * c /
                       (std::log1p(c) - c / (1.0 + c)) * rho_m;
  y3_cluster::OperatorMax1h2h_gl const op(
    [&](double r, double) { double x = r / rs; return rho_s / (rho_m * x * (1 + x) * (1 + x)); },
    [](double, double) { return 0.0; },
    [](double, double) { return 0.0; },
    rho_m, /*r_max=*/1.0e4, /*n_los=*/128, /*n_aperture=*/64);

  auto shape = [](double x, bool mean) {
    double const s = x < 1.0
      ? std::atanh(std::sqrt((1 - x) / (1 + x))) / std::sqrt(1 - x * x)
      : std::atan(std::sqrt((x - 1) / (1 + x))) / std::sqrt(x * x - 1);
    return mean ? (std::log(x / 2.0) + 2.0 * s) / (x * x)
                : (1.0 - 2.0 * s) / (x * x - 1.0);
  };
  for (double R : {0.1, 0.3, 1.0, 3.0}) {
    double const x = R / rs;
    double const sigma = 2.0 * rho_s * rs * shape(x, false) * 1.0e-12;
    double const mean = 4.0 * rho_s * rs * shape(x, true) * 1.0e-12;
    INFO("R " << R);
    CHECK(op.surface_density(R, lnM, 0.3) == Approx(sigma).epsilon(1e-4));
    CHECK(op.excess_surface_density(R, lnM, 0.3) ==
          Approx(mean - sigma).epsilon(1e-4));
  }
}

TEST_CASE("Shear1h2hMax gauss_legendre: tabulated wall == on-the-fly, finite")
{
  auto s = make_sample_block(SampleOpts{1.0, false, false, /*xi=*/true});
  auto cfg_wall = make_gl_cfg_block(/*with_wall=*/true);
  auto cfg_nowall = make_gl_cfg_block(/*with_wall=*/false);
  Shear1h2hMax cached(cfg_wall);
  Shear1h2hMax direct(cfg_nowall);
  cached.set_sample(s);
  direct.set_sample(s);
  for (int b : {0, 1})
    for (double R : R_QUERY) {
      auto const pt = Shear1h2hMax::grid_point_t{static_cast<double>(b), R};
      double const v = cached.evaluate(pt)[0];
      INFO("bin " << b << " R " << R);
      CHECK(std::isfinite(v));
      CHECK(v > 0.0);
      CHECK(v == Approx(direct.evaluate(pt)[0]).epsilon(1e-14));
    }

  // An unknown profile_method is a configuration error.
  auto cfg_bad = make_cfg_block();
  cfg_bad.put_val(Shear1h2hMax::module_label(), "profile_method",
                  std::string("bad"));
  CHECK_THROWS_AS(Shear1h2hMax{cfg_bad}, std::invalid_argument);
}

TEST_CASE("Shear1h2hMax gauss_legendre Bsel equals B DeltaSigma - M_sel/(pi R^2)")
{
  // The GL path weights the aperture nodes by B; integrating by parts this
  // is exactly the BselDeltaSigmaOperator (B') form.  Rebuild that form
  // independently from OperatorMax1h2h_gl and the same W2d stack.
  auto cfg = make_gl_cfg_block(true);
  cfg.put_val(Shear1h2hMax::module_label(), "bsel", 1);
  cfg.put_val(Shear1h2hMax::module_label(), "bsel_model",
              std::string("Sunayama23"));
  auto s = make_sample_block(SampleOpts{1.0, false, false, true});
  put_sunayama23(s, 0.45, -0.1);
  Shear1h2hMax mod(cfg);
  mod.set_sample(s);

  double const rho_m = RHO_M_REF;
  auto xi_nfw = [rho_m](double r, double lnM) {
    double const PI = 3.14159265358979323846, c = 4.0;
    double const r200 = std::cbrt(3.0 * std::exp(lnM) / (800.0 * PI * rho_m));
    double const x = r * c / r200;
    return (200.0 / 3.0) * c * c * c / (std::log1p(c) - c / (1.0 + c)) /
             (x * (1.0 + x) * (1.0 + x)) - 1.0;
  };
  auto const bias = y3_cluster::make_Interp2D(s, "haloModel", "lnM", "z", "bias");
  auto const xi_nl = y3_cluster::make_Interp2D(s, "xi_nl", "r", "z", "xi_nl", /*bicubic=*/true);
  y3_cluster::OperatorMax1h2h_gl const op(
    xi_nfw, [&](double m, double z) { return bias.clamp(m, z); },
    [&](double r, double z) { return xi_nl.clamp(r, z); }, rho_m, 1000.0);
  auto const bsel_op = y3_cluster::BselDeltaSigmaOperator::from_datablock(
    s, "Sunayama23");

  y3_cluster::HMF_t const hmf(s);
  y3_cluster::DV_DO_DZ_t const dv(s);
  y3_cluster::OMEGA_Z_DES const omega(s);
  auto const sci = y3_pipelines::load_sigma_crit_inv(s);
  std::vector<double> lnm_x, lnm_w, z_x, z_w;
  y3_pipelines::gl_nodes(LNM_LO, LNM_HI, N_LNM, lnm_x, lnm_w);
  y3_pipelines::gl_nodes(ZT_LO, ZT_HI, N_Z, z_x, z_w);

  int const b = 1;
  y3_cluster::SelFunction_t const sel(s, b);
  for (double R : R_QUERY) {
    double expected = 0.0;
    for (std::size_t k = 0; k != lnm_x.size(); ++k)
      for (std::size_t q = 0; q != z_x.size(); ++q) {
        double const lnM = lnm_x[k], z = z_x[q];
        double const w = lnm_w[k] * z_w[q] * dv(z) * omega(z) * sci.clamp(z) *
                         hmf(lnM, z) * sel(lnM, z);
        expected += w * bsel_op.apply(
          op.excess_surface_density(R, lnM, z),
          [&](double r) { return op.mean_surface_density(r, lnM, z); },
          R, b, LOB_CENTERS[b], z);
      }
    INFO("R " << R);
    CHECK(mod.evaluate({static_cast<double>(b), R})[0] ==
          Approx(expected).epsilon(1e-4));
  }
}

TEST_CASE("Shear1h2hMax gauss_legendre precision: default nodes vs doubled nodes")
{
  // GL convergence on log-spaced radii: the production nodes (n_los = 64,
  // n_aperture = 32) must agree with twice the nodes in both integrals.
  std::vector<double> radii;
  for (int i = 0; i != 12; ++i)
    radii.push_back(0.1 * std::pow(100.0, i / 11.0));   // 0.1 .. 10 Mpc/h
  auto s = make_sample_block(SampleOpts{3.0, false, false, /*xi=*/true});

  auto cfg = make_gl_cfg_block(false);
  cfg.put_val(Shear1h2hMax::module_label(), "r_perp", radii);
  auto cfg2 = cfg;
  cfg2.put_val(Shear1h2hMax::module_label(), "n_los", 128);
  cfg2.put_val(Shear1h2hMax::module_label(), "n_aperture", 64);
  Shear1h2hMax base(cfg), doubled(cfg2);
  base.set_sample(s);
  doubled.set_sample(s);

  double worst = 0.0;
  for (int b : {0, 1})
    for (double R : radii) {
      auto const pt = Shear1h2hMax::grid_point_t{static_cast<double>(b), R};
      double const v1 = base.evaluate(pt)[0];
      double const v2 = doubled.evaluate(pt)[0];
      worst = std::max(worst, std::abs(v1 / v2 - 1.0));
      INFO("bin " << b << " R " << R);
      CHECK(v1 == Approx(v2).epsilon(PRECISION_GL));
    }
  WARN("GL 64/32 vs 128/64: max |rel diff| = " << worst);
}

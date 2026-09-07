// Unit tests for the REAL traditional 1h+2h max-model header
// (src/pipelines/des_y3/shear_1h2h/cpp/0d/shear1h2h_max_t.hh, module
// Shear1h2hMax.cc).
//
// Why a second file next to test/shear1h2h_max.test.cc: that test was
// written when Shear1h2hMax lived entirely inside the module .cc and was
// private to that translation unit, so it pins a byte-for-byte MIRROR of
// the (lnM, z) accumulation rather than the class.  It is green and it
// still documents the composition contract, so it stays -- but it
// instantiates nothing from the header, which is why the coverage report
// shows shear1h2h_max_t.hh at 0% next to a passing test.  The class is
// now public in the header; this file drives it directly.
//
// Pinned here, on a self-contained synthetic DataBlock:
//
//   1. Shear1h2hMax::evaluate() vs an independently assembled
//      z-RESOLVED reference: the max model's two-halo term is
//      z-dependent, so unlike Shear1hGl the redshift integral cannot be
//      contracted out of the mass integral, and the reference below
//      reproduces that nesting from the leaf models directly.
//   2. Both branches of max(1h, b * 2h) are genuinely exercised (a
//      REQUIRE guards against the synthetic tables drifting into a
//      "the max model always returns 1h" degenerate case).
//   3. max(1h, 0) = 1h: an all-NaN dSigma_hh table (sanitized to zero by
//      the module's own make_sanitized_hh path, reached through
//      set_sample -- not a re-implementation) recovers the pure 1-halo
//      stack exactly, and no NaN reaches the output.
//   4. f_mis endpoints and include_miscentering = F.
//   5. The one_halo_physical_density opt-in path: the exact fixed-c
//      identity DSigma_phys(R|z) = (1+z)^2 DSigma_com(R (1+z)) applied at
//      every z node, with the 2-halo row untouched (CLAUDE.md, "Physical
//      density rho_m(z)").
//   6. The DataBlock contract: every required sample key, and every
//      required [Shear1h2hMax] ini option, fails LOUDLY when absent.
//      In particular miscentering/f_mis and miscentering/tau_mis are
//      required -- the header's docstring promises "no silent fallback to
//      the Y3 fiducial defaults", and that promise is what this pins.
//   7. Guard clauses: out-of-range bin_index, empty lob_centers.
//
// Requires Y3_CLUSTER_CPP_DIR (NFW_DSIGMA_MIS reads data/nfw_off_center/*).
#include "catch2/catch.hpp"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include "models/dv_do_dz_t.hh"
#include "models/hmf_t.hh"
#include "models/nfw_dsigma_mis.hh"
#include "models/omega_z_des.hh"
#include "models/sel_function_t.hh"
#include "pipelines/shared/lensing_helpers.hh"
#include "pipelines/shared/sel_gl_weights.hh"
#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"
#include "utils/make_interp_2d.hh"

#include "pipelines/des_y3/shear_1h2h/cpp/0d/shear1h2h_max_t.hh"

#include <algorithm>
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
  double constexpr F_MIS = 0.35;
  double constexpr TAU_MIS = 0.17;
  double constexpr RHO_M_REF = 0.3096 * 2.77533742639e+11;
  std::vector<double> const LOB_CENTERS{30.0, 90.0};
  std::vector<double> const R_QUERY{0.5, 1.0, 2.0};

  // dSigma_hh table shape.  `hh_amp` scales the whole two-halo table so a
  // single knob moves the composition between the two branches of the
  // max(); `hh_nan` replaces it with an all-NaN table (the historical
  // defect shape) to reach the module's sanitize path.
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

  struct SampleOpts {
    double f_mis = F_MIS;
    double tau_mis = TAU_MIS;
    double hh_amp = 1.0;
    bool hh_nan = false;
    bool phys_density = false;
  };

  // Two knots per axis, all tables either constant or linear along the one
  // axis the formula varies, so every interpolator is exact in range --
  // the same construction as shear1h_gl.test.cc.  bias is constant so the
  // max() branch is controlled purely by hh_amp.
  cosmosis::DataBlock
  make_sample_block(SampleOpts const& o = {})
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
              cosmosis::ndarray<double>(std::vector<double>(4, 7.0), {2, 2}));

    s.put_val("distances", "z", std::vector<double>{0.0, 1.0});
    s.put_val("distances", "d_a", std::vector<double>{1500.0, 1500.0});

    s.put_val("average_sigma_crit_inv", "zlense",
              std::vector<double>{0.0, 1.0});
    s.put_val("average_sigma_crit_inv", "sci_average",
              std::vector<double>{3.0e-4, 3.0e-4});

    s.put_val("sel_function", "lnM", std::vector<double>{LNM_LO, LNM_HI});
    s.put_val("sel_function", "z", std::vector<double>{ZT_LO, ZT_HI});
    std::vector<double> s_stack;
    for (double c : {2.0, 3.5})
      for (int i = 0; i != 4; ++i) s_stack.push_back(c);
    s.put_val("sel_function", "S_stack",
              cosmosis::ndarray<double>(s_stack, {2, 2, 2}));

    // haloModel: r_sigma x lnM for dSigma_nfw, lnM x z for bias,
    // r_sigma x z for dSigma_hh.  The physical-density branch queries
    // dSigma_nfw at R (1 + z), so the radial axis reaches well past
    // max(R_QUERY).
    s.put_val("haloModel", "r_sigma", std::vector<double>{0.05, 10.0});
    s.put_val("haloModel", "lnM", std::vector<double>{28.0, 37.0});
    s.put_val("haloModel", "z", std::vector<double>{0.0, 1.0});
    s.put_val("haloModel", "rho_m_ref", RHO_M_REF);
    auto const dsig = [](double r) { return 5.0 - 0.3 * r; };
    s.put_val("haloModel", "dSigma_nfw",
              cosmosis::ndarray<double>(
                std::vector<double>{dsig(0.05), dsig(10.0),
                                    dsig(0.05), dsig(10.0)},
                {2, 2}));
    // bias(lnM, z): constant, so it never chooses the max() branch by
    // itself -- hh_amp does.
    s.put_val("haloModel", "bias",
              cosmosis::ndarray<double>(std::vector<double>(4, 2.0), {2, 2}));
    // dSigma_hh(r_sigma, z) as the (n_z, n_r) row-major ndarray the module
    // documents; constant in both so the composition is hand-checkable.
    double const nan_d = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> hh(4, o.hh_nan ? nan_d : o.hh_amp);
    s.put_val("haloModel", "dSigma_hh",
              cosmosis::ndarray<double>(hh, {2, 2}));

    if (o.phys_density)
      s.put_val("haloModel", "one_halo_physical_density", 1);

    s.put_val("miscentering", "f_mis", o.f_mis);
    s.put_val("miscentering", "tau_mis", o.tau_mis);
    return s;
  }

  // Independent z-RESOLVED reference: recombines the same leaf models
  // Shear1h2hMax composes, but written as an explicit (lnM, z) double sum
  // rather than by reusing any of the module's own caches.
  struct ReferenceMax {
    y3_cluster::HMF_t hmf;
    y3_cluster::DV_DO_DZ_t dv;
    y3_cluster::OMEGA_Z_DES omega;
    y3_cluster::Interp1D sci;
    y3_cluster::Interp2D dsigma_nfw;
    y3_cluster::Interp2D bias;
    y3_cluster::Interp2D dsigma_hh;
    y3_cluster::NFW_DSIGMA_MIS dsigma_mis;
    std::vector<double> lnm_x, lnm_w, z_x, z_w;

    explicit ReferenceMax(cosmosis::DataBlock& s)
      : hmf(s)
      , dv(s)
      , omega(s)
      , sci(y3_pipelines::load_sigma_crit_inv(s))
      , dsigma_nfw(y3_cluster::make_Interp2D(s, "haloModel", "r_sigma",
                                             "lnM", "dSigma_nfw"))
      , bias(y3_cluster::make_Interp2D(s, "haloModel", "lnM", "z", "bias"))
      , dsigma_hh(sanitized_hh(s))
      , dsigma_mis(y3_cluster::CONC, y3_cluster::RHOC, y3_cluster::GAMMA)
    {
      dsigma_mis.set_rho_ref(s.view<double>("haloModel", "rho_m_ref"));
      y3_pipelines::gl_nodes(LNM_LO, LNM_HI, N_LNM, lnm_x, lnm_w);
      y3_pipelines::gl_nodes(ZT_LO, ZT_HI, N_Z, z_x, z_w);
    }

    static y3_cluster::Interp2D
    sanitized_hh(cosmosis::DataBlock& s)
    {
      auto const r = s.view<std::vector<double>>("haloModel", "r_sigma");
      auto const z = s.view<std::vector<double>>("haloModel", "z");
      auto const& nd =
        s.view<cosmosis::ndarray<double>>("haloModel", "dSigma_hh");
      std::vector<double> vals(nd.begin(), nd.end());
      for (auto& v : vals)
        if (!std::isfinite(v)) v = 0.0;
      return y3_cluster::Interp2D(r, z, vals);
    }

    double
    acc(cosmosis::DataBlock& s, int bin, double R, double f_mis,
        double r_mis, bool phys) const
    {
      y3_cluster::SelFunction_t const sel(s, bin);
      double total = 0.0;
      for (std::size_t k = 0; k != lnm_x.size(); ++k) {
        double const lnM = lnm_x[k];
        for (std::size_t q = 0; q != z_x.size(); ++q) {
          double const z = z_x[q];
          double const w = lnm_w[k] * z_w[q] * dv(z) * omega(z) *
                           sci.clamp(z) * hmf(lnM, z) * sel(lnM, z);
          double one;
          if (!phys) {
            one = (1.0 - f_mis) * dsigma_nfw.clamp(R, lnM) +
                  f_mis * dsigma_mis(R, r_mis, lnM);
          } else {
            double const qf = 1.0 + z;
            one = qf * qf *
                  ((1.0 - f_mis) * dsigma_nfw.clamp(R * qf, lnM) +
                   f_mis * dsigma_mis(R * qf, r_mis * qf, lnM));
          }
          double const two = bias.clamp(lnM, z) * dsigma_hh.clamp(R, z);
          total += w * std::max(one, two);
        }
      }
      return total;
    }
  };

  struct DbKey { char const* section; char const* key; };

  std::vector<DbKey> const REQUIRED_SAMPLE_KEYS{
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
    {"haloModel", "rho_m_ref"},
    {"haloModel", "dSigma_nfw"},
    {"haloModel", "bias"},
    {"haloModel", "dSigma_hh"},
    {"miscentering", "f_mis"},
    {"miscentering", "tau_mis"},
  };

  // The full block minus exactly one required key.
  cosmosis::DataBlock
  make_sample_block_omitting(DbKey const& skip)
  {
    auto keep = [&](char const* sec, char const* key) {
      return !(std::string(skip.section) == sec &&
               std::string(skip.key) == key);
    };
    cosmosis::DataBlock s;

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
      s.put_val("mass_function", "dndlnmh",
                cosmosis::ndarray<double>(std::vector<double>(4, 7.0),
                                          {2, 2}));

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

    if (keep("haloModel", "r_sigma"))
      s.put_val("haloModel", "r_sigma", std::vector<double>{0.05, 10.0});
    if (keep("haloModel", "lnM"))
      s.put_val("haloModel", "lnM", std::vector<double>{28.0, 37.0});
    if (keep("haloModel", "z"))
      s.put_val("haloModel", "z", std::vector<double>{0.0, 1.0});
    if (keep("haloModel", "rho_m_ref"))
      s.put_val("haloModel", "rho_m_ref", RHO_M_REF);
    if (keep("haloModel", "dSigma_nfw")) {
      auto const dsig = [](double r) { return 5.0 - 0.3 * r; };
      s.put_val("haloModel", "dSigma_nfw",
                cosmosis::ndarray<double>(
                  std::vector<double>{dsig(0.05), dsig(10.0),
                                      dsig(0.05), dsig(10.0)},
                  {2, 2}));
    }
    if (keep("haloModel", "bias"))
      s.put_val("haloModel", "bias",
                cosmosis::ndarray<double>(std::vector<double>(4, 2.0),
                                          {2, 2}));
    if (keep("haloModel", "dSigma_hh"))
      s.put_val("haloModel", "dSigma_hh",
                cosmosis::ndarray<double>(std::vector<double>(4, 1.0),
                                          {2, 2}));

    if (keep("miscentering", "f_mis"))
      s.put_val("miscentering", "f_mis", F_MIS);
    if (keep("miscentering", "tau_mis"))
      s.put_val("miscentering", "tau_mis", TAU_MIS);
    return s;
  }

  double
  r_mis_for(int bin, double tau_mis)
  {
    return tau_mis * y3_pipelines::R_lambda(LOB_CENTERS[bin]);
  }

} // namespace

TEST_CASE("Shear1h2hMax matches an independently assembled z-resolved reference")
{
  auto cfg = make_cfg_block();
  // hh_amp chosen below the 1-halo scale so BOTH branches appear across
  // the radial range (asserted explicitly in the next test case).
  auto s = make_sample_block(SampleOpts{F_MIS, TAU_MIS, /*hh_amp=*/1.5});

  Shear1h2hMax mod(cfg);
  mod.set_sample(s);
  ReferenceMax const ref(s);

  for (int b : {0, 1})
    for (double R : R_QUERY) {
      auto const out = mod.evaluate({static_cast<double>(b), R});
      double const expected = ref.acc(s, b, R, F_MIS, r_mis_for(b, TAU_MIS),
                                      /*phys=*/false);
      INFO("bin " << b << " R " << R);
      CHECK(out[0] == Approx(expected).epsilon(1e-9));
      CHECK(std::isfinite(out[0]));
      CHECK(out[0] > 0.0);
    }
}

TEST_CASE("Shear1h2hMax exercises both branches of max(1h, b * 2h)")
{
  // Guard against the synthetic tables silently degenerating into "the
  // max model always returns the one-halo term": with a constant
  // bias = 2 and a constant dSigma_hh, a large enough hh_amp must lift
  // the two-halo branch above the one-halo term at every node and make
  // the whole stack scale linearly with hh_amp; a tiny hh_amp must leave
  // the result identical to the 2h = 0 case.
  auto cfg = make_cfg_block();

  Shear1h2hMax lo(cfg);
  auto s_lo = make_sample_block(SampleOpts{F_MIS, TAU_MIS, /*hh_amp=*/1e-6});
  lo.set_sample(s_lo);

  Shear1h2hMax zero(cfg);
  auto s_zero = make_sample_block(SampleOpts{F_MIS, TAU_MIS, /*hh_amp=*/0.0});
  zero.set_sample(s_zero);

  Shear1h2hMax hi(cfg);
  auto s_hi = make_sample_block(SampleOpts{F_MIS, TAU_MIS, /*hh_amp=*/1e4});
  hi.set_sample(s_hi);

  Shear1h2hMax hi2(cfg);
  auto s_hi2 = make_sample_block(SampleOpts{F_MIS, TAU_MIS, /*hh_amp=*/2e4});
  hi2.set_sample(s_hi2);

  for (int b : {0, 1})
    for (double R : R_QUERY) {
      auto const pt = Shear1h2hMax::grid_point_t{static_cast<double>(b), R};
      // 1-halo branch wins everywhere.
      CHECK(lo.evaluate(pt)[0] == Approx(zero.evaluate(pt)[0]).epsilon(1e-12));
      // 2-halo branch wins everywhere -> exactly linear in hh_amp.
      CHECK(hi2.evaluate(pt)[0] == Approx(2.0 * hi.evaluate(pt)[0])
                                     .epsilon(1e-9));
      // ...and the two regimes really are different.
      CHECK(hi.evaluate(pt)[0] > 10.0 * zero.evaluate(pt)[0]);
    }
}

TEST_CASE("Shear1h2hMax sanitizes an all-NaN dSigma_hh to zero: max(1h, 0) = 1h")
{
  // The historical dSigma_hh defect shape (docs/known_issues/
  // dsigma_hh_debug_flag.md), driven through the module's OWN
  // make_sanitized_hh path via set_sample -- not through a
  // re-implementation of it, which is all the mirror test could do.
  auto cfg = make_cfg_block();

  Shear1h2hMax nan_mod(cfg);
  auto s_nan = make_sample_block(SampleOpts{F_MIS, TAU_MIS, 1.0,
                                            /*hh_nan=*/true});
  nan_mod.set_sample(s_nan);

  Shear1h2hMax zero_mod(cfg);
  auto s_zero = make_sample_block(SampleOpts{F_MIS, TAU_MIS, /*hh_amp=*/0.0});
  zero_mod.set_sample(s_zero);

  for (int b : {0, 1})
    for (double R : R_QUERY) {
      auto const pt = Shear1h2hMax::grid_point_t{static_cast<double>(b), R};
      double const v = nan_mod.evaluate(pt)[0];
      CHECK(std::isfinite(v));
      CHECK(v == Approx(zero_mod.evaluate(pt)[0]).epsilon(1e-12));
    }
}

TEST_CASE("Shear1h2hMax f_mis endpoints and include_miscentering = F")
{
  auto cfg = make_cfg_block();

  // f_mis = 0: the mixture is pure dSigma_nfw, so tau_mis and lob_centers
  // cannot matter at all.
  auto cfg_a = make_cfg_block(std::vector<double>{30.0, 90.0});
  auto cfg_b = make_cfg_block(std::vector<double>{5.0, 500.0});
  Shear1h2hMax a(cfg_a);
  auto s_a = make_sample_block(SampleOpts{0.0, 0.17});
  a.set_sample(s_a);
  Shear1h2hMax b(cfg_b);
  auto s_b = make_sample_block(SampleOpts{0.0, 0.9});
  b.set_sample(s_b);

  for (int bin : {0, 1})
    for (double R : R_QUERY) {
      auto const pt = Shear1h2hMax::grid_point_t{static_cast<double>(bin), R};
      CHECK(a.evaluate(pt)[0] == Approx(b.evaluate(pt)[0]).epsilon(1e-12));
      CHECK(a.evaluate(pt)[0] > 0.0);
    }

  // include_miscentering = F forces f_mis -> 0 regardless of the
  // datablock value (tau_mis is still read, and still required).
  auto cfg_nomis = make_cfg_block();
  cfg_nomis.put_val(Shear1h2hMax::module_label(), "include_miscentering", 0);
  Shear1h2hMax off(cfg_nomis);
  auto s_on = make_sample_block(SampleOpts{/*f_mis=*/0.8, TAU_MIS});
  off.set_sample(s_on);

  for (int bin : {0, 1})
    for (double R : R_QUERY) {
      auto const pt = Shear1h2hMax::grid_point_t{static_cast<double>(bin), R};
      CHECK(off.evaluate(pt)[0] == Approx(a.evaluate(pt)[0]).epsilon(1e-12));
    }
}

TEST_CASE("Shear1h2hMax one_halo_physical_density applies the exact (1+z)^2 identity")
{
  // CLAUDE.md, "Physical density rho_m(z)": the opt-in mode replaces the
  // z-free 1-halo mixture with (1+z)^2 * DSigma_com(R (1+z)) evaluated at
  // every z node, leaving the 2-halo row on its own convention.  The
  // reference reproduces exactly that, node by node.
  auto cfg = make_cfg_block();
  auto s = make_sample_block(SampleOpts{F_MIS, TAU_MIS, /*hh_amp=*/1.5,
                                        /*hh_nan=*/false,
                                        /*phys_density=*/true});

  Shear1h2hMax mod(cfg);
  mod.set_sample(s);
  ReferenceMax const ref(s);

  for (int b : {0, 1})
    for (double R : R_QUERY) {
      auto const out = mod.evaluate({static_cast<double>(b), R});
      double const expected = ref.acc(s, b, R, F_MIS, r_mis_for(b, TAU_MIS),
                                      /*phys=*/true);
      INFO("bin " << b << " R " << R);
      CHECK(out[0] == Approx(expected).epsilon(1e-9));
    }

  // The flag genuinely changes the answer (a no-op would make this test
  // vacuous).
  Shear1h2hMax com(cfg);
  auto s_com = make_sample_block(SampleOpts{F_MIS, TAU_MIS, 1.5});
  com.set_sample(s_com);
  auto const pt = Shear1h2hMax::grid_point_t{0.0, 1.0};
  CHECK(mod.evaluate(pt)[0] != Approx(com.evaluate(pt)[0]).epsilon(1e-6));
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
  //   include_miscentering                    optional, default T
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

  // lob_centers omitted -> the DES-Y3 default {25, 37.5, 52.5, 130}, so
  // construction succeeds and the miscentring radius follows that default.
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
  // Unlike NumCountsSijGl, this backend DOES require
  // average_sigma_crit_inv (the shear weight folds Sigma_crit^-1 in) and
  // the full haloModel lensing trio (dSigma_nfw, bias, dSigma_hh, plus
  // rho_m_ref for the unified rho_m convention).  miscentering/f_mis and
  // miscentering/tau_mis are required too -- the header promises "no
  // silent fallback to the Y3 fiducial defaults".
  auto cfg = make_cfg_block();

  for (auto const& k : REQUIRED_SAMPLE_KEYS) {
    auto s = make_sample_block_omitting(k);
    Shear1h2hMax mod(cfg);
    INFO("omitted " << k.section << "/" << k.key);
    CHECK_THROWS(mod.set_sample(s));
  }

  auto s_full = make_sample_block_omitting(DbKey{"none", "none"});
  Shear1h2hMax mod(cfg);
  REQUIRE_NOTHROW(mod.set_sample(s_full));
  CHECK(mod.evaluate({0.0, 1.0})[0] > 0.0);
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

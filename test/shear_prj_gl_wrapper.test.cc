// End-to-end unit tests for the projection-shear 0d C++ driver
// (src/pipelines/des_y3/shear_projection/cpp/0d/shear_prj_gl_t.hh, module
// ShearPrjGl.cc) and, through it, the sample-level machinery of
// y3_cluster::sp_detail::ShearPrjCore
// (src/pipelines/systematics/shear_prj/cpp/sigma_prj_t.hh).
//
// The existing test/shear_prj_gl.test.cc deliberately stops at the
// DataBlock-free helpers (build_theta_grid, theta_excl_at_z, R_lambda),
// on the grounds that ShearPrjCore::set_sample needs "essentially all of
// real_pipeline_extract.ini".  That left the wrapper at 0% and the core at
// ~30%: set_sample, the per-slice caches, accumulate_, locate_, and
// evaluate() were never executed by any test.  This file builds that
// sample block -- small but complete -- and drives the wrapper.
//
// It does NOT re-derive the projection integral: reproducing set_sample's
// z-grid, theta-grid and (theta, M) caches independently would be a
// second copy of the implementation, and the accuracy claim against the
// adaptive reference is the job of the validate_*.py harnesses (CLAUDE.md,
// "Testing precision and cost").  What is pinned here are the exact
// structural and linearity invariants the algorithm must satisfy for ANY
// input -- the kind of property a re-derivation cannot accidentally share:
//
//   1. Output layout: the 6 outputs are
//      {dsigma_prj vals, rnd, cl, shear_prj vals, rnd, cl} in that order,
//      and the two sections are the des_y3-namespaced ones.
//   2. total == rnd + cl exactly, on both observables.
//   3. shear channel == dsigma channel * Sigma_crit_inv(zob), exactly and
//      channel by channel -- the one algebraic step the wrapper's
//      shear_prj() adds on top of dsigma_prj().
//   4. Cache correctness: re-running set_sample on the same block is
//      idempotent, and a changed block really does recompute (the caches
//      are rebuilt from scratch, not reused across samples).
//   5. Sigma_crit_inv is OPTIONAL: with no average_sigma_crit_inv section
//      the dsigma channels are unchanged and the shear channels collapse
//      to exactly 0 (the documented smoke-pipeline behaviour).
//   6. b_sel(theta) enters ONLY the clustered channel, linearly:
//      b_small = b_large = B makes cl exactly proportional to B, and
//      B = 0 kills cl while leaving rnd bit-identical.
//   7. xi_NL = 0 kills the clustered channel entirely (total == rnd).
//   8. The HMF amplitude scales both channels linearly; the halo-bias
//      amplitude scales only the clustered one.
//   9. The line-of-sight slab exclusion actually excludes: a larger
//      R_excl (driven by a larger richness-bin centre through R_lambda)
//      removes clustered contribution.
//  10. The ini-option and DataBlock contracts: which [ShearPrjGl] keys are
//      required vs defaulted, and which sample sections must be present.
//
// Requires Y3_CLUSTER_CPP_DIR (the NFW miscentering tables under data/).
#include "catch2/catch.hpp"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include "pipelines/des_y3/shear_projection/cpp/0d/shear_prj_gl_t.hh"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  double constexpr ZT_LO = 0.15;
  double constexpr ZT_HI = 0.75;
  double constexpr LNM_LO = 30.0;
  double constexpr LNM_HI = 35.5;
  double constexpr RHO_M_REF = 0.3096 * 2.77533742639e+11;
  double constexpr SCI = 3.0e-4;

  // Small but structurally complete wall: 2 richness bins x 2 observed-z
  // bins x 2 radii = 8 grid points, so slice grouping (unique (lam, zob)
  // pairs sharing one theta grid) and the per-R caches are both exercised.
  std::vector<double> const WALL_LAM{0, 0, 1, 1, 0, 0, 1, 1};
  std::vector<double> const WALL_ZLO{0.20, 0.20, 0.20, 0.20,
                                     0.35, 0.35, 0.35, 0.35};
  std::vector<double> const WALL_ZHI{0.35, 0.35, 0.35, 0.35,
                                     0.50, 0.50, 0.50, 0.50};
  std::vector<double> const WALL_R{0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5};

  // Keep the quadrature small: this test is about structure, not accuracy,
  // and ShearPrjCore's defaults (24 x 30 x 20 x 20) would make every one of
  // the ~15 set_sample calls below expensive.
  // NOTE: cosmosis put_val does NOT overwrite an existing key (CLAUDE.md),
  // so every knob a test wants to vary has to be a parameter here rather
  // than something put on top of a pre-built block.
  cosmosis::DataBlock
  make_cfg_block(double r_max_cmpch = 35.0)
  {
    cosmosis::DataBlock cfg;
    char const* label = ShearPrjGl::module_label();
    cfg.put_val(label, "zt_low", ZT_LO);
    cfg.put_val(label, "zt_high", ZT_HI);
    cfg.put_val(label, "lnm_low", LNM_LO);
    cfg.put_val(label, "lnm_high", LNM_HI);
    cfg.put_val(label, "R_max_cMpch", r_max_cmpch);
    cfg.put_val(label, "n_lnm", 6);
    cfg.put_val(label, "n_per_seg", 4);
    cfg.put_val(label, "n_zring", 6);
    cfg.put_val(label, "n_zouter", 8);
    cfg.put_val(label, "lambda_bin", WALL_LAM);
    cfg.put_val(label, "zo_low", WALL_ZLO);
    cfg.put_val(label, "zo_high", WALL_ZHI);
    cfg.put_val(label, "radii", WALL_R);
    return cfg;
  }

  struct SampleOpts {
    double hmf_amp = 7.0;
    double bias_amp = 2.0;
    double xi_amp = 1.0;
    double b_small = 1.2;
    double b_large = 2.4;
    bool with_sci = true;
    // Richness-bin centres on the b_sel wall.  R_excl = R_lambda(lob) *
    // (1 + zob), so raising these widens the line-of-sight slab exclusion.
    double lob0 = 25.0;
    double lob1 = 37.5;
  };

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
    s.put_val("mass_function", "z", std::vector<double>{0.0, 2.5});
    s.put_val("mass_function", "dndlnmh",
              cosmosis::ndarray<double>(std::vector<double>(4, o.hmf_amp),
                                        {2, 2}));

    // chi(z) = 3000 z Mpc (comoving); d_a = chi / (1 + z).  Linear in z on
    // a grid that comfortably covers both the integration window and the
    // z in [0.001, 2] range ShearPrjCore's invert_chi_ bisects over.
    std::vector<double> const z_grid{0.0, 0.5, 1.0, 1.5, 2.0, 2.5};
    std::vector<double> d_c, d_a;
    for (double z : z_grid) {
      d_c.push_back(3000.0 * z);
      d_a.push_back(3000.0 * z / (1.0 + z));
    }
    s.put_val("distances", "z", z_grid);
    s.put_val("distances", "d_c", d_c);
    s.put_val("distances", "d_a", d_a);

    if (o.with_sci) {
      s.put_val("average_sigma_crit_inv", "zlense",
                std::vector<double>{0.0, 2.5});
      s.put_val("average_sigma_crit_inv", "sci_average",
                std::vector<double>{SCI, SCI});
    }

    // haloModel: lnM x z bias table plus the unified rho_m reference.
    s.put_val("haloModel", "lnM", std::vector<double>{28.0, 37.0});
    s.put_val("haloModel", "z", std::vector<double>{0.0, 2.5});
    s.put_val("haloModel", "bias",
              cosmosis::ndarray<double>(std::vector<double>(4, o.bias_amp),
                                        {2, 2}));
    s.put_val("haloModel", "rho_m_ref", RHO_M_REF);

    // xi_NL(r, z): constant in z, mildly decreasing in r, over a radial
    // range that covers every |dchi| the geometry can produce (queries are
    // clamped, so the ends are safe).
    std::vector<double> const xi_r{0.01, 10.0, 100.0, 3000.0};
    std::vector<double> const xi_z{0.0, 2.5};
    std::vector<double> xi;
    for (double z : xi_z) {
      (void)z;
      for (double r : xi_r) xi.push_back(o.xi_amp * 10.0 / (1.0 + r));
    }
    s.put_val("xi_nl", "r", xi_r);
    s.put_val("xi_nl", "z", xi_z);
    s.put_val("xi_nl", "xi_nl",
              cosmosis::ndarray<double>(xi, {xi_z.size(), xi_r.size()}));

    // The b_sel_marginalised wall: one row per (lambda_bin, zo bin).
    s.put_val("b_sel_marginalised", "lambda_bin",
              std::vector<double>{0, 1, 0, 1});
    s.put_val("b_sel_marginalised", "zo_low",
              std::vector<double>{0.20, 0.20, 0.35, 0.35});
    s.put_val("b_sel_marginalised", "zo_high",
              std::vector<double>{0.35, 0.35, 0.50, 0.50});
    s.put_val("b_sel_marginalised", "zob",
              std::vector<double>{0.275, 0.275, 0.425, 0.425});
    s.put_val("b_sel_marginalised", "lob",
              std::vector<double>{o.lob0, o.lob1, o.lob0, o.lob1});
    s.put_val("b_sel_marginalised", "b_small",
              std::vector<double>(4, o.b_small));
    s.put_val("b_sel_marginalised", "b_large",
              std::vector<double>(4, o.b_large));
    return s;
  }

  struct DbKey { char const* section; char const* key; };

  // The full sample block minus exactly one required key.  DataBlock has
  // no erase(), so the block is rebuilt with that key skipped.
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
      s.put_val("mass_function", "z", std::vector<double>{0.0, 2.5});
    if (keep("mass_function", "dndlnmh"))
      s.put_val("mass_function", "dndlnmh",
                cosmosis::ndarray<double>(std::vector<double>(4, 7.0),
                                          {2, 2}));

    std::vector<double> const z_grid{0.0, 0.5, 1.0, 1.5, 2.0, 2.5};
    std::vector<double> d_c, d_a;
    for (double z : z_grid) {
      d_c.push_back(3000.0 * z);
      d_a.push_back(3000.0 * z / (1.0 + z));
    }
    if (keep("distances", "z")) s.put_val("distances", "z", z_grid);
    if (keep("distances", "d_c")) s.put_val("distances", "d_c", d_c);
    if (keep("distances", "d_a")) s.put_val("distances", "d_a", d_a);

    s.put_val("average_sigma_crit_inv", "zlense",
              std::vector<double>{0.0, 2.5});
    s.put_val("average_sigma_crit_inv", "sci_average",
              std::vector<double>{SCI, SCI});

    if (keep("haloModel", "lnM"))
      s.put_val("haloModel", "lnM", std::vector<double>{28.0, 37.0});
    if (keep("haloModel", "z"))
      s.put_val("haloModel", "z", std::vector<double>{0.0, 2.5});
    if (keep("haloModel", "bias"))
      s.put_val("haloModel", "bias",
                cosmosis::ndarray<double>(std::vector<double>(4, 2.0),
                                          {2, 2}));
    if (keep("haloModel", "rho_m_ref"))
      s.put_val("haloModel", "rho_m_ref", RHO_M_REF);

    std::vector<double> const xi_r{0.01, 10.0, 100.0, 3000.0};
    std::vector<double> const xi_z{0.0, 2.5};
    std::vector<double> xi;
    for (std::size_t j = 0; j != xi_z.size(); ++j)
      for (double r : xi_r) xi.push_back(10.0 / (1.0 + r));
    if (keep("xi_nl", "r")) s.put_val("xi_nl", "r", xi_r);
    if (keep("xi_nl", "z")) s.put_val("xi_nl", "z", xi_z);
    if (keep("xi_nl", "xi_nl"))
      s.put_val("xi_nl", "xi_nl",
                cosmosis::ndarray<double>(xi, {xi_z.size(), xi_r.size()}));

    if (keep("b_sel_marginalised", "lambda_bin"))
      s.put_val("b_sel_marginalised", "lambda_bin",
                std::vector<double>{0, 1, 0, 1});
    if (keep("b_sel_marginalised", "zo_low"))
      s.put_val("b_sel_marginalised", "zo_low",
                std::vector<double>{0.20, 0.20, 0.35, 0.35});
    if (keep("b_sel_marginalised", "zo_high"))
      s.put_val("b_sel_marginalised", "zo_high",
                std::vector<double>{0.35, 0.35, 0.50, 0.50});
    if (keep("b_sel_marginalised", "zob"))
      s.put_val("b_sel_marginalised", "zob",
                std::vector<double>{0.275, 0.275, 0.425, 0.425});
    if (keep("b_sel_marginalised", "lob"))
      s.put_val("b_sel_marginalised", "lob",
                std::vector<double>{25.0, 37.5, 25.0, 37.5});
    if (keep("b_sel_marginalised", "b_small"))
      s.put_val("b_sel_marginalised", "b_small", std::vector<double>(4, 1.2));
    if (keep("b_sel_marginalised", "b_large"))
      s.put_val("b_sel_marginalised", "b_large", std::vector<double>(4, 2.4));
    return s;
  }

  // The 8 wall points as evaluator grid points (lam, zo_low, zo_high, R).
  std::vector<ShearPrjGl::grid_point_t>
  wall_points()
  {
    std::vector<ShearPrjGl::grid_point_t> pts;
    for (std::size_t i = 0; i != WALL_LAM.size(); ++i)
      pts.push_back({WALL_LAM[i], WALL_ZLO[i], WALL_ZHI[i], WALL_R[i]});
    return pts;
  }

  std::vector<std::array<double, 6>>
  evaluate_wall(cosmosis::DataBlock& cfg, cosmosis::DataBlock& s)
  {
    ShearPrjGl mod(cfg);
    mod.set_sample(s);
    std::vector<std::array<double, 6>> out;
    for (auto const& pt : wall_points()) out.push_back(mod.evaluate(pt));
    return out;
  }

} // namespace

TEST_CASE("ShearPrjGl publishes six outputs in the documented order and sections")
{
  REQUIRE(ShearPrjGl::n_outputs == 6);
  auto const secs = ShearPrjGl::output_sections();
  auto const names = ShearPrjGl::output_names();
  CHECK(std::string(ShearPrjGl::module_label()) == "ShearPrjGl");
  for (int i = 0; i != 3; ++i) {
    CHECK(std::string(secs[i]) == "dsigma_prj_gl");
    CHECK(std::string(secs[i + 3]) == "shear_prj_gl");
  }
  CHECK(std::string(names[0]) == "vals");
  CHECK(std::string(names[1]) == "rnd");
  CHECK(std::string(names[2]) == "cl");
  CHECK(std::string(names[3]) == "vals");
  CHECK(std::string(names[4]) == "rnd");
  CHECK(std::string(names[5]) == "cl");
}

TEST_CASE("ShearPrjGl evaluates the whole wall: finite, decomposed, and Sigma_crit-scaled")
{
  auto cfg = make_cfg_block();
  auto s = make_sample_block();
  auto const out = evaluate_wall(cfg, s);
  REQUIRE(out.size() == WALL_LAM.size());

  bool any_cl = false;
  for (std::size_t i = 0; i != out.size(); ++i) {
    auto const& v = out[i];
    INFO("wall point " << i);
    for (double x : v) CHECK(std::isfinite(x));

    // SIGN NOTE (not a defect): both channels come out NEGATIVE here.
    // The projection branch uses the SINGLE (delta-offset) miscentering
    // kernel, whose signed DSigma_mis(R, R_off) has a negative lobe
    // wherever R << R_off -- and the theta integral runs out to
    // R_max_cMpch = 35 Mpc/h, so almost every (theta, M) node contributes
    // from that lobe at these small R.  This is the same signed-table
    // behaviour test/signed_mis_table.test.py pins directly.  The
    // invariants below are therefore stated on magnitudes and exact
    // decompositions, never on a presumed sign.
    CHECK(v[0] != 0.0);
    CHECK(v[1] != 0.0);
    CHECK(v[0] == Approx(v[1] + v[2]).epsilon(1e-12));
    CHECK(v[3] == Approx(v[4] + v[5]).epsilon(1e-12));
    if (v[2] != 0.0) any_cl = true;

    // gamma_t^prj = DSigma_prj * Sigma_crit^-1, channel by channel.  The
    // synthetic sci table is flat, so the factor is exactly SCI.
    CHECK(v[3] == Approx(v[0] * SCI).epsilon(1e-12));
    CHECK(v[4] == Approx(v[1] * SCI).epsilon(1e-12));
    CHECK(v[5] == Approx(v[2] * SCI).epsilon(1e-12));
  }
  // If the clustered channel were identically zero for every wall point,
  // most of the checks below would be vacuous.
  CHECK(any_cl);

  // The two radii on each (lambda, zob) slice must give DIFFERENT answers:
  // Smis/DSmis get their own per-R cache row while the theta grid and the
  // z-contracted weights are shared, so a mis-keyed per-R cache would hand
  // both radii the same row.
  for (std::size_t i = 0; i < out.size(); i += 2) {
    INFO("slice starting at wall point " << i);
    CHECK(out[i][0] != Approx(out[i + 1][0]).epsilon(1e-6));
    CHECK(out[i][1] != Approx(out[i + 1][1]).epsilon(1e-6));
  }
}

TEST_CASE("ShearPrjGl caches: set_sample is idempotent and a new sample recomputes")
{
  auto cfg = make_cfg_block();
  auto s = make_sample_block();

  ShearPrjGl mod(cfg);
  mod.set_sample(s);
  std::vector<std::array<double, 6>> first;
  for (auto const& pt : wall_points()) first.push_back(mod.evaluate(pt));

  // Re-running set_sample on the identical block must rebuild the caches
  // to exactly the same state -- no accumulation into stale buffers.
  mod.set_sample(s);
  for (std::size_t i = 0; i != first.size(); ++i) {
    auto const v = mod.evaluate(wall_points()[i]);
    for (int c = 0; c != 6; ++c) {
      INFO("wall point " << i << " channel " << c);
      CHECK(v[c] == Approx(first[i][c]).epsilon(1e-15));
    }
  }

  // ...and feeding a genuinely different sample through the SAME object
  // must produce the new answer, not the cached one.
  auto s2 = make_sample_block(SampleOpts{/*hmf_amp=*/14.0});
  mod.set_sample(s2);
  for (std::size_t i = 0; i != first.size(); ++i) {
    auto const v = mod.evaluate(wall_points()[i]);
    INFO("wall point " << i);
    CHECK(v[0] == Approx(2.0 * first[i][0]).epsilon(1e-9));
  }

  // A fresh object on the second sample agrees with the reused one: no
  // state leaks across set_sample calls.
  auto const fresh = evaluate_wall(cfg, s2);
  for (std::size_t i = 0; i != fresh.size(); ++i) {
    auto const v = mod.evaluate(wall_points()[i]);
    for (int c = 0; c != 6; ++c)
      CHECK(v[c] == Approx(fresh[i][c]).epsilon(1e-15));
  }
}

TEST_CASE("ShearPrjGl treats average_sigma_crit_inv as optional")
{
  // Documented behaviour: "when absent (e.g. smoke pipelines that only care
  // about Sigma_prj / DSigma_prj) sci_ is left empty and g_t outputs
  // collapse to 0" -- the DSigma channels must be untouched.
  auto cfg = make_cfg_block();
  auto s_with = make_sample_block();
  auto s_without = make_sample_block(SampleOpts{7.0, 2.0, 1.0, 1.2, 2.4,
                                                /*with_sci=*/false});

  auto const a = evaluate_wall(cfg, s_with);
  auto const b = evaluate_wall(cfg, s_without);

  for (std::size_t i = 0; i != a.size(); ++i) {
    INFO("wall point " << i);
    for (int c = 0; c != 3; ++c)
      CHECK(b[i][c] == Approx(a[i][c]).epsilon(1e-12));
    for (int c = 3; c != 6; ++c)
      CHECK(b[i][c] == Approx(0.0).margin(1e-300));
  }
}

TEST_CASE("ShearPrjGl b_sel(theta) enters only the clustered channel, linearly")
{
  // With b_small = b_large = B the sigmoid reconstruction is the constant
  // B at every theta node, so cl must be exactly proportional to B and rnd
  // must not move at all.  This is the cleanest available handle on the
  // b_sel wall -> b_sel(theta) -> accumulate_ path.
  auto cfg = make_cfg_block();

  auto s1 = make_sample_block(SampleOpts{7.0, 2.0, 1.0,
                                         /*b_small=*/1.0, /*b_large=*/1.0});
  auto s3 = make_sample_block(SampleOpts{7.0, 2.0, 1.0,
                                         /*b_small=*/3.0, /*b_large=*/3.0});
  auto s0 = make_sample_block(SampleOpts{7.0, 2.0, 1.0,
                                         /*b_small=*/0.0, /*b_large=*/0.0});

  auto const v1 = evaluate_wall(cfg, s1);
  auto const v3 = evaluate_wall(cfg, s3);
  auto const v0 = evaluate_wall(cfg, s0);

  for (std::size_t i = 0; i != v1.size(); ++i) {
    INFO("wall point " << i);
    // rnd is b_sel-free.
    CHECK(v3[i][1] == Approx(v1[i][1]).epsilon(1e-12));
    CHECK(v0[i][1] == Approx(v1[i][1]).epsilon(1e-12));
    // cl is exactly linear in the (constant) b_sel.
    CHECK(v3[i][2] == Approx(3.0 * v1[i][2]).epsilon(1e-9));
    CHECK(v0[i][2] == Approx(0.0).margin(1e-300));
    // ...and with b_sel = 0 the total collapses onto rnd.
    CHECK(v0[i][0] == Approx(v0[i][1]).epsilon(1e-12));
  }
}

TEST_CASE("ShearPrjGl: xi_NL = 0 kills the clustered channel; HMF and bias scale as expected")
{
  auto cfg = make_cfg_block();
  auto base = make_sample_block();
  auto const v = evaluate_wall(cfg, base);

  // xi_NL = 0 -> no correlated line-of-sight structure -> cl = 0.
  auto s_noxi = make_sample_block(SampleOpts{7.0, 2.0, /*xi_amp=*/0.0});
  auto const v_noxi = evaluate_wall(cfg, s_noxi);
  for (std::size_t i = 0; i != v.size(); ++i) {
    INFO("wall point " << i);
    CHECK(v_noxi[i][2] == Approx(0.0).margin(1e-300));
    CHECK(v_noxi[i][0] == Approx(v_noxi[i][1]).epsilon(1e-12));
    // rnd is xi-free.
    CHECK(v_noxi[i][1] == Approx(v[i][1]).epsilon(1e-12));
  }

  // The halo-bias table multiplies only the clustered accumulator.
  auto s_bias = make_sample_block(SampleOpts{7.0, /*bias_amp=*/4.0});
  auto const v_bias = evaluate_wall(cfg, s_bias);
  for (std::size_t i = 0; i != v.size(); ++i) {
    INFO("wall point " << i);
    CHECK(v_bias[i][1] == Approx(v[i][1]).epsilon(1e-12));
    CHECK(v_bias[i][2] == Approx(2.0 * v[i][2]).epsilon(1e-9));
  }

  // The mass function multiplies both.
  auto s_hmf = make_sample_block(SampleOpts{/*hmf_amp=*/21.0});
  auto const v_hmf = evaluate_wall(cfg, s_hmf);
  for (std::size_t i = 0; i != v.size(); ++i) {
    INFO("wall point " << i);
    CHECK(v_hmf[i][1] == Approx(3.0 * v[i][1]).epsilon(1e-9));
    CHECK(v_hmf[i][2] == Approx(3.0 * v[i][2]).epsilon(1e-9));
  }
}

TEST_CASE("ShearPrjGl theta grid: R_max_cMpch sets how much structure is integrated")
{
  // theta_max = max(R_max_cMpch / D_A_o, 3 theta_R) is the outer bound of
  // the per-slice theta quadrature, and the ONLY breakpoint it moves is
  // the last one: build_theta_grid's breakpoint set is
  // { lower, theta_excl_o, theta_R(R_i)..., theta_lam, 2 theta_lam,
  //   theta_max }, all of which except theta_max are R_max-free.  So
  // widening R_max integrates the SAME integrand over a strictly larger
  // outermost segment, leaving every inner segment's nodes untouched --
  // the cleanest single-knob handle on the theta-grid half of set_sample,
  // since it moves neither the b_sel reconstruction nor the z grid.
  //
  // Out there theta * D_A >> R, which is exactly where the signed
  // single-offset DSigma_mis is NEGATIVE (see the SIGN NOTE above), so the
  // added contribution is negative and both accumulators must strictly
  // DECREASE.  This is asserted signed, not on |value|: |rnd| is a
  // cancellation between the positive core (theta * D_A < R) and that
  // negative tail, so it is not monotone (measured: at R = 1.5, |rnd|
  // shrinks as the tail grows).
  auto cfg_narrow = make_cfg_block(/*R_max_cMpch=*/8.0);
  auto cfg_wide = make_cfg_block(/*R_max_cMpch=*/35.0);

  auto s = make_sample_block();
  auto const narrow = evaluate_wall(cfg_narrow, s);
  auto const wide = evaluate_wall(cfg_wide, s);

  for (std::size_t i = 0; i != narrow.size(); ++i) {
    INFO("wall point " << i);
    CHECK(wide[i][1] < narrow[i][1]);
    CHECK(wide[i][2] < narrow[i][2]);
    CHECK(std::isfinite(wide[i][0]));
    CHECK(std::isfinite(narrow[i][0]));
  }
}

TEST_CASE("ShearPrjGl consumes the wall's lob through R_excl and theta_lambda")
{
  // b_sel_marginalised/lob is not an output passthrough: it drives
  // R_lambda(lob), hence R_excl (the line-of-sight slab exclusion and the
  // ring in z) AND theta_lambda (the b_sel sigmoid's scale and midpoint).
  // Changing it while holding b_small/b_large fixed must therefore change
  // the clustered channel.
  //
  // Deliberately NOT asserted as monotone in lob: lob moves the exclusion,
  // the theta breakpoints and the b_sel(theta) profile simultaneously and
  // in opposite directions, so |cl| is not monotone in it (measured: at
  // R = 0.5 raising lob lowers |cl|, at R = 1.5 it raises it).  Any
  // "exclusion always reduces cl" claim would be pinning an artefact of
  // one radius, not a property of the algorithm.
  auto cfg = make_cfg_block();
  auto s_small = make_sample_block(SampleOpts{7.0, 2.0, 1.0, 1.2, 2.4, true,
                                              /*lob0=*/25.0, /*lob1=*/25.0});
  auto s_large = make_sample_block(SampleOpts{7.0, 2.0, 1.0, 1.2, 2.4, true,
                                              /*lob0=*/2000.0,
                                              /*lob1=*/2000.0});
  auto const a = evaluate_wall(cfg, s_small);
  auto const b = evaluate_wall(cfg, s_large);

  for (std::size_t i = 0; i != a.size(); ++i) {
    INFO("wall point " << i);
    CHECK(b[i][2] != Approx(a[i][2]).epsilon(1e-6));
    CHECK(std::isfinite(b[i][2]));
    // rnd carries no b_sel factor but DOES see the changed theta grid, so
    // it must move too -- if it did not, lob would not be reaching the
    // geometry at all.
    CHECK(b[i][1] != Approx(a[i][1]).epsilon(1e-6));
  }
}

TEST_CASE("ShearPrjGl ini-option contract: [ShearPrjGl] required vs defaulted")
{
  // Cross-checked against cosmosis-models/des_y3.ini's [ShearPrjGl]:
  //
  //   file        = .../des_y3_shear_prj_0d_cpp/ShearPrjGl.so
  //   zt_low / zt_high / lnm_low / lnm_high    REQUIRED
  //   lambda_bin / zo_low / zo_high / radii    REQUIRED (the zipped wall)
  //   R_max_cMpch = 35.0                       optional, default 30.0
  //   n_lnm / n_per_seg / n_zring / n_zouter   optional (24/30/20/20)
  //   lob_centers / theta_breakpoints          optional
  char const* label = ShearPrjGl::module_label();
  auto base_cfg = [&]() {
    cosmosis::DataBlock c;
    c.put_val(label, "zt_low", ZT_LO);
    c.put_val(label, "zt_high", ZT_HI);
    c.put_val(label, "lnm_low", LNM_LO);
    c.put_val(label, "lnm_high", LNM_HI);
    c.put_val(label, "lambda_bin", WALL_LAM);
    c.put_val(label, "zo_low", WALL_ZLO);
    c.put_val(label, "zo_high", WALL_ZHI);
    c.put_val(label, "radii", WALL_R);
    return c;
  };

  for (char const* missing : {"zt_low", "zt_high", "lnm_low", "lnm_high",
                              "lambda_bin", "zo_low", "zo_high", "radii"}) {
    cosmosis::DataBlock c;
    auto put_d = [&](char const* k, double v) {
      if (std::string(k) != missing) c.put_val(label, k, v);
    };
    auto put_v = [&](char const* k, std::vector<double> const& v) {
      if (std::string(k) != missing) c.put_val(label, k, v);
    };
    put_d("zt_low", ZT_LO);
    put_d("zt_high", ZT_HI);
    put_d("lnm_low", LNM_LO);
    put_d("lnm_high", LNM_HI);
    put_v("lambda_bin", WALL_LAM);
    put_v("zo_low", WALL_ZLO);
    put_v("zo_high", WALL_ZHI);
    put_v("radii", WALL_R);
    INFO("missing [ShearPrjGl] " << missing);
    CHECK_THROWS(ShearPrjGl{c});
  }

  // Everything else defaulted: construction succeeds, and the wall parses
  // into one grid point per zipped row.
  auto cfg_default = base_cfg();
  REQUIRE_NOTHROW(ShearPrjGl{cfg_default});
  auto const grid = ShearPrjGl::make_grid_points(cfg_default);
  REQUIRE(grid.size() == WALL_LAM.size());
  for (std::size_t i = 0; i != grid.size(); ++i) {
    CHECK(grid.points[i][0] == Approx(WALL_LAM[i]));
    CHECK(grid.points[i][1] == Approx(WALL_ZLO[i]));
    CHECK(grid.points[i][2] == Approx(WALL_ZHI[i]));
    CHECK(grid.points[i][3] == Approx(WALL_R[i]));
  }
}

TEST_CASE("ShearPrjGl DataBlock contract: every required sample section fails loudly")
{
  // Projection is the hungriest stage in the pipeline; this enumerates
  // exactly what it needs, so a pipeline missing a producer fails at
  // set_sample rather than publishing a quietly wrong g_t^prj.
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
    {"distances", "d_c"},
    {"haloModel", "lnM"},
    {"haloModel", "z"},
    {"haloModel", "bias"},
    {"haloModel", "rho_m_ref"},
    {"xi_nl", "r"},
    {"xi_nl", "z"},
    {"xi_nl", "xi_nl"},
    {"b_sel_marginalised", "lambda_bin"},
    {"b_sel_marginalised", "zo_low"},
    {"b_sel_marginalised", "zo_high"},
    {"b_sel_marginalised", "zob"},
    {"b_sel_marginalised", "lob"},
    {"b_sel_marginalised", "b_small"},
    {"b_sel_marginalised", "b_large"},
  };

  auto cfg = make_cfg_block();
  for (auto const& k : required) {
    auto s = make_sample_block_omitting(k);
    ShearPrjGl mod(cfg);
    INFO("omitted " << k.section << "/" << k.key);
    CHECK_THROWS(mod.set_sample(s));
  }

  // The complete block succeeds -- so the list above is the whole contract,
  // not an arbitrary subset.
  auto s_full = make_sample_block_omitting(DbKey{"none", "none"});
  ShearPrjGl mod(cfg);
  REQUIRE_NOTHROW(mod.set_sample(s_full));
  CHECK(std::isfinite(mod.evaluate(wall_points()[0])[0]));
  CHECK(mod.evaluate(wall_points()[0])[0] != 0.0);
}

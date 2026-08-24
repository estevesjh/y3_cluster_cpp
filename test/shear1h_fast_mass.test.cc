// Unit tests for the fast_mass shear C++ backend (des_y3 Phase 2):
// src/pipelines/des_y3/shear_1h2h/fast_mass/cpp/Shear1hFastMass.cc
//
// Shear1hFastMassCpp composes two immutable production pieces under a new
// module label / output section (shear1h_fast_mass/vals): the header-only
// y3_cluster::nosel_gl_detail::SelGLCore (models/n_operator_sel_gl_t.hh),
// which contracts the z integral on fixed GL nodes into a per-bin mass
// weight W_ij(lnM) = int dz n(M,z) dV/dOmega/dz(z) Omega(z)
// Sigma_crit_inv(z) S_ij(lnM,z); and the production miscentred NFW mixture
// (haloModel/dSigma_nfw Interp2D + NFW_DSIGMA_MIS's gamma-kernel table).
// Per the backend's README.md (validated 2026-08-12) it is bitwise
// identical to production Shear1hMisSel.so and 8.4e-4 vs the adaptive
// full_ltmz reference -- that fiducial-accuracy number comes from a live
// CosmoSIS run through src/pipelines/des_y3/validate_against_fiducial.py /
// the backend's validate_vs_production.py (see CLAUDE.md's fiducial-
// accuracy policy: only the adaptive full_ltmz reference counts as ground
// truth, and it is validated by those Python harnesses, not re-derived
// bit-for-bit in every Catch2 unit test).
//
// What this file checks instead, on a small self-contained synthetic
// DataBlock (two richness bins, two knots per table axis -- enough because
// every table below is either bilinear-exact, i.e. constant, or linear
// along the one axis the production formula actually varies, so GSL's
// bilinear/linear interpolators reproduce it exactly at any query in
// range; see make_sample_block):
//   1. Shear1hFastMassCpp::evaluate() against an independently assembled
//      reference (ReferenceModel::acc) that recombines the SAME
//      lower-level, separately-tested production pieces (HMF_t,
//      DV_DO_DZ_t, OMEGA_Z_DES, Sigma_crit_inv's Interp1D, SelFunction_t,
//      Interp2D, NFW_DSIGMA_MIS, the shared GL-node generator) per the
//      W_ij(lnM)/mixture formula documented at the top of
//      n_operator_sel_gl_t.hh -- this exercises the composition logic
//      that is actually new in Shear1hFastMass.cc.
//   2. Shear1hFastMassCpp against y3_cluster::Shear1hMisSelGL (the
//      production-equivalent header class, `method = exact`) on the
//      identical sample -- the unit-test-level analogue of the
//      "bitwise identical to production" claim, without a live pipeline
//      run.
//   3. The f_mis = 0 structural invariant: the centred-only limit must not
//      depend on the miscentering-kernel parameters (lob_centers,
//      tau_mis) at all, mirroring the exact-endpoint style of
//      test/shear1h_radial_series.test.cc.
//   4. The constructor/evaluate guard clauses (empty lob_centers,
//      out-of-range bin_index).
//
// ASSUMPTION flagged for the coordinator: Shear1hFastMassCpp is defined
// only in the .cc (no separate header), so it is included directly below
// -- there is no other test in this suite doing that, but the class has
// no include guard issue (included exactly once here) and its
// DEFINE_COSMOSIS_SCALAR_EVALUATOR_MODULE-generated extern "C"
// setup/execute/cleanup are inert, unexported symbols in this executable.
//
// Requires Y3_CLUSTER_CPP_DIR to point at the source tree: NFW_DSIGMA_MIS
// reads the real data/nfw_off_center/*gamma* tables at construction (same
// requirement as shear1h_radial_series.test.cc).
#include "catch2/catch.hpp"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include "models/dv_do_dz_t.hh"
#include "models/hmf_t.hh"
#include "models/nfw_dsigma_mis.hh"
#include "models/omega_z_des.hh"
#include "models/p_operator_cuhre_t.hh"
#include "models/sel_function_t.hh"
#include "modules/num_counts_sel/lensing_weights.hh"
#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"
#include "utils/make_interp_1d.hh"
#include "utils/make_interp_2d.hh"

// The header-only model machinery Shear1hFastMass.cc composes: SelGLCore
// (the fixed-GL W_ij(lnM) builder) and Shear1hMisSelGL (the production
// module this backend re-expresses, used below for the identity check).
#include "models/n_operator_sel_gl_t.hh"

// The system under test itself -- see the ASSUMPTION note above.
#include "pipelines/des_y3/shear_1h2h/fast_mass/cpp/shear1h_fast_mass_t.hh"

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  // A window narrow enough that every GL node it produces is strictly
  // interior to every synthetic table axis below (some of those axes,
  // e.g. distances/z via DV_DO_DZ_t, are queried WITHOUT clamping, so
  // going out of range would throw rather than silently misbehave).
  // ZT_HI is kept below 0.504 so the whole window stays on OMEGA_Z_DES's
  // single smooth SDSS_fit branch (avoiding the fit/fit2 seam at z=0.504,
  // which is irrelevant to what this file checks).
  double constexpr ZT_LO = 0.10;
  double constexpr ZT_HI = 0.50;
  double constexpr LNM_LO = 30.0;
  double constexpr LNM_HI = 36.0;
  int constexpr N_Z = 6;
  int constexpr N_LNM = 8;
  double constexpr F_MIS = 0.35;
  double constexpr TAU_MIS = 0.17;
  std::array<double, 3> const R_QUERY{0.5, 1.0, 2.0};

  // Options section shared by Shear1hFastMass (the system under test) and
  // Shear1hMisSel (the production-equivalent class, for the identity
  // check in the second TEST_CASE) -- same knobs under both module
  // labels, one DataBlock.
  cosmosis::DataBlock
  make_cfg_block(std::vector<double> const& lob_centers)
  {
    cosmosis::DataBlock cfg;
    for (char const* label : {"Shear1hFastMass", "Shear1hMisSel"}) {
      cfg.put_val(label, "n_lnm", N_LNM);
      cfg.put_val(label, "n_z", N_Z);
      cfg.put_val(label, "zt_low", ZT_LO);
      cfg.put_val(label, "zt_high", ZT_HI);
      cfg.put_val(label, "lnm_low", LNM_LO);
      cfg.put_val(label, "lnm_high", LNM_HI);
      cfg.put_val(label, "lob_centers", lob_centers);
    }
    return cfg;
  }

  // The per-sample DataBlock: two richness bins, two knots per table
  // axis.  dSigma_nfw is linear in r_sigma and constant in lnM (catches
  // an R/lnM argument-order bug while staying bilinear-exact at any R in
  // range); dndlnmh, d_a, sci_average and each bin's S_stack plane are
  // flat constants (bilinear/linear-exact everywhere by construction;
  // hmf_s = 0 / hmf_q = 1 -- matching the real fiducial point's
  // cluster_abundance values -- makes HMF_t collapse to the raw table
  // value with no further lnM-dependent correction).  S_stack is
  // distinct across the two bins (2.0 vs 3.5) so bin-indexing into both
  // the selection function and r_mis_[b] is actually exercised.
  cosmosis::DataBlock
  make_sample_block(double f_mis, double tau_mis)
  {
    cosmosis::DataBlock s;

    // cosmological_parameters: "omega_m" backs both HMF_t/NFW_DSIGMA_MIS's
    // omega_M and EZ's omega_m -- DataBlock names are case-insensitive.
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
                                        {2, 2})); // (n_z, n_m), constant

    s.put_val("distances", "z", std::vector<double>{0.0, 1.0});
    s.put_val("distances", "d_a", std::vector<double>{1500.0, 1500.0});

    s.put_val("average_sigma_crit_inv", "zlense", std::vector<double>{0.0, 1.0});
    s.put_val("average_sigma_crit_inv", "sci_average",
              std::vector<double>{3.0e-4, 3.0e-4});

    s.put_val("sel_function", "lnM", std::vector<double>{LNM_LO, LNM_HI});
    s.put_val("sel_function", "z", std::vector<double>{ZT_LO, ZT_HI});
    std::vector<double> s_stack;
    for (double c : {2.0, 3.5})
      for (int i = 0; i != 4; ++i) s_stack.push_back(c);
    s.put_val("sel_function", "S_stack",
              cosmosis::ndarray<double>(s_stack, {2, 2, 2})); // (bin,z,lnm)

    s.put_val("haloModel", "r_sigma", std::vector<double>{0.05, 10.0});
    s.put_val("haloModel", "lnM", std::vector<double>{28.0, 37.0});
    auto const dsig = [](double r) { return 5.0 - 0.3 * r; };
    s.put_val("haloModel", "dSigma_nfw",
              cosmosis::ndarray<double>(
                std::vector<double>{dsig(0.05), dsig(10.0), dsig(0.05),
                                    dsig(10.0)},
                {2, 2}));

    s.put_val("miscentering", "f_mis", f_mis);
    s.put_val("miscentering", "tau_mis", tau_mis);
    return s;
  }

  // Recombines the SAME lower-level production pieces SelGLCore and
  // Shear1hFastMassCpp compose, per the W_ij(lnM)/mixture formula
  // documented at the top of n_operator_sel_gl_t.hh -- an independent
  // check of the composition, not a re-derivation of any one piece's own
  // physics (each of those already has its own dedicated unit test).
  struct ReferenceModel {
    y3_cluster::HMF_t hmf;
    y3_cluster::DV_DO_DZ_t dv;
    y3_cluster::OMEGA_Z_DES omega;
    y3_cluster::Interp1D sci;
    y3_cluster::Interp2D dsigma_nfw;
    y3_cluster::NFW_DSIGMA_MIS dsigma_mis;
    std::vector<double> z_x, z_w, lnm_x, lnm_w;

    explicit ReferenceModel(cosmosis::DataBlock& s)
      : hmf(s)
      , dv(s)
      , omega(s)
      , sci(y3_cluster_sel_weights::load_sigma_crit_inv(s))
      , dsigma_nfw(y3_cluster::make_Interp2D(s, "haloModel", "r_sigma",
                                             "lnM", "dSigma_nfw"))
      , dsigma_mis(4.0, 2.77533742639e+11, y3_cluster::GAMMA)
    {
      dsigma_mis.set_rho_mult(s.view<double>("cosmological_parameters",
                                             "omega_m"));
      y3_cluster::p_op_detail::gl_nodes(ZT_LO, ZT_HI, N_Z, z_x, z_w);
      y3_cluster::p_op_detail::gl_nodes(LNM_LO, LNM_HI, N_LNM, lnm_x, lnm_w);
    }

    double
    acc(cosmosis::DataBlock& s, int bin, double R, double f_mis,
        double r_mis) const
    {
      y3_cluster::SelFunction_t const sel(s, bin);
      double total = 0.0;
      for (std::size_t k = 0; k != lnm_x.size(); ++k) {
        double const lnM = lnm_x[k];
        double Wk = 0.0;
        for (std::size_t q = 0; q != z_x.size(); ++q) {
          double const z = z_x[q];
          Wk += z_w[q] * dv(z) * omega(z) * sci.clamp(z) * hmf(lnM, z) *
                sel(lnM, z);
        }
        double const d_cen = dsigma_nfw.clamp(R, lnM);
        double const d_mis = dsigma_mis(R, r_mis, lnM);
        total += lnm_w[k] * Wk * ((1.0 - f_mis) * d_cen + f_mis * d_mis);
      }
      return total;
    }
  };

} // namespace

TEST_CASE("Shear1hFastMass matches an independently assembled W_ij x mixture reference")
{
  auto cfg = make_cfg_block(std::vector<double>{30.0, 90.0});
  auto s = make_sample_block(F_MIS, TAU_MIS);

  Shear1hFastMassCpp mod(cfg);
  mod.set_sample(s);

  ReferenceModel const ref(s);
  double const lob_centers[2] = {30.0, 90.0};

  for (int b : {0, 1}) {
    double const r_mis =
      TAU_MIS * y3_cluster_sel_weights::mis_detail::R_lambda(lob_centers[b]);
    for (double R : R_QUERY) {
      auto const out = mod.evaluate({static_cast<double>(b), R});
      double const expected = ref.acc(s, b, R, F_MIS, r_mis);
      CHECK(out[0] == Approx(expected).epsilon(1e-9));
    }
  }
}

TEST_CASE("Shear1hFastMass agrees with the production-equivalent Shear1hMisSelGL")
{
  // Same claim as the README's "bitwise identical to production
  // Shear1hMisSel.so", exercised here without a live CosmoSIS run:
  // Shear1hMisSelGL (method = exact, the default) IS the production
  // algorithm this backend re-expresses under a new label/section.
  auto cfg = make_cfg_block(std::vector<double>{30.0, 90.0});
  auto s = make_sample_block(F_MIS, TAU_MIS);

  Shear1hFastMassCpp fast(cfg);
  fast.set_sample(s);
  y3_cluster::Shear1hMisSelGL prod(cfg);
  prod.set_sample(s);

  for (int b : {0, 1}) {
    for (double R : R_QUERY) {
      auto const a = fast.evaluate({static_cast<double>(b), R});
      auto const c = prod.evaluate({static_cast<double>(b), R});
      CHECK(a[0] == Approx(c[0]).epsilon(1e-9));
    }
  }
}

TEST_CASE("Shear1hFastMass at f_mis = 0 is invariant to the miscentering-kernel parameters")
{
  // At f_mis = 0 the mixture is pure dSigma_nfw; lob_centers and tau_mis
  // only ever enter through the f_mis-weighted miscentred term, so two
  // very different miscentering configurations must give bit-identical
  // output -- the fast_mass analogue of shear1h_radial_series.test.cc's
  // "mixture decomposition has exact endpoints".
  auto cfg_a = make_cfg_block(std::vector<double>{30.0, 90.0});
  auto cfg_b = make_cfg_block(std::vector<double>{5.0, 500.0});
  auto s = make_sample_block(/*f_mis=*/0.0, /*tau_mis=*/0.17);
  auto s_other_tau = make_sample_block(/*f_mis=*/0.0, /*tau_mis=*/0.9);

  Shear1hFastMassCpp mod_a(cfg_a);
  mod_a.set_sample(s);
  Shear1hFastMassCpp mod_b(cfg_b);
  mod_b.set_sample(s_other_tau);

  for (int b : {0, 1}) {
    for (double R : R_QUERY) {
      auto const va = mod_a.evaluate({static_cast<double>(b), R});
      auto const vb = mod_b.evaluate({static_cast<double>(b), R});
      CHECK(va[0] == Approx(vb[0]).epsilon(1e-12));
      CHECK(va[0] > 0.0); // sanity: not silently zero
    }
  }
}

TEST_CASE("Shear1hFastMass guard clauses: empty lob_centers, out-of-range bin_index")
{
  cosmosis::DataBlock cfg_empty;
  cfg_empty.put_val("Shear1hFastMass", "n_lnm", N_LNM);
  cfg_empty.put_val("Shear1hFastMass", "n_z", N_Z);
  cfg_empty.put_val("Shear1hFastMass", "zt_low", ZT_LO);
  cfg_empty.put_val("Shear1hFastMass", "zt_high", ZT_HI);
  cfg_empty.put_val("Shear1hFastMass", "lnm_low", LNM_LO);
  cfg_empty.put_val("Shear1hFastMass", "lnm_high", LNM_HI);
  cfg_empty.put_val("Shear1hFastMass", "lob_centers", std::vector<double>{});
  CHECK_THROWS_AS(Shear1hFastMassCpp(cfg_empty), std::runtime_error);

  auto cfg = make_cfg_block(std::vector<double>{30.0, 90.0});
  auto s = make_sample_block(F_MIS, TAU_MIS);
  Shear1hFastMassCpp mod(cfg);
  mod.set_sample(s);
  CHECK_THROWS_AS(mod.evaluate({2.0, 0.5}), std::out_of_range);
  CHECK_THROWS_AS(mod.evaluate({-1.0, 0.5}), std::out_of_range);
}

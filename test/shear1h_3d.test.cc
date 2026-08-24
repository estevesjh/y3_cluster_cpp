// Unit tests for the full_ltmz shear C++ reference backend
// (src/pipelines/des_y3/shear_1h2h/cpp/3d,
// des_y3 Phase 2): one adaptive-Cuhre triple integral per (bin, R) over
// the explicit (lambda_true, lnM, z) integrand, production miscentred
// mixture profile.
//
// Shear1h3d has no committed numeric fiducial corpus to pin
// against: its own validate_vs_production.py compares against a
// real-pipeline dump (cosmosis-models/real_pipeline_extract_output) that is
// gitignored and not present in this tree, and CLAUDE.md's fiducial
// accuracy policy quotes accuracy only against the adaptive full_ltmz
// Python reference (also driven from that same dump) -- neither is
// reproducible offline without a real cosmosis run.
//
// So instead of pinning a number from that script, this file builds one
// small, fully self-contained DataBlock (synthetic tables/parameters,
// but the real production sub-model types: HMF_t, DV_DO_DZ_t,
// OMEGA_Z_DES, MOR_HOD_t, PlobLtrEMG_t + RichnessKernel_t, Interp1D
// sigma_crit_inv, Interp2D dSigma_nfw, NFW_DSIGMA_MIS) covering every
// section Shear1h3d::set_sample reads, and checks operator()
// against those same sub-models assembled by hand per the
// header-documented O_ij(R) integrand. That is an algorithm-identity
// check in CLAUDE.md's sense: it pins that Shear1h3d composes the
// documented terms correctly (right factors, right order, right lnM/R
// argument convention), not an absolute physics number. The
// miscentering-mixture-affinity and monotonic-radial-falloff checks
// below are purely structural and hold regardless of which synthetic
// table values are chosen.
//
// Requires Y3_CLUSTER_CPP_DIR: NFW_DSIGMA_MIS reads data/nfw_off_center/*
// at construction, same as shear1h_radial_series.test.cc.
#include "catch2/catch.hpp"

#include "pipelines/des_y3/shear_1h2h/cpp/3d/shear1h_3d_t.hh"

#include "models/dv_do_dz_t.hh"
#include "models/hmf_t.hh"
#include "models/mor_hod_t.hh"
#include "models/nfw_dsigma_mis.hh"
#include "models/omega_z_des.hh"
#include "models/plob_ltr_emg_t.hh"
#include "models/richness_kernel_t.hh"
#include "modules/num_counts_sel/lensing_weights.hh"
#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"
#include "utils/make_interp_1d.hh"
#include "utils/make_interp_2d.hh"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace {
  // Single richness/redshift bin; fiducial evaluation point.
  constexpr double LAM_MIN = 20.0, LAM_MAX = 30.0;
  constexpr double ZOB_MIN = 0.20, ZOB_MAX = 0.35, SIGMA_Z = 0.03;
  constexpr double LT_Q = 25.0;              // lambda_true query (bin centre)
  double const ZT_Q = 0.275;                 // z query (bin centre)
  double const LNM_Q = std::log(1.0e14);     // raw ln(M) query, M = 1e14 Msun/h
  constexpr double R_Q = 1.0;                // radius query (exact table node)

  // Populate every DataBlock section Shear1h3d's constructor and
  // set_sample read. cfg doubles as both the "options" and "sample"
  // block, which the class is happy to accept.
  void
  populate(cosmosis::DataBlock& cfg, double f_mis, double tau_mis)
  {
    std::string const label = "Shear1h3d";
    cfg.put_val(label, "lam_min", std::vector<double>{LAM_MIN});
    cfg.put_val(label, "lam_max", std::vector<double>{LAM_MAX});
    cfg.put_val(label, "zob_min", std::vector<double>{ZOB_MIN});
    cfg.put_val(label, "zob_max", std::vector<double>{ZOB_MAX});
    cfg.put_val(label, "sigma_z", std::vector<double>{SIGMA_Z});
    // lob_centers intentionally omitted: falls back to the DES-Y3
    // default {25, 37.5, 52.5, 130}; bin 0 -> 25.0, matching the R_lambda
    // calls below.

    cfg.put_val("cosmological_parameters", "omega_m", 0.3);   // EZ (lowercase key)
    cfg.put_val("cosmological_parameters", "omega_M", 0.3);   // HMF_t / rho_mult (uppercase key)
    cfg.put_val("cosmological_parameters", "omega_lambda", 0.7);
    cfg.put_val("cosmological_parameters", "omega_k", 0.0);
    cfg.put_val("cosmological_parameters", "omega_nu", 0.0);
    cfg.put_val("cosmological_parameters", "h0", 0.7);

    cfg.put_val("cluster_abundance", "hmf_s", 0.02);
    cfg.put_val("cluster_abundance", "hmf_q", 1.1);

    // mass_function: 3x3 table. LNM_Q lands strictly between the raw
    // m_h=1e14 and m_h=1e15 nodes after the (Omega_m - Omega_nu) shift
    // HMF_t applies internally -- a genuine bilinear interpolation, not
    // an exact-node lookup.
    cfg.put_val("mass_function", "m_h", std::vector<double>{1e13, 1e14, 1e15});
    cfg.put_val("mass_function", "z", std::vector<double>{0.05, 0.4, 0.9});
    std::vector<double> const dndlnmh{
      1e-3, 1e-5, 1e-8,    // z=0.05
      8e-4, 7e-6, 5e-9,    // z=0.4
      5e-4, 3e-6, 1e-9};   // z=0.9
    std::vector<std::size_t> const dndlnmh_extents{3, 3};
    cfg.put_val<cosmosis::ndarray<double>>(
      "mass_function", "dndlnmh", {dndlnmh, dndlnmh_extents});

    cfg.put_val("distances", "z", std::vector<double>{0.0, 0.5, 1.0});
    cfg.put_val("distances", "d_a", std::vector<double>{50.0, 700.0, 1200.0});

    cfg.put_val("average_sigma_crit_inv", "zlense",
               std::vector<double>{0.0, 0.5, 1.0});
    cfg.put_val("average_sigma_crit_inv", "sci_average",
               std::vector<double>{1.5e-4, 2.0e-4, 2.5e-4});

    // haloModel dSigma_nfw: 5 radii x 2 masses, strictly decreasing in R
    // at both mass nodes (and therefore at any interpolated lnM in
    // between) -- the table the monotonic-radial-falloff test below
    // relies on.
    cfg.put_val("haloModel", "r_sigma",
               std::vector<double>{0.2, 0.5, 1.0, 2.0, 5.0});
    cfg.put_val("haloModel", "lnM", std::vector<double>{30.0, 34.0});
    std::vector<double> const dsigma_nfw{
      40.0, 18.0, 9.0, 4.0, 1.5,     // lnM=30
      55.0, 24.0, 12.0, 5.5, 2.0};   // lnM=34
    std::vector<std::size_t> const dsigma_nfw_extents{2, 5};
    cfg.put_val<cosmosis::ndarray<double>>(
      "haloModel", "dSigma_nfw", {dsigma_nfw, dsigma_nfw_extents});

    cfg.put_val("cluster_mor", "log10_Mmin", 13.5);
    cfg.put_val("cluster_mor", "log10_ratio", 1.0);   // log10_M1 = 14.5
    cfg.put_val("cluster_mor", "alpha", 1.1);
    cfg.put_val("cluster_mor", "epsilon", 0.3);
    cfg.put_val("cluster_mor", "sigma_lambda", 0.3);

    cfg.put_val("plob_ltr_params", "z", std::vector<double>{0.1, 0.4, 0.8});
    cfg.put_val("plob_ltr_params", "a_mu", std::vector<double>{0.05, 0.02, 0.0});
    cfg.put_val("plob_ltr_params", "b_mu", std::vector<double>{0.95, 1.0, 1.05});
    cfg.put_val("plob_ltr_params", "a_sig", std::vector<double>{0.4, 0.45, 0.5});
    cfg.put_val("plob_ltr_params", "b_sig", std::vector<double>{0.15, 0.16, 0.18});
    cfg.put_val("plob_ltr_params", "a_tau", std::vector<double>{0.5, 0.55, 0.6});
    cfg.put_val("plob_ltr_params", "b_tau", std::vector<double>{0.08, 0.09, 0.10});
    cfg.put_val("plob_ltr_params", "a_fprj", std::vector<double>{1.0, 1.0, 1.0});
    cfg.put_val("plob_ltr_params", "b_fprj", std::vector<double>{0.25, 0.3, 0.35});

    cfg.put_val("miscentering", "f_mis", f_mis);
    cfg.put_val("miscentering", "tau_mis", tau_mis);
  }
}

TEST_CASE("Shear1h3d reproduces the documented O_ij integrand from its own production sub-models")
{
  cosmosis::DataBlock cfg;
  double const f_mis = 0.22, tau_mis = 0.17;
  populate(cfg, f_mis, tau_mis);

  Shear1h3d model(cfg);
  model.set_sample(cfg);
  model.set_grid_point({0.0, R_Q});

  // Independently assemble the same terms Shear1h3d::operator()
  // multiplies, from the same DataBlock, via freshly constructed
  // production sub-model objects -- not by calling into the class under
  // test. This catches a wrong composition order, a missing factor, or a
  // wrong lnM/R argument convention without duplicating any physics.
  y3_cluster::HMF_t hmf(cfg);
  y3_cluster::DV_DO_DZ_t dv(cfg);
  y3_cluster::OMEGA_Z_DES omega(cfg);
  y3_cluster::Interp1D sci = y3_cluster::make_Interp1D(
    cfg, "average_sigma_crit_inv", "zlense", "sci_average");
  y3_cluster::RichnessKernel_t k_i(LAM_MIN, LAM_MAX);
  y3_cluster::PlobLtrEMG_t plob(cfg);
  y3_cluster::MOR_HOD_t mor(cfg);
  y3_cluster::Interp2D dsigma_nfw = y3_cluster::make_Interp2D(
    cfg, "haloModel", "r_sigma", "lnM", "dSigma_nfw");
  y3_cluster::NFW_DSIGMA_MIS dsigma_mis(4.0, 2.77533742639e+11, y3_cluster::GAMMA);
  dsigma_mis.set_rho_mult(cfg.view<double>("cosmological_parameters", "omega_M"));

  double const r_mis =
    tau_mis * y3_cluster_sel_weights::mis_detail::R_lambda(25.0);
  double const k_j = y3_cluster::richness_zkernel(ZT_Q, ZOB_MIN, ZOB_MAX, SIGMA_Z);
  double const phi = (1.0 - f_mis) * dsigma_nfw.clamp(R_Q, LNM_Q) +
                     f_mis * dsigma_mis(R_Q, r_mis, LNM_Q);
  double const expected =
    hmf(LNM_Q, ZT_Q) * dv(ZT_Q) * omega(ZT_Q) * sci.clamp(ZT_Q) * k_j *
    k_i(LT_Q, ZT_Q, plob) * mor(LT_Q, LNM_Q, ZT_Q) * phi;

  CHECK(expected > 0.0);
  CHECK(model(LT_Q, ZT_Q, LNM_Q) == Approx(expected).epsilon(1e-3));
}

TEST_CASE("Shear1h3d miscentering mixture is affine in f_mis, anchored to the production dSigma_nfw/NFW_DSIGMA_MIS readers")
{
  cosmosis::DataBlock cfg0;
  populate(cfg0, /*f_mis=*/0.0, /*tau_mis=*/0.17);
  Shear1h3d centred(cfg0);
  centred.set_sample(cfg0);
  centred.set_grid_point({0.0, R_Q});
  double const val0 = centred(LT_Q, ZT_Q, LNM_Q);

  double const f_mis_F = 0.35, tau_mis_F = 0.20;
  cosmosis::DataBlock cfgF;
  populate(cfgF, f_mis_F, tau_mis_F);
  Shear1h3d mixed(cfgF);
  mixed.set_sample(cfgF);
  mixed.set_grid_point({0.0, R_Q});
  double const valF = mixed(LT_Q, ZT_Q, LNM_Q);

  y3_cluster::Interp2D dsigma_nfw = y3_cluster::make_Interp2D(
    cfg0, "haloModel", "r_sigma", "lnM", "dSigma_nfw");
  y3_cluster::NFW_DSIGMA_MIS dsigma_mis(4.0, 2.77533742639e+11, y3_cluster::GAMMA);
  dsigma_mis.set_rho_mult(cfg0.view<double>("cosmological_parameters", "omega_M"));
  double const r_mis_F =
    tau_mis_F * y3_cluster_sel_weights::mis_detail::R_lambda(25.0);
  double const d_cen = dsigma_nfw.clamp(R_Q, LNM_Q);
  double const d_mis = dsigma_mis(R_Q, r_mis_F, LNM_Q);

  // Every other factor in operator() (hmf, dv, omega, sci, k_j, k_i, mor)
  // is f_mis-independent, so it cancels in the ratio: this identity must
  // hold regardless of what those factors evaluate to.
  double const expected_ratio = (1.0 - f_mis_F) + f_mis_F * (d_mis / d_cen);
  CHECK(valF / val0 == Approx(expected_ratio).epsilon(1e-9));
}

TEST_CASE("Shear1h3d falls off monotonically with radius")
{
  cosmosis::DataBlock cfg;
  populate(cfg, /*f_mis=*/0.0, /*tau_mis=*/0.17); // isolate the centred NFW profile
  Shear1h3d model(cfg);
  model.set_sample(cfg);

  std::vector<double> const radii{0.2, 0.5, 1.0, 2.0, 5.0};
  double previous = std::numeric_limits<double>::max();
  for (double const R : radii) {
    model.set_grid_point({0.0, R});
    double const val = model(LT_Q, ZT_Q, LNM_Q);
    CHECK(val > 0.0);
    CHECK(val < previous);
    previous = val;
  }
}

// Projected tangential shear integrand gamma_t^prj (Costanzi-2026 Eq. Sprj),
// exposed through the CosmoSIS shear_prj scalar integrand.
//
// This model first integrates the projected excess surface density
// DeltaSigma_prj from neighbouring, miscentred haloes and then converts it to
// shear with
//
//   gamma_t^prj(R) = DeltaSigma_prj(R) * Sigma_crit^{-1}(z_ob).
//
// Sigma_crit^{-1} is read from average_sigma_crit_inv and evaluated at z_ob;
// it is not set to unity.  The legacy SigmaPrj and DSigmaPrj wrappers remain
// as compatibility/diagnostic outputs, but the physical observable is
// shear_prj.
//
// The numerical implementation mirrors the richness-selection projection
// model, with PAGANI driving the 3-D outer (z, lnM, theta) integral.  It reads
// b_sel_marginalised from the datablock (written by
// systematics/selection_bias/python/bsel.py). Each
// wall row supplies b_small and b_large; the theta dependence is evaluated
// from the shared sigmoid at the exact (lambda_bin, z_ob) key.
//
// The integrand body:
//
//   I_DeltaSigma(z, lnM, theta | lob_i, zob_j, R_k)
//     = dV/dzdOmega(z) * n(M, z) * [1 + b(M, z) * b_sel(theta) * xi_NL(Delta_chi)]
//       * 2 pi sin(theta) * Sigma_mis(R_k, theta * D_A(zob), M, z)
//       * w_z(z, zob) * (indicator: exclusion Delta_chi >= R_excl)
//
// Wall grid: (lob_bin, zob_low, zob_high, radii) zipped.  Same
// gt_mock_cpu/avgCentFull.cc shape, with lob_bin now a grid axis too.
// Output vals has shape (N_lob * N_zob * N_R,) -- a 1-D wall grid,
// reshaped by the downstream consumer.
#ifndef Y3_CLUSTER_CPP_SIGMA_PRJ_T_HH
#define Y3_CLUSTER_CPP_SIGMA_PRJ_T_HH

#include "utils/make_grid_points.hh"
#include "utils/make_integration_volumes.hh"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"
#include "cubacpp/cubacpp.hh"
#include "cubacpp/integration_volume.hh"

#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"
#include "utils/make_interp_1d.hh"
#include "utils/make_interp_2d.hh"
#include "models/z_kernel_data.hh"

#include "models/dv_do_dz_t.hh"
#include "models/hmf_t.hh"
#include "models/nfw_dsigma_mis.hh"
#include "models/nfw_sigma_mis.hh"
#include "models/omega_z_des.hh"
#include "pipelines/systematics/selection_bias/cpp/bsel_bins_t.hh"

#include "utils/datablock_reader.hh"

// For p_op_detail::gl_nodes (shared with P_operator).
#include "models/p_operator_cuhre_t.hh"

#include <gsl/gsl_integration.h>
#include <gsl/gsl_math.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace y3_cluster {

  namespace sp_detail {
    inline constexpr double PI = 3.14159265358979323846;

    inline double R_lambda(double lam) { return std::pow(lam / 100.0, 0.2); }

    // Default DES-Y3 richness-bin centres used by the sigma_prj module
    // when no per-bin "lob_centers" vector is provided in the ini.  The
    // edges are [20, 30, 45, 60, 200]; arithmetic centres are
    // {25, 37.5, 52.5, 130}.  Callers should prefer the ini-configured
    // vector (read once in the module ctor) over this fallback.
    inline std::array<double, 4> const&
    default_lob_centers()
    {
      static constexpr std::array<double, 4> centres{{25.0, 37.5, 52.5, 130.0}};
      return centres;
    }

    // Result of build_theta_grid: log-GL nodes on segments split at
    // feature breakpoints (peaks / bends of the Sigma_prj theta integrand).
    // Mirrors richness_selection/sigma_prj.py::_theta_grid (Eq. 5 in
    // RichnessSelection docs/sigma_prj_refactor.md).
    struct ThetaGrid {
      std::vector<double> theta;        // node positions
      std::vector<double> weight;       // GL weight with dtheta = theta du Jacobian folded in
      std::vector<double> breakpoints;  // for diagnostics / ini echo
    };

    // Build a log-GL theta grid on segments split at
    //   { lower, theta_excl,o, theta_R(R_i) for every R_i in R_vec,
    //     theta_lam, 2*theta_lam, theta_max } U extra_breakpoints
    // with n_per_seg nodes per segment.  The per-R breakpoint is
    // load-bearing: each requested R lands a node at its Sigma_mis peak
    // theta_R = R / D_A(zob).
    inline ThetaGrid
    build_theta_grid(double lob_center_val, double zob,
                     std::vector<double> const& R_vec,
                     double chi_o, double D_A_o, double R_excl,
                     std::size_t n_per_seg,
                     double R_max_cMpch,
                     std::vector<double> const& extra_breakpoints)
    {
      double const theta_lam    = R_lambda(lob_center_val)
                                  * (1.0 + zob) / chi_o;
      double const theta_excl_o = R_excl / chi_o;

      std::vector<double> theta_R_arr;
      theta_R_arr.reserve(R_vec.size());
      for (double const R : R_vec) theta_R_arr.push_back(R / D_A_o);
      double const theta_R_min = theta_R_arr.empty()
                                 ? theta_lam
                                 : *std::min_element(theta_R_arr.begin(),
                                                     theta_R_arr.end());
      double const theta_R_max = theta_R_arr.empty()
                                 ? theta_lam
                                 : *std::max_element(theta_R_arr.begin(),
                                                     theta_R_arr.end());

      double const theta_max = std::max(R_max_cMpch / D_A_o,
                                        3.0 * theta_R_max);
      double const lower = std::max(1.0e-8,
                                    0.1 * std::min({theta_excl_o,
                                                    theta_R_min,
                                                    theta_lam}));

      std::vector<double> bp;
      bp.reserve(R_vec.size() + extra_breakpoints.size() + 6);
      bp.push_back(lower);
      bp.push_back(theta_excl_o);
      for (double const t : theta_R_arr) bp.push_back(t);
      bp.push_back(theta_lam);
      bp.push_back(2.0 * theta_lam);
      bp.push_back(theta_max);
      for (double const t : extra_breakpoints) bp.push_back(t);

      // Keep only finite, positive, in-range entries.
      std::vector<double> bp_clean;
      bp_clean.reserve(bp.size());
      for (double const t : bp) {
        if (std::isfinite(t) && t > 0.0 && t <= theta_max) bp_clean.push_back(t);
      }
      std::sort(bp_clean.begin(), bp_clean.end());

      // Force the endpoints into the list.
      if (bp_clean.empty() || bp_clean.front() > lower)
        bp_clean.insert(bp_clean.begin(), lower);
      if (bp_clean.back() < theta_max) bp_clean.push_back(theta_max);

      // Dedupe close-by breakpoints (relative gap < 1e-6).
      std::vector<double> bp_dedup;
      bp_dedup.reserve(bp_clean.size());
      bp_dedup.push_back(bp_clean.front());
      for (std::size_t i = 1; i != bp_clean.size(); ++i) {
        if (bp_clean[i] > bp_dedup.back() * (1.0 + 1.0e-6))
          bp_dedup.push_back(bp_clean[i]);
      }

      // Log-GL per segment; fold dtheta = theta du Jacobian into weight.
      ThetaGrid tg;
      tg.breakpoints = bp_dedup;
      std::size_t const Nseg = bp_dedup.size() - 1;
      tg.theta .reserve(Nseg * n_per_seg);
      tg.weight.reserve(Nseg * n_per_seg);
      std::vector<double> u_x, u_w;
      for (std::size_t s = 0; s != Nseg; ++s) {
        double const a   = bp_dedup[s];
        double const b   = bp_dedup[s + 1];
        double const lna = std::log(a);
        double const lnb = std::log(b);
        p_op_detail::gl_nodes(lna, lnb, n_per_seg, u_x, u_w);
        for (std::size_t k = 0; k != n_per_seg; ++k) {
          double const th = std::exp(u_x[k]);
          tg.theta .push_back(th);
          tg.weight.push_back(u_w[k] * th);   // dtheta = theta d(ln theta)
        }
      }
      return tg;
    }

    // Per-z LoS-slab exclusion angle.
    //   cos theta_excl(z) = (chi(z)^2 + chi_o^2 - R_excl^2) / (2 chi_z chi_o)
    // Matches SelBias._P_operator (richness_selection/sel_bias.py:234-254
    // and sigma_prj.py:131-138).  Returns 0 when the 1-halo radius does
    // not reach z from zob.
    inline double
    theta_excl_at_z(double chi_z, double chi_o, double R_excl)
    {
      double const denom = 2.0 * chi_z * chi_o + 1.0e-30;
      double cos_ex = (chi_z * chi_z + chi_o * chi_o - R_excl * R_excl)
                      / denom;
      if (cos_ex >= 1.0 - 1.0e-12) return 0.0;
      if (cos_ex < -1.0) cos_ex = -1.0;
      return std::acos(cos_ex);
    }
  }

  // Retired classes: Sigma_prj_integrand / Sigma_prj_integrand_cuhre /
  // DSigma_prj_integrand (3-D Cuhre adaptive, deprecated) and
  // Sigma_prj_evaluator / DSigma_prj_evaluator (single-observable
  // fixed-GL variants superseded by the merged ShearPrjEvaluator
  // below).  Removed 2026-05-06; the new fixed-GL and Cuhre paths
  // publish {vals, rnd, cl} per observable via
  // CosmoSISScalarEvaluatorModule + output_names().


  // =====================================================================
  // sp_detail::ShearPrjCore -- shared cache + query backbone for the
  // fixed-GL projection evaluators.  Owns every (lob, zob) slice cache
  // and the per-R Smis/DSmis tables.  Three thin wrappers below expose
  // it as separate CosmoSIS modules:
  //
  //   SigmaPrjEvaluator   -> {Sigma_total,  Sigma_rnd,  Sigma_cl}
  //   DSigmaPrjEvaluator  -> {DSigma_total, DSigma_rnd, DSigma_cl}
  //   ShearPrjEvaluator   -> {gt_total,     gt_rnd,     gt_cl}
  //
  // Recipe (unchanged from the retired single-class ShearPrjEvaluator):
  //   * THETA is the outer integral; log-GL on segments split at sorted
  //     breakpoints { lower, theta_excl_o, theta_R(R_i) for each R_i,
  //     theta_lam, 2*theta_lam, theta_max }.  Shared per (lob, zob)
  //     slice; every R on that slice lands its own theta_R breakpoint.
  //   * Inner (z, lnM): z via the adaptive ring + fg/bg log-|Dchi| grid
  //     (build_z_grid_, matches SelBias._z_grid); lnM via fixed GL.
  //   * Exclusion: per-z LoS slab  theta > theta_excl(z).
  //   * rnd / cl split:
  //       X_rnd = 2pi int dtheta sin(theta) int dz dV w_z(z)
  //               int dM n(M,z) X_mis(R|M,z,theta D_A_o)
  //       X_cl  = 2pi int dtheta sin(theta) b_sel(theta)
  //               int dz dV w_z(z) xi_NL(|Dr|, zob) 1[theta>theta_excl(z)]
  //               int dM n(M,z) b(M,z) X_mis(R|M,z,theta D_A_o)
  //     where X is Sigma_mis or DSigma_mis.  gamma_t = DSigma * Sigma_crit_inv
  //     (linear; reduced-shear denominator retired 2026-05-11).
  // =====================================================================
  namespace sp_detail {

  class ShearPrjCore {
   public:
    using grid_t       = y3_cluster::grid_t<4>;
    using grid_point_t = grid_t::value_type;

    explicit ShearPrjCore(cosmosis::DataBlock& cfg, char const* section)
      : section_(section)
      , N_lnm_(cfg.has_val(section, "n_lnm")
                 ? cfg.view<int>(section, "n_lnm") : 24)
      , N_per_seg_(cfg.has_val(section, "n_per_seg")
                     ? cfg.view<int>(section, "n_per_seg") : 30)
      , N_zring_(cfg.has_val(section, "n_zring")
                   ? cfg.view<int>(section, "n_zring") : 20)
      , N_zouter_(cfg.has_val(section, "n_zouter")
                    ? cfg.view<int>(section, "n_zouter") : 20)
      , zt_lo_(cfg.view<double>(section, "zt_low"))
      , zt_hi_(cfg.view<double>(section, "zt_high"))
      , lnm_lo_(cfg.view<double>(section, "lnm_low"))
      , lnm_hi_(cfg.view<double>(section, "lnm_high"))
      , R_max_cMpch_(cfg.has_val(section, "R_max_cMpch")
                       ? cfg.view<double>(section, "R_max_cMpch")
                       : 30.0)   // Costanzi-2026 convention: theta_max = R_max/D_A
    {
      p_op_detail::gl_nodes(lnm_lo_, lnm_hi_, N_lnm_, lnm_x_, lnm_w_);

      // Optional user-supplied extra breakpoints (radians).  Passed
      // through to sp_detail::build_theta_grid; default empty.
      if (cfg.has_val(section, "theta_breakpoints")) {
        extra_breakpoints_ = get_vector_double(cfg, section,
                                               "theta_breakpoints");
      }

      // Per-bin richness "centres" used for R_excl and theta_lambda.
      // Default: DES-Y3 arithmetic centres {25, 37.5, 52.5, 130}
      // (edges [20, 30, 45, 60, 200]).  Override via the ini
      //   lob_centers = 25 37.5 52.5 130
      if (cfg.has_val(section, "lob_centers")) {
        lob_centers_ = get_vector_double(cfg, section,
                                         "lob_centers");
      } else {
        auto const& dflt = sp_detail::default_lob_centers();
        lob_centers_.assign(dflt.begin(), dflt.end());
      }

      // Build the NFW miscentering lookup tables once at ctor time.
      // Each default ctor reads 3 ASCII 1000×1000 tables from disk
      // (~0.3s each). These depend only on the lookup files, not on
      // cosmology, so rebuilding them every set_sample() was pure
      // overhead.
      //
      // For projection (Sigma_prj / DSigma_prj) we use the SINGLE
      // miscentering kernel (delta-function at R_mis) matching the
      // Python reference richness_selection.nfw.NFWMiscentered.  The
      // gamma kernel (Kelly et al. 2023 distribution) is for the DES
      // lensing pipeline, not for projection.
      sigma_mis_.emplace(4.0, 2.77533742639e+11, NFW_SIG_SINGLE);
      dsigma_mis_.emplace(4.0, 2.77533742639e+11, SINGLE);

      // Parse the wall-grid axes so we can pre-plan sample-level caches
      // keyed on unique (lam_bin, zob, R) tuples.  get_vector_double
      // coerces int-arrays (lambda_bin) to double on the fly.
      auto const lamb = get_vector_double(cfg, section, "lambda_bin");
      auto const zlo  = get_vector_double(cfg, section, "zo_low");
      auto const zhi  = get_vector_double(cfg, section, "zo_high");
      auto const rad  = get_vector_double(cfg, section, "radii");
      std::size_t const Ng = lamb.size();

      // Unique (lam_bin, zob) pairs -> determines D_A_o, R_excl, bsel slice.
      gp_lzob_idx_.resize(Ng);
      for (std::size_t i = 0; i != Ng; ++i) {
        int    const lb  = static_cast<int>(lamb[i]);
        double const zob = 0.5 * (zlo[i] + zhi[i]);
        int found = -1;
        for (std::size_t k = 0; k != lzob_lb_.size(); ++k) {
          if (lzob_lb_[k] == lb &&
              std::abs(lzob_zob_[k] - zob) < 1e-12) { found = int(k); break; }
        }
        if (found < 0) {
          lzob_lb_.push_back(lb);
          lzob_zob_.push_back(zob);
          found = int(lzob_lb_.size() - 1);
        }
        gp_lzob_idx_[i] = found;
      }

      // Group all R values that appear on each (lob_bin, zob) slice.
      // These share a single theta-grid built in set_sample().
      lzob_Rs_.assign(lzob_lb_.size(), {});
      lzob_Rgp_idx_.assign(lzob_lb_.size(), {});
      for (std::size_t i = 0; i != Ng; ++i) {
        int const lzob = gp_lzob_idx_[i];
        double const R = rad[i];
        auto& Rs  = lzob_Rs_[lzob];
        auto& Rgp = lzob_Rgp_idx_[lzob];
        int found = -1;
        for (std::size_t k = 0; k != Rs.size(); ++k) {
          if (std::abs(Rs[k] - R) < 1e-12) { found = int(k); break; }
        }
        if (found < 0) {
          Rs.push_back(R);
          found = int(Rs.size() - 1);
        }
        Rgp.push_back(found);   // per-grid-point index into Rs on this lzob
      }

      // Cache the grid (used in evaluate to locate the current point
      // by linear scan over (lam_bin, zob, R) — Ng = 120, cheap).
      gp_lam_bin_.resize(Ng);
      gp_zob_.resize(Ng);
      gp_R_.resize(Ng);
      for (std::size_t i = 0; i != Ng; ++i) {
        gp_lam_bin_[i] = static_cast<int>(lamb[i]);
        gp_zob_[i]     = 0.5 * (zlo[i] + zhi[i]);
        gp_R_[i]       = rad[i];
      }
    }

    void
    set_sample(cosmosis::DataBlock& sample)
    {
      hmf_.emplace(sample);
      hmb_.emplace(make_Interp2D(sample, "haloModel", "lnM", "z", "bias"));
      dv_do_dz_.emplace(sample);
      omega_z_.emplace(sample);
      xi_nl_.emplace(make_Interp2D(sample, "xi_nl", "r", "z", "xi_nl"));
      chi_.emplace(make_Interp1D(sample, "distances", "z", "d_c"));
      // Optional: average_sigma_crit_inv is only needed to compute
      // g_t^prj.  When absent (e.g. smoke pipelines that only care
      // about Sigma_prj / DSigma_prj), sci_ is left empty and g_t
      // outputs collapse to 0.
      if (sample.has_val("average_sigma_crit_inv", "zlense")) {
        sci_.emplace(make_Interp1D(sample, "average_sigma_crit_inv",
                                   "zlense", "sci_average"));
      } else {
        sci_.reset();
      }
      // Photo-z sigma(z) from models/z_kernel_data.hh (same convention as
      // p_operator).  Used for the w_z(z, zob) parabolic kernel.
      sigma_z_.emplace(Interp1D(y3_cluster::z_kernel_z(),
                                y3_cluster::z_kernel_sigma()));

      double const omm =
          sample.view<double>("cosmological_parameters", "omega_M");
      double const omn =
          sample.view<double>("cosmological_parameters", "omega_nu");
      // HMF_t already rescales its spline axis by (omm - omn) internally
      // (hmf_t.hh::_adjust_to_log), so the shift here must stay zero.
      ln_mass_shift_ = 0.0;
      h0_ = sample.view<double>("cosmological_parameters", "h0");

      // Switch the NFW normalisation to rho_mean = Omega_m * rho_crit
      // (matches Python richness_selection.nfw.NFWMiscentered which
      // builds rho_s from rho_mean).  Affects every Sigma_mis / DSigma_mis
      // call from here on for this sample.
      sigma_mis_ ->set_rho_mult(omm);
      dsigma_mis_->set_rho_mult(omm);

      // Sample-level zt-reference tables.
      // No zt_ref intermediate table: chi / dV / om / sigma_z / HMF /
      // HMB are read directly from their spline models at the adaptive
      // z-nodes inside the per-slice cache builder.

      bsel_.emplace(sample);

      // ------------------------------------------------------------------
      // Per (lob, zob)-slice theta grid: log-GL on segments split at
      // { lower, theta_excl_o, theta_R(R_i) for each R_i on this slice,
      //   theta_lam, 2*theta_lam, theta_max }.  All R values on the slice
      // share one theta-grid (every R lands its own theta_R breakpoint),
      // so b_sel(theta) and geometry tables are shared across R while
      // Sigma_mis / DSigma_mis depend on R and get their own per-R cache.
      // ------------------------------------------------------------------
      lzob_chi_o_ .assign(lzob_lb_.size(), 0.0);
      lzob_R_excl_.assign(lzob_lb_.size(), 0.0);
      lzob_sci_   .assign(lzob_lb_.size(), 0.0);
      for (std::size_t k = 0; k != lzob_lb_.size(); ++k) {
        int    const lb  = lzob_lb_[k];
        double const zob = lzob_zob_[k];
        double const lobc = bsel_->at(lb, zob).lob;
        lzob_chi_o_[k]  = (*chi_).clamp(zob) * h0_;
        lzob_R_excl_[k] = sp_detail::R_lambda(lobc) * (1.0 + zob);
        lzob_sci_[k]    = sci_ ? (*sci_).clamp(zob) : 0.0;
      }

      std::size_t const Nlz = lzob_lb_.size();
      lzob_theta_      .assign(Nlz, {});
      lzob_sin_theta_  .assign(Nlz, {});
      lzob_cos_theta_  .assign(Nlz, {});
      lzob_geom_       .assign(Nlz, {});     // w_theta * 2 pi sin theta
      lzob_bsel_       .assign(Nlz, {});
      lzob_Smis_       .assign(Nlz, {});     // flattened [i_R][i_theta * N_lnm + i_M]
      lzob_DSmis_      .assign(Nlz, {});
      lzob_breakpoints_.assign(Nlz, {});
      // Z-contracted per-slice weights (see member decl).
      lzob_wrnd_M_.assign(Nlz, {});
      lzob_wcl_M_ .assign(Nlz, {});

      for (std::size_t k = 0; k != Nlz; ++k) {
        int    const lob_bin = lzob_lb_[k];
        double const zob     = lzob_zob_[k];
        double const chi_o   = lzob_chi_o_[k];
        double const D_A_o   = chi_o / (1.0 + zob);
        double const R_excl  = lzob_R_excl_[k];
        auto const bsel_bin = bsel_->at(lob_bin, zob);
        double const lobc    = bsel_bin.lob;
        auto const& Rs       = lzob_Rs_[k];

        auto const tg = sp_detail::build_theta_grid(lobc, zob, Rs,
                                                    chi_o, D_A_o, R_excl,
                                                    N_per_seg_,
                                                    R_max_cMpch_,
                                                    extra_breakpoints_);
        std::size_t const Nth = tg.theta.size();
        auto& theta_k = lzob_theta_    [k]; theta_k = tg.theta;
        auto& sin_k   = lzob_sin_theta_[k]; sin_k.resize(Nth);
        auto& cos_k   = lzob_cos_theta_[k]; cos_k.resize(Nth);
        auto& geom_k  = lzob_geom_     [k]; geom_k.resize(Nth);
        for (std::size_t it = 0; it != Nth; ++it) {
          sin_k[it]  = std::sin(theta_k[it]);
          cos_k[it]  = std::cos(theta_k[it]);
          geom_k[it] = tg.weight[it] * 2.0 * sp_detail::PI * sin_k[it];
        }
        lzob_breakpoints_[k] = tg.breakpoints;

        // b_sel(theta) on this slice's theta grid.
        auto& bsel_k = lzob_bsel_[k];
        bsel_k.resize(Nth);
        {
          double const theta_lam = sp_detail::R_lambda(lobc)
                                   * (1.0 + zob) / chi_o;
          double const k_sig   = 2.5 / theta_lam;
          double const theta0  = 0.5 * theta_lam;
          double const delta_B = bsel_bin.b_large - bsel_bin.b_small;
          for (std::size_t it = 0; it != Nth; ++it) {
            double const sgm =
                1.0 / (1.0 + std::exp(-k_sig * (theta_k[it] - theta0)));
            bsel_k[it] = bsel_bin.b_small + delta_B * sgm;
          }
        }

        // Per-R NFW caches Smis(R | theta, M) and DSmis(R | theta, M).
        auto& Smis_k  = lzob_Smis_ [k];
        auto& DSmis_k = lzob_DSmis_[k];
        Smis_k .assign(Rs.size() * Nth * N_lnm_, 0.0);
        DSmis_k.assign(Rs.size() * Nth * N_lnm_, 0.0);
        for (std::size_t iR = 0; iR != Rs.size(); ++iR) {
          double const R = Rs[iR];
          double* Sbase  = &Smis_k [iR * Nth * N_lnm_];
          double* DSbase = &DSmis_k[iR * Nth * N_lnm_];
          for (std::size_t it = 0; it != Nth; ++it) {
            double const R_th = theta_k[it] * D_A_o;
            double* Srow  = Sbase  + it * N_lnm_;
            double* DSrow = DSbase + it * N_lnm_;
            for (std::size_t iM = 0; iM != N_lnm_; ++iM) {
              Srow[iM]  = (*sigma_mis_ )(R, R_th, lnm_x_[iM]);
              DSrow[iM] = (*dsigma_mis_)(R, R_th, lnm_x_[iM]);
            }
          }
        }

        // Per-slice z-axis caches (wall-independent at fixed lob, zob).
        // Builds:
        //   zs, wzs      -- adaptive ring+fg/bg grid
        //   common_z[iz] = dV(z) * om(z) * w_zgl(z) * w_photoz(z, zob)
        //                  with the parabolic photo-z window folded in.
        //   hmf_z, hmb_z -- (Nz, N_lnm_) evaluated directly from the
        //                   HMF / HMB splines at the adaptive z-nodes.
        //   xi_mask[iz*Nth + it] -- xi_NL(|dr|, zob) or 0 when
        //     theta <= theta_excl_z(iz).  Folds the LoS-slab exclusion so
        //     evaluate() reads a ready multiplier.
        // Z-contraction (Python sigma_prj.py:248-252): for every theta
        // node, the integrand separates into a (z, M)-only kernel times
        // Sigma_mis(R, theta*D_A, M).  We sum over z once per slice
        // into M-vectors and leave theta as the outer loop in evaluate:
        //   wrnd_M[iM]        = sum_z  wzs(z) * dV(z) * om(z) * wphot(z) * n(M,z) * w_lnM[iM]
        //   wcl_M[it, iM]     = sum_z  wzs(z) * dV(z) * om(z) * wphot(z) * n(M,z) * b(M,z)
        //                              * xi_NL(|dr(z, theta_it)|)
        //                              * 1[theta_it > theta_excl(z)] * w_lnM[iM]
        // Evaluate then only runs a theta-outer, M-inner dot product
        // against the pre-cached Smis(R | theta, M) table.
        std::vector<double> zs, wzs;
        build_z_grid_(zob, chi_o, R_excl, zs, wzs);
        std::size_t const Nz = zs.size();

        auto& wrnd_M = lzob_wrnd_M_[k];
        auto& wcl_M  = lzob_wcl_M_[k];
        wrnd_M.assign(N_lnm_, 0.0);
        wcl_M .assign(Nth * N_lnm_, 0.0);

        // One-time buffers reused per z (fresh vs the vector<double>
        // allocation overhead of the old xi_mask_k tables).
        std::vector<double> hmf_row(N_lnm_), hmb_row(N_lnm_);
        for (std::size_t iz = 0; iz != Nz; ++iz) {
          double const z     = zs[iz];
          double const chi_z = (*chi_).clamp(z) * h0_;
          double const dV    = (*dv_do_dz_)(z);
          // Omega(z) is the SDSS/DES effective survey solid angle
          // (rad^2), only relevant for number-count operators like
          // P_1/I_1/I_2 where the integral is a cluster count.  For
          // the Sigma_prj / DSigma_prj surface densities it cancels
          // between the numerator and the normalisation, so it must
          // NOT appear here -- that is why Python SigmaPrj has no
          // survey-area weight.
          double const sig_z = (*sigma_z_).clamp(z);
          double const u_phot = (z - zob) / sig_z;
          double const wz_phot = (std::abs(u_phot) < 1.0)
                                 ? (1.0 - u_phot * u_phot) : 0.0;
          double const common_z = dV * wzs[iz] * wz_phot;
          if (common_z == 0.0) continue;

          for (std::size_t iM = 0; iM != N_lnm_; ++iM) {
            hmf_row[iM] = (*hmf_)(lnm_x_[iM] + ln_mass_shift_, z);
            hmb_row[iM] = hmb_->clamp(lnm_x_[iM], z);
          }

          // rnd accumulator: z-independent of theta, sums straight in.
          for (std::size_t iM = 0; iM != N_lnm_; ++iM)
            wrnd_M[iM] += common_z * hmf_row[iM] * lnm_w_[iM];

          // cl accumulator: per-theta because of xi_NL and the
          // per-z LoS-slab exclusion.
          double const theta_excl_z = sp_detail::theta_excl_at_z(chi_z, chi_o,
                                                                 R_excl);
          for (std::size_t it = 0; it != Nth; ++it) {
            double const theta = theta_k[it];
            if (theta <= theta_excl_z) continue;
            double const dchi = std::sqrt(std::max(
                chi_z * chi_z + chi_o * chi_o
                - 2.0 * chi_z * chi_o * cos_k[it], 0.0));
            double const xi_val = (*xi_nl_).clamp(dchi, zob);
            if (xi_val == 0.0) continue;
            double const wc_ztheta = common_z * xi_val;
            double* wcl_row = &wcl_M[it * N_lnm_];
            for (std::size_t iM = 0; iM != N_lnm_; ++iM)
              wcl_row[iM] += wc_ztheta * hmf_row[iM]
                             * hmb_row[iM] * lnm_w_[iM];
          }
        }
      }
    }

    // Locate (lob_bin, zob, R) in the wall-grid; returns (lzob_idx,
    // iR_on_slice).  Linear scan; Ng=120 in production.
    std::pair<int, int>
    locate_(int lob_bin, double zob, double R) const
    {
      int gp_idx = -1;
      for (std::size_t i = 0; i != gp_zob_.size(); ++i) {
        if (gp_lam_bin_[i] == lob_bin &&
            std::abs(gp_zob_[i] - zob) < 1e-12 &&
            std::abs(gp_R_[i]   - R)   < 1e-12) { gp_idx = int(i); break; }
      }
      int const lzob_idx = gp_lzob_idx_[gp_idx];
      int iR_on_slice = -1;
      auto const& Rs = lzob_Rs_[lzob_idx];
      for (std::size_t k = 0; k != Rs.size(); ++k) {
        if (std::abs(Rs[k] - R) < 1.0e-12) {
          iR_on_slice = int(k);
          break;
        }
      }
      return {lzob_idx, iR_on_slice};
    }

    // Inner-product over (theta, M) of a per-(theta, M) miscentering
    // table (Smis or DSmis) with the z-contracted M-vectors.  Returns
    // {total = rnd + cl, rnd, cl}.  Used by sigma_prj/dsigma_prj.
    std::array<double, 3>
    accumulate_(int lzob_idx, double const* table) const
    {
      auto const& geom_k  = lzob_geom_[lzob_idx];
      auto const& bsel_k  = lzob_bsel_[lzob_idx];
      auto const& wrnd_M  = lzob_wrnd_M_[lzob_idx];
      auto const& wcl_M   = lzob_wcl_M_ [lzob_idx];
      std::size_t const Nth = lzob_theta_[lzob_idx].size();

      double acc_rnd = 0.0, acc_cl = 0.0;
      for (std::size_t it = 0; it != Nth; ++it) {
        double const g    = geom_k[it];
        double const bsel = bsel_k[it];
        double const* row     = &table[it * N_lnm_];
        double const* wcl_row = &wcl_M[it * N_lnm_];

        double a_r = 0.0, a_c = 0.0;
        #pragma omp simd reduction(+:a_r,a_c)
        for (std::size_t iM = 0; iM != N_lnm_; ++iM) {
          double const v = row[iM];
          a_r += wrnd_M [iM] * v;
          a_c += wcl_row[iM] * v;
        }
        acc_rnd += g * a_r;
        acc_cl  += g * bsel * a_c;
      }
      double const total = acc_rnd + acc_cl;
      return {total, acc_rnd, acc_cl};
    }

   public:
    // Σ_prj at (lob_bin, zob, R): {total, rnd, cl}.
    std::array<double, 3>
    sigma_prj(grid_point_t const& pt) const
    {
      int    const lob_bin = static_cast<int>(pt[0]);
      double const zob     = 0.5 * (pt[1] + pt[2]);
      double const R       = pt[3];
      auto const [lzob_idx, iR] = locate_(lob_bin, zob, R);
      std::size_t const Nth = lzob_theta_[lzob_idx].size();
      double const* const Smis =
          &lzob_Smis_[lzob_idx][iR * Nth * N_lnm_];
      return accumulate_(lzob_idx, Smis);
    }

    // ΔΣ_prj at (lob_bin, zob, R): {total, rnd, cl}.
    std::array<double, 3>
    dsigma_prj(grid_point_t const& pt) const
    {
      int    const lob_bin = static_cast<int>(pt[0]);
      double const zob     = 0.5 * (pt[1] + pt[2]);
      double const R       = pt[3];
      auto const [lzob_idx, iR] = locate_(lob_bin, zob, R);
      std::size_t const Nth = lzob_theta_[lzob_idx].size();
      double const* const DSmis =
          &lzob_DSmis_[lzob_idx][iR * Nth * N_lnm_];
      return accumulate_(lzob_idx, DSmis);
    }

    // gamma_t^prj = ΔΣ_prj · Σ_crit^-1 (linear; reduced-shear
    // denominator retired 2026-05-11 to preserve the additive
    // decomposition γ_t^total = γ_t^1h + γ_t^prj used by the
    // likelihood).
    std::array<double, 3>
    shear_prj(grid_point_t const& pt) const
    {
      int    const lob_bin = static_cast<int>(pt[0]);
      double const zob     = 0.5 * (pt[1] + pt[2]);
      double const R       = pt[3];
      auto const [lzob_idx, iR] = locate_(lob_bin, zob, R);
      std::size_t const Nth = lzob_theta_[lzob_idx].size();
      double const* const DSmis =
          &lzob_DSmis_[lzob_idx][iR * Nth * N_lnm_];
      auto const ds = accumulate_(lzob_idx, DSmis);
      double const sci_v = lzob_sci_[lzob_idx];
      return {ds[0] * sci_v, ds[1] * sci_v, ds[2] * sci_v};
    }

    // --- Ring + fg/bg log-|Δchi| zt grid -----------------------------------
    // Mirrors p_operator_t.hh::build_z_grid_ but clips to the ini-configured
    // integration range [zt_lo_, zt_hi_] instead of photo-z support (Σ_prj
    // has no photo-z kernel).
    void
    build_z_grid_(double zob, double chi_o, double R_excl,
                  std::vector<double>& zs,
                  std::vector<double>& wzs) const
    {
      zs.clear();
      wzs.clear();

      double const z_fg_lo = zt_lo_;        // no photo-z clip: full z range
      double const z_bg_hi = zt_hi_;
      double const chi_fg_lo = (*chi_).clamp(z_fg_lo) * h0_;
      double const chi_bg_hi = (*chi_).clamp(z_bg_hi) * h0_;

      double const dz = 1.0e-3;
      double const chi_plus  = (*chi_).clamp(zob + dz) * h0_;
      double const chi_minus = (*chi_).clamp(zob - dz) * h0_;
      double const dchi_dz = (chi_plus - chi_minus) / (2.0 * dz);

      double const dz_excl = R_excl / dchi_dz;
      double const z_ring_lo = std::max(zob - dz_excl, z_fg_lo);
      double const z_ring_hi = std::min(zob + dz_excl, z_bg_hi);

      std::vector<double> z_ring, w_ring;
      if (z_ring_hi > z_ring_lo)
        p_op_detail::gl_nodes(z_ring_lo, z_ring_hi, N_zring_, z_ring, w_ring);

      std::vector<double> z_fg, w_fg;
      double const dis_fg_max = chi_o - chi_fg_lo;
      if (R_excl < dis_fg_max) {
        std::vector<double> u_fg, w_u_fg;
        p_op_detail::gl_nodes(std::log(R_excl), std::log(dis_fg_max),
                              N_zouter_, u_fg, w_u_fg);
        z_fg.resize(N_zouter_);
        w_fg.resize(N_zouter_);
        for (std::size_t i = 0; i != N_zouter_; ++i) {
          double const dis = std::exp(u_fg[i]);
          double const z_i = invert_chi_(chi_o - dis);
          double const chip = (*chi_).clamp(z_i + dz) * h0_;
          double const chim = (*chi_).clamp(z_i - dz) * h0_;
          double const ddz  = (chip - chim) / (2.0 * dz);
          z_fg[i] = z_i;
          w_fg[i] = w_u_fg[i] * dis / ddz;
        }
      }

      std::vector<double> z_bg, w_bg;
      double const dis_bg_max = chi_bg_hi - chi_o;
      if (R_excl < dis_bg_max) {
        std::vector<double> u_bg, w_u_bg;
        p_op_detail::gl_nodes(std::log(R_excl), std::log(dis_bg_max),
                              N_zouter_, u_bg, w_u_bg);
        z_bg.resize(N_zouter_);
        w_bg.resize(N_zouter_);
        for (std::size_t i = 0; i != N_zouter_; ++i) {
          double const dis = std::exp(u_bg[i]);
          double const z_i = invert_chi_(chi_o + dis);
          double const chip = (*chi_).clamp(z_i + dz) * h0_;
          double const chim = (*chi_).clamp(z_i - dz) * h0_;
          double const ddz  = (chip - chim) / (2.0 * dz);
          z_bg[i] = z_i;
          w_bg[i] = w_u_bg[i] * dis / ddz;
        }
      }

      zs.reserve(z_fg.size() + z_ring.size() + z_bg.size());
      wzs.reserve(zs.capacity());
      for (std::size_t i = z_fg.size(); i--;) {
        zs.push_back(z_fg[i]);
        wzs.push_back(w_fg[i]);
      }
      for (std::size_t i = 0; i != z_ring.size(); ++i) {
        zs.push_back(z_ring[i]);
        wzs.push_back(w_ring[i]);
      }
      for (std::size_t i = 0; i != z_bg.size(); ++i) {
        zs.push_back(z_bg[i]);
        wzs.push_back(w_bg[i]);
      }
    }

    // Invert chi(z) = chi_target by bisection.  Same as p_operator's.
    double
    invert_chi_(double chi_target) const
    {
      double lo = 0.001, hi = 2.0;
      for (int it = 0; it < 40; ++it) {
        double const mid = 0.5 * (lo + hi);
        double const c   = (*chi_).clamp(mid) * h0_;
        if (c < chi_target) lo = mid;
        else                hi = mid;
      }
      return 0.5 * (lo + hi);
    }

    static grid_t
    make_grid_points(cosmosis::DataBlock& cfg, char const* section)
    {
      return y3_cluster::make_grid_points_wall_of_numbers(
          cfg, section,
          "lambda_bin", "zo_low", "zo_high", "radii");
    }

   private:
    char const* section_;
    std::size_t N_lnm_, N_per_seg_;
    std::size_t N_zring_, N_zouter_;
    double zt_lo_, zt_hi_;
    double lnm_lo_, lnm_hi_;
    double R_max_cMpch_;      // cMpc/h; theta_max(zob) = R_max_cMpch / D_A(zob)
    std::vector<double> extra_breakpoints_;   // optional ini input, radians
    std::vector<double> lob_centers_;   // per-lambda-bin effective richness

    std::vector<double> lnm_x_, lnm_w_;

    std::optional<y3_cluster::HMF_t>       hmf_;
    std::optional<Interp2D>                hmb_;
    std::optional<y3_cluster::DV_DO_DZ_t>  dv_do_dz_;
    std::optional<y3_cluster::OMEGA_Z_DES> omega_z_;
    std::optional<Interp2D>                xi_nl_;
    std::optional<NFW_SIGMA_MIS>           sigma_mis_;
    std::optional<NFW_DSIGMA_MIS>          dsigma_mis_;
    std::optional<Interp1D>                chi_;
    std::optional<Interp1D>                sci_;
    std::optional<Interp1D>                sigma_z_;   // photo-z kernel sigma(z)

    std::optional<sp_detail::BSelBins> bsel_;

    // Grid axes parsed at ctor (one entry per wall-grid point).
    std::vector<int>    gp_lam_bin_;
    std::vector<double> gp_zob_, gp_R_;

    // Unique (lam_bin, zob) -> (chi_o, R_excl, sci) cache and per-slice
    // grouping of R values.
    std::vector<int>    lzob_lb_;
    std::vector<double> lzob_zob_;
    std::vector<int>    gp_lzob_idx_;
    std::vector<double> lzob_chi_o_, lzob_R_excl_, lzob_sci_;

    // R values per (lob, zob) slice (drives the theta-breakpoint set).
    std::vector<std::vector<double>> lzob_Rs_;
    // For each (lob, zob) slice: indices into Rs[]  of every wall-grid
    // point that falls on this slice, in the order the wall-grid visits them.
    // Unused once evaluate() looks up R directly; kept for diagnostics.
    std::vector<std::vector<int>>    lzob_Rgp_idx_;

    // Per-slice theta grids + caches.  Layout:
    //   lzob_theta_[k].size() == Nth_k    (one per slice)
    //   lzob_Smis_ [k].size() == Rs.size() * Nth_k * N_lnm_
    //   lzob_DSmis_[k] same shape.
    std::vector<std::vector<double>> lzob_theta_;
    std::vector<std::vector<double>> lzob_sin_theta_;
    std::vector<std::vector<double>> lzob_cos_theta_;
    std::vector<std::vector<double>> lzob_geom_;        // w_theta * 2pi sin theta
    std::vector<std::vector<double>> lzob_bsel_;
    std::vector<std::vector<double>> lzob_Smis_;
    std::vector<std::vector<double>> lzob_DSmis_;
    std::vector<std::vector<double>> lzob_breakpoints_; // diagnostics
    // z-contracted per-slice weights (theta outer, z middle, M inner):
    //   wrnd_M[iM] = sum_z  common_z(z) * n(M,z)
    //   wcl_M[it * N_lnm_ + iM] = sum_z  common_z(z) * n(M,z) * b(M,z)
    //                                    * xi_NL(|dr(z,theta)|) * 1[theta>theta_excl(z)]
    // The dot product over M with Smis_k[iR, it, :] yields the
    // rnd / cl contribution at fixed theta, reused across every R on
    // the slice.  Computed once per (lob, zob) slice in set_sample.
    std::vector<std::vector<double>> lzob_wrnd_M_;      // (N_lnm_,) per slice
    std::vector<std::vector<double>> lzob_wcl_M_;       // (Nth * N_lnm_,) per slice

    double ln_mass_shift_ = 0.0;
    double h0_            = 1.0;
  };

  }  // namespace sp_detail

  // =====================================================================
  // SigmaPrjEvaluator / DSigmaPrjEvaluator / ShearPrjEvaluator --
  // thin CosmoSIS wrappers around sp_detail::ShearPrjCore.  Each one
  // owns its own ShearPrjCore (separate cache) and reads its own
  // [section] block from the .ini.  Output is the {total, rnd, cl}
  // triple for one observable; the three classes mirror the public
  // API expected by CosmoSISScalarEvaluatorModule (constructor +
  // set_sample + evaluate + module_label + output_sections +
  // output_names + make_grid_points).
  // =====================================================================
  class SigmaPrjEvaluator {
   public:
    using grid_t       = sp_detail::ShearPrjCore::grid_t;
    using grid_point_t = sp_detail::ShearPrjCore::grid_point_t;
    static constexpr std::size_t n_outputs = 3;

    explicit SigmaPrjEvaluator(cosmosis::DataBlock& cfg)
      : core_(cfg, module_label()) {}

    void set_sample(cosmosis::DataBlock& s) { core_.set_sample(s); }

    std::array<double, 3>
    evaluate(grid_point_t const& pt) const { return core_.sigma_prj(pt); }

    static char const* module_label() { return "sigma_prj"; }
    static std::array<char const*, 3>
    output_sections() { return {"sigma_prj", "sigma_prj", "sigma_prj"}; }
    static std::array<char const*, 3>
    output_names() { return {"vals", "rnd", "cl"}; }
    static grid_t
    make_grid_points(cosmosis::DataBlock& cfg)
    { return sp_detail::ShearPrjCore::make_grid_points(cfg, module_label()); }

   private:
    sp_detail::ShearPrjCore core_;
  };

  class DSigmaPrjEvaluator {
   public:
    using grid_t       = sp_detail::ShearPrjCore::grid_t;
    using grid_point_t = sp_detail::ShearPrjCore::grid_point_t;
    static constexpr std::size_t n_outputs = 3;

    explicit DSigmaPrjEvaluator(cosmosis::DataBlock& cfg)
      : core_(cfg, module_label()) {}

    void set_sample(cosmosis::DataBlock& s) { core_.set_sample(s); }

    std::array<double, 3>
    evaluate(grid_point_t const& pt) const { return core_.dsigma_prj(pt); }

    static char const* module_label() { return "dsigma_prj"; }
    static std::array<char const*, 3>
    output_sections() { return {"dsigma_prj", "dsigma_prj", "dsigma_prj"}; }
    static std::array<char const*, 3>
    output_names() { return {"vals", "rnd", "cl"}; }
    static grid_t
    make_grid_points(cosmosis::DataBlock& cfg)
    { return sp_detail::ShearPrjCore::make_grid_points(cfg, module_label()); }

   private:
    sp_detail::ShearPrjCore core_;
  };

  class ShearPrjEvaluator {
   public:
    using grid_t       = sp_detail::ShearPrjCore::grid_t;
    using grid_point_t = sp_detail::ShearPrjCore::grid_point_t;
    static constexpr std::size_t n_outputs = 3;

    explicit ShearPrjEvaluator(cosmosis::DataBlock& cfg)
      : core_(cfg, module_label()) {}

    void set_sample(cosmosis::DataBlock& s) { core_.set_sample(s); }

    std::array<double, 3>
    evaluate(grid_point_t const& pt) const { return core_.shear_prj(pt); }

    static char const* module_label() { return "shear_prj"; }
    static std::array<char const*, 3>
    output_sections() { return {"shear_prj", "shear_prj", "shear_prj"}; }
    static std::array<char const*, 3>
    output_names() { return {"vals", "rnd", "cl"}; }
    static grid_t
    make_grid_points(cosmosis::DataBlock& cfg)
    { return sp_detail::ShearPrjCore::make_grid_points(cfg, module_label()); }

   private:
    sp_detail::ShearPrjCore core_;
  };

  // =====================================================================
  // ShearPrjGsl -- same integrand as the ShearPrjCore family, but the
  // outer z-integral is driven by GSL QAGP (piecewise-QAG with an
  // explicit singular point at z=zob).  QAGP adapts around the xi_nl(Δchi)
  // peak by construction, so no ring+fg/bg grid is needed.  Inner (lnM, θ)
  // still use fixed GL, because those axes are smooth.
  //
  // Writes sigma_prj_gsl / dsigma_prj_gsl / shear_prj_gsl — coexists
  // with the main evaluator so they can be diffed head-to-head.
  // =====================================================================
  class ShearPrjGsl {
   public:
    using grid_t       = y3_cluster::grid_t<4>;
    using grid_point_t = grid_t::value_type;

    static constexpr std::size_t n_outputs = 3;

    explicit ShearPrjGsl(cosmosis::DataBlock& cfg)
      : N_lnm_(cfg.has_val(module_label(), "n_lnm")
                 ? cfg.view<int>(module_label(), "n_lnm") : 24)
      , N_th_(cfg.has_val(module_label(), "n_theta")
                ? cfg.view<int>(module_label(), "n_theta") : 30)
      , N_zt_ref_(cfg.has_val(module_label(), "n_zt_ref")
                    ? cfg.view<int>(module_label(), "n_zt_ref") : 80)
      , eps_rel_(cfg.has_val(module_label(), "eps_rel")
                   ? cfg.view<double>(module_label(), "eps_rel") : 1.0e-3)
      , eps_abs_(cfg.has_val(module_label(), "eps_abs")
                   ? cfg.view<double>(module_label(), "eps_abs") : 1.0e-12)
      , max_eval_(cfg.has_val(module_label(), "max_eval")
                    ? cfg.view<int>(module_label(), "max_eval") : 1000)
      , zt_lo_(cfg.view<double>(module_label(), "zt_low"))
      , zt_hi_(cfg.view<double>(module_label(), "zt_high"))
      , lnm_lo_(cfg.view<double>(module_label(), "lnm_low"))
      , lnm_hi_(cfg.view<double>(module_label(), "lnm_high"))
      , th_lo_(cfg.view<double>(module_label(), "theta_low"))
      , th_hi_(cfg.view<double>(module_label(), "theta_high"))
    {
      p_op_detail::gl_nodes(lnm_lo_, lnm_hi_, N_lnm_, lnm_x_, lnm_w_);
      p_op_detail::gl_nodes(std::log(th_lo_), std::log(th_hi_),
                            N_th_, u_th_x_, u_th_w_);
      th_x_.resize(N_th_);
      th_w_.resize(N_th_);
      sin_th_.resize(N_th_);
      cos_th_.resize(N_th_);
      geom_.resize(N_th_);
      for (std::size_t it = 0; it != N_th_; ++it) {
        th_x_[it]  = std::exp(u_th_x_[it]);
        th_w_[it]  = u_th_w_[it] * th_x_[it];
        sin_th_[it] = std::sin(th_x_[it]);
        cos_th_[it] = std::cos(th_x_[it]);
        geom_[it]   = th_w_[it] * 2.0 * sp_detail::PI * sin_th_[it];
      }
      zt_ref_.resize(N_zt_ref_);
      double const dz_ref = (zt_hi_ - zt_lo_) / (N_zt_ref_ - 1);
      for (std::size_t i = 0; i != N_zt_ref_; ++i)
        zt_ref_[i] = zt_lo_ + i * dz_ref;
      inv_dz_ref_ = 1.0 / dz_ref;

      // Projection convention: SINGLE miscentering kernel, c=4,
      // rho_crit-based tables.  rho_mult is set to Omega_m in
      // set_sample to match Python's rho_mean normalisation.
      sigma_mis_.emplace(4.0, 2.77533742639e+11, NFW_SIG_SINGLE);
      dsigma_mis_.emplace(4.0, 2.77533742639e+11, SINGLE);

      // Per-bin richness centres (DES-Y3 default, ini-overridable).
      if (cfg.has_val(module_label(), "lob_centers")) {
        lob_centers_ = get_vector_double(cfg, module_label(),
                                         "lob_centers");
      } else {
        auto const& dflt = sp_detail::default_lob_centers();
        lob_centers_.assign(dflt.begin(), dflt.end());
      }

      // Persistent GSL workspace -- allocated once per module lifetime.
      // max_eval_ bounds the number of subinterval evaluations per QAGP
      // call (each subinterval = one 21-pt Gauss-Kronrod call).
      gsl_ws_ = gsl_integration_workspace_alloc(max_eval_);

      auto const lamb = get_vector_double(cfg, module_label(), "lambda_bin");
      auto const zlo  = get_vector_double(cfg, module_label(), "zo_low");
      auto const zhi  = get_vector_double(cfg, module_label(), "zo_high");
      auto const rad  = get_vector_double(cfg, module_label(), "radii");
      std::size_t const Ng = lamb.size();

      gp_lzob_idx_.resize(Ng);
      for (std::size_t i = 0; i != Ng; ++i) {
        int    const lb  = static_cast<int>(lamb[i]);
        double const zob = 0.5 * (zlo[i] + zhi[i]);
        int found = -1;
        for (std::size_t k = 0; k != lzob_lb_.size(); ++k) {
          if (lzob_lb_[k] == lb &&
              std::abs(lzob_zob_[k] - zob) < 1e-12) { found = int(k); break; }
        }
        if (found < 0) {
          lzob_lb_.push_back(lb);
          lzob_zob_.push_back(zob);
          found = int(lzob_lb_.size() - 1);
        }
        gp_lzob_idx_[i] = found;
      }

      gp_zobR_idx_.resize(Ng);
      for (std::size_t i = 0; i != Ng; ++i) {
        double const zob = 0.5 * (zlo[i] + zhi[i]);
        double const R   = rad[i];
        int found = -1;
        for (std::size_t k = 0; k != zobR_zob_.size(); ++k) {
          if (std::abs(zobR_zob_[k] - zob) < 1e-12 &&
              std::abs(zobR_R_[k]   - R)   < 1e-12) { found = int(k); break; }
        }
        if (found < 0) {
          zobR_zob_.push_back(zob);
          zobR_R_  .push_back(R);
          found = int(zobR_zob_.size() - 1);
        }
        gp_zobR_idx_[i] = found;
      }
    }

    ~ShearPrjGsl()
    {
      if (gsl_ws_) gsl_integration_workspace_free(gsl_ws_);
    }

    void
    set_sample(cosmosis::DataBlock& sample)
    {
      hmf_.emplace(sample);
      hmb_.emplace(make_Interp2D(sample, "haloModel", "lnM", "z", "bias"));
      dv_do_dz_.emplace(sample);
      omega_z_.emplace(sample);
      xi_nl_.emplace(make_Interp2D(sample, "xi_nl", "r", "z", "xi_nl"));
      chi_.emplace(make_Interp1D(sample, "distances", "z", "d_c"));
      sci_.emplace(make_Interp1D(sample, "average_sigma_crit_inv",
                                 "zlense", "sci_average"));

      double const omm =
          sample.view<double>("cosmological_parameters", "omega_M");
      double const omn =
          sample.view<double>("cosmological_parameters", "omega_nu");
      // HMF_t already rescales its spline axis by (omm - omn) internally
      // (hmf_t.hh::_adjust_to_log), so the shift here must stay zero.
      ln_mass_shift_ = 0.0;
      h0_ = sample.view<double>("cosmological_parameters", "h0");

      // NFW rho_mean normalisation (see ShearPrjEvaluator).
      sigma_mis_ ->set_rho_mult(omm);
      dsigma_mis_->set_rho_mult(omm);

      chi_ref_.resize(N_zt_ref_);
      dv_ref_.resize(N_zt_ref_);
      om_ref_.resize(N_zt_ref_);
      for (std::size_t iz = 0; iz != N_zt_ref_; ++iz) {
        double const z = zt_ref_[iz];
        chi_ref_[iz] = (*chi_).clamp(z) * h0_;
        dv_ref_[iz]  = (*dv_do_dz_)(z);
        om_ref_[iz]  = (*omega_z_)(z);
      }
      hmf_ref_.assign(N_zt_ref_ * N_lnm_, 0.0);
      hmb_ref_.assign(N_zt_ref_ * N_lnm_, 0.0);
      for (std::size_t iz = 0; iz != N_zt_ref_; ++iz) {
        double const z = zt_ref_[iz];
        double* nrow = &hmf_ref_[iz * N_lnm_];
        double* brow = &hmb_ref_[iz * N_lnm_];
        for (std::size_t iM = 0; iM != N_lnm_; ++iM) {
          nrow[iM] = (*hmf_)(lnm_x_[iM] + ln_mass_shift_, z);
          brow[iM] = hmb_->clamp(lnm_x_[iM], z);
        }
      }

      bsel_.emplace(sample);

      // NFW cache on unique (zob, R) — same as main evaluator.
      std::size_t const Np = zobR_zob_.size();
      Smis_cache_ .assign(Np * N_th_ * N_lnm_, 0.0);
      DSmis_cache_.assign(Np * N_th_ * N_lnm_, 0.0);
      for (std::size_t ip = 0; ip != Np; ++ip) {
        double const zob   = zobR_zob_[ip];
        double const R     = zobR_R_[ip];
        double const chi_o = (*chi_).clamp(zob) * h0_;
        double const D_A_o = chi_o / (1.0 + zob);
        double* Sbase  = &Smis_cache_ [ip * N_th_ * N_lnm_];
        double* DSbase = &DSmis_cache_[ip * N_th_ * N_lnm_];
        for (std::size_t it = 0; it != N_th_; ++it) {
          double const R_th = th_x_[it] * D_A_o;
          double* Srow  = Sbase  + it * N_lnm_;
          double* DSrow = DSbase + it * N_lnm_;
          for (std::size_t iM = 0; iM != N_lnm_; ++iM) {
            Srow[iM]  = (*sigma_mis_ )(R, R_th, lnm_x_[iM]);
            DSrow[iM] = (*dsigma_mis_)(R, R_th, lnm_x_[iM]);
          }
        }
      }

      // bsel cache on unique (lam_bin, zob).
      std::size_t const Nlz = lzob_lb_.size();
      bsel_th_cache_.assign(Nlz * N_th_, 0.0);
      lzob_sci_     .assign(Nlz, 0.0);
      lzob_R_excl_  .assign(Nlz, 0.0);
      lzob_chi_o_   .assign(Nlz, 0.0);
      for (std::size_t k = 0; k != Nlz; ++k) {
        int    const lb  = lzob_lb_[k];
        double const zob = lzob_zob_[k];
        auto const bsel_bin = bsel_->at(lb, zob);
        double const lobc = bsel_bin.lob;
        double const theta_lam = sp_detail::R_lambda(lobc)
                                 * (1.0 + zob) / ((*chi_).clamp(zob) * h0_);
        double const k_sig = 2.5 / theta_lam;
        double const theta0 = 0.5 * theta_lam;
        double* bsel_row = &bsel_th_cache_[k * N_th_];
        for (std::size_t it = 0; it != N_th_; ++it)
          bsel_row[it] = bsel_bin.b_small
                       + (bsel_bin.b_large - bsel_bin.b_small)
                       / (1.0 + std::exp(-k_sig * (th_x_[it] - theta0)));

        lzob_chi_o_[k]  = (*chi_).clamp(zob) * h0_;
        lzob_R_excl_[k] = sp_detail::R_lambda(lobc)
                          * (1.0 + zob);
        lzob_sci_[k]    = (*sci_).clamp(zob);
      }
    }

    // Struct passed through gsl_function::params to the callback.  Holds
    // everything the inner (lnM, θ) sum needs at the current z node and
    // a selector for Σ vs ΔΣ (Σ is output 0; ΔΣ is output 1).
    struct qagp_ctx {
      ShearPrjGsl const* self;
      double chi_o;
      double R_excl;
      double zob;
      double const* bsel_th_vec;
      double const* Smis;
      double const* DSmis;
      int which;                // 0 = Σ, 1 = ΔΣ
    };

    // GSL integrand: f(z) = scalar inner integral at z.
    static double
    qagp_integrand(double z, void* p)
    {
      auto const* c = static_cast<qagp_ctx const*>(p);
      return c->self->inner_integral_(z, c->chi_o, c->R_excl, c->zob,
                                      c->bsel_th_vec, c->Smis, c->DSmis,
                                      c->which);
    }

    // Evaluate the inner (lnM, θ) sum at one z, returning Σ or ΔΣ
    // contribution (including dV * Ω(z), but no outer dz weight).
    double
    inner_integral_(double z, double chi_o, double R_excl, double zob,
                    double const* bsel_th_vec,
                    double const* Smis, double const* DSmis,
                    int which) const
    {
      // Interp zt-dependent tables onto this z.
      double const u = (z - zt_lo_) * inv_dz_ref_;
      std::size_t const i = std::min<std::size_t>(
                              std::max<std::size_t>(0, std::size_t(u)),
                              N_zt_ref_ - 2);
      double const f = u - i;

      double const cz = chi_ref_[i] + f * (chi_ref_[i+1] - chi_ref_[i]);
      double const dV = dv_ref_ [i] + f * (dv_ref_ [i+1] - dv_ref_ [i]);
      double const om = om_ref_ [i] + f * (om_ref_ [i+1] - om_ref_ [i]);

      // hmf/hmb rows at this z.
      double const* h0r = &hmf_ref_[i       * N_lnm_];
      double const* h1r = &hmf_ref_[(i + 1) * N_lnm_];
      double const* b0r = &hmb_ref_[i       * N_lnm_];
      double const* b1r = &hmb_ref_[(i + 1) * N_lnm_];

      // xi_nl and halo-exclusion per θ.
      double xi_bsel[256];   // N_th_ <= 256 in practice
      for (std::size_t it = 0; it != N_th_; ++it) {
        double const dchi = std::sqrt(std::max(
            cz*cz + chi_o*chi_o - 2.0 * cz * chi_o * cos_th_[it],
            0.0));
        double const xi_val = (dchi < R_excl) ? 0.0
                                              : (*xi_nl_).clamp(dchi, zob);
        xi_bsel[it] = xi_val * bsel_th_vec[it];
      }

      double const common_z = dV * om;

      double out_S  = 0.0;
      double out_DS = 0.0;
      for (std::size_t iM = 0; iM != N_lnm_; ++iM) {
        double const w_M = lnm_w_[iM];
        double const nM  = h0r[iM] + f * (h1r[iM] - h0r[iM]);
        double const bM  = b0r[iM] + f * (b1r[iM] - b0r[iM]);
        double const common_M = common_z * nM * w_M;

        double acc_S  = 0.0;
        double acc_DS = 0.0;
        #pragma omp simd reduction(+:acc_S,acc_DS)
        for (std::size_t it = 0; it != N_th_; ++it) {
          double const gb = geom_[it] * (1.0 + bM * xi_bsel[it]);
          acc_S  += gb * Smis [it * N_lnm_ + iM];
          acc_DS += gb * DSmis[it * N_lnm_ + iM];
        }
        out_S  += common_M * acc_S;
        out_DS += common_M * acc_DS;
      }
      return (which == 0) ? out_S : out_DS;
    }

    std::array<double, 3>
    evaluate(grid_point_t const& pt) const
    {
      int    const lob_bin = static_cast<int>(pt[0]);
      double const zob     = 0.5 * (pt[1] + pt[2]);
      double const R       = pt[3];

      int lzob_idx = -1;
      for (std::size_t k = 0; k != lzob_lb_.size(); ++k) {
        if (lzob_lb_[k] == lob_bin &&
            std::abs(lzob_zob_[k] - zob) < 1e-12) { lzob_idx = int(k); break; }
      }
      int zobR_idx = -1;
      for (std::size_t k = 0; k != zobR_zob_.size(); ++k) {
        if (std::abs(zobR_zob_[k] - zob) < 1e-12 &&
            std::abs(zobR_R_[k]   - R)   < 1e-12) { zobR_idx = int(k); break; }
      }
      double const chi_o  = lzob_chi_o_ [lzob_idx];
      double const R_excl = lzob_R_excl_[lzob_idx];
      double const sci_v  = lzob_sci_   [lzob_idx];
      double const* const bsel_th_vec = &bsel_th_cache_[lzob_idx * N_th_];
      double const* const Smis  =
          &Smis_cache_ [zobR_idx * N_th_ * N_lnm_];
      double const* const DSmis =
          &DSmis_cache_[zobR_idx * N_th_ * N_lnm_];

      qagp_ctx ctx{this, chi_o, R_excl, zob, bsel_th_vec, Smis, DSmis, 0};
      gsl_function F;
      F.function = &qagp_integrand;
      F.params   = &ctx;

      // QAGP breakpoints: {zt_lo, zob, zt_hi}.  The singular point at
      // z = zob is handled by QAGP's explicit-breakpoint split -- the
      // adaptive subdivision clusters evaluations on both sides of the
      // cusp, which is the mechanism Cuhre/Vegas were using implicitly.
      double pts[3] = {zt_lo_, zob, zt_hi_};
      if (zob <= zt_lo_ || zob >= zt_hi_) {
        // Fall back to plain QAG if zob is on the boundary (shouldn't
        // happen with our ini but be safe).
        double result_S = 0.0, err_S = 0.0;
        gsl_integration_qag(&F, zt_lo_, zt_hi_, eps_abs_, eps_rel_,
                            max_eval_, GSL_INTEG_GAUSS21,
                            gsl_ws_, &result_S, &err_S);
        ctx.which = 1;
        double result_DS = 0.0, err_DS = 0.0;
        gsl_integration_qag(&F, zt_lo_, zt_hi_, eps_abs_, eps_rel_,
                            max_eval_, GSL_INTEG_GAUSS21,
                            gsl_ws_, &result_DS, &err_DS);
        // γ_t = ΔΣ · Σ_crit^-1 (linear; reduced-shear form retired 2026-05-11).
        double const g_t = result_DS * sci_v;
        return {result_S, result_DS, g_t};
      }

      double result_S = 0.0, err_S = 0.0;
      gsl_integration_qagp(&F, pts, 3, eps_abs_, eps_rel_,
                           max_eval_, gsl_ws_, &result_S, &err_S);

      ctx.which = 1;
      double result_DS = 0.0, err_DS = 0.0;
      gsl_integration_qagp(&F, pts, 3, eps_abs_, eps_rel_,
                           max_eval_, gsl_ws_, &result_DS, &err_DS);

      // γ_t = ΔΣ · Σ_crit^-1 (linear).
      double const g_t = result_DS * sci_v;

      return {result_S, result_DS, g_t};
    }

    static char const* module_label() { return "shear_prj_gsl"; }

    static std::array<char const*, 3>
    output_sections()
    {
      return {"sigma_prj_gsl", "dsigma_prj_gsl", "shear_prj_gsl"};
    }

    static grid_t
    make_grid_points(cosmosis::DataBlock& cfg)
    {
      return y3_cluster::make_grid_points_wall_of_numbers(
          cfg, module_label(),
          "lambda_bin", "zo_low", "zo_high", "radii");
    }

   private:
    std::size_t N_lnm_, N_th_, N_zt_ref_;
    double eps_rel_, eps_abs_;
    std::size_t max_eval_;
    double zt_lo_, zt_hi_;
    double lnm_lo_, lnm_hi_;
    double th_lo_, th_hi_;

    std::vector<double> lnm_x_, lnm_w_;
    std::vector<double> u_th_x_, u_th_w_;
    std::vector<double> th_x_,  th_w_;
    std::vector<double> sin_th_, cos_th_, geom_;

    std::vector<double> zt_ref_;
    double inv_dz_ref_ = 0.0;

    std::optional<y3_cluster::HMF_t>       hmf_;
    std::optional<Interp2D>                hmb_;
    std::optional<y3_cluster::DV_DO_DZ_t>  dv_do_dz_;
    std::optional<y3_cluster::OMEGA_Z_DES> omega_z_;
    std::optional<Interp2D>                xi_nl_;
    std::optional<NFW_SIGMA_MIS>           sigma_mis_;
    std::optional<NFW_DSIGMA_MIS>          dsigma_mis_;
    std::optional<Interp1D>                chi_;
    std::optional<Interp1D>                sci_;

    std::vector<double> chi_ref_, dv_ref_, om_ref_;
    std::vector<double> hmf_ref_, hmb_ref_;

    std::optional<sp_detail::BSelBins> bsel_;

    std::vector<int>    lzob_lb_;
    std::vector<double> lzob_zob_;
    std::vector<int>    gp_lzob_idx_;
    std::vector<double> bsel_th_cache_;
    std::vector<double> lzob_chi_o_, lzob_R_excl_, lzob_sci_;

    std::vector<double> zobR_zob_, zobR_R_;
    std::vector<int>    gp_zobR_idx_;
    std::vector<double> Smis_cache_, DSmis_cache_;
    std::vector<double> lob_centers_;   // per-bin effective richness

    double ln_mass_shift_ = 0.0;
    double h0_            = 1.0;

    gsl_integration_workspace* gsl_ws_ = nullptr;
  };

  // =====================================================================
  // ShearPrjCuhre -- same theta recipe as ShearPrjEvaluator
  // (outer log-GL in theta on segments split at feature breakpoints),
  // but the INNER 2-D (z, lnM) integral is driven by cubacpp::Cuhre or
  // cubacpp::Vegas.  Acts as the adaptive reference path that must
  // converge to the fixed-GL evaluator and to Python SigmaPrj.
  //
  // Writes sigma_prj_cuhre / dsigma_prj_cuhre / shear_prj_cuhre,
  // each with {vals, rnd, cl} subfields (9-vector per grid point),
  // matching the fixed-GL evaluator's layout so downstream diffing is
  // straightforward.
  //
  // Reference-table strategy (same as ShearPrjGsl): chi(z), dV(z),
  // om(z), sigma_z(z), HMF(lnM, z), HMB(lnM, z) are pre-evaluated on a
  // linear reference zt grid during set_sample() and linearly
  // interpolated inside the Cuhre callback -- so the adaptive integrand
  // is cheap per evaluation.
  // =====================================================================
  class ShearPrjCuhre {
   public:
    using grid_t       = y3_cluster::grid_t<4>;
    using grid_point_t = grid_t::value_type;

    static constexpr std::size_t n_outputs = 9;

    explicit ShearPrjCuhre(cosmosis::DataBlock& cfg)
      : N_lnm_(cfg.has_val(module_label(), "n_lnm")
                 ? cfg.view<int>(module_label(), "n_lnm") : 24)
      , N_per_seg_(cfg.has_val(module_label(), "n_per_seg")
                     ? cfg.view<int>(module_label(), "n_per_seg") : 30)
      , N_zt_ref_(cfg.has_val(module_label(), "n_zt_ref")
                    ? cfg.view<int>(module_label(), "n_zt_ref") : 80)
      , eps_rel_(cfg.has_val(module_label(), "eps_rel")
                   ? cfg.view<double>(module_label(), "eps_rel") : 1.0e-3)
      , eps_abs_(cfg.has_val(module_label(), "eps_abs")
                   ? cfg.view<double>(module_label(), "eps_abs") : 1.0e-12)
      , max_eval_(cfg.has_val(module_label(), "max_eval")
                    ? cfg.view<int>(module_label(), "max_eval") : 2000000)
      , algorithm_(cfg.has_val(module_label(), "algorithm")
                     ? cfg.view<std::string>(module_label(), "algorithm")
                     : std::string("cuhre"))
      , zt_lo_(cfg.view<double>(module_label(), "zt_low"))
      , zt_hi_(cfg.view<double>(module_label(), "zt_high"))
      , lnm_lo_(cfg.view<double>(module_label(), "lnm_low"))
      , lnm_hi_(cfg.view<double>(module_label(), "lnm_high"))
      , R_max_cMpch_(cfg.has_val(module_label(), "R_max_cMpch")
                       ? cfg.view<double>(module_label(), "R_max_cMpch")
                       : 30.0)
    {
      p_op_detail::gl_nodes(lnm_lo_, lnm_hi_, N_lnm_, lnm_x_, lnm_w_);

      if (cfg.has_val(module_label(), "theta_breakpoints")) {
        extra_breakpoints_ = get_vector_double(cfg, module_label(),
                                               "theta_breakpoints");
      }

      if (cfg.has_val(module_label(), "lob_centers")) {
        lob_centers_ = get_vector_double(cfg, module_label(),
                                         "lob_centers");
      } else {
        auto const& dflt = sp_detail::default_lob_centers();
        lob_centers_.assign(dflt.begin(), dflt.end());
      }

      zt_ref_.resize(N_zt_ref_);
      double const dz_ref = (zt_hi_ - zt_lo_) / (N_zt_ref_ - 1);
      for (std::size_t i = 0; i != N_zt_ref_; ++i)
        zt_ref_[i] = zt_lo_ + i * dz_ref;
      inv_dz_ref_ = 1.0 / dz_ref;

      // Projection convention: SINGLE miscentering kernel, c=4,
      // rho_crit-based tables; rho_mult set per sample.
      sigma_mis_.emplace(4.0, 2.77533742639e+11, NFW_SIG_SINGLE);
      dsigma_mis_.emplace(4.0, 2.77533742639e+11, SINGLE);

      auto const lamb = get_vector_double(cfg, module_label(), "lambda_bin");
      auto const zlo  = get_vector_double(cfg, module_label(), "zo_low");
      auto const zhi  = get_vector_double(cfg, module_label(), "zo_high");
      auto const rad  = get_vector_double(cfg, module_label(), "radii");
      std::size_t const Ng = lamb.size();

      gp_lzob_idx_.resize(Ng);
      for (std::size_t i = 0; i != Ng; ++i) {
        int    const lb  = static_cast<int>(lamb[i]);
        double const zob = 0.5 * (zlo[i] + zhi[i]);
        int found = -1;
        for (std::size_t k = 0; k != lzob_lb_.size(); ++k) {
          if (lzob_lb_[k] == lb &&
              std::abs(lzob_zob_[k] - zob) < 1e-12) { found = int(k); break; }
        }
        if (found < 0) {
          lzob_lb_.push_back(lb);
          lzob_zob_.push_back(zob);
          found = int(lzob_lb_.size() - 1);
        }
        gp_lzob_idx_[i] = found;
      }

      lzob_Rs_.assign(lzob_lb_.size(), {});
      for (std::size_t i = 0; i != Ng; ++i) {
        int const lzob = gp_lzob_idx_[i];
        double const R = rad[i];
        auto& Rs = lzob_Rs_[lzob];
        bool have = false;
        for (double const x : Rs) if (std::abs(x - R) < 1e-12) { have = true; break; }
        if (!have) Rs.push_back(R);
      }

      gp_lam_bin_.resize(Ng);
      gp_zob_.resize(Ng);
      gp_R_.resize(Ng);
      for (std::size_t i = 0; i != Ng; ++i) {
        gp_lam_bin_[i] = static_cast<int>(lamb[i]);
        gp_zob_[i]     = 0.5 * (zlo[i] + zhi[i]);
        gp_R_[i]       = rad[i];
      }
    }

    void
    set_sample(cosmosis::DataBlock& sample)
    {
      hmf_.emplace(sample);
      hmb_.emplace(make_Interp2D(sample, "haloModel", "lnM", "z", "bias"));
      dv_do_dz_.emplace(sample);
      omega_z_.emplace(sample);
      xi_nl_.emplace(make_Interp2D(sample, "xi_nl", "r", "z", "xi_nl"));
      chi_.emplace(make_Interp1D(sample, "distances", "z", "d_c"));
      if (sample.has_val("average_sigma_crit_inv", "zlense")) {
        sci_.emplace(make_Interp1D(sample, "average_sigma_crit_inv",
                                   "zlense", "sci_average"));
      } else {
        sci_.reset();
      }
      sigma_z_.emplace(Interp1D(y3_cluster::z_kernel_z(),
                                y3_cluster::z_kernel_sigma()));

      double const omm =
          sample.view<double>("cosmological_parameters", "omega_M");
      double const omn =
          sample.view<double>("cosmological_parameters", "omega_nu");
      // HMF_t already rescales its spline axis by (omm - omn) internally
      // (hmf_t.hh::_adjust_to_log), so the shift here must stay zero.
      ln_mass_shift_ = 0.0;
      h0_ = sample.view<double>("cosmological_parameters", "h0");

      // NFW rho_mean normalisation (see ShearPrjEvaluator).
      sigma_mis_ ->set_rho_mult(omm);
      dsigma_mis_->set_rho_mult(omm);

      chi_ref_.resize(N_zt_ref_);
      dv_ref_.resize(N_zt_ref_);
      om_ref_.resize(N_zt_ref_);
      sig_ref_.resize(N_zt_ref_);
      for (std::size_t iz = 0; iz != N_zt_ref_; ++iz) {
        double const z = zt_ref_[iz];
        chi_ref_[iz] = (*chi_).clamp(z) * h0_;
        dv_ref_[iz]  = (*dv_do_dz_)(z);
        om_ref_[iz]  = (*omega_z_)(z);
        sig_ref_[iz] = (*sigma_z_).clamp(z);
      }
      hmf_ref_.assign(N_zt_ref_ * N_lnm_, 0.0);
      hmb_ref_.assign(N_zt_ref_ * N_lnm_, 0.0);
      for (std::size_t iz = 0; iz != N_zt_ref_; ++iz) {
        double const z = zt_ref_[iz];
        double* nrow = &hmf_ref_[iz * N_lnm_];
        double* brow = &hmb_ref_[iz * N_lnm_];
        for (std::size_t iM = 0; iM != N_lnm_; ++iM) {
          nrow[iM] = (*hmf_)(lnm_x_[iM] + ln_mass_shift_, z);
          brow[iM] = hmb_->clamp(lnm_x_[iM], z);
        }
      }

      bsel_.emplace(sample);

      // Per-slice caches.
      std::size_t const Nlz = lzob_lb_.size();
      lzob_chi_o_ .assign(Nlz, 0.0);
      lzob_R_excl_.assign(Nlz, 0.0);
      lzob_sci_   .assign(Nlz, 0.0);
      lzob_theta_    .assign(Nlz, {});
      lzob_sin_theta_.assign(Nlz, {});
      lzob_cos_theta_.assign(Nlz, {});
      lzob_geom_     .assign(Nlz, {});
      lzob_bsel_     .assign(Nlz, {});
      for (std::size_t k = 0; k != Nlz; ++k) {
        int    const lb  = lzob_lb_[k];
        double const zob = lzob_zob_[k];
        auto const bsel_bin = bsel_->at(lb, zob);
        double const chi_o = (*chi_).clamp(zob) * h0_;
        double const D_A_o = chi_o / (1.0 + zob);
        double const lobc = bsel_bin.lob;
        double const R_excl = sp_detail::R_lambda(lobc) * (1.0 + zob);
        lzob_chi_o_[k]  = chi_o;
        lzob_R_excl_[k] = R_excl;
        lzob_sci_[k]    = sci_ ? (*sci_).clamp(zob) : 0.0;

        auto const tg = sp_detail::build_theta_grid(lobc, zob, lzob_Rs_[k],
                                                    chi_o, D_A_o, R_excl,
                                                    N_per_seg_,
                                                    R_max_cMpch_,
                                                    extra_breakpoints_);
        std::size_t const Nth = tg.theta.size();
        auto& th_k = lzob_theta_[k];      th_k = tg.theta;
        auto& s_k  = lzob_sin_theta_[k];  s_k.resize(Nth);
        auto& c_k  = lzob_cos_theta_[k];  c_k.resize(Nth);
        auto& g_k  = lzob_geom_[k];       g_k.resize(Nth);
        for (std::size_t it = 0; it != Nth; ++it) {
          s_k[it] = std::sin(th_k[it]);
          c_k[it] = std::cos(th_k[it]);
          g_k[it] = tg.weight[it] * 2.0 * sp_detail::PI * s_k[it];
        }

        auto& bsel_k = lzob_bsel_[k]; bsel_k.resize(Nth);
        double const theta_lam = sp_detail::R_lambda(lobc)
                                 * (1.0 + zob) / chi_o;
        double const k_sig = 2.5 / theta_lam;
        double const theta0 = 0.5 * theta_lam;
        for (std::size_t it = 0; it != Nth; ++it) {
          double const sgm =
              1.0 / (1.0 + std::exp(-k_sig * (th_k[it] - theta0)));
          bsel_k[it] = bsel_bin.b_small
                     + (bsel_bin.b_large - bsel_bin.b_small) * sgm;
        }
      }
    }

    // Interpolate reference tables on-the-fly at an arbitrary z.
    // Returns (chi, dV, om, sig).
    struct ZFactors { double chi; double dV; double om; double sig; };
    ZFactors interp_z_factors(double z) const {
      double const u = (z - zt_lo_) * inv_dz_ref_;
      std::size_t const i = std::min<std::size_t>(
          std::max<std::size_t>(0, std::size_t(u)), N_zt_ref_ - 2);
      double const f = u - double(i);
      return {
        chi_ref_[i] + f * (chi_ref_[i+1] - chi_ref_[i]),
        dv_ref_ [i] + f * (dv_ref_ [i+1] - dv_ref_ [i]),
        om_ref_ [i] + f * (om_ref_ [i+1] - om_ref_ [i]),
        sig_ref_[i] + f * (sig_ref_[i+1] - sig_ref_[i])
      };
    }

    // Linear-in-lnM lookup of HMF/HMB onto the reference lnM grid at
    // arbitrary z (linear interp in z between reference slices).
    // lnm_x_[0..N_lnm_-1] are GL nodes -- use linear interp on (lnM, z).
    std::pair<double, double>
    interp_hmf_hmb(double lnM, double z) const {
      // Find lnM bracket in lnm_x_ (monotone increasing, 24 nodes).
      std::size_t jM = 0;
      for (; jM + 1 < N_lnm_; ++jM) {
        if (lnm_x_[jM + 1] > lnM) break;
      }
      if (jM + 1 >= N_lnm_) jM = N_lnm_ - 2;
      double const fM = (lnM - lnm_x_[jM]) / (lnm_x_[jM+1] - lnm_x_[jM] + 1.0e-30);

      double const u = (z - zt_lo_) * inv_dz_ref_;
      std::size_t const iz = std::min<std::size_t>(
          std::max<std::size_t>(0, std::size_t(u)), N_zt_ref_ - 2);
      double const fz = u - double(iz);

      auto at = [&](std::size_t iz_, std::size_t jM_) {
        return std::pair<double,double>{
          hmf_ref_[iz_ * N_lnm_ + jM_],
          hmb_ref_[iz_ * N_lnm_ + jM_]
        };
      };
      auto [n00, b00] = at(iz,     jM);
      auto [n01, b01] = at(iz,     jM + 1);
      auto [n10, b10] = at(iz + 1, jM);
      auto [n11, b11] = at(iz + 1, jM + 1);
      double const nM = (1.0 - fz) * ((1.0 - fM) * n00 + fM * n01)
                         + fz      * ((1.0 - fM) * n10 + fM * n11);
      double const bM = (1.0 - fz) * ((1.0 - fM) * b00 + fM * b01)
                         + fz      * ((1.0 - fM) * b10 + fM * b11);
      return {nM, bM};
    }

    // Adaptive 2-D inner integral at fixed (theta, R, lob_bin, zob).
    // Returns (I_rnd, I_cl), where the cl piece already multiplies by
    // the mass-loop's bM * xi_NL * 1[theta > theta_excl(z)] but NOT by
    // b_sel(theta) (that is applied in the outer theta loop, matching
    // the Python sigma_prj.py convention).
    // The inner integrand is
    //   rnd_z_M = dV(z) * om(z) * wz(z, zob) * n(M,z) * Sigma_mis(R|M,z,theta*D_A_o)
    //   cl_z_M  = rnd_z_M * b(M,z) * xi_NL(|dr|, zob) * 1[theta > theta_excl(z)]
    // integrated over (z, lnM).
    std::array<double, 4>
    inner_2d_integral(double theta, double chi_o, double D_A_o,
                      double R_excl, double zob, double R, double cos_th) const
    {
      auto integrand = [&](double z, double lnM) -> std::vector<double> {
        auto const zf = interp_z_factors(z);
        double const u_phot = (z - zob) / (zf.sig + 1.0e-30);
        double const wz     = (std::abs(u_phot) < 1.0)
                              ? (1.0 - u_phot * u_phot) : 0.0;
        double const theta_excl_z = sp_detail::theta_excl_at_z(zf.chi,
                                                                chi_o,
                                                                R_excl);
        double const dchi = std::sqrt(std::max(
            zf.chi * zf.chi + chi_o * chi_o - 2.0 * zf.chi * chi_o * cos_th,
            0.0));
        double xi_val = 0.0;
        if (theta > theta_excl_z) {
          xi_val = (*xi_nl_).clamp(dchi, zob);
        }

        auto [nM, bM] = interp_hmf_hmb(lnM, z);
        double const R_th = theta * D_A_o;
        double const Smis = (*sigma_mis_)(R, R_th, lnM);
        double const DSmis = (*dsigma_mis_)(R, R_th, lnM);

        double const base = zf.dV * zf.om * wz * nM;
        double const cl_w = bM * xi_val;
        // Output: {sigma_rnd, sigma_cl, dsigma_rnd, dsigma_cl}
        return {
          base * Smis,
          base * cl_w * Smis,
          base * DSmis,
          base * cl_w * DSmis
        };
      };

      using iv_t = cubacpp::IntegrationVolume<2>;
      iv_t vol{{zt_lo_, lnm_lo_}, {zt_hi_, lnm_hi_}};
      std::array<double, 4> out{0.0, 0.0, 0.0, 0.0};
      if (algorithm_ == "vegas") {
        cubacpp::Vegas alg;
        alg.maxeval = max_eval_;
        auto res = alg.integrate(integrand, eps_rel_, eps_abs_, vol);
        for (std::size_t k = 0; k != 4; ++k)
          out[k] = (k < res.value.size()) ? res.value[k] : 0.0;
      } else {
        cubacpp::Cuhre alg;
        alg.maxeval = max_eval_;
        auto res = alg.integrate(integrand, eps_rel_, eps_abs_, vol);
        for (std::size_t k = 0; k != 4; ++k)
          out[k] = (k < res.value.size()) ? res.value[k] : 0.0;
      }
      return out;
    }

    std::array<double, 9>
    evaluate(grid_point_t const& pt) const
    {
      int    const lob_bin = static_cast<int>(pt[0]);
      double const zob     = 0.5 * (pt[1] + pt[2]);
      double const R       = pt[3];

      int lzob_idx = -1;
      for (std::size_t k = 0; k != lzob_lb_.size(); ++k) {
        if (lzob_lb_[k] == lob_bin &&
            std::abs(lzob_zob_[k] - zob) < 1e-12) { lzob_idx = int(k); break; }
      }
      double const chi_o  = lzob_chi_o_ [lzob_idx];
      double const R_excl = lzob_R_excl_[lzob_idx];
      double const sci_v  = lzob_sci_   [lzob_idx];
      double const D_A_o  = chi_o / (1.0 + zob);
      auto const& th_k    = lzob_theta_    [lzob_idx];
      auto const& cos_k   = lzob_cos_theta_[lzob_idx];
      auto const& geom_k  = lzob_geom_     [lzob_idx];
      auto const& bsel_k  = lzob_bsel_     [lzob_idx];
      std::size_t const Nth = th_k.size();

      double sigma_rnd = 0.0, sigma_cl = 0.0;
      double dsigma_rnd = 0.0, dsigma_cl = 0.0;
      for (std::size_t it = 0; it != Nth; ++it) {
        auto const inner = inner_2d_integral(th_k[it], chi_o, D_A_o,
                                              R_excl, zob, R, cos_k[it]);
        double const g = geom_k[it];
        double const b = bsel_k[it];
        sigma_rnd  += g * inner[0];
        sigma_cl   += g * b * inner[1];
        dsigma_rnd += g * inner[2];
        dsigma_cl  += g * b * inner[3];
      }

      double const sigma_total  = sigma_rnd  + sigma_cl;
      double const dsigma_total = dsigma_rnd + dsigma_cl;
      // γ_t = ΔΣ · Σ_crit^-1 (linear; see note in ShearPrjEvaluator).
      auto g_t = [sci_v](double /*s*/, double ds) { return ds * sci_v; };
      return {
        sigma_total,  sigma_rnd,  sigma_cl,
        dsigma_total, dsigma_rnd, dsigma_cl,
        g_t(sigma_total, dsigma_total),
        g_t(sigma_rnd,   dsigma_rnd),
        g_t(sigma_cl,    dsigma_cl)
      };
    }

    static char const* module_label() { return "shear_prj_cuhre"; }

    static std::array<char const*, 9>
    output_sections()
    {
      return {
        "sigma_prj_cuhre",  "sigma_prj_cuhre",  "sigma_prj_cuhre",
        "dsigma_prj_cuhre", "dsigma_prj_cuhre", "dsigma_prj_cuhre",
        "shear_prj_cuhre",  "shear_prj_cuhre",  "shear_prj_cuhre"
      };
    }

    static std::array<char const*, 9>
    output_names()
    {
      return {
        "vals", "rnd", "cl",
        "vals", "rnd", "cl",
        "vals", "rnd", "cl"
      };
    }

    static grid_t
    make_grid_points(cosmosis::DataBlock& cfg)
    {
      return y3_cluster::make_grid_points_wall_of_numbers(
          cfg, module_label(),
          "lambda_bin", "zo_low", "zo_high", "radii");
    }

   private:
    std::size_t N_lnm_, N_per_seg_, N_zt_ref_;
    double eps_rel_, eps_abs_;
    std::size_t max_eval_;
    std::string algorithm_;
    double zt_lo_, zt_hi_;
    double lnm_lo_, lnm_hi_;
    double R_max_cMpch_;
    std::vector<double> extra_breakpoints_;
    std::vector<double> lob_centers_;

    std::vector<double> lnm_x_, lnm_w_;
    std::vector<double> zt_ref_;
    double inv_dz_ref_ = 0.0;

    std::optional<y3_cluster::HMF_t>       hmf_;
    std::optional<Interp2D>                hmb_;
    std::optional<y3_cluster::DV_DO_DZ_t>  dv_do_dz_;
    std::optional<y3_cluster::OMEGA_Z_DES> omega_z_;
    std::optional<Interp2D>                xi_nl_;
    std::optional<NFW_SIGMA_MIS>           sigma_mis_;
    std::optional<NFW_DSIGMA_MIS>          dsigma_mis_;
    std::optional<Interp1D>                chi_;
    std::optional<Interp1D>                sci_;
    std::optional<Interp1D>                sigma_z_;

    std::vector<double> chi_ref_, dv_ref_, om_ref_, sig_ref_;
    std::vector<double> hmf_ref_, hmb_ref_;
    std::optional<sp_detail::BSelBins> bsel_;

    std::vector<int>    gp_lam_bin_;
    std::vector<double> gp_zob_, gp_R_;

    std::vector<int>    lzob_lb_;
    std::vector<double> lzob_zob_;
    std::vector<int>    gp_lzob_idx_;
    std::vector<double> lzob_chi_o_, lzob_R_excl_, lzob_sci_;
    std::vector<std::vector<double>> lzob_Rs_;

    std::vector<std::vector<double>> lzob_theta_;
    std::vector<std::vector<double>> lzob_sin_theta_;
    std::vector<std::vector<double>> lzob_cos_theta_;
    std::vector<std::vector<double>> lzob_geom_;
    std::vector<std::vector<double>> lzob_bsel_;

    double ln_mass_shift_ = 0.0;
    double h0_            = 1.0;
  };

}  // namespace y3_cluster

#endif

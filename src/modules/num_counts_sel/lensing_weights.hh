// Shared weight functors for the radial lensing N-operator drivers.
//
// All lensing profiles are read from the "haloModel" section populated by
// y3_buzzard/haloModelCosmosis.py (see memory/halo_model_cosmosis.md):
//
//   Sigma_nfw (r_sigma, lnM)    -- 1-halo projected surface density
//   Sigma_hh  (r_sigma, z)      -- 2-halo projected surface density (no bias)
//   dSigma_nfw(r_sigma, lnM)    -- 1-halo differential
//   dSigma_hh (r_sigma, z)      -- 2-halo differential (no bias)
//   bias      (lnM,     z)      -- halo bias b(M, z)
//
// Sigma_crit_inv(z_lens) is a source-bin-weighted 1-D function read from
// section "average_sigma_crit_inv" (populated by
// src/modules/average_sigma_crit_inv/average_sigma_crit_inv.py, axes
// "zlense" / "sci_average"). Σ_cr^-1 has no intrinsic R dependence;
// any R dependence in a full weighted pipeline comes from a per-R source
// selection, which the existing smoke module does not model.
//
// Total 1h+2h surface densities (used by the _tot variants):
//   Sigma_tot (R, M, z) = Sigma_nfw(R, M)  + b(M, z) * Sigma_hh (R, z)
//   dSigma_tot(R, M, z) = dSigma_nfw(R, M) + b(M, z) * dSigma_hh(R, z)
//
// Shear uses Sigma_crit_inv to rescale:
//   gamma_t = dSigma * Sigma_crit_inv(z_lens)
//
// The old reduced-shear form g_t = gamma_t / (1 - Sigma * Sigma_crit_inv)
// was retired 2026-05-11: its 1/(1 - x) denominator breaks the additive
// decomposition gamma_t^total = gamma_t^1h + gamma_t^prj that the
// Costanzi-2026 projection observable requires.
//
// Each weight functor exposes:
//   static char const* module_label();
//   explicit Weight(cosmosis::DataBlock&);
//   void set_sample(cosmosis::DataBlock&);
//   double operator()(double R, double lnM, double zt) const;
#ifndef Y3_CLUSTER_CPP_NUM_COUNTS_SEL_LENSING_WEIGHTS_HH
#define Y3_CLUSTER_CPP_NUM_COUNTS_SEL_LENSING_WEIGHTS_HH

#include "cosmosis/datablock/datablock.hh"
#include "models/nfw_dsigma_mis.hh"
#include "utils/datablock_reader.hh"
#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"
#include "utils/make_interp_1d.hh"
#include "utils/make_interp_2d.hh"

#include <cmath>
#include <optional>
#include <vector>

namespace y3_cluster_sel_weights {

  // Helper — load Sigma_crit_inv^{-1}(z_lens) 1-D table.
  inline y3_cluster::Interp1D
  load_sigma_crit_inv(cosmosis::DataBlock& s)
  {
    return y3_cluster::make_Interp1D(s, "average_sigma_crit_inv",
                                     "zlense", "sci_average");
  }

  // ---- 1-halo surface density ------------------------------------------------
  struct Sigma1hWeight {
    static char const* module_label() { return "Sigma1hSel"; }
    explicit Sigma1hWeight(cosmosis::DataBlock&) {}
    void
    set_sample(cosmosis::DataBlock& s)
    {
      sigma_nfw_.emplace(
        y3_cluster::make_Interp2D(s, "haloModel", "r_sigma", "lnM", "Sigma_nfw"));
    }
    double
    operator()(double R, double lnM, double /*zt*/) const
    {
      return sigma_nfw_->clamp(R, lnM);
    }
  private:
    std::optional<y3_cluster::Interp2D> sigma_nfw_;
  };


  // ---- Total (1h + b·2h) surface density ------------------------------------
  struct SigmaTotWeight {
    static char const* module_label() { return "SigmaTotSel"; }
    explicit SigmaTotWeight(cosmosis::DataBlock&) {}
    void
    set_sample(cosmosis::DataBlock& s)
    {
      sigma_nfw_.emplace(
        y3_cluster::make_Interp2D(s, "haloModel", "r_sigma", "lnM", "Sigma_nfw"));
      sigma_hh_.emplace(
        y3_cluster::make_Interp2D(s, "haloModel", "r_sigma", "z",   "Sigma_hh"));
      bias_.emplace(
        y3_cluster::make_Interp2D(s, "haloModel", "lnM",     "z",   "bias"));
    }
    double
    operator()(double R, double lnM, double zt) const
    {
      double const b  = bias_->clamp(lnM, zt);
      double const s1 = sigma_nfw_->clamp(R, lnM);
      double const s2 = sigma_hh_ ->clamp(R, zt);
      return s1 + b * s2;
    }
  private:
    std::optional<y3_cluster::Interp2D> sigma_nfw_;
    std::optional<y3_cluster::Interp2D> sigma_hh_;
    std::optional<y3_cluster::Interp2D> bias_;
  };


  // ---- 1-halo differential ΔΣ -----------------------------------------------
  struct DSigma1hWeight {
    static char const* module_label() { return "DSigma1hSel"; }
    explicit DSigma1hWeight(cosmosis::DataBlock&) {}
    void
    set_sample(cosmosis::DataBlock& s)
    {
      dsigma_nfw_.emplace(
        y3_cluster::make_Interp2D(s, "haloModel", "r_sigma", "lnM", "dSigma_nfw"));
    }
    double
    operator()(double R, double lnM, double /*zt*/) const
    {
      return dsigma_nfw_->clamp(R, lnM);
    }
  private:
    std::optional<y3_cluster::Interp2D> dsigma_nfw_;
  };


  // ---- Miscentering helpers shared by DSigma1hMis / Shear1hMis -------------
  //
  // DES-Y3 redMaPPer cluster centring is imperfect: a fraction f_mis of
  // clusters in the sample are offset from the true halo centre by a 2-D
  // radial distance R_mis drawn from a Rayleigh-shaped kernel
  //
  //     P_gamma(R_mis | tau_mis, R_lambda) = (R_mis / (tau_mis R_lambda)^2)
  //                                           * exp(-R_mis / (tau_mis R_lambda))
  //
  // (Kelly et al. 2023 / Costanzi-2026 fit).  The lensing observable is
  // therefore a two-component mixture of the centred NFW profile and the
  // offset-averaged miscentered convolution:
  //
  //     DSigma_cl(R | M, z) = (1 - f_mis) * DSigma_NFW(R, M)
  //                         + f_mis      * DSigma_mis(R, M; tau_mis * R_lambda)
  //
  // where R_lambda(lambda^ob) = (lambda^ob / 100)^0.2 [h^-1 Mpc] is the
  // richness-derived cluster aperture.  DSigma_mis is read from the
  // pre-computed look-up tables under data/nfw_off_center/ via
  // NFW_DSIGMA_MIS, the same machinery red_shear_prj uses for the 2-halo
  // projection-effect integrand.  The lensing branch picks the gamma
  // (Rayleigh) kernel; red_shear_prj uses the single (delta-function)
  // kernel because its theta integral already plays the role of the
  // R_mis integration.
  //
  // R_lambda(lambda^ob) is bin-dependent.  The N_i[f] template now
  // forwards the current lambda-bin index to the weight via the optional
  // set_bin(int) hook (see src/models/n_operator_sel_t.hh); the weight
  // resolves it to lambda^ob through a lob_centers vector loaded once
  // per sample (default DES-Y3 arithmetic centres {25, 37.5, 52.5, 130};
  // ini-overridable via "lob_centers" on the module section).
  //
  // Defaults: f_mis = 0.22, tau_mis = 0.17 (DES-Y3 fiducial calibration
  // jointly against X-ray and SZ-derived true centres).  Both knobs are
  // read from the "miscentering" datablock section if present, else fall
  // back to the defaults so chains that have not yet wired the section
  // continue to run.
  namespace mis_detail {
    inline constexpr double F_MIS_DEFAULT   = 0.22;
    inline constexpr double TAU_MIS_DEFAULT = 0.17;

    // R_lambda(lambda^ob) = (lambda^ob / 100)^0.2  [h^-1 Mpc]
    inline double R_lambda(double lob) { return std::pow(lob / 100.0, 0.2); }

    // Default DES-Y3 richness-bin centres (edges [20, 30, 45, 60, 200]).
    inline std::vector<double> default_lob_centers()
    {
      return {25.0, 37.5, 52.5, 130.0};
    }

    // Read scalar param from "miscentering" with a default fallback.
    inline double
    read_mis_param(cosmosis::DataBlock& s, char const* key, double dflt)
    {
      double out = dflt;
      auto const status = s.get_val<double>("miscentering", key, out);
      if (status != DBS_SUCCESS) return dflt;
      return out;
    }

    // Load lob_centers from the module section if present, else default.
    inline std::vector<double>
    read_lob_centers(cosmosis::DataBlock& s, char const* module_section)
    {
      if (s.has_val(module_section, "lob_centers"))
        return get_vector_double(s, module_section, "lob_centers");
      return default_lob_centers();
    }
  }  // namespace mis_detail


  // ---- Miscentered 1-halo ΔΣ_cl(R, M, z) ----------------------------------
  //
  //   ΔΣ_cl = (1 - f_mis) ΔΣ_NFW(R, M) + f_mis ΔΣ_mis(R, M; tau_mis R_λ)
  //
  // R_lambda is set per (lob_bin, R) wall point through set_bin.
  struct DSigma1hMisWeight {
    static char const* module_label() { return "DSigma1hMisSel"; }
    explicit DSigma1hMisWeight(cosmosis::DataBlock&)
      // Construct the miscentered NFW table once at module load (3
      // ~1000x1000 ASCII tables read from data/nfw_off_center/), the
      // same convention the projection evaluator uses.  The kernel
      // ("gamma" -> Kelly et al. 2023 Rayleigh) is fixed; only the
      // (f_mis, tau_mis) mixture weights change per sample.
      : dsigma_mis_(4.0, 2.77533742639e+11, y3_cluster::GAMMA)
    {}
    void
    set_sample(cosmosis::DataBlock& s)
    {
      dsigma_nfw_.emplace(
        y3_cluster::make_Interp2D(s, "haloModel", "r_sigma", "lnM", "dSigma_nfw"));
      lob_centers_ = mis_detail::read_lob_centers(s, module_label());
      f_mis_   = mis_detail::read_mis_param(s, "f_mis",
                                            mis_detail::F_MIS_DEFAULT);
      tau_mis_ = mis_detail::read_mis_param(s, "tau_mis",
                                            mis_detail::TAU_MIS_DEFAULT);
      // rho_mean normalisation (matches the Python reference's
      // richness_selection.nfw.NFWMiscentered).
      double const omm =
          s.view<double>("cosmological_parameters", "omega_M");
      dsigma_mis_.set_rho_mult(omm);
      // Feed haloModel/concentration (Child18 x concentration_amplitude) into
      // the miscentered NFW, so the 1h miscentered part matches the centered 1h
      // (dSigma_nfw table) and the 2-halo (ShearPrjCore). Without this it stays
      // at the fixed c=4 and ignores the concentration (and c_amp) entirely.
      if (s.has_val("haloModel", "concentration"))
        dsigma_mis_.set_concentration_table(
            y3_cluster::make_Interp1D(s, "haloModel", "lnM", "concentration"));
      // Stay in a defined state until set_bin is called by the
      // operator template; first integrand evaluation will overwrite.
      current_R_lambda_ = mis_detail::R_lambda(
          lob_centers_.empty() ? 25.0 : lob_centers_.front());
    }
    // Forwarded by the N-operator template every time the wall-grid
    // visits a new (lambda_bin, ...) slice; no-op for out-of-range
    // indices so an ill-formed wall-grid does not abort inside Cuhre.
    void
    set_bin(int lob_bin)
    {
      if (lob_bin < 0 || static_cast<std::size_t>(lob_bin) >= lob_centers_.size())
        return;
      current_R_lambda_ = mis_detail::R_lambda(lob_centers_[lob_bin]);
    }
    double
    operator()(double R, double lnM, double /*zt*/) const
    {
      double const d_cen = dsigma_nfw_->clamp(R, lnM);
      double const r_mis = tau_mis_ * current_R_lambda_;
      double const d_mis = dsigma_mis_(R, r_mis, lnM);
      return (1.0 - f_mis_) * d_cen + f_mis_ * d_mis;
    }
  private:
    std::optional<y3_cluster::Interp2D> dsigma_nfw_;
    y3_cluster::NFW_DSIGMA_MIS          dsigma_mis_;
    std::vector<double>                 lob_centers_{};
    double f_mis_  {mis_detail::F_MIS_DEFAULT};
    double tau_mis_{mis_detail::TAU_MIS_DEFAULT};
    double current_R_lambda_{mis_detail::R_lambda(25.0)};
  };


  // ---- Total (1h + b·2h) differential ΔΣ -----------------------------------
  struct DSigmaTotWeight {
    static char const* module_label() { return "DSigmaTotSel"; }
    explicit DSigmaTotWeight(cosmosis::DataBlock&) {}
    void
    set_sample(cosmosis::DataBlock& s)
    {
      dsigma_nfw_.emplace(
        y3_cluster::make_Interp2D(s, "haloModel", "r_sigma", "lnM", "dSigma_nfw"));
      dsigma_hh_.emplace(
        y3_cluster::make_Interp2D(s, "haloModel", "r_sigma", "z",   "dSigma_hh"));
      bias_.emplace(
        y3_cluster::make_Interp2D(s, "haloModel", "lnM",     "z",   "bias"));
    }
    double
    operator()(double R, double lnM, double zt) const
    {
      double const b  = bias_->clamp(lnM, zt);
      double const d1 = dsigma_nfw_->clamp(R, lnM);
      double const d2 = dsigma_hh_ ->clamp(R, zt);
      return d1 + b * d2;
    }
  private:
    std::optional<y3_cluster::Interp2D> dsigma_nfw_;
    std::optional<y3_cluster::Interp2D> dsigma_hh_;
    std::optional<y3_cluster::Interp2D> bias_;
  };


  // ---- 1-halo shear γ_t = ΔΣ_nfw · Σ_crit^-1(z_lens) -----------------------
  struct Shear1hWeight {
    static char const* module_label() { return "Shear1hSel"; }
    explicit Shear1hWeight(cosmosis::DataBlock&) {}
    void
    set_sample(cosmosis::DataBlock& s)
    {
      dsigma_nfw_.emplace(
        y3_cluster::make_Interp2D(s, "haloModel", "r_sigma", "lnM", "dSigma_nfw"));
      sigma_crit_inv_.emplace(load_sigma_crit_inv(s));
    }
    double
    operator()(double R, double lnM, double zt) const
    {
      double const d = dsigma_nfw_->clamp(R, lnM);
      double const sci = sigma_crit_inv_->clamp(zt);
      return d * sci;
    }
  private:
    std::optional<y3_cluster::Interp2D> dsigma_nfw_;
    std::optional<y3_cluster::Interp1D> sigma_crit_inv_;
  };


  // ---- Miscentered 1-halo shear γ_t^mis = ΔΣ_cl · Σ_crit^-1(z_lens) ---------
  //
  //   ΔΣ_cl = (1 - f_mis) ΔΣ_NFW(R, M) + f_mis ΔΣ_mis(R, M; tau_mis R_λ)
  //
  // Same mixture as DSigma1hMisWeight, scaled by Σ_crit^-1(z_lens) so the
  // ratio Shear1hMisSel / NumCountsSel gives the per-cluster
  // <γ_t^1h, full>(R) including miscentering.
  struct Shear1hMisWeight {
    static char const* module_label() { return "Shear1hMisSel"; }
    explicit Shear1hMisWeight(cosmosis::DataBlock&)
      : dsigma_mis_(4.0, 2.77533742639e+11, y3_cluster::GAMMA)
    {}
    void
    set_sample(cosmosis::DataBlock& s)
    {
      dsigma_nfw_.emplace(
        y3_cluster::make_Interp2D(s, "haloModel", "r_sigma", "lnM", "dSigma_nfw"));
      sigma_crit_inv_.emplace(load_sigma_crit_inv(s));
      lob_centers_ = mis_detail::read_lob_centers(s, module_label());
      f_mis_   = mis_detail::read_mis_param(s, "f_mis",
                                            mis_detail::F_MIS_DEFAULT);
      tau_mis_ = mis_detail::read_mis_param(s, "tau_mis",
                                            mis_detail::TAU_MIS_DEFAULT);
      double const omm =
          s.view<double>("cosmological_parameters", "omega_M");
      dsigma_mis_.set_rho_mult(omm);
      // See DSigma1hMisWeight: feed haloModel/concentration into the
      // miscentered NFW so it matches the centered 1h and the 2-halo,
      // instead of the fixed c=4.
      if (s.has_val("haloModel", "concentration"))
        dsigma_mis_.set_concentration_table(
            y3_cluster::make_Interp1D(s, "haloModel", "lnM", "concentration"));
      current_R_lambda_ = mis_detail::R_lambda(
          lob_centers_.empty() ? 25.0 : lob_centers_.front());
    }
    void
    set_bin(int lob_bin)
    {
      if (lob_bin < 0 || static_cast<std::size_t>(lob_bin) >= lob_centers_.size())
        return;
      current_R_lambda_ = mis_detail::R_lambda(lob_centers_[lob_bin]);
    }
    double
    operator()(double R, double lnM, double zt) const
    {
      double const d_cen = dsigma_nfw_->clamp(R, lnM);
      double const r_mis = tau_mis_ * current_R_lambda_;
      double const d_mis = dsigma_mis_(R, r_mis, lnM);
      double const d     = (1.0 - f_mis_) * d_cen + f_mis_ * d_mis;
      double const sci   = sigma_crit_inv_->clamp(zt);
      return d * sci;
    }
  private:
    std::optional<y3_cluster::Interp2D> dsigma_nfw_;
    std::optional<y3_cluster::Interp1D> sigma_crit_inv_;
    y3_cluster::NFW_DSIGMA_MIS          dsigma_mis_;
    std::vector<double>                 lob_centers_{};
    double f_mis_  {mis_detail::F_MIS_DEFAULT};
    double tau_mis_{mis_detail::TAU_MIS_DEFAULT};
    double current_R_lambda_{mis_detail::R_lambda(25.0)};
  };


  // ---- Total shear γ_t = ΔΣ_tot · Σ_crit^-1(z_lens) ------------------------
  struct ShearTotWeight {
    static char const* module_label() { return "ShearTotSel"; }
    explicit ShearTotWeight(cosmosis::DataBlock&) {}
    void
    set_sample(cosmosis::DataBlock& s)
    {
      dsigma_nfw_.emplace(
        y3_cluster::make_Interp2D(s, "haloModel", "r_sigma", "lnM", "dSigma_nfw"));
      dsigma_hh_.emplace(
        y3_cluster::make_Interp2D(s, "haloModel", "r_sigma", "z",   "dSigma_hh"));
      bias_.emplace(
        y3_cluster::make_Interp2D(s, "haloModel", "lnM",     "z",   "bias"));
      sigma_crit_inv_.emplace(load_sigma_crit_inv(s));
    }
    double
    operator()(double R, double lnM, double zt) const
    {
      double const b = bias_->clamp(lnM, zt);
      double const d = dsigma_nfw_->clamp(R, lnM) +
                       b * dsigma_hh_->clamp(R, zt);
      double const sci = sigma_crit_inv_->clamp(zt);
      return d * sci;
    }
  private:
    std::optional<y3_cluster::Interp2D> dsigma_nfw_;
    std::optional<y3_cluster::Interp2D> dsigma_hh_;
    std::optional<y3_cluster::Interp2D> bias_;
    std::optional<y3_cluster::Interp1D> sigma_crit_inv_;
  };


  // ReducedShear{1h,Tot}Weight structs retired 2026-05-11.  We now use
  // the linear γ_t = ΔΣ · Σ_crit^-1 form (Shear{1h,Tot}Weight above);
  // the 1h + prj observable is assembled additively in the likelihood,
  // which was not valid under the old 1/(1 - ΣΣ_crit^-1) denominator.

} // namespace y3_cluster_sel_weights

#endif

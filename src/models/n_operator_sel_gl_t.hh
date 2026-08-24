// Fixed-GL N-operator evaluators -- the z-marginalised replacements for
// the Cuhre-based NumCountsSel and Shear1hMisSel modules.
//
// Both observables share the structure (docs/shear1h_radial_factorization.tex)
//
//   N_ij[f]   = int dlnM  Wij(lnM) * f(...)
//   Wij(lnM)  = int dz  n(M,z) dV/dOmega/dz(z) Omega(z) S_ij(lnM,z)
//               [ * Sigma_crit_inv(z), shear only ]
//
// where every z-dependent factor lives in Wij and the remaining factor f
// is z-free: f = 1 for number counts, f = Phi_i(R, lnM) (the miscentered
// 1-halo profile, fixed concentration and rho_crit) for shear.  Building
// Wij once per sample on fixed Gauss-Legendre nodes removes the entire
// per-grid-point adaptive quadrature the old modules ran: NumCountsSel
// one 2-D Cuhre integral per bin (12 per sample), Shear1hMisSel one per
// (bin, radius) pair (120 per sample).
//
// Beyond the mean speed-up, fixed GL makes the cost DETERMINISTIC.
// Adaptive Cuhre's runtime varies strongly with the sampled parameter
// point (measured over ~1e6 MCMC realisations: NumCountsSel mean
// 0.107 s but max 0.98 s, Shear1hMisSel mean 0.575 s but max 4.0 s);
// the GL evaluators do an identical node count at every sample.
//
// SelGLCore builds Wij (and its plain lnM moments) on the GL grid;
// NumCountsSelGL / Shear1hMisSelGL wrap it behind the
// DEFINE_COSMOSIS_SCALAR_EVALUATOR_MODULE contract with the SAME module
// labels, grid semantics (bin_index wall / bin_index x r_perp cartesian
// product, bin slow, R fast) and output sections (numcountssel/vals,
// shear1hmissel/vals) as the modules they replace, so production inis
// and y3_buzzard/likelihood_cp.py work unchanged.  The old adaptive-
// integrator knobs (algorithm, eps_rel, eps_abs, max_eval,
// use_cartesian_product) are simply ignored; the new optional knobs are
// n_lnm (default 96) and n_z (default 64 -- S_ij has compact z-support,
// ~0.15 wide inside the [zt_low, zt_high] window, and 32 nodes leave a
// ~2e-3 residual vs a 128-node reference where 64 reach ~4e-4).
//
// One deliberate physics change vs the old Shear1hMisSel: the richness
// bin for R_lambda is bin_index % lob_centers.size() (the sel_function
// bin layout is richness-fast, z-block-slow).  The old
// Shear1hMisWeight::set_bin was a silent no-op for bin_index >= 4,
// leaving bins 4-11 on bin 3's R_lambda(130); bins 4-11 therefore
// intentionally differ from the old module (up to ~2%).
#ifndef Y3_CLUSTER_CPP_N_OPERATOR_SEL_GL_T_HH
#define Y3_CLUSTER_CPP_N_OPERATOR_SEL_GL_T_HH

#include "cosmosis/datablock/datablock.hh"

#include "models/dv_do_dz_t.hh"
#include "models/hmf_t.hh"
#include "models/n_operator_sel_t.hh"
#include "models/nfw_dsigma_mis.hh"
#include "models/omega_z_des.hh"
#include "models/p_operator_cuhre_t.hh"
#include "models/sel_function_t.hh"
#include "modules/num_counts_sel/lensing_weights.hh"
#include "utils/datablock_reader.hh"
#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"
#include "utils/make_grid_points.hh"
#include "utils/make_interp_2d.hh"

#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace y3_cluster {

  namespace nosel_gl_detail {

    // Per-sample builder for the z-marginalised mass weight Wij(lnM) on
    // fixed GL nodes, plus its plain (amplitude-free) lnM moments.
    class SelGLCore {
     public:
      SelGLCore(cosmosis::DataBlock& cfg, char const* label)
        : N_lnm_(cfg.has_val(label, "n_lnm")
                   ? cfg.view<int>(label, "n_lnm") : 96)
        , N_z_(cfg.has_val(label, "n_z")
                 ? cfg.view<int>(label, "n_z") : 64)
        , zt_lo_(cfg.view<double>(label, "zt_low"))
        , zt_hi_(cfg.view<double>(label, "zt_high"))
        , lnm_lo_(cfg.view<double>(label, "lnm_low"))
        , lnm_hi_(cfg.view<double>(label, "lnm_high"))
      {
        p_op_detail::gl_nodes(lnm_lo_, lnm_hi_, N_lnm_, lnm_x_, lnm_w_);
        p_op_detail::gl_nodes(zt_lo_, zt_hi_, N_z_, z_x_, z_w_);
      }

      void
      build_weights(cosmosis::DataBlock& s, bool include_sci)
      {
        y3_cluster::HMF_t const       hmf(s);
        y3_cluster::DV_DO_DZ_t const  dv(s);
        y3_cluster::OMEGA_Z_DES const omega(s);
        std::optional<y3_cluster::Interp1D> sci;
        if (include_sci)
          sci.emplace(y3_cluster_sel_weights::load_sigma_crit_inv(s));

        // z-only factors, shared by every bin and mass node.
        std::vector<double> zfac(z_x_.size());
        for (std::size_t q = 0; q != z_x_.size(); ++q) {
          double const z = z_x_[q];
          zfac[q] = z_w_[q] * dv(z) * omega(z) *
                    (sci ? sci->clamp(z) : 1.0);
        }

        int const n_bins = nosel_detail::n_bins_from_block(s);
        W_.assign(n_bins, std::vector<double>(lnm_x_.size(), 0.0));
        norm_.assign(n_bins, 0.0);
        lnm_eff_.assign(n_bins, 0.0);
        mu2_.assign(n_bins, 0.0);

        for (int b = 0; b != n_bins; ++b) {
          SelFunction_t const sel(s, b);
          auto& Wb = W_[b];
          for (std::size_t k = 0; k != lnm_x_.size(); ++k) {
            double const lnM = lnm_x_[k];
            double acc = 0.0;
            for (std::size_t q = 0; q != z_x_.size(); ++q)
              acc += zfac[q] * hmf(lnM, z_x_[q]) * sel(lnM, z_x_[q]);
            Wb[k] = acc;
          }

          // Plain moments of lnM under Wij -- the pairing that makes
          // the moment expansion's linear term vanish (tex,
          // sec:approach2).  Cheap enough to always compute.
          double n0 = 0.0, n1 = 0.0;
          for (std::size_t k = 0; k != lnm_x_.size(); ++k) {
            n0 += lnm_w_[k] * Wb[k];
            n1 += lnm_w_[k] * Wb[k] * lnm_x_[k];
          }
          norm_[b]    = n0;
          lnm_eff_[b] = (n0 != 0.0) ? n1 / n0 : 0.5 * (lnm_lo_ + lnm_hi_);
          double m2 = 0.0;
          for (std::size_t k = 0; k != lnm_x_.size(); ++k) {
            double const d = lnm_x_[k] - lnm_eff_[b];
            m2 += lnm_w_[k] * Wb[k] * d * d;
          }
          mu2_[b] = (n0 != 0.0) ? m2 / n0 : 0.0;
        }
      }

      std::size_t
      n_bins() const
      {
        return W_.size();
      }
      std::vector<double> const&
      weights(int b) const
      {
        return W_[b];
      }
      double norm(int b) const { return norm_[b]; }
      double lnm_eff(int b) const { return lnm_eff_[b]; }
      double mu2(int b) const { return mu2_[b]; }
      std::vector<double> const& lnm_x() const { return lnm_x_; }
      std::vector<double> const& lnm_w() const { return lnm_w_; }

     private:
      std::size_t N_lnm_;
      std::size_t N_z_;
      double zt_lo_, zt_hi_, lnm_lo_, lnm_hi_;
      std::vector<double> lnm_x_, lnm_w_;
      std::vector<double> z_x_, z_w_;

      std::vector<std::vector<double>> W_;   // [bin][lnM node]
      std::vector<double> norm_;
      std::vector<double> lnm_eff_;
      std::vector<double> mu2_;
    };

  } // namespace nosel_gl_detail


  // ---- NumCountsSel: N_ij = int dlnM Wij(lnM) --------------------------
  class NumCountsSelGL {
   public:
    using grid_t       = y3_cluster::grid_t<1>;
    using grid_point_t = grid_t::value_type;
    static constexpr std::size_t n_outputs = 1;

    explicit NumCountsSelGL(cosmosis::DataBlock& cfg)
      : core_(cfg, module_label())
    {}

    void
    set_sample(cosmosis::DataBlock& s)
    {
      core_.build_weights(s, /*include_sci=*/false);
    }

    std::array<double, n_outputs>
    evaluate(grid_point_t const& pt) const
    {
      int const b = static_cast<int>(pt[0]);
      if (b < 0 || static_cast<std::size_t>(b) >= core_.n_bins())
        throw std::out_of_range(
          "NumCountsSelGL: bin_index outside sel_function/S_stack range");
      return {core_.norm(b)};
    }

    static char const* module_label() { return "NumCountsSel"; }

    // NOTE: hardcoded, deliberately NOT an ini knob -- CosmoSIS [DEFAULT]
    // blocks propagate keys like output_section into every module
    // section, which would silently redirect the write.
    static std::array<char const*, n_outputs>
    output_sections()
    {
      return {"numcountssel"};
    }

    static grid_t
    make_grid_points(cosmosis::DataBlock& cfg)
    {
      return y3_cluster::make_grid_points_wall_of_numbers(
        cfg, module_label(), "bin_index");
    }

   private:
    nosel_gl_detail::SelGLCore core_;
  };


  // ---- Shear1hMisSel: N_ij(R) = int dlnM Wij(lnM) Phi_i(R, lnM) ---------
  //
  //   Phi_i(R, lnM) = (1 - f_mis) dSigma_nfw(R, lnM)
  //                 + f_mis dSigma_mis(R, tau_mis * R_lambda(lob_i), lnM)
  //
  // Wij here includes the Sigma_crit_inv(z) factor (z-only, folds into
  // the weight).  Two service methods, ini knob `method`:
  //   exact (default) -- 1-D GL mass sum, no approximation beyond the
  //                      fixed grid itself.
  //   idea2           -- moment expansion (tex eq. approach2_general),
  //                      norm * [Phi + (mu2/2) d2Phi/dlnM^2] at the
  //                      weight-mean lnM_eff, 3-point stencil.
  class Shear1hMisSelGL {
   public:
    using grid_t       = y3_cluster::grid_t<2>;
    using grid_point_t = grid_t::value_type;
    static constexpr std::size_t n_outputs = 1;

    explicit Shear1hMisSelGL(cosmosis::DataBlock& cfg)
      : core_(cfg, module_label())
      , stencil_h_(cfg.has_val(module_label(), "stencil_h")
                     ? cfg.view<double>(module_label(), "stencil_h") : 0.15)
      , dsigma_mis_(4.0, 2.77533742639e+11, y3_cluster::GAMMA)
    {
      lob_centers_ =
        y3_cluster_sel_weights::mis_detail::read_lob_centers(cfg,
                                                             module_label());
      if (lob_centers_.empty())
        throw std::runtime_error("Shear1hMisSelGL: lob_centers is empty");

      if (cfg.has_val(module_label(), "method")) {
        std::string const m =
          cfg.view<std::string>(module_label(), "method");
        if (m == "exact")      use_idea2_ = false;
        else if (m == "idea2") use_idea2_ = true;
        else
          throw std::runtime_error(
            "Shear1hMisSelGL: method must be 'exact' or 'idea2', got '" +
            m + "'");
      }
    }

    void
    set_sample(cosmosis::DataBlock& s)
    {
      namespace w = y3_cluster_sel_weights;

      core_.build_weights(s, /*include_sci=*/true);

      dsigma_nfw_.emplace(
        make_Interp2D(s, "haloModel", "r_sigma", "lnM", "dSigma_nfw"));
      f_mis_   = w::mis_detail::read_mis_param(s, "f_mis",
                                               w::mis_detail::F_MIS_DEFAULT);
      tau_mis_ = w::mis_detail::read_mis_param(s, "tau_mis",
                                               w::mis_detail::TAU_MIS_DEFAULT);
      double const omm = s.view<double>("cosmological_parameters", "omega_M");
      dsigma_mis_.set_rho_mult(omm);
      // OPT-IN (miscentering/use_halo_model_conc != 0, default 0 = the
      // ratified fixed-c production path): feed haloModel/concentration
      // (Child18 x concentration_amplitude) into the miscentered NFW so
      // the 1h miscentered part uses the SAME c(M) as the centered 1h
      // (dSigma_nfw table) and the 2-halo (ShearPrjCore). Kept opt-in
      // until the fixed-c physical-density default has passed the
      // report-based validation (issue #14; same contract as the
      // use_halo_model_conc option on the projection cores).
      if (w::mis_detail::read_mis_param(s, "use_halo_model_conc", 0.0) != 0.0
          && s.has_val("haloModel", "concentration"))
        dsigma_mis_.set_concentration_table(
            make_Interp1D(s, "haloModel", "lnM", "concentration"));

      // Richness bin = bin_index % 4 (richness-fast bin layout); see the
      // header comment on the bins-4-11 fix.
      r_mis_.assign(core_.n_bins(), 0.0);
      for (std::size_t b = 0; b != core_.n_bins(); ++b)
        r_mis_[b] = tau_mis_ * w::mis_detail::R_lambda(
                                 lob_centers_[b % lob_centers_.size()]);
    }

    std::array<double, n_outputs>
    evaluate(grid_point_t const& pt) const
    {
      int const b    = static_cast<int>(pt[0]);
      double const R = pt[1];
      if (b < 0 || static_cast<std::size_t>(b) >= core_.n_bins())
        throw std::out_of_range(
          "Shear1hMisSelGL: bin_index outside sel_function/S_stack range");

      if (!use_idea2_) {
        double acc = 0.0;
        auto const& Wb = core_.weights(b);
        auto const& xs = core_.lnm_x();
        auto const& ws = core_.lnm_w();
        for (std::size_t k = 0; k != xs.size(); ++k)
          acc += ws[k] * Wb[k] * phi(b, R, xs[k]);
        return {acc};
      }

      double const h  = stencil_h_;
      double const y0 = core_.lnm_eff(b);
      double const c  = phi(b, R, y0);
      double const p  = phi(b, R, y0 + h);
      double const m  = phi(b, R, y0 - h);
      double const d2 = (p - 2.0 * c + m) / (h * h);
      return {core_.norm(b) * (c + 0.5 * core_.mu2(b) * d2)};
    }

    static char const* module_label() { return "Shear1hMisSel"; }

    // NOTE: hardcoded, deliberately NOT an ini knob -- CosmoSIS [DEFAULT]
    // blocks propagate keys like output_section into every module
    // section, which would silently redirect the write.
    static std::array<char const*, n_outputs>
    output_sections()
    {
      return {"shear1hmissel"};
    }

    static grid_t
    make_grid_points(cosmosis::DataBlock& cfg)
    {
      return y3_cluster::make_grid_points_cartesian_product(
        cfg, module_label(), "bin_index", "r_perp");
    }

   private:
    // Phi_i(R, lnM): the z-free radial profile of richness bin i.
    // Identical to the old Shear1hMisWeight::operator() minus the
    // Sigma_crit_inv(zt) factor, which is folded into Wij.
    double
    phi(int b, double R, double lnM) const
    {
      double const d_cen = dsigma_nfw_->clamp(R, lnM);
      double const d_mis = dsigma_mis_(R, r_mis_[b], lnM);
      return (1.0 - f_mis_) * d_cen + f_mis_ * d_mis;
    }

    nosel_gl_detail::SelGLCore core_;
    double stencil_h_;
    bool use_idea2_{false};
    std::vector<double> lob_centers_;

    std::optional<y3_cluster::Interp2D> dsigma_nfw_;
    y3_cluster::NFW_DSIGMA_MIS          dsigma_mis_;
    double f_mis_  {y3_cluster_sel_weights::mis_detail::F_MIS_DEFAULT};
    double tau_mis_{y3_cluster_sel_weights::mis_detail::TAU_MIS_DEFAULT};
    std::vector<double> r_mis_;
  };

} // namespace y3_cluster

#endif

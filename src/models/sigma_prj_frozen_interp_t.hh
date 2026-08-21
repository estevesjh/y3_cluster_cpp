// Option E (frozen-physics DeltaSigma_prj benchmark backend) -- see
// RichnessSelection/docs/richness_selection_frozen.tex section "Extension:
// frozen physics for <DeltaSigma_prj>" and richness_selection.FrozenDeltaSigmaPrj
// for the Python reference this ports, and the plan doc "Port frozen-physics
// b_sel/DeltaSigma_prj recipe into the cosmosis pipeline" (Part III, Option E)
// for the design rationale.
//
// Keeps the same frozen-physics algebraic reduction as Option C
// (ShearPrjFrozenPhysics): the z-dependent work (exact rnd-channel hoist,
// cl-channel r_s(M)-anchored amplitude drift a_b(z)) is done once per
// (lob, zob) slice on a fixed ring+outer z-grid (build_z_grid_, copied
// verbatim from sp_detail::ShearPrjCore). What differs from Option C: the
// remaining (lnM, theta) assembly is NOT an explicit N_theta x N_M grid +
// dot product -- w_rnd(lnM), w_cl(lnM), and Psi(theta) are tabulated as
// continuous Interp1D functions, and the final integral is evaluated by a
// genuine 2-D adaptive Cuhre integral over (theta, lnM) (cubacpp, same
// integration backend already used by ShearPrjCuhre in sigma_prj_t.hh),
// instead of building and summing a fixed grid.
//
// No adaptive integration over z: that axis is still eliminated by the
// frozen reduction, same as Option C. Only (theta, lnM) are Cuhre
// integration variables.
#ifndef Y3_CLUSTER_CPP_SIGMA_PRJ_FROZEN_INTERP_T_HH
#define Y3_CLUSTER_CPP_SIGMA_PRJ_FROZEN_INTERP_T_HH

#include "models/sigma_prj_t.hh"   // sp_detail::{PI, R_lambda, default_lob_centers, theta_excl_at_z}

namespace y3_cluster {

  class ShearPrjFrozenCuhre {
   public:
    using grid_t       = y3_cluster::grid_t<4>;
    using grid_point_t = grid_t::value_type;

    // dsigma_{total,rnd,cl}, gt_{total,rnd,cl}
    static constexpr std::size_t n_outputs = 6;

    explicit ShearPrjFrozenCuhre(cosmosis::DataBlock& cfg)
      : N_lnm_(cfg.has_val(module_label(), "n_lnm")
                 ? cfg.view<int>(module_label(), "n_lnm") : 24)
      , N_lnm_tab_(cfg.has_val(module_label(), "n_lnm_tab")
                     ? cfg.view<int>(module_label(), "n_lnm_tab") : 64)
      , N_theta_tab_(cfg.has_val(module_label(), "n_theta_tab")
                       ? cfg.view<int>(module_label(), "n_theta_tab") : 64)
      , N_zring_(cfg.has_val(module_label(), "n_zring")
                   ? cfg.view<int>(module_label(), "n_zring") : 20)
      , N_zouter_(cfg.has_val(module_label(), "n_zouter")
                    ? cfg.view<int>(module_label(), "n_zouter") : 20)
      , zt_lo_(cfg.view<double>(module_label(), "zt_low"))
      , zt_hi_(cfg.view<double>(module_label(), "zt_high"))
      , lnm_lo_(cfg.view<double>(module_label(), "lnm_low"))
      , lnm_hi_(cfg.view<double>(module_label(), "lnm_high"))
      , R_max_cMpch_(cfg.has_val(module_label(), "R_max_cMpch")
                       ? cfg.view<double>(module_label(), "R_max_cMpch") : 30.0)
      , eps_rel_(cfg.has_val(module_label(), "eps_rel")
                   ? cfg.view<double>(module_label(), "eps_rel") : 1.0e-3)
      , eps_abs_(cfg.has_val(module_label(), "eps_abs")
                   ? cfg.view<double>(module_label(), "eps_abs") : 1.0e-12)
      , max_eval_(cfg.has_val(module_label(), "max_eval")
                    ? cfg.view<int>(module_label(), "max_eval") : 20000)
      , algorithm_(cfg.has_val(module_label(), "algorithm")
                     ? cfg.view<std::string>(module_label(), "algorithm") : "cuhre")
      , include_omega_z_(cfg.has_val(module_label(), "include_omega_z")
                            ? cfg.view<int>(module_label(), "include_omega_z") != 0 : true)
    {
      p_op_detail::gl_nodes(lnm_lo_, lnm_hi_, N_lnm_, lnm_x_, lnm_w_);

      if (cfg.has_val(module_label(), "lob_centers")) {
        lob_centers_ = get_vector_double(cfg, module_label(), "lob_centers");
      } else {
        auto const& dflt = sp_detail::default_lob_centers();
        lob_centers_.assign(dflt.begin(), dflt.end());
      }

      // DeltaSigma-only scope (no Sigma_prj counterpart -- FrozenDeltaSigmaPrj
      // has no Sigma analogue in the Python reference either).
      dsigma_mis_.emplace(4.0, 2.77533742639e+11, SINGLE);
      // issue #14: honor use_halo_model_conc (was silently ignored here).
      use_halo_model_conc_ =
          cfg.has_val(module_label(), "use_halo_model_conc") &&
          cfg.view<bool>(module_label(), "use_halo_model_conc");

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

      double const omm = sample.view<double>("cosmological_parameters", "omega_M");
      h0_ = sample.view<double>("cosmological_parameters", "h0");
      dsigma_mis_->set_rho_mult(omm);
      if (use_halo_model_conc_)
        dsigma_mis_->set_concentration_table(
            make_Interp1D(sample, "haloModel", "lnM", "concentration"));

      bsel_.emplace(sample);

      // Explicit ascending lnM tabulation grid for the Interp1D functions
      // (kept separate from the GL nodes lnm_x_, which are only used for
      // the a_b(z)/w_cl_M anchor evaluation below -- Cuhre integrates over
      // lnM continuously, so it must query a plain linearly-interpolated
      // function of lnM, not a GL-weighted sum).
      lnm_tab_.resize(N_lnm_tab_);
      for (std::size_t i = 0; i != N_lnm_tab_; ++i) {
        lnm_tab_[i] = lnm_lo_ + (lnm_hi_ - lnm_lo_) * double(i)
                     / double(N_lnm_tab_ - 1);
      }

      std::size_t const Nlz = lzob_lb_.size();
      lzob_D_A_o_.assign(Nlz, 0.0);
      lzob_sci_.assign(Nlz, 0.0);
      lzob_theta_lo_.assign(Nlz, 0.0);
      lzob_theta_hi_.assign(Nlz, 0.0);
      lzob_theta_lam_.assign(Nlz, 0.0);
      lzob_Bs_.assign(Nlz, 0.0);
      lzob_Bl_.assign(Nlz, 0.0);
      lzob_wrnd_interp_.clear(); lzob_wrnd_interp_.resize(Nlz);
      lzob_wcl_interp_.clear();  lzob_wcl_interp_.resize(Nlz);
      lzob_psi_interp_.clear();  lzob_psi_interp_.resize(Nlz);
      // Interp1D has no default constructor, so these are
      // std::optional<Interp1D> (see member decl) -- .emplace(...) below
      // constructs each one in place once its tabulated values are ready.

      std::vector<double> n_o(N_lnm_), b_o(N_lnm_), anchor_M(N_lnm_);
      std::vector<double> hmf_row(N_lnm_), hmb_row(N_lnm_);

      for (std::size_t k = 0; k != Nlz; ++k) {
        int    const lob_bin = lzob_lb_[k];
        double const zob     = lzob_zob_[k];
        auto const bsel_bin = bsel_->at(lob_bin, zob);
        double const lobc    = bsel_bin.lob;
        double const chi_o   = (*chi_).clamp(zob) * h0_;
        double const D_A_o   = chi_o / (1.0 + zob);
        double const R_excl  = sp_detail::R_lambda(lobc) * (1.0 + zob);
        double const theta_lam    = sp_detail::R_lambda(lobc) * (1.0 + zob) / chi_o;
        double const theta_excl_o = R_excl / chi_o;

        double Rmax_slice = 0.0, Rmin_slice = std::numeric_limits<double>::infinity();
        for (double const R : lzob_Rs_[k]) {
          Rmax_slice = std::max(Rmax_slice, R);
          Rmin_slice = std::min(Rmin_slice, R);
        }
        double const theta_hi = std::max(R_max_cMpch_ / D_A_o,
                                         3.0 * Rmax_slice / D_A_o);
        // Must include theta_R_min = Rmin_slice/D_A_o in the min(), matching
        // sp_detail::build_theta_grid's `lower` formula exactly -- omitting
        // it truncates away the smallest-R breakpoint's support region
        // whenever min(R) sits below theta_lam*D_A_o (this repo's radii
        // grids go down to R=0.2 cMpc/h, well below the lambda scale for
        // most bins), causing a systematic under-estimate across every R
        // on the slice (shared theta domain, not per-R).
        double const theta_lo = std::max(1.0e-8,
                                         0.1 * std::min({theta_excl_o,
                                                         Rmin_slice / D_A_o,
                                                         theta_lam}));

        lzob_D_A_o_[k]     = D_A_o;
        lzob_sci_[k]       = sci_ ? (*sci_).clamp(zob) : 0.0;
        lzob_theta_lo_[k]  = theta_lo;
        lzob_theta_hi_[k]  = theta_hi;
        lzob_theta_lam_[k] = theta_lam;

        lzob_Bs_[k] = bsel_bin.b_small;
        lzob_Bl_[k] = bsel_bin.b_large;

        // n(M,zob), b(M,zob), and the r_s(M)-anchored amplitude-drift
        // denominator, on the fixed GL lnM grid (lnm_x_/lnm_w_).
        double denom = 0.0;
        for (std::size_t iM = 0; iM != N_lnm_; ++iM) {
          n_o[iM] = (*hmf_)(lnm_x_[iM], zob);
          b_o[iM] = hmb_->clamp(lnm_x_[iM], zob);
          // hmf_() already returns dn/dlnM (not dn/dM -- see hmf_t.hh), so no
          // extra exp(lnM) mass-Jacobian belongs here: matches production's
          // own wrnd_M[iM] += common_z*hmf_row[iM]*lnm_w_[iM] convention.
          anchor_M[iM] = lnm_w_[iM] * dsigma_mis_->r_s(lnm_x_[iM]);
          denom += anchor_M[iM] * n_o[iM] * b_o[iM];
        }
        denom += 1.0e-300;

        // w_cl(lnM): frozen shape n(M,zob)*b(M,zob), z-independent --
        // tabulated directly at the lnM_tab_ nodes (no GL weight: this is
        // a continuous density Cuhre will integrate, not a discrete sum).
        std::vector<double> h_cl_tab(N_lnm_tab_);
        for (std::size_t i = 0; i != N_lnm_tab_; ++i) {
          h_cl_tab[i] = (*hmf_)(lnm_tab_[i], zob) * hmb_->clamp(lnm_tab_[i], zob);
        }

        // z-grid (ring+outer split, matches SelBias._z_grid).
        std::vector<double> zs, wzs;
        build_z_grid_(zob, chi_o, R_excl, zs, wzs);
        std::size_t const Nz = zs.size();

        // rnd-channel hoist h_rnd(lnM) = sum_z common_z(z) * n(M,z), and
        // the a_b(z)-weighted Psi(theta) tabulation, built together in one
        // pass over the z-grid (matches FrozenDeltaSigmaPrj's rnd/cl split).
        std::vector<double> h_rnd_tab(N_lnm_tab_, 0.0);
        std::vector<double> theta_tab(N_theta_tab_);
        for (std::size_t j = 0; j != N_theta_tab_; ++j) {
          double const lth = std::log(theta_lo)
                            + (std::log(theta_hi) - std::log(theta_lo))
                              * double(j) / double(N_theta_tab_ - 1);
          theta_tab[j] = std::exp(lth);
        }
        std::vector<double> psi_tab(N_theta_tab_, 0.0);
        std::vector<double> cos_tab(N_theta_tab_), sin_tab(N_theta_tab_);
        for (std::size_t j = 0; j != N_theta_tab_; ++j) {
          cos_tab[j] = std::cos(theta_tab[j]);
          sin_tab[j] = std::sin(theta_tab[j]);
        }

        for (std::size_t iz = 0; iz != Nz; ++iz) {
          double const z     = zs[iz];
          double const chi_z = (*chi_).clamp(z) * h0_;
          double const dV    = (*dv_do_dz_)(z);
          double const om    = include_omega_z_ ? (*omega_z_)(z) : 1.0;
          double const sig_z = (*sigma_z_).clamp(z);
          double const u_phot = (z - zob) / sig_z;
          double const wz_phot = (std::abs(u_phot) < 1.0)
                                 ? (1.0 - u_phot * u_phot) : 0.0;
          double const common_z = dV * om * wzs[iz] * wz_phot;
          if (common_z == 0.0) continue;

          for (std::size_t iM = 0; iM != N_lnm_; ++iM) {
            hmf_row[iM] = (*hmf_)(lnm_x_[iM], z);
            hmb_row[iM] = hmb_->clamp(lnm_x_[iM], z);
          }
          double numer_ab = 0.0;
          for (std::size_t iM = 0; iM != N_lnm_; ++iM)
            numer_ab += anchor_M[iM] * hmf_row[iM] * hmb_row[iM];
          double const a_b_z = numer_ab / denom;

          for (std::size_t i = 0; i != N_lnm_tab_; ++i)
            h_rnd_tab[i] += common_z * (*hmf_)(lnm_tab_[i], z);

          double const theta_excl_z = sp_detail::theta_excl_at_z(chi_z, chi_o, R_excl);
          for (std::size_t j = 0; j != N_theta_tab_; ++j) {
            if (theta_tab[j] <= theta_excl_z) continue;
            double const dchi = std::sqrt(std::max(
                chi_z * chi_z + chi_o * chi_o
                - 2.0 * chi_z * chi_o * cos_tab[j], 0.0));
            double const xi_val = (*xi_nl_).clamp(dchi, zob);
            psi_tab[j] += common_z * a_b_z * xi_val;
          }
        }

        lzob_wrnd_interp_[k].emplace(lnm_tab_, h_rnd_tab);
        lzob_wcl_interp_[k].emplace(lnm_tab_, h_cl_tab);
        lzob_psi_interp_[k].emplace(theta_tab, psi_tab);
      }
    }

    std::array<double, 6>
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

      double const D_A_o    = lzob_D_A_o_[lzob_idx];
      double const sci_v    = lzob_sci_[lzob_idx];
      double const theta_lo = lzob_theta_lo_[lzob_idx];
      double const theta_hi = lzob_theta_hi_[lzob_idx];
      double const theta_lam = lzob_theta_lam_[lzob_idx];
      double const Bs = lzob_Bs_[lzob_idx];
      double const Bl = lzob_Bl_[lzob_idx];
      double const k_sig  = 2.5 / theta_lam;
      double const theta0 = 0.5 * theta_lam;
      double const delta_B = Bl - Bs;

      Interp1D const& w_rnd = *lzob_wrnd_interp_[lzob_idx];
      Interp1D const& w_cl  = *lzob_wcl_interp_ [lzob_idx];
      Interp1D const& psi   = *lzob_psi_interp_ [lzob_idx];
      NFW_DSIGMA_MIS const& ds_mis = *dsigma_mis_;
      double const lnm_lo = lnm_lo_, lnm_hi = lnm_hi_;

      // Cuhre integrates in u = ln(theta), not theta itself: theta_lo/hi
      // span several decades (theta_lo can be ~1e-8, theta_hi ~0.05-0.1),
      // and every feature of interest (exclusion-mask transition, b_sel
      // sigmoid at theta0, xi_nl chord feature) sits in a thin low-theta
      // sliver of that linear range. A cubature rule sampling uniformly in
      // linear theta can miss that sliver entirely while its own embedded
      // error estimate still looks converged (both rules agree on the
      // "flat" bulk) -- the same reason production's own theta grid is
      // log-GL with explicit breakpoints (see CLAUDE.md / sigma_prj_t.hh),
      // not linear. dtheta = theta*du supplies the Jacobian.
      double const u_lo = std::log(theta_lo), u_hi = std::log(theta_hi);

      auto integrand = [&](double u, double lnM) -> std::vector<double> {
        double const theta  = std::exp(u);
        double const sinth = std::sin(theta);
        double const geo   = 2.0 * sp_detail::PI * sinth * theta;
        double const sgm   = 1.0 / (1.0 + std::exp(-k_sig * (theta - theta0)));
        double const bsel  = Bs + delta_B * sgm;
        double const Rmis  = theta * D_A_o;
        double const ds    = ds_mis(R, Rmis, lnM);
        double const wr    = w_rnd.clamp(lnM);
        double const wc    = w_cl.clamp(lnM);
        double const ps    = psi.clamp(theta);
        // No mass-Jacobian here: hmf_() (feeding w_rnd/w_cl) already
        // returns dn/dlnM directly, so the Cuhre dlnM measure matches it
        // as-is -- see the anchor_M fix above for the same reasoning.
        double const f_rnd = geo * wr * ds;
        double const f_cl  = geo * bsel * ps * wc * ds;
        return {f_rnd, f_cl};
      };

      cubacpp::IntegrationVolume<2> vol{{u_lo, lnm_lo}, {u_hi, lnm_hi}};
      double dsigma_rnd = 0.0, dsigma_cl = 0.0;
      if (algorithm_ == "vegas") {
        cubacpp::Vegas alg;
        alg.maxeval = max_eval_;
        auto res = alg.integrate(integrand, eps_rel_, eps_abs_, vol);
        dsigma_rnd = (res.value.size() > 0) ? res.value[0] : 0.0;
        dsigma_cl  = (res.value.size() > 1) ? res.value[1] : 0.0;
      } else {
        cubacpp::Cuhre alg;
        alg.maxeval = max_eval_;
        auto res = alg.integrate(integrand, eps_rel_, eps_abs_, vol);
        dsigma_rnd = (res.value.size() > 0) ? res.value[0] : 0.0;
        dsigma_cl  = (res.value.size() > 1) ? res.value[1] : 0.0;
      }
      double const dsigma_total = dsigma_rnd + dsigma_cl;
      auto gt = [sci_v](double d) { return d * sci_v; };
      return {dsigma_total, dsigma_rnd, dsigma_cl,
              gt(dsigma_total), gt(dsigma_rnd), gt(dsigma_cl)};
    }

    static char const* module_label() { return "shear_prj_frozen_cuhre"; }

    static std::array<char const*, 6>
    output_sections()
    {
      return {"dsigma_prj_frozen_cuhre", "dsigma_prj_frozen_cuhre", "dsigma_prj_frozen_cuhre",
              "shear_prj_frozen_cuhre",  "shear_prj_frozen_cuhre",  "shear_prj_frozen_cuhre"};
    }

    static std::array<char const*, 6>
    output_names()
    {
      return {"vals", "rnd", "cl", "vals", "rnd", "cl"};
    }

    static grid_t
    make_grid_points(cosmosis::DataBlock& cfg)
    {
      return y3_cluster::make_grid_points_wall_of_numbers(
          cfg, module_label(), "lambda_bin", "zo_low", "zo_high", "radii");
    }

   private:
    // --- z-grid builder: copied verbatim from sp_detail::ShearPrjCore
    // (same first-cut duplication tradeoff noted in the plan: a shared
    // sp_detail free function is a reasonable follow-up once this class
    // is stable, but out of scope here). ---
    void
    build_z_grid_(double zob, double chi_o, double R_excl,
                  std::vector<double>& zs, std::vector<double>& wzs) const
    {
      zs.clear();
      wzs.clear();

      double const z_fg_lo = zt_lo_;
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

    std::size_t N_lnm_, N_lnm_tab_, N_theta_tab_;
    std::size_t N_zring_, N_zouter_;
    double zt_lo_, zt_hi_;
    double lnm_lo_, lnm_hi_;
    double R_max_cMpch_;
    double eps_rel_, eps_abs_;
    int max_eval_;
    std::string algorithm_;
    bool include_omega_z_;
    std::vector<double> lob_centers_;

    std::vector<double> lnm_x_, lnm_w_;   // fixed GL nodes for a_b(z)/anchor
    std::vector<double> lnm_tab_;         // explicit ascending grid for Interp1D

    std::optional<y3_cluster::HMF_t>       hmf_;
    std::optional<Interp2D>                hmb_;
    std::optional<y3_cluster::DV_DO_DZ_t>  dv_do_dz_;
    std::optional<y3_cluster::OMEGA_Z_DES> omega_z_;
    std::optional<Interp2D>                xi_nl_;
    std::optional<NFW_DSIGMA_MIS>          dsigma_mis_;
    bool use_halo_model_conc_ = false;   // issue #14: feed haloModel/concentration
    std::optional<Interp1D>                chi_;
    std::optional<Interp1D>                sci_;
    std::optional<Interp1D>                sigma_z_;
    double h0_ = 0.0;

    std::optional<sp_detail::BSelBins> bsel_;

    std::vector<int>    gp_lam_bin_;
    std::vector<double> gp_zob_, gp_R_;
    std::vector<int>    gp_lzob_idx_;
    std::vector<int>    lzob_lb_;
    std::vector<double> lzob_zob_;
    std::vector<std::vector<double>> lzob_Rs_;

    std::vector<double> lzob_D_A_o_, lzob_sci_;
    std::vector<double> lzob_theta_lo_, lzob_theta_hi_, lzob_theta_lam_;
    std::vector<double> lzob_Bs_, lzob_Bl_;
    std::vector<std::optional<Interp1D>> lzob_wrnd_interp_, lzob_wcl_interp_,
        lzob_psi_interp_;
  };

}  // namespace y3_cluster

#endif

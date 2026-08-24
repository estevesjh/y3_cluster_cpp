#ifndef Y3_CLUSTER_CPP_SIGMA_PRJ_FROZEN_T_HH
#define Y3_CLUSTER_CPP_SIGMA_PRJ_FROZEN_T_HH

#include "pipelines/systematics/shear_prj/cpp/sigma_prj_t.hh"   // sp_detail::{PI, R_lambda, default_lob_centers, theta_excl_at_z, build_theta_grid}

namespace y3_cluster {

  class ShearPrjFrozenPhysics {
   public:
    using grid_t       = y3_cluster::grid_t<4>;
    using grid_point_t = grid_t::value_type;

    // dsigma_{total,rnd,cl}, gt_{total,rnd,cl}, plus a shear_prj/{vals,rnd,cl}
    // alias of the gt_* triple for drop-in compatibility with
    // y3_buzzard/likelihood_cp.py (reads shear_prj/vals), matching the same
    // alias Option B (shear_prj_richness.py) already publishes.
    static constexpr std::size_t n_outputs = 9;

    explicit ShearPrjFrozenPhysics(cosmosis::DataBlock& cfg)
      : N_lnm_(cfg.has_val(module_label(), "n_lnm")
                 ? cfg.view<int>(module_label(), "n_lnm") : 24)
      , N_per_seg_(cfg.has_val(module_label(), "n_per_seg")
                     ? cfg.view<int>(module_label(), "n_per_seg") : 10)
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

      bsel_.emplace(sample);

      std::size_t const Nlz = lzob_lb_.size();
      lzob_D_A_o_.assign(Nlz, 0.0);
      lzob_sci_.assign(Nlz, 0.0);
      lzob_theta_.assign(Nlz, {});
      lzob_geom_.assign(Nlz, {});
      lzob_bsel_.assign(Nlz, {});
      lzob_wrnd_M_.assign(Nlz, {});
      lzob_wcl_M_.assign(Nlz, {});
      lzob_psi_.assign(Nlz, {});
      lzob_DSmis_.assign(Nlz, {});

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
        auto const& Rs       = lzob_Rs_[k];

        lzob_D_A_o_[k] = D_A_o;
        lzob_sci_[k]   = sci_ ? (*sci_).clamp(zob) : 0.0;

        // Per-slice theta grid: log-GL on segments split at feature
        // breakpoints, per-R breakpoints included -- identical recipe to
        // production's ShearPrjEvaluator (sp_detail::build_theta_grid), so
        // this class inherits the same domain/resolution guarantees rather
        // than the hand-rolled bounds Option E needed to debug.
        auto const tg = sp_detail::build_theta_grid(lobc, zob, Rs,
                                                     chi_o, D_A_o, R_excl,
                                                     N_per_seg_,
                                                     R_max_cMpch_, {});
        std::size_t const Nth = tg.theta.size();
        auto& theta_k = lzob_theta_[k]; theta_k = tg.theta;
        std::vector<double> sin_k(Nth), cos_k(Nth);
        auto& geom_k = lzob_geom_[k]; geom_k.resize(Nth);
        for (std::size_t it = 0; it != Nth; ++it) {
          sin_k[it] = std::sin(theta_k[it]);
          cos_k[it] = std::cos(theta_k[it]);
          geom_k[it] = tg.weight[it] * 2.0 * sp_detail::PI * sin_k[it];
        }

        double const theta_lam = sp_detail::R_lambda(lobc) * (1.0 + zob) / chi_o;
        double const k_sig   = 2.5 / theta_lam;
        double const theta0  = 0.5 * theta_lam;
        double const delta_B = bsel_bin.b_large - bsel_bin.b_small;
        auto& bsel_k = lzob_bsel_[k]; bsel_k.resize(Nth);
        for (std::size_t it = 0; it != Nth; ++it) {
          double const sgm = 1.0 / (1.0 + std::exp(-k_sig * (theta_k[it] - theta0)));
          bsel_k[it] = bsel_bin.b_small + delta_B * sgm;
        }

        // n(M,zob), b(M,zob), the r_s(M)-anchored amplitude-drift
        // denominator, and the frozen w_cl(M) shape -- on the fixed GL lnM
        // grid (lnm_x_/lnm_w_).
        double denom = 0.0;
        auto& wcl_M = lzob_wcl_M_[k]; wcl_M.resize(N_lnm_);
        for (std::size_t iM = 0; iM != N_lnm_; ++iM) {
          n_o[iM] = (*hmf_)(lnm_x_[iM], zob);
          b_o[iM] = hmb_->clamp(lnm_x_[iM], zob);
          // hmf_() already returns dn/dlnM (not dn/dM -- see hmf_t.hh), so no
          // extra exp(lnM) mass-Jacobian belongs here (same reasoning as
          // Option E's anchor_M fix).
          anchor_M[iM] = lnm_w_[iM] * dsigma_mis_->r_s(lnm_x_[iM]);
          denom += anchor_M[iM] * n_o[iM] * b_o[iM];
          wcl_M[iM] = lnm_w_[iM] * n_o[iM] * b_o[iM];
        }
        denom += 1.0e-300;

        // z-grid (ring+outer split, matches SelBias._z_grid).
        std::vector<double> zs, wzs;
        build_z_grid_(zob, chi_o, R_excl, zs, wzs);
        std::size_t const Nz = zs.size();

        auto& wrnd_M = lzob_wrnd_M_[k]; wrnd_M.assign(N_lnm_, 0.0);
        auto& psi_k  = lzob_psi_[k];    psi_k.assign(Nth, 0.0);

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

          // rnd accumulator: theta-independent, sums straight in.
          for (std::size_t iM = 0; iM != N_lnm_; ++iM)
            wrnd_M[iM] += common_z * hmf_row[iM] * lnm_w_[iM];

          // Psi(theta): per-z LoS-slab exclusion + xi_NL chord distance,
          // weighted by the frozen amplitude drift a_b(z).
          double const theta_excl_z = sp_detail::theta_excl_at_z(chi_z, chi_o, R_excl);
          for (std::size_t it = 0; it != Nth; ++it) {
            if (theta_k[it] <= theta_excl_z) continue;
            double const dchi = std::sqrt(std::max(
                chi_z * chi_z + chi_o * chi_o
                - 2.0 * chi_z * chi_o * cos_k[it], 0.0));
            double const xi_val = (*xi_nl_).clamp(dchi, zob);
            psi_k[it] += common_z * a_b_z * xi_val;
          }
        }

        // Per-R DSigma_mis(R | theta, M) cache, built once per slice
        // (mirrors sp_detail::ShearPrjCore's own Smis_k/DSmis_k caching):
        // evaluate() is then a pure dot product with no NFW lookup inside.
        auto& DSmis_k = lzob_DSmis_[k];
        DSmis_k.assign(Rs.size() * Nth * N_lnm_, 0.0);
        for (std::size_t iR = 0; iR != Rs.size(); ++iR) {
          double const R = Rs[iR];
          double* base = &DSmis_k[iR * Nth * N_lnm_];
          for (std::size_t it = 0; it != Nth; ++it) {
            double const Rmis = theta_k[it] * D_A_o;
            double* row = base + it * N_lnm_;
            for (std::size_t iM = 0; iM != N_lnm_; ++iM)
              row[iM] = (*dsigma_mis_)(R, Rmis, lnm_x_[iM]);
          }
        }
      }
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
      int iR = -1;
      auto const& Rs = lzob_Rs_[lzob_idx];
      for (std::size_t j = 0; j != Rs.size(); ++j) {
        if (std::abs(Rs[j] - R) < 1.0e-12) { iR = int(j); break; }
      }

      double const sci_v = lzob_sci_[lzob_idx];
      auto const& geom_k = lzob_geom_[lzob_idx];
      auto const& bsel_k = lzob_bsel_[lzob_idx];
      auto const& psi_k  = lzob_psi_[lzob_idx];
      auto const& wrnd_M = lzob_wrnd_M_[lzob_idx];
      auto const& wcl_M  = lzob_wcl_M_[lzob_idx];
      std::size_t const Nth = geom_k.size();
      double const* DSmis = &lzob_DSmis_[lzob_idx][std::size_t(iR) * Nth * N_lnm_];

      double acc_rnd = 0.0, acc_cl = 0.0;
      for (std::size_t it = 0; it != Nth; ++it) {
        double const* row = DSmis + it * N_lnm_;
        double a_r = 0.0, a_c = 0.0;
        for (std::size_t iM = 0; iM != N_lnm_; ++iM) {
          double const v = row[iM];
          a_r += wrnd_M[iM] * v;
          a_c += wcl_M[iM]  * v;
        }
        acc_rnd += geom_k[it] * a_r;
        acc_cl  += geom_k[it] * bsel_k[it] * psi_k[it] * a_c;
      }
      double const dsigma_total = acc_rnd + acc_cl;
      auto gt = [sci_v](double d) { return d * sci_v; };
      double const gt_total = gt(dsigma_total), gt_rnd = gt(acc_rnd), gt_cl = gt(acc_cl);
      return {dsigma_total, acc_rnd, acc_cl,
              gt_total, gt_rnd, gt_cl,
              gt_total, gt_rnd, gt_cl};
    }

    static char const* module_label() { return "shear_prj_frozen_physics"; }

    static std::array<char const*, 9>
    output_sections()
    {
      return {"dsigma_prj_frozen_physics", "dsigma_prj_frozen_physics", "dsigma_prj_frozen_physics",
              "shear_prj_frozen_physics",  "shear_prj_frozen_physics",  "shear_prj_frozen_physics",
              "shear_prj", "shear_prj", "shear_prj"};
    }

    static std::array<char const*, 9>
    output_names()
    {
      return {"vals", "rnd", "cl", "vals", "rnd", "cl", "vals", "rnd", "cl"};
    }

    static grid_t
    make_grid_points(cosmosis::DataBlock& cfg)
    {
      return y3_cluster::make_grid_points_wall_of_numbers(
          cfg, module_label(), "lambda_bin", "zo_low", "zo_high", "radii");
    }

   private:
    // --- z-grid builder: copied verbatim from sp_detail::ShearPrjCore
    // (same duplication tradeoff as Option E -- a shared sp_detail free
    // function is a reasonable follow-up once both classes are stable). ---
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

    std::size_t N_lnm_, N_per_seg_;
    std::size_t N_zring_, N_zouter_;
    double zt_lo_, zt_hi_;
    double lnm_lo_, lnm_hi_;
    double R_max_cMpch_;
    bool include_omega_z_;
    std::vector<double> lob_centers_;

    std::vector<double> lnm_x_, lnm_w_;   // fixed GL nodes (a_b(z), anchor, wcl_M)

    std::optional<y3_cluster::HMF_t>       hmf_;
    std::optional<Interp2D>                hmb_;
    std::optional<y3_cluster::DV_DO_DZ_t>  dv_do_dz_;
    std::optional<y3_cluster::OMEGA_Z_DES> omega_z_;
    std::optional<Interp2D>                xi_nl_;
    std::optional<NFW_DSIGMA_MIS>          dsigma_mis_;
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
    std::vector<std::vector<double>> lzob_theta_;   // per-slice theta grid
    std::vector<std::vector<double>> lzob_geom_;    // weight * 2 pi sin(theta)
    std::vector<std::vector<double>> lzob_bsel_;    // b_sel(theta)
    std::vector<std::vector<double>> lzob_psi_;     // frozen cl-channel Psi(theta)
    std::vector<std::vector<double>> lzob_wrnd_M_;  // rnd-channel M hoist
    std::vector<std::vector<double>> lzob_wcl_M_;   // frozen cl-channel M shape
    std::vector<std::vector<double>> lzob_DSmis_;   // [iR][it*N_lnm_+iM]
  };

}  // namespace y3_cluster

#endif

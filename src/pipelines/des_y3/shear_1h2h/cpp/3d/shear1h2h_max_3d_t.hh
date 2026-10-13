// Traditional 1h+2h max-model shear — full (lambda_true, lnM, z)
// reference, C++.
//
// The brute-force adaptive counterpart of the Python explicit-GL max
// backend (../python/shear1h2h_max_explicit_gl.py): one Cuhre triple
// integral per (bin, R) point,
//
//   O_ij(R) = ∫∫∫ dlt dzt dlnM  n(M,zt) dV/dΩdz(zt) Ω(zt)
//             Σ_crit^-1(zt) S_j(zt) S_i(lt, zt) P_HOD(lt | M, zt)
//             d_tot(R, lnM, zt | bin),
//
// where d_tot is formulated via 3D halo exclusion and piecewise approximation:
// 1. In 3D real space, halo exclusion is defined on the correlation function:
//      xi_hm(r) = max(xi_1h(r), b(M, z) * xi_mm(r, z))
//    with transition radius r_trans defined by xi_1h(r_trans) = b(M, z) * xi_mm(r_trans, z).
// 2. The projected excess surface density DeltaSigma_max is approximated around transition radius and is assembled by building a piecewise function:
//      For R < r_trans:
//        DeltaSigma_1h(R) + Delta_gap^1 * (R / r_trans)^2
//      For R >= r_trans:
//        b * DeltaSigma_hh(R) + (C_0 / R^2) * 1e-12
//    where C_0 = Delta M_3D(r_trans) / pi (in Msun/h) and
//    Delta_gap^1 = DeltaSigma_max(r_trans) - DeltaSigma_1h(r_trans).
//
// with DeltaSigma_1h the production miscentred mixture ((1 - f_mis)
// haloModel ΔΣ_nfw + f_mis gamma-table ΔΣ_mis, Ω_m amplitude). Unlike
// the 1-halo Shear1h3d, d_tot is z-dependent (the biased two-halo
// term) and the max is nonlinear, so nothing can be contracted past the
// profile — the whole integrand rides inside the adaptive volume, with
// every selection kernel evaluated at the quadrature points (no S_ij
// tabulation). This is the 3d adaptive reference the fixed-GL
// Shear1h2hMax backends validate against.
//
// dSigma_hh carries NaN over ~60% of its (R, z) table by construction
// (see docs/known_issues/dsigma_hh_debug_flag.md); the values are sanitized to
// 0 BEFORE being handed to Interp2D, which is exact for a max model
// (max(1h, 0) = 1h where the 2h term is undefined) and keeps NaN out
// of the interpolator's stencil. Requires compute_lensing_2h = T.
//
// f_mis and tau_mis are REQUIRED datablock values (miscentering/f_mis,
// miscentering/tau_mis): set_sample throws if the section is missing —
// no silent fallback to the Y3 fiducial defaults.
//
// Configuration: the Shear1h3d options (bin definitions +
// per-row (lt, zt, lnm) volumes zipped with the (bin_index, r_perp)
// wall, lob_centers) plus include_miscentering (default T).
// Output: shear1h2hmax3d/{vals, errors, probs, status, nregions}.
// Status: reference backend; the fixed-GL Shear1h2hMax.so is the fast
// re-expression, production remains the traditional Y1-era stack.
#ifndef Y3_CLUSTER_CPP_SHEAR1H2H_MAX_3D_T_HH
#define Y3_CLUSTER_CPP_SHEAR1H2H_MAX_3D_T_HH

#include "utils/datablock_reader.hh"
#include "utils/make_grid_points.hh"
#include "utils/make_integration_volumes.hh"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"
#include "cubacpp/integration_volume.hh"

#include "models/dv_do_dz_t.hh"
#include "models/hmf_t.hh"
#include "models/mor_hod_t.hh"
#include "models/nfw_dsigma_mis.hh"
#include "models/omega_z_des.hh"
#include "models/plob_ltr_emg_t.hh"
#include "models/richness_kernel_t.hh"
#include "pipelines/shared/lensing_helpers.hh"
#include "pipelines/systematics/selection_boost/Bsel.hh"
#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"
#include "utils/make_interp_2d.hh"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

class Shear1h2hMax3d {
public:
  using grid_t = y3_cluster::grid_t<2>;
  using grid_point_t = grid_t::value_type;

private:
  using volume_t = cubacpp::IntegrationVolume<3>;

  std::vector<y3_cluster::RichnessKernel_t> s_i_;
  std::vector<double> zob_min_, zob_max_, sigma_z_;
  std::vector<double> lob_centers_;
  bool include_mis_{true};
  bool bsel_enabled_{false};
  std::string bsel_model_;
  std::optional<y3_cluster::BselDeltaSigmaOperator> bsel_op_;
  std::optional<y3_cluster::Interp2D> sigma_bar_hh_, sigma_bar_nfw_;

  std::optional<y3_cluster::HMF_t> hmf_;
  std::optional<y3_cluster::DV_DO_DZ_t> dv_do_dz_;
  std::optional<y3_cluster::OMEGA_Z_DES> omega_z_;
  std::optional<y3_cluster::MOR_HOD_t> mor_;
  std::optional<y3_cluster::PlobLtrEMG_t> plob_;
  std::optional<y3_cluster::Interp2D> dsigma_nfw_, bias_, dsigma_hh_;
  std::optional<y3_cluster::Interp2D> r_trans_, c0_trans_, delta_gap_1_, delta_gap_0_;
  std::optional<y3_cluster::Interp1D> sci_;
  // TODO(#14): drop the hardcoded c=4 — once claude/issue-4-dsigma-hh-2h-term
  // merges, call dsigma_mis_.set_concentration_table(
  //   make_Interp1D(sample, "haloModel", "lnM", "concentration"))
  // in set_sample so the miscentred term uses the same per-mass Child18
  // concentration as the centred dSigma_nfw table.
  y3_cluster::NFW_DSIGMA_MIS dsigma_mis_{y3_cluster::CONC, y3_cluster::RHOC,
                                         y3_cluster::GAMMA};
  double f_mis_{0.0};
  bool phys_density_{false};
  double tau_mis_{0.0};

  int current_bin_{0};
  double current_R_{0.0};
  double current_r_mis_{0.0};

public:
  explicit Shear1h2hMax3d(cosmosis::DataBlock& cfg)
  {
    auto const lam_min = get_vector_double(cfg, module_label(), "lam_min");
    auto const lam_max = get_vector_double(cfg, module_label(), "lam_max");
    zob_min_ = get_vector_double(cfg, module_label(), "zob_min");
    zob_max_ = get_vector_double(cfg, module_label(), "zob_max");
    sigma_z_ = get_vector_double(cfg, module_label(), "sigma_z");
    std::size_t const n = lam_min.size();
    if (lam_max.size() != n || zob_min_.size() != n ||
        zob_max_.size() != n || sigma_z_.size() != n)
      throw std::runtime_error(
        "Shear1h2hMax3d: bin definition arrays have unequal lengths");
    s_i_.reserve(n);
    for (std::size_t i = 0; i != n; ++i)
      s_i_.emplace_back(lam_min[i], lam_max[i]);
    lob_centers_ =
      y3_pipelines::read_lob_centers(cfg, module_label());
    if (lob_centers_.empty())
      throw std::runtime_error("Shear1h2hMax3d: lob_centers is empty");
    include_mis_ =
      read_bool_opt(cfg, module_label(), "include_miscentering", true);
    bsel_enabled_ =
      read_bool_opt(cfg, module_label(), "bsel", false);
    if (bsel_enabled_ && cfg.has_val(module_label(), "bsel_model"))
      bsel_model_ = cfg.view<std::string>(module_label(), "bsel_model");
  }

  static bool read_bool_opt(cosmosis::DataBlock& cfg,
                            char const* sec,
                            char const* key,
                            bool default_val)
  {
    if (!cfg.has_val(sec, key))
      return default_val;
    try {
      return cfg.view<bool>(sec, key);
    } catch (...) {
      try {
        return cfg.view<int>(sec, key) != 0;
      } catch (...) {
        auto const str = cfg.view<std::string>(sec, key);
        return str == "T" || str == "true" || str == "True" || str == "1";
      }
    }
  }

  void
  set_sample(cosmosis::DataBlock& sample)
  {
    namespace w = y3_pipelines;
    bsel_op_.reset();
    if (bsel_enabled_) {
      bsel_op_.emplace(
        y3_cluster::BselDeltaSigmaOperator::from_datablock(
          sample, bsel_model_));
    }
    hmf_.emplace(sample);
    dv_do_dz_.emplace(sample);
    omega_z_.emplace(sample);
    mor_.emplace(sample);
    plob_.emplace(sample);
    dsigma_nfw_.emplace(y3_cluster::make_Interp2D(
      sample, "haloModel", "r_sigma", "lnM", "dSigma_nfw"));
    bias_.emplace(
      y3_cluster::make_Interp2D(sample, "haloModel", "lnM", "z", "bias"));
    dsigma_hh_.emplace(make_sanitized_hh(sample));
    if (sample.has_val("haloModel", "Sigma_bar_hh"))
      sigma_bar_hh_.emplace(make_sanitized_table(sample, "Sigma_bar_hh"));
    if (sample.has_val("haloModel", "Sigma_bar_nfw"))
      sigma_bar_nfw_.emplace(y3_cluster::make_Interp2D(
        sample, "haloModel", "r_sigma", "lnM", "Sigma_bar_nfw"));
    if (sample.has_val("haloModel", "r_trans")) {
      r_trans_.emplace(
        y3_cluster::make_Interp2D(sample, "haloModel", "lnM", "z", "r_trans"));
      c0_trans_.emplace(
        y3_cluster::make_Interp2D(sample, "haloModel", "lnM", "z", "c0_trans"));
      delta_gap_1_.emplace(
        y3_cluster::make_Interp2D(sample, "haloModel", "lnM", "z", "delta_gap_1"));
      if (sample.has_val("haloModel", "delta_gap_0")) {
        delta_gap_0_.emplace(
          y3_cluster::make_Interp2D(sample, "haloModel", "lnM", "z", "delta_gap_0"));
      }
    } else {
      r_trans_.reset();
      c0_trans_.reset();
      delta_gap_1_.reset();
      delta_gap_0_.reset();
    }
    sci_.emplace(w::load_sigma_crit_inv(sample));
    // Required: no fallback to the fiducial defaults — a pipeline that
    // has not published the miscentering section must fail loudly.
    f_mis_ =
      include_mis_ ? sample.view<double>("miscentering", "f_mis") : 0.0;
    tau_mis_ = sample.view<double>("miscentering", "tau_mis");
    // UNIFIED rho_m convention (2026-08-24): boundary AND amplitude on
    // haloModel/rho_m_ref (same density as the centred dSigma_nfw table).
    dsigma_mis_.set_rho_ref(sample.view<double>("haloModel", "rho_m_ref"));
    // Physical mean density (opt-in): exact per-zt identity on the
    // 1-halo mixture ONLY (the 2-halo term keeps its own convention).
    int phys = 0;
    if (sample.has_val("haloModel", "one_halo_physical_density"))
      sample.get_val("haloModel", "one_halo_physical_density", phys);
    phys_density_ = (phys != 0);
  }

  void
  set_grid_point(grid_point_t const& pt)
  {
    current_bin_ = static_cast<int>(pt[0]);
    current_R_ = pt[1];
    if (current_bin_ < 0 ||
        static_cast<std::size_t>(current_bin_) >= s_i_.size())
      throw std::out_of_range(
        "Shear1h2hMax3d: bin_index outside the configured bin set");
    current_r_mis_ =
      tau_mis_ * y3_pipelines::R_lambda(
                   lob_centers_[current_bin_ % lob_centers_.size()]);
  }

  double
  operator()(double lt, double zt, double lnM) const
  {
    double const s_j = y3_cluster::richness_zkernel(
      zt, zob_min_[current_bin_], zob_max_[current_bin_],
      sigma_z_[current_bin_]);
    double const q = phys_density_ ? 1.0 + zt : 1.0;
    double const d_cen = dsigma_nfw_->clamp(current_R_ * q, lnM);
    double const d_mis = dsigma_mis_(current_R_ * q, current_r_mis_ * q, lnM);
    double const d_1h = (q * q) *
                        ((1.0 - f_mis_) * d_cen + f_mis_ * d_mis);
    double const d_2h =
      bias_->clamp(lnM, zt) * dsigma_hh_->clamp(current_R_, zt);
    double d_tot;
    if (r_trans_ && c0_trans_ && delta_gap_1_) {
      double const rt = r_trans_->clamp(lnM, zt);
      if (current_R_ >= rt) {
        double const c0 = c0_trans_->clamp(lnM, zt);
        d_tot = d_2h + (c0 / (current_R_ * current_R_)) * 1.0e-12;
      } else {
        double const gap1 = delta_gap_1_->clamp(lnM, zt);
        double const r_ratio = current_R_ / rt;
        d_tot = d_1h + gap1 * r_ratio * r_ratio;
      }
    } else {
      d_tot = std::max(d_1h, d_2h);
    }
    double d_val = d_tot;
    if (bsel_op_) {
      // Apply the selection boost operator:
      // DeltaSigma_sel(R) = B_sel(R) * DeltaSigma(R) - M_sel(R) / (pi R^2).
      // The lambda supplies comoving barSigma(<r) matching the comoving B_sel(R) grid.
      double const lob = lob_centers_[current_bin_ % lob_centers_.size()];
      d_val = bsel_op_->apply(
        d_tot,
        [this, lnM, zt](double r) {
          return mean_surface_density(r, lnM, zt);
        },
        current_R_, current_bin_, lob, zt);
    }
    return (*hmf_)(lnM, zt) * (*dv_do_dz_)(zt) * (*omega_z_)(zt) *
           sci_->clamp(zt) * s_j * s_i_[current_bin_](lt, zt, *plob_) *
           (*mor_)(lt, lnM, zt) * d_val;
  }

  static char const*
  module_label()
  {
    return "Shear1h2hMax3d";
  }

  static std::vector<volume_t>
  make_integration_volumes(cosmosis::DataBlock& cfg)
  {
    return y3_cluster::make_integration_volumes_wall_of_numbers(
      cfg, module_label(), "lt", "zt", "lnm");
  }

  static grid_t
  make_grid_points(cosmosis::DataBlock& cfg)
  {
    return y3_cluster::make_grid_points_wall_of_numbers(
      cfg, module_label(), "bin_index", "r_perp");
  }

private:
  static y3_cluster::Interp2D
  make_sanitized_table(cosmosis::DataBlock& s, char const* key)
  {
    using doubles = std::vector<double>;
    auto const r = s.view<doubles>("haloModel", "r_sigma");
    auto const z = s.view<doubles>("haloModel", "z");
    auto const& nd =
      s.view<cosmosis::ndarray<double>>("haloModel", key);
    std::vector<double> vals(nd.begin(), nd.end());
    if (vals.size() != r.size() * z.size())
      throw std::runtime_error(
        std::string("Shear1h2hMax3d: ") + key + " extents mismatch");
    for (auto& v : vals)
      if (!std::isfinite(v)) v = 0.0;
    return y3_cluster::Interp2D(r, z, vals);
  }

  static y3_cluster::Interp2D
  make_sanitized_hh(cosmosis::DataBlock& s)
  {
    return make_sanitized_table(s, "dSigma_hh");
  }

  // barSigma(<r) = Sigma(r) + DeltaSigma(r) for a given (lnM, zt); the
  // integrand input to BselDeltaSigmaOperator::apply in comoving coordinates.
  double mean_surface_density(double r, double lnM, double zt) const
  {
    if (r_trans_ && c0_trans_ && delta_gap_1_) {
      double const rt = r_trans_->clamp(lnM, zt);
      if (r >= rt) {
        double const s_2h = sigma_bar_hh_ ? bias_->clamp(lnM, zt) * sigma_bar_hh_->clamp(r, zt) : 0.0;
        double const c0 = c0_trans_->clamp(lnM, zt);
        return s_2h + (c0 / (r * r)) * 1.0e-12;
      } else {
        double const s_1h = sigma_bar_nfw_ ? sigma_bar_nfw_->clamp(r, lnM)
                                           : (dsigma_nfw_ ? dsigma_nfw_->clamp(r, lnM) : 0.0);
        double const gap0 = delta_gap_0_ ? delta_gap_0_->clamp(lnM, zt) : 0.0;
        double const gap1 = delta_gap_1_->clamp(lnM, zt);
        double const r_ratio = r / rt;
        return s_1h + gap0 + 2.0 * gap1 - gap1 * r_ratio * r_ratio;
      }
    }
    if (sigma_bar_nfw_ && sigma_bar_hh_) {
      return std::max(sigma_bar_nfw_->clamp(r, lnM), bias_->clamp(lnM, zt) * sigma_bar_hh_->clamp(r, zt));
    }
    return dsigma_nfw_ ? dsigma_nfw_->clamp(r, lnM) : 0.0;
  }
};

#endif

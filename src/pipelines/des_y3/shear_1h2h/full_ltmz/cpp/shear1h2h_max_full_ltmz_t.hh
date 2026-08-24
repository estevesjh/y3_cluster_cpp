// Traditional 1h+2h max-model shear — full (lambda_true, lnM, z)
// reference, C++.
//
// The brute-force adaptive counterpart of the Python full_ltmz max
// backend (../python/shear1h2h_max_full_ltmz.py): one Cuhre triple
// integral per (bin, R) point,
//
//   O_ij(R) = ∫∫∫ dlt dzt dlnM  n(M,zt) dV/dΩdz(zt) Ω(zt)
//             Σ_crit^-1(zt) S_j(zt) S_i(lt, zt) P_HOD(lt | M, zt)
//             d_tot(R, lnM, zt | bin),
//
//   d_tot = max( DSigma_cl(R, lnM | bin),
//                b(lnM, zt) * DSigma_hh(R, zt) )
//
// with DSigma_cl the production miscentred mixture ((1 - f_mis)
// haloModel ΔΣ_nfw + f_mis gamma-table ΔΣ_mis, Ω_m amplitude). Unlike
// the 1-halo Shear1hFullLtmz, d_tot is z-dependent (the biased two-halo
// term) and the max is nonlinear, so nothing can be contracted past the
// profile — the whole integrand rides inside the adaptive volume, with
// every selection kernel evaluated at the quadrature points (no S_ij
// tabulation). This is the full_ltmz reference the fast_mass
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
// Configuration: the Shear1hFullLtmz options (bin definitions +
// per-row (lt, zt, lnm) volumes zipped with the (bin_index, r_perp)
// wall, lob_centers) plus include_miscentering (default T).
// Output: shear1h2hmaxfullltmz/{vals, errors, probs, status, nregions}.
// Status: reference backend; the fast_mass Shear1h2hMax.so is the fast
// re-expression, production remains the traditional Y1-era stack.
#ifndef Y3_CLUSTER_CPP_SHEAR1H2H_MAX_FULL_LTMZ_T_HH
#define Y3_CLUSTER_CPP_SHEAR1H2H_MAX_FULL_LTMZ_T_HH

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
#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"
#include "utils/make_interp_2d.hh"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

class Shear1h2hMaxFullLtmz {
public:
  using grid_t = y3_cluster::grid_t<2>;
  using grid_point_t = grid_t::value_type;

private:
  using volume_t = cubacpp::IntegrationVolume<3>;

  std::vector<y3_cluster::RichnessKernel_t> s_i_;
  std::vector<double> zob_min_, zob_max_, sigma_z_;
  std::vector<double> lob_centers_;
  bool include_mis_{true};

  std::optional<y3_cluster::HMF_t> hmf_;
  std::optional<y3_cluster::DV_DO_DZ_t> dv_do_dz_;
  std::optional<y3_cluster::OMEGA_Z_DES> omega_z_;
  std::optional<y3_cluster::MOR_HOD_t> mor_;
  std::optional<y3_cluster::PlobLtrEMG_t> plob_;
  std::optional<y3_cluster::Interp2D> dsigma_nfw_, bias_, dsigma_hh_;
  std::optional<y3_cluster::Interp1D> sci_;
  // TODO(#14): drop the hardcoded c=4 — once claude/issue-4-dsigma-hh-2h-term
  // merges, call dsigma_mis_.set_concentration_table(
  //   make_Interp1D(sample, "haloModel", "lnM", "concentration"))
  // in set_sample so the miscentred term uses the same per-mass Child18
  // concentration as the centred dSigma_nfw table.
  y3_cluster::NFW_DSIGMA_MIS dsigma_mis_{y3_cluster::CONC, y3_cluster::RHOC,
                                         y3_cluster::GAMMA};
  double f_mis_{0.0};
  double tau_mis_{0.0};

  int current_bin_{0};
  double current_R_{0.0};
  double current_r_mis_{0.0};

public:
  explicit Shear1h2hMaxFullLtmz(cosmosis::DataBlock& cfg)
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
        "Shear1h2hMaxFullLtmz: bin definition arrays have unequal lengths");
    s_i_.reserve(n);
    for (std::size_t i = 0; i != n; ++i)
      s_i_.emplace_back(lam_min[i], lam_max[i]);
    lob_centers_ =
      y3_pipelines::read_lob_centers(cfg, module_label());
    if (lob_centers_.empty())
      throw std::runtime_error("Shear1h2hMaxFullLtmz: lob_centers is empty");
    if (cfg.has_val(module_label(), "include_miscentering"))
      include_mis_ =
        cfg.view<int>(module_label(), "include_miscentering") != 0;
  }

  void
  set_sample(cosmosis::DataBlock& sample)
  {
    namespace w = y3_pipelines;
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
    sci_.emplace(w::load_sigma_crit_inv(sample));
    // Required: no fallback to the fiducial defaults — a pipeline that
    // has not published the miscentering section must fail loudly.
    f_mis_ =
      include_mis_ ? sample.view<double>("miscentering", "f_mis") : 0.0;
    tau_mis_ = sample.view<double>("miscentering", "tau_mis");
    dsigma_mis_.set_rho_mult(
      sample.view<double>("cosmological_parameters", "omega_M"));
  }

  void
  set_grid_point(grid_point_t const& pt)
  {
    current_bin_ = static_cast<int>(pt[0]);
    current_R_ = pt[1];
    if (current_bin_ < 0 ||
        static_cast<std::size_t>(current_bin_) >= s_i_.size())
      throw std::out_of_range(
        "Shear1h2hMaxFullLtmz: bin_index outside the configured bin set");
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
    double const d_cen = dsigma_nfw_->clamp(current_R_, lnM);
    double const d_mis = dsigma_mis_(current_R_, current_r_mis_, lnM);
    double const d_1h = (1.0 - f_mis_) * d_cen + f_mis_ * d_mis;
    double const d_2h =
      bias_->clamp(lnM, zt) * dsigma_hh_->clamp(current_R_, zt);
    double const d_tot = std::max(d_1h, d_2h);
    return (*hmf_)(lnM, zt) * (*dv_do_dz_)(zt) * (*omega_z_)(zt) *
           sci_->clamp(zt) * s_j * s_i_[current_bin_](lt, zt, *plob_) *
           (*mor_)(lt, lnM, zt) * d_tot;
  }

  static char const*
  module_label()
  {
    return "Shear1h2hMaxFullLtmz";
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
  // haloModel/dSigma_hh through Interp2D, with the producer's NaNs
  // replaced by 0 first (docs/known_issues/dsigma_hh_debug_flag.md), same
  // convention as the fast_mass Shear1h2hMax backends.
  static y3_cluster::Interp2D
  make_sanitized_hh(cosmosis::DataBlock& s)
  {
    using doubles = std::vector<double>;
    auto const r = s.view<doubles>("haloModel", "r_sigma");
    auto const z = s.view<doubles>("haloModel", "z");
    auto const& nd =
      s.view<cosmosis::ndarray<double>>("haloModel", "dSigma_hh");
    std::vector<double> vals(nd.begin(), nd.end());
    if (vals.size() != r.size() * z.size())
      throw std::runtime_error(
        "Shear1h2hMaxFullLtmz: dSigma_hh extents mismatch");
    for (auto& v : vals)
      if (!std::isfinite(v)) v = 0.0;
    return y3_cluster::Interp2D(r, z, vals);
  }
};

#endif

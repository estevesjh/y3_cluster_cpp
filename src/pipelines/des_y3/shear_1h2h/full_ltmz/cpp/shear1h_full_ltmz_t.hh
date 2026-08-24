// Miscentred one-halo shear — full (lambda_true, lnM, z) reference, C++.
//
// The brute-force adaptive counterpart of the Python full_ltmz shear
// backend (../python/shear1h_full_ltmz.py): one Cuhre triple integral
// per (bin, R) point,
//
//   O_ij(R) = ∫∫∫ dlt dzt dlnM  n(M,zt) dV/dΩdz(zt) Ω(zt)
//             Σ_crit^-1(zt) S_j(zt) S_i(lt, zt) P_HOD(lt | M, zt)
//             d_tot(R, lnM),
//
// with S_i the observed-richness kernel, S_j the observed-redshift
// kernel, and d_tot the production miscentred mixture ((1 - f_mis)
// haloModel ΔΣ_nfw + f_mis gamma-table ΔΣ_mis, Ω_m amplitude). All
// physics terms are existing immutable models; the module instantiates
// the immutable scalar-integration template. Deliberately makes no use
// of d_tot being z-free — the whole point of this backend is an
// integration strategy with no shared structure with the fixed-GL
// references it validates.
//
// f_mis and tau_mis are REQUIRED datablock values (miscentering/f_mis,
// miscentering/tau_mis): set_sample throws if the section is missing —
// no silent fallback to the Y3 fiducial defaults.
//
// Configuration: the counts full_ltmz options (bin definitions +
// per-bin lt/zt/lnm volumes, zipped with bin_index) PLUS r_perp.
// The volumes-and-gridpoints pairing is zipped per wall entry: supply
// one wall row per (bin, R) output — bin_index and r_perp arrays of
// length n_bins * n_R (bin slow, R fast), with the volume arrays
// repeated per radius. eps_rel = 1e-4 per the counts backend's
// finding (the lt ridge at lambda_true ~ 1).
//
// Output: shear1hfullltmz/{vals, errors, probs, status, nregions}.
// Status: reference backend. Production remains Shear1hMisSel.so.
#ifndef Y3_CLUSTER_CPP_SHEAR1H_FULL_LTMZ_T_HH
#define Y3_CLUSTER_CPP_SHEAR1H_FULL_LTMZ_T_HH

#include "utils/datablock_reader.hh"
#include "utils/make_grid_points.hh"
#include "utils/make_integration_volumes.hh"

#include "cosmosis/datablock/datablock.hh"
#include "cubacpp/integration_volume.hh"

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
#include "utils/make_interp_2d.hh"

#include <optional>
#include <stdexcept>
#include <vector>

class Shear1hFullLtmz {
public:
  using grid_t = y3_cluster::grid_t<2>;
  using grid_point_t = grid_t::value_type;

private:
  using volume_t = cubacpp::IntegrationVolume<3>;

  std::vector<y3_cluster::RichnessKernel_t> s_i_;
  std::vector<double> zob_min_, zob_max_, sigma_z_;
  std::vector<double> lob_centers_;

  std::optional<y3_cluster::HMF_t> hmf_;
  std::optional<y3_cluster::DV_DO_DZ_t> dv_do_dz_;
  std::optional<y3_cluster::OMEGA_Z_DES> omega_z_;
  std::optional<y3_cluster::MOR_HOD_t> mor_;
  std::optional<y3_cluster::PlobLtrEMG_t> plob_;
  std::optional<y3_cluster::Interp2D> dsigma_nfw_;
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
  explicit Shear1hFullLtmz(cosmosis::DataBlock& cfg)
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
        "Shear1hFullLtmz: bin definition arrays have unequal lengths");
    s_i_.reserve(n);
    for (std::size_t i = 0; i != n; ++i)
      s_i_.emplace_back(lam_min[i], lam_max[i]);
    lob_centers_ =
      y3_cluster_sel_weights::mis_detail::read_lob_centers(cfg,
                                                           module_label());
    if (lob_centers_.empty())
      throw std::runtime_error("Shear1hFullLtmz: lob_centers is empty");
  }

  void
  set_sample(cosmosis::DataBlock& sample)
  {
    namespace w = y3_cluster_sel_weights;
    hmf_.emplace(sample);
    dv_do_dz_.emplace(sample);
    omega_z_.emplace(sample);
    mor_.emplace(sample);
    plob_.emplace(sample);
    dsigma_nfw_.emplace(y3_cluster::make_Interp2D(
      sample, "haloModel", "r_sigma", "lnM", "dSigma_nfw"));
    sci_.emplace(w::load_sigma_crit_inv(sample));
    // Required: no fallback to the fiducial defaults — a pipeline that
    // has not published the miscentering section must fail loudly.
    f_mis_ = sample.view<double>("miscentering", "f_mis");
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
        "Shear1hFullLtmz: bin_index outside the configured bin set");
    current_r_mis_ =
      tau_mis_ * y3_cluster_sel_weights::mis_detail::R_lambda(
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
    double const d_tot = (1.0 - f_mis_) * d_cen + f_mis_ * d_mis;
    return (*hmf_)(lnM, zt) * (*dv_do_dz_)(zt) * (*omega_z_)(zt) *
           sci_->clamp(zt) * s_j * s_i_[current_bin_](lt, zt, *plob_) *
           (*mor_)(lt, lnM, zt) * d_tot;
  }

  static char const*
  module_label()
  {
    return "Shear1hFullLtmz";
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
};

#endif

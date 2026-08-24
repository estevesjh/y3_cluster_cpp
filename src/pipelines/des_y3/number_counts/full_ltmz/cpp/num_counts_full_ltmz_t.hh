// Full (lambda_true, lnM, z) reference number counts — `full_ltmz`, C++.
//
// The complete maintained observable as one adaptive triple integral per
// (richness, photo-z) bin:
//
//   N_ij = ∫∫∫ dlt dzt dlnM  n(M,zt) · dV/dΩdz(zt) · Ω(zt)
//                            · S_j(zt) · S_i(lt, zt) · P_HOD(lt | M, zt)
//
// This is the C++ backend of the des_y3 `full_ltmz` strategy
// (src/pipelines/des_y3/README.md): unlike the older
// NumCountsFullScalarIntegrand diagnostic it includes the Gaussian
// observed-redshift kernel S_j and the Costanzi EMG observed-richness
// kernel S_i,
// so it computes the same quantity as the production fast path
// (sel_function.py + NumCountsSel.so) with no intermediate S_ij
// tabulation, no redshift contraction, and no fixed-node quadrature —
// adaptive Cuhre over an explicit per-bin (lt, zt, lnM) volume.
//
// Every physics term is an existing immutable model reused as-is:
// HMF_t, DV_DO_DZ_t, OMEGA_Z_DES, MOR_HOD_t (shifted-Poisson HOD),
// PlobLtrEMG_t + RichnessKernel_t (EMG S_i), richness_zkernel (S_j).
// PlobLtrEMG_t reads the datablock section `plob_ltr_params`, which the
// systematics/selection_function/prj_params.py publishes at pipeline setup (the Python
// backend's in-code default is the same frozen table).
//
// Configuration (wall-of-numbers, one entry per bin, zipped with
// bin_index): lam_min/lam_max/zob_min/zob_max/sigma_z (bin definitions),
// lt_low/lt_high/zt_low/zt_high/lnm_low/lnm_high (integration volume),
// bin_index, plus the standard integration-module knobs (algorithm,
// eps_rel, eps_abs, max_eval, use_cartesian_product = F).
//
// The lt volume must cover the HOD support inside the mass envelope
// (lt_high ~ 2000 at the Y3 fiducial; the integrand decays to zero well
// inside, and Cuhre concentrates points adaptively). lt_low must be
// strictly positive: the EMG parameter provider is singular at lt = 0
// (sigma ~ lt^a), and richness support below lt ~ 0.1 contributes
// nothing to lambda_ob >= 20 bins.
//
// Output: numcountsfullltmz/{vals, errors, probs, status, nregions},
// one entry per bin.
//
// Status: reference backend (validated against the Python full_ltmz
// reference and production NumCountsSel.so; see README.md). Not a
// production entry point.
#ifndef Y3_CLUSTER_CPP_NUM_COUNTS_FULL_LTMZ_T_HH
#define Y3_CLUSTER_CPP_NUM_COUNTS_FULL_LTMZ_T_HH

#include "utils/datablock_reader.hh"
#include "utils/make_grid_points.hh"
#include "utils/make_integration_volumes.hh"

#include "cosmosis/datablock/datablock.hh"
#include "cubacpp/integration_volume.hh"

#include "models/dv_do_dz_t.hh"
#include "models/hmf_t.hh"
#include "models/mor_hod_t.hh"
#include "models/omega_z_des.hh"
#include "models/plob_ltr_emg_t.hh"
#include "models/richness_kernel_t.hh"

#include <optional>
#include <stdexcept>
#include <vector>

class NumCountsFullLtmz {
public:
  using grid_t = y3_cluster::grid_t<1>;
  using grid_point_t = grid_t::value_type;

private:
  using volume_t = cubacpp::IntegrationVolume<3>;

  // Bin definitions from configuration.
  std::vector<y3_cluster::RichnessKernel_t> s_i_;
  std::vector<double> zob_min_, zob_max_, sigma_z_;

  // Per-sample models.
  std::optional<y3_cluster::HMF_t> hmf_;
  std::optional<y3_cluster::DV_DO_DZ_t> dv_do_dz_;
  std::optional<y3_cluster::OMEGA_Z_DES> omega_z_;
  std::optional<y3_cluster::MOR_HOD_t> mor_;
  std::optional<y3_cluster::PlobLtrEMG_t> plob_;

  int current_bin_{0};

public:
  explicit NumCountsFullLtmz(cosmosis::DataBlock& cfg)
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
        "NumCountsFullLtmz: bin definition arrays have unequal lengths");
    s_i_.reserve(n);
    for (std::size_t i = 0; i != n; ++i)
      s_i_.emplace_back(lam_min[i], lam_max[i]);
  }

  void
  set_sample(cosmosis::DataBlock& sample)
  {
    hmf_.emplace(sample);
    dv_do_dz_.emplace(sample);
    omega_z_.emplace(sample);
    mor_.emplace(sample);
    plob_.emplace(sample);
  }

  void
  set_grid_point(grid_point_t const& pt)
  {
    current_bin_ = static_cast<int>(pt[0]);
    if (current_bin_ < 0 ||
        static_cast<std::size_t>(current_bin_) >= s_i_.size())
      throw std::out_of_range(
        "NumCountsFullLtmz: bin_index outside the configured bin set");
  }

  double
  operator()(double lt, double zt, double lnM) const
  {
    double const s_j = y3_cluster::richness_zkernel(
      zt, zob_min_[current_bin_], zob_max_[current_bin_],
      sigma_z_[current_bin_]);
    return (*hmf_)(lnM, zt) * (*dv_do_dz_)(zt) * (*omega_z_)(zt) * s_j *
           s_i_[current_bin_](lt, zt, *plob_) * (*mor_)(lt, lnM, zt);
  }

  static char const*
  module_label()
  {
    return "NumCountsFullLtmz";
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
    return y3_cluster::make_grid_points_wall_of_numbers(cfg, module_label(),
                                                        "bin_index");
  }
};

#endif

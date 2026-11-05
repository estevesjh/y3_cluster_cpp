// GPU 1h+2h max-model shear — full (lambda_true, lnM, z) reference, PAGANI
//
// GPU: the whole integral, one adaptive triple integral per (bin, R)
//   O_ij(R) = int dlt dzt dlnM n(M,zt) dV/dOmegadz(zt) Omega(zt)
//             * Sigma_crit^-1(zt) S_j(zt) S_i(lt,zt) P_HOD(lt | M,zt)
//             * d_tot(R, lnM, zt | bin)
//   - max model, piecewise around r_trans:
//       R <  r_trans: DSigma_1h + Delta_gap^1 (R / r_trans)^2
//       R >= r_trans: b DSigma_hh + (C_0 / R^2) 1e-12
//     falls back to max(1h, 2h) when the r_trans tables are absent
//   - DSigma_1h = (1 - f_mis) DSigma_nfw + f_mis DSigma_mis, mixed BEFORE the max
//
// CPU: models and tables from the datablock.
//
// Note: Bsel is not implemented here; bsel = T throws.
// Output: shear1h2hmax3dgpu/{vals, errors, probs, status, nregions}.

#ifndef Y3_CLUSTER_CPP_SHEAR1H2H_MAX_3D_GPU_T_CUH
#define Y3_CLUSTER_CPP_SHEAR1H2H_MAX_3D_GPU_T_CUH

#include "utils/datablock_reader.hh"
#include "utils/make_cuda_integration_volumes.cuh"
#include "utils/make_grid_points.hh"
#include "utils/make_interp_1d.cuh"
#include "utils/make_interp_2d.cuh"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/datablock_status.h"
#include "cosmosis/datablock/ndarray.hh"
#include "cubacpp/integration_result.hh"
#include "common/cuda/Volume.cuh"

#include "models/dv_do_dz_t.cuh"
#include "models/emg_des_t.cuh"
#include "models/hmf_t.cuh"
#include "models/mor_hod_t.cuh"
#include "models/nfw_dsigma_mis.cuh"
#include "models/omega_z_des.cuh"

// zkernel_sj (observed-redshift kernel S_j).
#include "pipelines/des_y3/number_counts/cuda/3d/num_counts_3d_gpu_t.cuh"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

class Shear1h2hMax3dGpu {
public:
  using grid_t = y3_cluster::grid_t<2>;
  using grid_point_t = grid_t::value_type;

private:
  using volume_t = quad::Volume<double, 3>;
  static constexpr int MAX_BINS = 32;

  int n_bins_{0};
  double lam_min_[MAX_BINS], lam_max_[MAX_BINS];
  double zob_min_[MAX_BINS], zob_max_[MAX_BINS], sigma_z_[MAX_BINS];
  double lob_centers_[MAX_BINS];
  int n_lob_{0};
  bool include_mis_{true};

  std::optional<y3_cuda::HMF_t> hmf_;
  std::optional<y3_cuda::DV_DO_DZ_t> dv_do_dz_;
  std::optional<y3_cuda::OMEGA_Z_DES> omega_z_;
  std::optional<quad::Interp2D> dsigma_nfw_;
  std::optional<quad::Interp1D> sci_;
  std::optional<y3_cuda::NFW_DSIGMA_MIS> dsigma_mis_;
  bool use_halo_model_conc_ = false;   // issue #14, opt-in diagnostic leg
  std::optional<y3_cuda::MOR_HOD_t> mor_;
  std::optional<y3_cuda::EMG_DES_t> emg_;

  // Two-halo term and the piecewise max-model tables.
  std::optional<quad::Interp2D> bias_, dsigma_hh_;
  std::optional<quad::Interp2D> r_trans_, c0_trans_, delta_gap_1_;
  bool has_piecewise_{false};

  double f_mis_{0.0}, tau_mis_{0.0};
  bool phys_density_{false};

  double cur_lam_min_{0}, cur_lam_max_{0}, cur_zob_min_{0}, cur_zob_max_{0},
    cur_sigma_z_{1}, cur_R_{0}, cur_r_mis_{0};

public:
  explicit Shear1h2hMax3dGpu(cosmosis::DataBlock& cfg)
  {
    auto const lam_min = get_vector_double(cfg, module_label(), "lam_min");
    auto const lam_max = get_vector_double(cfg, module_label(), "lam_max");
    auto const zob_min = get_vector_double(cfg, module_label(), "zob_min");
    auto const zob_max = get_vector_double(cfg, module_label(), "zob_max");
    auto const sigma_z = get_vector_double(cfg, module_label(), "sigma_z");
    std::size_t const n = lam_min.size();
    if (n > MAX_BINS || lam_max.size() != n || zob_min.size() != n ||
        zob_max.size() != n || sigma_z.size() != n)
      throw std::runtime_error("Shear1h2hMax3dGpu: bad bin arrays");
    n_bins_ = static_cast<int>(n);
    for (std::size_t i = 0; i != n; ++i) {
      lam_min_[i] = lam_min[i];
      lam_max_[i] = lam_max[i];
      zob_min_[i] = zob_min[i];
      zob_max_[i] = zob_max[i];
      sigma_z_[i] = sigma_z[i];
    }
    std::vector<double> lob{25.0, 37.5, 52.5, 130.0};
    if (cfg.has_val(module_label(), "lob_centers"))
      lob = get_vector_double(cfg, module_label(), "lob_centers");
    if (lob.empty() || lob.size() > MAX_BINS)
      throw std::runtime_error("Shear1h2hMax3dGpu: bad lob_centers");
    n_lob_ = static_cast<int>(lob.size());
    for (int i = 0; i != n_lob_; ++i) lob_centers_[i] = lob[i];
    use_halo_model_conc_ =
      read_bool_opt(cfg, module_label(), "use_halo_model_conc", false);
    include_mis_ =
      read_bool_opt(cfg, module_label(), "include_miscentering", true);
    if (read_bool_opt(cfg, module_label(), "bsel", false))
      throw std::runtime_error(
        "Shear1h2hMax3dGpu: bsel = T is not implemented yet (phase 1)");
  }

  // Same tolerant reader as the CPU twin: accepts T/F, 0/1 or a string.
  static bool
  read_bool_opt(cosmosis::DataBlock& cfg,
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
  set_sample(cosmosis::DataBlock& s)
  {
    hmf_.emplace(s);
    dv_do_dz_.emplace(s);
    omega_z_.emplace(s);
    dsigma_nfw_.emplace(
      make_Interp2D(s, "haloModel", "r_sigma", "lnM", "dSigma_nfw"));
    sci_.emplace(make_Interp1D(s, "average_sigma_crit_inv", "zlense",
                               "sci_average"));
    dsigma_mis_.emplace(y3_cuda::DSIGMA_MIS_CONC, y3_cuda::DSIGMA_MIS_RHOC,
                        y3_cuda::DSIGMA_MIS_GAMMA);
    // UNIFIED rho_m convention (2026-08-24): boundary AND amplitude on
    // haloModel/rho_m_ref (same density as the centred dSigma_nfw table).
    dsigma_mis_->set_rho_ref(s.view<double>("haloModel", "rho_m_ref"));
    if (use_halo_model_conc_)
      dsigma_mis_->set_concentration_table(s);
    mor_.emplace(s);
    emg_.emplace(s);

    bias_.emplace(make_Interp2D(s, "haloModel", "lnM", "z", "bias"));
    dsigma_hh_.emplace(make_sanitized_table(s, "dSigma_hh"));
    has_piecewise_ = s.has_val("haloModel", "r_trans");
    if (has_piecewise_) {
      r_trans_.emplace(make_Interp2D(s, "haloModel", "lnM", "z", "r_trans"));
      c0_trans_.emplace(
        make_Interp2D(s, "haloModel", "lnM", "z", "c0_trans"));
      delta_gap_1_.emplace(
        make_Interp2D(s, "haloModel", "lnM", "z", "delta_gap_1"));
    } else {
      r_trans_.reset();
      c0_trans_.reset();
      delta_gap_1_.reset();
    }

    // Required: no fallback to the fiducial defaults — a pipeline that
    // has not published the miscentering section must fail loudly.
    f_mis_ = include_mis_ ? s.view<double>("miscentering", "f_mis") : 0.0;
    tau_mis_ = s.view<double>("miscentering", "tau_mis");
    // Physical mean density (opt-in): exact per-zt identity on the
    // 1-halo mixture ONLY (the 2-halo term keeps its own convention).
    int phys = 0;
    if (s.has_val("haloModel", "one_halo_physical_density"))
      s.get_val("haloModel", "one_halo_physical_density", phys);
    phys_density_ = (phys != 0);
  }

  void
  set_grid_point(grid_point_t const& pt)
  {
    int const b = static_cast<int>(pt[0]);
    if (b < 0 || b >= n_bins_)
      throw std::out_of_range("Shear1h2hMax3dGpu: bin_index");
    cur_lam_min_ = lam_min_[b];
    cur_lam_max_ = lam_max_[b];
    cur_zob_min_ = zob_min_[b];
    cur_zob_max_ = zob_max_[b];
    cur_sigma_z_ = sigma_z_[b];
    cur_R_ = pt[1];
    cur_r_mis_ =
      tau_mis_ * std::pow(lob_centers_[b % n_lob_] / 100.0, 0.2);
  }

  __host__ __device__ double
  operator()(double lt, double zt, double lnM) const
  {
    double const s_j = y3_cuda_des_y3::zkernel_sj(zt, cur_zob_min_,
                                                  cur_zob_max_, cur_sigma_z_);
    double const s_i = emg_->cdf(cur_lam_max_, lt, zt) -
                       emg_->cdf(cur_lam_min_, lt, zt);
    double const q = phys_density_ ? 1.0 + zt : 1.0;
    double const d_cen = dsigma_nfw_->clamp(cur_R_ * q, lnM);
    double const d_mis = (*dsigma_mis_)(cur_R_ * q, cur_r_mis_ * q, lnM);
    double const d_1h = (q * q) *
                        ((1.0 - f_mis_) * d_cen + f_mis_ * d_mis);
    double const d_2h =
      bias_->clamp(lnM, zt) * dsigma_hh_->clamp(cur_R_, zt);
    double d_tot;
    if (has_piecewise_) {
      double const rt = r_trans_->clamp(lnM, zt);
      if (cur_R_ >= rt) {
        double const c0 = c0_trans_->clamp(lnM, zt);
        d_tot = d_2h + (c0 / (cur_R_ * cur_R_)) * 1.0e-12;
      } else {
        double const gap1 = delta_gap_1_->clamp(lnM, zt);
        double const r_ratio = cur_R_ / rt;
        d_tot = d_1h + gap1 * r_ratio * r_ratio;
      }
    } else {
      d_tot = fmax(d_1h, d_2h);
    }
    return (*hmf_)(lnM, zt) * (*dv_do_dz_)(zt) *
           (*omega_z_)(zt) * sci_->clamp(zt) * s_j *
           fmax(0.0, s_i) * (*mor_)(lt, lnM, zt) * d_tot;
  }

  static char const* module_label() { return "Shear1h2hMax3dGpu"; }

  static std::vector<volume_t>
  make_integration_volumes(cosmosis::DataBlock& cfg)
  {
    return y3_cuda::make_integration_volumes_wall_of_numbers(
      cfg, module_label(), "lt", "zt", "lnm");
  }

  static grid_t
  make_grid_points(cosmosis::DataBlock& cfg)
  {
    return y3_cluster::make_grid_points_wall_of_numbers(
      cfg, module_label(), "bin_index", "r_perp");
  }

private:
  // haloModel (r_sigma, z) table with NaN/inf set to 0, as in the CPU twin.
  static quad::Interp2D
  make_sanitized_table(cosmosis::DataBlock& s, char const* key)
  {
    auto const r = s.view<doubles>("haloModel", "r_sigma");
    auto const z = s.view<doubles>("haloModel", "z");
    auto const& nd = s.view<cosmosis::ndarray<double>>("haloModel", key);
    std::vector<double> vals(nd.begin(), nd.end());
    if (vals.size() != r.size() * z.size())
      throw std::runtime_error(std::string("Shear1h2hMax3dGpu: ") + key +
                               " extents mismatch");
    for (auto& v : vals)
      if (!std::isfinite(v)) v = 0.0;
    // Same brace-construction as make_Interp2D (make_interp_2d.cuh).
    return {r, z, vals};
  }
};

#endif

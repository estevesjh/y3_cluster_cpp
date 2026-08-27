#ifndef Y3_CLUSTER_CPP_SHEAR1H_3D_GPU_T_CUH
#define Y3_CLUSTER_CPP_SHEAR1H_3D_GPU_T_CUH

#include "utils/datablock_reader.hh"
#include "utils/make_cuda_integration_volumes.cuh"
#include "utils/make_grid_points.hh"
#include "utils/make_interp_1d.cuh"
#include "utils/make_interp_2d.cuh"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/datablock_status.h"
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
#include <vector>

class Shear1h3dGpu {
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

  std::optional<y3_cuda::HMF_t> hmf_;
  std::optional<y3_cuda::DV_DO_DZ_t> dv_do_dz_;
  std::optional<y3_cuda::OMEGA_Z_DES> omega_z_;
  std::optional<quad::Interp2D> dsigma_nfw_;
  std::optional<quad::Interp1D> sci_;
  // Issue #14 (CUDA half): the concentration-table mirror in
  // nfw_dsigma_mis.cuh is wired below as an OPT-IN path
  // (use_halo_model_conc); the production default stays the fixed-c
  // kernel (DSIGMA_MIS_CONC) so cross-backend identity pins stay valid.
  std::optional<y3_cuda::NFW_DSIGMA_MIS> dsigma_mis_;
  bool use_halo_model_conc_ = false;   // issue #14, opt-in diagnostic leg
  // Device HOD MOR (central-shifted Costanzi-2019 form) — the SAME
  // convention as the CPU twin's y3_cluster::MOR_HOD_t, so the
  // CPU<->GPU cross-backend pins compare identical physics. The
  // no-central MOR_SAT_ONLY_t is reserved for the P[X]/b_sel
  // operators (p_operator_gpu_t.cuh).
  std::optional<y3_cuda::MOR_HOD_t> mor_;
  std::optional<y3_cuda::EMG_DES_t> emg_;
  double f_mis_{0.0}, tau_mis_{0.0};
  bool phys_density_{false};

  double cur_lam_min_{0}, cur_lam_max_{0}, cur_zob_min_{0}, cur_zob_max_{0},
    cur_sigma_z_{1}, cur_R_{0}, cur_r_mis_{0};

public:
  explicit Shear1h3dGpu(cosmosis::DataBlock& cfg)
  {
    auto const lam_min = get_vector_double(cfg, module_label(), "lam_min");
    auto const lam_max = get_vector_double(cfg, module_label(), "lam_max");
    auto const zob_min = get_vector_double(cfg, module_label(), "zob_min");
    auto const zob_max = get_vector_double(cfg, module_label(), "zob_max");
    auto const sigma_z = get_vector_double(cfg, module_label(), "sigma_z");
    std::size_t const n = lam_min.size();
    if (n > MAX_BINS || lam_max.size() != n || zob_min.size() != n ||
        zob_max.size() != n || sigma_z.size() != n)
      throw std::runtime_error("Shear1h3dGpu: bad bin arrays");
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
      throw std::runtime_error("Shear1h3dGpu: bad lob_centers");
    n_lob_ = static_cast<int>(lob.size());
    for (int i = 0; i != n_lob_; ++i) lob_centers_[i] = lob[i];
    // issue #14: honor use_halo_model_conc (per-mass c(lnM) into the
    // miscentered NFW); default keeps fixed c=4.
    use_halo_model_conc_ =
        cfg.has_val(module_label(), "use_halo_model_conc") &&
        cfg.view<bool>(module_label(), "use_halo_model_conc");
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
    // issue #14 (opt-in): per-mass c(lnM) from haloModel/concentration;
    // default (flag off) keeps the fixed-c production path.
    if (use_halo_model_conc_)
      dsigma_mis_->set_concentration_table(s);
    mor_.emplace(s);
    emg_.emplace(s);
    // Required: no fallback to the fiducial defaults — a pipeline that
    // has not published the miscentering section must fail loudly.
    f_mis_ = s.view<double>("miscentering", "f_mis");
    tau_mis_ = s.view<double>("miscentering", "tau_mis");
    // Physical mean density (opt-in): exact per-zt identity in the
    // integrand, DSigma_phys(R|zt) = (1+zt)^2 DSigma_com(R (1+zt)).
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
      throw std::out_of_range("Shear1h3dGpu: bin_index");
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
    double const d_tot = (q * q) *
                         ((1.0 - f_mis_) * d_cen + f_mis_ * d_mis);
    return (*hmf_)(lnM, zt) * (*dv_do_dz_)(zt) *
           (*omega_z_)(zt) * sci_->clamp(zt) * s_j *
           fmax(0.0, s_i) * (*mor_)(lt, lnM, zt) * d_tot;
  }

  static char const* module_label() { return "Shear1h3dGpu"; }

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
};

#endif

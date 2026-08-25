// Full (lambda_true, lnM, z) reference number counts — explicit 3d adaptive (formerly `explicit-3d`), CUDA.
//
// PAGANI backend of the same triple integral as the C++/Cuhre module
// (../cpp/num_counts_3d_t.hh):
//
//   N_ij = ∫∫∫ dlt dzt dlnM  n(M,zt) · dV/dΩdz(zt) · Ω(zt)
//                            · S_j(zt) · S_i(lt, zt) · P_HOD(lt | M, zt)
//
// HMF, dV/dΩdz and Ω(z) come from the existing y3_cuda device models
// (same conventions as their host twins, including the HMF mass-axis
// shift and hmf_s/hmf_q nuisance). The HOD and the EMG observed-richness
// kernel S_i are the gpu_prj_costanzi2026 device models (Arwa Qadi,
// upstream PR #3): y3_cuda::MOR_HOD_t (central-shifted Costanzi-2019
// shifted-Poisson HOD, the device mirror of the CPU MOR_HOD_t —
// same cluster_mor input contract, same algebra) and
// y3_cuda::EMG_DES_t (analytic EMG CDF over plob_ltr_params, so
// S_i = F(lam_max | lt, zt) − F(lam_min | lt, zt) with no lambda_ob
// quadrature).
//
// MOR convention note: this backend previously used
// MOR_SHIFTED_POISSON_t — the Costanzi-2026 P-operator form
// (p_operator_t.hh, x = ltr + δ, no central-count shift) — which made
// the GPU integrand the CPU one offset by one ltr unit
// (MOR_SP(ltr) = MOR_HOD(ltr + 1) above Mmin; identity pinned in
// test/num_counts_3d_gpu.test.cu). The explicit-3d backends now all
// use the HOD form so CPU↔GPU cross-backend pins compare identical
// physics; MOR_SHIFTED_POISSON_t remains the P[X]/b_sel operator MOR
// (p_operator_gpu_t.cuh), matching its CPU twin.
//
// The observed-redshift kernel S_j is the 3-line Gaussian
// CDF difference below, verbatim the CPU richness_zkernel with the
// per-bin constant sigma_z this backend is validated against — NOT the
// σ(z)-model INT_ZO_ZT_DES_t, which is a different photo-z convention.
//
// Configuration is identical to the C++ backend except the section
// name (NumCounts3dGpu) and algorithm = pagani; the two backends
// are validated against each other and against the Python reference.
// Output: numcounts3dgpu/{vals, errors, ...}, one entry per bin.
//
// Status: reference backend. Not a production entry point.
#ifndef Y3_CLUSTER_CPP_NUM_COUNTS_3D_GPU_T_CUH
#define Y3_CLUSTER_CPP_NUM_COUNTS_3D_GPU_T_CUH

#include "utils/datablock_reader.hh"
#include "utils/make_cuda_integration_volumes.cuh"
#include "utils/make_grid_points.hh"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/datablock_status.h"
#include "cubacpp/integration_result.hh"
#include "common/cuda/Volume.cuh"

#include "models/dv_do_dz_t.cuh"
#include "models/emg_des_t.cuh"
#include "models/hmf_t.cuh"
#include "models/mor_hod_t.cuh"
#include "models/omega_z_des.cuh"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

namespace y3_cuda_des_y3 {

  // Observed-redshift kernel S_j — verbatim the CPU richness_zkernel
  // (Gaussian CDF difference at the per-bin constant sigma_z).
  __host__ __device__ inline double
  zkernel_sj(double zt, double zob_min, double zob_max, double sigma_z)
  {
    constexpr double SQRT2_INV = 0.7071067811865475;
    return 0.5 * (erf((zob_max - zt) / sigma_z * SQRT2_INV) -
                  erf((zob_min - zt) / sigma_z * SQRT2_INV));
  }

}  // namespace y3_cuda_des_y3

class NumCounts3dGpu {
public:
  using grid_t = y3_cluster::grid_t<1>;
  using grid_point_t = grid_t::value_type;

private:
  using volume_t = quad::Volume<double, 3>;
  static constexpr int MAX_BINS = 32;

  // Bin definitions from configuration (fixed-size so the integrand
  // object stays trivially copyable to the device).
  int n_bins_{0};
  double lam_min_[MAX_BINS], lam_max_[MAX_BINS];
  double zob_min_[MAX_BINS], zob_max_[MAX_BINS], sigma_z_[MAX_BINS];

  // Per-sample models (existing device models + PR #3 device models).
  std::optional<y3_cuda::HMF_t> hmf_;
  std::optional<y3_cuda::DV_DO_DZ_t> dv_do_dz_;
  std::optional<y3_cuda::OMEGA_Z_DES> omega_z_;
  std::optional<y3_cuda::MOR_HOD_t> mor_;
  std::optional<y3_cuda::EMG_DES_t> emg_;

  // Current-bin scalars, set by set_grid_point.
  double cur_lam_min_{0.0}, cur_lam_max_{0.0};
  double cur_zob_min_{0.0}, cur_zob_max_{0.0}, cur_sigma_z_{1.0};

public:
  explicit NumCounts3dGpu(cosmosis::DataBlock& cfg)
  {
    auto const lam_min = get_vector_double(cfg, module_label(), "lam_min");
    auto const lam_max = get_vector_double(cfg, module_label(), "lam_max");
    auto const zob_min = get_vector_double(cfg, module_label(), "zob_min");
    auto const zob_max = get_vector_double(cfg, module_label(), "zob_max");
    auto const sigma_z = get_vector_double(cfg, module_label(), "sigma_z");
    std::size_t const n = lam_min.size();
    if (lam_max.size() != n || zob_min.size() != n || zob_max.size() != n ||
        sigma_z.size() != n)
      throw std::runtime_error(
        "NumCounts3dGpu: bin definition arrays have unequal lengths");
    if (n > MAX_BINS)
      throw std::runtime_error("NumCounts3dGpu: too many bins");
    n_bins_ = static_cast<int>(n);
    for (std::size_t i = 0; i != n; ++i) {
      lam_min_[i] = lam_min[i];
      lam_max_[i] = lam_max[i];
      zob_min_[i] = zob_min[i];
      zob_max_[i] = zob_max[i];
      sigma_z_[i] = sigma_z[i];
    }
  }

  void
  set_sample(cosmosis::DataBlock& sample)
  {
    hmf_.emplace(sample);
    dv_do_dz_.emplace(sample);
    omega_z_.emplace(sample);
    mor_.emplace(sample);
    emg_.emplace(sample);
  }

  void
  set_grid_point(grid_point_t const& pt)
  {
    int const b = static_cast<int>(pt[0]);
    if (b < 0 || b >= n_bins_)
      throw std::out_of_range(
        "NumCounts3dGpu: bin_index outside the configured bin set");
    cur_lam_min_ = lam_min_[b];
    cur_lam_max_ = lam_max_[b];
    cur_zob_min_ = zob_min_[b];
    cur_zob_max_ = zob_max_[b];
    cur_sigma_z_ = sigma_z_[b];
  }

  __host__ __device__ double
  operator()(double lt, double zt, double lnM) const
  {
    double const s_j = y3_cuda_des_y3::zkernel_sj(zt, cur_zob_min_,
                                                  cur_zob_max_, cur_sigma_z_);
    double const s_i = emg_->cdf(cur_lam_max_, lt, zt) -
                       emg_->cdf(cur_lam_min_, lt, zt);
    return (*hmf_)(lnM, zt) * (*dv_do_dz_)(zt) * (*omega_z_)(zt) * s_j *
           fmax(0.0, s_i) * (*mor_)(lt, lnM, zt);
  }

  static char const*
  module_label()
  {
    return "NumCounts3dGpu";
  }

  static std::vector<volume_t>
  make_integration_volumes(cosmosis::DataBlock& cfg)
  {
    return y3_cuda::make_integration_volumes_wall_of_numbers(
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

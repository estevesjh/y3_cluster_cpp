// Full (lambda_true, lnM, z) reference number counts — `full_ltmz`, CUDA.
//
// PAGANI backend of the same triple integral as the C++/Cuhre module
// (../cpp/NumCountsFullLtmz.cc):
//
//   N_ij = ∫∫∫ dlt dzt dlnM  n(M,zt) · dV/dΩdz(zt) · Ω(zt)
//                            · K_j(zt) · K_i(lt, zt) · P_HOD(lt | M, zt)
//
// HMF, dV/dΩdz and Ω(z) come from the existing y3_cuda device models
// (same conventions as their host twins, including the HMF mass-axis
// shift and hmf_s/hmf_q nuisance); the HOD, EMG richness kernel and
// photo-z kernel have no pre-existing device versions and are the
// verbatim ports in full_ltmz_device_kernels.cuh (des_y3 scoped — no
// existing model or template is modified).
//
// Configuration is identical to the C++ backend except the section
// name (NumCountsFullLtmzGpu) and algorithm = pagani; the two backends
// are validated against each other and against the Python reference.
// Output: numcountsfullltmzgpu/{vals, errors, ...}, one entry per bin.
//
// Status: reference backend. Not a production entry point.
#include "utils/cuda_module_macros.cuh"
#include "utils/datablock_reader.hh"
#include "utils/make_cuda_integration_volumes.cuh"
#include "utils/make_grid_points.hh"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/datablock_status.h"
#include "cubacpp/integration_result.hh"
#include "common/cuda/Volume.cuh"

#include "models/dv_do_dz_t.cuh"
#include "models/hmf_t.cuh"
#include "models/omega_z_des.cuh"

#include "pipelines/des_y3/observables/number_counts/full_ltmz/cuda/full_ltmz_device_kernels.cuh"

#include <optional>
#include <stdexcept>
#include <vector>

class NumCountsFullLtmzGpu {
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

  // Per-sample models (existing device models + the des_y3 ports).
  std::optional<y3_cuda::HMF_t> hmf_;
  std::optional<y3_cuda::DV_DO_DZ_t> dv_do_dz_;
  std::optional<y3_cuda::OMEGA_Z_DES> omega_z_;
  y3_cuda_des_y3::MorHodDevice mor_;
  y3_cuda_des_y3::PlobEmgDevice plob_;

  // Current-bin scalars, set by set_grid_point.
  double cur_lam_min_{0.0}, cur_lam_max_{0.0};
  double cur_zob_min_{0.0}, cur_zob_max_{0.0}, cur_sigma_z_{1.0};

public:
  explicit NumCountsFullLtmzGpu(cosmosis::DataBlock& cfg)
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
        "NumCountsFullLtmzGpu: bin definition arrays have unequal lengths");
    if (n > MAX_BINS)
      throw std::runtime_error("NumCountsFullLtmzGpu: too many bins");
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
    mor_ = y3_cuda_des_y3::MorHodDevice::from_datablock(sample);
    plob_ = y3_cuda_des_y3::PlobEmgDevice::from_datablock(sample);
  }

  void
  set_grid_point(grid_point_t const& pt)
  {
    int const b = static_cast<int>(pt[0]);
    if (b < 0 || b >= n_bins_)
      throw std::out_of_range(
        "NumCountsFullLtmzGpu: bin_index outside the configured bin set");
    cur_lam_min_ = lam_min_[b];
    cur_lam_max_ = lam_max_[b];
    cur_zob_min_ = zob_min_[b];
    cur_zob_max_ = zob_max_[b];
    cur_sigma_z_ = sigma_z_[b];
  }

  __host__ __device__ double
  operator()(double lt, double zt, double lnM) const
  {
    double const k_j = y3_cuda_des_y3::zkernel(zt, cur_zob_min_,
                                               cur_zob_max_, cur_sigma_z_);
    double const mu = plob_.mu(lt, zt);
    double const sigma = plob_.sigma(lt, zt);
    double const tau = plob_.tau(lt, zt);
    double const fprj = plob_.fprj(lt, zt);
    double const k_i = y3_cuda_des_y3::richness_kernel(
      cur_lam_min_, cur_lam_max_, mu, sigma, tau, fprj);
    return (*hmf_)(lnM, zt) * (*dv_do_dz_)(zt) * (*omega_z_)(zt) * k_j *
           k_i * mor_(lt, lnM, zt);
  }

  static char const*
  module_label()
  {
    return "NumCountsFullLtmzGpu";
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

DEFINE_COSMOSIS_CUDA_INTEGRATION_MODULE(NumCountsFullLtmzGpu)

// Shear1hMisSel GPU module with 3D integration.
//
// Integrating over (lt, zt, lnM). The lo (observed richness) integral is done
// analytically via the EMG CDF, matching the pattern used in numberCountsFull_t.
//
// Output:
//
//   N_i[gamma_1h_full](R) = int dlt int dzt int dlnM
//       Omega(z) * (dV/dOmega/dz)(z) * n(M,z)
//       * P_HOD(lt | M,z)
//       * K_i(lo_low, lo_high | lt, z)     [EMG CDF difference]
//       * K_j(zo_low, zo_high | z)          [photo-z kernel]
//       * gamma_1h_full(R, M, z)
//
// Important notes for future Arwa: gamma_prj is NOT included here.
// Final theory is: gamma_theory = shear1hmissel / numcountssel + shear_prj
//
// Grid: (lo_bin_low, lo_bin_high, zo_low, zo_high, radius)

#include "utils/make_cuda_integration_volumes.cuh"
#include "utils/datablock_reader.hh"
#include "utils/cuda_module_macros.cuh"
#include "utils/make_grid_points.hh"
#include "utils/read_vector.hh"

#include "cubacpp/integration_result.hh"
#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/datablock_status.h"
#include "common/cuda/Volume.cuh"

#include "models/omega_z_des.cuh"
#include "models/dv_do_dz_t.cuh"
#include "models/hmf_t.cuh"
#include "models/mor_des_log_t.cuh"
#include "models/emg_des_t.cuh"        // EMG CDF for analytic lo integral
#include "models/int_zo_zt_des_t.cuh"  // photo-z kernel
#include "models/gamma_1h_nfw.cuh"

#include <optional>
#include <vector>

using cosmosis::DataBlock;
using cosmosis::ndarray;
using cubacpp::integration_result;

class Shear1hMisSel {
public:
  // Grid point: (lo_bin_low, lo_bin_high, zo_low, zo_high, radius)
  using grid_t = y3_cluster::grid_t<5>;
  using grid_point_t = grid_t::value_type;

private:
  // 3D integration volume: (lt, zt, lnM) — lo integrated analytically via CDF.
  using volume_t = quad::Volume<double, 3>;

  std::optional<y3_cuda::OMEGA_Z_DES>     omega_z_;
  std::optional<y3_cuda::DV_DO_DZ_t>      dv_do_dz_;
  std::optional<y3_cuda::HMF_t>           hmf_;
  std::optional<y3_cuda::MOR_DES_LOG_t>   mor_;
  std::optional<y3_cuda::EMG_DES_t>       emg_;
  std::optional<y3_cuda::INT_ZO_ZT_DES_t> int_zo_zt_;
  std::optional<y3_cuda::GAMMA_1H_NFW>    gamma_1h_;

  // Current bin edges and radius from grid point.
  double lo_low_;
  double lo_high_;
  double zo_low_;
  double zo_high_;
  double radius_;

public:
  explicit Shear1hMisSel(cosmosis::DataBlock& /*cfg*/) {}

  void set_sample(cosmosis::DataBlock& sample) {
    omega_z_.emplace(sample);
    dv_do_dz_.emplace(sample);
    hmf_.emplace(sample);
    mor_.emplace(sample);
    emg_.emplace(sample);
    int_zo_zt_.emplace();
    gamma_1h_.emplace(sample);
  }

  void set_grid_point(grid_point_t const& pt) {
    lo_low_  = pt[0];
    lo_high_ = pt[1];
    zo_low_  = pt[2];
    zo_high_ = pt[3];
    radius_  = pt[4];

    // Miscentering R_mis = tau_mis * R_lambda(lambda^ob) depends on the
    // richness bin, not on the integration variables -- push the bin's
    // arithmetic centre into gamma_1h_ each time the grid point changes.
    gamma_1h_->set_lob_centre(0.5 * (lo_low_ + lo_high_));
  }

  __host__ __device__ double
  operator()(double lt, double zt, double lnM) const
  {
    // Richness selection kernel: EMG CDF difference over [lo_low, lo_high].
    double const Ki_raw = emg_->cdf(lo_high_, lt, zt) - emg_->cdf(lo_low_, lt, zt);
    double const Ki = fmin(1.0, fmax(0.0, Ki_raw));
    if (Ki <= 0.0) return 0.0;

    // Photo-z kernel: integral of P(zo | zt) over [zo_low, zo_high].
    double const Kj = (*int_zo_zt_)(zo_low_, zo_high_, zt);
    if (Kj <= 0.0) return 0.0;

    // MOR: P_HOD(lt | M, z).
    double const p_hod = (*mor_)(lt, lnM, zt);
    if (p_hod <= 0.0) return 0.0;

    // Cosmology factors.
    double const common = (*omega_z_)(zt) * (*dv_do_dz_)(zt) * (*hmf_)(lnM, zt);

    // 1-halo shear only. The projection shear is added later, after this
    // numerator is divided by NumCountsSel.
    double const g1h = (*gamma_1h_)(radius_, lnM, zt);

    return common * p_hod * Ki * Kj * (isfinite(g1h) ? g1h : 0.0);
  }

  static char const* module_label() { return "shear1hmissel"; }

  static std::vector<volume_t>
  make_integration_volumes(cosmosis::DataBlock& cfg) {
    return y3_cuda::make_integration_volumes_wall_of_numbers(
      cfg, module_label(), "ltr", "zt", "lnm");
  }

  static grid_t make_grid_points(cosmosis::DataBlock& cfg) {
    return y3_cluster::make_grid_points_wall_of_numbers(
      cfg, module_label(),
      "lob_bin_low", "lob_bin_high",
      "zo_low", "zo_high",
      "radii");
  }
};

DEFINE_COSMOSIS_CUDA_INTEGRATION_MODULE(Shear1hMisSel)

// Number counts GPU module with 3D integration.
// Integrating over (lambda_tr, z, lnM). 
//
// Integrand:
//   N = int dltr int dz int dlnM
//       Omega(z) * (dV/dOmega/dz)(z) * n(M,z)
//       * P_HOD(ltr|M,z) * P_lob_bin(lob_bin|ltr,z) * K(zo|z)
//
// where:
//   - P_HOD(ltr|M,z) is the MOR: mor_des_log_t
//   - P_lob_bin is the precomputed richness-bin kernel: int_lc_lt_des_t2
//   - K(zo|z) is the photo-z bin kernel: int_zo_zt_des_t
//
// Grid: (lob_low, lob_high, zo_low, zo_high)

#include "utils/cuda_module_macros.cuh"
#include "utils/datablock_reader.hh"
#include "utils/make_cuda_integration_volumes.cuh"
#include "utils/make_grid_points.hh"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/datablock_status.h"
#include "common/cuda/Volume.cuh"

#include "models/omega_z_des.cuh"
#include "models/dv_do_dz_t.cuh"
#include "models/hmf_t.cuh"
#include "models/mor_des_log_t.cuh"
#include "models/int_zo_zt_des_t.cuh"
#include "models/int_lc_lt_des_t2.cuh"

#include <optional>
#include <vector>

using cosmosis::DataBlock;

class numberCountsFull_t {
public:
  using grid_t = y3_cluster::grid_t<4>;
  using grid_point_t = grid_t::value_type;

private:
  using volume_t = quad::Volume<double, 3>;

  std::optional<y3_cuda::OMEGA_Z_DES> omega_z_;
  std::optional<y3_cuda::DV_DO_DZ_t> dv_do_dz_;
  std::optional<y3_cuda::HMF_t> hmf_;
  std::optional<y3_cuda::MOR_DES_LOG_t> mor_;
  std::optional<y3_cuda::INT_LC_LT_DES_t2> lc_lt_;
  std::optional<y3_cuda::INT_ZO_ZT_DES_t> int_zo_zt_;

  double lob_low_;
  double lob_high_;
  double zo_low_;
  double zo_high_;

public:
  explicit numberCountsFull_t(cosmosis::DataBlock& /*cfg*/)
  {
  }

  void set_sample(cosmosis::DataBlock& sample)
  {
    omega_z_.emplace(sample);
    dv_do_dz_.emplace(sample);
    hmf_.emplace(sample);
    mor_.emplace(sample);

    // Reads precomputed richness-bin interpolation tables.
    lc_lt_.emplace(sample);

    // Default constructor is correct.
    int_zo_zt_.emplace();
  }

  void set_grid_point(grid_point_t const& grid_point)
  {
    lob_low_ = grid_point[0];
    lob_high_ = grid_point[1];
    zo_low_ = grid_point[2];
    zo_high_ = grid_point[3];
  }

  __host__ __device__ double
  operator()(double ltr, double zt, double lnM) const
  {
    double const photoz = (*int_zo_zt_)(zo_low_, zo_high_, zt);
    if (photoz <= 0.0) {
      return 0.0;
    }

    double const p_hod = (*mor_)(ltr, lnM, zt);
    if (p_hod <= 0.0) {
      return 0.0;
    }

    // INT_LC_LT_DES_t2 uses lob_low_ only to choose the richness bin:
    // [20,30), [30,45), [45,60), or [60,inf).
    double const richness_bin = (*lc_lt_)(lob_low_, ltr, zt);
    if (richness_bin <= 0.0) {
      return 0.0;
    }

    double const common =
        (*omega_z_)(zt)
      * (*dv_do_dz_)(zt)
      * (*hmf_)(lnM, zt);

    return common * p_hod * richness_bin * photoz;
  }

  static char const* module_label()
  {
    return "numberCountsFull_t";
  }

  static std::vector<volume_t>
  make_integration_volumes(cosmosis::DataBlock& cfg)
  {
    return y3_cuda::make_integration_volumes_wall_of_numbers(
      cfg,
      numberCountsFull_t::module_label(),
      "ltr",
      "zt",
      "lnm");
  }

  static grid_t make_grid_points(cosmosis::DataBlock& cfg)
  {
    return y3_cluster::make_grid_points_wall_of_numbers(
      cfg,
      numberCountsFull_t::module_label(),
      "lob_bin_low",
      "lob_bin_high",
      "zo_low",
      "zo_high");
  }
};

DEFINE_COSMOSIS_CUDA_INTEGRATION_MODULE(numberCountsFull_t)
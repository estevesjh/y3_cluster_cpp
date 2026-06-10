// Number counts GPU module with 3D integration.
//
// Integrating over (lambda_tr, z, lnM). The lambda_ob integral is done
// analytically via the EMG CDF, matching the template pattern used for
// photo-z bins.
//
// Integrand:
//
//   N = int dltr int dz int dlnM
//       Omega(z) * (dV/dOmega/dz)(z) * n(M,z)
//       * P_HOD(ltr | M,z)
//       * P_EMG_bin(lob_low,lob_high | ltr,z)
//       * K(zo | z)
//
// where:
//   - P_HOD(ltr | M,z) is the MOR (mass-observable relation): mor_des_log_t
//   - P_EMG_bin = CDF(lob_high | ltr,z) - CDF(lob_low | ltr,z) is the richness
//     bin kernel
//   - K(zo | z) is the photo-z bin kernel: int_zo_zt_des_t
//
// Grid: (lob_low, lob_high, zo_low, zo_high) - per richness and redshift bin

#include "utils/cuda_module_macros.cuh"
#include "utils/datablock_reader.hh"
#include "utils/make_cuda_integration_volumes.cuh"
#include "utils/make_grid_points.hh"

#include "cubacpp/integration_result.hh"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/datablock_status.h"

#include "common/cuda/Volume.cuh"

#include "models/omega_z_des.cuh"
#include "models/dv_do_dz_t.cuh"
#include "models/hmf_t.cuh"
#include "models/mor_des_log_t.cuh"
#include "models/int_zo_zt_des_t.cuh"
#include "models/emg_des_t.cuh"

#include <iostream>
#include <optional>
#include <vector>

using cosmosis::DataBlock;
using cubacpp::integration_result;

// numberCountsFull_t: 3D integration over (ltr, zt, lnM) with lob integrated
// analytically via EMG CDF.
//
// Follows the CosmoSISScalarIntegrand concept for use with
// CosmoSISScalarIntegrationModule.
class numberCountsFull_t {
public:
    // Grid point: (lob_low, lob_high, zo_low, zo_high)
    // This defines the richness bin [lob_low, lob_high] and redshift bin
    // [zo_low, zo_high].
    using grid_t = y3_cluster::grid_t<4>;
    using grid_point_t = grid_t::value_type;

private:
    // 3D integration volume: (ltr, zt, lnM) - lob integrated analytically via CDF.
    using volume_t = quad::Volume<double, 3>;

    // Models from datablock. std::optional allows deferred initialization.
    std::optional<y3_cuda::OMEGA_Z_DES> omega_z_;
    std::optional<y3_cuda::DV_DO_DZ_t> dv_do_dz_;
    std::optional<y3_cuda::HMF_t> hmf_;
    std::optional<y3_cuda::MOR_DES_LOG_t> mor_;          // P_HOD(ltr | M, z)
    std::optional<y3_cuda::EMG_DES_t> emg_;              // P_EMG(lob | ltr, z)
    std::optional<y3_cuda::INT_ZO_ZT_DES_t> int_zo_zt_;  // Photo-z kernel K(zo | z)

    // Current bin edges from grid point.
    double lob_low_;
    double lob_high_;
    double zo_low_;
    double zo_high_;

public:
    // Initialize integrand object from configuration block.
    // No special configuration is needed at setup time for this module.
    explicit numberCountsFull_t(cosmosis::DataBlock& /*cfg*/) {
        // Configuration parameters could be read here if needed.
    }

    // Set data members from values read from the current sample.
    // Called once per MCMC sample before integration begins.
    void set_sample(cosmosis::DataBlock& sample) {
        omega_z_.emplace(sample);
        dv_do_dz_.emplace(sample);
        hmf_.emplace(sample);
        mor_.emplace(sample);
        emg_.emplace(sample);

        // INT_ZO_ZT_DES_t has default constructor; no datablock needed.
        int_zo_zt_.emplace();
    }

    // Set the data for the current bin/grid point.
    // Called before each integration over a specific bin.
    void set_grid_point(grid_point_t const& grid_point) {
        lob_low_ = grid_point[0];
        lob_high_ = grid_point[1];
        zo_low_ = grid_point[2];
        zo_high_ = grid_point[3];
    }

    // Integrand function called at each (ltr, zt, lnM) point.
    // lob is integrated analytically via CDF difference over the bin.
    __host__ __device__ double operator()(double ltr, double zt, double lnM) const {
        // Photo-z kernel: integral of P(z_ob | z_true) over [zo_low, zo_high].
        double const photoz = (*int_zo_zt_)(zo_low_, zo_high_, zt);
        if (photoz <= 0.0) {
            return 0.0;
        }

        // MOR: P_HOD(lambda_tr | M, z).
        double const p_hod = (*mor_)(ltr, lnM, zt);
        if (p_hod <= 0.0) {
            return 0.0;
        }

        // Richness bin kernel:
        //
        //   P(lob_low < lob < lob_high | ltr, z)
        //     = CDF(lob_high) - CDF(lob_low)
        //
        // fmax guards against tiny negative values from floating-point subtraction.
        double const richness_bin = fmax(
            0.0,
            emg_->cdf(lob_high_, ltr, zt) - emg_->cdf(lob_low_, ltr, zt)
        );

        if (richness_bin == 0.0) {
            return 0.0;
        }

        // Common cosmology factors: Omega(z) * dV/dOmega/dz * dn/dlnM.
        double const common = (*omega_z_)(zt) * (*dv_do_dz_)(zt) * (*hmf_)(lnM, zt);

        return common * p_hod * richness_bin * photoz;
    }

    // Module label - must match section name in pipeline.ini file.
    static char const* module_label() {
        return "numberCountsFull_t";
    }

    // Create 3D integration volumes from ini config.
    // The volume is over (ltr, zt, lnM). lob is integrated via CDF.
    static std::vector<volume_t> make_integration_volumes(cosmosis::DataBlock& cfg) {
        return y3_cuda::make_integration_volumes_wall_of_numbers(
            cfg,
            numberCountsFull_t::module_label(),
            "ltr",
            "zt",
            "lnm"
        );
    }

    // Create grid points from ini config.
    // Each grid point is (lob_low, lob_high, zo_low, zo_high) for a bin.
    static grid_t make_grid_points(cosmosis::DataBlock& cfg) {
        return y3_cluster::make_grid_points_wall_of_numbers(
            cfg,
            numberCountsFull_t::module_label(),
            "lob_bin_low",
            "lob_bin_high",
            "zo_low",
            "zo_high"
        );
    }
};

DEFINE_COSMOSIS_CUDA_INTEGRATION_MODULE(numberCountsFull_t)

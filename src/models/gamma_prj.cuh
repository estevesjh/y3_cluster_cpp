// gamma_prj.cuh -- 2-halo shear term with selection bias correction
//
// Computes gamma_prj = DSigma_hh * bias * b_sel(theta) * sigma_crit_inv
//
// Where b_sel(theta) = B_small + (B_large - B_small) * sigmoid(theta)
// and B_small, B_large come from bSelMargGPU.cu output.
//
// Usage:
//   GAMMA_PRJ gamma_prj(sample);
//   gamma_prj.set_grid_point(zo_low, zo_high);
//   double val = gamma_prj(r, lnM, zt, lo);  // lo = observed richness

#ifndef Y3_CLUSTER_GAMMA_PRJ_CUH
#define Y3_CLUSTER_GAMMA_PRJ_CUH

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"
#include "common/cuda/Interp2D.cuh"
#include "utils/make_interp_1d.cuh"
#include "utils/make_interp_2d.cuh"
#include "utils/datablock_reader.hh"
#include "models/b_sel.cuh"  // For b_sel_sigmoid, R_lambda, theta_lob

#include <cmath>
#include <vector>

namespace y3_cuda {

  class GAMMA_PRJ {
  private:
    // 2-halo surface density DSigma_hh(r, z)
    quad::Interp2D dsigma_hh_;

    // Halo bias b(lnM, z)
    quad::Interp2D bias_;

    // Inverse critical surface density sigma_crit_inv(z, r)
    quad::Interp2D sigma_crit_inv_;

    // Comoving distance chi(z) for theta calculation
    quad::Interp1D chi_;

    // B_small, B_large from bSelMargGPU - stored as vectors for lookup
    // Shape: (n_grid,) where grid is (zob, lob) pairs
    std::vector<double> b_small_vec_;
    std::vector<double> b_large_vec_;
    std::vector<double> lob_vec_;
    std::vector<double> zob_vec_;
    int n_zob_ = 0;
    int n_lob_ = 0;

    // Current grid point state
    double zob_ = 0.0;
    double chi_o_ = 0.0;  // chi(zob)

    // Cosmology
    double h0_ = 1.0;

  public:
    size_t get_device_mem_footprint()
    {
      size_t size = 0;
      size += dsigma_hh_.get_device_mem_footprint();
      size += bias_.get_device_mem_footprint();
      size += sigma_crit_inv_.get_device_mem_footprint();
      return size;
    }

    explicit GAMMA_PRJ(cosmosis::DataBlock& sample)
      : dsigma_hh_(make_Interp2D(sample, "haloModel", "r_sigma", "z", "DSigma_hh"))
      , bias_(make_Interp2D(sample, "haloModel", "lnM", "z", "bias"))
      , sigma_crit_inv_(make_Interp2D(sample,
                                      "haloModel", "z",
                                      "haloModel", "r_sigma",
                                      "sigmaCritInv", "sigma_crit_inv"))
      , chi_(make_Interp1D(sample, "distances", "z", "d_c"))
    {
      h0_ = sample.view<double>("cosmological_parameters", "h0");

      // Read B_small, B_large from bSelMargGPU output
      if (sample.has_val("b_sel_marginalised", "b_small")) {
        b_small_vec_ = get_vector_double(sample, "b_sel_marginalised", "b_small");
        b_large_vec_ = get_vector_double(sample, "b_sel_marginalised", "b_large");
        lob_vec_ = get_vector_double(sample, "b_sel_marginalised", "lob");
        zob_vec_ = get_vector_double(sample, "b_sel_marginalised", "zob");

        // Determine grid dimensions (assume regular grid)
        // Count unique zob values
        std::vector<double> unique_zob;
        for (double z : zob_vec_) {
          bool found = false;
          for (double uz : unique_zob) {
            if (std::abs(z - uz) < 1e-6) { found = true; break; }
          }
          if (!found) unique_zob.push_back(z);
        }
        n_zob_ = unique_zob.size();
        n_lob_ = (n_zob_ > 0) ? (zob_vec_.size() / n_zob_) : 0;
      }
    }

    void set_grid_point(double zo_low, double zo_high)
    {
      zob_ = 0.5 * (zo_low + zo_high);
      chi_o_ = chi_.clamp(zob_) * h0_;  // Convert to cMpc/h
    }

    // Lookup B_small, B_large for given (zob, lob)
    __host__ __device__ void
    get_B_values(double lob, double& B_small, double& B_large) const
    {
      // Default values if no b_sel data available
      B_small = 1.0;
      B_large = 1.0;

      if (b_small_vec_.empty() || n_zob_ == 0 || n_lob_ == 0) {
        return;
      }

      // Find nearest zob index
      int iz = 0;
      double min_dz = 1e30;
      for (int i = 0; i < n_zob_; ++i) {
        // zob values repeat every n_lob_ entries
        double z_i = zob_vec_[i * n_lob_];
        double dz = std::abs(z_i - zob_);
        if (dz < min_dz) { min_dz = dz; iz = i; }
      }

      // Find nearest lob index
      int il = 0;
      double min_dl = 1e30;
      for (int i = 0; i < n_lob_; ++i) {
        double l_i = lob_vec_[i];
        double dl = std::abs(l_i - lob);
        if (dl < min_dl) { min_dl = dl; il = i; }
      }

      // Linear index into flat arrays
      int idx = iz * n_lob_ + il;
      if (idx >= 0 && idx < static_cast<int>(b_small_vec_.size())) {
        B_small = b_small_vec_[idx];
        B_large = b_large_vec_[idx];
      }
    }

    // Main evaluation: gamma_prj(r, lnM, zt, lo)
    // r = radius (h^-1 Mpc)
    // lnM = log mass
    // zt = true redshift
    // lo = observed richness (used to look up B_small, B_large)
    __host__ __device__ double
    operator()(double r, double lnM, double zt, double lo) const
    {
      // 2-halo surface density
      double const dsigma_2h = dsigma_hh_.clamp(r, zt);

      // Halo bias
      double const b_Mz = bias_.clamp(lnM, zt);

      // Inverse critical surface density
      double const sigc_inv = sigma_crit_inv_.clamp(zt, r);

      // Selection bias correction
      // theta = angular separation = r / D_A(zob)
      // D_A = chi / (1 + z)
      double const D_A_o = chi_o_ / (1.0 + zob_);
      double const theta = (D_A_o > 0.0) ? (r / D_A_o) : 0.0;

      // theta_lob for sigmoid
      double const th_lob = theta_lob(lo, zob_, chi_o_);

      // Get B_small, B_large for this (zob, lo)
      double B_small, B_large;
      get_B_values(lo, B_small, B_large);

      // b_sel(theta) = B_small + (B_large - B_small) * sigmoid(theta)
      double const sig = b_sel_sigmoid(theta, th_lob);
      double const b_sel = B_small + (B_large - B_small) * sig;

      // gamma_prj = DSigma_2h * bias * b_sel * sigma_crit_inv
      return dsigma_2h * b_Mz * b_sel * sigc_inv;
    }

    // Overload without lo - uses default b_sel = 1 (no correction)
    __host__ __device__ double
    operator()(double r, double lnM, double zt) const
    {
      double const dsigma_2h = dsigma_hh_.clamp(r, zt);
      double const b_Mz = bias_.clamp(lnM, zt);
      double const sigc_inv = sigma_crit_inv_.clamp(zt, r);
      // No selection bias correction
      return dsigma_2h * b_Mz * sigc_inv;
    }
  };

}  // namespace y3_cuda

#endif

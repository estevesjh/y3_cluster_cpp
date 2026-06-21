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

  // Maximum number of (zob, lob) grid points for B_small/B_large lookup
  // Typically 3 z-bins x 4 lambda-bins = 12 points
  static constexpr int MAX_BSEL_GRID = 64;

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

    // B_small, B_large from bSelMargGPU - GPU-compatible fixed arrays
    double b_small_arr_[MAX_BSEL_GRID];
    double b_large_arr_[MAX_BSEL_GRID];
    double lob_arr_[MAX_BSEL_GRID];
    double zob_arr_[MAX_BSEL_GRID];
    int n_grid_ = 0;
    int n_zob_ = 0;
    int n_lob_ = 0;

    // Current grid point state
    double zob_ = 0.0;
    double chi_o_ = 0.0;  // chi(zob)

    // Pre-computed B values for current zob (averaged over lob)
    // This simplifies GPU lookup
    double B_small_current_ = 1.0;
    double B_large_current_ = 1.0;

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
      // Initialize arrays to defaults
      for (int i = 0; i < MAX_BSEL_GRID; ++i) {
        b_small_arr_[i] = 1.0;
        b_large_arr_[i] = 1.0;
        lob_arr_[i] = 0.0;
        zob_arr_[i] = 0.0;
      }

      h0_ = sample.view<double>("cosmological_parameters", "h0");

      // Read B_small, B_large from bSelMargGPU output
      // These are written as ndarray<double> so read them that way
      if (sample.has_val("b_sel_marginalised", "b_small")) {
        auto b_small_nd = sample.view<cosmosis::ndarray<double>>("b_sel_marginalised", "b_small");
        auto b_large_nd = sample.view<cosmosis::ndarray<double>>("b_sel_marginalised", "b_large");
        auto lob_nd = sample.view<cosmosis::ndarray<double>>("b_sel_marginalised", "lob");
        auto zob_nd = sample.view<cosmosis::ndarray<double>>("b_sel_marginalised", "zob");

        n_grid_ = static_cast<int>(b_small_nd.size());
        if (n_grid_ > MAX_BSEL_GRID) n_grid_ = MAX_BSEL_GRID;

        // Copy to fixed arrays
        for (int i = 0; i < n_grid_; ++i) {
          b_small_arr_[i] = b_small_nd[i];
          b_large_arr_[i] = b_large_nd[i];
          lob_arr_[i] = lob_nd[i];
          zob_arr_[i] = zob_nd[i];
        }

        // Determine grid dimensions (assume regular grid)
        // Count unique zob values
        double unique_zob[MAX_BSEL_GRID];
        int n_unique = 0;
        for (int i = 0; i < n_grid_; ++i) {
          bool found = false;
          for (int j = 0; j < n_unique; ++j) {
            if (fabs(zob_arr_[i] - unique_zob[j]) < 1e-6) { found = true; break; }
          }
          if (!found && n_unique < MAX_BSEL_GRID) {
            unique_zob[n_unique++] = zob_arr_[i];
          }
        }
        n_zob_ = n_unique;
        n_lob_ = (n_zob_ > 0) ? (n_grid_ / n_zob_) : 0;
      }
    }

    void set_grid_point(double zo_low, double zo_high)
    {
      zob_ = 0.5 * (zo_low + zo_high);
      chi_o_ = chi_.clamp(zob_) * h0_;  // Convert to cMpc/h

      // Pre-compute average B_small and B_large for this zob
      // (averaging over lob bins)
      if (n_grid_ > 0 && n_zob_ > 0) {
        // Find nearest zob index
        int iz = 0;
        double min_dz = 1e30;
        for (int i = 0; i < n_zob_; ++i) {
          double z_i = zob_arr_[i * n_lob_];
          double dz = fabs(z_i - zob_);
          if (dz < min_dz) { min_dz = dz; iz = i; }
        }

        // Average B_small and B_large over lob bins for this zob
        double sum_small = 0.0, sum_large = 0.0;
        int count = 0;
        for (int il = 0; il < n_lob_; ++il) {
          int idx = iz * n_lob_ + il;
          if (idx < n_grid_) {
            sum_small += b_small_arr_[idx];
            sum_large += b_large_arr_[idx];
            ++count;
          }
        }
        if (count > 0) {
          B_small_current_ = sum_small / count;
          B_large_current_ = sum_large / count;
        }
      }
    }

    // Main evaluation: gamma_prj(r, lnM, zt, lo)
    // r = radius (h^-1 Mpc)
    // lnM = log mass
    // zt = true redshift
    // lo = observed richness (used for theta_lob calculation)
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

      // Use pre-computed B_small/B_large for current zob
      // (GPU-compatible: no std::vector access)
      double const B_small = B_small_current_;
      double const B_large = B_large_current_;

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

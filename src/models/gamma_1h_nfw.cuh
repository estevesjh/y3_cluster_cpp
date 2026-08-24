#ifndef Y3_CLUSTER_GAMMA_1H_NFW_CUH
#define Y3_CLUSTER_GAMMA_1H_NFW_CUH
//
// This file defines the GPU-compatible 1-halo shear model, including
// miscentering.
//
// DES-Y3 redMaPPer centring is imperfect: a fraction f_mis of clusters
// are offset from the true halo centre by a 2-D radial distance R_mis
// drawn from a Gamma/Rayleigh-shaped kernel with characteristic scale
// R_mis = tau_mis * R_lambda(lambda^ob) (Zhang et al. 2019 measurement;
// see arXiv:2002.11124 Sec. "Cluster centering": f_cen = 0.75 +/- 0.08,
// tau_mis = 0.17 +/- 0.04). The lensing observable is a two-component
// mixture:
//
//     DSigma_cl(R | M, z) = (1 - f_mis) * DSigma_NFW(R, M)
//                         + f_mis      * DSigma_mis(R, M; tau_mis R_lambda)
//
// This mirrors the CPU reference implementation in
// src/modules/num_counts_sel/lensing_weights.hh (Shear1hMisWeight /
// DSigma1hMisWeight), which uses the same NFW_DSIGMA_MIS machinery with
// the "gamma" kernel. red_shear_prj instead uses the "single" (delta
// function) kernel for its own miscentering table
// (sigma_prj_gpu_t.cuh) because its theta integral already plays the
// role of the R_mis integration -- the two do not double-count.
//
#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"
#include "common/cuda/Interp2D.cuh"
#include "models/ez.hh"
#include "models/nfw_dsigma_mis.cuh"
#include "utils/make_interp_2d.cuh"
#include "utils/primitives.hh"

namespace y3_cuda {

  namespace gamma_1h_detail {
    inline quad::Interp2D make_dsigma_nfw(cosmosis::DataBlock& sample)
    {
      // quad::Interp2D(xs, ys, zs) stores xs→cols, ys→rows.
      // dSigma_nfw shape (n_M, n_r): M rows, r cols. Call as clamp(r, lnM).
      return make_Interp2D(sample, "haloModel", "r_sigma", "lnM", "dSigma_nfw");
    }

    inline double read_or_default(cosmosis::DataBlock& sample,
                                  char const* section,
                                  char const* name,
                                  double fallback)
    {
      return sample.has_val(section, name) ? sample.view<double>(section, name) : fallback;
    }

    // R_lambda(lambda^ob) = (lambda^ob / 100)^0.2  [h^-1 Mpc], matching
    // src/modules/num_counts_sel/lensing_weights.hh's mis_detail::R_lambda.
    inline double R_lambda(double lob) { return std::pow(lob / 100.0, 0.2); }
  }

  class GAMMA_1H_NFW {
  private:
    quad::Interp2D _dsigma_nfw;
    quad::Interp2D _sigma_crit_inv;
    NFW_DSIGMA_MIS _dsigma_mis;
    double f_mis_   = 0.22;
    double tau_mis_ = 0.17;
    // tau_mis * R_lambda for the richness bin currently being integrated.
    // Wall-grid point-constant (set from the bin edges, not an
    // integration variable) -- see set_lob_centre().
    double r_mis_ = 0.17 * gamma_1h_detail::R_lambda(25.0);

  public:
    size_t
    get_device_mem_footprint()
    {
      size_t size = 0;
      size += _dsigma_nfw.get_device_mem_footprint();
      size += _sigma_crit_inv.get_device_mem_footprint();
      return size;
    }

    explicit GAMMA_1H_NFW(cosmosis::DataBlock& sample)
      : _dsigma_nfw(gamma_1h_detail::make_dsigma_nfw(sample))
      // sigma_crit_inv shape (n_z, n_r): z rows, r cols. Call as clamp(r, z).
      , _sigma_crit_inv(make_Interp2D(sample,
                                      "sigmaCritInv",
                                      "r_sigma",
                                      "sigmaCritInv",
                                      "z",
                                      "sigmaCritInv",
                                      "sigma_crit_inv"))
      , _dsigma_mis(4.0, 2.77533742639e+11, std::string("gamma"))
      , f_mis_(gamma_1h_detail::read_or_default(sample, "miscentering", "f_mis", 0.22))
      , tau_mis_(gamma_1h_detail::read_or_default(sample, "miscentering", "tau_mis", 0.17))
    {
      // Legacy hybrid normalisation preserved: rho_crit boundary with an
      // Omega_m amplitude factor, now applied OUTSIDE the profile class
      // (rho_mult was removed from NFW_DSIGMA_MIS; the unified des_y3
      // chain uses set_rho_ref(haloModel/rho_m_ref) instead).
      omega_m_ = sample.view<double>("cosmological_parameters", "omega_M");
      r_mis_ = tau_mis_ * gamma_1h_detail::R_lambda(25.0);
    }

    // Called once per wall-grid point (from Shear1hMisSel::set_grid_point)
    // whenever the richness bin changes, with the bin's own arithmetic
    // centre lob_centre = 0.5*(lo_low + lo_high).
    void
    set_lob_centre(double lob_centre)
    {
      r_mis_ = tau_mis_ * gamma_1h_detail::R_lambda(lob_centre);
    }

    __device__ __host__ double
    operator()(double r, double lnM, double zt) const
    /* r in h^-1 Mpc */ /* M in h^-1 M_solar, represented by lnM */
    {
      double const dsigma_cen = _dsigma_nfw.clamp(r, lnM);
      double const dsigma_mis = omega_m_ * _dsigma_mis(r, r_mis_, lnM);

      // Full 1-halo DeltaSigma mixture:
      // DeltaSigma_cl = (1 - f_mis) DeltaSigma_NFW + f_mis DeltaSigma_mis.
      double const dsigma_1h = (1.0 - f_mis_) * dsigma_cen + f_mis_ * dsigma_mis;

      double const sigc_inv = _sigma_crit_inv.clamp(r, zt);
      return dsigma_1h * sigc_inv;
    }
  };
}

#endif

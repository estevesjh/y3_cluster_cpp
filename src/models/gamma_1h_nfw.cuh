#ifndef Y3_CLUSTER_GAMMA_1H_NFW_CUH
#define Y3_CLUSTER_GAMMA_1H_NFW_CUH

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"
#include "common/cuda/Interp2D.cuh"
#include "models/ez.hh"
#include "utils/make_interp_2d.cuh"
#include "utils/primitives.hh"

namespace y3_cuda {
  class GAMMA_1H_NFW {
  private:
    quad::Interp2D _dsigma_nfw;
    quad::Interp2D _sigma_crit_inv;

  public:
    size_t
    get_device_mem_footprint()
    {
      size_t size = 0;
      size += _dsigma_nfw.get_device_mem_footprint();
      size += _sigma_crit_inv.get_device_mem_footprint();
      return size;
    }

    GAMMA_1H_NFW(quad::Interp2D const& dsigma_nfw,
                 quad::Interp2D const& sigma_crit_inv)
      : _dsigma_nfw(dsigma_nfw), _sigma_crit_inv(sigma_crit_inv)
    {}

    explicit GAMMA_1H_NFW(cosmosis::DataBlock& sample)
      // DSigma_nfw has shape (n_M, n_R), so x-axis = lnM (rows), y-axis = r_sigma (cols)
      : _dsigma_nfw(make_Interp2D(sample,
                                  "haloModel",
                                  "lnM",
                                  "r_sigma",
                                  "DSigma_nfw"))
      // sigma_crit_inv has shape (n_z, n_r), so x-axis = z (rows), y-axis = r_sigma (cols)
      , _sigma_crit_inv(make_Interp2D(sample,
                                      "sigmaCritInv",
                                      "z",
                                      "sigmaCritInv",
                                      "r_sigma",
                                      "sigmaCritInv",
                                      "sigma_crit_inv"))
    {}

    __device__ __host__ double
    operator()(double r, double lnM, double zt) const
    /* r in h^-1 Mpc */ /* M in h^-1 M_solar, represented by lnM */
    {
      // DSigma_nfw(lnM, r) - x=lnM, y=r
      double const dsigma_1h = _dsigma_nfw.clamp(lnM, r);
      // sigma_crit_inv(z, r) - x=z, y=r
      double const sigc_inv = _sigma_crit_inv.clamp(zt, r);
      return dsigma_1h * sigc_inv;
    }
  };
}

#endif

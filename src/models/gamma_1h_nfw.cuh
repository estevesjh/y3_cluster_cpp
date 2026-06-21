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
      : _dsigma_nfw(make_Interp2D(sample,
                                  "haloModel",
                                  "r_sigma",
                                  "lnM",
                                  "DSigma_nfw"))
      , _sigma_crit_inv(make_Interp2D(sample,
                                      "haloModel",
                                      "z",
                                      "haloModel",
                                      "r_sigma",
                                      "sigmaCritInv",
                                      "sigma_crit_inv"))
    {}

    __device__ __host__ double
    operator()(double r, double lnM, double zt) const
    /* r in h^-1 Mpc */ /* M in h^-1 M_solar, represented by lnM */
    {
      double const dsigma_1h = _dsigma_nfw.clamp(r, lnM);
      double const sigc_inv = _sigma_crit_inv.clamp(zt, r);
      return dsigma_1h * sigc_inv;
    }
  };
}

#endif

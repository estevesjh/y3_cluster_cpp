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
      // quad::Interp2D(xs, ys, zs) stores xs→cols, ys→rows.
      // dSigma_nfw shape (n_M, n_r) from Python means M varies on rows, r on cols.
      // Swap axis order so cols=n_r, rows=n_M. Call as clamp(lnM, r) works after swapping.
      : _dsigma_nfw(make_Interp2D(sample,
                                  "haloModel",
                                  "r_sigma",
                                  "lnM",
                                  "dSigma_nfw"))
      // sigma_crit_inv shape (n_z, n_r): z→rows, r→cols. Swap: xs=r_sigma, ys=z.
      , _sigma_crit_inv(make_Interp2D(sample,
                                      "sigmaCritInv",
                                      "r_sigma",
                                      "sigmaCritInv",
                                      "z",
                                      "sigmaCritInv",
                                      "sigma_crit_inv"))
    {}

    __device__ __host__ double
    operator()(double r, double lnM, double zt) const
    /* r in h^-1 Mpc */ /* M in h^-1 M_solar, represented by lnM */
    {
      // After axis swap: xs=r_sigma (cols), ys=lnM (rows). Call as clamp(r, lnM).
      double const dsigma_1h = _dsigma_nfw.clamp(r, lnM);
      // After axis swap: xs=r_sigma (cols), ys=z (rows). Call as clamp(r, z).
      double const sigc_inv = _sigma_crit_inv.clamp(r, zt);
      return dsigma_1h * sigc_inv;
    }
  };
}

#endif
